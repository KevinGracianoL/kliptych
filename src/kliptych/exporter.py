"""Exportador de paquetes de entrega: solo piezas que pasan el gate.

El paquete se construye en un directorio de staging y se publica reemplazando
el directorio anterior: ante un error gestionado, el paquete previo queda
intacto y una publicación exitosa no conserva piezas de corridas anteriores.
El reemplazo son dos renombres de directorio, así que existe una ventana breve
en la que el destino no está visible para lectores concurrentes y una
interrupción dura del proceso puede dejar un ``*.backup-*``/``*.staging-*``
huérfano. Dentro del paquete, cada pieza vive en ``<campaña>/<plataforma>/``
junto a su metadata final, el reporte del gate y un ``delivery_report.json``
con el resumen y los recordatorios post-publicación (geo, views para payout,
analytics y revisión manual).

Los identificadores derivados del brief se validan como segmentos de ruta
seguros y los nombres de salida se comprueban contra colisiones entre piezas;
una pieza cuyo gate no apruebe no se copia.
"""

import re
import shutil
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import ClassVar, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict

from kliptych.assets import AssetRegistry
from kliptych.contract import (
    Attribution,
    AudioRule,
    Contract,
    Languages,
    OfficialAudio,
    Platform,
    PlatformRules,
)
from kliptych.gate import (
    CheckStatus,
    Gate,
    GateResult,
    GateStatus,
    Piece,
)
from kliptych.hashing import sha256_bytes

_SAFE_SEGMENT = re.compile(r"[A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?")
_SUFFIX = re.compile(r"\.[A-Za-z0-9]{1,10}")
_MAX_SEGMENT_LENGTH = 64
_RESERVED_NAMES = frozenset(
    {
        "aux",
        "com1",
        "com2",
        "com3",
        "com4",
        "com5",
        "com6",
        "com7",
        "com8",
        "com9",
        "con",
        "lpt1",
        "lpt2",
        "lpt3",
        "lpt4",
        "lpt5",
        "lpt6",
        "lpt7",
        "lpt8",
        "lpt9",
        "nul",
        "prn",
    }
)
_SUMMARY_NAME = "delivery_report.json"


class ExportError(Exception):
    """El paquete de entrega no se pudo construir de forma segura."""


class ExportStatus(StrEnum):
    """Resultado global de un paquete de entrega."""

    EXPORTED = "exported"
    PARTIAL = "partial"
    BLOCKED = "blocked"


class _ExporterBase(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", frozen=True)


class PieceMetadata(_ExporterBase):
    """Metadata final que acompaña al artefacto en el paquete."""

    schema_version: Literal["1.0"] = "1.0"
    piece_id: str
    platform: Platform
    caption: str
    hashtags: tuple[str, ...]
    required_mentions: tuple[str, ...]
    required_hashtags: tuple[str, ...]
    attribution: Attribution
    link_in_bio: bool
    audio_rule: AudioRule
    official_audio: OfficialAudio | None
    languages: Languages
    watermark_required: bool
    artifact_sha256: str
    artifact_size_bytes: int


class ExportedPiece(_ExporterBase):
    """Pieza exportada, con rutas relativas al paquete."""

    piece_id: str
    platform: Platform
    artifact_path: str
    artifact_sha256: str
    metadata_path: str
    gate_path: str
    gate_status: GateStatus


class RejectedPiece(_ExporterBase):
    """Pieza que no se exportó, con el resultado completo del gate."""

    piece_id: str
    platform: Platform
    reason: str
    gate: GateResult

    @property
    def gate_status(self) -> GateStatus:
        """Estado del gate que motivó el rechazo.

        Returns:
            El estado global del resultado del gate embebido.
        """
        return self.gate.status


class Reminder(_ExporterBase):
    """Recordatorio post-publicación que el exportador lista para el owner."""

    kind: str
    detail: str


class DeliveryReport(_ExporterBase):
    """Resumen del paquete: lo exportado, lo rechazado y los recordatorios."""

    schema_version: Literal["1.0"] = "1.0"
    campaign_id: str
    package: str
    status: ExportStatus
    exported: tuple[ExportedPiece, ...]
    rejected: tuple[RejectedPiece, ...]
    reminders: tuple[Reminder, ...]


@dataclass(frozen=True, slots=True)
class _PiecePlan:
    piece: Piece
    rules: PlatformRules
    artifact_name: str
    metadata_name: str
    gate_name: str
    platform_dir: str


@dataclass(frozen=True, slots=True)
class _ExportContext:
    contract: Contract
    campaign: str
    staging: Path


def export_delivery(
    *,
    contract: Contract,
    pieces: Sequence[Piece],
    gate: Gate,
    assets: AssetRegistry,
    destination: Path,
) -> DeliveryReport:
    """Construye el paquete de entrega de las piezas que pasan el gate.

    El paquete se arma en staging y se publica reemplazando el directorio
    anterior; ante cualquier error gestionado, el destino queda como estaba
    antes de la llamada.

    Args:
        contract: Contrato validado que declara las reglas.
        pieces: Piezas a exportar (artefacto final más textos).
        gate: Gate configurado con su probe de medios.
        assets: Registro de assets del workspace.
        destination: Raíz del paquete de entrega.

    Returns:
        El reporte del paquete, escrito también en ``delivery_report.json``.

    Raises:
        ExportError: Si no hay piezas, un identificador no es seguro para el
            árbol de entrega, hay colisiones de nombres, la plataforma de una
            pieza no está declarada, o el artefacto no se puede copiar de
            forma verificable.
    """
    if not pieces:
        msg = "no hay piezas para exportar"
        raise ExportError(msg)
    if not destination.name:
        msg = f"destino de entrega inválido: {destination}"
        raise ExportError(msg)
    if (destination.exists() or destination.is_symlink()) and not destination.is_dir():
        msg = f"el destino de entrega existe y no es un directorio: {destination}"
        raise ExportError(msg)
    campaign = _safe_segment(contract.campaign_id, field="campaign_id")
    if campaign.casefold() == _SUMMARY_NAME:
        msg = f"campaign_id reservado por el exportador: {contract.campaign_id!r}"
        raise ExportError(msg)
    segments = [_safe_segment(piece.piece_id, field="piece_id") for piece in pieces]
    if len(set(segments)) != len(segments):
        msg = "piece_id duplicado en el paquete de entrega"
        raise ExportError(msg)
    undeclared = sorted(
        {piece.platform.value for piece in pieces if piece.platform not in contract.platforms}
    )
    if undeclared:
        msg = f"plataformas no declaradas en el contrato: {', '.join(undeclared)}"
        raise ExportError(msg)

    plans = [
        _plan_piece(piece, segment, contract.platforms[piece.platform])
        for piece, segment in zip(pieces, segments, strict=True)
    ]
    _validate_output_names(plans)

    staging = destination.with_name(f"{destination.name}.staging-{uuid4().hex}")
    context = _ExportContext(contract=contract, campaign=campaign, staging=staging)
    published = False
    try:
        staging.mkdir(parents=True)
        report = _build_report(
            context,
            plans,
            gate=gate,
            assets=assets,
            package=destination.name,
        )
        _ = (staging / _SUMMARY_NAME).write_text(report.model_dump_json(indent=2), encoding="utf-8")
        _publish(staging, destination)
        published = True
    except OSError as error:
        msg = f"falló la escritura del paquete de entrega: {error}"
        raise ExportError(msg) from error
    finally:
        if not published:
            _remove_tree(staging)
    return report


def _build_report(
    context: _ExportContext,
    plans: Sequence[_PiecePlan],
    *,
    gate: Gate,
    assets: AssetRegistry,
    package: str,
) -> DeliveryReport:
    exported, rejected = _build_package(context, plans, gate=gate, assets=assets)
    return DeliveryReport(
        campaign_id=context.contract.campaign_id,
        package=package,
        status=_status(exported=exported, rejected=rejected),
        exported=tuple(exported),
        rejected=tuple(rejected),
        reminders=_reminders(context.contract),
    )


def _plan_piece(piece: Piece, segment: str, rules: PlatformRules) -> _PiecePlan:
    suffix = piece.artifact_path.suffix
    if suffix and not _SUFFIX.fullmatch(suffix):
        msg = f"extensión de artefacto no permitida: {suffix!r}"
        raise ExportError(msg)
    return _PiecePlan(
        piece=piece,
        rules=rules,
        artifact_name=f"{segment}{suffix}",
        metadata_name=f"{segment}.metadata.json",
        gate_name=f"{segment}.gate.json",
        platform_dir=piece.platform.value,
    )


def _validate_output_names(plans: Sequence[_PiecePlan]) -> None:
    seen: set[tuple[str, str]] = set()
    for plan in plans:
        for name in (plan.artifact_name, plan.metadata_name, plan.gate_name):
            key = (plan.platform_dir.casefold(), name.casefold())
            if key in seen:
                msg = f"colisión de nombres en el paquete: {plan.platform_dir}/{name}"
                raise ExportError(msg)
            seen.add(key)


def _safe_segment(value: str, *, field: str) -> str:
    invalid = (
        len(value) > _MAX_SEGMENT_LENGTH
        or value in {".", ".."}
        or not _SAFE_SEGMENT.fullmatch(value)
        or value.split(".", 1)[0].casefold() in _RESERVED_NAMES
    )
    if invalid:
        msg = f"{field} no es un identificador seguro para el paquete: {value!r}"
        raise ExportError(msg)
    return value


def _build_package(
    context: _ExportContext,
    plans: Sequence[_PiecePlan],
    *,
    gate: Gate,
    assets: AssetRegistry,
) -> tuple[list[ExportedPiece], list[RejectedPiece]]:
    exported: list[ExportedPiece] = []
    rejected: list[RejectedPiece] = []
    for plan in plans:
        gate_result = gate.run(contract=context.contract, piece=plan.piece, assets=assets)
        if not gate_result.passed:
            rejected.append(
                RejectedPiece(
                    piece_id=plan.piece.piece_id,
                    platform=plan.piece.platform,
                    reason=_rejection_reason(gate_result),
                    gate=gate_result,
                )
            )
            continue
        exported.append(_export_piece(plan, context, gate_result))
    return exported, rejected


def _export_piece(
    plan: _PiecePlan,
    context: _ExportContext,
    gate_result: GateResult,
) -> ExportedPiece:
    piece = plan.piece
    destination = context.staging
    directory = destination / context.campaign / plan.platform_dir
    directory.mkdir(parents=True, exist_ok=True)
    if gate_result.artifact_sha256 is None:
        msg = f"el gate aprobó la pieza sin hash de artefacto: {piece.piece_id}"
        raise ExportError(msg)
    artifact_size = _copy_verified(
        piece.artifact_path,
        directory / plan.artifact_name,
        gate_result.artifact_sha256,
    )

    rules = plan.rules
    metadata = PieceMetadata(
        piece_id=piece.piece_id,
        platform=piece.platform,
        caption=piece.caption,
        hashtags=piece.hashtags,
        required_mentions=tuple(
            dict.fromkeys([*rules.required_mentions, *rules.caption_rules.must_mention])
        ),
        required_hashtags=tuple(rules.required_hashtags),
        attribution=rules.attribution,
        link_in_bio=rules.link_rules.link_in_bio,
        audio_rule=rules.audio_rule,
        official_audio=context.contract.official_audio,
        languages=context.contract.languages,
        watermark_required=context.contract.watermark.required,
        artifact_sha256=gate_result.artifact_sha256,
        artifact_size_bytes=artifact_size,
    )
    _ = (directory / plan.metadata_name).write_text(
        metadata.model_dump_json(indent=2), encoding="utf-8"
    )
    _ = (directory / plan.gate_name).write_text(
        gate_result.model_dump_json(indent=2), encoding="utf-8"
    )
    return ExportedPiece(
        piece_id=piece.piece_id,
        platform=piece.platform,
        artifact_path=(directory / plan.artifact_name).relative_to(destination).as_posix(),
        artifact_sha256=gate_result.artifact_sha256,
        metadata_path=(directory / plan.metadata_name).relative_to(destination).as_posix(),
        gate_path=(directory / plan.gate_name).relative_to(destination).as_posix(),
        gate_status=gate_result.status,
    )


def _copy_verified(source: Path, target: Path, expected_sha256: str) -> int:
    try:
        data = source.read_bytes()
    except OSError as error:
        msg = f"no se pudo leer el artefacto a exportar: {source}"
        raise ExportError(msg) from error
    if sha256_bytes(data) != expected_sha256:
        msg = f"el artefacto cambió respecto del hash validado por el gate: {source}"
        raise ExportError(msg)
    _ = target.write_bytes(data)
    return len(data)


def _publish(staging: Path, destination: Path) -> None:
    backup: Path | None = None
    if destination.exists() or destination.is_symlink():
        backup = destination.with_name(f"{destination.name}.backup-{uuid4().hex}")
        _ = destination.rename(backup)
    try:
        _ = staging.rename(destination)
    except OSError as error:
        if backup is not None:
            try:
                _ = backup.rename(destination)
            except OSError:
                msg = (
                    "no se pudo publicar el paquete ni restaurar el anterior; "
                    f"el paquete previo quedó en {backup.name}"
                )
                raise ExportError(msg) from error
        raise
    if backup is not None:
        _remove_tree(backup)


def _remove_tree(path: Path) -> None:
    if path.is_junction():
        path.rmdir()
        return
    if path.is_symlink() or path.is_file():
        path.unlink(missing_ok=True)
        return
    shutil.rmtree(path, ignore_errors=True)


def _status(
    *,
    exported: Sequence[ExportedPiece],
    rejected: Sequence[RejectedPiece],
) -> ExportStatus:
    if not rejected:
        return ExportStatus.EXPORTED
    if exported:
        return ExportStatus.PARTIAL
    return ExportStatus.BLOCKED


def _rejection_reason(gate_result: GateResult) -> str:
    failed = ", ".join(
        check.id for check in gate_result.checks if check.status is not CheckStatus.PASS
    )
    return f"el gate no aprobó la pieza ({gate_result.status.value}): {failed}"


def _reminders(contract: Contract) -> tuple[Reminder, ...]:
    reminders: list[Reminder] = []
    geo = contract.geo_target
    if geo is not None:
        reminders.append(
            Reminder(
                kind="geo_target",
                detail=(
                    f"verificar al menos {geo.min_pct}% de audiencia en {geo.country} tras publicar"
                ),
            )
        )
    payout = contract.min_views_for_payout.value
    if payout is not None:
        reminders.append(
            Reminder(
                kind="min_views_for_payout",
                detail=f"verificar al menos {payout} views para el payout",
            )
        )
    if contract.analytics_proof_required.value:
        reminders.append(
            Reminder(
                kind="analytics_proof_required",
                detail="guardar la prueba de analytics después de publicar",
            )
        )
    reminders.extend(
        Reminder(kind="manual_review", detail=f"revisión manual pendiente: {rule_id}")
        for rule_id in contract.rules.manual_review
    )
    return tuple(reminders)
