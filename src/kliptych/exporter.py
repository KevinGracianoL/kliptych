"""Exportador de paquetes de entrega: solo piezas que pasan el gate.

El paquete se organiza por campaña y plataforma:
``<destino>/<campaign_id>/<platform>/<piece_id>.<ext>`` junto a la metadata
final (caption, hashtags, menciones, audio y atribución), el reporte del gate
y un ``delivery_report.json`` con el resumen y los recordatorios
post-publicación (geo, views para payout, analytics y revisión manual).

Una pieza cuyo gate no apruebe no se copia: el reporte explica por qué.
"""

import re
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import ClassVar

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
    CheckResult,
    CheckStatus,
    Gate,
    GateResult,
    GateStatus,
    Piece,
)
from kliptych.hashing import sha256_file

_SAFE_SEGMENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
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
    gate_status: GateStatus
    reason: str
    checks: tuple[CheckResult, ...]


class Reminder(_ExporterBase):
    """Recordatorio post-publicación que el exportador lista para el owner."""

    kind: str
    detail: str


class DeliveryReport(_ExporterBase):
    """Resumen del paquete: lo exportado, lo rechazado y los recordatorios."""

    campaign_id: str
    destination: str
    status: ExportStatus
    exported: tuple[ExportedPiece, ...]
    rejected: tuple[RejectedPiece, ...]
    reminders: tuple[Reminder, ...]


@dataclass(frozen=True, slots=True)
class _PieceContext:
    contract: Contract
    rules: PlatformRules
    gate_result: GateResult
    campaign: str
    destination: Path


def export_delivery(
    *,
    contract: Contract,
    pieces: Sequence[Piece],
    gate: Gate,
    assets: AssetRegistry,
    destination: Path,
) -> DeliveryReport:
    """Construye el paquete de entrega de las piezas que pasan el gate.

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
            árbol de entrega, hay ``piece_id`` duplicados o el artefacto no se
            puede copiar de forma verificable.
    """
    if not pieces:
        msg = "no hay piezas para exportar"
        raise ExportError(msg)
    campaign = _safe_segment(contract.campaign_id, field="campaign_id")
    segments = [_safe_segment(piece.piece_id, field="piece_id") for piece in pieces]
    if len(set(segments)) != len(segments):
        msg = "piece_id duplicado en el paquete de entrega"
        raise ExportError(msg)

    exported: list[ExportedPiece] = []
    rejected: list[RejectedPiece] = []
    for piece, segment in zip(pieces, segments, strict=True):
        gate_result = gate.run(contract=contract, piece=piece, assets=assets)
        if not gate_result.passed:
            rejected.append(
                RejectedPiece(
                    piece_id=piece.piece_id,
                    platform=piece.platform,
                    gate_status=gate_result.status,
                    reason=_rejection_reason(gate_result),
                    checks=gate_result.checks,
                )
            )
            continue
        exported.append(
            _export_piece(
                piece,
                segment,
                _PieceContext(
                    contract=contract,
                    rules=contract.platforms[piece.platform],
                    gate_result=gate_result,
                    campaign=campaign,
                    destination=destination,
                ),
            )
        )

    status = _status(exported=exported, rejected=rejected)
    report = DeliveryReport(
        campaign_id=contract.campaign_id,
        destination=str(destination),
        status=status,
        exported=tuple(exported),
        rejected=tuple(rejected),
        reminders=_reminders(contract),
    )
    destination.mkdir(parents=True, exist_ok=True)
    _ = (destination / _SUMMARY_NAME).write_text(report.model_dump_json(indent=2), encoding="utf-8")
    return report


def _safe_segment(value: str, *, field: str) -> str:
    if value in {".", ".."} or not _SAFE_SEGMENT.match(value):
        msg = f"{field} no es un identificador seguro para el paquete: {value!r}"
        raise ExportError(msg)
    return value


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


def _export_piece(piece: Piece, segment: str, context: _PieceContext) -> ExportedPiece:
    rules = context.rules
    contract = context.contract
    gate_result = context.gate_result
    campaign = context.campaign
    destination = context.destination
    directory = destination / campaign / piece.platform.value
    directory.mkdir(parents=True, exist_ok=True)
    artifact_name = f"{segment}{piece.artifact_path.suffix}"
    metadata_name = f"{segment}.metadata.json"
    gate_name = f"{segment}.gate.json"
    source = piece.artifact_path
    if gate_result.artifact_sha256 is None:
        msg = f"el gate aprobó la pieza sin hash de artefacto: {piece.piece_id}"
        raise ExportError(msg)
    _copy_verified(source, directory / artifact_name, gate_result.artifact_sha256)

    metadata = PieceMetadata(
        piece_id=piece.piece_id,
        platform=piece.platform,
        caption=piece.caption,
        hashtags=piece.hashtags,
        required_mentions=tuple(rules.required_mentions),
        required_hashtags=tuple(rules.required_hashtags),
        attribution=rules.attribution,
        link_in_bio=rules.link_rules.link_in_bio,
        audio_rule=rules.audio_rule,
        official_audio=contract.official_audio,
        languages=contract.languages,
        watermark_required=contract.watermark.required,
        artifact_sha256=gate_result.artifact_sha256,
        artifact_size_bytes=(directory / artifact_name).stat().st_size,
    )
    _ = (directory / metadata_name).write_text(metadata.model_dump_json(indent=2), encoding="utf-8")
    _ = (directory / gate_name).write_text(gate_result.model_dump_json(indent=2), encoding="utf-8")
    return ExportedPiece(
        piece_id=piece.piece_id,
        platform=piece.platform,
        artifact_path=(directory / artifact_name).relative_to(destination).as_posix(),
        artifact_sha256=gate_result.artifact_sha256,
        metadata_path=(directory / metadata_name).relative_to(destination).as_posix(),
        gate_path=(directory / gate_name).relative_to(destination).as_posix(),
        gate_status=gate_result.status,
    )


def _copy_verified(source: Path, target: Path, expected_sha256: str) -> None:
    try:
        data = source.read_bytes()
    except OSError as error:
        msg = f"no se pudo leer el artefacto a exportar: {source}"
        raise ExportError(msg) from error
    _ = target.write_bytes(data)
    if sha256_file(target) != expected_sha256:
        msg = f"el artefacto copiado no coincide con el hash validado: {target}"
        raise ExportError(msg)
