"""Orquestador del walking skeleton ``given_clips`` (Fase B).

Encadena contrato -> resolución -> ensamblado -> caption -> gate -> paquete de
entrega y deja un ``run_manifest.json`` reproducible. El gate se ejecuta sobre
el artefacto final y el exportador solo publica piezas que pasan: un caption
sin la mención obligatoria queda rechazado y el paquete lo reporta.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import ClassVar, Protocol

from pydantic import BaseModel, ConfigDict, Field

from kliptych.assembler import FFmpegAssembler
from kliptych.assets import AssetNotFoundError, AssetRegistry
from kliptych.config import Settings
from kliptych.contract import (
    Contract,
    ContractDraft,
    Format,
    Mode,
    Watermark,
    contract_digest,
    contract_mutes_audio,
)
from kliptych.environment import EnvironmentReport
from kliptych.exporter import DeliveryReport, export_delivery
from kliptych.gate import Gate, GateResult, Piece, SubtitleSegment
from kliptych.gate.brand_safety import (
    ChatJsonModel,
    make_brand_safety_validator,
    make_model_assessor,
)
from kliptych.gate.probe import FFprobeProbe
from kliptych.hashing import brief_key, sha256_file
from kliptych.lyrics import (
    LyricsError,
    lyric_lines_to_subtitle_segments,
    lyric_lines_to_subtitle_text,
    resolve_synced_lyrics,
)
from kliptych.manifest import OutputHash, RunManifest, write_manifest
from kliptych.resolver import (
    IssueCode,
    ProvenanceError,
    ResolutionIssue,
    ResolutionResult,
    ResolutionStatus,
    resolve_contract,
)
from kliptych.runtime import CampaignModel, PieceContext
from kliptych.subtitle_text import (
    hydrate_piece_subtitle_segments,
    hydrate_piece_subtitle_text,
)

_VIDEO_KIND = "video"
_SHA256_PATTERN = r"^[0-9a-f]{64}$"


class PipelineError(Exception):
    """La corrida no se pudo completar."""


class RunOutcome(StrEnum):
    """Resultado global de una corrida."""

    EXPORTED = "exported"
    PARTIAL = "partial"
    BLOCKED = "blocked"
    MANUAL_REVIEW = "manual_review"
    NEW_ARCHETYPE = "new_archetype"
    UNSUPPORTED = "unsupported"


class RunResult(BaseModel):
    """Resultado de una corrida del pipeline."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", frozen=True)

    run_id: str = Field(min_length=1)
    outcome: RunOutcome
    resolution_status: ResolutionStatus
    contract_sha256: str | None = Field(default=None, pattern=_SHA256_PATTERN)
    package_path: str | None = None
    manifest_path: str = Field(min_length=1)
    issues: tuple[ResolutionIssue, ...] = ()
    delivery: DeliveryReport | None = None


class PieceAssembler(Protocol):
    """Ensambla el clip entregado en el artefacto final de la pieza."""

    def assemble(
        self,
        *,
        clip: Path,
        destination: Path,
        watermark: Path | None,
        watermark_config: Watermark | None = None,
        mute_audio: bool = False,
    ) -> Path:
        """Ensambla el clip y devuelve la ruta del artefacto."""
        ...

    def render_arguments(
        self,
        *,
        clip: Path,
        destination: Path,
        watermark: Path | None,
        watermark_config: Watermark | None = None,
        mute_audio: bool = False,
    ) -> tuple[str, ...]:
        """Devuelve la receta de render que se registra en el manifiesto."""
        ...


@dataclass(frozen=True, slots=True)
class RunRequest:
    """Entradas de una corrida, con dependencias externas inyectables."""

    brief: str
    destination: Path
    environment: EnvironmentReport
    model_version: str
    prompt_version: str
    caption_prompt_version: str
    run_id: str | None = None
    assembler: PieceAssembler | None = None
    gate: Gate | None = None
    registry: AssetRegistry | None = None
    brief_path: Path | None = None
    approve_manual_review: bool = False
    approved_by: str | None = None
    contract_draft: ContractDraft | None = None
    subtitle_texts: Mapping[str, str] | None = None
    subtitle_segments: Mapping[str, Sequence[SubtitleSegment]] | None = None


@dataclass(frozen=True, slots=True)
class _RunContext:
    """Estado y dependencias de una corrida ya preparada."""

    request: RunRequest
    identifier: str
    started_at: datetime
    run_dir: Path
    settings: Settings
    registry: AssetRegistry
    model: CampaignModel
    engine: Gate
    builder: PieceAssembler


def run_given_clips(
    *,
    model: CampaignModel,
    settings: Settings,
    request: RunRequest,
) -> RunResult:
    """Corre el modo ``given_clips`` de punta a punta.

    Args:
        model: Modelo de runtime (real o grabado) que extrae y redacta.
        settings: Rutas del workspace; ``runs/<run_id>`` guarda artefactos y
            manifiesto.
        request: Entradas de la corrida.

    Returns:
        El resultado con el estado, el paquete y la ruta del manifiesto.

    Raises:
        PipelineError: Si el contrato no declara clips de video que ensamblar.
    """
    started_at = datetime.now(UTC)
    identifier = request.run_id if request.run_id is not None else _run_id(started_at)
    registry = request.registry if request.registry is not None else AssetRegistry(settings.root)
    draft = (
        request.contract_draft
        if request.contract_draft is not None
        else model.extract_contract(request.brief)
    )
    try:
        resolution = resolve_contract(draft, registry=registry, brief_text=request.brief)
    except ProvenanceError as error:
        msg = f"error de procedencia en el contrato: {error}"
        raise PipelineError(msg) from error
    gate = Gate(FFprobeProbe()) if request.gate is None else request.gate
    if isinstance(model, ChatJsonModel) and hasattr(gate, "register_validator"):
        gate.register_validator(
            "brand.safety",
            make_brand_safety_validator(make_model_assessor(model)),
        )
    context = _RunContext(
        request=request,
        identifier=identifier,
        started_at=started_at,
        run_dir=settings.runs_dir / identifier,
        settings=settings,
        registry=registry,
        model=model,
        engine=gate,
        builder=FFmpegAssembler() if request.assembler is None else request.assembler,
    )
    if resolution.status is not ResolutionStatus.RESOLVED or resolution.contract is None:
        return _unresolved(context, resolution)
    contract = resolution.contract
    if contract.mode is not Mode.GIVEN_CLIPS:
        return _halted(
            context,
            contract,
            outcome=RunOutcome.UNSUPPORTED,
            issue=ResolutionIssue(
                code=IssueCode.MODE_NOT_IMPLEMENTED,
                field="mode",
                detail=f"modo no implementado en el pipeline: {contract.mode.value}",
            ),
        )
    unsafe = _unsafe_scope_assets(context, contract)
    if unsafe:
        return _halted(
            context,
            contract,
            outcome=RunOutcome.MANUAL_REVIEW,
            issue=ResolutionIssue(
                code=IssueCode.UNSAFE_ASSET_SCOPE,
                field="assets.required",
                detail=(
                    "assets fuera del alcance permitido (private/ o runs/): "
                    + ", ".join(str(path) for path in unsafe)
                ),
            ),
        )
    return _resolved(context, contract)


def _halted(
    context: _RunContext,
    contract: Contract,
    *,
    outcome: RunOutcome,
    issue: ResolutionIssue,
) -> RunResult:
    manifest_path = _write_manifest(context, contract=contract)
    return RunResult(
        run_id=context.identifier,
        outcome=outcome,
        resolution_status=ResolutionStatus.RESOLVED,
        contract_sha256=contract_digest(contract),
        issues=(issue,),
        manifest_path=str(manifest_path),
    )


def _unsafe_scope_assets(context: _RunContext, contract: Contract) -> list[Path]:
    private = context.settings.private_campaigns_dir.resolve()
    runs = context.settings.runs_dir.resolve()
    allowed = _allowed_private_root(context, private)
    unsafe: list[Path] = []
    for asset in contract.assets.required:
        path = context.registry.path_for(asset.asset_id).resolve()
        if _outside_scope(path, private=private, runs=runs, allowed=allowed):
            unsafe.append(path)
    watermark = _watermark_path(contract, context.registry)
    if watermark is not None:
        resolved = watermark.resolve()
        if _outside_scope(resolved, private=private, runs=runs, allowed=allowed):
            unsafe.append(resolved)
    return unsafe


def _allowed_private_root(context: _RunContext, private: Path) -> Path | None:
    brief_path = context.request.brief_path
    if brief_path is None:
        return None
    resolved = brief_path.resolve()
    return resolved.parent if resolved.is_relative_to(private) else None


def _outside_scope(path: Path, *, private: Path, runs: Path, allowed: Path | None) -> bool:
    if path.is_relative_to(runs):
        return True
    if not path.is_relative_to(private):
        return False
    return allowed is None or not path.is_relative_to(allowed)


def _unresolved(context: _RunContext, resolution: ResolutionResult) -> RunResult:
    outcome = (
        RunOutcome.NEW_ARCHETYPE
        if resolution.status is ResolutionStatus.NEW_ARCHETYPE
        else RunOutcome.MANUAL_REVIEW
    )
    manifest_path = _write_manifest(context)
    return RunResult(
        run_id=context.identifier,
        outcome=outcome,
        resolution_status=resolution.status,
        issues=resolution.issues,
        manifest_path=str(manifest_path),
    )


def _resolved(context: _RunContext, contract: Contract) -> RunResult:
    pieces, gates, outputs = _assemble_pieces(context, contract)
    if not pieces:
        msg = "el contrato no declara clips de video para ensamblar"
        raise PipelineError(msg)
    manifest_path = _write_manifest(context, contract=contract, outputs=outputs, gates=gates)
    delivery = export_delivery(
        contract=contract,
        pieces=pieces,
        gate=context.engine,
        assets=context.registry,
        destination=context.request.destination,
        approve_manual_review=context.request.approve_manual_review,
        approved_by=context.request.approved_by,
    )
    if delivery.manually_approved_rules:
        manifest_path = _write_manifest(
            context, contract=contract, outputs=outputs, gates=gates, delivery=delivery
        )
    return RunResult(
        run_id=context.identifier,
        outcome=RunOutcome(delivery.status.value),
        resolution_status=ResolutionStatus.RESOLVED,
        contract_sha256=contract_digest(contract),
        package_path=str(context.request.destination),
        manifest_path=str(manifest_path),
        delivery=delivery,
    )


def _resolve_piece_subtitles(
    context: _RunContext,
    contract: Contract,
    asset_id: str,
) -> tuple[str | None, tuple[SubtitleSegment, ...]]:
    subtitle_texts = context.request.subtitle_texts or {}
    subtitle_segments = context.request.subtitle_segments or {}
    sub_segments = tuple(subtitle_segments.get(asset_id, ()))
    sub_text = subtitle_texts.get(asset_id)
    if contract.format is Format.LYRIC_VIDEO:
        try:
            lines = resolve_synced_lyrics(contract=contract, registry=context.registry)
            if not sub_segments:
                sub_segments = lyric_lines_to_subtitle_segments(lines)
            if sub_text is None:
                sub_text = lyric_lines_to_subtitle_text(lines)
        except (LyricsError, OSError):
            pass
    if not sub_segments:
        sub_segments = hydrate_piece_subtitle_segments(
            transcript=None,
            segment=None,
            work_dir=context.run_dir,
        )
    if sub_text is None:
        sub_text = hydrate_piece_subtitle_text(
            transcript=None,
            segment=None,
            work_dir=context.run_dir,
        )
    return sub_text, sub_segments


def _assemble_pieces(
    context: _RunContext,
    contract: Contract,
) -> tuple[list[Piece], list[GateResult], list[OutputHash]]:
    pieces: list[Piece] = []
    gates: list[GateResult] = []
    outputs: list[OutputHash] = []
    watermark = _watermark_path(contract, context.registry)
    watermark_config = contract.watermark if watermark is not None else None
    mute_audio = contract_mutes_audio(contract)
    for platform in sorted(contract.platforms, key=lambda item: item.value):
        for asset in contract.assets.required:
            if asset.kind != _VIDEO_KIND:
                continue
            clip = context.registry.path_for(asset.asset_id)
            artifact = context.run_dir / "artifacts" / platform.value / f"{asset.asset_id}.mp4"
            _ = context.builder.assemble(
                clip=clip,
                destination=artifact,
                watermark=watermark,
                watermark_config=watermark_config,
                mute_audio=mute_audio,
            )
            caption = context.model.write_caption(
                contract,
                PieceContext(piece_id=asset.asset_id, platform=platform),
            )
            sub_text, sub_segments = _resolve_piece_subtitles(context, contract, asset.asset_id)
            piece = Piece(
                piece_id=asset.asset_id,
                platform=platform,
                caption=caption.caption,
                hashtags=caption.hashtags,
                subtitle_text=sub_text,
                subtitle_segments=sub_segments,
                artifact_path=artifact,
            )
            pieces.append(piece)
            gates.append(
                context.engine.run(contract=contract, piece=piece, assets=context.registry)
            )
            outputs.append(
                OutputHash(
                    path=artifact.relative_to(context.run_dir).as_posix(),
                    sha256=sha256_file(artifact),
                    size_bytes=artifact.stat().st_size,
                    render_arguments=context.builder.render_arguments(
                        clip=clip,
                        destination=artifact,
                        watermark=watermark,
                        watermark_config=watermark_config,
                        mute_audio=mute_audio,
                    ),
                )
            )
    return pieces, gates, outputs


def _watermark_path(contract: Contract, registry: AssetRegistry) -> Path | None:
    asset_id = contract.watermark.asset_id
    if asset_id is None:
        return None
    try:
        return registry.path_for(asset_id)
    except AssetNotFoundError:
        return None


def _write_manifest(
    context: _RunContext,
    *,
    contract: Contract | None = None,
    outputs: Sequence[OutputHash] = (),
    gates: Sequence[GateResult] = (),
    delivery: DeliveryReport | None = None,
) -> Path:
    request = context.request
    manually_approved_rules = delivery.manually_approved_rules if delivery is not None else ()
    approved_by = delivery.approved_by if delivery is not None else None
    approved_at_utc = delivery.approved_at_utc if delivery is not None else None
    manifest = RunManifest(
        run_id=context.identifier,
        started_at=context.started_at,
        finished_at=datetime.now(UTC),
        brief_sha256=brief_key(request.brief),
        contract_schema_version=None if contract is None else contract.schema_version,
        contract_sha256=(None if contract is None else contract_digest(contract)),
        model_version=request.model_version,
        prompt_version=request.prompt_version,
        caption_prompt_version=request.caption_prompt_version,
        ffmpeg_version=request.environment.ffmpeg_version,
        environment=request.environment,
        assets=tuple(context.registry.assets.values()),
        outputs=tuple(outputs),
        gates=tuple(gates),
        degradations=request.environment.degradations,
        manually_approved_rules=manually_approved_rules,
        approved_by=approved_by,
        approved_at_utc=approved_at_utc,
    )
    return write_manifest(manifest, context.run_dir)


def _run_id(started_at: datetime) -> str:
    return started_at.strftime("%Y%m%dT%H%M%S%fZ")
