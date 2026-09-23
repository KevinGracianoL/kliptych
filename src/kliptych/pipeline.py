"""Orquestador del walking skeleton ``given_clips`` (Fase B).

Encadena contrato -> resolución -> ensamblado -> caption -> gate -> paquete de
entrega y deja un ``run_manifest.json`` reproducible. El gate se ejecuta sobre
el artefacto final y el exportador solo publica piezas que pasan: un caption
sin la mención obligatoria queda rechazado y el paquete lo reporta.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import ClassVar, Protocol

from pydantic import BaseModel, ConfigDict, Field

from kliptych.assembler import FFmpegAssembler
from kliptych.assets import AssetNotFoundError, AssetRegistry
from kliptych.config import Settings
from kliptych.contract import Contract, Mode, contract_digest
from kliptych.environment import EnvironmentReport
from kliptych.exporter import DeliveryReport, export_delivery
from kliptych.gate import Gate, GateResult, Piece
from kliptych.gate.probe import FFprobeProbe
from kliptych.hashing import brief_key, sha256_file
from kliptych.manifest import OutputHash, RunManifest, write_manifest
from kliptych.resolver import (
    IssueCode,
    ResolutionIssue,
    ResolutionResult,
    ResolutionStatus,
    resolve_contract,
)
from kliptych.runtime import CampaignModel, PieceContext

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

    def assemble(self, *, clip: Path, destination: Path, watermark: Path | None) -> Path:
        """Ensambla el clip y devuelve la ruta del artefacto."""
        ...

    def render_arguments(
        self, *, clip: Path, destination: Path, watermark: Path | None
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


@dataclass(frozen=True, slots=True)
class _RunContext:
    """Estado y dependencias de una corrida ya preparada."""

    request: RunRequest
    identifier: str
    started_at: datetime
    run_dir: Path
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
    resolution = resolve_contract(model.extract_contract(request.brief), registry=registry)
    context = _RunContext(
        request=request,
        identifier=identifier,
        started_at=started_at,
        run_dir=settings.runs_dir / identifier,
        registry=registry,
        model=model,
        engine=Gate(FFprobeProbe()) if request.gate is None else request.gate,
        builder=FFmpegAssembler() if request.assembler is None else request.assembler,
    )
    if resolution.status is not ResolutionStatus.RESOLVED or resolution.contract is None:
        return _unresolved(context, resolution)
    contract = resolution.contract
    if contract.mode is not Mode.GIVEN_CLIPS:
        return _unsupported_mode(context, contract)
    return _resolved(context, contract)


def _unsupported_mode(context: _RunContext, contract: Contract) -> RunResult:
    issue = ResolutionIssue(
        code=IssueCode.MODE_NOT_IMPLEMENTED,
        field="mode",
        detail=f"modo no implementado en el pipeline: {contract.mode.value}",
    )
    manifest_path = _write_manifest(context)
    return RunResult(
        run_id=context.identifier,
        outcome=RunOutcome.UNSUPPORTED,
        resolution_status=ResolutionStatus.RESOLVED,
        issues=(issue,),
        manifest_path=str(manifest_path),
    )


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
    pieces, gates, render_arguments, outputs = _assemble_pieces(context, contract)
    if not pieces:
        msg = "el contrato no declara clips de video para ensamblar"
        raise PipelineError(msg)
    manifest_path = _write_manifest(
        context,
        contract=contract,
        outputs=outputs,
        gates=gates,
        render_arguments=render_arguments,
    )
    delivery = export_delivery(
        contract=contract,
        pieces=pieces,
        gate=context.engine,
        assets=context.registry,
        destination=context.request.destination,
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


def _assemble_pieces(
    context: _RunContext,
    contract: Contract,
) -> tuple[list[Piece], list[GateResult], list[str], list[OutputHash]]:
    pieces: list[Piece] = []
    gates: list[GateResult] = []
    render_arguments: list[str] = []
    outputs: list[OutputHash] = []
    watermark = _watermark_path(contract, context.registry)
    for platform in sorted(contract.platforms, key=lambda item: item.value):
        for asset in contract.assets.required:
            if asset.kind != _VIDEO_KIND:
                continue
            clip = context.registry.path_for(asset.asset_id)
            artifact = context.run_dir / "artifacts" / platform.value / f"{asset.asset_id}.mp4"
            _ = context.builder.assemble(clip=clip, destination=artifact, watermark=watermark)
            caption = context.model.write_caption(
                contract,
                PieceContext(piece_id=asset.asset_id, platform=platform),
            )
            piece = Piece(
                piece_id=asset.asset_id,
                platform=platform,
                caption=caption.caption,
                hashtags=caption.hashtags,
                subtitle_text=None,
                artifact_path=artifact,
            )
            pieces.append(piece)
            gates.append(
                context.engine.run(contract=contract, piece=piece, assets=context.registry)
            )
            render_arguments.extend(
                context.builder.render_arguments(
                    clip=clip,
                    destination=artifact,
                    watermark=watermark,
                )
            )
            outputs.append(
                OutputHash(
                    path=artifact.relative_to(context.run_dir).as_posix(),
                    sha256=sha256_file(artifact),
                    size_bytes=artifact.stat().st_size,
                )
            )
    return pieces, gates, render_arguments, outputs


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
    render_arguments: Sequence[str] = (),
) -> Path:
    request = context.request
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
        render_arguments=tuple(render_arguments),
        environment=request.environment,
        assets=tuple(context.registry.assets.values()),
        outputs=tuple(outputs),
        gates=tuple(gates),
        degradations=request.environment.degradations,
    )
    return write_manifest(manifest, context.run_dir)


def _run_id(started_at: datetime) -> str:
    return started_at.strftime("%Y%m%dT%H%M%S%fZ")
