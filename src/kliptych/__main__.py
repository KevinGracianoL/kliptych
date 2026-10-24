r"""Punto de entrada de línea de comandos de Kliptych.

La salida JSON es ASCII-safe (escapes ``\uXXXX``): así stdout y stderr se
pueden redirigir a otro proceso sin depender del codepage de la consola.
A partir de F1-PR1, todo output pasa por logging (prohibido ``print()``).
"""

import argparse
import json
import logging
import os
import signal
import sys
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, cast, runtime_checkable

from pydantic import ValidationError

from kliptych import __version__, orchestrator
from kliptych.assembler import AssembleError
from kliptych.assets import AssetRegistry
from kliptych.campaign_manager import CampaignManager, CampaignManagerError, CampaignOutcome
from kliptych.campaign_types import Campaign, CampaignStatus
from kliptych.config import Settings
from kliptych.contract import Contract, ContractDraft, vanilla_contract
from kliptych.contract.enums import Aspect, WatermarkPosition
from kliptych.contract.schema import LayoutSpec
from kliptych.download import MediaDownloader
from kliptych.encoding import RenderConfig
from kliptych.environment import EnvironmentReport, SubprocessRunner, detect_environment
from kliptych.exporter import ExportError
from kliptych.gate import Gate
from kliptych.gate.probe import FFprobeProbe
from kliptych.gc import clean_temporary_directories
from kliptych.git_proposals import GitHubCliProvider, ProposalEngine
from kliptych.hashing import brief_key
from kliptych.ingest import MAX_BRIEF_BYTES, IngestError, ingest_bytes, ingest_file
from kliptych.intelligence import (
    Archetype,
    CampaignClassifier,
    LLMCampaignClassifier,
)
from kliptych.logging_setup import setup_logging
from kliptych.pipeline import PipelineError, RunOutcome, RunRequest, RunResult, run_given_clips
from kliptych.resolver import ProvenanceError, resolve_contract
from kliptych.runtime import (
    CAPTION_PROMPT_VERSION,
    PROMPT_VERSION,
    CampaignModel,
    ModelError,
    ModelUnavailableError,
    OpenAIChatModel,
    RecordedModel,
)
from kliptych.studio.analyzer import (
    AnalysisCancelled,
    AnalysisEngineError,
    AnalysisRequest,
    analyze_project,
)
from kliptych.studio.models import ANALYSIS_SCHEMA_VERSION, AnalysisDocument
from kliptych.studio.preview import PreviewError, PreviewRequest, generate_preview
from kliptych.studio.project_models import AbsorbNeighbor, ProjectDocument
from kliptych.studio.project_store import ProjectStoreError, open_project_store
from kliptych.transcribe import FasterWhisperTranscriber

logger = logging.getLogger("kliptych.cli")
_ARGPARSE_USAGE_ERROR_STATUS = 2
_ANALYZE_INVOCATION_ERROR_STATUS = 1
_STUDIO_COMMAND_INVALID = "studio_command_invalid"
_PROJECT_ANALYSIS_MISSING = "project_analysis_missing"


@runtime_checkable
class _CampaignManagerProtocol(Protocol):
    def process(
        self,
        campaign: Campaign,
        *,
        mode: str = "long_video",
        url: str | None = None,
        resume: bool = False,
        approve_manual_review: bool = False,
        approved_by: str | None = None,
        audio_track_path: Path | None = None,
        audio_track_url: str | None = None,
        output_format: str | None = None,
    ) -> CampaignOutcome: ...


class _StudioStore(Protocol):
    @property
    def existing_analysis(self) -> AnalysisDocument | None:
        """Optional published B1 analysis."""
        ...

    def read_project(self) -> ProjectDocument | None:
        """Read optional editable Studio project state."""
        ...

    def initialize_project(self, *, probe: FFprobeProbe) -> ProjectDocument:
        """Initialize measured editable state for the current B1 project."""
        ...

    def create_variant(self, *, candidate_id: str, name: str) -> ProjectDocument:
        """Create an initial editable variant from a B1 candidate."""
        ...

    def update_position(
        self, *, variant_id: uuid.UUID, position_id: uuid.UUID, layout_spec: LayoutSpec
    ) -> ProjectDocument:
        """Persist a validated layout update for one variant position."""
        ...

    def split_position(
        self, *, variant_id: uuid.UUID, position_id: uuid.UUID, at_ms: int
    ) -> ProjectDocument:
        """Split one timeline position at the requested timestamp."""
        ...

    def delete_position(
        self,
        *,
        variant_id: uuid.UUID,
        position_id: uuid.UUID,
        absorb_neighbor: AbsorbNeighbor | None,
    ) -> ProjectDocument:
        """Delete one timeline position while preserving continuous coverage."""
        ...

    def duplicate_variant(self, *, variant_id: uuid.UUID, name: str) -> ProjectDocument:
        """Duplicate a variant with fresh IDs."""
        ...

    def import_watermark(
        self,
        *,
        variant_id: uuid.UUID,
        source_path: Path,
        position: WatermarkPosition | None = None,
    ) -> ProjectDocument:
        """Import a local PNG watermark into controlled project storage."""
        ...

    def set_watermark(
        self,
        *,
        variant_id: uuid.UUID,
        enabled: bool | None = None,
        position: WatermarkPosition | None = None,
    ) -> ProjectDocument:
        """Update watermark enabled state or zone for one variant."""
        ...


_SEGMENT_SYSTEM_PROMPT = (
    "Eres el selector de segmentos de Kliptych. Recibes un JSON con la "
    "transcripción, los momentos candidatos y las cotas de duración, y "
    "devuelves ÚNICAMENTE un JSON con la forma "
    '{"segments": [{"start_s": float, "end_s": float}], "rationale": str}. '
    "Reglas: identifica y extrae todos los momentos virales, divertidos, polémicos "
    "o destacados que encuentres a lo largo del vídeo (lista con todos los segmentos "
    "relevantes sin solaparse); cada segmento queda dentro del vídeo y respeta estrictamente "
    "las duraciones mínima y máxima; el rationale explica la selección."
)


class _ChatSegmentModel:
    """Modelo LongVideoModel sobre el backend OpenAI-compatible de la CLI."""

    def __init__(self, backend: OpenAIChatModel) -> None:
        """Configura el selector con su backend chat.

        Args:
            backend: Backend OpenAI-compatible ya configurado.
        """
        self._backend: OpenAIChatModel = backend

    def select_segments(self, prompt: Mapping[str, object]) -> object:
        """Selecciona los segmentos enviando el payload al backend chat.

        Args:
            prompt: Payload serializable construido por el selector.

        Returns:
            El objeto JSON devuelto por el backend.
        """
        return self._backend.chat_json(
            system_prompt=_SEGMENT_SYSTEM_PROMPT,
            user_content=json.dumps(dict(prompt), ensure_ascii=False, sort_keys=True),
        )


class _DefaultVideoOrchestrator:
    """Orquestador de video real para CampaignManager (sin inyección manual)."""

    def __init__(
        self,
        *,
        work_dir: Path,
        model: _ChatSegmentModel | None,
        render: RenderConfig,
        assets: AssetRegistry | None = None,
    ) -> None:
        """Configura el directorio de trabajo, el modelo y el render.

        Args:
            work_dir: Directorio donde el pipeline publica sus artefactos.
            model: Modelo de selección de segmentos sobre el backend LLM. Es
                ``None`` en el modo zero-contract, que fuerza repost y nunca
                selecciona segmentos.
            render: Binario, timeout y NVENC compartidos por los renders.
            assets: Registro de assets opcional del workspace.
        """
        self._work_dir: Path = work_dir
        self._model: _ChatSegmentModel | None = model
        self._render: RenderConfig = render
        self._assets: AssetRegistry | None = assets

    def run_long_video(self, url: str, **kwargs: object) -> orchestrator.PipelineResult:
        """Renderiza un vídeo largo, con audio o repost según los flags.

        Args:
            url: URL http/https del vídeo fuente.
            **kwargs: resume, contract, audio_locked, repost_mode,
                audio_track_path y audio_track_url.

        Returns:
            El resultado del pipeline long_video.

        Raises:
            CampaignManagerError: Si el modo exige selección de segmentos con
                el LLM y no hay backend configurado.
        """
        assets_in = kwargs.get("assets")
        assets = assets_in if isinstance(assets_in, AssetRegistry) else self._assets
        config = self._pipeline_config(
            kwargs.get("contract"),
            audio_locked=kwargs.get("audio_locked") is True,
            repost_mode=_is_repost_mode(kwargs.get("repost_mode"), kwargs.get("mode")),
            audio_track_path=kwargs.get("audio_track_path"),
            audio_track_url=kwargs.get("audio_track_url"),
            assets=assets,
        )
        resume = kwargs.get("resume") is True
        if config.repost_mode:
            return orchestrator.run_repost(url, model=self._model, config=config, resume=resume)
        if self._model is None:
            msg = (
                "este modo necesita un modelo de selección de segmentos y no hay "
                "backend LLM configurado; el modo zero-contract solo admite repost"
            )
            raise CampaignManagerError(msg)
        if config.audio_locked:
            return orchestrator.run_audio_locked(
                url, model=self._model, config=config, resume=resume
            )
        return orchestrator.run_long_video(url, model=self._model, config=config, resume=resume)

    def run_slideshow(
        self, images: Sequence[Path], **kwargs: object
    ) -> orchestrator.SlideshowResult:
        """Renderiza un slideshow delegando en el entry point real.

        Args:
            images: Rutas locales de las imágenes, en orden de montaje.
            **kwargs: resume, contract, audio_track_path y audio_track_url.

        Returns:
            El resultado del pipeline slideshow.
        """
        return _run_slideshow_pipeline(
            images, work_dir=self._work_dir, render=self._render, kwargs=kwargs
        )

    def _pipeline_config(
        self,
        contract: object,
        *,
        audio_locked: bool = False,
        repost_mode: bool = False,
        audio_track_path: object = None,
        audio_track_url: object = None,
        assets: AssetRegistry | None = None,
    ) -> orchestrator.PipelineConfig:
        """Construye la config del pipeline validando el contrato y la pista.

        Args:
            contract: Contrato validado de la campaña.
            audio_locked: Si se inyecta pista de audio externa.
            repost_mode: Si se usa el vídeo completo sin inteligencia.
            audio_track_path: Ruta local de la pista externa, o None.
            audio_track_url: URL de la pista externa, o None.
            assets: Registro de assets del workspace.

        Returns:
            La configuración lista para el pipeline.

        Raises:
            CampaignManagerError: Si el contrato o la pista no son válidos.
        """
        if not isinstance(contract, Contract):
            msg = "el orquestador por defecto requiere el contrato de la campaña"
            raise CampaignManagerError(msg)
        track_path = _coerce_audio_track_path(audio_track_path)
        track_url = _coerce_audio_track_url(audio_track_url)
        face_model_path = _resolve_face_model_path()
        max_size_raw = os.environ.get("KLIPTYCH_MAX_DOWNLOAD_BYTES")
        timeout_raw = os.environ.get("KLIPTYCH_DOWNLOAD_TIMEOUT_S")
        download_max_size_bytes = int(max_size_raw) if max_size_raw is not None else 10 * 1024**3
        download_timeout_s = float(timeout_raw) if timeout_raw is not None else 1800.0
        return orchestrator.PipelineConfig(
            output_dir=self._work_dir,
            contract=contract,
            render=self._render,
            face_model_path=face_model_path,
            download_timeout_s=download_timeout_s,
            download_max_size_bytes=download_max_size_bytes,
            audio_locked=audio_locked,
            audio_track_path=track_path,
            audio_track_url=track_url,
            repost_mode=repost_mode,
            assets=assets,
        )


class _DefaultSlideshowOrchestrator:
    """Orquestador de slideshow real para CampaignManager."""

    def __init__(self, *, work_dir: Path, render: RenderConfig) -> None:
        """Configura el directorio de trabajo y el render.

        Args:
            work_dir: Directorio donde el pipeline publica sus artefactos.
            render: Binario, timeout y NVENC compartidos por los renders.
        """
        self._work_dir: Path = work_dir
        self._render: RenderConfig = render

    def run(self, images: Sequence[Path], **kwargs: object) -> orchestrator.SlideshowResult:
        """Renderiza un slideshow delegando en el entry point real.

        Args:
            images: Rutas locales de las imágenes, en orden de montaje.
            **kwargs: resume, contract, audio_track_path y audio_track_url.

        Returns:
            El resultado del pipeline slideshow.
        """
        return _run_slideshow_pipeline(
            images, work_dir=self._work_dir, render=self._render, kwargs=kwargs
        )


def _is_repost_mode(repost_flag: object, mode: object) -> bool:
    """Indica si los flags del orquestador activan el pipeline repost.

    Args:
        repost_flag: Valor del flag ``repost_mode`` (activo solo si es True).
        mode: Modo de campaña opcional; ``"repost"`` y el canónico
            ``"repost_ugc"`` activan el pipeline repost.

    Returns:
        True si el pipeline repost debe ejecutarse.
    """
    if repost_flag is True:
        return True
    return isinstance(mode, str) and mode in {"repost", "repost_ugc"}


def _coerce_audio_track_path(value: object) -> Path | None:
    """Convierte la pista local a Path sin perderla en silencio.

    Args:
        value: Ruta como ``Path`` o ``str``, o None si no hay pista local.

    Returns:
        La ruta como ``Path``, o None si no hay pista local.

    Raises:
        TypeError: Si el valor no es ``Path``, ``str`` ni None.
    """
    if value is None:
        return None
    if isinstance(value, Path):
        return value
    if isinstance(value, str):
        return Path(value)
    msg = f"audio_track_path debe ser Path, str o None, no {type(value).__name__}"
    raise TypeError(msg)


def _coerce_audio_track_url(value: object) -> str | None:
    """Conserva la URL de la pista sin perderla en silencio.

    Args:
        value: URL como ``str``, o None si no hay pista remota.

    Returns:
        La URL, o None si no hay pista remota.

    Raises:
        TypeError: Si el valor no es ``str`` ni None.
    """
    if value is None:
        return None
    if isinstance(value, str):
        return value
    msg = f"audio_track_url debe ser str o None, no {type(value).__name__}"
    raise TypeError(msg)


def _resolve_face_model_path() -> Path | None:
    """Resuelve el modelo de caras para el reframe 9:16 con MediaPipe.

    Returns:
        La ruta al modelo ``.tflite`` si existe en ``KLIPTYCH_FACE_MODEL`` o en
        la ubicación por defecto, o None si no hay modelo disponible.
    """
    env_value = os.environ.get("KLIPTYCH_FACE_MODEL")
    if env_value:
        env_path = Path(env_value)
        if env_path.is_file():
            return env_path
    default_path = Path.home() / ".kliptych" / "models" / "blaze_face_short_range.tflite"
    if default_path.is_file():
        return default_path
    return None


def _run_slideshow_pipeline(
    images: Sequence[Path],
    *,
    work_dir: Path,
    render: RenderConfig,
    kwargs: Mapping[str, object],
) -> orchestrator.SlideshowResult:
    """Ejecuta el slideshow real con la pista externa obligatoria.

    Args:
        images: Rutas locales de las imágenes, en orden de montaje.
        work_dir: Directorio donde el pipeline publica sus artefactos.
        render: Binario, timeout y NVENC compartidos por los renders.
        kwargs: resume, contract, audio_track_path y audio_track_url.

    Returns:
        El resultado del pipeline slideshow.

    Raises:
        CampaignManagerError: Si falta el contrato de la campaña.
    """
    contract = kwargs.get("contract")
    if not isinstance(contract, Contract):
        msg = "el orquestador por defecto requiere el contrato de la campaña"
        raise CampaignManagerError(msg)
    track_path = kwargs.get("audio_track_path")
    track_url = kwargs.get("audio_track_url")
    slide_duration = kwargs.get("slide_duration_s")
    config = orchestrator.PipelineConfig(
        output_dir=work_dir,
        contract=contract,
        render=render,
        audio_locked=True,
        audio_track_path=_coerce_audio_track_path(track_path),
        audio_track_url=_coerce_audio_track_url(track_url),
    )
    return orchestrator.run_slideshow(
        images,
        config=config,
        slide_duration_s=float(slide_duration) if isinstance(slide_duration, (int, float)) else 3.0,
        resume=kwargs.get("resume") is True,
    )


def main(
    argv: Sequence[str] | None = None,
    *,
    manager: _CampaignManagerProtocol | None = None,
) -> int:
    """Ejecuta la CLI de Kliptych.

    Args:
        argv: Argumentos de línea de comandos; por defecto ``sys.argv``.
        manager: CampaignManager inyectable para tests (DI).

    Returns:
        Código de salida del proceso (``0`` cuando la ejecución fue válida).
    """
    parser = argparse.ArgumentParser(
        prog="kliptych",
        description="Pipeline headless de Kliptych.",
    )
    _ = parser.add_argument("--version", action="version", version=__version__)
    _ = parser.add_argument("--verbose", "-v", action="store_true", help="activa logging DEBUG")
    _ = parser.add_argument("--quiet", "-q", action="store_true", help="solo muestra ERROR")
    subcommands = parser.add_subparsers(dest="command")
    _ = subcommands.add_parser("env", help="detecta el entorno local y lo imprime como JSON")
    analyze_parser = subcommands.add_parser("analyze", help="analiza una URL para Kliptych Studio")
    _add_analyze_arguments(analyze_parser)
    preview_parser = subcommands.add_parser(
        "preview", help="genera un JPEG de composición para Kliptych Studio"
    )
    _add_preview_arguments(preview_parser)
    studio_parser = subcommands.add_parser(
        "studio", help="frontera de proyecto para Kliptych Studio"
    )
    _add_studio_subcommands(studio_parser)
    ingest_parser = subcommands.add_parser(
        "ingest", help="ingiere un brief local y lo imprime como JSON"
    )
    _ = ingest_parser.add_argument(
        "path", help="ruta del brief (txt, md, pdf, docx) o '-' para texto pegado por stdin"
    )
    run_parser = subcommands.add_parser(
        "run", help="corre el pipeline given_clips y arma el paquete de entrega"
    )
    _ = run_parser.add_argument(
        "brief", nargs="?", default=None, help="ruta del brief (txt, md, pdf, docx)"
    )
    _ = run_parser.add_argument(
        "--brief", dest="brief_option", default=None, help="ruta del brief (txt, md, pdf, docx)"
    )
    _ = run_parser.add_argument(
        "--contract-draft", default=None, help="ruta del borrador de contrato JSON"
    )
    _ = run_parser.add_argument("--out", required=True, help="directorio del paquete de entrega")
    _ = run_parser.add_argument(
        "--recorded",
        default=None,
        help="directorio de grabaciones para reproducir sin llamar a un modelo real",
    )
    _ = run_parser.add_argument(
        "--root",
        default=None,
        help="raíz del workspace (por defecto KLIPTYCH_ROOT o el directorio actual)",
    )
    _ = run_parser.add_argument(
        "--approve-manual-review",
        action="store_true",
        help="aprueba piezas con advertencias de revisión manual en el Gate",
    )
    _ = run_parser.add_argument(
        "--approved-by",
        default=None,
        help="identificador del operador o sistema que aprueba la revisión manual",
    )
    run_group = run_parser.add_mutually_exclusive_group()
    _ = run_group.add_argument(
        "--resume",
        action="store_true",
        help="reanuda la ejecución desde el último punto de control",
    )
    _ = run_group.add_argument(
        "--restart",
        action="store_true",
        help="reinicia la ejecución descartando puntos de control previos",
    )
    campaign_parser = subcommands.add_parser(
        "campaign", help="procesa una campaña mediante clasificación y enrutamiento inteligente"
    )
    _ = campaign_parser.add_argument(
        "brief", nargs="?", default=None, help="ruta del brief (txt, md, pdf, docx)"
    )
    _ = campaign_parser.add_argument(
        "--brief", dest="brief_option", default=None, help="ruta del brief (txt, md, pdf, docx)"
    )
    _ = campaign_parser.add_argument(
        "--contract-draft", default=None, help="ruta del borrador de contrato JSON"
    )
    _ = campaign_parser.add_argument("--out", required=True, help="directorio de salida")
    _ = campaign_parser.add_argument(
        "--format",
        dest="output_format",
        default=None,
        choices=sorted(aspect.value for aspect in Aspect),
        help=(
            "aspecto del render y del veredicto del gate (default: el de "
            "RenderConfig). Las opciones son exactamente las de `Aspect`: "
            "agregar un formato al enum `Aspect` es un cambio de contrato y "
            "necesita su propia revision, no una opcion mas aqui"
        ),
    )
    _ = campaign_parser.add_argument(
        "--mode",
        default="long_video",
        choices=["long_video", "audio_locked", "repost", "repost_ugc", "slideshow"],
        help="modo de procesamiento (default: long_video)",
    )
    _ = campaign_parser.add_argument(
        "--url", default=None, help="URL del vídeo fuente (modo long_video)"
    )
    _ = campaign_parser.add_argument(
        "--audio-track-path",
        default=None,
        help="pista de audio local a inyectar (modo audio_locked)",
    )
    _ = campaign_parser.add_argument(
        "--audio-track-url",
        default=None,
        help="URL de la pista de audio a inyectar (modo audio_locked)",
    )
    _ = campaign_parser.add_argument(
        "--approve-manual-review",
        action="store_true",
        help="aprueba piezas con advertencias de revisión manual en el Gate",
    )
    _ = campaign_parser.add_argument(
        "--approved-by",
        default=None,
        help="identificador del operador o sistema que aprueba la revisión manual",
    )
    campaign_group = campaign_parser.add_mutually_exclusive_group()
    _ = campaign_group.add_argument(
        "--resume",
        action="store_true",
        help="reanuda la ejecución desde el último punto de control",
    )
    _ = campaign_group.add_argument(
        "--restart",
        action="store_true",
        help="reinicia la ejecución descartando puntos de control previos",
    )
    clean_parser = subcommands.add_parser("clean", help="elimina directorios temporales obsoletos")
    _ = clean_parser.add_argument(
        "--days",
        type=float,
        default=7.0,
        help="antigüedad mínima en días para eliminar (default: 7)",
    )
    _ = clean_parser.add_argument(
        "--root",
        default=None,
        help="directorio raíz donde escanear (por defecto 'campaigns')",
    )
    args = _parse_cli_arguments(parser, argv)
    setup_logging(
        verbose=cast("bool", getattr(args, "verbose", False)),
        quiet=cast("bool", getattr(args, "quiet", False)),
    )
    return _dispatch_command(args=args, parser=parser, manager=manager)


def _parse_cli_arguments(
    parser: argparse.ArgumentParser, argv: Sequence[str] | None
) -> argparse.Namespace:
    """Parse CLI arguments, reserving status two only for Studio cancellation.

    Existing commands retain argparse's historical status-two behavior. An
    invalid `analyze` invocation instead reports status one so a Tauri parent
    cannot confuse it with the documented controlled-cancellation status.

    Returns:
        The parsed command namespace.

    Raises:
        SystemExit: If arguments are invalid; `analyze` syntax errors use status
            one while every existing command retains argparse's status two.
    """
    effective_argv = tuple(sys.argv[1:] if argv is None else argv)
    try:
        return parser.parse_args(effective_argv)
    except SystemExit as error:
        if _is_analyze_invocation(effective_argv) and error.code == _ARGPARSE_USAGE_ERROR_STATUS:
            raise SystemExit(_ANALYZE_INVOCATION_ERROR_STATUS) from error
        raise


def _is_analyze_invocation(argv: Sequence[str]) -> bool:
    """Return whether the selected subcommand is `analyze`, ignoring global flags."""
    global_flags = frozenset({"--quiet", "-q", "--verbose", "-v"})
    command = next((argument for argument in argv if argument not in global_flags), None)
    return command == "analyze"


def _add_analyze_arguments(parser: argparse.ArgumentParser) -> None:
    """Register the machine-oriented B1 arguments on an existing parser."""
    _ = parser.add_argument("--url", required=True, help="URL del vídeo fuente")
    _ = parser.add_argument("--project-dir", required=True, help="directorio local del proyecto")
    _ = parser.add_argument("--project-name", required=True, help="nombre visible del proyecto")


def _add_preview_arguments(parser: argparse.ArgumentParser) -> None:
    """Register the exact-frame B2.2 preview command arguments."""
    _ = parser.add_argument("--project-dir", required=True, help="directorio local del proyecto")
    _ = parser.add_argument("--variant-id", required=True, help="UUID de la variante activa")
    _ = parser.add_argument("--position-id", required=True, help="UUID de la posición activa")
    _ = parser.add_argument(
        "--at-ms", required=True, type=int, help="timestamp absoluto del source"
    )


def _add_studio_project_argument(parser: argparse.ArgumentParser) -> None:
    """Register the project directory argument shared by Studio operations."""
    _ = parser.add_argument("--project-dir", required=True, help="directorio local del proyecto")


def _add_studio_subcommands(parser: argparse.ArgumentParser) -> None:
    """Register the narrow project operations consumed by the desktop shell."""
    studio_subcommands = parser.add_subparsers(dest="studio_command", required=True)
    studio_inspect = studio_subcommands.add_parser("inspect", help="inspecciona un proyecto Studio")
    _add_studio_project_argument(studio_inspect)
    studio_initialize = studio_subcommands.add_parser(
        "initialize", help="crea estado editable medido para un proyecto Studio"
    )
    _add_studio_project_argument(studio_initialize)
    studio_variant = studio_subcommands.add_parser(
        "create-variant", help="crea una variante inicial desde un candidato B1"
    )
    _add_studio_project_argument(studio_variant)
    _ = studio_variant.add_argument("--candidate-id", required=True, help="ID del candidato B1")
    _ = studio_variant.add_argument("--name", required=True, help="nombre visible de la variante")
    studio_update = studio_subcommands.add_parser(
        "update-position", help="persiste un layout validado para una posicion"
    )
    _add_studio_project_argument(studio_update)
    _ = studio_update.add_argument("--variant-id", required=True, help="ID de la variante")
    _ = studio_update.add_argument("--position-id", required=True, help="ID de la posicion")
    _ = studio_update.add_argument("--layout-json", required=True, help="LayoutSpec JSON validado")
    studio_split = studio_subcommands.add_parser(
        "split-position", help="divide una posicion en el playhead"
    )
    _add_studio_project_argument(studio_split)
    _ = studio_split.add_argument("--variant-id", required=True, help="ID de la variante")
    _ = studio_split.add_argument("--position-id", required=True, help="ID de la posicion")
    _ = studio_split.add_argument("--at-ms", required=True, type=int, help="marca de corte en ms")
    studio_delete = studio_subcommands.add_parser(
        "delete-position", help="elimina una posicion preservando cobertura"
    )
    _add_studio_project_argument(studio_delete)
    _ = studio_delete.add_argument("--variant-id", required=True, help="ID de la variante")
    _ = studio_delete.add_argument("--position-id", required=True, help="ID de la posicion")
    _ = studio_delete.add_argument(
        "--absorb", choices=["previous", "next"], help="vecino que absorbe (interior)"
    )
    studio_duplicate = studio_subcommands.add_parser(
        "duplicate-variant", help="duplica una variante con nuevos IDs"
    )
    _add_studio_project_argument(studio_duplicate)
    _ = studio_duplicate.add_argument("--variant-id", required=True, help="ID de la variante")
    _ = studio_duplicate.add_argument("--name", required=True, help="nombre de la copia")
    studio_import_wm = studio_subcommands.add_parser(
        "import-watermark", help="importa un PNG local como watermark"
    )
    _add_studio_project_argument(studio_import_wm)
    _ = studio_import_wm.add_argument("--variant-id", required=True, help="ID de la variante")
    _ = studio_import_wm.add_argument("--source", required=True, help="ruta del PNG local")
    _ = studio_import_wm.add_argument(
        "--position", default="bottom_right", help="zona del enum WatermarkPosition"
    )
    studio_set_wm = studio_subcommands.add_parser(
        "set-watermark", help="activa/desactiva o cambia la zona del watermark"
    )
    _add_studio_project_argument(studio_set_wm)
    _ = studio_set_wm.add_argument("--variant-id", required=True, help="ID de la variante")
    _ = studio_set_wm.add_argument(
        "--enabled", choices=["true", "false"], help="activar o desactivar"
    )
    _ = studio_set_wm.add_argument("--position", help="zona del enum WatermarkPosition")


def _dispatch_command(
    *,
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
    manager: _CampaignManagerProtocol | None,
) -> int:
    """Route one parsed command to its thin CLI handler.

    Returns:
        The selected handler's process exit status.
    """
    command = cast("str | None", getattr(args, "command", None))
    match command:
        case "env":
            status = _cmd_env()
        case "analyze":
            status = _cmd_analyze(args)
        case "preview":
            status = _cmd_preview(args)
        case "studio":
            status = _cmd_studio(args)
        case "ingest":
            status = _cmd_ingest(args)
        case "run":
            status = _run_command(args)
        case "campaign":
            status = _cmd_campaign(args, manager=manager)
        case "clean":
            status = _cmd_clean(args)
        case _:
            parser.print_help()
            status = 0
    return status


def _cmd_env() -> int:
    report = detect_environment(SubprocessRunner())
    _ = sys.stdout.write(json.dumps(report.model_dump(mode="json"), indent=2) + "\n")
    return 0


def _cmd_analyze(args: argparse.Namespace) -> int:
    """Run the B1 Studio analysis protocol with NDJSON-only stdout.

    Returns:
        Zero on success, one on analysis failure, or two on cancellation.
    """
    run_id = str(uuid.uuid4())

    def emit(event_type: str, stage: str, **fields: object) -> None:
        event = {
            "schema_version": ANALYSIS_SCHEMA_VERSION,
            "run_id": run_id,
            "type": event_type,
            "stage": stage,
            **fields,
        }
        _ = sys.stdout.write(json.dumps(event, ensure_ascii=True, separators=(",", ":")) + "\n")
        _ = sys.stdout.flush()

    def cancel_handler(_signum: int, _frame: object) -> None:
        raise AnalysisCancelled

    request = AnalysisRequest(
        url=cast("str", args.url),
        project_dir=Path(cast("str", args.project_dir)),
        project_name=cast("str", args.project_name),
    )
    previous_term = signal.signal(signal.SIGTERM, cancel_handler)
    previous_int = signal.signal(signal.SIGINT, cancel_handler)
    try:
        document = analyze_project(
            request,
            downloader=MediaDownloader(),
            transcriber=FasterWhisperTranscriber(),
            progress=lambda stage: emit("progress", stage, percent=None),
        )
    except AnalysisCancelled:
        emit("cancelled", "cancelled")
        return 2
    except AnalysisEngineError as error:
        emit("error", "failed", error_code=error.code, message=error.message)
        return 1
    except ProjectStoreError as error:
        emit("error", "failed", error_code=error.code, message=error.message)
        return 1
    finally:
        _ = signal.signal(signal.SIGTERM, previous_term)
        _ = signal.signal(signal.SIGINT, previous_int)
    emit(
        "completed",
        "completed",
        analysis_path="analysis.json",
        candidate_count=len(document.candidates),
    )
    return 0


def _cmd_preview(args: argparse.Namespace) -> int:
    """Generate one structured B2.2 JPEG preview result.

    Returns:
        Zero on success or one for invalid request/project/preview errors.
    """
    project_dir = Path(cast("str", args.project_dir))
    try:
        request = PreviewRequest(
            project_dir=project_dir,
            variant_id=uuid.UUID(cast("str", args.variant_id)),
            position_id=uuid.UUID(cast("str", args.position_id)),
            at_ms=cast("int", args.at_ms),
        )
    except ValueError as error:
        _write_preview_event(
            {
                "type": "error",
                "error_code": "preview_request_invalid",
                "message": str(error),
            }
        )
        return 1
    try:
        result = generate_preview(request, probe=FFprobeProbe(), runner=SubprocessRunner())
    except PreviewError as error:
        _write_preview_event(
            {"type": "error", "error_code": "preview_failed", "message": error.message}
        )
        return 1
    _write_preview_event(
        {
            "type": "completed",
            "preview_path": result.preview_path.relative_to(project_dir).as_posix(),
            "source_frame_path": result.source_frame_path.relative_to(project_dir).as_posix(),
            "preview_fingerprint": result.preview_fingerprint,
            "frame_time_ms": result.frame_time_ms,
            "width": result.width,
            "height": result.height,
        }
    )
    return 0


def _cmd_studio(args: argparse.Namespace) -> int:
    """Run an inspect or initialization operation for the desktop shell.

    Returns:
        Zero on success and one when the project boundary rejects the operation.
    """
    project_dir = Path(cast("str", args.project_dir))
    try:
        store = open_project_store(project_dir)
        event = _studio_event(store, args=args)
    except ProjectStoreError as error:
        _write_studio_event({"type": "error", "error_code": error.code, "message": error.message})
        return 1
    _write_studio_event(event)
    return 0


def _studio_event(store: _StudioStore, *, args: argparse.Namespace) -> dict[str, object]:
    command = cast("str", args.studio_command)
    dispatch = {
        "inspect": lambda: _studio_inspect_event(store),
        "initialize": lambda: _studio_initialize_event(store),
        "create-variant": lambda: _studio_create_variant_event(
            store,
            candidate_id=cast("str", args.candidate_id),
            name=cast("str", args.name),
        ),
        "update-position": lambda: _studio_update_position_event(
            store,
            variant_id=cast("str", args.variant_id),
            position_id=cast("str", args.position_id),
            layout_json=cast("str", args.layout_json),
        ),
        "split-position": lambda: _studio_split_position_event(
            store,
            variant_id=cast("str", args.variant_id),
            position_id=cast("str", args.position_id),
            at_ms=cast("int", args.at_ms),
        ),
        "delete-position": lambda: _studio_delete_position_event(
            store,
            variant_id=cast("str", args.variant_id),
            position_id=cast("str", args.position_id),
            absorb=cast("str | None", getattr(args, "absorb", None)),
        ),
        "duplicate-variant": lambda: _studio_duplicate_variant_event(
            store,
            variant_id=cast("str", args.variant_id),
            name=cast("str", args.name),
        ),
        "import-watermark": lambda: _studio_import_watermark_event(
            store,
            variant_id=cast("str", args.variant_id),
            source=cast("str", args.source),
            position=cast("str", args.position),
        ),
        "set-watermark": lambda: _studio_set_watermark_event(
            store,
            variant_id=cast("str", args.variant_id),
            enabled=cast("str | None", getattr(args, "enabled", None)),
            position=cast("str | None", getattr(args, "position", None)),
        ),
    }
    handler = dispatch.get(command)
    if handler is None:
        msg = f"Unknown Studio command: {command}"
        raise ProjectStoreError(_STUDIO_COMMAND_INVALID, msg)
    return handler()


def _studio_inspect_event(store: _StudioStore) -> dict[str, object]:
    analysis = store.existing_analysis
    if analysis is None:
        msg = "Studio inspect requires a completed B1 analysis."
        raise ProjectStoreError(_PROJECT_ANALYSIS_MISSING, msg)
    project = store.read_project()
    return {
        "type": "project",
        "project_id": analysis.project_id,
        "project_name": analysis.project_name,
        "source_sha256": analysis.source.sha256,
        "candidates": [candidate.model_dump(mode="json") for candidate in analysis.candidates],
        "editable_state": "ready" if project is not None else "missing",
        "variants": []
        if project is None
        else [variant.model_dump(mode="json") for variant in project.variants],
    }


def _studio_initialize_event(store: _StudioStore) -> dict[str, object]:
    project = store.initialize_project(probe=FFprobeProbe())
    return {
        "type": "initialized",
        "project_id": project.project_id,
        "source_duration_ms": project.source_duration_ms,
    }


def _studio_create_variant_event(
    store: _StudioStore, *, candidate_id: str, name: str
) -> dict[str, object]:
    project = store.create_variant(candidate_id=candidate_id, name=name)
    variant = project.variants[-1]
    return {
        "type": "variant_created",
        "variant": variant.model_dump(mode="json"),
    }


def _studio_update_position_event(
    store: _StudioStore, *, variant_id: str, position_id: str, layout_json: str
) -> dict[str, object]:
    """Persist a validated layout update for one Studio position.

    Returns:
        A JSON event describing the updated variant.

    Raises:
        ProjectStoreError: If the layout or project boundary rejects the update.
        ValueError: If an identifier is not a valid UUID.
    """
    try:
        layout_spec = LayoutSpec.model_validate(json.loads(layout_json))
    except Exception as error:  # boundary validation
        msg = f"Invalid layout JSON: {error}"
        raise ProjectStoreError(_STUDIO_COMMAND_INVALID, msg) from error
    try:
        project = store.update_position(
            variant_id=uuid.UUID(variant_id),
            position_id=uuid.UUID(position_id),
            layout_spec=layout_spec,
        )
    except (ProjectStoreError, ValueError) as error:
        if isinstance(error, ProjectStoreError):
            raise
        msg = str(error)
        raise ProjectStoreError(_STUDIO_COMMAND_INVALID, msg) from error
    variant = next((item for item in project.variants if str(item.variant_id) == variant_id), None)
    if variant is None:
        msg = "Updated variant is missing from project."
        raise ProjectStoreError(_STUDIO_COMMAND_INVALID, msg)
    return {
        "type": "position_updated",
        "variant": variant.model_dump(mode="json"),
    }


def _studio_split_position_event(
    store: _StudioStore, *, variant_id: str, position_id: str, at_ms: int
) -> dict[str, object]:
    try:
        project = store.split_position(
            variant_id=uuid.UUID(variant_id),
            position_id=uuid.UUID(position_id),
            at_ms=at_ms,
        )
    except (ProjectStoreError, ValueError) as error:
        if isinstance(error, ProjectStoreError):
            raise
        msg = str(error)
        raise ProjectStoreError(_STUDIO_COMMAND_INVALID, msg) from error
    variant = next((item for item in project.variants if str(item.variant_id) == variant_id), None)
    if variant is None:
        msg = "Split variant is missing from project."
        raise ProjectStoreError(_STUDIO_COMMAND_INVALID, msg)
    return {"type": "position_split", "variant": variant.model_dump(mode="json")}


def _studio_delete_position_event(
    store: _StudioStore, *, variant_id: str, position_id: str, absorb: str | None
) -> dict[str, object]:
    neighbor = None if absorb is None else AbsorbNeighbor(absorb)
    try:
        project = store.delete_position(
            variant_id=uuid.UUID(variant_id),
            position_id=uuid.UUID(position_id),
            absorb_neighbor=neighbor,
        )
    except (ProjectStoreError, ValueError) as error:
        if isinstance(error, ProjectStoreError):
            raise
        msg = str(error)
        raise ProjectStoreError(_STUDIO_COMMAND_INVALID, msg) from error
    variant = next((item for item in project.variants if str(item.variant_id) == variant_id), None)
    if variant is None:
        msg = "Deleted position variant is missing from project."
        raise ProjectStoreError(_STUDIO_COMMAND_INVALID, msg)
    return {"type": "position_deleted", "variant": variant.model_dump(mode="json")}


def _studio_duplicate_variant_event(
    store: _StudioStore, *, variant_id: str, name: str
) -> dict[str, object]:
    try:
        project = store.duplicate_variant(variant_id=uuid.UUID(variant_id), name=name)
    except (ProjectStoreError, ValueError) as error:
        if isinstance(error, ProjectStoreError):
            raise
        msg = str(error)
        raise ProjectStoreError(_STUDIO_COMMAND_INVALID, msg) from error
    variant = project.variants[-1]
    return {"type": "variant_duplicated", "variant": variant.model_dump(mode="json")}


def _studio_import_watermark_event(
    store: _StudioStore, *, variant_id: str, source: str, position: str
) -> dict[str, object]:
    try:
        project = store.import_watermark(
            variant_id=uuid.UUID(variant_id),
            source_path=Path(source),
            position=WatermarkPosition(position),
        )
    except (ProjectStoreError, ValueError) as error:
        if isinstance(error, ProjectStoreError):
            raise
        msg = str(error)
        raise ProjectStoreError(_STUDIO_COMMAND_INVALID, msg) from error
    variant = next((item for item in project.variants if str(item.variant_id) == variant_id), None)
    if variant is None:
        msg = "Watermark variant is missing from project."
        raise ProjectStoreError(_STUDIO_COMMAND_INVALID, msg)
    return {"type": "watermark_imported", "variant": variant.model_dump(mode="json")}


def _studio_set_watermark_event(
    store: _StudioStore, *, variant_id: str, enabled: str | None, position: str | None
) -> dict[str, object]:
    enabled_value = None if enabled is None else enabled == "true"
    position_value = None if position is None else WatermarkPosition(position)
    try:
        project = store.set_watermark(
            variant_id=uuid.UUID(variant_id),
            enabled=enabled_value,
            position=position_value,
        )
    except (ProjectStoreError, ValueError) as error:
        if isinstance(error, ProjectStoreError):
            raise
        msg = str(error)
        raise ProjectStoreError(_STUDIO_COMMAND_INVALID, msg) from error
    variant = next((item for item in project.variants if str(item.variant_id) == variant_id), None)
    if variant is None:
        msg = "Watermark variant is missing from project."
        raise ProjectStoreError(_STUDIO_COMMAND_INVALID, msg)
    return {"type": "watermark_updated", "variant": variant.model_dump(mode="json")}


def _write_preview_event(event: dict[str, object]) -> None:
    """Write one JSON preview result to stdout without human log text."""
    payload = {"schema_version": ANALYSIS_SCHEMA_VERSION, **event}
    _ = sys.stdout.write(json.dumps(payload, ensure_ascii=True, separators=(",", ":")) + "\n")
    _ = sys.stdout.flush()


def _write_studio_event(event: dict[str, object]) -> None:
    """Write one JSON Studio project operation event to stdout."""
    _write_preview_event(event)


def _cmd_ingest(args: argparse.Namespace) -> int:
    brief_path = cast("str", getattr(args, "path", ""))
    try:
        brief = (
            ingest_bytes(sys.stdin.buffer.read(MAX_BRIEF_BYTES + 1), source="<stdin>")
            if brief_path == "-"
            else ingest_file(Path(brief_path))
        )
    except IngestError as error:
        _ = sys.stderr.write(json.dumps({"error": str(error)}, indent=2) + "\n")
        return 1
    _ = sys.stdout.write(
        json.dumps(
            {
                "source": brief.source,
                "media_type": brief.media_type,
                "sha256": brief.sha256,
                "chars": len(brief.text),
                "text": brief.text,
            },
            indent=2,
        )
        + "\n"
    )
    return 0


def _is_json_document(text: str) -> bool:
    stripped = text.strip()
    if (stripped.startswith("{") and stripped.endswith("}")) or (
        stripped.startswith("[") and stripped.endswith("]")
    ):
        try:
            parsed = cast("object", json.loads(stripped))
            return isinstance(parsed, (dict, list))
        except (ValueError, TypeError):
            return False
    return False


def _load_campaign_contract(
    brief_text: str,
    contract_draft_path: str | None,
) -> tuple[Contract | None, str | None]:
    if contract_draft_path is None:
        return None, None
    try:
        draft_text = Path(contract_draft_path).read_text(encoding="utf-8")
        draft = ContractDraft.model_validate_json(draft_text)
        contract = resolve_contract(
            draft,
            registry=AssetRegistry(Path.cwd()),
            brief_text=brief_text,
        ).contract
    except ProvenanceError as error:
        return None, f"Error de procedencia en el borrador de contrato: {error}"
    except (ValidationError, ValueError) as error:
        return None, f"Error al validar el borrador de contrato: {error}"
    else:
        return contract, None


def _handle_campaign_result(result: CampaignOutcome) -> int:
    if result.error is not None:
        logger.error("La campaña falló: %s", result.error)
        return 1
    if result.pull_request is not None:
        logger.info("Propuesta creada: %s", result.pull_request.url)
        return 0
    if result.status != CampaignStatus.COMPLETED:
        if result.delivery_report is not None:
            logger.error(
                "La campaña no se completó: estado=%s, delivery_status=%s",
                result.status.value,
                result.delivery_report.status.value,
            )
            for rejected in result.delivery_report.rejected:
                logger.error(
                    "Pieza rechazada: %s [%s] -> %s",
                    rejected.piece_id,
                    rejected.platform.value,
                    rejected.reason,
                )
        else:
            logger.error("La campaña no se completó: estado=%s", result.status.value)
        return 1
    if result.pipeline_result is not None:
        logger.info("Video final: %s", result.pipeline_result.final_video)
        return 0
    logger.info("Estado: %s", result.status.value)
    return 0


def _cmd_campaign(args: argparse.Namespace, *, manager: _CampaignManagerProtocol | None) -> int:
    brief_path = cast(
        "str", getattr(args, "brief_option", None) or getattr(args, "brief", "") or ""
    )
    out_dir = cast("str", getattr(args, "out", ""))
    dest_path = Path(out_dir) if out_dir else None
    url = cast("str | None", getattr(args, "url", None))

    if not brief_path:
        return _cmd_campaign_zero_contract(
            args,
            manager=manager,
            destination=dest_path,
            url=url,
        )

    try:
        brief = ingest_file(Path(brief_path))
    except IngestError:
        logger.exception("Error al ingerir el brief de campaña")
        return 1
    if Path(brief_path).suffix.lower() == ".json" or _is_json_document(brief.text):
        msg = (
            "Error de procedencia: el brief no puede ser un archivo JSON; "
            "se requiere documento fuente markdown/texto y --contract-draft por separado"
        )
        logger.error("%s", msg)
        return 1
    logger.info(
        "Procesando campaña %s en modo %s",
        Path(brief_path).name,
        getattr(args, "mode", "long_video"),
    )
    logger.info("Directorio de salida: %s", out_dir)

    contract, err = _load_campaign_contract(brief.text, getattr(args, "contract_draft", None))
    if err is not None:
        logger.error("%s", err)
        return 1

    campaign = Campaign(
        campaign_id=brief.sha256[:12],
        brief=brief.text,
        status=CampaignStatus.PENDING,
        contract=contract,
    )
    try:
        active_manager = (
            manager
            if manager is not None
            else _make_default_campaign_manager(
                destination=dest_path / "delivery" if dest_path else None,
                output_format=cast("str | None", getattr(args, "output_format", None)),
            )
        )
    except RuntimeError:
        logger.exception("Error de configuración")
        return 1

    return _handle_campaign_result(_process_campaign(active_manager, campaign, args))


_ZERO_CONTRACT_IGNORED_FLAGS = (
    ("contract_draft", "--contract-draft"),
    ("audio_track_path", "--audio-track-path"),
    ("audio_track_url", "--audio-track-url"),
)
# Modos que el modo zero-contract HONRA de verdad: el gestor fuerza repost
# porque es el único camino sin selección LLM. La lista es de los que se
# aceptan, no de los que se rechazan (#55): con una denylist, cualquier modo
# nuevo que se añadiera a `--mode` colaría en silencio y se ejecutaría como
# repost debajo, que es exactamente el fallo que se quiere evitar. Fail-closed
# significa que lo no listado se avisa.
_ZERO_CONTRACT_HONORED_MODES = frozenset({"long_video", "repost", "repost_ugc"})


def _ignored_zero_contract_flags(args: argparse.Namespace) -> list[str]:
    """Lista los flags que el modo zero-contract acepta pero no puede honrar.

    Sin brief el contrato se sintetiza, así que ``--contract-draft`` no tiene
    sobre qué actuar; y el modo efectivo es repost, que no recibe pista de
    audio externa ni monta slideshows. Aceptarlos en silencio devolvería
    exit 0 con una entrega distinta de la pedida, así que se avisa en vez de
    descartarlos.

    Args:
        args: Argumentos parseados del subcomando ``campaign``.

    Returns:
        Los nombres de los flags presentes y de los modos no honrados.
    """
    ignored: list[str] = []
    for attribute, flag in _ZERO_CONTRACT_IGNORED_FLAGS:
        value = getattr(args, attribute, None)
        if value not in {None, ""}:
            ignored.append(flag)
    mode = cast("str", getattr(args, "mode", ""))
    if mode and mode not in _ZERO_CONTRACT_HONORED_MODES:
        ignored.append(f"--mode {mode}")
    return ignored


def _cmd_campaign_zero_contract(
    args: argparse.Namespace,
    *,
    manager: _CampaignManagerProtocol | None,
    destination: Path | None,
    url: str | None,
) -> int:
    """Procesa la campaña sin brief: contrato vainilla y ninguna llamada al LLM.

    Es la vía de un solo argumento (``--url``). No se ingiere ningún brief, no
    se extrae contrato y no se exige ``KLIPTYCH_LLM_*``: el contrato se
    sintetiza en memoria y el modo se fuerza a repost, el único camino que no
    selecciona segmentos con el LLM.

    Args:
        args: Argumentos parseados del subcomando ``campaign``.
        manager: Manager inyectado, si la prueba lo provee.
        destination: Directorio de salida resuelto de ``--out``.
        url: URL del vídeo fuente.

    Returns:
        ``0`` si la campaña se procesó, ``1`` si falta la URL o falla la config.
    """
    if url is None or not url.strip():
        logger.error("Se requiere --url cuando no se pasa la ruta del brief")
        return 1
    ignored = _ignored_zero_contract_flags(args)
    if ignored:
        logger.warning(
            "el modo zero-contract ignora %s: sin brief no hay contrato que sobreescribir",
            ", ".join(ignored),
        )
    campaign_id = brief_key(url.strip())[:12]
    logger.info(
        "Modo zero-contract: contrato vainilla para la campaña %s, modo efectivo repost_ugc",
        campaign_id,
    )
    logger.info("Directorio de salida: %s", destination if destination is not None else "")
    campaign = Campaign(
        campaign_id=campaign_id,
        brief="",
        status=CampaignStatus.PENDING,
        contract=vanilla_contract(campaign_id=campaign_id),
    )
    try:
        active_manager = (
            manager
            if manager is not None
            else _make_default_campaign_manager(
                destination=destination / "delivery" if destination else None,
                require_llm=False,
                output_format=cast("str | None", getattr(args, "output_format", None)),
            )
        )
    except RuntimeError:
        logger.exception("Error de configuración")
        return 1
    return _handle_campaign_result(_process_campaign(active_manager, campaign, args))


def _process_campaign(
    active_manager: _CampaignManagerProtocol, campaign: Campaign, args: argparse.Namespace
) -> CampaignOutcome:
    """Ejecuta el manager de campaña con los argumentos del subcomando.

    Args:
        active_manager: Manager inyectado o construido por defecto.
        campaign: Campaña con su brief y contrato validado.
        args: Argumentos parseados del subcomando campaign.

    Returns:
        El resultado del procesamiento de la campaña.
    """
    mode = cast("str", getattr(args, "mode", "long_video"))
    url = cast("str | None", getattr(args, "url", None))
    resume = cast("bool", getattr(args, "resume", False))
    approve_manual_review = cast("bool", getattr(args, "approve_manual_review", False))
    approved_by = cast("str | None", getattr(args, "approved_by", None))
    if approve_manual_review and (approved_by is None or not approved_by.strip()):
        return CampaignOutcome(
            campaign_id=campaign.campaign_id,
            archetype=Archetype.NEW_ARCHETYPE,
            status=CampaignStatus.PENDING,
            error="se requiere --approved-by cuando --approve-manual-review está activo",
        )
    audio_track_path, audio_track_url = _campaign_audio_tracks(args)

    return active_manager.process(
        campaign,
        mode=mode,
        url=url,
        resume=resume,
        approve_manual_review=approve_manual_review,
        approved_by=approved_by,
        audio_track_path=audio_track_path,
        audio_track_url=audio_track_url,
        output_format=cast("str | None", getattr(args, "output_format", None)),
    )


def _campaign_audio_tracks(args: argparse.Namespace) -> tuple[Path | None, str | None]:
    """Extrae la pista de audio externa de los argumentos de campaña.

    Args:
        args: Argumentos parseados del subcomando campaign.

    Returns:
        La ruta local (o None) y la URL (o None) de la pista externa.
    """
    track_path_arg = cast("str | None", getattr(args, "audio_track_path", None))
    track_url = cast("str | None", getattr(args, "audio_track_url", None))
    return (Path(track_path_arg) if track_path_arg else None, track_url)


def _load_run_brief(brief_path: str) -> str:
    brief = ingest_file(Path(brief_path))
    if Path(brief_path).suffix.lower() == ".json" or _is_json_document(brief.text):
        msg = (
            "Error de procedencia: el brief no puede ser un archivo JSON; "
            "se requiere documento fuente markdown/texto"
        )
        raise IngestError(msg)
    return brief.text


def _load_contract_draft_file(path_str: str | None) -> tuple[ContractDraft | None, str | None]:
    if path_str is None:
        return None, None
    try:
        text = Path(path_str).read_text(encoding="utf-8")
        return ContractDraft.model_validate_json(text), None
    except (ValidationError, OSError, ValueError) as error:
        return None, f"Error en contract_draft: {error}"


def _run_command(args: argparse.Namespace) -> int:
    """Punto de entrada de `kliptych run`.

    Args:
        args: Argumentos parseados del comando.

    Returns:
        El codigo de salida del proceso.
    """
    brief_path = cast(
        "str",
        getattr(args, "brief_option", None) or getattr(args, "brief", "") or "",
    )
    if not brief_path:
        _ = sys.stderr.write(
            json.dumps({"error": "Se requiere la ruta del brief"}, indent=2) + "\n"
        )
        return 1
    out = cast("str", getattr(args, "out", ""))
    approve_manual_review = cast("bool", getattr(args, "approve_manual_review", False))
    approved_by = cast("str | None", getattr(args, "approved_by", None))
    if approve_manual_review and (approved_by is None or not approved_by.strip()):
        _ = sys.stderr.write(
            json.dumps(
                {"error": "se requiere --approved-by cuando --approve-manual-review está activo"},
                indent=2,
            )
            + "\n"
        )
        return 1

    contract_draft, draft_err = _load_contract_draft_file(
        cast("str | None", getattr(args, "contract_draft", None))
    )
    if draft_err is not None:
        _ = sys.stderr.write(json.dumps({"error": draft_err}, indent=2) + "\n")
        return 1

    try:
        peticion = _construir_peticion_run(
            args,
            root=getattr(args, "root", None),
            out=out,
            brief_path=brief_path,
            approve_manual_review=approve_manual_review,
            approved_by=approved_by,
            contract_draft=contract_draft,
        )
        result = run_given_clips(
            model=peticion.model, settings=peticion.settings, request=peticion.run
        )
    except (IngestError, ModelError, PipelineError, AssembleError, ExportError) as error:
        _ = sys.stderr.write(json.dumps({"error": str(error)}, indent=2) + "\n")
        return 1
    _ = sys.stdout.write(json.dumps(_run_payload(result), indent=2) + "\n")
    return 0 if result.outcome is RunOutcome.EXPORTED else 1


@dataclass(frozen=True, slots=True)
class _PeticionRun:
    """La peticion de corrida junto al modelo que la resuelve."""

    run: RunRequest
    model: CampaignModel
    settings: Settings


def _construir_peticion_run(
    args: argparse.Namespace,
    *,
    root: str | None,
    out: str,
    brief_path: str,
    approve_manual_review: bool,
    approved_by: str | None,
    contract_draft: ContractDraft | None,
) -> _PeticionRun:
    """Construye la peticion de `kliptych run` con el entorno ya detectado.

    Un solo `detect_environment()` por corrida: alimenta el manifiesto Y la
    config de render. Antes el manifiesto recibia la deteccion y el
    `RenderConfig` se construia con el default, asi que NVENC nunca llegaba a
    ningun render aunque la maquina lo tuviera.

    Args:
        args: Argumentos parseados del comando.
        root: Raiz del workspace, o ``None`` para usar la del entorno.
        out: Directorio de salida de la corrida.
        brief_path: Ruta del brief ya validada.
        approve_manual_review: Si se admite aprobacion manual.
        approved_by: Responsable de la aprobacion, si la hay.
        contract_draft: Borrador de contrato ya cargado, o ``None``.

    Returns:
        La peticion, el modelo y las rutas resueltas.
    """
    brief_text = _load_run_brief(
        cast("str", getattr(args, "brief_option", None) or getattr(args, "brief", ""))
    )
    model, model_version = _build_model(cast("str | None", getattr(args, "recorded", None)))
    settings = Settings.from_root(Path(root)) if root is not None else Settings.from_env()
    environment = detect_environment(SubprocessRunner())
    request = RunRequest(
        brief=brief_text,
        destination=Path(out),
        environment=environment,
        model_version=model_version,
        prompt_version=PROMPT_VERSION,
        caption_prompt_version=CAPTION_PROMPT_VERSION,
        brief_path=Path(brief_path),
        approve_manual_review=approve_manual_review,
        approved_by=approved_by,
        contract_draft=contract_draft,
        # La CLI NO declara todavia la tolerancia: no hay bandera para
        # cambiarla, asi que esto equivale al default de la clase y el
        # pipeline haria lo mismo con `render=None`. Se pone de forma
        # explicita para que el punto de entrada sea visible como UNO de los
        # dos, y no para insinuar que la CLI puede elegir la geometria.
        # Cuando exista la bandera, este es el sitio.
        # El encoder se decide aqui, donde esta la deteccion. Los cinco
        # lectores de `nvenc_available` leen el campo; esta es la escritura.
        render=RenderConfig(nvenc_available=environment.nvenc_available),
    )
    return _PeticionRun(run=request, model=model, settings=settings)


def _cmd_clean(args: argparse.Namespace) -> int:
    days = cast("float", getattr(args, "days", 7.0))
    root_arg = cast("str | None", getattr(args, "root", None))
    target = Path(root_arg) if root_arg is not None else Path("campaigns")
    if not target.exists():
        logger.info("Directorio no encontrado: %s", target)
        _ = sys.stdout.write("[]\n")
        return 0
    try:
        removed = clean_temporary_directories(target, days=days)
    except ValueError:
        logger.exception("Parámetro inválido")
        return 1
    logger.info("Limpieza completada: %d directorios eliminados", len(removed))
    _ = sys.stdout.write(json.dumps([str(p) for p in removed], indent=2) + "\n")
    return 0


def _build_model(recorded: str | None) -> tuple[CampaignModel, str]:
    if recorded is not None:
        try:
            model = RecordedModel.from_directory(
                Path(recorded),
                expected_prompt_version=PROMPT_VERSION,
                expected_caption_prompt_version=CAPTION_PROMPT_VERSION,
            )
        except ValidationError:
            msg = f"grabaciones inválidas en {recorded}: error de validación"
            raise ModelUnavailableError(msg) from None
        except ValueError:
            msg = f"grabaciones inválidas en {recorded}: contenido duplicado o ilegible"
            raise ModelUnavailableError(msg) from None
        return model, RecordedModel.model_version
    backend = OpenAIChatModel.from_env()
    return backend, backend.model_version


def _render_config_para(
    output_format: str | None, environment: EnvironmentReport | None = None
) -> RenderConfig:
    """La config de render de la corrida, derivada de `--format` y del entorno.

    La tolerancia no se toca: `--format` cambia el ASPECTO, que es lo que el
    render produce y lo que el gate mide. Un flag que tambien moviera la
    tolerancia daria dos ejes por un knob y haria imposible saber cual de los dos
    produjo un veredicto.

    `nvenc_available` sale de la deteccion. Sin esto el campo se quedaba en su
    default y los cinco lectores de todo el pipeline leian siempre `False`, con
    NVENC disponible en la maquina y sin usar.

    Args:
        output_format: El valor de `--format`, o ``None`` para no overriding.
        environment: Deteccion del entorno; ``None`` usa el default de la clase.

    Returns:
        La config con el aspecto objetivo pedido y el encoder disponible.

    Raises:
        ValueError: Si `output_format` no corresponde a un miembro de `Aspect`.
    """
    nvenc = environment.nvenc_available if environment is not None else False
    if output_format is None:
        return RenderConfig(nvenc_available=nvenc)
    try:
        aspecto = Aspect(output_format)
    except ValueError as error:
        msg = f"aspecto de salida no soportado: {output_format!r}"
        raise ValueError(msg) from error
    return RenderConfig(target_aspect=aspecto.ratio, nvenc_available=nvenc)


def _make_default_campaign_manager(
    destination: Path | None = None,
    gate: Gate | None = None,
    assets: AssetRegistry | None = None,
    *,
    require_llm: bool = True,
    output_format: str | None = None,
) -> _CampaignManagerProtocol:
    """Construye el gestor de campañas real.

    Args:
        destination: Directorio de la entrega; por defecto ``./delivery``.
        gate: Gate de validación; por defecto el de ffprobe.
        assets: Registro de assets; por defecto el del directorio actual.
        require_llm: Si es True exige ``KLIPTYCH_LLM_*`` y cablea el
            clasificador y el modelo de segmentos con el backend LLM. El modo
            zero-contract pasa False: sin brief no hay nada que clasificar ni
            que seleccionar, así que no debe exigir credenciales ni construir
            un cliente de chat que nunca se usaría.
        output_format: Aspecto pedido con ``--format`` (los valores del enum
            `Aspect`), o ``None`` para el default de `RenderConfig`. Cuando se
            pasa, la config que sale de aqui es la que reciben el gate Y el
            orquestador: son el mismo objeto.

    Returns:
        El gestor configurado.

    Raises:
        ValueError: Si `output_format` no es un aspecto soportado por `Aspect`.

    Raises:
        RuntimeError: Si ``require_llm`` y falta alguna variable ``KLIPTYCH_LLM_*``.
    """
    source = os.environ
    classifier: CampaignClassifier | None
    backend: OpenAIChatModel | None
    if require_llm:
        missing = [
            name
            for name in ("KLIPTYCH_LLM_BASE_URL", "KLIPTYCH_LLM_API_KEY", "KLIPTYCH_LLM_MODEL")
            if not source.get(name)
        ]
        if missing:
            msg = f"faltan variables de entorno: {', '.join(missing)}"
            raise RuntimeError(msg)
        classifier = LLMCampaignClassifier(
            base_url=source["KLIPTYCH_LLM_BASE_URL"],
            api_key=source["KLIPTYCH_LLM_API_KEY"],
            model=source["KLIPTYCH_LLM_MODEL"],
        )
        backend = OpenAIChatModel(
            base_url=source["KLIPTYCH_LLM_BASE_URL"],
            api_key=source["KLIPTYCH_LLM_API_KEY"],
            model=source["KLIPTYCH_LLM_MODEL"],
        )
    else:
        classifier = None
        backend = None
    repo = os.environ.get("KLIPTYCH_GIT_REPO", "owner/repo")
    provider = GitHubCliProvider(workdir=Path.cwd(), repo=repo)
    # Una sola deteccion para las dos cosas que la necesitan: la config de render
    # y el entorno que va al manifiesto.
    environment = detect_environment(SubprocessRunner())
    # Una sola config para gate y orquestador. Que compartan el objeto es lo que
    # hace que "el aspecto con el que se acepto no reencuadrar" sea el mismo en
    # los dos lados de la comparacion. Con `--format` esto ya no es el default de
    # la clase: antes lo era siempre, y entonces `--format` no habria podido
    # cambiar nada aunque existiera.
    render = _render_config_para(output_format, environment)
    effective_gate = gate if gate is not None else Gate(probe=FFprobeProbe(), render=render)
    effective_dest = destination if destination is not None else Path.cwd() / "delivery"
    effective_assets = assets if assets is not None else AssetRegistry(Path.cwd())
    work_dir = effective_dest.parent / f"{effective_dest.name}-work"
    segment_model = _ChatSegmentModel(backend) if backend is not None else None
    return CampaignManager(
        classifier=classifier,
        proposal_engine=ProposalEngine(provider=provider),
        video_orchestrator=_DefaultVideoOrchestrator(
            work_dir=work_dir, model=segment_model, render=render, assets=effective_assets
        ),
        slideshow_orchestrator=_DefaultSlideshowOrchestrator(work_dir=work_dir, render=render),
        gate=effective_gate,
        destination=effective_dest,
        assets=effective_assets,
        model=backend,
    )


_build_campaign_manager = _make_default_campaign_manager


def _run_payload(result: RunResult) -> dict[str, object]:
    payload: dict[str, object] = {
        "run_id": result.run_id,
        "outcome": result.outcome.value,
        "contract_sha256": result.contract_sha256,
        "package": result.package_path,
        "manifest": result.manifest_path,
    }
    if result.delivery is None:
        payload["issues"] = [issue.code.value for issue in result.issues]
        return payload
    payload["exported"] = [piece.piece_id for piece in result.delivery.exported]
    payload["rejected"] = [
        {
            "piece_id": piece.piece_id,
            "platform": piece.platform.value,
            "reason": piece.reason,
        }
        for piece in result.delivery.rejected
    ]
    return payload


if __name__ == "__main__":  # pragma: no cover - entrada directa
    raise SystemExit(main())
