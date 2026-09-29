r"""Punto de entrada de línea de comandos de Kliptych.

La salida JSON es ASCII-safe (escapes ``\uXXXX``): así stdout y stderr se
pueden redirigir a otro proceso sin depender del codepage de la consola.
A partir de F1-PR1, todo output pasa por logging (prohibido ``print()``).
"""

import argparse
import json
import logging
import os
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Protocol, cast, runtime_checkable

from pydantic import ValidationError

from kliptych import __version__, orchestrator
from kliptych.assembler import AssembleError
from kliptych.assets import AssetRegistry
from kliptych.campaign_manager import CampaignManager, CampaignManagerError, CampaignOutcome
from kliptych.campaign_types import Campaign, CampaignStatus
from kliptych.config import Settings
from kliptych.contract import Contract, ContractDraft
from kliptych.encoding import RenderConfig
from kliptych.environment import SubprocessRunner, detect_environment
from kliptych.exporter import ExportError
from kliptych.gate import Gate
from kliptych.gate.probe import FFprobeProbe
from kliptych.gc import clean_temporary_directories
from kliptych.git_proposals import GitHubCliProvider, ProposalEngine
from kliptych.ingest import MAX_BRIEF_BYTES, IngestError, ingest_bytes, ingest_file
from kliptych.intelligence import Archetype, LLMCampaignClassifier
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

logger = logging.getLogger("kliptych.cli")


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
    ) -> CampaignOutcome: ...


_SEGMENT_SYSTEM_PROMPT = (
    "Eres el selector de segmentos de Kliptych. Recibes un JSON con la "
    "transcripción, los momentos candidatos y las cotas de duración, y "
    "devuelves ÚNICAMENTE un JSON con la forma "
    '{"segments": [{"start_s": float, "end_s": float}], "rationale": str}. '
    "Reglas: cada segmento queda dentro del vídeo y respeta las duraciones "
    "mínima y máxima; el rationale explica la selección en una frase."
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

    def __init__(self, *, work_dir: Path, model: _ChatSegmentModel, render: RenderConfig) -> None:
        """Configura el directorio de trabajo, el modelo y el render.

        Args:
            work_dir: Directorio donde el pipeline publica sus artefactos.
            model: Modelo de selección de segmentos sobre el backend LLM.
            render: Binario, timeout y NVENC compartidos por los renders.
        """
        self._work_dir: Path = work_dir
        self._model: _ChatSegmentModel = model
        self._render: RenderConfig = render

    def run_long_video(self, url: str, **kwargs: object) -> orchestrator.PipelineResult:
        """Renderiza un vídeo largo, con audio o repost según los flags.

        Args:
            url: URL http/https del vídeo fuente.
            **kwargs: resume, contract, audio_locked, repost_mode,
                audio_track_path y audio_track_url.

        Returns:
            El resultado del pipeline long_video.
        """
        config = self._pipeline_config(
            kwargs.get("contract"),
            audio_locked=kwargs.get("audio_locked") is True,
            repost_mode=_is_repost_mode(kwargs.get("repost_mode"), kwargs.get("mode")),
            audio_track_path=kwargs.get("audio_track_path"),
            audio_track_url=kwargs.get("audio_track_url"),
        )
        resume = kwargs.get("resume") is True
        if config.repost_mode:
            return orchestrator.run_repost(url, model=self._model, config=config, resume=resume)
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
    ) -> orchestrator.PipelineConfig:
        """Construye la config del pipeline validando el contrato y la pista.

        Args:
            contract: Contrato validado de la campaña.
            audio_locked: Si se inyecta pista de audio externa.
            repost_mode: Si se usa el vídeo completo sin inteligencia.
            audio_track_path: Ruta local de la pista externa, o None.
            audio_track_url: URL de la pista externa, o None.

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
        return orchestrator.PipelineConfig(
            output_dir=self._work_dir,
            contract=contract,
            render=self._render,
            audio_locked=audio_locked,
            audio_track_path=track_path,
            audio_track_url=track_url,
            repost_mode=repost_mode,
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
    args = parser.parse_args(argv)
    setup_logging(
        verbose=cast("bool", getattr(args, "verbose", False)),
        quiet=cast("bool", getattr(args, "quiet", False)),
    )
    command = cast("str | None", getattr(args, "command", None))
    if command == "env":
        return _cmd_env()
    if command == "ingest":
        return _cmd_ingest(args)
    if command == "run":
        return _run_command(args)
    if command == "campaign":
        return _cmd_campaign(args, manager=manager)
    if command == "clean":
        return _cmd_clean(args)
    parser.print_help()
    return 0


def _cmd_env() -> int:
    report = detect_environment(SubprocessRunner())
    _ = sys.stdout.write(json.dumps(report.model_dump(mode="json"), indent=2) + "\n")
    return 0


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
    if not brief_path:
        logger.error("Se requiere la ruta del brief")
        return 1
    out_dir = cast("str", getattr(args, "out", ""))
    dest_path = Path(out_dir) if out_dir else None

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
                destination=dest_path / "delivery" if dest_path else None
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
    recorded = cast("str | None", getattr(args, "recorded", None))
    root = cast("str | None", getattr(args, "root", None))
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
        brief_text = _load_run_brief(brief_path)
        model, model_version = _build_model(recorded)
        settings = Settings.from_root(Path(root)) if root is not None else Settings.from_env()
        request = RunRequest(
            brief=brief_text,
            destination=Path(out),
            environment=detect_environment(SubprocessRunner()),
            model_version=model_version,
            prompt_version=PROMPT_VERSION,
            caption_prompt_version=CAPTION_PROMPT_VERSION,
            brief_path=Path(brief_path),
            approve_manual_review=approve_manual_review,
            approved_by=approved_by,
            contract_draft=contract_draft,
        )
        result = run_given_clips(model=model, settings=settings, request=request)
    except (IngestError, ModelError, PipelineError, AssembleError, ExportError) as error:
        _ = sys.stderr.write(json.dumps({"error": str(error)}, indent=2) + "\n")
        return 1
    _ = sys.stdout.write(json.dumps(_run_payload(result), indent=2) + "\n")
    return 0 if result.outcome is RunOutcome.EXPORTED else 1


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


def _make_default_campaign_manager(
    destination: Path | None = None,
    gate: Gate | None = None,
    assets: AssetRegistry | None = None,
) -> _CampaignManagerProtocol:
    source = os.environ
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
    repo = os.environ.get("KLIPTYCH_GIT_REPO", "owner/repo")
    provider = GitHubCliProvider(workdir=Path.cwd(), repo=repo)
    effective_gate = gate if gate is not None else Gate(probe=FFprobeProbe())
    effective_dest = destination if destination is not None else Path.cwd() / "delivery"
    effective_assets = assets if assets is not None else AssetRegistry(Path.cwd())
    render = RenderConfig()
    work_dir = effective_dest.parent / f"{effective_dest.name}-work"
    segment_model = _ChatSegmentModel(backend)
    return CampaignManager(
        classifier=classifier,
        proposal_engine=ProposalEngine(provider=provider),
        video_orchestrator=_DefaultVideoOrchestrator(
            work_dir=work_dir, model=segment_model, render=render
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
