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
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol, cast, runtime_checkable

from pydantic import ValidationError

from kliptych import __version__
from kliptych.assembler import AssembleError
from kliptych.assets import AssetRegistry
from kliptych.campaign_manager import CampaignManager, CampaignOutcome
from kliptych.campaign_types import Campaign, CampaignStatus
from kliptych.config import Settings
from kliptych.contract import Contract, ContractDraft
from kliptych.environment import SubprocessRunner, detect_environment
from kliptych.exporter import ExportError
from kliptych.gc import clean_temporary_directories
from kliptych.git_proposals import GitHubCliProvider, ProposalEngine
from kliptych.ingest import MAX_BRIEF_BYTES, IngestError, ingest_bytes, ingest_file
from kliptych.intelligence import LLMCampaignClassifier
from kliptych.logging_setup import setup_logging
from kliptych.pipeline import PipelineError, RunRequest, RunResult, run_given_clips
from kliptych.resolver import resolve_contract
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
        mode: str,
        url: str | None,
        images: Sequence[Path] | None,
        resume: bool = False,
    ) -> CampaignOutcome: ...


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
    _ = run_parser.add_argument("brief", help="ruta del brief (txt, md, pdf, docx)")
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
    _ = campaign_parser.add_argument("brief", help="ruta del brief (txt, md, pdf, docx)")
    _ = campaign_parser.add_argument("--out", required=True, help="directorio de salida")
    _ = campaign_parser.add_argument(
        "--mode",
        default="long_video",
        choices=["long_video", "repost", "slideshow"],
        help="modo de procesamiento (default: long_video)",
    )
    _ = campaign_parser.add_argument(
        "--url", default=None, help="URL del vídeo fuente (modo long_video)"
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


def _extract_contract(brief_text: str) -> Contract | None:
    try:
        draft = ContractDraft.model_validate_json(brief_text)
        return resolve_contract(
            draft,
            registry=AssetRegistry(Path.cwd()),
            brief_text=brief_text,
        ).contract
    except (ValidationError, ValueError):
        return None


def _cmd_campaign(args: argparse.Namespace, *, manager: _CampaignManagerProtocol | None) -> int:
    brief_path = cast("str", getattr(args, "brief", ""))
    out_dir = cast("str", getattr(args, "out", ""))
    mode = cast("str", getattr(args, "mode", "long_video"))
    url = cast("str | None", getattr(args, "url", None))
    try:
        brief = ingest_file(Path(brief_path))
    except IngestError:
        logger.exception("Error al ingerir el brief de campaña")
        return 1
    logger.info("Procesando campaña %s en modo %s", Path(brief_path).name, mode)
    logger.info("Directorio de salida: %s", out_dir)
    contract = _extract_contract(brief.text)
    campaign = Campaign(
        campaign_id=brief.sha256[:12],
        brief=brief.text,
        status=CampaignStatus.PENDING,
        contract=contract,
    )
    try:
        active_manager = manager if manager is not None else _build_campaign_manager()
    except RuntimeError:
        logger.exception("Error de configuración")
        return 1
    resume = cast("bool", getattr(args, "resume", False))
    result = active_manager.process(campaign, mode=mode, url=url, images=None, resume=resume)
    if result.error is not None:
        logger.error("La campaña falló: %s", result.error)
        return 1
    if result.pull_request is not None:
        logger.info("Propuesta creada: %s", result.pull_request.url)
        return 0
    if result.pipeline_result is not None:
        logger.info("Video final: %s", result.pipeline_result.final_video)
        return 0
    logger.info("Estado: %s", result.status.value)
    return 0


def _run_command(args: argparse.Namespace) -> int:
    brief_path = cast("str", getattr(args, "brief", ""))
    out = cast("str", getattr(args, "out", ""))
    recorded = cast("str | None", getattr(args, "recorded", None))
    root = cast("str | None", getattr(args, "root", None))
    try:
        brief = ingest_file(Path(brief_path))
        model, model_version = _build_model(recorded)
        settings = Settings.from_root(Path(root)) if root is not None else Settings.from_env()
        request = RunRequest(
            brief=brief.text,
            destination=Path(out),
            environment=detect_environment(SubprocessRunner()),
            model_version=model_version,
            prompt_version=PROMPT_VERSION,
            caption_prompt_version=CAPTION_PROMPT_VERSION,
            brief_path=Path(brief_path),
        )
        result = run_given_clips(model=model, settings=settings, request=request)
    except (IngestError, ModelError, PipelineError, AssembleError, ExportError) as error:
        _ = sys.stderr.write(json.dumps({"error": str(error)}, indent=2) + "\n")
        return 1
    _ = sys.stdout.write(json.dumps(_run_payload(result), indent=2) + "\n")
    return 0


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


def _build_campaign_manager() -> _CampaignManagerProtocol:
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
    repo = os.environ.get("KLIPTYCH_GIT_REPO", "owner/repo")
    provider = GitHubCliProvider(workdir=Path.cwd(), repo=repo)
    return CampaignManager(
        classifier=classifier,
        proposal_engine=ProposalEngine(provider=provider),
    )


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
