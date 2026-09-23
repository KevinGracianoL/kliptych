r"""Punto de entrada de línea de comandos de Kliptych.

La salida JSON es ASCII-safe (escapes ``\uXXXX``): así stdout y stderr se
pueden redirigir a otro proceso sin depender del codepage de la consola.
"""

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import cast

from pydantic import ValidationError

from kliptych import __version__
from kliptych.assembler import AssembleError
from kliptych.config import Settings
from kliptych.environment import SubprocessRunner, detect_environment
from kliptych.exporter import ExportError
from kliptych.ingest import MAX_BRIEF_BYTES, IngestError, ingest_bytes, ingest_file
from kliptych.pipeline import PipelineError, RunRequest, RunResult, run_given_clips
from kliptych.runtime import (
    CAPTION_PROMPT_VERSION,
    PROMPT_VERSION,
    CampaignModel,
    ModelError,
    ModelUnavailableError,
    OpenAIChatModel,
    RecordedModel,
)


def main(argv: Sequence[str] | None = None) -> int:
    """Ejecuta la CLI de Kliptych.

    Args:
        argv: Argumentos de línea de comandos; por defecto ``sys.argv``.

    Returns:
        Código de salida del proceso (``0`` cuando la ejecución fue válida).
    """
    parser = argparse.ArgumentParser(
        prog="kliptych",
        description="Pipeline headless de Kliptych.",
    )
    _ = parser.add_argument("--version", action="version", version=__version__)
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
    args = parser.parse_args(argv)
    command = cast("str | None", getattr(args, "command", None))
    if command == "env":
        report = detect_environment(SubprocessRunner())
        print(json.dumps(report.model_dump(mode="json"), indent=2))
        return 0
    if command == "ingest":
        brief_path = cast("str", getattr(args, "path", ""))
        try:
            brief = (
                ingest_bytes(sys.stdin.buffer.read(MAX_BRIEF_BYTES + 1), source="<stdin>")
                if brief_path == "-"
                else ingest_file(Path(brief_path))
            )
        except IngestError as error:
            print(json.dumps({"error": str(error)}, indent=2), file=sys.stderr)
            return 1
        print(
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
        )
        return 0
    if command == "run":
        return _run_command(args)
    parser.print_help()
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
        print(json.dumps({"error": str(error)}, indent=2), file=sys.stderr)
        return 1
    print(json.dumps(_run_payload(result), indent=2))
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
