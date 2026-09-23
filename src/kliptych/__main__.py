"""Punto de entrada de línea de comandos de Kliptych."""

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import cast

from kliptych import __version__
from kliptych.environment import SubprocessRunner, detect_environment
from kliptych.ingest import IngestError, ingest_file


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
    _ = ingest_parser.add_argument("path", help="ruta del brief (txt, md, pdf, docx)")
    args = parser.parse_args(argv)
    command = cast("str | None", getattr(args, "command", None))
    if command == "env":
        report = detect_environment(SubprocessRunner())
        print(json.dumps(report.model_dump(mode="json"), indent=2, ensure_ascii=False))
        return 0
    if command == "ingest":
        brief_path = Path(cast("str", getattr(args, "path", "")))
        try:
            brief = ingest_file(brief_path)
        except IngestError as error:
            print(json.dumps({"error": str(error)}, indent=2, ensure_ascii=False), file=sys.stderr)
            return 1
        print(
            json.dumps(
                {
                    "source": brief.source,
                    "media_type": brief.media_type,
                    "sha256": brief.sha256,
                    "chars": len(brief.text),
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        return 0
    parser.print_help()
    return 0


if __name__ == "__main__":  # pragma: no cover - entrada directa
    raise SystemExit(main())
