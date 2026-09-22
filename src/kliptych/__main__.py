"""Punto de entrada de línea de comandos de Kliptych."""

import argparse
import json
from collections.abc import Sequence
from typing import cast

from kliptych import __version__
from kliptych.environment import SubprocessRunner, detect_environment


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
    args = parser.parse_args(argv)
    command = cast("str | None", getattr(args, "command", None))
    if command == "env":
        report = detect_environment(SubprocessRunner())
        print(json.dumps(report.model_dump(mode="json"), indent=2, ensure_ascii=False))
        return 0
    parser.print_help()
    return 0


if __name__ == "__main__":  # pragma: no cover - entrada directa
    raise SystemExit(main())
