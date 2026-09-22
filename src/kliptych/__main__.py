"""Punto de entrada de línea de comandos de Kliptych."""

import argparse
from collections.abc import Sequence

from kliptych import __version__


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
    _ = parser.parse_args(argv)
    parser.print_help()
    return 0


if __name__ == "__main__":  # pragma: no cover - entrada directa
    raise SystemExit(main())
