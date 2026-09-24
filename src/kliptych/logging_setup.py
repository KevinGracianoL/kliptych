"""Configuración de logging estructurado para Kliptych."""

import logging
import sys


def setup_logging(*, verbose: bool = False, quiet: bool = False) -> None:
    """Configura el logging estructurado de la aplicación.

    Args:
        verbose: Si es True, fija el nivel a DEBUG.
        quiet: Si es True, fija el nivel a ERROR (ignorado si verbose es True).
    """
    level = logging.DEBUG if verbose else (logging.ERROR if quiet else logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
        stream=sys.stderr,
        force=True,
    )
    logging.getLogger().setLevel(level)
    for name in ("urllib3", "httpcore", "httpx", "asyncio"):
        logging.getLogger(name).setLevel(logging.WARNING)
