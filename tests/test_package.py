"""Sanidad del paquete instalado."""

from importlib.metadata import version

import kliptych


def test_version_matches_installed_metadata() -> None:
    assert kliptych.__version__ == version("kliptych")
