"""Sanidad del paquete instalado."""

from importlib.metadata import version

import kliptych
from kliptych._version import __version__ as version_internal


def test_version_matches_installed_metadata() -> None:
    assert kliptych.__version__ == version("kliptych")
    assert kliptych.__version__ == "1.1.0"
    assert version_internal == "1.1.0"
