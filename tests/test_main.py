"""Tests de la CLI."""

import pytest

from kliptych import __version__
from kliptych.__main__ import main


def test_main_without_args_prints_help(capsys: pytest.CaptureFixture[str]) -> None:
    assert main([]) == 0
    assert "usage" in capsys.readouterr().out


def test_main_version_flag_prints_version(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as excinfo:
        _ = main(["--version"])
    assert excinfo.value.code == 0
    assert __version__ in capsys.readouterr().out
