"""Tests de la CLI."""

import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import cast

import pytest

from kliptych import __version__
from kliptych.__main__ import main
from kliptych.environment import EnvironmentReport


def _parse(text: str) -> dict[str, object]:
    return cast("dict[str, object]", json.loads(text))


def test_console_script_is_installed() -> None:
    script = shutil.which("kliptych", path=str(Path(sys.executable).parent))
    assert script is not None
    completed = subprocess.run(
        [script, "--version"],
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    assert completed.returncode == 0
    assert completed.stdout.strip() == __version__


def test_main_without_args_prints_help(capsys: pytest.CaptureFixture[str]) -> None:
    assert main([]) == 0
    assert "usage" in capsys.readouterr().out


def test_main_version_flag_prints_version(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as excinfo:
        _ = main(["--version"])
    assert excinfo.value.code == 0
    assert __version__ in capsys.readouterr().out


def test_main_env_prints_environment(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    report = EnvironmentReport(ffmpeg_version="N-118380", nvenc_available=True)

    def fake_detect(_runner: object) -> EnvironmentReport:
        return report

    monkeypatch.setattr("kliptych.__main__.detect_environment", fake_detect)
    assert main(["env"]) == 0
    payload = _parse(capsys.readouterr().out)
    assert payload["ffmpeg_version"] == "N-118380"
    assert payload["nvenc_available"] is True
