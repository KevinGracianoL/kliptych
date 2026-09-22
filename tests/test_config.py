"""Tests de resolución de rutas del workspace."""

from pathlib import Path

import pytest

from kliptych.config import ENV_ROOT, Settings


def test_from_root_resolves_to_absolute(tmp_path: Path) -> None:
    nested = tmp_path / "sub" / ".."
    settings = Settings.from_root(nested)
    assert settings.root == tmp_path.resolve()


def test_from_env_reads_configured_root(tmp_path: Path) -> None:
    settings = Settings.from_env({ENV_ROOT: str(tmp_path)})
    assert settings.root == tmp_path.resolve()


def test_from_env_defaults_to_current_directory(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    settings = Settings.from_env({})
    assert settings.root == tmp_path.resolve()


def test_derived_directories_live_under_root(tmp_path: Path) -> None:
    settings = Settings.from_root(tmp_path)
    assert settings.runs_dir == tmp_path / "runs"
    assert settings.campaigns_dir == tmp_path / "campaigns"
    assert settings.private_campaigns_dir == tmp_path / "campaigns" / "private"
    assert settings.assets_dir == tmp_path / "assets"
