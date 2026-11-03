"""Distribution guards for the native WebKitGTK MP4 playback runtime."""

import json
from pathlib import Path
from typing import cast

import pytest

from kliptych.environment import SubprocessRunner

_ROOT = Path(__file__).resolve().parents[2]
_RUNTIME_PACKAGES = {
    "gstreamer1.0-plugins-good",
    "gstreamer1.0-plugins-bad",
    "gstreamer1.0-libav",
}
_ACTION = "./.github/actions/studio-linux-deps"


def _mapping(value: object) -> dict[str, object]:
    assert isinstance(value, dict)
    return cast("dict[str, object]", value)


def _linux_config() -> dict[str, object]:
    payload = cast("object", json.loads((_ROOT / "studio/src-tauri/tauri.conf.json").read_text()))
    config = _mapping(payload)
    return _mapping(_mapping(config["bundle"])["linux"])


def test_debian_installer_declares_native_playback_runtime() -> None:
    dependencies = _mapping(_linux_config()["deb"])["depends"]
    assert isinstance(dependencies, list)
    assert set(cast("list[str]", dependencies)) >= _RUNTIME_PACKAGES


def test_appimage_includes_native_media_framework() -> None:
    appimage = _mapping(_linux_config().get("appimage", {}))
    assert appimage.get("bundleMediaFramework") is True


@pytest.mark.parametrize("workflow", ["ci.yml", "release.yml"])
def test_linux_builds_share_dependency_installation(workflow: str) -> None:
    content = (_ROOT / ".github/workflows" / workflow).read_text()
    assert f"uses: {_ACTION}" in content


def test_shared_action_derives_media_packages_from_installer_metadata() -> None:
    action = (_ROOT / ".github/actions/studio-linux-deps/action.yml").read_text()
    installer = (_ROOT / ".github/actions/studio-linux-deps/install-runtime.mjs").read_text()
    assert 'node "$GITHUB_ACTION_PATH/install-runtime.mjs"' in action
    assert "studio/src-tauri/tauri.conf.json" in installer
    assert "config.bundle.linux.deb.depends" in installer
    for package in _RUNTIME_PACKAGES:
        assert package not in action


def test_build_action_checks_native_factories_instead_of_only_ffmpeg() -> None:
    action = (_ROOT / ".github/actions/studio-linux-deps/action.yml").read_text()
    assert 'gst-inspect-1.0 "$element"' in action
    assert "set -euo pipefail" in action
    for factory in ("autoaudiosink", "qtdemux", "h264parse", "avdec_h264", "avdec_aac"):
        assert factory in action


def test_full_python_gate_pins_node_for_installer_policy_tests() -> None:
    workflow = (_ROOT / ".github/workflows/ci.yml").read_text()
    gate_job = workflow.split("\n  gate:", maxsplit=1)[1]
    assert "uses: actions/setup-node@v5" in gate_job
    assert "node-version: 26" in gate_job


@pytest.mark.parametrize(
    "packages", [[], ["--allow-unauthenticated"], ["good;evil"], [None], "name"]
)
def test_runtime_installer_rejects_invalid_package_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, packages: object
) -> None:
    record = _prepare_runtime_installer(tmp_path, monkeypatch, packages=packages)
    result = SubprocessRunner().run(
        (
            "node",
            "--import",
            (tmp_path / "process-boundary.mjs").as_uri(),
            str(_ROOT / ".github/actions/studio-linux-deps/install-runtime.mjs"),
        ),
    )
    assert not result.ok
    assert "explicit package names" in result.stderr
    assert not record.exists()


@pytest.mark.parametrize("exit_code", [0, 17])
def test_runtime_installer_uses_argument_list_and_propagates_apt_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, exit_code: int
) -> None:
    packages = ["gstreamer1.0-plugins-good", "gstreamer1.0-libav"]
    record = _prepare_runtime_installer(tmp_path, monkeypatch, packages=packages)
    monkeypatch.setenv("KLIPTYCH_APT_TEST_EXIT", str(exit_code))
    result = SubprocessRunner().run(
        (
            "node",
            "--import",
            (tmp_path / "process-boundary.mjs").as_uri(),
            str(_ROOT / ".github/actions/studio-linux-deps/install-runtime.mjs"),
        ),
    )
    assert result.ok is (exit_code == 0)
    call = _mapping(cast("object", json.loads(record.read_text())))
    assert call["executable"] == "sudo"
    assert call["args"] == ["apt-get", "install", "-y", "-o", "Acquire::Retries=3", *packages]


def _prepare_runtime_installer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, packages: object
) -> Path:
    config_dir = tmp_path / "studio/src-tauri"
    config_dir.mkdir(parents=True)
    _ = (config_dir / "tauri.conf.json").write_text(
        json.dumps(
            {
                "bundle": {"linux": {"deb": {"depends": packages}}},
            }
        )
    )
    record = tmp_path / "apt-call.json"
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("KLIPTYCH_APT_TEST_RECORD", str(record))
    _ = (tmp_path / "process-boundary.mjs").write_text("""
import { createRequire, syncBuiltinESMExports } from 'node:module';
import { writeFileSync } from 'node:fs';
const childProcess = createRequire(import.meta.url)('node:child_process');
childProcess.spawnSync = (executable, args) => {
  writeFileSync(process.env.KLIPTYCH_APT_TEST_RECORD, JSON.stringify({ executable, args }));
  return { status: Number(process.env.KLIPTYCH_APT_TEST_EXIT ?? '0') };
};
syncBuiltinESMExports();
""")
    return record
