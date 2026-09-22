"""Tests del manifiesto de corrida."""

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import pytest
from pydantic import ValidationError

from kliptych.environment import EnvironmentReport, GpuInfo
from kliptych.gate import CheckResult, CheckStatus, GateResult, GateStatus
from kliptych.manifest import OutputHash, RunManifest, read_manifest, write_manifest

_STARTED = datetime(2026, 9, 22, 10, 0, tzinfo=UTC)


def _parse(text: str) -> dict[str, object]:
    return cast("dict[str, object]", json.loads(text))


def _gate_result() -> GateResult:
    return GateResult(
        status=GateStatus.REJECTED,
        checks=(
            CheckResult(
                id="caption.required_mention",
                status=CheckStatus.FAIL,
                evidence={"missing": ["@marca"]},
            ),
        ),
        artifact_sha256="a" * 64,
        contract_sha256="b" * 64,
    )


def _manifest(**overrides: object) -> RunManifest:
    defaults: dict[str, object] = {
        "run_id": "run-001",
        "started_at": _STARTED,
        "environment": EnvironmentReport(
            ffmpeg_version="N-118380",
            nvenc_available=False,
            gpu=GpuInfo(name="GTX 1650 Ti", vram_mib=4096, driver_version="610.74"),
        ),
        "brief_sha256": "c" * 64,
        "contract_schema_version": "1.0",
        "contract_sha256": "b" * 64,
        "model_version": "recorded-v1",
        "prompt_version": "extract-v1",
        "render_arguments": ("-c:v", "h264_nvenc"),
        "outputs": (OutputHash(path="piece.mp4", sha256="d" * 64, size_bytes=10),),
        "gate": _gate_result(),
    }
    defaults.update(overrides)
    return RunManifest.model_validate(defaults)


def test_write_and_read_manifest_round_trip(tmp_path: Path) -> None:
    manifest = _manifest()
    path = write_manifest(manifest, tmp_path / "runs" / "run-001")
    assert path.name == "run_manifest.json"
    restored = read_manifest(path)
    assert restored == manifest


def test_manifest_json_records_gate_and_hashes(tmp_path: Path) -> None:
    path = write_manifest(_manifest(), tmp_path)
    payload = _parse(path.read_text(encoding="utf-8"))
    assert payload["brief_sha256"] == "c" * 64
    assert payload["contract_schema_version"] == "1.0"
    gate = payload["gate"]
    assert isinstance(gate, dict)
    gate_payload = cast("dict[str, object]", gate)
    assert gate_payload["status"] == "rejected"
    checks = gate_payload["checks"]
    assert isinstance(checks, list)
    check_payload = cast("dict[str, object]", checks[0])
    assert check_payload["id"] == "caption.required_mention"
    outputs = payload["outputs"]
    assert isinstance(outputs, list)
    assert outputs[0]["sha256"] == "d" * 64


def test_output_hash_requires_full_digest() -> None:
    with pytest.raises(ValidationError, match="sha256"):
        _ = OutputHash(path="piece.mp4", sha256="abc", size_bytes=1)


def test_manifest_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError, match="Extra inputs"):
        _ = _manifest(promesa="subir por API")


def test_manifest_schema_version_is_locked() -> None:
    with pytest.raises(ValidationError, match="contract_schema_version"):
        _ = _manifest(contract_schema_version="2.0")


def test_manifest_requires_timezone_aware_start() -> None:
    naive = datetime.fromisoformat("2026-09-22T10:00:00")
    with pytest.raises(ValidationError, match="started_at"):
        _ = _manifest(started_at=naive)
