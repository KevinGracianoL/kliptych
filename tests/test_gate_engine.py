"""Tests del motor del gate: estados fail-closed, hashes y orden de checks."""

from hashlib import sha256
from pathlib import Path

import pytest

from kliptych.assets import AssetRegistry
from kliptych.contract import Platform
from kliptych.gate import (
    DEFAULT_VALIDATORS,
    CheckOutcome,
    CheckResult,
    CheckStatus,
    Gate,
    GateContext,
    GateError,
    GateResult,
    GateStatus,
    ProbeError,
)
from kliptych.hashing import sha256_canonical_json
from tests.support import ALL_HARD_RULES, FakeProbe, make_contract, make_media, make_piece


def _artifact(tmp_path: Path) -> Path:
    path = tmp_path / "piece.mp4"
    _ = path.write_bytes(b"video")
    return path


def _gate(probe: FakeProbe | None = None) -> Gate:
    return Gate(FakeProbe(info=make_media()) if probe is None else probe)


def _check(result: GateResult, rule_id: str) -> CheckResult:
    matches = [check for check in result.checks if check.id == rule_id]
    assert len(matches) == 1
    return matches[0]


def _always_manual_review(_context: GateContext) -> CheckOutcome:
    return CheckOutcome(status=CheckStatus.MANUAL_REVIEW, evidence={"reason": "revisión visual"})


def test_all_checks_pass_yields_passed(tmp_path: Path) -> None:
    artifact = _artifact(tmp_path)
    result = _gate().run(
        contract=make_contract(),
        piece=make_piece(artifact),
        assets=AssetRegistry(tmp_path),
    )
    assert result.status is GateStatus.PASSED
    assert result.passed is True
    assert [check.id for check in result.checks] == list(ALL_HARD_RULES)
    assert all(check.status is CheckStatus.PASS for check in result.checks)
    assert result.artifact_sha256 == sha256(b"video").hexdigest()
    expected_contract_hash = sha256_canonical_json(make_contract().model_dump(mode="json"))
    assert result.contract_sha256 == expected_contract_hash


def test_hard_rule_without_validator_never_passes(tmp_path: Path) -> None:
    artifact = _artifact(tmp_path)
    contract = make_contract(hard=[*ALL_HARD_RULES, "watermark.full_video"])
    result = _gate().run(
        contract=contract,
        piece=make_piece(artifact),
        assets=AssetRegistry(tmp_path),
    )
    assert result.status is GateStatus.UNSUPPORTED
    assert result.passed is False
    check = _check(result, "watermark.full_video")
    assert check.status is CheckStatus.UNSUPPORTED
    assert "reason" in check.evidence


def test_hard_failure_rejects_and_wins_over_unsupported(tmp_path: Path) -> None:
    artifact = _artifact(tmp_path)
    contract = make_contract(hard=[*ALL_HARD_RULES, "watermark.full_video"])
    result = _gate().run(
        contract=contract,
        piece=make_piece(artifact, caption="sin mención #marca"),
        assets=AssetRegistry(tmp_path),
    )
    assert result.status is GateStatus.REJECTED
    assert _check(result, "caption.required_mention").status is CheckStatus.FAIL


def test_manual_review_class_is_non_blocking(tmp_path: Path) -> None:
    artifact = _artifact(tmp_path)
    contract = make_contract(manual_review=["audio.official_selection"])
    result = _gate().run(
        contract=contract,
        piece=make_piece(artifact),
        assets=AssetRegistry(tmp_path),
    )
    assert result.status is GateStatus.PASSED
    assert _check(result, "audio.official_selection").status is CheckStatus.MANUAL_REVIEW


def test_recommended_failure_is_recorded_but_non_blocking(tmp_path: Path) -> None:
    artifact = _artifact(tmp_path)
    contract = make_contract(recommended=["duration.max"], min_s=5, max_s=6)
    result = _gate().run(
        contract=contract,
        piece=make_piece(artifact),
        assets=AssetRegistry(tmp_path),
    )
    assert result.status is GateStatus.PASSED
    assert _check(result, "duration.max").status is CheckStatus.FAIL


def test_hard_manual_review_status_blocks_as_manual_review(tmp_path: Path) -> None:
    artifact = _artifact(tmp_path)
    contract = make_contract(hard=[*ALL_HARD_RULES, "watermark.full_video"])
    validators = {**DEFAULT_VALIDATORS, "watermark.full_video": _always_manual_review}
    gate = Gate(FakeProbe(info=make_media()), validators=validators)
    result = gate.run(
        contract=contract,
        piece=make_piece(artifact),
        assets=AssetRegistry(tmp_path),
    )
    assert result.status is GateStatus.MANUAL_REVIEW
    assert _check(result, "watermark.full_video").status is CheckStatus.MANUAL_REVIEW


def test_probe_failure_blocks_media_checks(tmp_path: Path) -> None:
    artifact = _artifact(tmp_path)
    probe = FakeProbe(error=ProbeError("ffprobe no disponible"))
    contract = make_contract(hard=[*ALL_HARD_RULES, "artifact.video_stream"])
    result = _gate(probe).run(
        contract=contract,
        piece=make_piece(artifact),
        assets=AssetRegistry(tmp_path),
    )
    assert result.status is GateStatus.UNSUPPORTED
    assert _check(result, "duration.min").status is CheckStatus.UNSUPPORTED
    assert _check(result, "audio.present").status is CheckStatus.UNSUPPORTED
    assert _check(result, "artifact.video_stream").status is CheckStatus.UNSUPPORTED
    assert _check(result, "artifact.integrity").status is CheckStatus.PASS
    assert _check(result, "caption.required_mention").status is CheckStatus.PASS


def test_missing_artifact_fails_integrity(tmp_path: Path) -> None:
    result = _gate().run(
        contract=make_contract(),
        piece=make_piece(tmp_path / "no-existe.mp4"),
        assets=AssetRegistry(tmp_path),
    )
    assert result.status is GateStatus.REJECTED
    assert _check(result, "artifact.integrity").status is CheckStatus.FAIL
    assert result.artifact_sha256 is None


def test_piece_platform_must_exist_in_contract(tmp_path: Path) -> None:
    artifact = _artifact(tmp_path)
    with pytest.raises(GateError, match="plataforma"):
        _ = _gate().run(
            contract=make_contract(),
            piece=make_piece(artifact, platform=Platform.X),
            assets=AssetRegistry(tmp_path),
        )


def test_gate_result_round_trips_through_json(tmp_path: Path) -> None:
    artifact = _artifact(tmp_path)
    result = _gate().run(
        contract=make_contract(),
        piece=make_piece(artifact),
        assets=AssetRegistry(tmp_path),
    )
    restored = GateResult.model_validate_json(result.model_dump_json())
    assert restored == result
