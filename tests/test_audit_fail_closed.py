"""Tests del hallazgo BLOCKING: el gate es fail-closed ante cualquier FAIL.

Un check con estado FAIL deriva REJECTED con independencia de la categoría de
la regla, y el exportador no publica un FAIL aunque se apruebe la revisión
manual: la aprobación solo cubre checks MANUAL_REVIEW.
"""

from pathlib import Path

from kliptych.assets import AssetRegistry
from kliptych.contract import Contract
from kliptych.exporter import ExportStatus, export_delivery
from kliptych.gate import CheckStatus, Gate, GateStatus
from kliptych.gate.checks import DEFAULT_VALIDATORS, CheckOutcome, GateContext
from tests.support import ALL_HARD_RULES, FakeProbe, make_contract, make_media, make_piece


def _probe_15s() -> FakeProbe:
    return FakeProbe(info=make_media(duration_s=15.0, has_video=True, has_audio=True))


def _artifact(tmp_path: Path) -> Path:
    path = tmp_path / "piece.mp4"
    _ = path.write_bytes(b"video payload")
    return path


def _hal_contract() -> Contract:
    hard = [rule for rule in ALL_HARD_RULES if rule != "caption.required_mention"]
    return make_contract(
        hard=hard,
        manual_review=["caption.required_mention"],
        audio_rule="any",
    )


def test_hal_manual_review_fail_derives_rejected(tmp_path: Path) -> None:
    """El vector Hal: mención en manual_review + mención ausente → REJECTED."""
    piece = make_piece(_artifact(tmp_path), caption="mira #marca sin mencion")
    result = Gate(probe=_probe_15s()).evaluate_piece(
        piece, contract=_hal_contract(), assets=AssetRegistry(tmp_path)
    )
    assert result.status is GateStatus.REJECTED
    mention = next(check for check in result.checks if check.id == "caption.required_mention")
    assert mention.status is CheckStatus.FAIL


def test_hal_exporter_rejects_fail_despite_manual_approval(tmp_path: Path) -> None:
    """El exportador no publica un FAIL aunque se apruebe la revisión manual."""
    piece = make_piece(_artifact(tmp_path), caption="mira #marca sin mencion")
    report = export_delivery(
        contract=_hal_contract(),
        pieces=[piece],
        gate=Gate(probe=_probe_15s()),
        assets=AssetRegistry(tmp_path),
        destination=tmp_path / "delivery",
        approve_manual_review=True,
        approved_by="auditor-hal",
    )
    assert report.status is ExportStatus.BLOCKED
    assert report.exported == ()
    assert len(report.rejected) == 1
    assert report.rejected[0].gate.status is GateStatus.REJECTED
    assert report.manually_approved_rules == ()
    assert not (tmp_path / "delivery").exists()


def _needs_human_eyes(_context: GateContext) -> CheckOutcome:
    return CheckOutcome(status=CheckStatus.MANUAL_REVIEW, evidence={"reason": "ojos humanos"})


def test_manual_review_without_fail_stays_approvable(tmp_path: Path) -> None:
    """Control positivo: PENDING_REVIEW sin FAIL sí se puede aprobar."""
    gate = Gate(
        probe=_probe_15s(),
        validators={**DEFAULT_VALIDATORS, "watermark.present": _needs_human_eyes},
    )
    contract = make_contract(
        hard=[*ALL_HARD_RULES],
        manual_review=["watermark.present"],
        audio_rule="any",
    )
    piece = make_piece(_artifact(tmp_path), caption="mira @marca #marca")
    report = export_delivery(
        contract=contract,
        pieces=[piece],
        gate=gate,
        assets=AssetRegistry(tmp_path),
        destination=tmp_path / "delivery",
        approve_manual_review=True,
        approved_by="auditor-ok",
    )
    assert report.status is ExportStatus.EXPORTED
    assert report.manually_approved_rules == ("watermark.present",)
