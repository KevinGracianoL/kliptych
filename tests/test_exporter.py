"""Tests del exportador de paquetes de entrega (solo piezas que pasan el gate)."""

import json
import subprocess
import sys
from collections.abc import Sequence
from hashlib import sha256
from pathlib import Path
from typing import cast

import pytest
from pydantic import ValidationError

from kliptych.assets import AssetRegistry
from kliptych.contract import (
    AnalyticsProofRequired,
    Contract,
    GeoTarget,
    MinViewsForPayout,
    Platform,
    contract_digest,
)
from kliptych.exporter import (
    DeliveryReport,
    ExportError,
    ExportStatus,
    export_delivery,
)
from kliptych.gate import (
    DEFAULT_VALIDATORS,
    CheckOutcome,
    CheckStatus,
    Gate,
    GateContext,
    GateStatus,
    Piece,
)
from kliptych.resolver import resolve_contract
from kliptych.runtime import RecordedModel
from tests.support import (
    ALL_HARD_RULES,
    FakeProbe,
    make_contract,
    make_media,
    make_piece,
    write_fixture_clip,
)

_CAMPAIGN = "camp-test"


def _parse(text: str) -> dict[str, object]:
    return cast("dict[str, object]", json.loads(text))


def _gate() -> Gate:
    return Gate(FakeProbe(info=make_media()))


def _artifact(tmp_path: Path, name: str = "piece-01.mp4", content: bytes = b"video") -> Path:
    path = tmp_path / name
    _ = path.write_bytes(content)
    return path


def _export(
    tmp_path: Path,
    *,
    contract: Contract | None = None,
    pieces: Sequence[Piece] | None = None,
    destination: Path | None = None,
    gate: Gate | None = None,
    approve_manual_review: bool = False,
) -> DeliveryReport:
    return export_delivery(
        contract=contract if contract is not None else make_contract(),
        pieces=pieces if pieces is not None else [make_piece(_artifact(tmp_path))],
        gate=gate if gate is not None else _gate(),
        assets=AssetRegistry(tmp_path),
        destination=tmp_path / "delivery" if destination is None else destination,
        approve_manual_review=approve_manual_review,
    )


def test_exports_passing_piece_with_metadata_and_gate_report(tmp_path: Path) -> None:
    artifact = _artifact(tmp_path)
    piece = make_piece(artifact, caption="mira @marca #marca", hashtags=("#marca",))
    destination = tmp_path / "delivery"
    report = _export(tmp_path, pieces=[piece], destination=destination)

    assert report.status is ExportStatus.EXPORTED
    assert report.package == "delivery"
    assert report.schema_version == "1.1"
    assert len(report.exported) == 1
    exported = report.exported[0]
    assert exported.piece_id == "piece-01"
    assert exported.platform.value == "tiktok"
    assert exported.gate_status is GateStatus.PASSED
    assert exported.artifact_sha256 == sha256(b"video").hexdigest()
    assert exported.artifact_path == "camp-test/tiktok/piece-01.mp4"

    root = destination / _CAMPAIGN / "tiktok"
    assert (root / "piece-01.mp4").read_bytes() == b"video"
    metadata = _parse((root / "piece-01.metadata.json").read_text(encoding="utf-8"))
    assert metadata["schema_version"] == "1.1"
    assert metadata["piece_id"] == "piece-01"
    assert metadata["platform"] == "tiktok"
    assert metadata["caption"] == "mira @marca #marca"
    assert metadata["hashtags"] == ["#marca"]
    assert metadata["required_mentions"] == ["@marca"]
    assert metadata["required_hashtags"] == ["#marca"]
    assert metadata["attribution"] == {"type": "none", "value": None}
    assert metadata["link_in_bio"] is False
    assert metadata["audio_rule"] == "own_clip"
    assert metadata["official_audio"] is None
    languages = cast("dict[str, object]", metadata["languages"])
    assert languages["source"] == "es"
    assert metadata["watermark_required"] is False
    assert metadata["artifact_sha256"] == sha256(b"video").hexdigest()
    assert metadata["artifact_size_bytes"] == 5
    gate_payload = _parse((root / "piece-01.gate.json").read_text(encoding="utf-8"))
    assert gate_payload["status"] == "passed"
    assert gate_payload["artifact_sha256"] == sha256(b"video").hexdigest()
    assert gate_payload["contract_sha256"]
    checks = cast("list[object]", gate_payload["checks"])
    assert len(checks) > 0


def test_rejects_destination_that_is_a_file(tmp_path: Path) -> None:
    destination = tmp_path / "delivery"
    _ = destination.write_bytes(b"contenido del usuario")
    with pytest.raises(ExportError, match="no es un directorio"):
        _ = _export(tmp_path, destination=destination)
    assert destination.read_bytes() == b"contenido del usuario"


def test_locked_destination_keeps_previous_package(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "delivery"
    _ = _export(tmp_path, destination=destination)
    original_rename = Path.rename

    def failing_rename(self: Path, target: Path) -> Path:
        if self.name.startswith("delivery.staging-"):
            raise PermissionError(13, "bloqueado")
        return original_rename(self, target)

    monkeypatch.setattr(Path, "rename", failing_rename)
    with pytest.raises(ExportError, match="falló la escritura"):
        _ = _export(tmp_path, destination=destination)
    assert (destination / _CAMPAIGN / "tiktok" / "piece-01.mp4").exists()
    assert not list(tmp_path.glob("delivery.backup-*"))
    assert not list(tmp_path.glob("delivery.staging-*"))


def test_restore_failure_reports_backup_location(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "delivery"
    _ = _export(tmp_path, destination=destination)
    original_rename = Path.rename

    def failing_rename(self: Path, target: Path) -> Path:
        if self.name.startswith(("delivery.staging-", "delivery.backup-")):
            raise PermissionError(13, "bloqueado")
        return original_rename(self, target)

    monkeypatch.setattr(Path, "rename", failing_rename)
    with pytest.raises(ExportError, match="paquete previo quedó en") as excinfo:
        _ = _export(tmp_path, destination=destination)
    backups = list(tmp_path.glob("delivery.backup-*"))
    assert len(backups) == 1
    assert backups[0].name in str(excinfo.value)
    assert (backups[0] / _CAMPAIGN / "tiktok" / "piece-01.mp4").exists()
    assert not list(tmp_path.glob("delivery.staging-*"))


@pytest.mark.skipif(sys.platform != "win32", reason="junctions solo en Windows")
def test_junction_destination_is_replaced_without_residue(tmp_path: Path) -> None:
    target = tmp_path / "linked"
    destination = tmp_path / "delivery"
    target.mkdir()
    marker = target / "marker.txt"
    _ = marker.write_bytes(b"contenido previo")
    completed = subprocess.run(
        ["cmd", "/c", "mklink", "/J", str(destination), str(target)],
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        pytest.skip("no se pudo crear el junction")

    _ = _export(tmp_path, destination=destination)
    assert (destination / _CAMPAIGN / "tiktok" / "piece-01.mp4").exists()
    assert not destination.is_junction()
    assert marker.read_bytes() == b"contenido previo"
    assert not list(tmp_path.glob("delivery.backup-*"))
    assert not list(tmp_path.glob("delivery.staging-*"))


def test_piece_id_length_is_bounded_at_construction() -> None:
    with pytest.raises(ValidationError):
        _ = Piece(
            piece_id="a" * 65,
            platform=Platform.TIKTOK,
            caption="x",
            artifact_path=Path("x.mp4"),
        )


def test_metadata_merges_must_mention_with_required_mentions(tmp_path: Path) -> None:
    contract = make_contract(required_mentions=["@marca"], must_mention=["@jefe"])
    piece = make_piece(_artifact(tmp_path), caption="mira @marca y @jefe #marca")
    report = _export(tmp_path, contract=contract, pieces=[piece])
    metadata_path = tmp_path / "delivery" / _CAMPAIGN / "tiktok" / "piece-01.metadata.json"
    metadata = _parse(metadata_path.read_text(encoding="utf-8"))
    assert metadata["required_mentions"] == ["@marca", "@jefe"]
    assert report.status is ExportStatus.EXPORTED


def test_rejected_piece_is_not_exported_and_keeps_full_gate_result(tmp_path: Path) -> None:
    contract = make_contract()
    piece = make_piece(_artifact(tmp_path), caption="sin mención #marca")
    destination = tmp_path / "delivery"
    report = _export(tmp_path, contract=contract, pieces=[piece], destination=destination)

    assert report.status is ExportStatus.BLOCKED
    assert report.exported == ()
    assert len(report.rejected) == 1
    rejected = report.rejected[0]
    assert rejected.gate_status is GateStatus.REJECTED
    assert "caption.required_mention" in rejected.reason
    assert rejected.gate.contract_sha256 == contract_digest(contract)
    assert rejected.gate.artifact_sha256 == sha256(b"video").hexdigest()
    assert any(check.id == "caption.required_mention" for check in rejected.gate.checks)
    assert not (destination / _CAMPAIGN).exists()

    written = _parse((destination / "delivery_report.json").read_text(encoding="utf-8"))
    assert written["status"] == "blocked"
    assert written["schema_version"] == "1.1"


def test_unsupported_rule_is_not_exported(tmp_path: Path) -> None:
    contract = make_contract(hard=[*ALL_HARD_RULES, "watermark.full_video"])
    report = _export(tmp_path, contract=contract)
    assert report.status is ExportStatus.BLOCKED
    assert report.rejected[0].gate_status is GateStatus.UNSUPPORTED
    assert not (tmp_path / "delivery" / _CAMPAIGN).exists()


def test_hard_manual_review_rule_is_not_exported(tmp_path: Path) -> None:
    def manual_review(_context: GateContext) -> CheckOutcome:
        return CheckOutcome(status=CheckStatus.MANUAL_REVIEW, evidence={"reason": "visual"})

    gate = Gate(
        FakeProbe(info=make_media()),
        validators={**DEFAULT_VALIDATORS, "watermark.full_video": manual_review},
    )
    contract = make_contract(hard=[*ALL_HARD_RULES, "watermark.full_video"])
    report = _export(tmp_path, contract=contract, gate=gate)
    assert report.status is ExportStatus.BLOCKED
    assert report.rejected[0].gate_status is GateStatus.PENDING_REVIEW


def test_pending_review_exported_when_approved(tmp_path: Path) -> None:
    def manual_review(_context: GateContext) -> CheckOutcome:
        return CheckOutcome(status=CheckStatus.MANUAL_REVIEW, evidence={"reason": "visual"})

    gate = Gate(
        FakeProbe(info=make_media()),
        validators={**DEFAULT_VALIDATORS, "watermark.full_video": manual_review},
    )
    contract = make_contract(hard=[*ALL_HARD_RULES, "watermark.full_video"])

    # Without approval, it is blocked
    blocked = _export(tmp_path, contract=contract, gate=gate, approve_manual_review=False)
    assert blocked.status is ExportStatus.BLOCKED
    assert len(blocked.rejected) == 1
    assert blocked.rejected[0].gate_status is GateStatus.PENDING_REVIEW

    # With approval, it is exported
    destination = tmp_path / "delivery_approved"
    approved = _export(
        tmp_path,
        contract=contract,
        gate=gate,
        destination=destination,
        approve_manual_review=True,
    )
    assert approved.status is ExportStatus.EXPORTED
    assert len(approved.exported) == 1
    assert approved.exported[0].gate_status is GateStatus.PENDING_REVIEW


def test_partial_delivery_exports_only_passing_pieces(tmp_path: Path) -> None:
    good = make_piece(_artifact(tmp_path, "good.mp4"), caption="mira @marca #marca")
    bad = make_piece(_artifact(tmp_path, "bad.mp4"), caption="sin mención #marca").model_copy(
        update={"piece_id": "piece-02"}
    )
    destination = tmp_path / "delivery"
    report = _export(tmp_path, pieces=[good, bad], destination=destination)

    assert report.status is ExportStatus.PARTIAL
    assert [piece.piece_id for piece in report.exported] == ["piece-01"]
    assert [piece.piece_id for piece in report.rejected] == ["piece-02"]
    root = destination / _CAMPAIGN / "tiktok"
    assert (root / "piece-01.mp4").exists()
    assert not (root / "piece-02.mp4").exists()


def test_rejects_unsafe_campaign_id(tmp_path: Path) -> None:
    contract = make_contract().model_copy(update={"campaign_id": "../evil"})
    with pytest.raises(ExportError, match="campaign_id"):
        _ = _export(tmp_path, contract=contract)
    assert not (tmp_path / "delivery").exists()


def test_rejects_trailing_dot_campaign_id(tmp_path: Path) -> None:
    contract = make_contract().model_copy(update={"campaign_id": "camp-test."})
    with pytest.raises(ExportError, match="campaign_id"):
        _ = _export(tmp_path, contract=contract)
    assert not (tmp_path / "delivery").exists()


def test_rejects_campaign_id_reserved_by_exporter(tmp_path: Path) -> None:
    contract = make_contract().model_copy(update={"campaign_id": "delivery_report.json"})
    with pytest.raises(ExportError, match="reservado"):
        _ = _export(tmp_path, contract=contract)
    assert not (tmp_path / "delivery").exists()


def test_rejects_unsafe_piece_id(tmp_path: Path) -> None:
    piece = make_piece(_artifact(tmp_path)).model_copy(update={"piece_id": ".."})
    with pytest.raises(ExportError, match="piece_id"):
        _ = _export(tmp_path, pieces=[piece])


def test_rejects_piece_id_with_trailing_newline(tmp_path: Path) -> None:
    piece = make_piece(_artifact(tmp_path)).model_copy(update={"piece_id": "piece-01\n"})
    with pytest.raises(ExportError, match="piece_id"):
        _ = _export(tmp_path, pieces=[piece])


def test_rejects_overlong_piece_id(tmp_path: Path) -> None:
    piece = make_piece(_artifact(tmp_path)).model_copy(update={"piece_id": "a" * 65})
    with pytest.raises(ExportError, match="piece_id"):
        _ = _export(tmp_path, pieces=[piece])


def test_rejects_reserved_device_piece_id(tmp_path: Path) -> None:
    piece = make_piece(_artifact(tmp_path)).model_copy(update={"piece_id": "NUL"})
    with pytest.raises(ExportError, match="piece_id"):
        _ = _export(tmp_path, pieces=[piece])
    assert not (tmp_path / "delivery").exists()


def test_rejects_unsupported_artifact_suffix(tmp_path: Path) -> None:
    piece = make_piece(tmp_path / "clip.mp4:evil")
    with pytest.raises(ExportError, match="extensión"):
        _ = _export(tmp_path, pieces=[piece])


def test_rejects_duplicate_piece_ids(tmp_path: Path) -> None:
    artifact = _artifact(tmp_path)
    pieces = [make_piece(artifact), make_piece(artifact)]
    with pytest.raises(ExportError, match="duplicad"):
        _ = _export(tmp_path, pieces=pieces)
    assert not (tmp_path / "delivery" / _CAMPAIGN).exists()


def test_rejects_case_insensitive_collision(tmp_path: Path) -> None:
    first = make_piece(_artifact(tmp_path, "first.mp4", b"AAAA"), caption="mira @marca #marca")
    second = make_piece(
        _artifact(tmp_path, "second.mp4", b"BBBB"), caption="mira @marca #marca"
    ).model_copy(update={"piece_id": "PIECE-01"})
    with pytest.raises(ExportError, match="colisión"):
        _ = _export(tmp_path, pieces=[first, second])
    assert not (tmp_path / "delivery").exists()


def test_rejects_extension_collision(tmp_path: Path) -> None:
    first = make_piece(
        _artifact(tmp_path, "raw", b"AAAA"), caption="mira @marca #marca"
    ).model_copy(update={"piece_id": "clip.mp4"})
    second = make_piece(
        _artifact(tmp_path, "other.mp4", b"BBBB"), caption="mira @marca #marca"
    ).model_copy(update={"piece_id": "clip"})
    with pytest.raises(ExportError, match="colisión"):
        _ = _export(tmp_path, pieces=[first, second])


def test_rejects_metadata_namespace_collision(tmp_path: Path) -> None:
    first = make_piece(_artifact(tmp_path, "render.json", b"AAAA"), caption="mira @marca #marca")
    second = make_piece(
        _artifact(tmp_path, "other.json", b"BBBB"), caption="mira @marca #marca"
    ).model_copy(update={"piece_id": "piece-01.metadata"})
    with pytest.raises(ExportError, match="colisión"):
        _ = _export(tmp_path, pieces=[first, second])


def test_rejects_empty_piece_list(tmp_path: Path) -> None:
    with pytest.raises(ExportError, match="piezas"):
        _ = _export(tmp_path, pieces=[])


def test_rejects_undeclared_platform(tmp_path: Path) -> None:
    piece = make_piece(_artifact(tmp_path), platform=Platform.INSTAGRAM_REELS)
    with pytest.raises(ExportError, match="no declaradas"):
        _ = _export(tmp_path, pieces=[piece])
    assert not (tmp_path / "delivery").exists()


def test_copy_verification_rejects_mismatched_hash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_sha256(_data: bytes) -> str:
        return "0" * 64

    monkeypatch.setattr("kliptych.exporter.sha256_bytes", fake_sha256)
    with pytest.raises(ExportError, match="cambió respecto del hash"):
        _ = _export(tmp_path)
    assert not (tmp_path / "delivery").exists()


def test_unreadable_artifact_raises_export_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact = _artifact(tmp_path)
    original_read_bytes = Path.read_bytes

    def failing_read_bytes(self: Path) -> bytes:
        if self == artifact:
            raise PermissionError(13, "denegado")
        return original_read_bytes(self)

    monkeypatch.setattr(Path, "read_bytes", failing_read_bytes)
    with pytest.raises(ExportError, match="no se pudo leer"):
        _ = _export(tmp_path)


def test_mid_batch_failure_leaves_destination_untouched(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    good = make_piece(_artifact(tmp_path, "good.mp4", b"AAAA"), caption="mira @marca #marca")
    broken = _artifact(tmp_path, "broken.mp4", b"BBBB")
    bad = make_piece(broken, caption="mira @marca #marca").model_copy(
        update={"piece_id": "piece-02"}
    )
    original_read_bytes = Path.read_bytes

    def failing_read_bytes(self: Path) -> bytes:
        if self == broken:
            raise PermissionError(13, "denegado")
        return original_read_bytes(self)

    monkeypatch.setattr(Path, "read_bytes", failing_read_bytes)
    contract = make_contract(hard=[])
    with pytest.raises(ExportError, match="no se pudo leer"):
        _ = _export(tmp_path, contract=contract, pieces=[good, bad])
    assert not (tmp_path / "delivery").exists()
    assert not list(tmp_path.glob("delivery.staging-*"))


def test_reexport_removes_stale_artifacts(tmp_path: Path) -> None:
    destination = tmp_path / "delivery"
    good = make_piece(_artifact(tmp_path), caption="mira @marca #marca")
    first = _export(tmp_path, pieces=[good], destination=destination)
    assert first.status is ExportStatus.EXPORTED

    rejected_piece = make_piece(_artifact(tmp_path), caption="sin mención #marca")
    second = _export(tmp_path, pieces=[rejected_piece], destination=destination)
    assert second.status is ExportStatus.BLOCKED
    root = destination / _CAMPAIGN / "tiktok"
    assert not (root / "piece-01.mp4").exists()
    assert not (root / "piece-01.metadata.json").exists()
    assert not (root / "piece-01.gate.json").exists()
    written = _parse((destination / "delivery_report.json").read_text(encoding="utf-8"))
    assert written["status"] == "blocked"


def test_reexport_failure_keeps_previous_package(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "delivery"
    good = make_piece(_artifact(tmp_path), caption="mira @marca #marca")
    first = _export(tmp_path, pieces=[good], destination=destination)
    assert first.status is ExportStatus.EXPORTED

    def fake_sha256(_data: bytes) -> str:
        return "0" * 64

    monkeypatch.setattr("kliptych.exporter.sha256_bytes", fake_sha256)
    with pytest.raises(ExportError, match="cambió respecto del hash"):
        _ = _export(tmp_path, pieces=[good], destination=destination)
    root = destination / _CAMPAIGN / "tiktok"
    assert (root / "piece-01.mp4").read_bytes() == b"video"
    written = _parse((destination / "delivery_report.json").read_text(encoding="utf-8"))
    assert written["status"] == "exported"
    assert not list(tmp_path.glob("delivery.staging-*"))


def test_package_bytes_are_deterministic(tmp_path: Path) -> None:
    piece = make_piece(_artifact(tmp_path), caption="mira @marca #marca")
    first = tmp_path / "a" / "delivery"
    second = tmp_path / "b" / "delivery"
    _ = _export(tmp_path, pieces=[piece], destination=first)
    _ = _export(tmp_path, pieces=[piece], destination=second)
    for relative in (
        "delivery_report.json",
        "camp-test/tiktok/piece-01.mp4",
        "camp-test/tiktok/piece-01.metadata.json",
        "camp-test/tiktok/piece-01.gate.json",
    ):
        assert (first / relative).read_bytes() == (second / relative).read_bytes()


def test_report_lists_post_publication_reminders(tmp_path: Path) -> None:
    contract = make_contract().model_copy(
        update={
            "geo_target": GeoTarget(country="MX", min_pct=60),
            "min_views_for_payout": MinViewsForPayout(value=10000),
            "analytics_proof_required": AnalyticsProofRequired(value=True),
            "rules": make_contract().rules.model_copy(
                update={"manual_review": ["audio.official_selection"]}
            ),
        }
    )
    report = _export(tmp_path, contract=contract)
    details = {reminder.kind: reminder.detail for reminder in report.reminders}
    assert set(details) == {
        "geo_target",
        "min_views_for_payout",
        "analytics_proof_required",
        "manual_review",
    }
    assert "MX" in details["geo_target"]
    assert "60" in details["geo_target"]
    assert "10000" in details["min_views_for_payout"]
    assert "analytics" in details["analytics_proof_required"]
    assert "audio.official_selection" in details["manual_review"]


def test_delivery_report_round_trips_and_is_written(tmp_path: Path) -> None:
    destination = tmp_path / "delivery"
    report = _export(tmp_path, destination=destination)
    path = destination / "delivery_report.json"
    restored = DeliveryReport.model_validate_json(path.read_text(encoding="utf-8"))
    assert restored == report


def test_reexport_is_deterministic(tmp_path: Path) -> None:
    destination = tmp_path / "delivery"
    first = _export(tmp_path, destination=destination)
    second = _export(tmp_path, destination=destination)
    assert first == second
    assert (destination / _CAMPAIGN / "tiktok" / "piece-01.mp4").read_bytes() == b"video"


def test_recorded_fixture_flows_to_delivery(tmp_path: Path) -> None:
    fixtures = Path(__file__).resolve().parents[1] / "campaigns" / "fixtures" / "given-clips"
    brief = (fixtures / "brief.md").read_text(encoding="utf-8")
    draft = RecordedModel.from_directory(fixtures / "recorded").extract_contract(brief)
    _ = write_fixture_clip(tmp_path)
    registry = AssetRegistry(tmp_path)
    resolution = resolve_contract(draft, registry=registry)
    assert resolution.contract is not None
    artifact = tmp_path / "piece.mp4"
    _ = artifact.write_bytes(b"video")
    report = export_delivery(
        contract=resolution.contract,
        pieces=[make_piece(artifact, caption="mira @marca #marca")],
        gate=_gate(),
        assets=registry,
        destination=tmp_path / "delivery",
    )
    assert report.status is ExportStatus.EXPORTED
    assert report.exported[0].piece_id == "piece-01"
