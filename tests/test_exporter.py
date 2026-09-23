"""Tests del exportador de paquetes de entrega (solo piezas que pasan el gate)."""

import json
from collections.abc import Sequence
from hashlib import sha256
from pathlib import Path
from typing import cast

import pytest

from kliptych.assets import AssetRegistry
from kliptych.contract import (
    AnalyticsProofRequired,
    Contract,
    GeoTarget,
    MinViewsForPayout,
)
from kliptych.exporter import (
    DeliveryReport,
    ExportError,
    ExportStatus,
    export_delivery,
)
from kliptych.gate import Gate, GateStatus, Piece
from kliptych.resolver import resolve_contract
from kliptych.runtime import RecordedModel
from tests.support import FakeProbe, make_contract, make_media, make_piece

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
) -> DeliveryReport:
    return export_delivery(
        contract=contract if contract is not None else make_contract(),
        pieces=pieces if pieces is not None else [make_piece(_artifact(tmp_path))],
        gate=_gate(),
        assets=AssetRegistry(tmp_path),
        destination=tmp_path / "delivery" if destination is None else destination,
    )


def test_exports_passing_piece_with_metadata_and_gate_report(tmp_path: Path) -> None:
    artifact = _artifact(tmp_path)
    piece = make_piece(artifact, caption="mira @marca #marca", hashtags=("#marca",))
    destination = tmp_path / "delivery"
    report = _export(tmp_path, pieces=[piece], destination=destination)

    assert report.status is ExportStatus.EXPORTED
    assert report.destination == str(destination)
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
    assert metadata["caption"] == "mira @marca #marca"
    assert metadata["hashtags"] == ["#marca"]
    assert metadata["required_mentions"] == ["@marca"]
    assert metadata["audio_rule"] == "own_clip"
    assert metadata["artifact_sha256"] == sha256(b"video").hexdigest()
    gate_payload = _parse((root / "piece-01.gate.json").read_text(encoding="utf-8"))
    assert gate_payload["status"] == "passed"
    assert gate_payload["artifact_sha256"] == sha256(b"video").hexdigest()


def test_rejected_piece_is_not_exported(tmp_path: Path) -> None:
    piece = make_piece(_artifact(tmp_path), caption="sin mención #marca")
    destination = tmp_path / "delivery"
    report = _export(tmp_path, pieces=[piece], destination=destination)

    assert report.status is ExportStatus.BLOCKED
    assert report.exported == ()
    assert len(report.rejected) == 1
    rejected = report.rejected[0]
    assert rejected.gate_status is GateStatus.REJECTED
    assert "caption.required_mention" in rejected.reason
    assert any(check.id == "caption.required_mention" for check in rejected.checks)
    assert not (destination / _CAMPAIGN).exists()

    written = _parse((destination / "delivery_report.json").read_text(encoding="utf-8"))
    assert written["status"] == "blocked"


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


def test_rejects_unsafe_piece_id(tmp_path: Path) -> None:
    piece = make_piece(_artifact(tmp_path)).model_copy(update={"piece_id": ".."})
    with pytest.raises(ExportError, match="piece_id"):
        _ = _export(tmp_path, pieces=[piece])


def test_rejects_duplicate_piece_ids(tmp_path: Path) -> None:
    artifact = _artifact(tmp_path)
    pieces = [make_piece(artifact), make_piece(artifact)]
    with pytest.raises(ExportError, match="duplicad"):
        _ = _export(tmp_path, pieces=pieces)
    assert not (tmp_path / "delivery" / _CAMPAIGN).exists()


def test_rejects_empty_piece_list(tmp_path: Path) -> None:
    with pytest.raises(ExportError, match="piezas"):
        _ = _export(tmp_path, pieces=[])


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
    kinds = {reminder.kind for reminder in report.reminders}
    assert kinds == {
        "geo_target",
        "min_views_for_payout",
        "analytics_proof_required",
        "manual_review",
    }
    assert any("MX" in reminder.detail for reminder in report.reminders)


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
