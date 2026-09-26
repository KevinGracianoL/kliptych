"""Fix 1: --approved-by requerido cuando --approve-manual-review está activo."""

from pathlib import Path
from typing import override

import pytest

from kliptych.__main__ import main
from kliptych.assets import AssetRegistry
from kliptych.campaign_manager import CampaignManager, CampaignOutcome
from kliptych.campaign_types import Campaign, CampaignStatus
from kliptych.contract import Contract
from kliptych.exporter import ExportError, export_delivery
from kliptych.gate import Gate
from kliptych.git_proposals import GitHubCliProvider, ProposalEngine
from kliptych.intelligence import Archetype, ArchetypeClassification, CampaignClassifier
from tests.support import FakeProbe, make_contract, make_media, make_piece


class _FixedClassifier(CampaignClassifier):
    @override
    def classify(self, brief: str, contract: Contract) -> ArchetypeClassification:
        _ = (self, brief, contract)
        return ArchetypeClassification(archetype=Archetype.KNOWN, rationale="test")


def _gate() -> Gate:
    return Gate(probe=FakeProbe(info=make_media(duration_s=15.0)))


def test_exporter_rejects_missing_approved_by(tmp_path: Path) -> None:
    artifact = tmp_path / "piece.mp4"
    _ = artifact.write_bytes(b"video payload")
    with pytest.raises(ExportError, match="approved-by"):
        _ = export_delivery(
            contract=make_contract(audio_rule="any"),
            pieces=[make_piece(artifact)],
            gate=_gate(),
            assets=AssetRegistry(tmp_path),
            destination=tmp_path / "delivery",
            approve_manual_review=True,
            approved_by=None,
        )


def test_exporter_rejects_empty_and_whitespace_approved_by(tmp_path: Path) -> None:
    for bad in ("", "   "):
        artifact = tmp_path / f"piece-{len(bad)}.mp4"
        _ = artifact.write_bytes(b"video payload")
        with pytest.raises(ExportError, match="approved-by"):
            _ = export_delivery(
                contract=make_contract(audio_rule="any"),
                pieces=[make_piece(artifact)],
                gate=_gate(),
                assets=AssetRegistry(tmp_path),
                destination=tmp_path / f"delivery-{len(bad)}",
                approve_manual_review=True,
                approved_by=bad,
            )


def test_campaign_manager_rejects_missing_approved_by(tmp_path: Path) -> None:
    video_path = tmp_path / "rendered.mp4"
    _ = video_path.write_bytes(b"video")

    manager = CampaignManager(
        classifier=_FixedClassifier(),
        proposal_engine=ProposalEngine(provider=GitHubCliProvider(workdir=tmp_path, repo="o/r")),
        gate=_gate(),
        assets=AssetRegistry(tmp_path),
        destination=tmp_path / "delivery",
    )
    campaign = Campaign(
        campaign_id="camp-test",
        brief="brief",
        contract=make_contract(),
    )
    outcome = manager.process(
        campaign,
        mode="long_video",
        url="http://fake.mp4",
        approve_manual_review=True,
        approved_by=None,
    )
    assert outcome.error is not None
    lowered = outcome.error.lower()
    assert "approved-by" in lowered or "approved_by" in lowered
    assert outcome.status is not CampaignStatus.COMPLETED


class _RecordingManager:
    def __init__(self) -> None:
        self.calls: int = 0

    def process(
        self,
        campaign: Campaign,
        *,
        mode: str = "long_video",
        url: str | None = None,
        resume: bool = False,
        approve_manual_review: bool = False,
        approved_by: str | None = None,
        audio_track_path: Path | None = None,
        audio_track_url: str | None = None,
    ) -> CampaignOutcome:
        _ = (mode, url, resume, approve_manual_review, approved_by)
        _ = (audio_track_path, audio_track_url)
        self.calls += 1
        return CampaignOutcome(
            campaign_id=campaign.campaign_id,
            archetype=Archetype.KNOWN,
            status=CampaignStatus.COMPLETED,
        )


def test_cli_campaign_requires_approved_by(tmp_path: Path) -> None:
    brief_file = tmp_path / "brief.txt"
    _ = brief_file.write_text("texto del brief", encoding="utf-8")
    mgr = _RecordingManager()
    code = main(
        [
            "campaign",
            str(brief_file),
            "--out",
            str(tmp_path / "out"),
            "--approve-manual-review",
        ],
        manager=mgr,
    )
    assert code == 1
    assert mgr.calls == 0
