"""Tests de la CLI de Kliptych y el subcomando campaign (F1-PR1)."""

import json
from collections.abc import Sequence
from pathlib import Path
from typing import cast

import pytest

from kliptych.__main__ import main
from kliptych.campaign_manager import CampaignOutcome
from kliptych.campaign_types import Campaign, CampaignStatus
from kliptych.contract import Segment
from kliptych.exporter import DeliveryReport
from kliptych.git_proposals import PullRequest
from kliptych.intelligence import Archetype
from kliptych.orchestrator import PipelineResult, SlideshowResult
from kliptych.segment import SegmentSelection
from tests.support import make_contract, make_draft

_ = CampaignOutcome.model_rebuild(
    _types_namespace={
        "PullRequest": PullRequest,
        "PipelineResult": PipelineResult,
        "SlideshowResult": SlideshowResult,
        "DeliveryReport": DeliveryReport,
    }
)


class FakeCampaignManager:
    """Fake de CampaignManager para inyección de dependencias en la CLI."""

    def __init__(self, outcome: CampaignOutcome | None = None) -> None:
        self.outcome: CampaignOutcome = (
            outcome
            if outcome is not None
            else CampaignOutcome(
                campaign_id="test-camp",
                archetype=Archetype.KNOWN,
                status=CampaignStatus.COMPLETED,
            )
        )
        self.calls: list[dict[str, object]] = []

    def process(
        self,
        campaign: Campaign,
        *,
        mode: str = "long_video",
        url: str | None = None,
        images: Sequence[Path] | None = None,
        resume: bool = False,
        destination: Path | None = None,
        approve_manual_review: bool = False,
        **kwargs: object,
    ) -> CampaignOutcome:
        self.calls.append(
            {
                "campaign": campaign,
                "mode": mode,
                "url": url,
                "images": images,
                "resume": resume,
                "destination": destination,
                "approve_manual_review": approve_manual_review,
                **kwargs,
            }
        )
        return self.outcome


def test_cli_help_exits_zero() -> None:
    with pytest.raises(SystemExit) as exc_info:
        _ = main(["--help"])
    assert exc_info.value.code == 0


def test_cli_version_exits_zero() -> None:
    with pytest.raises(SystemExit) as exc_info:
        _ = main(["--version"])
    assert exc_info.value.code == 0


def test_cli_missing_required_arg_exits_two() -> None:
    with pytest.raises(SystemExit) as exc_info:
        _ = main(["campaign"])
    assert exc_info.value.code == 2


def test_cli_invalid_mode_exits_two(tmp_path: Path) -> None:
    brief = tmp_path / "brief.txt"
    _ = brief.write_text("brief content", encoding="utf-8")
    with pytest.raises(SystemExit) as exc_info:
        _ = main(["campaign", str(brief), "--out", str(tmp_path / "out"), "--mode", "invalid_mode"])
    assert exc_info.value.code == 2


def test_cli_campaign_delegates_to_manager(tmp_path: Path) -> None:
    brief = tmp_path / "brief.txt"
    _ = brief.write_text("brief content", encoding="utf-8")
    manager = FakeCampaignManager()

    code = main(
        ["campaign", str(brief), "--out", str(tmp_path / "out"), "--mode", "long_video"],
        manager=manager,
    )

    assert code == 0
    assert len(manager.calls) == 1
    assert manager.calls[0]["mode"] == "long_video"


def test_cli_campaign_error_outcome_returns_one(tmp_path: Path) -> None:
    brief = tmp_path / "brief.txt"
    _ = brief.write_text("brief content", encoding="utf-8")
    error_outcome = CampaignOutcome(
        campaign_id="test-camp",
        archetype=Archetype.NEW_ARCHETYPE,
        status=CampaignStatus.PENDING,
        error="fallo controlado",
    )
    manager = FakeCampaignManager(outcome=error_outcome)

    code = main(
        ["campaign", str(brief), "--out", str(tmp_path / "out")],
        manager=manager,
    )

    assert code == 1
    assert len(manager.calls) == 1


def test_cli_campaign_missing_brief_file_returns_one(tmp_path: Path) -> None:
    manager = FakeCampaignManager()
    code = main(
        ["campaign", str(tmp_path / "nonexistent.txt"), "--out", str(tmp_path / "out")],
        manager=manager,
    )
    assert code == 1
    assert len(manager.calls) == 0


def test_cli_campaign_logs_pr_url_and_exits_zero(tmp_path: Path) -> None:
    """When CampaignManager returns a PR, CLI logs URL and exits 0."""
    pr_outcome = CampaignOutcome(
        campaign_id="test-camp",
        archetype=Archetype.NEW_ARCHETYPE,
        status=CampaignStatus.MANUAL_REVIEW,
        pull_request=PullRequest(
            url="https://github.com/test/repo/pull/1",
            branch="proposal/test-camp",
            title="Propuesta test",
            body="body",
            campaign_id="test-camp",
            archetype=Archetype.NEW_ARCHETYPE,
        ),
    )
    manager = FakeCampaignManager(outcome=pr_outcome)
    brief = tmp_path / "brief.txt"
    _ = brief.write_text("brief content", encoding="utf-8")
    code = main(["campaign", str(brief), "--out", str(tmp_path / "out")], manager=manager)
    assert code == 0


def test_cli_campaign_logs_video_and_exits_zero(tmp_path: Path) -> None:
    """When CampaignManager returns pipeline_result, CLI logs path and exits 0."""
    video_outcome = CampaignOutcome(
        campaign_id="test-camp",
        archetype=Archetype.KNOWN,
        status=CampaignStatus.COMPLETED,
        pipeline_result=PipelineResult(
            source=tmp_path / "source.mp4",
            transcript=None,
            moments=(),
            selection=SegmentSelection(
                segments=(Segment(start_s=0.0, end_s=1.0),), rationale="test"
            ),
            reframe=None,
            subtitles=None,
            final_video=tmp_path / "final.mp4",
            cleaning=(),
        ),
    )
    manager = FakeCampaignManager(outcome=video_outcome)
    brief = tmp_path / "brief.txt"
    _ = brief.write_text("brief content", encoding="utf-8")
    code = main(["campaign", str(brief), "--out", str(tmp_path / "out")], manager=manager)
    assert code == 0


def test_cli_campaign_missing_env_returns_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When required env vars are missing, CLI logs config error and exits 1."""
    brief = tmp_path / "brief.txt"
    _ = brief.write_text("brief content", encoding="utf-8")
    monkeypatch.delenv("KLIPTYCH_LLM_BASE_URL", raising=False)
    monkeypatch.delenv("KLIPTYCH_LLM_API_KEY", raising=False)
    monkeypatch.delenv("KLIPTYCH_LLM_MODEL", raising=False)
    code = main(["campaign", str(brief), "--out", str(tmp_path / "out")])
    assert code == 1


def _make_provenance_draft_json() -> str:
    def update_spans(data: object, text: str) -> None:
        if isinstance(data, dict):
            dict_data = cast("dict[str, object]", data)
            if "quote" in dict_data and "start" in dict_data and "end" in dict_data:
                q = str(dict_data["quote"])
                idx = text.find(q)
                dict_data["start"] = idx
                dict_data["end"] = idx + len(q)
            else:
                for v in dict_data.values():
                    update_spans(v, text)
        elif isinstance(data, list):
            list_data = cast("list[object]", data)
            for item in list_data:
                update_spans(item, text)

    text = make_draft().model_dump_json()
    for _ in range(5):
        d = cast("object", json.loads(text))
        update_spans(d, text)
        new_text = json.dumps(d)
        if new_text == text:
            break
        text = new_text
    return text


def test_cli_campaign_resolves_contract_when_brief_is_json(tmp_path: Path) -> None:
    """When brief is valid ContractDraft JSON, contract is resolved and attached."""
    draft_json = _make_provenance_draft_json()
    brief = tmp_path / "brief.txt"
    _ = brief.write_text(draft_json, encoding="utf-8")
    manager = FakeCampaignManager()
    code = main(["campaign", str(brief), "--out", str(tmp_path / "out")], manager=manager)
    assert code == 0
    assert len(manager.calls) == 1
    campaign = cast("Campaign", manager.calls[0]["campaign"])
    assert campaign.contract is not None
    assert campaign.contract.campaign_id == "camp-01"


def test_cli_campaign_logs_out_dir(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """When campaign runs, output directory is logged."""
    brief = tmp_path / "brief.txt"
    _ = brief.write_text("brief content", encoding="utf-8")
    manager = FakeCampaignManager()
    code = main(
        ["campaign", str(brief), "--out", str(tmp_path / "custom_out")],
        manager=manager,
    )
    assert code == 0
    captured = capsys.readouterr()
    assert f"Directorio de salida: {tmp_path / 'custom_out'}" in captured.err


def test_cli_campaign_blocks_raw_json_contract_without_provenance(tmp_path: Path) -> None:
    """Raw Contract JSON without ContractDraft/provenance is rejected."""
    raw_contract_json = make_contract().model_dump_json()
    brief = tmp_path / "raw_contract.txt"
    _ = brief.write_text(raw_contract_json, encoding="utf-8")
    manager = FakeCampaignManager()
    code = main(["campaign", str(brief), "--out", str(tmp_path / "out")], manager=manager)
    assert code == 0
    assert len(manager.calls) == 1
    campaign = cast("Campaign", manager.calls[0]["campaign"])
    assert campaign.contract is None
