"""Tests de la CLI de Kliptych y el subcomando campaign (F1-PR1)."""

from collections.abc import Sequence
from pathlib import Path

import pytest

from kliptych.__main__ import main
from kliptych.campaign_manager import CampaignManager, CampaignOutcome
from kliptych.campaign_types import Campaign, CampaignStatus
from kliptych.git_proposals import PullRequest
from kliptych.intelligence import Archetype
from kliptych.orchestrator import PipelineResult, SlideshowResult

_ = CampaignOutcome.model_rebuild(
    _types_namespace={
        "PullRequest": PullRequest,
        "PipelineResult": PipelineResult,
        "SlideshowResult": SlideshowResult,
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
    ) -> CampaignOutcome:
        self.calls.append(
            {
                "campaign": campaign,
                "mode": mode,
                "url": url,
                "images": images,
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
