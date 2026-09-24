"""Tests de los tipos del ciclo de vida de una campaña (fase E)."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from kliptych.campaign_types import Campaign, CampaignStatus, PendingCampaign, route_campaign
from kliptych.intelligence import Archetype, ArchetypeClassification
from tests.support import make_contract


def test_campaign_status_enum_values() -> None:
    assert {status.value for status in CampaignStatus} == {
        "PENDING",
        "CLASSIFIED",
        "PROCESSING",
        "COMPLETED",
        "MANUAL_REVIEW",
    }


def test_campaign_defaults_to_pending() -> None:
    campaign = Campaign(campaign_id="camp-01", brief="brief crudo")
    assert campaign.status is CampaignStatus.PENDING
    assert campaign.archetype is None
    assert campaign.classification is None
    assert campaign.contract is None
    assert campaign.variations_log is None


def test_campaign_accepts_classification_and_contract() -> None:
    classification = ArchetypeClassification(
        archetype=Archetype.KNOWN_WITH_VARIATION,
        rationale="valor nuevo",
        variations=("duration.max=45",),
    )
    campaign = Campaign(
        campaign_id="camp-01",
        brief="brief crudo",
        status=CampaignStatus.PENDING,
        archetype=Archetype.KNOWN_WITH_VARIATION,
        classification=classification,
        contract=make_contract(),
        variations_log=Path("campaigns/variations.md"),
    )
    assert campaign.classification is classification
    assert campaign.archetype is Archetype.KNOWN_WITH_VARIATION
    assert campaign.variations_log == Path("campaigns/variations.md")


def test_campaign_is_frozen() -> None:
    campaign = Campaign(campaign_id="camp-01", brief="brief crudo")
    with pytest.raises(ValidationError):
        campaign.status = CampaignStatus.COMPLETED


def test_campaign_forbids_extra_fields() -> None:
    with pytest.raises(ValidationError, match="invented"):
        _ = Campaign.model_validate({"campaign_id": "camp-01", "brief": "brief", "invented": True})


@pytest.mark.parametrize(
    ("field_name", "payload"),
    [
        ("campaign_id", {"campaign_id": "", "brief": "brief"}),
        ("brief", {"campaign_id": "camp-01", "brief": ""}),
    ],
)
def test_campaign_requires_non_empty_text(field_name: str, payload: dict[str, str]) -> None:
    with pytest.raises(ValidationError, match=field_name):
        _ = Campaign.model_validate(payload)


@pytest.mark.parametrize(
    ("archetype", "expected"),
    [
        (Archetype.KNOWN, CampaignStatus.CLASSIFIED),
        (Archetype.KNOWN_WITH_VARIATION, CampaignStatus.PENDING),
        (Archetype.NEW_ARCHETYPE, CampaignStatus.MANUAL_REVIEW),
    ],
)
def test_every_archetype_has_a_status_transition(
    archetype: Archetype, expected: CampaignStatus
) -> None:
    variations = ("variacion",) if archetype is Archetype.KNOWN_WITH_VARIATION else ()
    classification = ArchetypeClassification(
        archetype=archetype, rationale="ok", variations=variations
    )
    assert route_campaign(classification) is expected


def test_pending_campaign_is_frozen() -> None:
    pending = PendingCampaign(
        campaign=Campaign(campaign_id="camp-01", brief="brief"),
        reason="falta un modo",
    )
    field_name = "reason"
    with pytest.raises(AttributeError):
        setattr(pending, field_name, "otro")
