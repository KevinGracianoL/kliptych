"""Tests de la generación de configuración de campañas (fase E).

La generación es pura: no toca red, Git ni el sistema de archivos.
"""

import json
from datetime import datetime
from typing import cast

import pytest
from pydantic import ValidationError

from kliptych.campaign_config import (
    CampaignConfig,
    CampaignConfigError,
    generate_campaign_config,
    variations_md_entry,
)
from kliptych.campaign_types import Campaign, CampaignStatus
from kliptych.intelligence import Archetype, ArchetypeClassification
from tests.support import make_contract


def make_campaign(
    *,
    campaign_id: str = "camp-01",
    archetype: Archetype = Archetype.KNOWN_WITH_VARIATION,
    variations: tuple[str, ...] = ("duration.max=45",),
    with_classification: bool = True,
) -> Campaign:
    classification = (
        ArchetypeClassification(
            archetype=archetype,
            rationale="valor nuevo en el contrato",
            variations=variations,
        )
        if with_classification
        else None
    )
    return Campaign(
        campaign_id=campaign_id,
        brief="brief crudo",
        status=CampaignStatus.PENDING,
        archetype=archetype,
        classification=classification,
        contract=make_contract(),
    )


def test_campaign_config_is_frozen() -> None:
    config = CampaignConfig(
        campaign_id="camp-01",
        archetype=Archetype.NEW_ARCHETYPE,
        contract={"campaign_id": "camp-01"},
        created_at="2026-09-23T00:00:00+00:00",
    )
    with pytest.raises(ValidationError):
        config.campaign_id = "camp-02"


def test_campaign_config_forbids_extra_fields() -> None:
    with pytest.raises(ValidationError, match="invented"):
        _ = CampaignConfig.model_validate(
            {
                "campaign_id": "camp-01",
                "archetype": "NEW_ARCHETYPE",
                "contract": {},
                "created_at": "2026-09-23T00:00:00+00:00",
                "invented": True,
            }
        )


def test_campaign_config_defaults_variations_to_empty() -> None:
    config = CampaignConfig(
        campaign_id="camp-01",
        archetype=Archetype.NEW_ARCHETYPE,
        contract={},
        created_at="2026-09-23T00:00:00+00:00",
    )
    assert config.variations == ()


def _payload(raw: str) -> dict[str, object]:
    return cast("dict[str, object]", json.loads(raw))


def test_generate_campaign_config_produces_valid_json() -> None:
    raw = generate_campaign_config(make_campaign(), make_contract())
    payload = _payload(raw)
    assert payload["campaign_id"] == "camp-01"
    assert payload["archetype"] == "KNOWN_WITH_VARIATION"
    assert payload["variations"] == ["duration.max=45"]
    contract = cast("dict[str, object]", payload["contract"])
    assert contract["campaign_id"] == "camp-test"
    assert contract["schema_version"] == "1.1"


def test_generate_campaign_config_new_archetype_has_no_variations() -> None:
    campaign = make_campaign(
        archetype=Archetype.NEW_ARCHETYPE,
        variations=(),
    )
    payload = _payload(generate_campaign_config(campaign, make_contract()))
    assert payload["archetype"] == "NEW_ARCHETYPE"
    assert payload["variations"] == []


def test_generate_campaign_config_captures_variations() -> None:
    campaign = make_campaign(variations=("duration.max=45", "audio.rule=official_required"))
    payload = _payload(generate_campaign_config(campaign, make_contract()))
    assert payload["variations"] == ["duration.max=45", "audio.rule=official_required"]


def test_generate_campaign_config_created_at_is_iso() -> None:
    payload = _payload(generate_campaign_config(make_campaign(), make_contract()))
    created_at = datetime.fromisoformat(cast("str", payload["created_at"]))
    assert created_at.tzinfo is not None


def test_generate_campaign_config_requires_archetype() -> None:
    campaign = Campaign(campaign_id="camp-01", brief="brief crudo")
    with pytest.raises(CampaignConfigError, match="arquetipo"):
        _ = generate_campaign_config(campaign, make_contract())


def test_generate_campaign_config_falls_back_to_classification() -> None:
    campaign = make_campaign()
    campaign_without_archetype = campaign.model_copy(update={"archetype": None})
    payload = _payload(generate_campaign_config(campaign_without_archetype, make_contract()))
    assert payload["archetype"] == "KNOWN_WITH_VARIATION"


def test_variations_md_entry_format() -> None:
    entry = variations_md_entry(make_campaign())
    assert entry.startswith("## camp-01\n")
    assert "- arquetipo: KNOWN_WITH_VARIATION" in entry
    assert "- variación: duration.max=45" in entry
    assert entry.endswith("\n")


def test_variations_md_entry_lists_every_variation() -> None:
    entry = variations_md_entry(make_campaign(variations=("uno", "dos")))
    assert "- variación: uno" in entry
    assert "- variación: dos" in entry


def test_variations_md_entry_requires_classification() -> None:
    campaign = make_campaign(with_classification=False)
    with pytest.raises(CampaignConfigError, match="clasificación"):
        _ = variations_md_entry(campaign)
