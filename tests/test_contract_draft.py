"""Tests del ContractDraft con evidencia por campo."""

import pytest
from pydantic import ValidationError

from kliptych.contract import Confidence, ContractDraft, Platform


def _evidence(quote: str) -> dict[str, object]:
    return {"quote": quote, "start": 0, "end": len(quote), "location": "brief.md#l1"}


def _candidate(value: object, quote: str, confidence: str = "explicit") -> dict[str, object]:
    return {"value": value, "evidence": _evidence(quote), "confidence": confidence}


def test_empty_draft_is_valid() -> None:
    draft = ContractDraft()
    assert draft.campaign_id is None
    assert draft.platforms == {}


def test_draft_carries_evidence_per_field() -> None:
    draft = ContractDraft.model_validate(
        {
            "schema_version": "1.0",
            "campaign_id": _candidate("camp-01", "Campaña camp-01"),
            "format": _candidate("video", "formato video vertical"),
            "mode": _candidate("given_clips", "usar los clips entregados"),
            "platforms": {
                "tiktok": {
                    "duration": {
                        "min_s": _candidate(8, "El video debe durar como mínimo ocho segundos"),
                    },
                    "required_hashtags": _candidate(["#marca"], "incluir #marca"),
                }
            },
        }
    )
    assert draft.format is not None
    assert draft.format.value is not None
    assert draft.format.confidence is Confidence.EXPLICIT

    platform = draft.platforms[Platform.TIKTOK]
    assert platform.duration is not None
    assert platform.duration.min_s is not None
    assert platform.duration.min_s.value == 8
    assert platform.duration.min_s.evidence is not None
    assert platform.required_hashtags is not None
    assert platform.required_hashtags.value == ["#marca"]


def test_draft_field_without_evidence_cannot_be_explicit() -> None:
    with pytest.raises(ValidationError, match="evidencia"):
        _ = ContractDraft.model_validate(
            {"campaign_id": {"value": "camp-01", "confidence": "explicit"}}
        )


def test_draft_forbids_unknown_fields() -> None:
    with pytest.raises(ValidationError, match="Extra inputs"):
        _ = ContractDraft.model_validate(
            {"campaign_id": _candidate("camp-01", "x"), "inventado": 1}
        )


def test_missing_candidate_in_draft_stays_unresolved() -> None:
    draft = ContractDraft.model_validate({"campaign_id": {"confidence": "missing"}})
    assert draft.campaign_id is not None
    assert draft.campaign_id.value is None
    assert draft.campaign_id.status is Confidence.MISSING


def test_draft_round_trips_through_json() -> None:
    draft = ContractDraft.model_validate(
        {
            "campaign_id": _candidate("camp-01", "Campaña camp-01"),
            "platforms": {
                "tiktok": {"audio_rule": _candidate("own_clip", "audio del propio clip")}
            },
        }
    )
    restored = ContractDraft.model_validate_json(draft.model_dump_json())
    assert restored == draft


def test_validation_errors_hide_input_values() -> None:
    canary = "CANARIO123"
    with pytest.raises(ValidationError) as excinfo:
        _ = ContractDraft.model_validate(
            {"campaign_id": {"value": canary, "confidence": "explicit"}}
        )
    assert canary not in str(excinfo.value)
