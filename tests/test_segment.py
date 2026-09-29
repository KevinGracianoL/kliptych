"""Tests unitarios de la selección de segmentos: sin red y sin llamadas al LLM."""

import json
import math
from typing import cast

import pytest
from pydantic import ValidationError

from kliptych.contract import Contract, Segment, TimestampRange
from kliptych.moments import Moment, MomentSource
from kliptych.segment import (
    SEGMENT_SYSTEM_PROMPT,
    LLMSegmentSelector,
    SegmentSelection,
    SegmentSelectionError,
)
from kliptych.transcribe import Transcript, Word


def _platform(*, min_s: int, max_s: int) -> dict[str, object]:
    return {
        "duration": {"min_s": min_s, "max_s": max_s},
        "caption_rules": {"must_mention": [], "first_line": None, "forbidden": []},
        "audio_rule": "own_clip",
        "required_hashtags": [],
        "required_mentions": [],
        "attribution": {"type": "none", "value": None},
        "link_rules": {"link_in_bio": False},
    }


def _contract(*, min_s: int = 8, max_s: int = 60) -> Contract:
    return Contract.model_validate(
        {
            "schema_version": "1.1",
            "campaign_id": "camp-long",
            "format": "video",
            "mode": "long_video",
            "platforms": {"tiktok": _platform(min_s=min_s, max_s=max_s)},
            "languages": {"source": "es", "subtitles": None, "caption": "es", "voice": None},
            "official_audio": None,
            "watermark": {"required": False, "asset_id": None, "visible_full_video": False},
            "spelling_locks": ["MarcaX"],
            "prohibitions": ["sin spoilers"],
            "rules": {
                "hard": [
                    "caption.forbidden",
                    "duration.min",
                    "duration.max",
                    "subtitles.spelling_lock",
                ],
                "recommended": [],
                "manual_review": ["audio.own_clip"],
            },
            "assets": {"required": [], "optional": []},
            "segments": [{"start_s": 0.0, "end_s": 8.0}],
            "geo_target": None,
            "min_views_for_payout": {"value": None, "enforcement": "post_publication_manual"},
            "analytics_proof_required": {"value": False, "enforcement": "post_publication_manual"},
        }
    )


def _incompatible_contract() -> Contract:
    return Contract.model_validate(
        {
            "schema_version": "1.1",
            "campaign_id": "camp-long",
            "format": "video",
            "mode": "long_video",
            "platforms": {
                "tiktok": _platform(min_s=0, max_s=10),
                "instagram_reels": _platform(min_s=20, max_s=30),
            },
            "languages": {"source": "es", "subtitles": None, "caption": "es", "voice": None},
            "official_audio": None,
            "watermark": {"required": False, "asset_id": None, "visible_full_video": False},
            "spelling_locks": [],
            "prohibitions": [],
            "rules": {
                "hard": ["duration.min", "duration.max"],
                "recommended": [],
                "manual_review": ["audio.own_clip"],
            },
            "assets": {"required": [], "optional": []},
            "segments": [{"start_s": 0.0, "end_s": 8.0}],
            "geo_target": None,
            "min_views_for_payout": {"value": None, "enforcement": "post_publication_manual"},
            "analytics_proof_required": {"value": False, "enforcement": "post_publication_manual"},
        }
    )


def _transcript(duration_s: float = 30.0) -> Transcript:
    words = (
        Word(start_s=0.0, end_s=0.5, text="hola", confidence=0.9, token_id=0),
        Word(start_s=0.6, end_s=1.0, text="mundo", confidence=0.8, token_id=1),
    )
    return Transcript(words=words, language="es", duration_s=duration_s, text="hola mundo")


def _moments() -> tuple[Moment, ...]:
    return (
        Moment(start_s=1.0, end_s=4.0, score=0.9, source=MomentSource.FUSED),
        Moment(start_s=5.0, end_s=9.0, score=0.5, source=MomentSource.FUSED),
    )


def _selector_with_prompt(
    *, transcript: Transcript | None = None, contract: Contract | None = None
) -> LLMSegmentSelector:
    selector = LLMSegmentSelector()
    _ = selector.build_prompt(
        _transcript() if transcript is None else transcript,
        _moments(),
        _contract() if contract is None else contract,
    )
    return selector


def _payload(selector: LLMSegmentSelector, contract: Contract | None = None) -> dict[str, object]:
    return selector.build_prompt(
        _transcript(),
        _moments(),
        _contract() if contract is None else contract,
    )


def test_build_prompt_includes_moments_transcript_and_contract_rules() -> None:
    payload = _payload(LLMSegmentSelector())
    moments = cast("list[dict[str, object]]", payload["moments"])
    assert moments == [
        {"start_s": 1.0, "end_s": 4.0, "score": 0.9, "source": "fused"},
        {"start_s": 5.0, "end_s": 9.0, "score": 0.5, "source": "fused"},
    ]
    transcript = cast("dict[str, object]", payload["transcript"])
    assert transcript["text"] == "hola mundo"
    assert transcript["duration_s"] == pytest.approx(30.0)
    contract = cast("dict[str, object]", payload["contract"])
    duration = cast("dict[str, object]", contract["duration"])
    assert duration == {"source_s": pytest.approx(30.0), "min_s": 8.0, "max_s": 60.0}
    assert contract["prohibitions"] == ["sin spoilers"]
    assert contract["spelling_locks"] == ["MarcaX"]
    assert contract["declared_segments"] == [{"start_s": 0.0, "end_s": 8.0}]


def test_build_prompt_uses_strictest_platform_duration_intersection() -> None:
    contract = Contract.model_validate(
        {
            "schema_version": "1.1",
            "campaign_id": "camp-long",
            "format": "video",
            "mode": "long_video",
            "platforms": {
                "tiktok": _platform(min_s=8, max_s=60),
                "instagram_reels": _platform(min_s=10, max_s=30),
            },
            "languages": {"source": "es", "subtitles": None, "caption": "es", "voice": None},
            "official_audio": None,
            "watermark": {"required": False, "asset_id": None, "visible_full_video": False},
            "spelling_locks": [],
            "prohibitions": [],
            "rules": {
                "hard": ["duration.min", "duration.max"],
                "recommended": [],
                "manual_review": ["audio.own_clip"],
            },
            "assets": {"required": [], "optional": []},
            "segments": [{"start_s": 0.0, "end_s": 8.0}],
            "geo_target": None,
            "min_views_for_payout": {"value": None, "enforcement": "post_publication_manual"},
            "analytics_proof_required": {"value": False, "enforcement": "post_publication_manual"},
        }
    )
    payload = _payload(LLMSegmentSelector(), contract=contract)
    rules = cast("dict[str, object]", payload["contract"])
    duration = cast("dict[str, object]", rules["duration"])
    assert duration["min_s"] == pytest.approx(10.0)
    assert duration["max_s"] == pytest.approx(30.0)


def test_build_prompt_rejects_incompatible_duration_ranges() -> None:
    selector = LLMSegmentSelector()
    with pytest.raises(SegmentSelectionError, match="incompatibles"):
        _ = selector.build_prompt(_transcript(), _moments(), _incompatible_contract())


def test_parse_response_accepts_mapping() -> None:
    selector = _selector_with_prompt()
    selection = selector.parse_response(
        {
            "segments": [{"start_s": 1.0, "end_s": 10.0}],
            "rationale": "el primer momento concentra la energía",
        }
    )
    assert selection == SegmentSelection(
        segments=(Segment(start_s=1.0, end_s=10.0),),
        rationale="el primer momento concentra la energía",
    )


def test_parse_response_accepts_json_string() -> None:
    selector = _selector_with_prompt()
    raw = json.dumps({"segments": [{"start_s": 0.0, "end_s": 8.0}], "rationale": "ok"})
    assert selector.parse_response(raw).segments == (Segment(start_s=0.0, end_s=8.0),)


def test_parse_response_accepts_json_bytes() -> None:
    selector = _selector_with_prompt()
    raw = b'{"segments": [{"start_s": 0.0, "end_s": 8.0}], "rationale": "ok"}'
    assert selector.parse_response(raw).segments == (Segment(start_s=0.0, end_s=8.0),)


def test_parse_response_rejects_out_of_bounds_segment() -> None:
    selector = _selector_with_prompt()
    with pytest.raises(SegmentSelectionError, match="excede el vídeo"):
        _ = selector.parse_response(
            {"segments": [{"start_s": 25.0, "end_s": 35.0}], "rationale": "tarde"}
        )


def test_parse_response_rejects_too_short_segment() -> None:
    selector = _selector_with_prompt()
    with pytest.raises(SegmentSelectionError, match="mínimo"):
        _ = selector.parse_response(
            {"segments": [{"start_s": 0.0, "end_s": 3.0}], "rationale": "corto"}
        )


def test_parse_response_rejects_too_long_segment() -> None:
    selector = _selector_with_prompt(contract=_contract(min_s=8, max_s=10))
    with pytest.raises(SegmentSelectionError, match="máximo"):
        _ = selector.parse_response(
            {"segments": [{"start_s": 0.0, "end_s": 25.0}], "rationale": "largo"}
        )


def test_parse_response_rejects_empty_segments() -> None:
    selector = _selector_with_prompt()
    with pytest.raises(SegmentSelectionError, match="no contiene segmentos"):
        _ = selector.parse_response({"segments": [], "rationale": "vacío"})


def test_parse_response_rejects_malformed_json() -> None:
    selector = _selector_with_prompt()
    with pytest.raises(SegmentSelectionError, match="SegmentSelection"):
        _ = selector.parse_response("no soy json")


def test_parse_response_rejects_non_mapping() -> None:
    selector = _selector_with_prompt()
    with pytest.raises(SegmentSelectionError, match="objeto JSON"):
        _ = selector.parse_response(["no", "es", "objeto"])


def test_parse_response_rejects_blank_rationale() -> None:
    selector = _selector_with_prompt()
    with pytest.raises(SegmentSelectionError, match="SegmentSelection"):
        _ = selector.parse_response({"segments": [{"start_s": 0.0, "end_s": 8.0}], "rationale": ""})


def test_parse_response_rejects_unordered_segment() -> None:
    selector = _selector_with_prompt()
    with pytest.raises(SegmentSelectionError, match="SegmentSelection"):
        _ = selector.parse_response(
            {"segments": [{"start_s": 8.0, "end_s": 8.0}], "rationale": "inválido"}
        )


def test_parse_response_requires_previous_build_prompt() -> None:
    with pytest.raises(SegmentSelectionError, match="build_prompt"):
        _ = LLMSegmentSelector().parse_response(
            {"segments": [{"start_s": 0.0, "end_s": 8.0}], "rationale": "ok"}
        )


def test_segment_selection_rejects_extra_fields() -> None:
    with pytest.raises(ValidationError):
        _ = SegmentSelection.model_validate(
            {
                "segments": [{"start_s": 0.0, "end_s": 8.0}],
                "rationale": "ok",
                "extra": 1,
            }
        )


def test_segment_selection_requires_rationale() -> None:
    with pytest.raises(ValidationError):
        _ = SegmentSelection(segments=(Segment(start_s=0.0, end_s=8.0),), rationale="")


def test_segment_selection_is_frozen() -> None:
    selection = SegmentSelection(segments=(Segment(start_s=0.0, end_s=8.0),), rationale="ok")
    with pytest.raises(ValidationError):
        selection.rationale = "otro"


def test_segment_respects_schema_constraints() -> None:
    with pytest.raises(ValidationError):
        _ = Segment(start_s=-1.0, end_s=8.0)
    with pytest.raises(ValidationError):
        _ = Segment(start_s=8.0, end_s=8.0)
    with pytest.raises(ValidationError):
        _ = Segment(start_s=0.0, end_s=float("inf"))


def test_system_prompt_describes_segment_selection() -> None:
    assert "segments" in SEGMENT_SYSTEM_PROMPT
    assert "rationale" in SEGMENT_SYSTEM_PROMPT


def test_mandatory_timestamp_ranges_override_llm_selection() -> None:
    base_contract = _contract(min_s=5, max_s=60)
    contract = base_contract.model_copy(
        update={"timestamp_ranges": (TimestampRange(start_sec=15.0, end_sec=30.0),)}
    )
    transcript = Transcript(
        language="es",
        duration_s=60.0,
        text="hola",
        words=(Word(start_s=0.0, end_s=1.0, text="hola", confidence=0.9, token_id=0),),
    )
    selector = LLMSegmentSelector()
    _ = selector.build_prompt(transcript, (), contract)

    llm_suggestion = {
        "segments": [{"start_s": 0.0, "end_s": 10.0}],
        "rationale": "llm proposed segment",
    }
    selection = selector.parse_response(llm_suggestion)
    assert len(selection.segments) == 1
    assert math.isclose(selection.segments[0].start_s, 15.0)
    assert math.isclose(selection.segments[0].end_s, 30.0)
    assert selection.rationale == "llm proposed segment"
