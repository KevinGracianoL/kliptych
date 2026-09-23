"""Tests del contrato validado (schema v1.0)."""

from datetime import UTC, datetime
from typing import cast

import pytest
from pydantic import ValidationError

from kliptych.contract import AssetRef, Contract, Platform, contract_digest

_SHA256 = "a" * 64


def _asset(**overrides: object) -> dict[str, object]:
    data: dict[str, object] = {
        "asset_id": "clip-01",
        "kind": "video",
        "uri": "assets/samples/clip-01.mp4",
        "sha256": _SHA256,
        "size_bytes": 1024,
        "mime": "video/mp4",
        "origin": "brief",
        "license": None,
        "resolved_at": datetime(2026, 9, 22, 12, 0, tzinfo=UTC).isoformat(),
    }
    data.update(overrides)
    return data


def _platform(**overrides: object) -> dict[str, object]:
    data: dict[str, object] = {
        "duration": {"min_s": 8, "max_s": 60},
        "caption_rules": {"must_mention": [], "first_line": None, "forbidden": []},
        "audio_rule": "own_clip",
        "required_hashtags": ["#marca"],
        "required_mentions": ["@marca"],
        "attribution": {"type": "tag", "value": "@marca"},
        "link_rules": {"link_in_bio": False},
    }
    data.update(overrides)
    return data


def _contract_data(**overrides: object) -> dict[str, object]:
    data: dict[str, object] = {
        "schema_version": "1.0",
        "campaign_id": "camp-given-clips-01",
        "format": "video",
        "mode": "given_clips",
        "platforms": {"tiktok": _platform()},
        "languages": {"source": "es", "subtitles": None, "caption": "es", "voice": None},
        "official_audio": None,
        "watermark": {"required": True, "asset_id": "wm-marca", "visible_full_video": True},
        "spelling_locks": ["MarcaX"],
        "prohibitions": ["música con copyright"],
        "rules": {
            "hard": [
                "artifact.integrity",
                "assets.required",
                "caption.forbidden",
                "caption.required_hashtag",
                "caption.required_mention",
                "duration.max",
                "duration.min",
                "subtitles.spelling_lock",
                "watermark.full_video",
            ],
            "recommended": [],
            "manual_review": [],
        },
        "assets": {"required": [_asset()], "optional": []},
        "geo_target": None,
        "min_views_for_payout": {"value": 10000, "enforcement": "post_publication_manual"},
        "analytics_proof_required": {"value": True, "enforcement": "post_publication_manual"},
    }
    data.update(overrides)
    return data


def _build(**overrides: object) -> Contract:
    return Contract.model_validate(_contract_data(**overrides))


def test_minimal_contract_is_valid() -> None:
    contract = _build()
    assert contract.schema_version == "1.0"
    assert contract.mode.value == "given_clips"
    assert contract.platforms[Platform.TIKTOK].duration.min_s == 8
    assert contract.assets.required[0].asset_id == "clip-01"


@pytest.mark.parametrize(
    "asset_id",
    [
        "..\\..\\pwned",
        "../../pwned",
        "C:\\abs\\pwned",
        "\\\\server\\share\\pwned",
        "con",
        "x" * 65,
    ],
)
def test_asset_id_rejects_unsafe_segments(asset_id: str) -> None:
    with pytest.raises(ValidationError, match="asset_id"):
        _ = AssetRef.model_validate(_asset(asset_id=asset_id))


def test_asset_id_accepts_safe_segment() -> None:
    assert AssetRef.model_validate(_asset()).asset_id == "clip-01"


def test_watermark_asset_id_rejects_unsafe_segment() -> None:
    with pytest.raises(ValidationError, match="asset_id"):
        _ = _build(
            watermark={
                "required": True,
                "asset_id": "../escape",
                "visible_full_video": True,
            }
        )


def test_contract_digest_ignores_resolved_at_but_not_asset_changes() -> None:
    base = _build()
    moved = {**_asset(), "resolved_at": datetime(2030, 1, 1, tzinfo=UTC).isoformat()}
    changed = {**_asset(), "sha256": "b" * 64}
    assert contract_digest(base) == contract_digest(
        _build(assets={"required": [moved], "optional": []})
    )
    assert contract_digest(base) != contract_digest(
        _build(assets={"required": [changed], "optional": []})
    )


def test_contract_round_trips_through_json() -> None:
    contract = _build()
    restored = Contract.model_validate_json(contract.model_dump_json())
    assert restored == contract


def test_duration_min_cannot_exceed_max() -> None:
    with pytest.raises(ValidationError, match="min_s"):
        _ = _build(platforms={"tiktok": _platform(duration={"min_s": 60, "max_s": 8})})


def test_negative_duration_is_rejected() -> None:
    with pytest.raises(ValidationError, match="duration"):
        _ = _build(platforms={"tiktok": _platform(duration={"min_s": -1, "max_s": 60})})


def test_rule_cannot_be_classified_twice() -> None:
    with pytest.raises(ValidationError, match="duplicad"):
        _ = _build(
            rules={
                "hard": ["duration.min"],
                "recommended": ["duration.min"],
                "manual_review": [],
            }
        )


def test_rule_ids_must_be_dotted() -> None:
    with pytest.raises(ValidationError, match=r"rules\.hard"):
        _ = _build(rules={"hard": ["duration"], "recommended": [], "manual_review": []})


def test_restriction_without_classified_rule_is_rejected() -> None:
    with pytest.raises(ValidationError, match="clasificada"):
        _ = _build(rules={"hard": [], "recommended": [], "manual_review": []})


def test_campaign_id_length_is_bounded() -> None:
    with pytest.raises(ValidationError, match="campaign_id"):
        _ = _build(campaign_id="a" * 65)


def test_mention_without_at_prefix_is_rejected() -> None:
    with pytest.raises(ValidationError, match="required_mentions"):
        _ = _build(platforms={"tiktok": _platform(required_mentions=["marca"])})


def test_must_mention_without_at_prefix_is_rejected() -> None:
    with pytest.raises(ValidationError, match="must_mention"):
        _ = _build(
            platforms={
                "tiktok": _platform(
                    caption_rules={"must_mention": ["marca"], "first_line": None, "forbidden": []}
                )
            }
        )


def test_hashtag_without_hash_prefix_is_rejected() -> None:
    with pytest.raises(ValidationError, match="required_hashtags"):
        _ = _build(platforms={"tiktok": _platform(required_hashtags=["marca"])})


def test_partial_watermark_requires_present_rule() -> None:
    data = _contract_data()
    data["watermark"] = {"required": True, "asset_id": "wm-marca", "visible_full_video": False}
    rules = cast("dict[str, object]", data["rules"])
    hard = cast("list[str]", rules["hard"])
    rules["hard"] = [rule for rule in hard if rule != "watermark.full_video"] + [
        "watermark.present"
    ]
    contract = Contract.model_validate(data)
    assert contract.watermark.required is True
    assert contract.watermark.visible_full_video is False


def test_partial_watermark_without_rule_is_rejected() -> None:
    data = _contract_data()
    data["watermark"] = {"required": True, "asset_id": "wm-marca", "visible_full_video": False}
    rules = cast("dict[str, object]", data["rules"])
    hard = cast("list[str]", rules["hard"])
    rules["hard"] = [rule for rule in hard if rule != "watermark.full_video"]
    with pytest.raises(ValidationError, match=r"watermark\.present"):
        _ = Contract.model_validate(data)


def test_full_video_watermark_requires_classified_rule() -> None:
    data = _contract_data()
    data["watermark"] = {"required": True, "asset_id": "wm-marca", "visible_full_video": True}
    rules = cast("dict[str, object]", data["rules"])
    hard = cast("list[str]", rules["hard"])
    rules["hard"] = [rule for rule in hard if rule != "watermark.full_video"]
    with pytest.raises(ValidationError, match=r"watermark\.full_video"):
        _ = Contract.model_validate(data)


def test_slideshow_mode_requires_slideshow_format() -> None:
    with pytest.raises(ValidationError, match="slideshow"):
        _ = _build(mode="slideshow")


def test_slideshow_format_requires_slideshow_mode() -> None:
    with pytest.raises(ValidationError, match="slideshow"):
        _ = _build(format="slideshow")


def test_official_required_audio_needs_official_audio_asset() -> None:
    with pytest.raises(ValidationError, match="official_audio"):
        _ = _build(platforms={"tiktok": _platform(audio_rule="official_required")})


def test_official_audio_requires_at_least_one_url() -> None:
    with pytest.raises(ValidationError, match="URL"):
        _ = _build(
            platforms={"tiktok": _platform(audio_rule="official_required")},
            official_audio={"tiktok_url": None, "instagram_url": None},
        )


def test_official_audio_with_url_is_valid() -> None:
    contract = _build(
        platforms={"tiktok": _platform(audio_rule="official_required")},
        official_audio={"tiktok_url": "https://example.com/audio", "instagram_url": None},
    )
    assert contract.official_audio is not None
    assert contract.official_audio.tiktok_url == "https://example.com/audio"


def test_platforms_cannot_be_empty() -> None:
    with pytest.raises(ValidationError, match="platforms"):
        _ = _build(platforms={})


def test_unknown_platform_is_rejected() -> None:
    with pytest.raises(ValidationError, match="platforms"):
        _ = _build(platforms={"youtube": _platform()})


def test_asset_sha256_must_be_a_full_digest() -> None:
    with pytest.raises(ValidationError, match="sha256"):
        _ = _build(assets={"required": [_asset(sha256="abc")], "optional": []})


def test_asset_ids_must_be_unique() -> None:
    with pytest.raises(ValidationError, match="asset_id"):
        _ = _build(
            assets={
                "required": [_asset(), _asset(kind="image")],
                "optional": [],
            }
        )


def test_attribution_requires_value_when_not_none() -> None:
    with pytest.raises(ValidationError, match="value"):
        _ = _build(platforms={"tiktok": _platform(attribution={"type": "tag", "value": None})})


def test_attribution_without_value_is_valid_when_type_is_none() -> None:
    contract = _build(platforms={"tiktok": _platform(attribution={"type": "none", "value": None})})
    assert contract.platforms[Platform.TIKTOK].attribution.type.value == "none"


def test_geo_min_pct_is_a_percentage() -> None:
    with pytest.raises(ValidationError, match="min_pct"):
        _ = _build(geo_target={"country": "MX", "min_pct": 150})


def test_geo_country_is_iso_alpha2() -> None:
    with pytest.raises(ValidationError, match="country"):
        _ = _build(geo_target={"country": "México", "min_pct": 60})


def test_post_publication_enforcement_is_fixed() -> None:
    with pytest.raises(ValidationError, match="enforcement"):
        _ = _build(min_views_for_payout={"value": 10000, "enforcement": "gate"})


def test_schema_version_is_locked_to_v1() -> None:
    with pytest.raises(ValidationError, match="schema_version"):
        _ = _build(schema_version="2.0")


def test_unknown_fields_are_rejected() -> None:
    with pytest.raises(ValidationError, match="Extra inputs"):
        _ = _build(promesa_extra="subir a TikTok por API")


def test_contract_is_immutable() -> None:
    contract = _build()
    with pytest.raises(ValidationError):
        contract.campaign_id = "otra"
