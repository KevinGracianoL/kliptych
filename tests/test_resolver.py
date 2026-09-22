"""Tests del resolutor ContractDraft -> Contract."""

from pathlib import Path

from kliptych.assets import AssetRegistry
from kliptych.contract import Platform
from kliptych.gate import CheckStatus, Gate, GateStatus
from kliptych.resolver import (
    IssueCode,
    ResolutionResult,
    ResolutionStatus,
    resolve_contract,
)
from tests.support import (
    FakeProbe,
    candidate,
    conflict_candidate,
    make_asset_draft,
    make_draft,
    make_media,
    make_piece,
)


def _issue_fields(result: ResolutionResult, code: IssueCode) -> list[str]:
    return [issue.field for issue in result.issues if issue.code is code]


def test_full_draft_resolves_cleanly(tmp_path: Path) -> None:
    result = resolve_contract(make_draft(), registry=AssetRegistry(tmp_path))
    assert result.status is ResolutionStatus.RESOLVED
    assert result.resolved is True
    assert result.issues == ()
    contract = result.contract
    assert contract is not None
    assert contract.campaign_id == "camp-01"
    rules = contract.platforms[Platform.TIKTOK]
    assert rules.duration.min_s == 8
    assert rules.required_hashtags == ["#marca"]
    assert rules.required_mentions == ["@marca"]
    assert set(contract.rules.hard) == {
        "duration.min",
        "caption.required_hashtag",
        "caption.required_mention",
    }


def test_resolved_contract_passes_the_gate(tmp_path: Path) -> None:
    result = resolve_contract(make_draft(), registry=AssetRegistry(tmp_path))
    assert result.contract is not None
    artifact = tmp_path / "piece.mp4"
    _ = artifact.write_bytes(b"video")
    gate = Gate(FakeProbe(info=make_media()))
    gate_result = gate.run(
        contract=result.contract,
        piece=make_piece(artifact, caption="mira @marca #marca"),
        assets=AssetRegistry(tmp_path),
    )
    assert gate_result.status is GateStatus.PASSED
    assert all(check.status is CheckStatus.PASS for check in gate_result.checks)


def test_missing_campaign_id_goes_to_manual_review(tmp_path: Path) -> None:
    result = resolve_contract(make_draft(campaign_id=None), registry=AssetRegistry(tmp_path))
    assert result.status is ResolutionStatus.MANUAL_REVIEW
    assert result.contract is None
    assert _issue_fields(result, IssueCode.MISSING_REQUIRED) == ["campaign_id"]


def test_missing_mode_is_a_new_archetype(tmp_path: Path) -> None:
    result = resolve_contract(make_draft(mode=None), registry=AssetRegistry(tmp_path))
    assert result.status is ResolutionStatus.NEW_ARCHETYPE
    assert result.contract is None


def test_conflicting_mode_goes_to_manual_review(tmp_path: Path) -> None:
    result = resolve_contract(
        make_draft(mode=conflict_candidate()),
        registry=AssetRegistry(tmp_path),
    )
    assert result.status is ResolutionStatus.MANUAL_REVIEW
    assert _issue_fields(result, IssueCode.CONFLICT) == ["mode"]


def test_conflicting_duration_goes_to_manual_review(tmp_path: Path) -> None:
    result = resolve_contract(
        make_draft(
            platforms={
                "tiktok": {
                    "duration": {"min_s": conflict_candidate()},
                    "required_hashtags": candidate(["#marca"]),
                    "required_mentions": candidate(["@marca"]),
                }
            },
            rules={"hard": candidate(["caption.required_hashtag", "caption.required_mention"])},
        ),
        registry=AssetRegistry(tmp_path),
    )
    assert result.status is ResolutionStatus.MANUAL_REVIEW
    assert _issue_fields(result, IssueCode.CONFLICT) == ["platforms.tiktok.duration.min_s"]


def test_unclassified_restriction_defaults_to_hard(tmp_path: Path) -> None:
    result = resolve_contract(make_draft(rules=None), registry=AssetRegistry(tmp_path))
    assert result.status is ResolutionStatus.RESOLVED
    assert result.contract is not None
    assert set(result.contract.rules.hard) == {
        "duration.min",
        "caption.required_hashtag",
        "caption.required_mention",
    }
    assert sorted(_issue_fields(result, IssueCode.RULE_DEFAULTED)) == [
        "rules.caption.required_hashtag",
        "rules.caption.required_mention",
        "rules.duration.min",
    ]


def test_required_asset_is_registered(tmp_path: Path) -> None:
    _ = (tmp_path / "clip.mp4").write_bytes(b"video")
    draft = make_draft(
        assets={"required": [make_asset_draft()], "optional": []},
        rules=None,
    )
    result = resolve_contract(draft, registry=AssetRegistry(tmp_path))
    assert result.status is ResolutionStatus.RESOLVED
    assert result.contract is not None
    ref = result.contract.assets.required[0]
    assert ref.asset_id == "clip-01"
    assert ref.size_bytes == 5
    assert "assets.required" in result.contract.rules.hard


def test_unresolved_required_asset_goes_to_manual_review(tmp_path: Path) -> None:
    draft = make_draft(assets={"required": [make_asset_draft(uri="nope.mp4")], "optional": []})
    result = resolve_contract(draft, registry=AssetRegistry(tmp_path))
    assert result.status is ResolutionStatus.MANUAL_REVIEW
    assert result.contract is None
    assert _issue_fields(result, IssueCode.UNRESOLVED_ASSET) == ["assets.required[0]"]


def test_unresolved_optional_asset_is_dropped(tmp_path: Path) -> None:
    draft = make_draft(assets={"required": [], "optional": [make_asset_draft(uri="nope.mp4")]})
    result = resolve_contract(draft, registry=AssetRegistry(tmp_path))
    assert result.status is ResolutionStatus.RESOLVED
    assert result.contract is not None
    assert result.contract.assets.optional == []
    assert _issue_fields(result, IssueCode.OPTIONAL_ASSET_DROPPED) == ["assets.optional[0]"]


def test_registered_asset_with_same_uri_is_reused(tmp_path: Path) -> None:
    _ = (tmp_path / "clip.mp4").write_bytes(b"video")
    registry = AssetRegistry(tmp_path)
    first = registry.register(asset_id="clip-01", kind="video", uri="clip.mp4", origin="brief")
    draft = make_draft(assets={"required": [make_asset_draft()], "optional": []})
    result = resolve_contract(draft, registry=registry)
    assert result.contract is not None
    assert result.contract.assets.required[0] == first


def test_registered_asset_with_other_uri_fails(tmp_path: Path) -> None:
    _ = (tmp_path / "clip.mp4").write_bytes(b"video")
    _ = (tmp_path / "other.mp4").write_bytes(b"otro")
    registry = AssetRegistry(tmp_path)
    _ = registry.register(asset_id="clip-01", kind="video", uri="clip.mp4", origin="brief")
    draft = make_draft(assets={"required": [make_asset_draft(uri="other.mp4")], "optional": []})
    result = resolve_contract(draft, registry=registry)
    assert result.status is ResolutionStatus.MANUAL_REVIEW
    assert _issue_fields(result, IssueCode.UNRESOLVED_ASSET) == ["assets.required[0]"]


def test_official_required_without_urls_goes_to_manual_review(tmp_path: Path) -> None:
    draft = make_draft(
        platforms={
            "tiktok": {
                "duration": {"min_s": candidate(8)},
                "required_hashtags": candidate(["#marca"]),
                "required_mentions": candidate(["@marca"]),
                "audio_rule": candidate("official_required"),
            }
        },
        rules=None,
    )
    result = resolve_contract(draft, registry=AssetRegistry(tmp_path))
    assert result.status is ResolutionStatus.MANUAL_REVIEW
    assert _issue_fields(result, IssueCode.MISSING_REQUIRED) == ["official_audio"]


def test_official_audio_with_url_resolves(tmp_path: Path) -> None:
    draft = make_draft(
        platforms={
            "tiktok": {
                "duration": {"min_s": candidate(8)},
                "required_hashtags": candidate(["#marca"]),
                "required_mentions": candidate(["@marca"]),
                "audio_rule": candidate("official_required"),
            }
        },
        official_audio={"tiktok_url": candidate("https://example.com/audio.mp3")},
        rules=None,
    )
    result = resolve_contract(draft, registry=AssetRegistry(tmp_path))
    assert result.status is ResolutionStatus.RESOLVED
    assert result.contract is not None
    assert result.contract.official_audio is not None
    assert result.contract.official_audio.tiktok_url == "https://example.com/audio.mp3"


def test_partial_geo_target_goes_to_manual_review(tmp_path: Path) -> None:
    draft = make_draft(geo_target={"country": candidate("MX")})
    result = resolve_contract(draft, registry=AssetRegistry(tmp_path))
    assert result.status is ResolutionStatus.MANUAL_REVIEW
    assert _issue_fields(result, IssueCode.MISSING_REQUIRED) == ["geo_target"]


def test_attribution_value_without_type_goes_to_manual_review(tmp_path: Path) -> None:
    draft = make_draft(
        platforms={
            "tiktok": {
                "duration": {"min_s": candidate(8)},
                "required_hashtags": candidate(["#marca"]),
                "required_mentions": candidate(["@marca"]),
                "attribution": {"value": candidate("@marca")},
            }
        },
        rules=None,
    )
    result = resolve_contract(draft, registry=AssetRegistry(tmp_path))
    assert result.status is ResolutionStatus.MANUAL_REVIEW
    assert _issue_fields(result, IssueCode.MISSING_REQUIRED) == [
        "platforms.tiktok.attribution.type"
    ]


def test_attribution_none_drops_value(tmp_path: Path) -> None:
    draft = make_draft(
        platforms={
            "tiktok": {
                "duration": {"min_s": candidate(8)},
                "required_hashtags": candidate(["#marca"]),
                "required_mentions": candidate(["@marca"]),
                "attribution": {"type": candidate("none"), "value": candidate("@marca")},
            }
        },
        rules=None,
    )
    result = resolve_contract(draft, registry=AssetRegistry(tmp_path))
    assert result.contract is not None
    attribution = result.contract.platforms[Platform.TIKTOK].attribution
    assert attribution.type.value == "none"
    assert attribution.value is None


def test_conflicting_duration_bounds_go_to_manual_review(tmp_path: Path) -> None:
    draft = make_draft(
        platforms={
            "tiktok": {
                "duration": {"min_s": candidate(60), "max_s": candidate(8)},
                "required_hashtags": candidate(["#marca"]),
                "required_mentions": candidate(["@marca"]),
            }
        },
        rules=None,
    )
    result = resolve_contract(draft, registry=AssetRegistry(tmp_path))
    assert result.status is ResolutionStatus.MANUAL_REVIEW
    assert _issue_fields(result, IssueCode.INVALID_CONTRACT) == ["contract"]


def test_absent_watermark_relaxes(tmp_path: Path) -> None:
    result = resolve_contract(make_draft(watermark=None), registry=AssetRegistry(tmp_path))
    assert result.status is ResolutionStatus.RESOLVED
    assert result.contract is not None
    assert result.contract.watermark.required is False
    assert result.contract.watermark.visible_full_video is False


def test_missing_platforms_goes_to_manual_review(tmp_path: Path) -> None:
    result = resolve_contract(make_draft(platforms={}), registry=AssetRegistry(tmp_path))
    assert result.status is ResolutionStatus.MANUAL_REVIEW
    assert _issue_fields(result, IssueCode.MISSING_REQUIRED) == ["platforms"]


def test_missing_languages_goes_to_manual_review(tmp_path: Path) -> None:
    result = resolve_contract(make_draft(languages=None), registry=AssetRegistry(tmp_path))
    assert result.status is ResolutionStatus.MANUAL_REVIEW
    assert _issue_fields(result, IssueCode.MISSING_REQUIRED) == ["languages"]


def test_asset_without_uri_goes_to_manual_review(tmp_path: Path) -> None:
    draft = make_draft(
        assets={
            "required": [{"asset_id": candidate("clip-01"), "kind": candidate("video")}],
            "optional": [],
        }
    )
    result = resolve_contract(draft, registry=AssetRegistry(tmp_path))
    assert result.status is ResolutionStatus.MANUAL_REVIEW
    assert _issue_fields(result, IssueCode.MISSING_REQUIRED) == ["assets.required[0]"]


def test_conflict_on_optional_field_goes_to_manual_review(tmp_path: Path) -> None:
    result = resolve_contract(
        make_draft(spelling_locks=conflict_candidate()),
        registry=AssetRegistry(tmp_path),
    )
    assert result.status is ResolutionStatus.MANUAL_REVIEW
    assert _issue_fields(result, IssueCode.CONFLICT) == ["spelling_locks"]


def test_official_audio_with_instagram_url_resolves(tmp_path: Path) -> None:
    draft = make_draft(
        platforms={
            "tiktok": {
                "duration": {"min_s": candidate(8)},
                "required_hashtags": candidate(["#marca"]),
                "required_mentions": candidate(["@marca"]),
                "audio_rule": candidate("official_required"),
            }
        },
        official_audio={"instagram_url": candidate("https://example.com/audio.mp3")},
        rules=None,
    )
    result = resolve_contract(draft, registry=AssetRegistry(tmp_path))
    assert result.status is ResolutionStatus.RESOLVED
    assert result.contract is not None
    assert result.contract.official_audio is not None
    assert result.contract.official_audio.instagram_url == "https://example.com/audio.mp3"


def test_attribution_tag_resolves_with_value(tmp_path: Path) -> None:
    draft = make_draft(
        platforms={
            "tiktok": {
                "duration": {"min_s": candidate(8)},
                "required_hashtags": candidate(["#marca"]),
                "required_mentions": candidate(["@marca"]),
                "attribution": {"type": candidate("tag"), "value": candidate("@marca")},
            }
        },
        rules=None,
    )
    result = resolve_contract(draft, registry=AssetRegistry(tmp_path))
    assert result.contract is not None
    attribution = result.contract.platforms[Platform.TIKTOK].attribution
    assert attribution.type.value == "tag"
    assert attribution.value == "@marca"


def test_languages_pass_through_optional_fields(tmp_path: Path) -> None:
    draft = make_draft(
        languages={
            "source": candidate("es"),
            "subtitles": candidate("es"),
            "caption": candidate("es"),
            "voice": candidate("es"),
        }
    )
    result = resolve_contract(draft, registry=AssetRegistry(tmp_path))
    assert result.contract is not None
    assert result.contract.languages.subtitles == "es"
    assert result.contract.languages.voice == "es"


def test_registered_asset_with_dot_slash_uri_is_reused(tmp_path: Path) -> None:
    _ = (tmp_path / "clip.mp4").write_bytes(b"video")
    registry = AssetRegistry(tmp_path)
    first = registry.register(asset_id="clip-01", kind="video", uri="clip.mp4", origin="brief")
    draft = make_draft(assets={"required": [make_asset_draft(uri="./clip.mp4")], "optional": []})
    result = resolve_contract(draft, registry=registry)
    assert result.contract is not None
    assert result.contract.assets.required[0] == first


def test_resolution_result_round_trips_json(tmp_path: Path) -> None:
    result = resolve_contract(make_draft(), registry=AssetRegistry(tmp_path))
    restored = ResolutionResult.model_validate_json(result.model_dump_json())
    assert restored == result
