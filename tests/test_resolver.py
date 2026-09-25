"""Tests del resolutor ContractDraft -> Contract."""

import os
from pathlib import Path

import pytest

from kliptych.assets import AssetRegistry
from kliptych.contract import ContractDraft, Platform, Segment
from kliptych.gate import CheckResult, CheckStatus, Gate, GateResult, GateStatus
from kliptych.resolver import (
    IssueCode,
    ProvenanceError,
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


def _check(result: GateResult, rule_id: str) -> CheckResult:
    matches = [check for check in result.checks if check.id == rule_id]
    assert len(matches) == 1
    return matches[0]


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
        "artifact.integrity",
        "artifact.video_stream",
        "duration.min",
        "caption.required_hashtag",
        "caption.required_mention",
    }


def test_resolved_contract_passes_the_gate(tmp_path: Path) -> None:
    registry = AssetRegistry(tmp_path)
    result = resolve_contract(make_draft(), registry=registry)
    assert result.contract is not None
    artifact = tmp_path / "piece.mp4"
    _ = artifact.write_bytes(b"video")
    gate = Gate(FakeProbe(info=make_media()))
    gate_result = gate.run(
        contract=result.contract,
        piece=make_piece(artifact, caption="mira @marca #marca"),
        assets=registry,
    )
    assert gate_result.status is GateStatus.PASSED
    assert all(check.status is CheckStatus.PASS for check in gate_result.checks)


def test_resolved_contract_with_required_asset_passes_the_gate(tmp_path: Path) -> None:
    _ = (tmp_path / "clip.mp4").write_bytes(b"video")
    registry = AssetRegistry(tmp_path)
    draft = make_draft(assets={"required": [make_asset_draft()], "optional": []})
    result = resolve_contract(draft, registry=registry)
    assert result.contract is not None
    assert "clip-01" in registry.assets
    artifact = tmp_path / "piece.mp4"
    _ = artifact.write_bytes(b"video")
    gate = Gate(FakeProbe(info=make_media()))
    gate_result = gate.run(
        contract=result.contract,
        piece=make_piece(artifact, caption="mira @marca #marca"),
        assets=registry,
    )
    assert gate_result.status is GateStatus.PASSED


def test_base_rules_block_a_missing_artifact(tmp_path: Path) -> None:
    result = resolve_contract(make_draft(rules=None), registry=AssetRegistry(tmp_path))
    assert result.contract is not None
    assert "artifact.integrity" in result.contract.rules.hard
    gate = Gate(FakeProbe(info=make_media()))
    gate_result = gate.run(
        contract=result.contract,
        piece=make_piece(tmp_path / "no-existe.mp4", caption="mira @marca #marca"),
        assets=AssetRegistry(tmp_path),
    )
    assert gate_result.status is GateStatus.REJECTED
    assert _check(gate_result, "artifact.integrity").status is CheckStatus.FAIL


def test_audio_rule_seeds_audio_present(tmp_path: Path) -> None:
    draft = make_draft(
        platforms={
            "tiktok": {
                "duration": {"min_s": candidate(8)},
                "required_hashtags": candidate(["#marca"]),
                "required_mentions": candidate(["@marca"]),
                "audio_rule": candidate("no_trending"),
            }
        },
        rules=None,
    )
    result = resolve_contract(draft, registry=AssetRegistry(tmp_path))
    assert result.contract is not None
    assert "audio.present" in result.contract.rules.hard


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
        "artifact.integrity",
        "artifact.video_stream",
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


def test_required_watermark_rule_is_forced_hard(tmp_path: Path) -> None:
    draft = make_draft(
        watermark={
            "required": candidate(value=True),
            "visible_full_video": candidate(value=True),
        },
        rules={
            "hard": candidate(["duration.min", "caption.required_hashtag"]),
            "recommended": candidate(["watermark.full_video"]),
            "manual_review": candidate([]),
        },
    )
    result = resolve_contract(draft, registry=AssetRegistry(tmp_path))
    assert result.status is ResolutionStatus.RESOLVED
    assert result.contract is not None
    assert "watermark.full_video" in result.contract.rules.hard
    assert "watermark.full_video" not in result.contract.rules.recommended
    assert "rules.watermark.full_video" in _issue_fields(result, IssueCode.RULE_DEFAULTED)


def test_unclassified_watermark_reports_rule_defaulted(tmp_path: Path) -> None:
    draft = make_draft(
        watermark={
            "required": candidate(value=True),
            "visible_full_video": candidate(value=True),
        },
        rules=None,
    )
    result = resolve_contract(draft, registry=AssetRegistry(tmp_path))
    assert result.contract is not None
    assert "watermark.full_video" in result.contract.rules.hard
    assert "rules.watermark.full_video" in _issue_fields(result, IssueCode.RULE_DEFAULTED)


def test_required_assets_rule_is_forced_hard(tmp_path: Path) -> None:
    _ = (tmp_path / "clip.mp4").write_bytes(b"video")
    draft = make_draft(
        assets={"required": [make_asset_draft()], "optional": []},
        rules={
            "hard": candidate(["duration.min", "caption.required_hashtag"]),
            "recommended": candidate(["assets.required"]),
            "manual_review": candidate([]),
        },
    )
    result = resolve_contract(draft, registry=AssetRegistry(tmp_path))
    assert result.contract is not None
    assert "assets.required" in result.contract.rules.hard
    assert "assets.required" not in result.contract.rules.recommended
    assert "rules.assets.required" in _issue_fields(result, IssueCode.RULE_DEFAULTED)


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


@pytest.mark.parametrize("field", ["asset_id", "kind", "uri", "origin"])
def test_optional_asset_with_missing_field_is_dropped(tmp_path: Path, field: str) -> None:
    asset = make_asset_draft()
    _ = asset.pop(field)
    draft = make_draft(assets={"required": [], "optional": [asset]})
    result = resolve_contract(draft, registry=AssetRegistry(tmp_path))
    assert result.status is ResolutionStatus.RESOLVED
    assert result.contract is not None
    assert result.contract.assets.optional == []
    assert _issue_fields(result, IssueCode.OPTIONAL_ASSET_DROPPED) == ["assets.optional[0]"]


def test_required_asset_with_missing_field_blocks(tmp_path: Path) -> None:
    asset = make_asset_draft()
    _ = asset.pop("kind")
    draft = make_draft(assets={"required": [asset], "optional": []})
    result = resolve_contract(draft, registry=AssetRegistry(tmp_path))
    assert result.status is ResolutionStatus.MANUAL_REVIEW
    assert _issue_fields(result, IssueCode.MISSING_REQUIRED) == ["assets.required[0]"]


def test_optional_asset_with_empty_kind_is_dropped(tmp_path: Path) -> None:
    _ = (tmp_path / "clip.mp4").write_bytes(b"video")
    asset = make_asset_draft()
    asset["kind"] = candidate("")
    draft = make_draft(assets={"required": [], "optional": [asset]})
    result = resolve_contract(draft, registry=AssetRegistry(tmp_path))
    assert result.status is ResolutionStatus.RESOLVED
    assert result.contract is not None
    assert result.contract.assets.optional == []
    assert _issue_fields(result, IssueCode.OPTIONAL_ASSET_DROPPED) == ["assets.optional[0]"]


def test_required_asset_with_empty_origin_blocks_at_asset_field(tmp_path: Path) -> None:
    _ = (tmp_path / "clip.mp4").write_bytes(b"video")
    asset = make_asset_draft()
    asset["origin"] = candidate("")
    draft = make_draft(assets={"required": [asset], "optional": []})
    result = resolve_contract(draft, registry=AssetRegistry(tmp_path))
    assert result.status is ResolutionStatus.MANUAL_REVIEW
    assert _issue_fields(result, IssueCode.MISSING_REQUIRED) == ["assets.required[0]"]


def test_conflicted_attribution_type_reports_once(tmp_path: Path) -> None:
    draft = make_draft(
        platforms={
            "tiktok": {
                "duration": {"min_s": candidate(8)},
                "required_hashtags": candidate(["#marca"]),
                "required_mentions": candidate(["@marca"]),
                "attribution": {"type": conflict_candidate(), "value": candidate("@marca")},
            }
        },
        rules=None,
    )
    result = resolve_contract(draft, registry=AssetRegistry(tmp_path))
    assert result.status is ResolutionStatus.MANUAL_REVIEW
    type_issues = [
        issue for issue in result.issues if issue.field == "platforms.tiktok.attribution.type"
    ]
    assert len(type_issues) == 1
    assert type_issues[0].code is IssueCode.CONFLICT


def test_conflicted_optional_asset_uri_reports_once(tmp_path: Path) -> None:
    asset = make_asset_draft()
    asset["uri"] = conflict_candidate()
    draft = make_draft(assets={"required": [], "optional": [asset]})
    result = resolve_contract(draft, registry=AssetRegistry(tmp_path))
    assert result.status is ResolutionStatus.MANUAL_REVIEW
    uri_issues = [issue for issue in result.issues if issue.field == "assets.optional[0].uri"]
    assert len(uri_issues) == 1
    assert uri_issues[0].code is IssueCode.CONFLICT
    assert _issue_fields(result, IssueCode.MISSING_REQUIRED) == []


def test_unc_asset_uri_is_rejected_without_resolving(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry = AssetRegistry(tmp_path)

    def forbidden_resolve(_self_path: Path) -> Path:
        msg = "resolve() no debe ejecutarse sobre rutas no relativas"
        raise AssertionError(msg)

    monkeypatch.setattr(Path, "resolve", forbidden_resolve)
    draft = make_draft(
        assets={"required": [make_asset_draft(uri=r"\\10.255.255.1\share\x.mp4")], "optional": []}
    )
    result = resolve_contract(draft, registry=registry)
    assert result.status is ResolutionStatus.MANUAL_REVIEW
    assert _issue_fields(result, IssueCode.UNRESOLVED_ASSET) == ["assets.required[0]"]


@pytest.mark.skipif(os.name != "nt", reason="normalización de rutas de Windows")
def test_reuse_matches_windows_path_variant(tmp_path: Path) -> None:
    _ = (tmp_path / "clip.mp4").write_bytes(b"video")
    registry = AssetRegistry(tmp_path)
    first = registry.register(asset_id="clip-01", kind="video", uri="clip.mp4", origin="brief")
    draft = make_draft(assets={"required": [make_asset_draft(uri=".\\clip.mp4")], "optional": []})
    result = resolve_contract(draft, registry=registry)
    assert result.contract is not None
    assert result.contract.assets.required[0] == first


def test_resolution_is_idempotent_with_same_registry(tmp_path: Path) -> None:
    _ = (tmp_path / "clip.mp4").write_bytes(b"video")
    registry = AssetRegistry(tmp_path)
    draft = make_draft(assets={"required": [make_asset_draft()], "optional": []})
    first = resolve_contract(draft, registry=registry)
    second = resolve_contract(draft, registry=registry)
    assert first.status is ResolutionStatus.RESOLVED
    assert second.status is ResolutionStatus.RESOLVED
    assert first.contract == second.contract


def test_prohibitions_default_to_hard_and_gate_rejects(tmp_path: Path) -> None:
    draft = make_draft(prohibitions=candidate(["sorteo"]), rules=None)
    result = resolve_contract(draft, registry=AssetRegistry(tmp_path))
    assert result.contract is not None
    assert "caption.forbidden" in result.contract.rules.hard
    assert "rules.caption.forbidden" in _issue_fields(result, IssueCode.RULE_DEFAULTED)
    artifact = tmp_path / "piece.mp4"
    _ = artifact.write_bytes(b"video")
    gate = Gate(FakeProbe(info=make_media()))
    gate_result = gate.run(
        contract=result.contract,
        piece=make_piece(artifact, caption="gran SORTEO @marca #marca"),
        assets=AssetRegistry(tmp_path),
    )
    assert gate_result.status is GateStatus.REJECTED
    assert _check(gate_result, "caption.forbidden").status is CheckStatus.FAIL


def test_spelling_locks_default_to_hard_and_gate_rejects(tmp_path: Path) -> None:
    draft = make_draft(spelling_locks=candidate(["MarcaX"]), rules=None)
    result = resolve_contract(draft, registry=AssetRegistry(tmp_path))
    assert result.contract is not None
    assert "subtitles.spelling_lock" in result.contract.rules.hard
    artifact = tmp_path / "piece.mp4"
    _ = artifact.write_bytes(b"video")
    gate = Gate(FakeProbe(info=make_media()))
    gate_result = gate.run(
        contract=result.contract,
        piece=make_piece(artifact, subtitle_text="bienvenidos a maracax"),
        assets=AssetRegistry(tmp_path),
    )
    assert gate_result.status is GateStatus.REJECTED
    assert _check(gate_result, "subtitles.spelling_lock").status is CheckStatus.FAIL


def test_base_rule_cannot_be_downgraded_to_recommended(tmp_path: Path) -> None:
    draft = make_draft(
        rules={
            "hard": candidate(
                [
                    "duration.min",
                    "caption.required_hashtag",
                    "caption.required_mention",
                ]
            ),
            "recommended": candidate(["artifact.integrity"]),
            "manual_review": candidate([]),
        }
    )
    result = resolve_contract(draft, registry=AssetRegistry(tmp_path))
    assert result.contract is not None
    assert "artifact.integrity" in result.contract.rules.hard
    assert "artifact.integrity" not in result.contract.rules.recommended
    assert "rules.artifact.integrity" in _issue_fields(result, IssueCode.RULE_DEFAULTED)


def test_base_rule_cannot_be_downgraded_to_manual_review(tmp_path: Path) -> None:
    draft = make_draft(
        rules={
            "hard": candidate(
                [
                    "duration.min",
                    "caption.required_hashtag",
                    "caption.required_mention",
                ]
            ),
            "recommended": candidate([]),
            "manual_review": candidate(["artifact.integrity"]),
        }
    )
    result = resolve_contract(draft, registry=AssetRegistry(tmp_path))
    assert result.contract is not None
    assert "artifact.integrity" in result.contract.rules.hard
    assert "artifact.integrity" not in result.contract.rules.manual_review
    assert "rules.artifact.integrity" in _issue_fields(result, IssueCode.RULE_DEFAULTED)


def test_audio_rule_is_scoped_per_platform(tmp_path: Path) -> None:
    draft = make_draft(
        platforms={
            "tiktok": {
                "duration": {"min_s": candidate(8)},
                "required_hashtags": candidate(["#marca"]),
                "required_mentions": candidate(["@marca"]),
                "audio_rule": candidate("no_trending"),
            },
            "instagram_reels": {"duration": {"min_s": candidate(8)}},
        },
        rules=None,
    )
    result = resolve_contract(draft, registry=AssetRegistry(tmp_path))
    assert result.contract is not None
    assert "audio.present" in result.contract.rules.hard
    artifact = tmp_path / "piece.mp4"
    _ = artifact.write_bytes(b"video")
    gate = Gate(FakeProbe(info=make_media(has_audio=False)))
    gate_result = gate.run(
        contract=result.contract,
        piece=make_piece(artifact, caption="libre", platform=Platform.INSTAGRAM_REELS),
        assets=AssetRegistry(tmp_path),
    )
    assert gate_result.status is GateStatus.PENDING_REVIEW
    assert _check(gate_result, "audio.present").status is CheckStatus.PASS
    assert _check(gate_result, "audio.no_trending").status is CheckStatus.MANUAL_REVIEW


def test_nul_uri_on_registered_asset_is_rejected(tmp_path: Path) -> None:
    _ = (tmp_path / "clip.mp4").write_bytes(b"video")
    registry = AssetRegistry(tmp_path)
    _ = registry.register(asset_id="clip-01", kind="video", uri="clip.mp4", origin="brief")
    draft = make_draft(
        assets={"required": [make_asset_draft(uri="clip.mp4\x00evil")], "optional": []}
    )
    result = resolve_contract(draft, registry=registry)
    assert result.status is ResolutionStatus.MANUAL_REVIEW
    assert _issue_fields(result, IssueCode.UNRESOLVED_ASSET) == ["assets.required[0]"]


def test_partial_watermark_blocks_until_validator_exists(tmp_path: Path) -> None:
    draft = make_draft(
        watermark={
            "required": candidate(value=True),
            "visible_full_video": candidate(value=False),
        },
        rules=None,
    )
    result = resolve_contract(draft, registry=AssetRegistry(tmp_path))
    assert result.contract is not None
    assert "watermark.present" in result.contract.rules.hard
    assert "watermark.full_video" not in result.contract.rules.hard
    artifact = tmp_path / "piece.mp4"
    _ = artifact.write_bytes(b"video")
    gate = Gate(FakeProbe(info=make_media()))
    gate_result = gate.run(
        contract=result.contract,
        piece=make_piece(artifact, caption="mira @marca #marca"),
        assets=AssetRegistry(tmp_path),
    )
    assert gate_result.status is GateStatus.UNSUPPORTED
    assert _check(gate_result, "watermark.present").status is CheckStatus.UNSUPPORTED


def test_absent_assets_resolve_to_empty_bundle(tmp_path: Path) -> None:
    result = resolve_contract(make_draft(assets=None), registry=AssetRegistry(tmp_path))
    assert result.status is ResolutionStatus.RESOLVED
    assert result.contract is not None
    assert result.contract.assets.required == []
    assert result.contract.assets.optional == []
    assert result.issues == ()


@pytest.mark.parametrize(
    ("platform_override", "expected_field"),
    [
        (
            {"caption_rules": {"must_mention": conflict_candidate()}},
            "platforms.tiktok.caption_rules.must_mention",
        ),
        (
            {"caption_rules": {"first_line": conflict_candidate()}},
            "platforms.tiktok.caption_rules.first_line",
        ),
        (
            {"caption_rules": {"forbidden": conflict_candidate()}},
            "platforms.tiktok.caption_rules.forbidden",
        ),
        (
            {"attribution": {"type": conflict_candidate()}},
            "platforms.tiktok.attribution.type",
        ),
        (
            {"attribution": {"value": conflict_candidate()}},
            "platforms.tiktok.attribution.value",
        ),
        (
            {"link_rules": {"link_in_bio": conflict_candidate()}},
            "platforms.tiktok.link_rules.link_in_bio",
        ),
        ({"audio_rule": conflict_candidate()}, "platforms.tiktok.audio_rule"),
        ({"required_mentions": conflict_candidate()}, "platforms.tiktok.required_mentions"),
        ({"required_hashtags": conflict_candidate()}, "platforms.tiktok.required_hashtags"),
    ],
)
def test_nested_conflicts_block_at_exact_field(
    tmp_path: Path,
    platform_override: dict[str, object],
    expected_field: str,
) -> None:
    platform: dict[str, object] = {
        "duration": {"min_s": candidate(8)},
        "required_hashtags": candidate(["#marca"]),
        "required_mentions": candidate(["@marca"]),
        **platform_override,
    }
    draft = make_draft(platforms={"tiktok": platform}, rules=None)
    result = resolve_contract(draft, registry=AssetRegistry(tmp_path))
    assert result.status is ResolutionStatus.MANUAL_REVIEW
    assert expected_field in _issue_fields(result, IssueCode.CONFLICT)


def test_complete_geo_target_is_preserved(tmp_path: Path) -> None:
    draft = make_draft(geo_target={"country": candidate("MX"), "min_pct": candidate(60)})
    result = resolve_contract(draft, registry=AssetRegistry(tmp_path))
    assert result.contract is not None
    assert result.contract.geo_target is not None
    assert result.contract.geo_target.country == "MX"
    assert result.contract.geo_target.min_pct == 60


def test_attribution_type_without_value_blocks_at_value_field(tmp_path: Path) -> None:
    draft = make_draft(
        platforms={
            "tiktok": {
                "duration": {"min_s": candidate(8)},
                "required_hashtags": candidate(["#marca"]),
                "required_mentions": candidate(["@marca"]),
                "attribution": {"type": candidate("tag")},
            }
        },
        rules=None,
    )
    result = resolve_contract(draft, registry=AssetRegistry(tmp_path))
    assert result.status is ResolutionStatus.MANUAL_REVIEW
    assert _issue_fields(result, IssueCode.MISSING_REQUIRED) == [
        "platforms.tiktok.attribution.value"
    ]


def test_asset_license_is_preserved(tmp_path: Path) -> None:
    _ = (tmp_path / "clip.mp4").write_bytes(b"video")
    asset = make_asset_draft()
    asset["license"] = candidate("CC0")
    draft = make_draft(assets={"required": [asset], "optional": []})
    result = resolve_contract(draft, registry=AssetRegistry(tmp_path))
    assert result.contract is not None
    assert result.contract.assets.required[0].license == "CC0"


def test_resolvable_optional_asset_is_kept(tmp_path: Path) -> None:
    _ = (tmp_path / "opt.mp4").write_bytes(b"video")
    draft = make_draft(
        assets={"required": [], "optional": [make_asset_draft(asset_id="opt-01", uri="opt.mp4")]}
    )
    result = resolve_contract(draft, registry=AssetRegistry(tmp_path))
    assert result.contract is not None
    assert [ref.asset_id for ref in result.contract.assets.optional] == ["opt-01"]
    assert _issue_fields(result, IssueCode.OPTIONAL_ASSET_DROPPED) == []


def test_resolve_segments_valid_draft(tmp_path: Path) -> None:
    draft = make_draft(
        mode=candidate("long_video"),
        segments={"segments": [{"start_s": candidate(0.0), "end_s": candidate(8.5)}]},
    )
    result = resolve_contract(draft, registry=AssetRegistry(tmp_path))
    assert result.status is ResolutionStatus.RESOLVED
    assert result.contract is not None
    assert result.contract.segments == (Segment(start_s=0.0, end_s=8.5),)


def test_resolve_segments_missing_field(tmp_path: Path) -> None:
    draft = make_draft(
        mode=candidate("long_video"),
        segments={"segments": [{"end_s": candidate(8.5)}]},
    )
    result = resolve_contract(draft, registry=AssetRegistry(tmp_path))
    assert result.status is ResolutionStatus.MANUAL_REVIEW
    assert _issue_fields(result, IssueCode.MISSING_REQUIRED) == ["segments[0]"]


def test_resolve_segments_conflict(tmp_path: Path) -> None:
    draft = make_draft(
        mode=candidate("long_video"),
        segments={"segments": [{"start_s": conflict_candidate(), "end_s": candidate(8.5)}]},
    )
    result = resolve_contract(draft, registry=AssetRegistry(tmp_path))
    assert result.status is ResolutionStatus.MANUAL_REVIEW
    assert _issue_fields(result, IssueCode.CONFLICT) == ["segments[0].start_s"]


def test_resolve_segments_empty_for_non_long_video(tmp_path: Path) -> None:
    draft = make_draft(
        segments={"segments": [{"start_s": candidate(0.0), "end_s": candidate(8.5)}]},
    )
    result = resolve_contract(draft, registry=AssetRegistry(tmp_path))
    assert result.status is ResolutionStatus.RESOLVED
    assert result.contract is not None
    assert result.contract.segments == ()
    assert [issue for issue in result.issues if issue.field.startswith("segments")] == []


def _valid_provenance_draft(brief: str, **overrides: object) -> ContractDraft:
    def _cand(value: object, quote: str) -> dict[str, object]:
        start = brief.index(quote)
        end = start + len(quote)
        return {
            "value": value,
            "evidence": {"quote": quote, "start": start, "end": end, "location": "brief.md#l1"},
            "confidence": "explicit",
        }

    data: dict[str, object] = {
        "schema_version": "1.1",
        "campaign_id": _cand("camp-01", "camp-01"),
        "format": _cand("video", "video"),
        "mode": _cand("given_clips", "given_clips"),
        "platforms": {
            "tiktok": {
                "duration": {"min_s": _cand(8, "8s")},
                "required_hashtags": _cand(["#marca"], "#marca"),
                "required_mentions": _cand(["@marca"], "@marca"),
            }
        },
        "languages": {"source": _cand("es", "es"), "caption": _cand("es", "es")},
        "watermark": {
            "required": _cand(value=False, quote="no_watermark"),
            "visible_full_video": _cand(value=False, quote="no_watermark"),
        },
        "rules": {
            "hard": _cand(
                ["duration.min", "caption.required_hashtag", "caption.required_mention"],
                "hard_rules",
            ),
            "recommended": _cand([], "no_rec"),
            "manual_review": _cand([], "no_rev"),
        },
        "assets": {"required": [], "optional": []},
    }
    data.update(overrides)
    return ContractDraft.model_validate(data)


_BRIEF_TEXT = (
    "ID: camp-01 | format: video | mode: given_clips | min: 8s | "
    "tags: #marca | mentions: @marca | lang: es | wm: no_watermark | "
    "rules: hard_rules | no_rec | no_rev"
)


def test_resolve_contract_verifies_brief_provenance(tmp_path: Path) -> None:
    draft = _valid_provenance_draft(_BRIEF_TEXT)
    result = resolve_contract(draft, registry=AssetRegistry(tmp_path), brief_text=_BRIEF_TEXT)
    assert result.status is ResolutionStatus.RESOLVED
    assert result.contract is not None
    assert result.contract.campaign_id == "camp-01"


def test_resolve_contract_rejects_fabricated_quote(tmp_path: Path) -> None:
    draft = _valid_provenance_draft(
        _BRIEF_TEXT,
        campaign_id={
            "value": "camp-01",
            "evidence": {
                "quote": "campaña_fantasma_inventada",
                "start": 0,
                "end": len("campaña_fantasma_inventada"),
                "location": "l1",
            },
            "confidence": "explicit",
        },
    )
    with pytest.raises(ProvenanceError, match=r"fabricada|no existe"):
        _ = resolve_contract(draft, registry=AssetRegistry(tmp_path), brief_text=_BRIEF_TEXT)


def test_resolve_contract_rejects_misaligned_offset(tmp_path: Path) -> None:
    start = _BRIEF_TEXT.index("camp-01") + 2
    end = start + len("camp-01")
    draft = _valid_provenance_draft(
        _BRIEF_TEXT,
        campaign_id={
            "value": "camp-01",
            "evidence": {
                "quote": "camp-01",
                "start": start,
                "end": end,
                "location": "l1",
            },
            "confidence": "explicit",
        },
    )
    with pytest.raises(ProvenanceError, match=r"desalinead|no coincide|esperaba"):
        _ = resolve_contract(draft, registry=AssetRegistry(tmp_path), brief_text=_BRIEF_TEXT)


def test_resolve_contract_rejects_out_of_bounds_offset(tmp_path: Path) -> None:
    start = len(_BRIEF_TEXT) + 10
    end = start + len("camp-01")
    draft = _valid_provenance_draft(
        _BRIEF_TEXT,
        campaign_id={
            "value": "camp-01",
            "evidence": {
                "quote": "camp-01",
                "start": start,
                "end": end,
                "location": "l1",
            },
            "confidence": "explicit",
        },
    )
    with pytest.raises(ProvenanceError, match=r"rango|longitud"):
        _ = resolve_contract(draft, registry=AssetRegistry(tmp_path), brief_text=_BRIEF_TEXT)
