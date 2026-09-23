"""Tests de los validadores deterministas del gate."""

import json
from datetime import UTC, datetime
from pathlib import Path

from kliptych.assets import AssetRegistry
from kliptych.contract import Contract
from kliptych.gate import (
    CheckResult,
    CheckStatus,
    Gate,
    GateResult,
    GateStatus,
    MediaInfo,
    Piece,
)
from tests.support import FakeProbe, make_asset_ref, make_contract, make_media, make_piece


def _artifact(tmp_path: Path, content: bytes = b"video") -> Path:
    path = tmp_path / "piece.mp4"
    _ = path.write_bytes(content)
    return path


def _result(
    tmp_path: Path,
    contract: Contract,
    piece: Piece,
    *,
    media: MediaInfo | None = None,
    registry: AssetRegistry | None = None,
) -> GateResult:
    gate = Gate(FakeProbe(info=make_media() if media is None else media))
    return gate.run(
        contract=contract,
        piece=piece,
        assets=AssetRegistry(tmp_path) if registry is None else registry,
    )


def _check(result: GateResult, rule_id: str) -> CheckResult:
    matches = [check for check in result.checks if check.id == rule_id]
    assert len(matches) == 1
    return matches[0]


def test_mention_from_must_mention_is_enforced(tmp_path: Path) -> None:
    contract = make_contract(required_mentions=(), must_mention=["@extra"])
    result = _result(tmp_path, contract, make_piece(_artifact(tmp_path)))
    check = _check(result, "caption.required_mention")
    assert check.status is CheckStatus.FAIL
    assert check.evidence["missing"] == ["@extra"]


def test_all_mentions_present_pass(tmp_path: Path) -> None:
    contract = make_contract(required_mentions=["@marca", "@extra"])
    piece = make_piece(_artifact(tmp_path), caption="gracias @marca y @extra #marca")
    assert _check(_result(tmp_path, contract, piece), "caption.required_mention").status is (
        CheckStatus.PASS
    )


def test_required_hashtag_can_live_in_hashtags_field(tmp_path: Path) -> None:
    contract = make_contract(required_hashtags=["#nueva"])
    piece = make_piece(_artifact(tmp_path), caption="texto sin etiquetas", hashtags=["#Nueva"])
    assert _check(_result(tmp_path, contract, piece), "caption.required_hashtag").status is (
        CheckStatus.PASS
    )


def test_missing_hashtag_fails_with_evidence(tmp_path: Path) -> None:
    contract = make_contract(required_hashtags=["#nueva"])
    piece = make_piece(_artifact(tmp_path), caption="solo @marca #marca")
    check = _check(_result(tmp_path, contract, piece), "caption.required_hashtag")
    assert check.status is CheckStatus.FAIL
    assert check.evidence["missing"] == ["#nueva"]


def test_forbidden_term_is_case_insensitive(tmp_path: Path) -> None:
    contract = make_contract(hard=["caption.forbidden"], forbidden=["sorteo"])
    piece = make_piece(_artifact(tmp_path), caption="gran SORTEO @marca #marca")
    check = _check(_result(tmp_path, contract, piece), "caption.forbidden")
    assert check.status is CheckStatus.FAIL
    assert check.evidence["found"] == ["sorteo"]


def test_forbidden_term_absent_passes(tmp_path: Path) -> None:
    contract = make_contract(hard=["caption.forbidden"], forbidden=["sorteo"])
    piece = make_piece(_artifact(tmp_path))
    assert _check(_result(tmp_path, contract, piece), "caption.forbidden").status is (
        CheckStatus.PASS
    )


def test_first_line_must_open_the_caption(tmp_path: Path) -> None:
    contract = make_contract(hard=["caption.first_line"], first_line="Escribe al 555-1234")
    good = make_piece(_artifact(tmp_path), caption="Escribe al 555-1234 y gana @marca #marca")
    assert _check(_result(tmp_path, contract, good), "caption.first_line").status is (
        CheckStatus.PASS
    )
    bad = make_piece(_artifact(tmp_path), caption="Gana ya\nEscribe al 555-1234 @marca #marca")
    check = _check(_result(tmp_path, contract, bad), "caption.first_line")
    assert check.status is CheckStatus.FAIL
    assert check.evidence["first_line"] == "Gana ya"


def test_first_line_without_requirement_passes(tmp_path: Path) -> None:
    contract = make_contract(hard=["caption.first_line"], first_line=None)
    result = _result(tmp_path, contract, make_piece(_artifact(tmp_path)))
    assert _check(result, "caption.first_line").status is CheckStatus.PASS


def test_spelling_lock_checked_in_subtitles(tmp_path: Path) -> None:
    contract = make_contract(hard=["subtitles.spelling_lock"], spelling_locks=["MarcaX"])
    bad = make_piece(_artifact(tmp_path), subtitle_text="bienvenidos a maracax")
    check = _check(_result(tmp_path, contract, bad), "subtitles.spelling_lock")
    assert check.status is CheckStatus.FAIL
    assert check.evidence["missing"] == ["MarcaX"]

    good = make_piece(_artifact(tmp_path), subtitle_text="bienvenidos a MarcaX")
    assert _check(_result(tmp_path, contract, good), "subtitles.spelling_lock").status is (
        CheckStatus.PASS
    )


def test_spelling_lock_without_subtitles_is_unsupported(tmp_path: Path) -> None:
    contract = make_contract(hard=["subtitles.spelling_lock"], spelling_locks=["MarcaX"])
    piece = make_piece(_artifact(tmp_path), subtitle_text=None)
    check = _check(_result(tmp_path, contract, piece), "subtitles.spelling_lock")
    assert check.status is CheckStatus.UNSUPPORTED
    assert "subtítulos" in str(check.evidence["reason"])


def test_spelling_lock_without_declared_locks_passes(tmp_path: Path) -> None:
    contract = make_contract(hard=["subtitles.spelling_lock"], spelling_locks=[])
    piece = make_piece(_artifact(tmp_path), subtitle_text=None)
    check = _check(_result(tmp_path, contract, piece), "subtitles.spelling_lock")
    assert check.status is CheckStatus.PASS


def test_duration_bounds_are_enforced(tmp_path: Path) -> None:
    contract = make_contract(hard=["duration.min", "duration.max"], min_s=8, max_s=20)
    short = _result(
        tmp_path, contract, make_piece(_artifact(tmp_path)), media=make_media(duration_s=5)
    )
    assert _check(short, "duration.min").status is CheckStatus.FAIL
    assert _check(short, "duration.max").status is CheckStatus.PASS

    long = _result(
        tmp_path, contract, make_piece(_artifact(tmp_path)), media=make_media(duration_s=25)
    )
    assert _check(long, "duration.min").status is CheckStatus.PASS
    assert _check(long, "duration.max").status is CheckStatus.FAIL

    inside = _result(
        tmp_path, contract, make_piece(_artifact(tmp_path)), media=make_media(duration_s=12)
    )
    assert _check(inside, "duration.min").status is CheckStatus.PASS
    assert _check(inside, "duration.max").status is CheckStatus.PASS


def test_duration_without_bounds_passes(tmp_path: Path) -> None:
    contract = make_contract(hard=["duration.min", "duration.max"], min_s=None, max_s=None)
    result = _result(tmp_path, contract, make_piece(_artifact(tmp_path)))
    assert _check(result, "duration.min").status is CheckStatus.PASS
    assert _check(result, "duration.max").status is CheckStatus.PASS


def test_unknown_duration_is_unsupported(tmp_path: Path) -> None:
    contract = make_contract(hard=["duration.min", "duration.max"], min_s=8, max_s=20)
    media = make_media(duration_s=None)
    result = _result(tmp_path, contract, make_piece(_artifact(tmp_path)), media=media)
    assert _check(result, "duration.min").status is CheckStatus.UNSUPPORTED
    assert _check(result, "duration.max").status is CheckStatus.UNSUPPORTED
    assert result.status is GateStatus.UNSUPPORTED


def test_missing_audio_track_fails(tmp_path: Path) -> None:
    contract = make_contract(hard=["audio.present"])
    media = make_media(has_audio=False)
    result = _result(tmp_path, contract, make_piece(_artifact(tmp_path)), media=media)
    assert _check(result, "audio.present").status is CheckStatus.FAIL


def test_audio_not_required_when_platform_rule_is_any(tmp_path: Path) -> None:
    contract = make_contract(hard=["audio.present"], audio_rule="any")
    media = make_media(has_audio=False)
    result = _result(tmp_path, contract, make_piece(_artifact(tmp_path)), media=media)
    check = _check(result, "audio.present")
    assert check.status is CheckStatus.PASS
    assert check.evidence == {"audio_rule": "any"}


def test_video_stream_required_for_video_format(tmp_path: Path) -> None:
    contract = make_contract(hard=["artifact.video_stream"])
    media = make_media(has_video=False)
    result = _result(tmp_path, contract, make_piece(_artifact(tmp_path)), media=media)
    assert _check(result, "artifact.video_stream").status is CheckStatus.FAIL


def test_video_stream_present_passes(tmp_path: Path) -> None:
    contract = make_contract(hard=["artifact.video_stream"])
    result = _result(tmp_path, contract, make_piece(_artifact(tmp_path)))
    check = _check(result, "artifact.video_stream")
    assert check.status is CheckStatus.PASS
    assert check.evidence == {"width": 1080, "height": 1920}


def test_required_asset_must_be_registered(tmp_path: Path) -> None:
    contract = make_contract(hard=["assets.required"], required_assets=[make_asset_ref("clip-01")])
    result = _result(tmp_path, contract, make_piece(_artifact(tmp_path)))
    check = _check(result, "assets.required")
    assert check.status is CheckStatus.FAIL
    assert check.evidence["missing"] == ["clip-01"]


def test_registered_asset_passes_and_tampering_fails(tmp_path: Path) -> None:
    clip = tmp_path / "clip.mp4"
    _ = clip.write_bytes(b"video")
    registry = AssetRegistry(tmp_path)
    ref = registry.register(asset_id="clip-01", kind="video", uri="clip.mp4", origin="brief")
    contract = make_contract(hard=["assets.required"], required_assets=[ref])
    piece = make_piece(_artifact(tmp_path))
    assert _check(
        _result(tmp_path, contract, piece, registry=registry), "assets.required"
    ).status is (CheckStatus.PASS)

    _ = clip.write_bytes(b"manipulado")
    check = _check(_result(tmp_path, contract, piece, registry=registry), "assets.required")
    assert check.status is CheckStatus.FAIL
    assert check.evidence["tampered"] == ["clip-01"]


def test_contract_asset_hash_mismatch_fails(tmp_path: Path) -> None:
    clip = tmp_path / "clip.mp4"
    _ = clip.write_bytes(b"video")
    registry = AssetRegistry(tmp_path)
    _ = registry.register(asset_id="clip-01", kind="video", uri="clip.mp4", origin="brief")
    wrong = make_asset_ref("clip-01", sha256="a" * 64, size_bytes=999)
    contract = make_contract(hard=["assets.required"], required_assets=[wrong])
    result = _result(tmp_path, contract, make_piece(_artifact(tmp_path)), registry=registry)
    check = _check(result, "assets.required")
    assert check.status is CheckStatus.FAIL
    assert check.evidence["mismatched"] == ["clip-01"]


def test_unsafe_registry_uri_fails_without_exception(tmp_path: Path) -> None:
    registry_file = tmp_path / "registry.json"
    payload = {
        "schema_version": "1.0",
        "assets": [
            {
                "asset_id": "malo",
                "kind": "file",
                "uri": "../fuera.txt",
                "sha256": "a" * 64,
                "size_bytes": 1,
                "mime": "text/plain",
                "origin": "brief",
                "license": None,
                "resolved_at": datetime(2026, 9, 22, tzinfo=UTC).isoformat(),
            }
        ],
    }
    _ = registry_file.write_text(json.dumps(payload), encoding="utf-8")
    registry = AssetRegistry.load(registry_file, tmp_path / "workspace")
    contract = make_contract(hard=["assets.required"], required_assets=[make_asset_ref("malo")])
    result = _result(tmp_path, contract, make_piece(_artifact(tmp_path)), registry=registry)
    check = _check(result, "assets.required")
    assert check.status is CheckStatus.FAIL
    assert check.evidence["unsafe"]
    assert result.status is GateStatus.REJECTED


def test_campaign_prohibitions_are_enforced(tmp_path: Path) -> None:
    contract = make_contract(hard=["caption.forbidden"], prohibitions=["sorteo"])
    piece = make_piece(_artifact(tmp_path), caption="gran SORTEO @marca #marca")
    check = _check(_result(tmp_path, contract, piece), "caption.forbidden")
    assert check.status is CheckStatus.FAIL
    assert check.evidence["found"] == ["sorteo"]


def test_mention_prefix_is_not_a_match(tmp_path: Path) -> None:
    contract = make_contract()
    piece = make_piece(_artifact(tmp_path), caption="gracias @marcado #marca")
    check = _check(_result(tmp_path, contract, piece), "caption.required_mention")
    assert check.status is CheckStatus.FAIL
    assert check.evidence["missing"] == ["@marca"]


def test_mention_inside_email_is_not_a_match(tmp_path: Path) -> None:
    contract = make_contract()
    piece = make_piece(_artifact(tmp_path), caption="correo@marca.com y #marca")
    check = _check(_result(tmp_path, contract, piece), "caption.required_mention")
    assert check.status is CheckStatus.FAIL
    assert check.evidence["missing"] == ["@marca"]


def test_mention_match_is_case_insensitive(tmp_path: Path) -> None:
    contract = make_contract(required_mentions=["@Marca"])
    piece = make_piece(_artifact(tmp_path), caption="hola @marca #marca")
    assert _check(_result(tmp_path, contract, piece), "caption.required_mention").status is (
        CheckStatus.PASS
    )


def test_hashtag_prefix_is_not_a_match(tmp_path: Path) -> None:
    contract = make_contract(required_hashtags=["#marca"])
    piece = make_piece(_artifact(tmp_path), caption="vamos #marcado @marca")
    check = _check(_result(tmp_path, contract, piece), "caption.required_hashtag")
    assert check.status is CheckStatus.FAIL
    assert check.evidence["missing"] == ["#marca"]


def test_no_required_assets_passes(tmp_path: Path) -> None:
    contract = make_contract(hard=["assets.required"])
    result = _result(tmp_path, contract, make_piece(_artifact(tmp_path)))
    assert _check(result, "assets.required").status is CheckStatus.PASS


def test_partial_watermark_is_unsupported(tmp_path: Path) -> None:
    contract = make_contract(watermark_required=True, watermark_visible_full_video=False)
    result = _result(tmp_path, contract, make_piece(_artifact(tmp_path)))
    assert _check(result, "watermark.present").status is CheckStatus.UNSUPPORTED
    assert result.status is GateStatus.UNSUPPORTED


def test_full_video_watermark_is_unsupported(tmp_path: Path) -> None:
    contract = make_contract(watermark_required=True, watermark_visible_full_video=True)
    result = _result(tmp_path, contract, make_piece(_artifact(tmp_path)))
    assert _check(result, "watermark.full_video").status is CheckStatus.UNSUPPORTED
    assert result.status is GateStatus.UNSUPPORTED
