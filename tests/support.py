"""Fixtures compartidas de los tests del gate."""

import subprocess
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import pytest

from kliptych.contract import AssetRef, AudioPolicy, Contract, ContractDraft, Platform
from kliptych.gate import MediaInfo, Piece, ProbeError, SubtitleSegment

ALL_HARD_RULES = (
    "artifact.integrity",
    "audio.present",
    "caption.required_hashtag",
    "caption.required_mention",
    "duration.min",
)

_AUDIO_MANUAL_RULES = {
    "own_clip": "audio.own_clip",
    "no_trending": "audio.no_trending",
}


def make_asset_ref(
    asset_id: str = "clip-01",
    *,
    sha256: str = "a" * 64,
    size_bytes: int = 5,
) -> AssetRef:
    return AssetRef(
        asset_id=asset_id,
        kind="video",
        uri="clip.mp4",
        sha256=sha256,
        size_bytes=size_bytes,
        mime="video/mp4",
        origin="brief",
        license=None,
        resolved_at=datetime(2026, 9, 22, tzinfo=UTC),
    )


def _caption_rule_ids(
    *,
    first_line: str | None,
    forbidden: Sequence[str],
    prohibitions: Sequence[str],
    must_mention: Sequence[str],
    required_mentions: Sequence[str],
    required_hashtags: Sequence[str],
) -> list[str]:
    active: list[str] = []
    if first_line is not None:
        active.append("caption.first_line")
    if forbidden or prohibitions:
        active.append("caption.forbidden")
    if must_mention or required_mentions:
        active.append("caption.required_mention")
    if required_hashtags:
        active.append("caption.required_hashtag")
    return active


def _active_rule_ids(
    *,
    min_s: int | None,
    max_s: int | None,
    first_line: str | None,
    forbidden: Sequence[str],
    prohibitions: Sequence[str],
    must_mention: Sequence[str],
    required_mentions: Sequence[str],
    required_hashtags: Sequence[str],
    spelling_locks: Sequence[str],
    required_assets: Sequence[AssetRef],
    watermark_required: bool,
    watermark_visible_full_video: bool,
    hook_keyword: str | None,
) -> list[str]:
    active: list[str] = []
    if min_s is not None:
        active.append("duration.min")
    if max_s is not None:
        active.append("duration.max")
    active.extend(
        _caption_rule_ids(
            first_line=first_line,
            forbidden=forbidden,
            prohibitions=prohibitions,
            must_mention=must_mention,
            required_mentions=required_mentions,
            required_hashtags=required_hashtags,
        )
    )
    if spelling_locks:
        active.append("subtitles.spelling_lock")
    if required_assets:
        active.append("assets.required")
    watermark_rule = _watermark_rule_id(
        watermark_required=watermark_required,
        watermark_visible_full_video=watermark_visible_full_video,
    )
    if watermark_rule is not None:
        active.append(watermark_rule)
    if hook_keyword is not None:
        active.append("hook.keyword")
    return active


def _watermark_rule_id(
    *,
    watermark_required: bool,
    watermark_visible_full_video: bool,
) -> str | None:
    if not watermark_required:
        return None
    if watermark_visible_full_video:
        return "watermark.full_video"
    return "watermark.present"


def make_contract(
    *,
    hard: Sequence[str] = ALL_HARD_RULES,
    recommended: Sequence[str] = (),
    manual_review: Sequence[str] = (),
    min_s: int | None = 8,
    max_s: int | None = None,
    required_mentions: Sequence[str] = ("@marca",),
    required_hashtags: Sequence[str] = ("#marca",),
    must_mention: Sequence[str] = (),
    forbidden: Sequence[str] = (),
    prohibitions: Sequence[str] = (),
    first_line: str | None = None,
    spelling_locks: Sequence[str] = (),
    required_assets: Sequence[AssetRef] = (),
    watermark_required: bool = False,
    watermark_visible_full_video: bool = False,
    watermark_position: str = "top_right",
    watermark_scale_ratio: float = 0.20,
    watermark_opacity: float = 1.0,
    watermark_min_width_ratio: float = 0.05,
    audio_rule: str = "own_clip",
    language: str | None = None,
    audio_policy: AudioPolicy | None = None,
    hook_keyword: str | None = None,
    brand_safety_required: bool = False,
    brand_safety_citation: str | None = None,
    unmapped: Sequence[tuple[str, str]] = (),
) -> Contract:
    plan = [*hard]
    manual = [*manual_review]
    classified = {*hard, *recommended, *manual_review}
    if brand_safety_required and "brand.safety" not in classified:
        plan.append("brand.safety")
        classified.add("brand.safety")
    audio_rule_id = _AUDIO_MANUAL_RULES.get(audio_rule)
    if audio_rule_id is not None and audio_rule_id not in classified:
        manual.append(audio_rule_id)
        classified.add(audio_rule_id)
    if audio_policy is AudioPolicy.INTERNAL_OFFICIAL_SOUND and "audio.policy" not in classified:
        manual.append("audio.policy")
        classified.add("audio.policy")
    if audio_policy is AudioPolicy.INTERNAL_OFFICIAL_SOUND and "audio.silence" not in classified:
        plan.append("audio.silence")
        classified.add("audio.silence")
    for rule_id in _active_rule_ids(
        min_s=min_s,
        max_s=max_s,
        first_line=first_line,
        forbidden=forbidden,
        prohibitions=prohibitions,
        must_mention=must_mention,
        required_mentions=required_mentions,
        required_hashtags=required_hashtags,
        spelling_locks=spelling_locks,
        required_assets=required_assets,
        watermark_required=watermark_required,
        watermark_visible_full_video=watermark_visible_full_video,
        hook_keyword=hook_keyword,
    ):
        if rule_id not in classified:
            if rule_id == "audio.policy":
                manual.append(rule_id)
            else:
                plan.append(rule_id)
            classified.add(rule_id)
    return Contract.model_validate(
        {
            "schema_version": "1.1",
            "campaign_id": "camp-test",
            "format": "video",
            "mode": "given_clips",
            "platforms": {
                "tiktok": {
                    "duration": {"min_s": min_s, "max_s": max_s},
                    "caption_rules": {
                        "must_mention": list(must_mention),
                        "first_line": first_line,
                        "forbidden": list(forbidden),
                    },
                    "audio_rule": audio_rule,
                    "required_hashtags": list(required_hashtags),
                    "required_mentions": list(required_mentions),
                    "attribution": {"type": "none", "value": None},
                    "link_rules": {"link_in_bio": False},
                }
            },
            "languages": {
                "source": "es",
                "subtitles": None,
                "caption": "es",
                "voice": None,
                "language": language,
            },
            "official_audio": None,
            "watermark": {
                "required": watermark_required,
                "asset_id": "wm-marca" if watermark_required else None,
                "visible_full_video": watermark_visible_full_video,
                "position": watermark_position,
                "scale_ratio": watermark_scale_ratio,
                "opacity": watermark_opacity,
                "min_width_ratio": watermark_min_width_ratio,
            },
            "spelling_locks": list(spelling_locks),
            "prohibitions": list(prohibitions),
            "audio_policy": audio_policy,
            "hook_keyword": hook_keyword,
            "brand_safety_required": brand_safety_required,
            "brand_safety_citation": brand_safety_citation,
            "unmapped": [{"rule": rule, "quote": quote} for rule, quote in unmapped],
            "rules": {
                "hard": plan,
                "recommended": list(recommended),
                "manual_review": manual,
            },
            "assets": {"required": list(required_assets), "optional": []},
            "geo_target": None,
            "min_views_for_payout": {"value": None, "enforcement": "post_publication_manual"},
            "analytics_proof_required": {"value": False, "enforcement": "post_publication_manual"},
        }
    )


def make_piece(
    artifact: Path,
    *,
    caption: str = "mira @marca #marca",
    hashtags: Sequence[str] = (),
    subtitle_text: str | None = None,
    subtitle_segments: Sequence[SubtitleSegment] = (),
    screen_text_segments: Sequence[SubtitleSegment] = (),
    platform: Platform = Platform.TIKTOK,
) -> Piece:
    return Piece(
        piece_id="piece-01",
        platform=platform,
        caption=caption,
        hashtags=tuple(hashtags),
        subtitle_text=subtitle_text,
        subtitle_segments=tuple(subtitle_segments),
        screen_text_segments=tuple(screen_text_segments),
        artifact_path=artifact,
    )


def make_media(
    *,
    duration_s: float | None = 12.0,
    has_video: bool = True,
    has_audio: bool = True,
) -> MediaInfo:
    return MediaInfo(
        format_name="mov,mp4,m4a",
        duration_s=duration_s,
        has_video=has_video,
        has_audio=has_audio,
        width=1080,
        height=1920,
    )


@dataclass
class FakeProbe:
    info: MediaInfo | None = None
    error: ProbeError | None = None
    probed: list[Path] = field(default_factory=list)

    def probe(self, path: Path) -> MediaInfo:
        self.probed.append(path)
        if self.error is not None:
            raise self.error
        assert self.info is not None
        return self.info


def candidate(value: object, quote: str = "cita del brief") -> dict[str, object]:
    return {
        "value": value,
        "evidence": {"quote": quote, "start": 0, "end": len(quote), "location": "brief.md#l1"},
        "confidence": "explicit",
    }


def conflict_candidate(quote: str = "el brief se contradice") -> dict[str, object]:
    return {
        "evidence": {"quote": quote, "start": 0, "end": len(quote), "location": "brief.md#l2"},
        "confidence": "conflict",
    }


def make_draft(**overrides: object) -> ContractDraft:
    data: dict[str, object] = {
        "schema_version": "1.1",
        "campaign_id": candidate("camp-01"),
        "format": candidate("video"),
        "mode": candidate("given_clips"),
        "platforms": {
            "tiktok": {
                "duration": {"min_s": candidate(8)},
                "required_hashtags": candidate(["#marca"]),
                "required_mentions": candidate(["@marca"]),
            }
        },
        "languages": {"source": candidate("es"), "caption": candidate("es")},
        "watermark": {
            "required": candidate(value=False),
            "visible_full_video": candidate(value=False),
        },
        "rules": {
            "hard": candidate(
                [
                    "duration.min",
                    "caption.required_hashtag",
                    "caption.required_mention",
                ]
            ),
            "recommended": candidate([]),
            "manual_review": candidate([]),
        },
        "assets": {"required": [], "optional": []},
    }
    data.update(overrides)
    return ContractDraft.model_validate(data)


def make_asset_draft(
    *,
    asset_id: str = "clip-01",
    uri: str = "clip.mp4",
) -> dict[str, object]:
    return {
        "asset_id": candidate(asset_id),
        "kind": candidate("video"),
        "uri": candidate(uri),
        "origin": candidate("brief"),
    }


def write_fixture_clip(root: Path, *, content: bytes = b"clip") -> Path:
    """Materializa el clip declarado por la fixture given_clips bajo ``root``.

    Args:
        root: Raíz del workspace/registry donde resolver la uri del contrato.
        content: Bytes del clip; en tests unitarios no se decodifica.

    Returns:
        La ruta del clip escrito.
    """
    path = root / "assets" / "samples" / "given-clips-sample.mp4"
    path.parent.mkdir(parents=True, exist_ok=True)
    _ = path.write_bytes(content)
    return path


SILENT_VOLUMEDETECT_STDERR = (
    "[Parsed_volumedetect_0 @ 0x7fab] n: 48000 | mean_volume: -91.0 dB | max_volume: -91.0 dB"
)


def mock_silent_volumedetect(monkeypatch: pytest.MonkeyPatch) -> None:
    """Simula un render correctamente silenciado para el gate, sin ffmpeg real.

    Los tests unitarios del gate usan artefactos falsos (bytes, no MP4): sin
    este mock, ``audio.silence`` fallaría en cerrado sobre ellos. La medición
    real con ffmpeg vive en ``tests/test_audio_silence.py``.

    Args:
        monkeypatch: Fixture de pytest para parchear ``subprocess.run``.
    """

    def fake_run(argv: Sequence[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        _ = kwargs
        assert isinstance(argv, list)
        if "-select_streams" in argv:
            return subprocess.CompletedProcess(args=argv, returncode=0, stdout="0\n", stderr="")
        return subprocess.CompletedProcess(
            args=argv, returncode=0, stdout="", stderr=SILENT_VOLUMEDETECT_STDERR
        )

    monkeypatch.setattr("kliptych.gate.checks.subprocess.run", fake_run)
