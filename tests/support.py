"""Fixtures compartidas de los tests del gate."""

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from kliptych.contract import AssetRef, Contract, Platform
from kliptych.gate import MediaInfo, Piece, ProbeError

ALL_HARD_RULES = (
    "artifact.integrity",
    "audio.present",
    "caption.required_hashtag",
    "caption.required_mention",
    "duration.min",
)


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
) -> list[str]:
    active: list[str] = []
    if min_s is not None:
        active.append("duration.min")
    if max_s is not None:
        active.append("duration.max")
    if first_line is not None:
        active.append("caption.first_line")
    if forbidden or prohibitions:
        active.append("caption.forbidden")
    if must_mention or required_mentions:
        active.append("caption.required_mention")
    if required_hashtags:
        active.append("caption.required_hashtag")
    if spelling_locks:
        active.append("subtitles.spelling_lock")
    if required_assets:
        active.append("assets.required")
    if watermark_required and watermark_visible_full_video:
        active.append("watermark.full_video")
    return active


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
) -> Contract:
    plan = [*hard]
    classified = {*hard, *recommended, *manual_review}
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
    ):
        if rule_id not in classified:
            plan.append(rule_id)
            classified.add(rule_id)
    return Contract.model_validate(
        {
            "schema_version": "1.0",
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
                    "audio_rule": "own_clip",
                    "required_hashtags": list(required_hashtags),
                    "required_mentions": list(required_mentions),
                    "attribution": {"type": "none", "value": None},
                    "link_rules": {"link_in_bio": False},
                }
            },
            "languages": {"source": "es", "subtitles": None, "caption": "es", "voice": None},
            "official_audio": None,
            "watermark": {
                "required": watermark_required,
                "asset_id": "wm-marca" if watermark_required else None,
                "visible_full_video": watermark_visible_full_video,
            },
            "spelling_locks": list(spelling_locks),
            "prohibitions": list(prohibitions),
            "rules": {
                "hard": plan,
                "recommended": list(recommended),
                "manual_review": list(manual_review),
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
    platform: Platform = Platform.TIKTOK,
) -> Piece:
    return Piece(
        piece_id="piece-01",
        platform=platform,
        caption=caption,
        hashtags=tuple(hashtags),
        subtitle_text=subtitle_text,
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
