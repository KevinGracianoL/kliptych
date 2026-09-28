"""Fixtures compartidas de los tests del gate."""

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from kliptych.contract import AssetRef, Contract, ContractDraft, Platform
from kliptych.gate import MediaInfo, Piece, ProbeError

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
    watermark_rule = _watermark_rule_id(
        watermark_required=watermark_required,
        watermark_visible_full_video=watermark_visible_full_video,
    )
    if watermark_rule is not None:
        active.append(watermark_rule)
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
    audio_rule: str = "own_clip",
    language: str | None = None,
) -> Contract:
    plan = [*hard]
    manual = [*manual_review]
    classified = {*hard, *recommended, *manual_review}
    audio_rule_id = _AUDIO_MANUAL_RULES.get(audio_rule)
    if audio_rule_id is not None and audio_rule_id not in classified:
        manual.append(audio_rule_id)
        classified.add(audio_rule_id)
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
            },
            "spelling_locks": list(spelling_locks),
            "prohibitions": list(prohibitions),
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
