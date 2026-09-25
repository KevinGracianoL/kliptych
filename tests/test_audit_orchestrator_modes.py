"""Tests de los entry points de modo run_audio_locked y run_repost.

Verifican que cada modo exige su flag en la configuración (fail-closed) y que
delega en el pipeline long_video con la misma config.
"""

from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

from kliptych import orchestrator as orchestrator_module
from kliptych.contract import Contract
from kliptych.encoding import RenderConfig
from kliptych.orchestrator import (
    LongVideoModel,
    PipelineConfig,
    PipelineError,
    run_audio_locked,
    run_repost,
)

_URL = "https://example.com/video"


def _contract() -> Contract:
    return Contract.model_validate(
        {
            "schema_version": "1.1",
            "campaign_id": "camp-modes",
            "format": "video",
            "mode": "long_video",
            "platforms": {
                "tiktok": {
                    "duration": {"min_s": 1, "max_s": 10},
                    "caption_rules": {"must_mention": [], "first_line": None, "forbidden": []},
                    "audio_rule": "any",
                    "required_hashtags": [],
                    "required_mentions": [],
                    "attribution": {"type": "none", "value": None},
                    "link_rules": {"link_in_bio": False},
                }
            },
            "languages": {"source": "es", "subtitles": None, "caption": "es", "voice": None},
            "official_audio": None,
            "watermark": {"required": False, "asset_id": None, "visible_full_video": False},
            "spelling_locks": [],
            "prohibitions": [],
            "rules": {
                "hard": ["duration.min", "duration.max"],
                "recommended": [],
                "manual_review": [],
            },
            "assets": {"required": [], "optional": []},
            "segments": [{"start_s": 0.0, "end_s": 1.0}],
            "geo_target": None,
            "min_views_for_payout": {"value": None, "enforcement": "post_publication_manual"},
            "analytics_proof_required": {"value": False, "enforcement": "post_publication_manual"},
        }
    )


def _config(
    tmp_path: Path,
    *,
    audio_locked: bool = False,
    audio_track_path: Path | None = None,
    repost_mode: bool = False,
) -> PipelineConfig:
    return PipelineConfig(
        output_dir=tmp_path / "out",
        contract=_contract(),
        render=RenderConfig(),
        audio_locked=audio_locked,
        audio_track_path=audio_track_path,
        repost_mode=repost_mode,
    )


def test_mode_wrappers_reject_mismatched_config(tmp_path: Path) -> None:
    """run_audio_locked/run_repost exigen su flag en la config (fail-closed)."""
    model = cast("LongVideoModel", SimpleNamespace())
    with pytest.raises(PipelineError, match="audio_locked"):
        _ = run_audio_locked(_URL, model=model, config=_config(tmp_path))
    with pytest.raises(PipelineError, match="repost"):
        _ = run_repost(_URL, model=model, config=_config(tmp_path))


def test_run_audio_locked_delegates_to_long_video(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """run_audio_locked delega en run_long_video con la misma config."""
    seen: dict[str, object] = {}
    sentinel = SimpleNamespace()

    def _fake_run_long_video(url: str, **kwargs: object) -> object:
        _ = url
        seen.update(kwargs)
        return sentinel

    monkeypatch.setattr(orchestrator_module, "run_long_video", _fake_run_long_video)
    track = tmp_path / "track.mp3"
    _ = track.write_bytes(b"audio")
    config = _config(tmp_path, audio_locked=True, audio_track_path=track)
    model = cast("LongVideoModel", SimpleNamespace())
    result = run_audio_locked(_URL, model=model, config=config, resume=True)
    assert result is sentinel
    assert seen["config"] is config
    assert seen["resume"] is True


def test_run_repost_delegates_to_long_video(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """run_repost delega en run_long_video con la misma config."""
    seen: dict[str, object] = {}
    sentinel = SimpleNamespace()

    def _fake_run_long_video(url: str, **kwargs: object) -> object:
        _ = url
        seen.update(kwargs)
        return sentinel

    monkeypatch.setattr(orchestrator_module, "run_long_video", _fake_run_long_video)
    config = _config(tmp_path, repost_mode=True)
    model = cast("LongVideoModel", SimpleNamespace())
    result = run_repost(_URL, model=model, config=config)
    assert result is sentinel
    assert seen["config"] is config
    assert seen["resume"] is False
