"""Hal C2: las combinaciones contradictorias de audio se rechazan (fail-closed).

`internal_official_sound` exige silenciar el render final (el sonido oficial
se añade al publicar): inyectar una pista externa (`audio_locked` o
`audio_track_path`/`audio_track_url`) y luego silenciarla es contradictorio
y debe fallar con error explícito, nunca degradarse en silencio.
"""

from pathlib import Path
from typing import cast

import pytest
from pydantic import ValidationError

from kliptych import orchestrator
from kliptych.contract import AudioPolicy, Contract
from kliptych.download import MediaDownloader
from kliptych.encoding import RenderConfig
from kliptych.orchestrator import (
    LongVideoModel,
    PipelineConfig,
    PipelineError,
    run_long_video,
    run_slideshow,
)
from tests.support import make_contract

_URL = "https://example.com/video"


def _muted_contract() -> Contract:
    return make_contract(audio_rule="any", audio_policy=AudioPolicy.INTERNAL_OFFICIAL_SOUND)


def _contradictory_config(tmp_path: Path, **overrides: object) -> PipelineConfig:
    track = tmp_path / "track.mp3"
    _ = track.write_bytes(b"audio")
    params: dict[str, object] = {
        "output_dir": tmp_path / "out",
        "contract": _muted_contract(),
        "render": RenderConfig(),
        "audio_locked": True,
        "audio_track_path": track,
    }
    params.update(overrides)
    return PipelineConfig(**params)  # type: ignore[arg-type]


def _model() -> LongVideoModel:
    return cast("LongVideoModel", object())


def test_contract_rejects_audio_locked_mode_with_internal_policy() -> None:
    data = _muted_contract().model_dump(mode="json")
    data["mode"] = "audio_locked"
    with pytest.raises(ValidationError, match="internal_official_sound"):
        _ = Contract.model_validate(data)


def test_run_long_video_rejects_external_audio_with_internal_policy(tmp_path: Path) -> None:
    config = _contradictory_config(tmp_path)
    with pytest.raises(PipelineError, match="internal_official_sound"):
        _ = run_long_video(_URL, model=_model(), config=config)


def test_run_long_video_rejects_track_without_flag_with_internal_policy(
    tmp_path: Path,
) -> None:
    config = _contradictory_config(tmp_path, audio_locked=False)
    with pytest.raises(PipelineError, match="internal_official_sound"):
        _ = run_long_video(_URL, model=_model(), config=config)


def test_run_slideshow_rejects_external_audio_with_internal_policy(tmp_path: Path) -> None:
    image = tmp_path / "slide.jpg"
    _ = image.write_bytes(b"imagen")
    config = _contradictory_config(tmp_path)
    with pytest.raises(PipelineError, match="internal_official_sound"):
        _ = run_slideshow([image], config=config)


def test_inject_audio_rejects_internal_policy(tmp_path: Path) -> None:
    inject_audio = getattr(orchestrator, "_inject_audio")
    cleanup_registry = getattr(orchestrator, "_CleanupRegistry")
    config = _contradictory_config(tmp_path)
    with pytest.raises(PipelineError, match="internal_official_sound"):
        _ = inject_audio(
            tmp_path / "clip.mp4",
            config=config,
            downloader=MediaDownloader(),
            registry=cleanup_registry(),
        )


def test_audible_audio_locked_combination_still_builds(tmp_path: Path) -> None:
    track = tmp_path / "track.mp3"
    _ = track.write_bytes(b"audio")
    config = PipelineConfig(
        output_dir=tmp_path / "out",
        contract=make_contract(audio_rule="any"),
        render=RenderConfig(),
        audio_locked=True,
        audio_track_path=track,
    )
    assert config.audio_locked is True
