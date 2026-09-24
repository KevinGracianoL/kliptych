"""Tests unitarios de la selección de codificador de render."""

import pytest

from kliptych.encoding import (
    RenderConfig,
    audio_and_container_arguments,
    video_encoder_arguments,
)


def test_video_encoder_prefers_nvenc() -> None:
    argv = video_encoder_arguments(nvenc_available=True)
    assert argv[:2] == ("-c:v", "h264_nvenc")
    assert "libx264" not in argv


def test_video_encoder_falls_back_to_libx264() -> None:
    argv = video_encoder_arguments(nvenc_available=False)
    assert argv[:2] == ("-c:v", "libx264")
    assert "h264_nvenc" not in argv


def test_audio_and_container_arguments() -> None:
    argv = audio_and_container_arguments()
    assert "aac" in argv
    assert "+faststart" in argv


def test_render_config_defaults() -> None:
    config = RenderConfig()
    assert config.ffmpeg == "ffmpeg"
    assert config.nvenc_available is False
    assert config.timeout_s > 0


@pytest.mark.parametrize("timeout_s", [0.0, -1.0])
def test_render_config_rejects_non_positive_timeout(timeout_s: float) -> None:
    with pytest.raises(ValueError, match="timeout"):
        _ = RenderConfig(timeout_s=timeout_s)
