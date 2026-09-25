"""Tests unitarios de la selección de codificador de render."""

import pytest

from kliptych.encoding import (
    RenderConfig,
    audio_and_container_arguments,
    audio_injection_arguments,
    fallback_encoder_arguments,
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


def test_fallback_encoder_arguments_replaces_nvenc() -> None:
    nvenc_argv = [
        "ffmpeg",
        "-i",
        "input.mp4",
        "-c:v",
        "h264_nvenc",
        "-preset",
        "p5",
        "-cq",
        "23",
        "-pix_fmt",
        "yuv420p",
        "output.mp4",
    ]
    fallback = fallback_encoder_arguments(nvenc_argv)
    assert "h264_nvenc" not in fallback
    assert "libx264" in fallback
    assert "-preset" in fallback
    assert fallback[fallback.index("-preset") + 1] == "medium"
    assert "-crf" in fallback
    assert fallback[fallback.index("-crf") + 1] == "20"


def test_fallback_encoder_arguments_replaces_non_contiguous_nvenc() -> None:
    nvenc_argv = [
        "ffmpeg",
        "-i",
        "input.mp4",
        "-preset",
        "p5",
        "-c:v",
        "h264_nvenc",
        "-cq",
        "23",
        "output.mp4",
    ]
    fallback = fallback_encoder_arguments(nvenc_argv)
    assert "h264_nvenc" not in fallback
    assert "libx264" in fallback
    assert fallback[fallback.index("-preset") + 1] == "medium"
    assert fallback[fallback.index("-crf") + 1] == "20"


def test_fallback_encoder_arguments_noop_without_nvenc() -> None:
    cpu_argv = ["ffmpeg", "-i", "input.mp4", "-c:v", "libx264", "output.mp4"]
    assert fallback_encoder_arguments(cpu_argv) == cpu_argv


def test_audio_injection_arguments_with_duration() -> None:
    argv = audio_injection_arguments(mix_ratio=1.0, video_duration_s=6.0)
    assert "-shortest" not in argv
    assert "-t" in argv
    assert argv[argv.index("-t") + 1] == "6.000"

    argv_mix = audio_injection_arguments(mix_ratio=0.5, video_duration_s=4.25)
    assert "-shortest" not in argv_mix
    assert "-t" in argv_mix
    assert argv_mix[argv_mix.index("-t") + 1] == "4.250"


@pytest.mark.parametrize("duration", [0.0, -2.5])
def test_audio_injection_arguments_rejects_non_positive_duration(duration: float) -> None:
    with pytest.raises(ValueError, match="video_duration_s"):
        _ = audio_injection_arguments(mix_ratio=1.0, video_duration_s=duration)
