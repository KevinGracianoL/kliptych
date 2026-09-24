"""Integración de detección de momentos con ffmpeg real.

El fixture es un vídeo sintético de tres escenas con un tramo central más
energético; se genera localmente con ffmpeg y nunca se descarga material de la
red. El módulo se salta entero si ffmpeg no está instalado.
"""

import shutil
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING, cast

import pytest

from kliptych.moments import ChatMessage, FFmpegMomentDetector, MomentSource
from kliptych.transcribe import Transcript, Word

if TYPE_CHECKING:
    from collections.abc import Callable

_FFMPEG = shutil.which("ffmpeg")

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(_FFMPEG is None, reason="ffmpeg no instalado"),
]

_FIXTURE_TIMEOUT_S = 120


def _method(detector: FFmpegMomentDetector, name: str) -> object:
    return cast("object", getattr(detector, name))


def _scene_events(detector: FFmpegMomentDetector, video: Path) -> tuple[tuple[float, float], ...]:
    method = cast(
        "Callable[[Path], tuple[tuple[float, float], ...]]",
        _method(detector, "_scene_events"),
    )
    return method(video)


def _energy_frames(detector: FFmpegMomentDetector, video: Path) -> tuple[tuple[float, float], ...]:
    method = cast(
        "Callable[[Path], tuple[tuple[float, float], ...]]",
        _method(detector, "_energy_frames"),
    )
    return method(video)


def _generate_fixture(path: Path) -> Path:
    assert _FFMPEG is not None
    argv = [
        _FFMPEG,
        "-hide_banner",
        "-v",
        "error",
        "-y",
        "-f",
        "lavfi",
        "-i",
        "color=c=red:s=160x120:r=10:d=1",
        "-f",
        "lavfi",
        "-i",
        "sine=frequency=440:duration=1",
        "-f",
        "lavfi",
        "-i",
        "color=c=blue:s=160x120:r=10:d=1",
        "-f",
        "lavfi",
        "-i",
        "sine=frequency=880:duration=1",
        "-f",
        "lavfi",
        "-i",
        "color=c=green:s=160x120:r=10:d=1",
        "-f",
        "lavfi",
        "-i",
        "sine=frequency=220:duration=1",
        "-filter_complex",
        (
            "[1:a]volume=0.05[a0];[3:a]volume=1.0[a1];[5:a]volume=0.05[a2];"
            "[0:v][a0][2:v][a1][4:v][a2]concat=n=3:v=1:a=1[v][a]"
        ),
        "-map",
        "[v]",
        "-map",
        "[a]",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        str(path),
    ]
    completed = subprocess.run(
        argv,
        capture_output=True,
        text=True,
        timeout=_FIXTURE_TIMEOUT_S,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    return path


def _transcript() -> Transcript:
    word = Word(start_s=0.0, end_s=1.0, text="hola", confidence=0.9, token_id=0)
    return Transcript(words=(word,), language="es", duration_s=3.0, text="hola")


def test_real_scene_detection_finds_cuts(tmp_path: Path) -> None:
    video = _generate_fixture(tmp_path / "fixture.mp4")
    events = _scene_events(FFmpegMomentDetector(), video)
    times = sorted(time_s for time_s, _ in events)
    assert any(abs(time_s - 1.0) < 0.2 for time_s in times)
    assert any(abs(time_s - 2.0) < 0.2 for time_s in times)


def test_real_audio_energy_detects_loud_window(tmp_path: Path) -> None:
    video = _generate_fixture(tmp_path / "fixture.mp4")
    frames = _energy_frames(FFmpegMomentDetector(), video)
    assert frames
    energies = [value for _, value in frames]
    loud = max(energies)
    quiet = min(energies)
    assert loud > quiet


def test_real_detect_returns_ranked_fused_moments(tmp_path: Path) -> None:
    video = _generate_fixture(tmp_path / "fixture.mp4")
    chat = tuple(
        ChatMessage(timestamp_s=1.0 + index * 0.05, text=f"mensaje {index}") for index in range(20)
    )
    moments = FFmpegMomentDetector().detect(video, transcript=_transcript(), chat=chat)
    assert moments
    assert all(moment.source is MomentSource.FUSED for moment in moments)
    assert all(0.0 <= moment.score <= 1.0 for moment in moments)
    assert all(moment.end_s > moment.start_s for moment in moments)
    assert all(moment.end_s <= 3.05 for moment in moments)
    assert [moment.score for moment in moments] == sorted(
        (moment.score for moment in moments), reverse=True
    )
