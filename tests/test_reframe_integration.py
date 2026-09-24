"""Integración de reframe 9:16 con MediaPipe real (CPU) y ffmpeg.

El fixture es un video sintético generado localmente con ffmpeg. El detector usa
la Tasks API de MediaPipe en CPU; el modelo corto de BlazeFace se busca en
``KLIPTYCH_FACE_MODEL`` o en una caché local y, si no está, se descarga una vez
(igual que el modelo de faster-whisper en la integración de transcripción). El
módulo se salta entero si MediaPipe no está instalado y está excluido del gate
de integración de CI, donde la extra ``reframe`` no se instala.
"""

import importlib.util
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, cast
from urllib.request import urlopen

import pytest

from kliptych.encoding import RenderConfig
from kliptych.gate.probe import FFprobeProbe
from kliptych.reframe import (
    FFmpegFrameSource,
    FFmpegReframer,
    MediaPipeFaceDetector,
    ReframeResult,
)

if TYPE_CHECKING:
    from collections.abc import Callable


class _HttpResponse(Protocol):
    def read(self) -> bytes: ...


_OPENER = cast("Callable[..., _HttpResponse]", urlopen)


_HAS_MEDIAPIPE = importlib.util.find_spec("mediapipe") is not None
_FFMPEG = shutil.which("ffmpeg")

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not _HAS_MEDIAPIPE, reason="mediapipe no instalado"),
    pytest.mark.skipif(_FFMPEG is None, reason="ffmpeg no instalado"),
]

_FIXTURE_TIMEOUT_S = 120
_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_detector/"
    "blaze_face_short_range/float16/1/blaze_face_short_range.tflite"
)


def _face_model() -> Path:
    override = os.environ.get("KLIPTYCH_FACE_MODEL")
    if override is not None and Path(override).is_file():
        return Path(override)
    cache = Path(tempfile.gettempdir()) / "kliptych-face-models" / "blaze_face_short_range.tflite"
    if cache.is_file():
        return cache
    cache.parent.mkdir(parents=True, exist_ok=True)
    try:
        data = _OPENER(_MODEL_URL, timeout=60).read()
    except OSError as error:
        pytest.skip(f"no se pudo obtener el modelo de caras: {error}")
    _ = cache.write_bytes(data)
    return cache


def _nvenc_available() -> bool:
    assert _FFMPEG is not None
    completed = subprocess.run(
        [_FFMPEG, "-hide_banner", "-encoders"],
        capture_output=True,
        text=True,
        timeout=_FIXTURE_TIMEOUT_S,
        check=False,
    )
    return completed.returncode == 0 and "h264_nvenc" in completed.stdout


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
        "testsrc=size=320x240:rate=10:duration=2",
        "-f",
        "lavfi",
        "-i",
        "sine=frequency=440:duration=2",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-shortest",
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


def test_real_frame_source_yields_frames(tmp_path: Path) -> None:
    video = _generate_fixture(tmp_path / "fixture.mp4")
    samples = list(FFmpegFrameSource().frames(video, sample_fps=2.0))
    assert samples
    assert samples[0].frame.width == 320
    assert samples[0].frame.height == 240
    assert len(samples[0].frame.data) == 320 * 240 * 3
    assert samples[0].timestamp_s == pytest.approx(0.0)


def test_real_mediapipe_reframe_produces_vertical_video(tmp_path: Path) -> None:
    video = _generate_fixture(tmp_path / "fixture.mp4")
    detector = MediaPipeFaceDetector(model_path=_face_model())
    reframer = FFmpegReframer(
        detector=detector,
        render=RenderConfig(nvenc_available=_nvenc_available()),
    )
    try:
        result = reframer.analyze(video)
        assert isinstance(result, ReframeResult)
        assert result.source_width == 320
        assert result.source_height == 240
        assert result.targets
        destination = tmp_path / "reframed.mp4"
        _ = reframer.reframe(video=video, destination=destination)
    finally:
        detector.close()
    assert destination.is_file()
    media = FFprobeProbe().probe(destination)
    assert media.has_video
    assert media.width is not None
    assert media.height is not None
    assert media.width / media.height == pytest.approx(9 / 16, abs=0.01)
