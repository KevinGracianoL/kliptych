"""Integración end-to-end del orquestador long_video con ffmpeg real.

El fixture es un vídeo sintético generado localmente (cero red). Se mockean la
descarga, la transcripción, el modelo LLM y MediaPipe; ffmpeg real ejecuta el
corte del segmento, el reframe 9:16 y el quemado de subtítulos. El módulo se
salta entero si falta ffmpeg.
"""

import shutil
import subprocess
from collections.abc import Mapping
from pathlib import Path

import pytest

from kliptych.contract import Contract
from kliptych.encoding import RenderConfig
from kliptych.gate.probe import FFprobeProbe
from kliptych.moments import FFmpegMomentDetector
from kliptych.orchestrator import PipelineConfig, PipelineResult, run_long_video
from kliptych.reframe import FaceBox, FFmpegReframer, RgbFrame
from kliptych.segment import LLMSegmentSelector
from kliptych.subtitles import SubtitleRenderer
from kliptych.transcribe import Transcript, Word

_FFMPEG = shutil.which("ffmpeg")

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(_FFMPEG is None, reason="ffmpeg no instalado"),
]

_FIXTURE_TIMEOUT_S = 120


def _contract() -> Contract:
    return Contract.model_validate(
        {
            "schema_version": "1.1",
            "campaign_id": "camp-01",
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
            "segments": [{"start_s": 0.0, "end_s": 1.5}],
            "geo_target": None,
            "min_views_for_payout": {"value": None, "enforcement": "post_publication_manual"},
            "analytics_proof_required": {"value": False, "enforcement": "post_publication_manual"},
        }
    )


def _transcript() -> Transcript:
    words = (
        Word(start_s=0.0, end_s=0.5, text="hola", confidence=0.9, token_id=0),
        Word(start_s=0.5, end_s=1.0, text="mundo", confidence=0.9, token_id=1),
        Word(start_s=1.0, end_s=1.5, text="vertical", confidence=0.9, token_id=2),
    )
    return Transcript(words=words, language="es", duration_s=2.0, text="hola mundo vertical")


class _StubTranscriber:
    def __init__(self, transcript: Transcript) -> None:
        self._transcript: Transcript = transcript

    def transcribe(self, audio: Path) -> Transcript:
        _ = audio
        return self._transcript


class _StaticFaceDetector:
    def __init__(self, confidence: float = 0.9) -> None:
        self._confidence: float = confidence

    def detect(self, frame: RgbFrame) -> tuple[FaceBox, ...]:
        _ = frame
        return (FaceBox(x=0.3, y=0.3, width=0.4, height=0.4, confidence=self._confidence),)


class _StubModel:
    def __init__(self, selection: dict[str, object]) -> None:
        self._selection: dict[str, object] = selection

    def select_segments(self, prompt: Mapping[str, object]) -> object:
        _ = prompt
        return self._selection


def _install_fixture_downloader(monkeypatch: pytest.MonkeyPatch, fixture: Path) -> None:
    class _Downloader:
        def __init__(self, *, timeout_s: float, max_size_bytes: int) -> None:
            _ = (timeout_s, max_size_bytes)
            self._fixture: Path = fixture

        def download_video(
            self, *, url: str, destination: Path, format_selector: str | None = None
        ) -> Path:
            _ = (url, format_selector)
            _ = shutil.copyfile(self._fixture, destination)
            return destination

    monkeypatch.setattr("kliptych.orchestrator.MediaDownloader", _Downloader)


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


def test_run_long_video_end_to_end(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fixture = _generate_fixture(tmp_path / "fixture.mp4")
    _install_fixture_downloader(monkeypatch, fixture)
    output_dir = tmp_path / "out"
    render = RenderConfig()
    config = PipelineConfig(output_dir=output_dir, contract=_contract(), render=render)
    result = run_long_video(
        "https://example.com/video",
        model=_StubModel(
            {"segments": [{"start_s": 0.0, "end_s": 1.5}], "rationale": "mejor tramo"}
        ),
        config=config,
        detector=FFmpegMomentDetector(),
        transcriber=_StubTranscriber(_transcript()),
        selector=LLMSegmentSelector(),
        reframer=FFmpegReframer(detector=_StaticFaceDetector(), render=render),
        subtitle_renderer=SubtitleRenderer(render=render),
    )
    assert isinstance(result, PipelineResult)
    assert result.source.is_file()
    assert result.final_video.is_file()
    media = FFprobeProbe().probe(result.final_video)
    assert media.has_video
    assert media.has_audio
    assert media.width is not None
    assert media.height is not None
    assert media.width / media.height == pytest.approx(9 / 16, abs=0.02)
    leftovers = [path.name for path in output_dir.iterdir() if ".part-" in path.name]
    assert leftovers == []
    assert result.subtitles is not None
    assert not result.subtitles.exists()
    assert result.cleaning
