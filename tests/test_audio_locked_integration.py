"""Integración del modo audio obligatorio con ffmpeg real.

El vídeo y la pista de audio son fixtures sintéticos generados localmente (cero
red); la descarga se mockea copiando esos fixtures. ffmpeg real corta el
segmento, hace el reframe 9:16, inyecta la pista de audio y quema los
subtítulos. El módulo se salta entero si falta ffmpeg o ffprobe.
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
_FFPROBE = shutil.which("ffprobe")

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        _FFMPEG is None or _FFPROBE is None,
        reason="ffmpeg/ffprobe no instalados",
    ),
]

_FIXTURE_TIMEOUT_S = 120
_SEGMENT_END_S = 1.5
_SOURCE_DURATION_S = 2.0
_VIDEO_URL = "https://example.com/video"
_AUDIO_URL = "https://cdn.example.com/track.mp3"


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
                    "audio_rule": "official_required",
                    "required_hashtags": [],
                    "required_mentions": [],
                    "attribution": {"type": "none", "value": None},
                    "link_rules": {"link_in_bio": False},
                }
            },
            "languages": {"source": "es", "subtitles": None, "caption": "es", "voice": None},
            "official_audio": {"tiktok_url": _AUDIO_URL, "instagram_url": None},
            "watermark": {"required": False, "asset_id": None, "visible_full_video": False},
            "spelling_locks": [],
            "prohibitions": [],
            "rules": {
                "hard": ["duration.min", "duration.max"],
                "recommended": [],
                "manual_review": ["audio.official_track"],
            },
            "assets": {"required": [], "optional": []},
            "segments": [{"start_s": 0.0, "end_s": _SEGMENT_END_S}],
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


def _install_fixture_downloader(
    monkeypatch: pytest.MonkeyPatch, *, video: Path, audio: Path
) -> None:
    class _Downloader:
        def __init__(self, *, timeout_s: float, max_size_bytes: int) -> None:
            _ = (timeout_s, max_size_bytes)
            self._video: Path = video
            self._audio: Path = audio

        def download_video(
            self, *, url: str, destination: Path, format_selector: str | None = None
        ) -> Path:
            _ = format_selector
            source = self._audio if url == _AUDIO_URL else self._video
            destination.parent.mkdir(parents=True, exist_ok=True)
            _ = shutil.copyfile(source, destination)
            return destination

    monkeypatch.setattr("kliptych.orchestrator.MediaDownloader", _Downloader)


def _generate_video(path: Path) -> Path:
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
        f"testsrc=size=320x240:rate=10:duration={_SOURCE_DURATION_S:g}",
        "-f",
        "lavfi",
        "-i",
        f"sine=frequency=440:duration={_SOURCE_DURATION_S:g}",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-shortest",
        str(path),
    ]
    _run_fixture(argv)
    return path


def _generate_audio(path: Path) -> Path:
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
        f"sine=frequency=880:duration={_SOURCE_DURATION_S:g}",
        "-c:a",
        "pcm_s16le",
        str(path),
    ]
    _run_fixture(argv)
    return path


def _run_fixture(argv: list[str]) -> None:
    completed = subprocess.run(
        argv,
        capture_output=True,
        text=True,
        timeout=_FIXTURE_TIMEOUT_S,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr


def _audio_duration_s(path: Path) -> float:
    assert _FFPROBE is not None
    argv = [
        _FFPROBE,
        "-v",
        "error",
        "-select_streams",
        "a:0",
        "-show_entries",
        "stream=duration",
        "-of",
        "default=nw=1:nk=1",
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
    return float(completed.stdout.strip())


def _run_pipeline(config: PipelineConfig) -> PipelineResult:
    render = config.render
    return run_long_video(
        _VIDEO_URL,
        model=_StubModel(
            {"segments": [{"start_s": 0.0, "end_s": _SEGMENT_END_S}], "rationale": "mejor tramo"}
        ),
        config=config,
        detector=FFmpegMomentDetector(),
        transcriber=_StubTranscriber(_transcript()),
        selector=LLMSegmentSelector(),
        reframer=FFmpegReframer(detector=_StaticFaceDetector(), render=render),
        subtitle_renderer=SubtitleRenderer(render=render),
    )


def _config(
    tmp_path: Path,
    *,
    mix_ratio: float,
    track: Path | None = None,
    url: str | None = None,
) -> PipelineConfig:
    return PipelineConfig(
        output_dir=tmp_path / "out",
        contract=_contract(),
        render=RenderConfig(),
        audio_locked=True,
        audio_track_path=track,
        audio_track_url=url,
        audio_mix_ratio=mix_ratio,
    )


def test_long_video_locked_audio_replaces_track(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    video = _generate_video(tmp_path / "fixture.mp4")
    track = _generate_audio(tmp_path / "track.wav")
    _install_fixture_downloader(monkeypatch, video=video, audio=track)
    config = _config(tmp_path, mix_ratio=1.0, track=track)
    result = _run_pipeline(config)
    assert result.final_video.is_file()
    media = FFprobeProbe().probe(result.final_video)
    assert media.has_video
    assert media.has_audio
    assert media.duration_s is not None
    assert media.duration_s / _SEGMENT_END_S == pytest.approx(1.0, abs=0.05)
    assert _audio_duration_s(result.final_video) == pytest.approx(_SEGMENT_END_S, abs=0.2)
    assert track.is_file()
    leftover = [path.name for path in (tmp_path / "out").iterdir() if ".part-" in path.name]
    assert leftover == []


def test_long_video_locked_audio_mixes_downloaded_track(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    video = _generate_video(tmp_path / "fixture.mp4")
    track = _generate_audio(tmp_path / "track.wav")
    _install_fixture_downloader(monkeypatch, video=video, audio=track)
    config = _config(tmp_path, mix_ratio=0.5, url=_AUDIO_URL)
    result = _run_pipeline(config)
    assert result.final_video.is_file()
    media = FFprobeProbe().probe(result.final_video)
    assert media.has_video
    assert media.has_audio
    audio_duration = _audio_duration_s(result.final_video)
    assert 0.0 < audio_duration < _SOURCE_DURATION_S
    assert abs(audio_duration - _SEGMENT_END_S) < abs(audio_duration - _SOURCE_DURATION_S)
    assert any("audio_track" in path for path in result.cleaning)
    leftover = [path.name for path in (tmp_path / "out").iterdir() if ".part-" in path.name]
    assert leftover == []
