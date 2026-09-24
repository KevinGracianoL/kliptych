"""Integración del modo Repost/UGC con ffmpeg real (cero red).

Los vídeos sintéticos (vertical 9:16 y horizontal 16:9) y la pista de audio se
generan localmente; la descarga se mockea copiando esos fixtures. Se verifica
que el modo repost omite transcripción y selección LLM (contadores en cero),
que hace passthrough cuando el vídeo ya es 9:16, que reframea cuando no lo es y
que la inyección de audio convive con repost. El módulo se salta entero si falta
ffmpeg o ffprobe.
"""

import shutil
import subprocess
from collections.abc import Mapping
from pathlib import Path

import pytest

from kliptych.contract import Contract
from kliptych.encoding import RenderConfig
from kliptych.gate.probe import FFprobeProbe
from kliptych.orchestrator import PipelineConfig, PipelineResult, run_long_video
from kliptych.reframe import FaceBox, FFmpegReframer, RgbFrame
from kliptych.transcribe import Transcript

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
_SOURCE_DURATION_S = 2.0
_VIDEO_URL = "https://example.com/repost"
_VERTICAL_WIDTH = 360
_VERTICAL_HEIGHT = 640
_HORIZONTAL_WIDTH = 640
_HORIZONTAL_HEIGHT = 360


def _contract() -> Contract:
    return Contract.model_validate(
        {
            "schema_version": "1.1",
            "campaign_id": "camp-repost",
            "format": "video",
            "mode": "repost_ugc",
            "platforms": {
                "tiktok": {
                    "duration": {"min_s": None, "max_s": None},
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
            "rules": {"hard": [], "recommended": [], "manual_review": []},
            "assets": {"required": [], "optional": []},
            "segments": [],
            "geo_target": None,
            "min_views_for_payout": {"value": None, "enforcement": "post_publication_manual"},
            "analytics_proof_required": {"value": False, "enforcement": "post_publication_manual"},
        }
    )


class _Counts:
    def __init__(self) -> None:
        self.transcribe: int = 0
        self.select: int = 0


class _CountingTranscriber:
    def __init__(self, counts: _Counts) -> None:
        self._counts: _Counts = counts

    def transcribe(self, audio: Path) -> Transcript:
        _ = audio
        self._counts.transcribe += 1
        message = "la transcripción no debe ejecutarse en modo repost"
        raise AssertionError(message)


class _CountingModel:
    def __init__(self, counts: _Counts) -> None:
        self._counts: _Counts = counts

    def select_segments(self, prompt: Mapping[str, object]) -> object:
        _ = prompt
        self._counts.select += 1
        message = "el modelo LLM no debe ejecutarse en modo repost"
        raise AssertionError(message)


class _StaticFaceDetector:
    def __init__(self, confidence: float = 0.9) -> None:
        self._confidence: float = confidence

    def detect(self, frame: RgbFrame) -> tuple[FaceBox, ...]:
        _ = frame
        return (FaceBox(x=0.3, y=0.3, width=0.4, height=0.4, confidence=self._confidence),)


def _install_fixture_downloader(monkeypatch: pytest.MonkeyPatch, fixture: Path) -> None:
    class _Downloader:
        def __init__(self, *, timeout_s: float, max_size_bytes: int) -> None:
            _ = (timeout_s, max_size_bytes)
            self._fixture: Path = fixture

        def download_video(
            self, *, url: str, destination: Path, format_selector: str | None = None
        ) -> Path:
            _ = (url, format_selector)
            destination.parent.mkdir(parents=True, exist_ok=True)
            _ = shutil.copyfile(self._fixture, destination)
            return destination

    monkeypatch.setattr("kliptych.orchestrator.MediaDownloader", _Downloader)


def _generate_video(path: Path, *, width: int, height: int) -> Path:
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
        f"testsrc=size={width}x{height}:rate=10:duration={_SOURCE_DURATION_S:g}",
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


def _config(
    tmp_path: Path,
    *,
    audio_locked: bool = False,
    audio_track_path: Path | None = None,
) -> PipelineConfig:
    return PipelineConfig(
        output_dir=tmp_path / "out",
        contract=_contract(),
        render=RenderConfig(),
        repost_mode=True,
        audio_locked=audio_locked,
        audio_track_path=audio_track_path,
    )


def _run(
    config: PipelineConfig,
    counts: _Counts,
    *,
    reframer: FFmpegReframer | None = None,
) -> PipelineResult:
    return run_long_video(
        _VIDEO_URL,
        model=_CountingModel(counts),
        config=config,
        transcriber=_CountingTranscriber(counts),
        reframer=reframer,
    )


def _leftovers(output_dir: Path) -> list[str]:
    return [path.name for path in output_dir.iterdir() if ".part-" in path.name]


def test_repost_vertical_passthrough_skips_ai(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _generate_video(
        tmp_path / "vertical.mp4", width=_VERTICAL_WIDTH, height=_VERTICAL_HEIGHT
    )
    _install_fixture_downloader(monkeypatch, fixture)
    counts = _Counts()
    result = _run(_config(tmp_path), counts)
    assert counts.transcribe == 0
    assert counts.select == 0
    assert result.transcript is None
    assert result.moments == ()
    assert result.reframe is None
    assert result.subtitles is None
    assert result.final_video.is_file()
    media = FFprobeProbe().probe(result.final_video)
    assert media.has_video
    assert media.width == _VERTICAL_WIDTH
    assert media.height == _VERTICAL_HEIGHT
    assert _leftovers(tmp_path / "out") == []


def test_repost_horizontal_triggers_reframe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _generate_video(
        tmp_path / "horizontal.mp4", width=_HORIZONTAL_WIDTH, height=_HORIZONTAL_HEIGHT
    )
    _install_fixture_downloader(monkeypatch, fixture)
    counts = _Counts()
    render = RenderConfig()
    result = _run(
        _config(tmp_path),
        counts,
        reframer=FFmpegReframer(detector=_StaticFaceDetector(), render=render),
    )
    assert counts.transcribe == 0
    assert counts.select == 0
    assert result.transcript is None
    assert result.reframe is not None
    assert result.final_video.is_file()
    media = FFprobeProbe().probe(result.final_video)
    assert media.width is not None
    assert media.height is not None
    assert media.width / media.height == pytest.approx(9 / 16, abs=0.02)
    assert _leftovers(tmp_path / "out") == []


def test_repost_with_audio_locked_injects_track(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _generate_video(
        tmp_path / "vertical.mp4", width=_VERTICAL_WIDTH, height=_VERTICAL_HEIGHT
    )
    track = _generate_audio(tmp_path / "track.wav")
    _install_fixture_downloader(monkeypatch, fixture)
    counts = _Counts()
    config = _config(tmp_path, audio_locked=True, audio_track_path=track)
    result = _run(config, counts)
    assert counts.transcribe == 0
    assert counts.select == 0
    assert result.transcript is None
    assert result.reframe is None
    assert result.final_video.is_file()
    media = FFprobeProbe().probe(result.final_video)
    assert media.has_video
    assert media.has_audio
    assert media.width == _VERTICAL_WIDTH
    assert media.height == _VERTICAL_HEIGHT
    assert _audio_duration_s(result.final_video) == pytest.approx(_SOURCE_DURATION_S, abs=0.2)
    assert track.is_file()
    assert _leftovers(tmp_path / "out") == []
