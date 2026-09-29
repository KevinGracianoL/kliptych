"""Tests E2E de coordenadas para marcas temporales mandatorias y cortes quirúrgicos (T1 + T2 + T4).

Verifica que el sistema de coordenadas sea coherente entre descarga quirúrgica,
selección forzada y corte de segmentos, que múltiples rangos generen clips válidos,
y que cortes fuera de duración fallen de forma controlada sin producir MP4s vacíos de 262 bytes.
"""

import math
import shutil
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import cast, final, override

import pytest

import kliptych.orchestrator as orch
from kliptych.contract import TimestampRange
from kliptych.download import DownloadError, SubprocessDownloadRunner
from kliptych.encoding import RenderConfig
from kliptych.moments import ChatMessage, Moment, MomentDetector
from kliptych.orchestrator import (
    LongVideoModel,
    PipelineConfig,
    PipelineError,
    Reframer,
    SubtitleBurner,
    run_long_video,
)
from kliptych.reframe import ReframeResult
from kliptych.segment import LLMSegmentSelector
from kliptych.transcribe import Transcriber, Transcript, Word
from tests.support import make_contract


def _attr(obj: object, name: str) -> object:
    return cast("object", getattr(obj, name))


def _probe(path: Path) -> float:
    fn = cast("Callable[..., object]", _attr(orch, "_probe_video"))
    info = fn(path, render=RenderConfig())
    return cast("float", _attr(info, "duration_s"))


def _generate_synthetic_video(path: Path, *, duration_s: int) -> Path:
    argv = [
        "ffmpeg",
        "-hide_banner",
        "-nostdin",
        "-v",
        "error",
        "-y",
        "-f",
        "lavfi",
        "-i",
        f"testsrc=duration={duration_s}:size=320x240:rate=1",
        "-f",
        "lavfi",
        "-i",
        f"sine=frequency=1000:duration={duration_s}",
        "-c:v",
        "libx264",
        "-c:a",
        "aac",
        str(path),
    ]
    res = SubprocessDownloadRunner.run(argv, timeout_s=30.0)
    if not res.ok:
        pytest.skip(f"ffmpeg no disponible o falló: {res.stderr}")
    return path


@final
class _FakeSectionDownloader:
    _source_video: Path

    def __init__(self, source_video: Path) -> None:
        self._source_video = source_video

    def has(self, tool: str) -> bool:
        _ = tool
        return bool(self._source_video)

    def download_video(
        self,
        *,
        url: str,
        destination: Path,
        format_selector: str | None = None,
        section: tuple[float, float] | None = None,
    ) -> Path:
        _ = (url, format_selector)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if section is None:
            _ = shutil.copyfile(self._source_video, destination)
            return destination

        start_sec, end_sec = section
        margin_start = max(0.0, start_sec - 10.0)
        margin_end = end_sec + 10.0
        duration = margin_end - margin_start

        argv = [
            "ffmpeg",
            "-hide_banner",
            "-nostdin",
            "-v",
            "error",
            "-y",
            "-ss",
            f"{margin_start:.3f}",
            "-t",
            f"{duration:.3f}",
            "-i",
            str(self._source_video),
            "-c:v",
            "libx264",
            "-c:a",
            "aac",
            str(destination),
        ]
        res = SubprocessDownloadRunner.run(argv, timeout_s=30.0)
        if not res.ok:
            msg = f"Fallo al descargar sección {section}: {res.stderr}"
            raise DownloadError(msg)
        return destination


class _FakeTranscriber(Transcriber):
    @override
    def transcribe(self, audio: Path) -> Transcript:
        dur = _probe(audio)
        words = (Word(start_s=0.0, end_s=min(1.0, dur), text="test", confidence=0.9, token_id=0),)
        return Transcript(language="es", duration_s=dur, text="test", words=words)


class _FakeDetector(MomentDetector):
    @override
    def detect(
        self,
        video: Path,
        *,
        transcript: Transcript | None = None,
        chat: Sequence[ChatMessage] | None = None,
    ) -> tuple[Moment, ...]:
        _ = (video, transcript, chat)
        return ()


class _FakeModel(LongVideoModel):
    @override
    def select_segments(self, prompt: Mapping[str, object]) -> dict[str, object]:
        _ = prompt
        return {"segments": [], "rationale": "mandatory ranges"}


class _FakeReframer(Reframer):
    @override
    def analyze(self, video: Path) -> ReframeResult:
        _ = video
        return ReframeResult(targets=(), source_width=320, source_height=240)

    @override
    def render(self, *, video: Path, destination: Path, result: ReframeResult) -> Path:
        _ = result
        destination.parent.mkdir(parents=True, exist_ok=True)
        _ = shutil.copyfile(video, destination)
        return destination


class _FakeSubtitleRenderer(SubtitleBurner):
    @override
    def write(self, words: Sequence[Word], destination: Path) -> Path:
        _ = words
        destination.parent.mkdir(parents=True, exist_ok=True)
        _ = destination.write_text("Dialogue: 0,0:00:00.00,0:00:01.00", encoding="utf-8")
        return destination

    @override
    def burn(
        self, *, video: Path, subtitles: Path, destination: Path, mute_audio: bool = False
    ) -> Path:
        _ = (subtitles, mute_audio)
        destination.parent.mkdir(parents=True, exist_ok=True)
        _ = shutil.copyfile(video, destination)
        return destination


def _run_e2e_pipeline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    source_video: Path,
    ranges: tuple[tuple[float, float], ...],
) -> tuple[Path, Path]:
    downloader = _FakeSectionDownloader(source_video)

    def _factory(*_args: object, **_kwargs: object) -> _FakeSectionDownloader:
        return downloader

    monkeypatch.setattr("kliptych.orchestrator.MediaDownloader", _factory)
    contract = make_contract(
        min_s=5,
        max_s=60,
        timestamp_ranges=tuple(TimestampRange(start_sec=s, end_sec=e) for s, e in ranges),
    )
    out_dir = tmp_path / "out"
    config = PipelineConfig(
        output_dir=out_dir,
        contract=contract,
        render=RenderConfig(),
    )
    result = run_long_video(
        "https://example.com/stream",
        config=config,
        model=_FakeModel(),
        detector=_FakeDetector(),
        transcriber=_FakeTranscriber(),
        selector=LLMSegmentSelector(),
        reframer=_FakeReframer(),
        subtitle_renderer=_FakeSubtitleRenderer(),
    )
    return result.final_video, out_dir


def test_e2e_single_range_60_to_75(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = _generate_synthetic_video(tmp_path / "source_120.mp4", duration_s=120)
    final_video, _ = _run_e2e_pipeline(
        tmp_path, monkeypatch, source_video=source, ranges=((60.0, 75.0),)
    )
    assert final_video.is_file()
    assert final_video.stat().st_size > 1000
    assert math.isclose(_probe(final_video), 15.0, abs_tol=0.5)


def test_e2e_single_range_10_to_40(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = _generate_synthetic_video(tmp_path / "source_120.mp4", duration_s=120)
    final_video, _ = _run_e2e_pipeline(
        tmp_path, monkeypatch, source_video=source, ranges=((10.0, 40.0),)
    )
    assert final_video.is_file()
    assert final_video.stat().st_size > 1000
    assert math.isclose(_probe(final_video), 30.0, abs_tol=0.5)


def test_e2e_two_ranges_60_75_and_90_100(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = _generate_synthetic_video(tmp_path / "source_120.mp4", duration_s=120)
    _, out_dir = _run_e2e_pipeline(
        tmp_path, monkeypatch, source_video=source, ranges=((60.0, 75.0), (90.0, 100.0))
    )
    f0 = out_dir / "final_00.mp4"
    f1 = out_dir / "final_01.mp4"
    assert f0.is_file()
    assert f1.is_file()
    assert f0.stat().st_size > 1000
    assert f1.stat().st_size > 1000
    assert math.isclose(_probe(f0), 15.0, abs_tol=0.5)
    assert math.isclose(_probe(f1), 10.0, abs_tol=0.5)


def test_e2e_start_beyond_duration_controlled_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _generate_synthetic_video(tmp_path / "source_60.mp4", duration_s=60)
    with pytest.raises(PipelineError):
        _ = _run_e2e_pipeline(tmp_path, monkeypatch, source_video=source, ranges=((70.0, 80.0),))
    out_dir = tmp_path / "out"
    if out_dir.exists():
        for mp4 in out_dir.glob("*.mp4"):
            assert mp4.stat().st_size != 262
