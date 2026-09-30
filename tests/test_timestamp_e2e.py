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

import cv2
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
    color_segments: tuple[tuple[int, str], ...] = (
        (10, "0xFF0000"),
        (30, "0x00FF00"),
        (20, "0x0000FF"),
        (15, "0xFFFF00"),
        (15, "0xFF00FF"),
        (10, "0x00FFFF"),
    )
    filter_parts: list[str] = []
    concat_inputs: list[str] = []
    remaining = duration_s
    for idx, (dur, color) in enumerate(color_segments):
        if remaining <= 0:
            break
        use_dur = min(remaining, dur)
        filter_parts.append(f"color=c={color}:d={use_dur}:s=320x240:r=1[v{idx}];")
        concat_inputs.append(f"[v{idx}]")
        remaining -= use_dur

    if remaining > 0:
        idx = len(filter_parts)
        filter_parts.append(f"color=c=white:d={remaining}:s=320x240:r=1[v{idx}];")
        concat_inputs.append(f"[v{idx}]")

    n = len(concat_inputs)
    filter_complex = "".join(filter_parts) + "".join(concat_inputs) + f"concat=n={n}:v=1:a=0[outv]"
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
        f"sine=frequency=1000:duration={duration_s}",
        "-filter_complex",
        filter_complex,
        "-map",
        "[outv]",
        "-map",
        "0:a",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
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


def _read_frame_pixel(path: Path, frame_idx: int) -> tuple[int, int, int]:
    cap = cv2.VideoCapture(str(path))
    try:
        _ = cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            msg = f"No se pudo leer el frame {frame_idx} de {path}"
            raise AssertionError(msg)
        cy = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) // 2
        cx = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) // 2
        mv = memoryview(frame)
        b = int(mv[cy, cx, 0])
        g = int(mv[cy, cx, 1])
        r = int(mv[cy, cx, 2])
        return b, g, r
    finally:
        cap.release()


def _frame_count(path: Path) -> int:
    cap = cv2.VideoCapture(str(path))
    try:
        return int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        cap.release()


def _assert_frame_color(path: Path, frame_idx: int, color_name: str) -> None:
    b, g, r = _read_frame_pixel(path, frame_idx)
    match color_name:
        case "yellow":
            assert b < 50
            assert g > 200
            assert r > 200
        case "green":
            assert b < 50
            assert g > 200
            assert r < 50
        case "cyan":
            assert b > 200
            assert g > 200
            assert r < 50
        case "red":
            assert b < 50
            assert g < 50
            assert r > 200
        case "blue":
            assert b > 200
            assert g < 50
            assert r < 50
        case "magenta":
            assert b > 200
            assert g < 50
            assert r > 200
        case _:
            msg = f"Color desconocido: {color_name}"
            raise ValueError(msg)


def test_e2e_single_range_60_to_75(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = _generate_synthetic_video(tmp_path / "source_120.mp4", duration_s=120)
    final_video, _ = _run_e2e_pipeline(
        tmp_path, monkeypatch, source_video=source, ranges=((60.0, 75.0),)
    )
    assert final_video.is_file()
    assert final_video.stat().st_size > 1000
    assert math.isclose(_probe(final_video), 15.0, abs_tol=0.5)
    count = _frame_count(final_video)
    assert count > 0
    _assert_frame_color(final_video, 0, "yellow")
    _assert_frame_color(final_video, count - 1, "yellow")


def test_e2e_single_range_10_to_40(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = _generate_synthetic_video(tmp_path / "source_120.mp4", duration_s=120)
    final_video, _ = _run_e2e_pipeline(
        tmp_path, monkeypatch, source_video=source, ranges=((10.0, 40.0),)
    )
    assert final_video.is_file()
    assert final_video.stat().st_size > 1000
    assert math.isclose(_probe(final_video), 30.0, abs_tol=0.5)
    count = _frame_count(final_video)
    assert count > 0
    _assert_frame_color(final_video, 0, "green")
    _assert_frame_color(final_video, count - 1, "green")


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
    count0 = _frame_count(f0)
    assert count0 > 0
    _assert_frame_color(f0, 0, "yellow")
    _assert_frame_color(f0, count0 - 1, "yellow")

    count1 = _frame_count(f1)
    assert count1 > 0
    _assert_frame_color(f1, 0, "cyan")
    _assert_frame_color(f1, count1 - 1, "cyan")


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
