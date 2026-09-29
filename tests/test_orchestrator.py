"""Tests unitarios del orquestador long_video con todos los límites simulados.

Ninguna etapa real se ejecuta: descarga, transcripción, momentos, modelo,
selector, reframe y subtítulos se inyectan como fakes; ffmpeg se simula. Se
verifica el orden de las etapas, la estructura del resultado, la limpieza de
temporales en éxito y en cada fallo, y la traducción de errores a PipelineError.
"""

import logging
import subprocess
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, cast

import pytest

from kliptych import orchestrator
from kliptych.contract import Contract, Segment
from kliptych.download import DownloadError, MediaDownloader
from kliptych.encoding import RenderConfig
from kliptych.moments import (
    ChatMessage,
    FFmpegMomentDetector,
    Moment,
    MomentDetectionError,
    MomentDetector,
    MomentSource,
)
from kliptych.orchestrator import (
    PipelineConfig,
    PipelineError,
    PipelineResult,
    Reframer,
    SubtitleBurner,
    compute_long_video_fingerprint,
    compute_slideshow_fingerprint,
    run_long_video,
)
from kliptych.reframe import (
    FFmpegReframer,
    ReframeError,
    ReframeResult,
    ReframeTarget,
)
from kliptych.runtime import ModelError
from kliptych.segment import (
    LLMSegmentSelector,
    SegmentSelection,
    SegmentSelectionError,
    SegmentSelector,
)
from kliptych.subtitles import SubtitleError, SubtitleRenderer
from kliptych.transcribe import (
    FasterWhisperTranscriber,
    Transcriber,
    Transcript,
    TranscriptionError,
    Word,
)

_URL = "https://example.com/video"


class _Registry(Protocol):
    """Superficie del registro de limpieza que usan los tests."""

    paths: list[Path]

    def register(self, path: Path) -> Path: ...

    def cleanup(self) -> tuple[str, ...]: ...


class _ResolvedDeps(Protocol):
    """Superficie de las dependencias resueltas que usan los tests."""

    downloader: MediaDownloader
    detector: MomentDetector
    transcriber: Transcriber
    selector: SegmentSelector
    reframer: Reframer
    subtitle_renderer: SubtitleBurner


def _private(name: str) -> object:
    return cast("object", getattr(orchestrator, name))


_cleanup_registry = cast("Callable[[], _Registry]", _private("_CleanupRegistry"))
_cut_segment = cast("Callable[..., Path]", _private("_cut_segment"))
_cut_exact_ffmpeg = cast("Callable[..., Path]", _private("_cut_exact_ffmpeg"))
_default_reframer = cast(
    "Callable[[PipelineConfig], FFmpegReframer]", _private("_default_reframer")
)
_file_signature = cast("Callable[[Path | str | None], str | None]", _private("_file_signature"))
_primary_segment = cast("Callable[[SegmentSelection], Segment]", _private("_primary_segment"))
_resolve_dependencies = cast("Callable[..., _ResolvedDeps]", _private("_resolve_dependencies"))
_run_ffmpeg = cast("Callable[..., None]", _private("_run_ffmpeg"))
_segment_words = cast(
    "Callable[[Transcript, Segment], tuple[Word, ...]]", _private("_segment_words")
)


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
            "segments": [{"start_s": 0.0, "end_s": 1.0}],
            "geo_target": None,
            "min_views_for_payout": {"value": None, "enforcement": "post_publication_manual"},
            "analytics_proof_required": {"value": False, "enforcement": "post_publication_manual"},
        }
    )


def _transcript() -> Transcript:
    words = (
        Word(start_s=0.0, end_s=0.5, text="hola", confidence=0.9, token_id=0),
        Word(start_s=0.5, end_s=1.0, text="mundo", confidence=0.9, token_id=1),
        Word(start_s=1.0, end_s=1.5, text="fuera", confidence=0.9, token_id=2),
    )
    return Transcript(words=words, language="es", duration_s=2.0, text="hola mundo fuera")


def _moment() -> Moment:
    return Moment(start_s=0.0, end_s=1.0, score=0.8, source=MomentSource.FUSED)


def _selection() -> SegmentSelection:
    return SegmentSelection(segments=(Segment(start_s=0.0, end_s=1.0),), rationale="mejor tramo")


def _reframe_result() -> ReframeResult:
    return ReframeResult(
        targets=(ReframeTarget(x=0, y=0, width=134, height=240),),
        source_width=320,
        source_height=240,
    )


@dataclass
class _Harness:
    """Errores a inyectar por etapa y bitácora de llamadas."""

    events: list[str] = field(default_factory=list)
    download_error: Exception | None = None
    transcribe_error: Exception | None = None
    detect_error: Exception | None = None
    prompt_error: Exception | None = None
    model_error: Exception | None = None
    reframe_error: Exception | None = None
    write_error: Exception | None = None
    burn_error: Exception | None = None
    ffmpeg_error: Exception | None = None
    ffmpeg_returncode: int = 0


class _FakeDownloader:
    def __init__(self, events: list[str], error: Exception | None) -> None:
        self._events: list[str] = events
        self._error: Exception | None = error

    def download_video(
        self, *, url: str, destination: Path, format_selector: str | None = None
    ) -> Path:
        _ = (url, format_selector)
        self._events.append("download")
        if self._error is not None:
            raise self._error
        _ = destination.write_bytes(b"source")
        return destination


class _FakeTranscriber:
    def __init__(self, events: list[str], error: Exception | None) -> None:
        self._events: list[str] = events
        self._error: Exception | None = error

    def transcribe(self, audio: Path) -> Transcript:
        _ = audio
        self._events.append("transcribe")
        if self._error is not None:
            raise self._error
        return _transcript()


class _FakeDetector:
    def __init__(self, events: list[str], error: Exception | None) -> None:
        self._events: list[str] = events
        self._error: Exception | None = error

    def detect(
        self,
        video: Path,
        *,
        transcript: Transcript | None = None,
        chat: Sequence[ChatMessage] | None = None,
    ) -> tuple[Moment, ...]:
        _ = (video, transcript, chat)
        self._events.append("detect")
        if self._error is not None:
            raise self._error
        return (_moment(),)


class _FakeSelector:
    def __init__(self, events: list[str], error: Exception | None) -> None:
        self._events: list[str] = events
        self._error: Exception | None = error

    def build_prompt(
        self,
        transcript: Transcript,
        moments: tuple[Moment, ...],
        contract: Contract,
    ) -> dict[str, object]:
        _ = (transcript, moments, contract)
        self._events.append("build_prompt")
        if self._error is not None:
            raise self._error
        return {"campaign_id": "camp-01"}

    def parse_response(self, raw: object) -> SegmentSelection:
        _ = raw
        self._events.append("parse_response")
        return _selection()


class _FakeModel:
    def __init__(self, events: list[str], error: Exception | None) -> None:
        self._events: list[str] = events
        self._error: Exception | None = error

    def select_segments(self, prompt: Mapping[str, object]) -> object:
        _ = prompt
        self._events.append("select_segments")
        if self._error is not None:
            raise self._error
        return {"segments": [{"start_s": 0.0, "end_s": 1.0}], "rationale": "ok"}


class _FakeReframer:
    def __init__(self, events: list[str], error: Exception | None) -> None:
        self._events: list[str] = events
        self._error: Exception | None = error

    def analyze(self, video: Path) -> ReframeResult:
        _ = video
        self._events.append("analyze")
        if self._error is not None:
            raise self._error
        return _reframe_result()

    def render(self, *, video: Path, destination: Path, result: ReframeResult) -> Path:
        _ = (video, result)
        self._events.append("render")
        _ = destination.write_bytes(b"reframed")
        return destination


class _FakeSubtitleRenderer:
    def __init__(
        self,
        events: list[str],
        *,
        write_error: Exception | None,
        burn_error: Exception | None,
    ) -> None:
        self._events: list[str] = events
        self._write_error: Exception | None = write_error
        self._burn_error: Exception | None = burn_error

    def write(self, words: Sequence[Word], destination: Path) -> Path:
        _ = words
        self._events.append("write")
        if self._write_error is not None:
            raise self._write_error
        _ = destination.write_text("ass", encoding="utf-8")
        return destination

    def burn(
        self, *, video: Path, subtitles: Path, destination: Path, mute_audio: bool = False
    ) -> Path:
        _ = (video, subtitles, mute_audio)
        self._events.append("burn")
        if self._burn_error is not None:
            raise self._burn_error
        _ = destination.write_bytes(b"final")
        return destination


@dataclass(frozen=True, slots=True)
class _Injectables:
    model: _FakeModel
    detector: _FakeDetector
    transcriber: _FakeTranscriber
    selector: _FakeSelector
    reframer: _FakeReframer
    subtitle_renderer: _FakeSubtitleRenderer


def _ffmpeg(harness: _Harness) -> object:
    last_duration = ["1.0"]

    def run(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        _ = kwargs
        if argv and "ffprobe" in argv[0]:
            return subprocess.CompletedProcess(
                args=argv,
                returncode=0,
                stdout=f"width=1080\nheight=1920\nduration={last_duration[0]}\n",
                stderr="",
            )
        if "-t" in argv:
            last_duration[0] = argv[argv.index("-t") + 1]
        harness.events.append("cut")
        _ = Path(argv[-1]).write_bytes(b"clip")
        if harness.ffmpeg_error is not None:
            raise harness.ffmpeg_error
        return subprocess.CompletedProcess(
            args=argv,
            returncode=harness.ffmpeg_returncode,
            stdout="",
            stderr="boom",
        )

    return run


def _install(monkeypatch: pytest.MonkeyPatch, harness: _Harness) -> _Injectables:
    def factory(*, timeout_s: float, max_size_bytes: int) -> _FakeDownloader:
        _ = (timeout_s, max_size_bytes)
        return _FakeDownloader(harness.events, harness.download_error)

    monkeypatch.setattr("kliptych.orchestrator.MediaDownloader", factory)
    monkeypatch.setattr("kliptych.orchestrator.subprocess.run", _ffmpeg(harness))
    return _Injectables(
        model=_FakeModel(harness.events, harness.model_error),
        detector=_FakeDetector(harness.events, harness.detect_error),
        transcriber=_FakeTranscriber(harness.events, harness.transcribe_error),
        selector=_FakeSelector(harness.events, harness.prompt_error),
        reframer=_FakeReframer(harness.events, harness.reframe_error),
        subtitle_renderer=_FakeSubtitleRenderer(
            harness.events,
            write_error=harness.write_error,
            burn_error=harness.burn_error,
        ),
    )


def _config(tmp_path: Path) -> PipelineConfig:
    return PipelineConfig(
        output_dir=tmp_path / "out",
        contract=_contract(),
        render=RenderConfig(),
    )


def _run(tmp_path: Path, injectables: _Injectables) -> PipelineResult:
    return run_long_video(
        _URL,
        model=injectables.model,
        config=_config(tmp_path),
        detector=injectables.detector,
        transcriber=injectables.transcriber,
        selector=injectables.selector,
        reframer=injectables.reframer,
        subtitle_renderer=injectables.subtitle_renderer,
    )


def _leftovers(output_dir: Path) -> list[str]:
    return [path.name for path in output_dir.iterdir() if ".part-" in path.name]


def test_pipeline_config_defaults() -> None:
    config = PipelineConfig(output_dir=Path("out"), contract=_contract(), render=RenderConfig())
    assert config.face_model_path is None
    assert config.download_timeout_s == pytest.approx(600.0)
    assert config.download_max_size_bytes == 2 * 1024**3


def test_run_long_video_returns_result(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    harness = _Harness()
    injectables = _install(monkeypatch, harness)
    result = _run(tmp_path, injectables)
    assert isinstance(result, PipelineResult)
    assert result.source == tmp_path / "out" / "source.mp4"
    assert result.source.is_file()
    assert result.final_video == tmp_path / "out" / "final.mp4"
    assert result.final_video.is_file()
    assert result.transcript == _transcript()
    assert result.moments == (_moment(),)
    assert result.selection == _selection()
    assert result.reframe == _reframe_result()
    assert result.cleaning


def test_run_long_video_batch_produces_all_segments(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness = _Harness()
    injectables = _install(monkeypatch, harness)
    three_segments = SegmentSelection(
        segments=(
            Segment(start_s=0.0, end_s=1.0),
            Segment(start_s=1.0, end_s=2.0),
            Segment(start_s=2.0, end_s=3.0),
        ),
        rationale="tres tramos",
    )

    def _fake_parse(_raw: object) -> SegmentSelection:
        return three_segments

    monkeypatch.setattr(injectables.selector, "parse_response", _fake_parse)
    result = _run(tmp_path, injectables)
    assert isinstance(result, PipelineResult)
    assert len(result.final_videos) == 3
    assert result.final_videos[0] == tmp_path / "out" / "final_00.mp4"
    assert result.final_videos[1] == tmp_path / "out" / "final_01.mp4"
    assert result.final_videos[2] == tmp_path / "out" / "final_02.mp4"
    assert result.final_video == result.final_videos[0]
    for video in result.final_videos:
        assert video.is_file()


def test_run_long_video_stage_order(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    harness = _Harness()
    injectables = _install(monkeypatch, harness)
    _ = _run(tmp_path, injectables)
    assert harness.events == [
        "download",
        "transcribe",
        "detect",
        "build_prompt",
        "select_segments",
        "parse_response",
        "cut",
        "analyze",
        "render",
        "write",
        "burn",
    ]


def test_cleanup_removes_temps_on_success(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    harness = _Harness()
    injectables = _install(monkeypatch, harness)
    result = _run(tmp_path, injectables)
    output_dir = tmp_path / "out"
    assert _leftovers(output_dir) == []
    assert result.subtitles is not None
    assert not result.subtitles.exists()
    assert result.source.is_file()
    assert len(result.cleaning) == 3


@pytest.mark.parametrize(
    ("field_name", "error"),
    [
        ("download_error", DownloadError("boom")),
        ("transcribe_error", TranscriptionError("boom")),
        ("detect_error", MomentDetectionError("boom")),
        ("prompt_error", SegmentSelectionError("boom")),
        ("model_error", ModelError("boom")),
        ("reframe_error", ReframeError("boom")),
        ("write_error", SubtitleError("boom")),
        ("burn_error", SubtitleError("boom")),
    ],
)
def test_failure_raises_pipeline_error_and_cleans(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field_name: str,
    error: Exception,
) -> None:
    harness = _Harness()
    setattr(harness, field_name, error)
    injectables = _install(monkeypatch, harness)
    with pytest.raises(PipelineError) as info:
        _ = _run(tmp_path, injectables)
    assert info.value.__cause__ is error
    assert _leftovers(tmp_path / "out") == []


def test_cut_failure_cleans_partial(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    harness = _Harness(ffmpeg_returncode=1)
    injectables = _install(monkeypatch, harness)
    with pytest.raises(PipelineError, match="ffmpeg falló con código 1"):
        _ = _run(tmp_path, injectables)
    assert _leftovers(tmp_path / "out") == []


def test_run_ffmpeg_missing_binary(monkeypatch: pytest.MonkeyPatch) -> None:
    def run(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        _ = (argv, kwargs)
        message = "ffmpeg"
        raise FileNotFoundError(message)

    monkeypatch.setattr("kliptych.orchestrator.subprocess.run", run)
    with pytest.raises(PipelineError, match="no está disponible"):
        _run_ffmpeg(["ffmpeg"], render=RenderConfig())


def test_run_ffmpeg_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    def run(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        _ = (argv, kwargs)
        raise subprocess.TimeoutExpired(cmd="ffmpeg", timeout=1.0)

    monkeypatch.setattr("kliptych.orchestrator.subprocess.run", run)
    with pytest.raises(PipelineError, match="timeout"):
        _run_ffmpeg(["ffmpeg"], render=RenderConfig())


def test_run_ffmpeg_os_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def run(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        _ = (argv, kwargs)
        message = "permiso"
        raise OSError(message)

    monkeypatch.setattr("kliptych.orchestrator.subprocess.run", run)
    with pytest.raises(PipelineError, match="no se pudo ejecutar"):
        _run_ffmpeg(["ffmpeg"], render=RenderConfig())


def test_run_ffmpeg_nonzero(monkeypatch: pytest.MonkeyPatch) -> None:
    def run(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        _ = kwargs
        return subprocess.CompletedProcess(args=argv, returncode=2, stdout="", stderr="boom")

    monkeypatch.setattr("kliptych.orchestrator.subprocess.run", run)
    with pytest.raises(PipelineError, match="código 2"):
        _run_ffmpeg(["ffmpeg"], render=RenderConfig())


def test_run_ffmpeg_nvenc_fallback_to_libx264(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    calls: list[list[str]] = []

    def mock_run(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        _ = kwargs
        calls.append(list(argv))
        if "h264_nvenc" in argv:
            return subprocess.CompletedProcess(
                args=argv,
                returncode=1,
                stdout="",
                stderr="NVENC out of memory",
            )
        return subprocess.CompletedProcess(
            args=argv,
            returncode=0,
            stdout="",
            stderr="",
        )

    monkeypatch.setattr("kliptych.orchestrator.subprocess.run", mock_run)
    render = RenderConfig(nvenc_available=True)
    argv = [
        "ffmpeg",
        "-i",
        "in.mp4",
        "-c:v",
        "h264_nvenc",
        "-preset",
        "p5",
        "-cq",
        "23",
        "-pix_fmt",
        "yuv420p",
        "out.mp4",
    ]

    with caplog.at_level(logging.WARNING, logger="kliptych.orchestrator"):
        _run_ffmpeg(argv, render=render)

    assert len(calls) == 2
    assert "h264_nvenc" in calls[0]
    assert "libx264" in calls[1]
    assert "h264_nvenc" not in calls[1]
    assert "-preset" in calls[1]
    assert calls[1][calls[1].index("-preset") + 1] == "medium"
    assert "-crf" in calls[1]
    assert calls[1][calls[1].index("-crf") + 1] == "20"
    assert any(
        "NVENC falló, reintentando con libx264" in record.message for record in caplog.records
    )


def test_run_ffmpeg_nvenc_fallback_propagates_when_libx264_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[list[str]] = []

    def mock_run(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        _ = kwargs
        calls.append(list(argv))
        return subprocess.CompletedProcess(
            args=argv,
            returncode=1,
            stdout="",
            stderr="all encoders failed",
        )

    monkeypatch.setattr("kliptych.orchestrator.subprocess.run", mock_run)
    render = RenderConfig(nvenc_available=True)
    argv = [
        "ffmpeg",
        "-i",
        "in.mp4",
        "-c:v",
        "h264_nvenc",
        "-preset",
        "p5",
        "-cq",
        "23",
        "-pix_fmt",
        "yuv420p",
        "out.mp4",
    ]

    with pytest.raises(PipelineError, match="ffmpeg falló con código 1"):
        _run_ffmpeg(argv, render=render)

    assert len(calls) == 2


def test_run_ffmpeg_called_process_error_nvenc_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[list[str]] = []

    def mock_run(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        _ = kwargs
        calls.append(list(argv))
        if "h264_nvenc" in argv:
            raise subprocess.CalledProcessError(
                returncode=1, cmd=argv, output="", stderr="nvenc error"
            )
        return subprocess.CompletedProcess(args=argv, returncode=0, stdout="", stderr="")

    monkeypatch.setattr("kliptych.orchestrator.subprocess.run", mock_run)
    render = RenderConfig(nvenc_available=True)
    argv = [
        "ffmpeg",
        "-i",
        "in.mp4",
        "-c:v",
        "h264_nvenc",
        "-preset",
        "p5",
        "-cq",
        "23",
        "-pix_fmt",
        "yuv420p",
        "out.mp4",
    ]
    _run_ffmpeg(argv, render=render)
    assert len(calls) == 2
    assert "libx264" in calls[1]


def test_run_ffmpeg_called_process_error_without_nvenc_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def mock_run(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        _ = (argv, kwargs)
        raise subprocess.CalledProcessError(returncode=1, cmd=argv, output="", stderr="cpu error")

    monkeypatch.setattr("kliptych.orchestrator.subprocess.run", mock_run)
    render = RenderConfig(nvenc_available=False)
    argv = ["ffmpeg", "-i", "in.mp4", "-c:v", "libx264", "out.mp4"]
    with pytest.raises(PipelineError, match="ffmpeg falló"):
        _run_ffmpeg(argv, render=render)


def test_cut_segment_builds_list_argv(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, list[str]] = {}

    def run(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        _ = kwargs
        if argv and "ffprobe" in argv[0]:
            return subprocess.CompletedProcess(
                args=argv, returncode=0, stdout="width=1080\nheight=1920\nduration=1.5\n", stderr=""
            )
        captured["argv"] = argv
        _ = Path(argv[-1]).write_bytes(b"clip")
        return subprocess.CompletedProcess(args=argv, returncode=0, stdout="", stderr="")

    monkeypatch.setattr("kliptych.orchestrator.subprocess.run", run)
    source = tmp_path / "source.mp4"
    _ = source.write_bytes(b"source")
    registry = _cleanup_registry()
    destination = _cut_segment(
        source,
        segment=Segment(start_s=1.0, end_s=2.5),
        render=RenderConfig(),
        registry=registry,
    )
    argv = captured["argv"]
    assert isinstance(argv, list)
    assert argv[argv.index("-ss") + 1] == "1.000"
    assert argv[argv.index("-t") + 1] == "1.500"
    assert destination.is_file()
    assert destination in registry.paths


def test_cut_segment_fails_when_probe_duration_zero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def run(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        _ = kwargs
        if argv and argv[0] == "ffprobe":
            return subprocess.CompletedProcess(
                args=argv, returncode=0, stdout="width=1080\nheight=1920\nduration=0.0\n", stderr=""
            )
        _ = Path(argv[-1]).write_bytes(b"clip")
        return subprocess.CompletedProcess(args=argv, returncode=0, stdout="", stderr="")

    monkeypatch.setattr("kliptych.orchestrator.subprocess.run", run)
    source = tmp_path / "source.mp4"
    _ = source.write_bytes(b"source")
    with pytest.raises(PipelineError, match="duración inválida"):
        _ = _cut_segment(
            source,
            segment=Segment(start_s=1.0, end_s=2.5),
            render=RenderConfig(),
            registry=_cleanup_registry(),
        )


def test_cut_segment_fails_when_probe_duration_mismatches_tolerance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def run(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        _ = kwargs
        if argv and argv[0] == "ffprobe":
            return subprocess.CompletedProcess(
                args=argv, returncode=0, stdout="width=1080\nheight=1920\nduration=0.8\n", stderr=""
            )
        _ = Path(argv[-1]).write_bytes(b"clip")
        return subprocess.CompletedProcess(args=argv, returncode=0, stdout="", stderr="")

    monkeypatch.setattr("kliptych.orchestrator.subprocess.run", run)
    source = tmp_path / "source.mp4"
    _ = source.write_bytes(b"source")
    with pytest.raises(PipelineError, match="tolerancia"):
        _ = _cut_segment(
            source,
            segment=Segment(start_s=1.0, end_s=2.5),
            render=RenderConfig(),
            registry=_cleanup_registry(),
        )


def test_cut_exact_ffmpeg_fails_when_probe_duration_mismatches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def run(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        _ = kwargs
        if argv and argv[0] == "ffprobe":
            return subprocess.CompletedProcess(
                args=argv, returncode=0, stdout="width=1080\nheight=1920\nduration=1.0\n", stderr=""
            )
        _ = Path(argv[-1]).write_bytes(b"clip")
        return subprocess.CompletedProcess(args=argv, returncode=0, stdout="", stderr="")

    monkeypatch.setattr("kliptych.orchestrator.subprocess.run", run)
    source = tmp_path / "source.mp4"
    dest = tmp_path / "dest.mp4"
    _ = source.write_bytes(b"source")
    with pytest.raises(PipelineError, match="tolerancia"):
        _ = _cut_exact_ffmpeg(
            source=source,
            destination=dest,
            start_s=10.0,
            duration_s=5.0,
            render=RenderConfig(),
        )


def test_segment_words_shifts_and_clamps() -> None:
    words = _segment_words(_transcript(), Segment(start_s=0.25, end_s=1.25))
    assert [word.text for word in words] == ["hola", "mundo", "fuera"]
    assert [word.token_id for word in words] == [0, 1, 2]
    assert words[0].start_s == pytest.approx(0.0)
    assert words[0].end_s == pytest.approx(0.25)
    assert words[1].start_s == pytest.approx(0.25)
    assert words[2].end_s == pytest.approx(1.0)


def test_segment_words_excludes_outside() -> None:
    words = _segment_words(_transcript(), Segment(start_s=0.5, end_s=1.0))
    assert [word.text for word in words] == ["mundo"]
    assert words[0].start_s == pytest.approx(0.0)
    assert words[0].end_s == pytest.approx(0.5)


def test_segment_words_empty_without_overlap() -> None:
    assert _segment_words(_transcript(), Segment(start_s=5.0, end_s=6.0)) == ()


def test_segment_words_skips_zero_length_overlap() -> None:
    zero = Word(start_s=0.6, end_s=0.6, text="x", confidence=0.9, token_id=0)
    transcript = Transcript(words=(zero,), language="es", duration_s=1.0, text="x")
    assert _segment_words(transcript, Segment(start_s=0.5, end_s=1.0)) == ()


def test_run_long_video_rejects_unwritable_output_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness = _Harness()
    injectables = _install(monkeypatch, harness)
    blocker = tmp_path / "out"
    _ = blocker.write_bytes(b"soy un archivo")
    config = PipelineConfig(output_dir=blocker, contract=_contract(), render=RenderConfig())
    with pytest.raises(PipelineError, match="directorio de salida"):
        _ = run_long_video(
            _URL,
            model=injectables.model,
            config=config,
            detector=injectables.detector,
            transcriber=injectables.transcriber,
            selector=injectables.selector,
            reframer=injectables.reframer,
            subtitle_renderer=injectables.subtitle_renderer,
        )


def test_primary_segment_empty_raises() -> None:
    with pytest.raises(PipelineError, match="segmentos"):
        _ = _primary_segment(SegmentSelection(segments=(), rationale="vacío"))


def test_default_reframer_requires_model_path(tmp_path: Path) -> None:
    config = PipelineConfig(
        output_dir=tmp_path / "out",
        contract=_contract(),
        render=RenderConfig(),
    )
    with pytest.raises(PipelineError, match="face_model_path"):
        _ = _default_reframer(config)


def test_resolve_dependencies_builds_defaults(tmp_path: Path) -> None:
    config = PipelineConfig(
        output_dir=tmp_path / "out",
        contract=_contract(),
        render=RenderConfig(),
        face_model_path=tmp_path / "face.tflite",
    )
    dependencies = _resolve_dependencies(
        config=config,
        detector=None,
        transcriber=None,
        selector=None,
        reframer=None,
        subtitle_renderer=None,
    )
    assert isinstance(dependencies.detector, FFmpegMomentDetector)
    assert isinstance(dependencies.transcriber, FasterWhisperTranscriber)
    assert isinstance(dependencies.selector, LLMSegmentSelector)
    assert isinstance(dependencies.reframer, FFmpegReframer)
    assert isinstance(dependencies.subtitle_renderer, SubtitleRenderer)


def test_resolve_dependencies_keeps_injected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness = _Harness()
    injectables = _install(monkeypatch, harness)
    dependencies = _resolve_dependencies(
        config=_config(tmp_path),
        detector=injectables.detector,
        transcriber=injectables.transcriber,
        selector=injectables.selector,
        reframer=injectables.reframer,
        subtitle_renderer=injectables.subtitle_renderer,
    )
    assert dependencies.detector is injectables.detector
    assert dependencies.transcriber is injectables.transcriber
    assert dependencies.selector is injectables.selector
    assert dependencies.reframer is injectables.reframer
    assert dependencies.subtitle_renderer is injectables.subtitle_renderer


def test_cleanup_registry_removes_existing(tmp_path: Path) -> None:
    first = tmp_path / "first.part"
    second = tmp_path / "second.part"
    _ = first.write_bytes(b"a")
    _ = second.write_bytes(b"b")
    registry = _cleanup_registry()
    assert registry.register(first) == first
    _ = registry.register(second)
    _ = registry.register(tmp_path / "missing.part")
    removed = registry.cleanup()
    assert set(removed) == {str(first), str(second)}
    assert not first.exists()
    assert not second.exists()
    assert registry.cleanup() == ()


def test_cleanup_registry_ignores_directories(tmp_path: Path) -> None:
    directory = tmp_path / "dir.part"
    directory.mkdir()
    registry = _cleanup_registry()
    _ = registry.register(directory)
    assert registry.cleanup() == ()
    assert directory.is_dir()


def test_file_signature_content_change_and_fallback(tmp_path: Path) -> None:
    file_path = tmp_path / "track.mp3"
    _ = file_path.write_bytes(b"initial content")
    sig1 = _file_signature(file_path)
    assert sig1 is not None
    assert sig1.startswith(f"{file_path.as_posix()}:")

    _ = file_path.write_bytes(b"modified content")
    sig2 = _file_signature(file_path)
    assert sig2 is not None
    assert sig1 != sig2

    missing = tmp_path / "missing.mp3"
    assert _file_signature(missing) == str(missing)
    assert _file_signature("https://example.com/stream.mp4") == "https://example.com/stream.mp4"
    assert _file_signature(None) is None


def test_file_signature_handles_oserror(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def mock_is_file(_self: Path) -> bool:
        msg = "Permission denied"
        raise OSError(msg)

    monkeypatch.setattr(Path, "is_file", mock_is_file)
    test_path = tmp_path / "file.txt"
    assert _file_signature(test_path) == str(test_path)


def test_long_video_fingerprint_file_content_change(tmp_path: Path) -> None:
    track = tmp_path / "audio.mp3"
    _ = track.write_bytes(b"version 1")
    config = PipelineConfig(
        output_dir=tmp_path / "out",
        contract=_contract(),
        render=RenderConfig(),
        audio_locked=True,
        audio_track_path=track,
    )
    fp1 = compute_long_video_fingerprint("https://example.com/video.mp4", config=config)

    _ = track.write_bytes(b"version 2 with different bytes")
    fp2 = compute_long_video_fingerprint("https://example.com/video.mp4", config=config)
    assert fp1 != fp2


def test_long_video_fingerprint_local_source_content_change(tmp_path: Path) -> None:
    source_file = tmp_path / "source.mp4"
    _ = source_file.write_bytes(b"video bytes v1")
    config = PipelineConfig(
        output_dir=tmp_path / "out",
        contract=_contract(),
        render=RenderConfig(),
    )
    fp1 = compute_long_video_fingerprint(str(source_file), config=config)

    _ = source_file.write_bytes(b"video bytes v2 modified")
    fp2 = compute_long_video_fingerprint(str(source_file), config=config)
    assert fp1 != fp2


def test_long_video_fingerprint_face_model_path_change(tmp_path: Path) -> None:
    config_none = PipelineConfig(
        output_dir=tmp_path / "out",
        contract=_contract(),
        render=RenderConfig(),
        face_model_path=None,
    )
    config_with_model = PipelineConfig(
        output_dir=tmp_path / "out",
        contract=_contract(),
        render=RenderConfig(),
        face_model_path=tmp_path / "face.tflite",
    )
    fp1 = compute_long_video_fingerprint("https://example.com/video.mp4", config=config_none)
    fp2 = compute_long_video_fingerprint("https://example.com/video.mp4", config=config_with_model)
    assert fp1 != fp2


def test_slideshow_fingerprint_file_content_change(tmp_path: Path) -> None:
    img1 = tmp_path / "1.png"
    img2 = tmp_path / "2.png"
    _ = img1.write_bytes(b"image 1 v1")
    _ = img2.write_bytes(b"image 2")
    config = PipelineConfig(
        output_dir=tmp_path / "out",
        contract=_contract(),
        render=RenderConfig(),
    )
    fp1 = compute_slideshow_fingerprint([img1, img2], config=config, slide_duration_s=3.0)

    _ = img1.write_bytes(b"image 1 v2 modified")
    fp2 = compute_slideshow_fingerprint([img1, img2], config=config, slide_duration_s=3.0)
    assert fp1 != fp2


def test_slideshow_fingerprint_face_model_path_change(tmp_path: Path) -> None:
    img = tmp_path / "1.png"
    _ = img.write_bytes(b"img")
    config_none = PipelineConfig(
        output_dir=tmp_path / "out",
        contract=_contract(),
        render=RenderConfig(),
        face_model_path=None,
    )
    config_with_model = PipelineConfig(
        output_dir=tmp_path / "out",
        contract=_contract(),
        render=RenderConfig(),
        face_model_path=tmp_path / "face.tflite",
    )
    fp1 = compute_slideshow_fingerprint([img], config=config_none, slide_duration_s=3.0)
    fp2 = compute_slideshow_fingerprint([img], config=config_with_model, slide_duration_s=3.0)
    assert fp1 != fp2
