"""Tests unitarios del modo Repost/UGC (D2-PR2) con todos los límites simulados.

Descarga, transcripción, momentos, modelo LLM, reframe, subtítulos, ffmpeg y
ffprobe se simulan: cero red y cero subproceso real. Se verifica que el modo
repost omite transcripción, detección de momentos y selección LLM; que usa el
vídeo completo como segmento; que sólo reframea cuando el vídeo no es 9:16; que
la inyección de audio convive con repost; y que ``repost_mode=False`` conserva
el pipeline exactamente igual que antes.
"""

import subprocess
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, cast

import pytest

from kliptych import orchestrator
from kliptych.contract import Contract, Segment
from kliptych.encoding import RenderConfig
from kliptych.moments import ChatMessage, Moment, MomentSource
from kliptych.orchestrator import (
    PipelineConfig,
    PipelineError,
    PipelineResult,
    run_long_video,
)
from kliptych.reframe import ReframeResult, ReframeTarget
from kliptych.segment import SegmentSelection
from kliptych.transcribe import Transcript, Word

_URL = "https://example.com/repost"
_DEFAULT_PROBE = "width=1080\nheight=1920\nduration=2.000000\n"


class _Registry(Protocol):
    """Superficie del registro de limpieza que usan los tests."""

    paths: list[Path]

    def register(self, path: Path) -> Path: ...


class _VideoInfo(Protocol):
    """Superficie de las dimensiones y duración sondeadas."""

    width: int
    height: int
    duration_s: float


class _ResolvedDeps(Protocol):
    """Superficie de las dependencias resueltas que usan los tests."""

    reframer: object | None


def _private(name: str) -> object:
    return cast("object", getattr(orchestrator, name))


_cleanup_registry = cast("Callable[[], _Registry]", _private("_CleanupRegistry"))
_probe_video = cast("Callable[..., _VideoInfo]", _private("_probe_video"))
_needs_reframe = cast("Callable[..., bool]", _private("_needs_reframe"))
_full_video_selection = cast("Callable[..., SegmentSelection]", _private("_full_video_selection"))
_publish = cast("Callable[..., Path]", _private("_publish"))
_resolve_dependencies = cast("Callable[..., _ResolvedDeps]", _private("_resolve_dependencies"))


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


def _long_video_contract() -> Contract:
    return Contract.model_validate(
        {
            "schema_version": "1.1",
            "campaign_id": "camp-long",
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
    )
    return Transcript(words=words, language="es", duration_s=1.0, text="hola mundo")


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
    """Bitácora de llamadas, comandos y respuestas simuladas del sondeo."""

    events: list[str] = field(default_factory=list)
    commands: list[list[str]] = field(default_factory=list)
    probe_stdout: str = _DEFAULT_PROBE
    probe_returncode: int = 0
    ffmpeg_returncode: int = 0


class _Downloader:
    def __init__(self, events: list[str]) -> None:
        self._events: list[str] = events

    def download_video(
        self, *, url: str, destination: Path, format_selector: str | None = None
    ) -> Path:
        _ = format_selector
        self._events.append(f"download:{url}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        _ = destination.write_bytes(b"media")
        return destination


class _Transcriber:
    def __init__(self, events: list[str]) -> None:
        self._events: list[str] = events

    def transcribe(self, audio: Path) -> Transcript:
        _ = audio
        self._events.append("transcribe")
        return _transcript()


class _Detector:
    def __init__(self, events: list[str]) -> None:
        self._events: list[str] = events

    def detect(
        self,
        video: Path,
        *,
        transcript: Transcript | None = None,
        chat: Sequence[ChatMessage] | None = None,
    ) -> tuple[Moment, ...]:
        _ = (video, transcript, chat)
        self._events.append("detect")
        return (_moment(),)


class _Selector:
    def __init__(self, events: list[str]) -> None:
        self._events: list[str] = events

    def build_prompt(
        self,
        transcript: Transcript,
        moments: tuple[Moment, ...],
        contract: Contract,
    ) -> dict[str, object]:
        _ = (transcript, moments, contract)
        self._events.append("build_prompt")
        return {"campaign_id": "camp-long"}

    def parse_response(self, raw: object) -> SegmentSelection:
        _ = raw
        self._events.append("parse_response")
        return _selection()


class _Model:
    def __init__(self, events: list[str]) -> None:
        self._events: list[str] = events

    def select_segments(self, prompt: Mapping[str, object]) -> object:
        _ = prompt
        self._events.append("select_segments")
        return {"segments": [{"start_s": 0.0, "end_s": 1.0}], "rationale": "ok"}


class _Reframer:
    def __init__(self, events: list[str]) -> None:
        self._events: list[str] = events

    def analyze(self, video: Path) -> ReframeResult:
        _ = video
        self._events.append("analyze")
        return _reframe_result()

    def render(self, *, video: Path, destination: Path, result: ReframeResult) -> Path:
        _ = (video, result)
        self._events.append("render")
        destination.parent.mkdir(parents=True, exist_ok=True)
        _ = destination.write_bytes(b"reframed")
        return destination


class _SubtitleRenderer:
    def __init__(self, events: list[str]) -> None:
        self._events: list[str] = events

    def write(self, words: Sequence[Word], destination: Path) -> Path:
        _ = words
        self._events.append("write")
        destination.parent.mkdir(parents=True, exist_ok=True)
        _ = destination.write_text("ass", encoding="utf-8")
        return destination

    def burn(self, *, video: Path, subtitles: Path, destination: Path) -> Path:
        _ = (video, subtitles)
        self._events.append("burn")
        destination.parent.mkdir(parents=True, exist_ok=True)
        _ = destination.write_bytes(b"final")
        return destination


def _ffmpeg(harness: _Harness) -> Callable[..., object]:
    def run(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        _ = kwargs
        harness.commands.append(list(argv))
        if str(argv[0]).lower().endswith("ffprobe"):
            harness.events.append("probe")
            return subprocess.CompletedProcess(
                args=argv,
                returncode=harness.probe_returncode,
                stdout=harness.probe_stdout,
                stderr="probe boom",
            )
        if "-ss" in argv:
            tag = "cut"
        elif "-filter_complex" in argv:
            tag = "inject_mix"
        elif "-map" in argv:
            tag = "inject_replace"
        else:
            tag = "passthrough"
        harness.events.append(tag)
        _ = Path(argv[-1]).write_bytes(b"render")
        return subprocess.CompletedProcess(
            args=argv,
            returncode=harness.ffmpeg_returncode,
            stdout="",
            stderr="boom",
        )

    return run


def _install(monkeypatch: pytest.MonkeyPatch, harness: _Harness) -> None:
    def factory(*, timeout_s: float, max_size_bytes: int) -> _Downloader:
        _ = (timeout_s, max_size_bytes)
        return _Downloader(harness.events)

    monkeypatch.setattr("kliptych.orchestrator.MediaDownloader", factory)
    monkeypatch.setattr("kliptych.orchestrator.subprocess.run", _ffmpeg(harness))


def _config(
    tmp_path: Path,
    *,
    repost_mode: bool = True,
    long_video: bool = False,
    audio_locked: bool = False,
    audio_track_path: Path | None = None,
) -> PipelineConfig:
    return PipelineConfig(
        output_dir=tmp_path / "out",
        contract=_long_video_contract() if long_video else _contract(),
        render=RenderConfig(),
        repost_mode=repost_mode,
        audio_locked=audio_locked,
        audio_track_path=audio_track_path,
    )


def _run(
    events: list[str],
    config: PipelineConfig,
    *,
    reframer: _Reframer | None = None,
) -> PipelineResult:
    return run_long_video(
        _URL,
        model=_Model(events),
        config=config,
        detector=_Detector(events),
        transcriber=_Transcriber(events),
        selector=_Selector(events),
        reframer=reframer if reframer is not None else _Reframer(events),
        subtitle_renderer=_SubtitleRenderer(events),
    )


def _leftovers(output_dir: Path) -> list[str]:
    return [path.name for path in output_dir.iterdir() if ".part-" in path.name]


def test_pipeline_config_repost_defaults() -> None:
    config = PipelineConfig(output_dir=Path("out"), contract=_contract(), render=RenderConfig())
    assert config.repost_mode is False


def test_repost_skips_transcribe_detect_and_select(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness = _Harness()
    _install(monkeypatch, harness)
    result = _run(harness.events, _config(tmp_path))
    assert harness.events == [
        f"download:{_URL}",
        "probe",
        "probe",
        "passthrough",
        "probe",
    ]
    assert "cut" not in harness.events
    assert "transcribe" not in harness.events
    assert "detect" not in harness.events
    assert "build_prompt" not in harness.events
    assert "select_segments" not in harness.events
    assert "parse_response" not in harness.events
    assert result.transcript is None
    assert result.moments == ()
    assert result.reframe is None
    assert result.subtitles is None
    assert result.selection.segments == (Segment(start_s=0.0, end_s=2.0),)
    assert result.final_video == tmp_path / "out" / "final.mp4"
    assert result.final_video.is_file()
    assert _leftovers(tmp_path / "out") == []


def test_repost_skips_reframe_when_vertical(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness = _Harness(probe_stdout="width=1080\nheight=1920\nduration=2.000000\n")
    _install(monkeypatch, harness)
    result = _run(harness.events, _config(tmp_path))
    assert "analyze" not in harness.events
    assert "render" not in harness.events
    assert result.reframe is None


def test_repost_vertical_uses_stream_copy_passthrough(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness = _Harness(probe_stdout="width=1080\nheight=1920\nduration=2.000000\n")
    _install(monkeypatch, harness)
    result = _run(harness.events, _config(tmp_path))
    assert "cut" not in harness.events
    assert "passthrough" in harness.events
    copies = [
        argv for argv in harness.commands if "-c" in argv and argv[argv.index("-c") + 1] == "copy"
    ]
    assert len(copies) == 1
    argv = copies[0]
    assert "-i" in argv
    assert argv[argv.index("-i") + 1] == str(tmp_path / "out" / "source.mp4")
    assert "+faststart" in argv
    assert result.final_video.is_file()
    assert _leftovers(tmp_path / "out") == []


def test_repost_reframes_when_not_vertical(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    harness = _Harness(probe_stdout="width=1920\nheight=1080\nduration=2.000000\n")
    _install(monkeypatch, harness)
    result = _run(harness.events, _config(tmp_path))
    assert "analyze" in harness.events
    assert "render" in harness.events
    assert result.reframe == _reframe_result()
    assert result.final_video.is_file()


def test_repost_with_audio_locked_injects_audio(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness = _Harness()
    _install(monkeypatch, harness)
    track = tmp_path / "track.wav"
    _ = track.write_bytes(b"audio")
    config = _config(tmp_path, audio_locked=True, audio_track_path=track)
    result = _run(harness.events, config)
    assert "inject_replace" in harness.events
    assert "analyze" not in harness.events
    assert result.transcript is None
    assert result.final_video.is_file()
    assert _leftovers(tmp_path / "out") == []


def test_backward_compatible_when_repost_disabled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness = _Harness(probe_stdout="width=320\nheight=240\nduration=1.000000\n")
    _install(monkeypatch, harness)
    result = _run(harness.events, _config(tmp_path, repost_mode=False, long_video=True))
    assert harness.events == [
        f"download:{_URL}",
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
    assert "probe" not in harness.events
    assert result.transcript == _transcript()
    assert result.moments == (_moment(),)
    assert result.reframe == _reframe_result()
    assert result.subtitles is not None
    assert not result.subtitles.exists()


def test_repost_generic_failure_is_pipeline_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness = _Harness(probe_stdout="width=1080\nheight=1920\nduration=2.000000\n")
    harness.ffmpeg_returncode = 1
    _install(monkeypatch, harness)
    with pytest.raises(PipelineError, match="ffmpeg falló"):
        _ = _run(harness.events, _config(tmp_path))
    assert _leftovers(tmp_path / "out") == []


@pytest.mark.parametrize(
    ("width", "height", "expected"),
    [
        (1080, 1920, False),
        (720, 1280, False),
        (1920, 1080, True),
        (1080, 1080, True),
        (1080, 1910, False),
        (1080, 1800, True),
    ],
)
def test_needs_reframe_aspect_ratios(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    width: int,
    height: int,
    expected: bool,
) -> None:
    monkeypatch.setattr(
        "kliptych.orchestrator.subprocess.run",
        _probe_runner(f"width={width}\nheight={height}\nduration=2.000000\n"),
    )
    assert _needs_reframe(tmp_path / "clip.mp4", RenderConfig()) is expected


def test_probe_video_reads_dimensions_and_duration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "kliptych.orchestrator.subprocess.run",
        _probe_runner("width=640\nheight=360\nduration=3.500000\n"),
    )
    info = _probe_video(tmp_path / "clip.mp4", render=RenderConfig())
    assert info.width == 640
    assert info.height == 360
    assert info.duration_s == pytest.approx(3.5)


@pytest.mark.parametrize(
    ("rotation_line", "expected_width", "expected_height"),
    [
        ("rotation=90\n", 1080, 1920),
        ("rotation=270\n", 1080, 1920),
        ("rotation=-90\n", 1080, 1920),
        ("rotation=0\n", 1920, 1080),
        ("rotation=abc\n", 1920, 1080),
    ],
)
def test_probe_video_swaps_dimensions_for_rotation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    rotation_line: str,
    expected_width: int,
    expected_height: int,
) -> None:
    monkeypatch.setattr(
        "kliptych.orchestrator.subprocess.run",
        _probe_runner(f"width=1920\nheight=1080\n{rotation_line}duration=2.000000\n"),
    )
    info = _probe_video(tmp_path / "clip.mp4", render=RenderConfig())
    assert info.width == expected_width
    assert info.height == expected_height
    assert info.duration_s == pytest.approx(2.0)


@pytest.mark.parametrize(
    ("stdout", "error"),
    [
        ("", "dimensiones"),
        ("width=abc\nheight=240\nduration=1.0\n", "dimensiones"),
        ("width=320\nheight=240\nduration=N/A\n", "duración"),
        ("width=320\nheight=240\n", "duración"),
        ("width=320\nheight=0\nduration=1.0\n", "dimensiones"),
    ],
)
def test_probe_video_rejects_malformed_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stdout: str, error: str
) -> None:
    monkeypatch.setattr("kliptych.orchestrator.subprocess.run", _probe_runner(stdout))
    with pytest.raises(PipelineError, match=error):
        _ = _probe_video(tmp_path / "clip.mp4", render=RenderConfig())


def test_probe_video_ignores_lines_without_separator(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "kliptych.orchestrator.subprocess.run",
        _probe_runner("ruido\nwidth=320\nheight=240\nduration=1.000000\n"),
    )
    info = _probe_video(tmp_path / "clip.mp4", render=RenderConfig())
    assert info.width == 320
    assert info.height == 240
    assert info.duration_s == pytest.approx(1.0)


def test_probe_video_missing_binary(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def run(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        _ = (argv, kwargs)
        message = "ffprobe"
        raise FileNotFoundError(message)

    monkeypatch.setattr("kliptych.orchestrator.subprocess.run", run)
    with pytest.raises(PipelineError, match="no está disponible"):
        _ = _probe_video(tmp_path / "clip.mp4", render=RenderConfig())


def test_probe_video_timeout(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def run(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        _ = (argv, kwargs)
        raise subprocess.TimeoutExpired(cmd="ffprobe", timeout=1.0)

    monkeypatch.setattr("kliptych.orchestrator.subprocess.run", run)
    with pytest.raises(PipelineError, match="timeout"):
        _ = _probe_video(tmp_path / "clip.mp4", render=RenderConfig())


def test_probe_video_os_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def run(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        _ = (argv, kwargs)
        message = "permiso"
        raise OSError(message)

    monkeypatch.setattr("kliptych.orchestrator.subprocess.run", run)
    with pytest.raises(PipelineError, match="no se pudo ejecutar"):
        _ = _probe_video(tmp_path / "clip.mp4", render=RenderConfig())


def test_probe_video_nonzero(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def run(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        _ = kwargs
        return subprocess.CompletedProcess(args=argv, returncode=2, stdout="", stderr="boom")

    monkeypatch.setattr("kliptych.orchestrator.subprocess.run", run)
    with pytest.raises(PipelineError, match="código 2"):
        _ = _probe_video(tmp_path / "clip.mp4", render=RenderConfig())


def test_full_video_selection_spans_whole_video(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "kliptych.orchestrator.subprocess.run",
        _probe_runner("width=1080\nheight=1920\nduration=4.250000\n"),
    )
    selection = _full_video_selection(tmp_path / "source.mp4", render=RenderConfig())
    assert selection.segments == (Segment(start_s=0.0, end_s=4.25),)
    assert selection.rationale


def test_full_video_selection_rejects_zero_duration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "kliptych.orchestrator.subprocess.run",
        _probe_runner("width=1080\nheight=1920\nduration=0.000000\n"),
    )
    with pytest.raises(PipelineError, match="duración"):
        _ = _full_video_selection(tmp_path / "source.mp4", render=RenderConfig())


def test_publish_copies_to_final(tmp_path: Path) -> None:
    video = tmp_path / "clip.mp4"
    _ = video.write_bytes(b"clip-bytes")
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    registry = _cleanup_registry()
    final = _publish(video, output_dir=output_dir, registry=registry)
    assert final == output_dir / "final.mp4"
    assert final.read_bytes() == b"clip-bytes"
    assert registry.paths


def test_resolve_dependencies_repost_without_face_model(tmp_path: Path) -> None:
    config = _config(tmp_path, repost_mode=True)
    dependencies = _resolve_dependencies(
        config=config,
        detector=None,
        transcriber=None,
        selector=None,
        reframer=None,
        subtitle_renderer=None,
    )
    assert dependencies.reframer is None


def test_repost_horizontal_without_reframer_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness = _Harness(probe_stdout="width=1920\nheight=1080\nduration=2.000000\n")
    _install(monkeypatch, harness)
    config = PipelineConfig(
        output_dir=tmp_path / "out",
        contract=_contract(),
        render=RenderConfig(),
        repost_mode=True,
    )
    with pytest.raises(PipelineError, match="reframer"):
        _ = run_long_video(
            _URL,
            model=_Model(harness.events),
            config=config,
            detector=_Detector(harness.events),
            transcriber=_Transcriber(harness.events),
            selector=_Selector(harness.events),
            subtitle_renderer=_SubtitleRenderer(harness.events),
        )


def _probe_runner(stdout: str) -> Callable[..., object]:
    def run(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        _ = kwargs
        return subprocess.CompletedProcess(args=argv, returncode=0, stdout=stdout, stderr="")

    return run
