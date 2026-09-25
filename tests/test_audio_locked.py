"""Tests unitarios del modo audio obligatorio (D1-PR1).

Descarga, transcripción, momentos, modelo, selector, reframe y subtítulos se
inyectan como fakes y ffmpeg se simula: cero red y cero ffmpeg real. Se verifica
la configuración, los argumentos de reemplazo y mezcla, la resolución de la
pista (existente, ausente o por URL), la limpieza de temporales de audio y la
compatibilidad hacia atrás con ``audio_locked=False``.
"""

import subprocess
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Protocol, cast

import pytest

from kliptych import orchestrator
from kliptych.contract import Contract, Segment
from kliptych.encoding import RenderConfig, audio_injection_arguments
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

_URL = "https://example.com/video"
_AUDIO_URL = "https://cdn.example.com/track.mp3"


class _Registry(Protocol):
    """Superficie del registro de limpieza que usan los tests."""

    paths: list[Path]

    def register(self, path: Path) -> Path: ...


def _private(name: str) -> object:
    return cast("object", getattr(orchestrator, name))


_cleanup_registry = cast("Callable[[], _Registry]", _private("_CleanupRegistry"))
_inject_audio = cast("Callable[..., Path]", _private("_inject_audio"))
_resolve_audio_track = cast("Callable[..., Path]", _private("_resolve_audio_track"))


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


def _config(
    tmp_path: Path,
    *,
    audio_locked: bool = False,
    audio_track_path: Path | None = None,
    audio_track_url: str | None = None,
    audio_mix_ratio: float = 1.0,
) -> PipelineConfig:
    return PipelineConfig(
        output_dir=tmp_path / "out",
        contract=_contract(),
        render=RenderConfig(),
        audio_locked=audio_locked,
        audio_track_path=audio_track_path,
        audio_track_url=audio_track_url,
        audio_mix_ratio=audio_mix_ratio,
    )


class _FakeDownloader:
    """Descargador acotado que materializa el destino sin tocar la red."""

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
        return {"campaign_id": "camp-01"}

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


def _ffmpeg(events: list[str], captured: list[list[str]]) -> Callable[..., object]:
    def run(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        _ = kwargs
        if str(argv[0]).lower().endswith("ffprobe") or "ffprobe" in str(argv[0]).lower():
            return subprocess.CompletedProcess(
                args=argv,
                returncode=0,
                stdout="width=1080\nheight=1920\nduration=10.000\n",
                stderr="",
            )
        if "-ss" in argv:
            tag = "cut"
        elif "-filter_complex" in argv:
            tag = "inject_mix"
        else:
            tag = "inject_replace"
        events.append(tag)
        captured.append(list(argv))
        _ = Path(argv[-1]).write_bytes(b"render")
        return subprocess.CompletedProcess(args=argv, returncode=0, stdout="", stderr="")

    return run


def _install(
    monkeypatch: pytest.MonkeyPatch,
    events: list[str],
    captured: list[list[str]],
) -> None:
    def factory(*, timeout_s: float, max_size_bytes: int) -> _FakeDownloader:
        _ = (timeout_s, max_size_bytes)
        return _FakeDownloader(events)

    monkeypatch.setattr("kliptych.orchestrator.MediaDownloader", factory)
    monkeypatch.setattr("kliptych.orchestrator.subprocess.run", _ffmpeg(events, captured))


def _run(events: list[str], config: PipelineConfig) -> PipelineResult:
    return run_long_video(
        _URL,
        model=_Model(events),
        config=config,
        detector=_Detector(events),
        transcriber=_Transcriber(events),
        selector=_Selector(events),
        reframer=_Reframer(events),
        subtitle_renderer=_SubtitleRenderer(events),
    )


def _leftovers(output_dir: Path) -> list[str]:
    return [path.name for path in output_dir.iterdir() if ".part-" in path.name]


def test_pipeline_config_audio_defaults() -> None:
    config = _config(Path("out"))
    assert config.audio_locked is False
    assert config.audio_track_path is None
    assert config.audio_track_url is None
    assert config.audio_mix_ratio == pytest.approx(1.0)


@pytest.mark.parametrize("ratio", [-0.1, 1.1, 2.0])
def test_pipeline_config_rejects_mix_ratio_out_of_range(ratio: float) -> None:
    with pytest.raises(ValueError, match="audio_mix_ratio"):
        _ = _config(Path("out"), audio_mix_ratio=ratio)


def test_audio_injection_arguments_replace() -> None:
    argv = audio_injection_arguments(mix_ratio=1.0)
    assert argv[argv.index("-map") + 1] == "0:v:0"
    assert argv[argv.index("-map", argv.index("-map") + 1) + 1] == "1:a:0"
    assert "-filter_complex" not in argv
    assert "copy" in argv
    assert "-shortest" not in argv


def test_audio_injection_arguments_mix() -> None:
    argv = audio_injection_arguments(mix_ratio=0.5)
    filter_graph = argv[argv.index("-filter_complex") + 1]
    assert "amix=inputs=2:duration=first:dropout_transition=2[aout]" in filter_graph
    assert "duration=longest" not in filter_graph
    assert "aformat=sample_fmts=fltp:channel_layouts=stereo" in filter_graph
    assert (
        "[0:a]aformat=sample_fmts=fltp:channel_layouts=stereo,volume=0.500[original]"
        in filter_graph
    )
    assert (
        "[1:a]aformat=sample_fmts=fltp:channel_layouts=stereo,volume=0.500[external]"
        in filter_graph
    )
    assert argv[argv.index("-map", argv.index("-map") + 1) + 1] == "[aout]"


def test_resolve_audio_track_missing_file_raises(tmp_path: Path) -> None:
    config = _config(
        tmp_path,
        audio_locked=True,
        audio_track_path=tmp_path / "missing.mp3",
    )
    with pytest.raises(PipelineError, match="no existe"):
        _ = _resolve_audio_track(
            config,
            downloader=_FakeDownloader([]),
            registry=_cleanup_registry(),
        )


def test_resolve_audio_track_requires_source(tmp_path: Path) -> None:
    config = _config(tmp_path, audio_locked=True)
    with pytest.raises(PipelineError, match="audio_track_path"):
        _ = _resolve_audio_track(
            config,
            downloader=_FakeDownloader([]),
            registry=_cleanup_registry(),
        )


def test_resolve_audio_track_rejects_both_sources(tmp_path: Path) -> None:
    track = tmp_path / "track.mp3"
    _ = track.write_bytes(b"audio")
    config = _config(
        tmp_path,
        audio_locked=True,
        audio_track_path=track,
        audio_track_url=_AUDIO_URL,
    )
    with pytest.raises(PipelineError, match="no ambos"):
        _ = _resolve_audio_track(
            config,
            downloader=_FakeDownloader([]),
            registry=_cleanup_registry(),
        )


def test_resolve_audio_track_downloads_url(tmp_path: Path) -> None:
    events: list[str] = []
    config = _config(tmp_path, audio_locked=True, audio_track_url=_AUDIO_URL)
    registry = _cleanup_registry()
    track = _resolve_audio_track(
        config,
        downloader=_FakeDownloader(events),
        registry=registry,
    )
    assert track.is_file()
    assert events == [f"download:{_AUDIO_URL}"]
    assert track in registry.paths


def test_inject_audio_builds_replace_argv(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    captured: list[list[str]] = []
    monkeypatch.setattr("kliptych.orchestrator.subprocess.run", _ffmpeg([], captured))
    video = tmp_path / "reframed.mp4"
    _ = video.write_bytes(b"video")
    track = tmp_path / "track.mp3"
    _ = track.write_bytes(b"audio")
    registry = _cleanup_registry()
    config = _config(tmp_path, audio_locked=True, audio_track_path=track)
    destination = _inject_audio(
        video,
        config=config,
        downloader=_FakeDownloader([]),
        registry=registry,
    )
    argv = captured[0]
    assert argv.count("-i") == 2
    assert argv[argv.index("-map", argv.index("-map") + 1) + 1] == "1:a:0"
    assert "-shortest" not in argv
    assert "-t" in argv
    assert argv[argv.index("-t") + 1] == "10.000"
    assert destination.is_file()
    assert destination in registry.paths


def test_inject_audio_rejects_shorter_external_audio(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_probe(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        _ = kwargs
        target = Path(argv[-1])
        duration = "6.000" if target.suffix == ".mp4" else "2.000"
        return subprocess.CompletedProcess(
            args=argv,
            returncode=0,
            stdout=f"width=1080\nheight=1920\nduration={duration}\n",
            stderr="",
        )

    monkeypatch.setattr("kliptych.orchestrator.subprocess.run", fake_probe)
    video = tmp_path / "reframed.mp4"
    _ = video.write_bytes(b"video")
    track = tmp_path / "track.mp3"
    _ = track.write_bytes(b"audio")
    config = _config(tmp_path, audio_locked=True, audio_track_path=track)
    with pytest.raises(PipelineError, match="El audio externo es más corto que el video"):
        _ = _inject_audio(
            video,
            config=config,
            downloader=_FakeDownloader([]),
            registry=_cleanup_registry(),
        )


def test_inject_audio_accepts_longer_audio_without_shortest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: list[list[str]] = []

    def fake_run(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        _ = kwargs
        if str(argv[0]).lower().endswith("ffprobe") or "ffprobe" in str(argv[0]).lower():
            target = Path(argv[-1])
            duration = "6.000" if target.suffix == ".mp4" else "8.000"
            return subprocess.CompletedProcess(
                args=argv,
                returncode=0,
                stdout=f"width=1080\nheight=1920\nduration={duration}\n",
                stderr="",
            )
        captured.append(list(argv))
        _ = Path(argv[-1]).write_bytes(b"rendered")
        return subprocess.CompletedProcess(args=argv, returncode=0, stdout="", stderr="")

    monkeypatch.setattr("kliptych.orchestrator.subprocess.run", fake_run)
    video = tmp_path / "reframed.mp4"
    _ = video.write_bytes(b"video")
    track = tmp_path / "track.mp3"
    _ = track.write_bytes(b"audio")
    config = _config(tmp_path, audio_locked=True, audio_track_path=track)
    dest = _inject_audio(
        video,
        config=config,
        downloader=_FakeDownloader([]),
        registry=_cleanup_registry(),
    )
    assert dest.is_file()
    ffmpeg_argv = captured[0]
    assert "-shortest" not in ffmpeg_argv
    assert "-t" in ffmpeg_argv
    assert ffmpeg_argv[ffmpeg_argv.index("-t") + 1] == "6.000"


def test_inject_audio_builds_mix_argv(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    captured: list[list[str]] = []
    monkeypatch.setattr("kliptych.orchestrator.subprocess.run", _ffmpeg([], captured))
    video = tmp_path / "reframed.mp4"
    _ = video.write_bytes(b"video")
    track = tmp_path / "track.mp3"
    _ = track.write_bytes(b"audio")
    config = _config(tmp_path, audio_locked=True, audio_track_path=track, audio_mix_ratio=0.5)
    _ = _inject_audio(
        video,
        config=config,
        downloader=_FakeDownloader([]),
        registry=_cleanup_registry(),
    )
    argv = captured[0]
    assert "-filter_complex" in argv
    filter_graph = argv[argv.index("-filter_complex") + 1]
    assert "amix=inputs=2:duration=first:dropout_transition=2[aout]" in filter_graph
    assert "duration=longest" not in filter_graph
    assert "aformat=sample_fmts=fltp:channel_layouts=stereo" in filter_graph


def test_pipeline_without_audio_locked_is_unchanged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    events: list[str] = []
    captured: list[list[str]] = []
    _install(monkeypatch, events, captured)
    config = _config(tmp_path)
    result = _run(events, config)
    assert isinstance(result, PipelineResult)
    assert events == [
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
    assert len(result.cleaning) == 3
    assert _leftovers(tmp_path / "out") == []


def test_pipeline_with_audio_locked_replaces_audio(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    events: list[str] = []
    captured: list[list[str]] = []
    _install(monkeypatch, events, captured)
    track = tmp_path / "track.mp3"
    _ = track.write_bytes(b"audio")
    config = _config(tmp_path, audio_locked=True, audio_track_path=track)
    result = _run(events, config)
    assert "inject_replace" in events
    assert result.final_video.is_file()
    assert len(result.cleaning) == 4
    assert _leftovers(tmp_path / "out") == []


def test_cleanup_registry_captures_downloaded_audio(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    events: list[str] = []
    captured: list[list[str]] = []
    _install(monkeypatch, events, captured)
    config = _config(tmp_path, audio_locked=True, audio_track_url=_AUDIO_URL)
    result = _run(events, config)
    assert f"download:{_AUDIO_URL}" in events
    assert "inject_replace" in events
    assert len(result.cleaning) == 5
    assert _leftovers(tmp_path / "out") == []
