"""Tests de los hallazgos 5 y 6: consistencia de --resume.

El checkpoint reutilizado debe coincidir con el nombre esperado según la
cardinalidad del lote (final.mp4 para 1 segmento, final_00.mp4... para varios)
y una fuente remota se re-descarga antes de validar el source local, de modo
que un cambio de bytes en el servidor invalida las etapas hijas.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from typing import cast, override

import pytest

from kliptych.contract import Contract, Segment, TimestampRange
from kliptych.encoding import RenderConfig
from kliptych.moments import ChatMessage, Moment, MomentDetector, MomentSource
from kliptych.orchestrator import (
    LongVideoModel,
    PipelineConfig,
    PipelineResult,
    Reframer,
    SubtitleBurner,
    run_audio_locked,
    run_long_video,
    run_repost,
)
from kliptych.reframe import ReframeResult, ReframeTarget
from kliptych.segment import SegmentSelection, SegmentSelector
from kliptych.transcribe import Transcriber, Transcript, Word

_URL = "https://example.com/video"


@dataclass
class _Counter:
    download_calls: int = 0
    transcribe_calls: int = 0
    detect_calls: int = 0
    select_calls: int = 0
    reframe_calls: int = 0
    subtitles_calls: int = 0


@dataclass
class _PayloadBox:
    data: bytes


@dataclass(frozen=True, slots=True)
class _ResumeDeps:
    model: LongVideoModel
    detector: MomentDetector
    transcriber: Transcriber
    selector: SegmentSelector
    reframer: Reframer
    subtitle_renderer: SubtitleBurner


def _resume_contract() -> Contract:
    return Contract.model_validate(
        {
            "schema_version": "1.1",
            "campaign_id": "camp-resume",
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


_last_ffmpeg_duration: list[str] = ["2.0"]


def _ffmpeg_run(argv: list[str], **kwargs: object) -> object:
    _ = kwargs
    if argv and "ffprobe" in argv[0]:
        return SimpleNamespace(
            args=argv,
            returncode=0,
            stdout=f"width=320\nheight=240\nduration={_last_ffmpeg_duration[0]}\n",
            stderr="",
        )
    if "-t" in argv:
        _last_ffmpeg_duration[0] = argv[argv.index("-t") + 1]
    out_path = Path(argv[-1])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _ = out_path.write_bytes(b"ffmpeg_output")
    return SimpleNamespace(
        args=argv,
        returncode=0,
        stdout=f"width=320\nheight=240\nduration={_last_ffmpeg_duration[0]}\n",
        stderr="",
    )


class _FakeDownloader:
    def __init__(self, counter: _Counter, box: _PayloadBox) -> None:
        self._counter: _Counter = counter
        self._box: _PayloadBox = box

    def download_video(
        self,
        *,
        url: str,
        destination: Path,
        format_selector: str | None = None,
        section: tuple[float, float] | None = None,
    ) -> Path:
        _ = (url, format_selector, section)
        self._counter.download_calls += 1
        _ = destination.write_bytes(self._box.data)
        return destination


class _FakeTranscriber(Transcriber):
    def __init__(self, counter: _Counter) -> None:
        self._counter: _Counter = counter

    @override
    def transcribe(self, audio: Path) -> Transcript:
        _ = audio
        self._counter.transcribe_calls += 1
        words = (
            Word(start_s=0.0, end_s=0.5, text="hola", confidence=0.9, token_id=0),
            Word(start_s=0.5, end_s=1.0, text="mundo", confidence=0.9, token_id=1),
        )
        return Transcript(words=words, language="es", duration_s=1.0, text="hola mundo")


class _FakeDetector(MomentDetector):
    def __init__(self, counter: _Counter) -> None:
        self._counter: _Counter = counter

    @override
    def detect(
        self,
        video: Path,
        *,
        transcript: Transcript | None = None,
        chat: Sequence[ChatMessage] | None = None,
    ) -> tuple[Moment, ...]:
        _ = (video, transcript, chat)
        self._counter.detect_calls += 1
        return (Moment(start_s=0.0, end_s=1.0, score=0.8, source=MomentSource.FUSED),)


class _FakeModel:
    def select_segments(self, prompt: Mapping[str, object]) -> object:
        _ = (self, prompt)
        return {"segments": [{"start_s": 0.0, "end_s": 1.0}], "rationale": "resumen"}


class _FakeSelector(SegmentSelector):
    def __init__(self, counter: _Counter) -> None:
        self._counter: _Counter = counter

    @override
    def build_prompt(
        self, transcript: Transcript, moments: tuple[Moment, ...], contract: Contract
    ) -> dict[str, object]:
        _ = (transcript, moments, contract)
        self._counter.select_calls += 1
        return {}

    @override
    def parse_response(self, raw: object) -> SegmentSelection:
        _ = raw
        return SegmentSelection(segments=(Segment(start_s=0.0, end_s=1.0),), rationale="resumen")


class _FakeReframer(Reframer):
    def __init__(self, counter: _Counter) -> None:
        self._counter: _Counter = counter

    @override
    def analyze(self, video: Path) -> ReframeResult:
        _ = video
        self._counter.reframe_calls += 1
        return ReframeResult(
            targets=(ReframeTarget(x=0.0, y=0.0, width=134 / 320, height=240 / 240),),
            source_width=320,
            source_height=240,
        )

    @override
    def render(self, *, video: Path, destination: Path, result: ReframeResult) -> Path:
        _ = (video, result)
        _ = destination.write_bytes(b"reframed_bytes")
        return destination


class _FakeSubtitleRenderer(SubtitleBurner):
    def __init__(self, counter: _Counter) -> None:
        self._counter: _Counter = counter

    @override
    def write(self, words: Sequence[Word], destination: Path) -> Path:
        _ = words
        self._counter.subtitles_calls += 1
        _ = destination.write_text("Dialogue: 0,0:00:00.00,0:00:01.00", encoding="utf-8")
        return destination

    @override
    def burn(
        self, *, video: Path, subtitles: Path, destination: Path, mute_audio: bool = False
    ) -> Path:
        _ = (video, subtitles, mute_audio)
        _ = destination.write_bytes(b"final_video_bytes")
        return destination


def _setup_resume(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, box: _PayloadBox
) -> tuple[PipelineConfig, _ResumeDeps, _Counter]:
    counter = _Counter()

    def _downloader_factory(*, timeout_s: float, max_size_bytes: int) -> _FakeDownloader:
        _ = (timeout_s, max_size_bytes)
        return _FakeDownloader(counter, box)

    monkeypatch.setattr("kliptych.orchestrator.subprocess.run", _ffmpeg_run)
    monkeypatch.setattr("kliptych.orchestrator.MediaDownloader", _downloader_factory)
    config = PipelineConfig(
        output_dir=tmp_path / "out",
        contract=_resume_contract(),
        render=RenderConfig(),
    )
    deps = _ResumeDeps(
        model=cast("LongVideoModel", _FakeModel()),
        detector=_FakeDetector(counter),
        transcriber=_FakeTranscriber(counter),
        selector=_FakeSelector(counter),
        reframer=_FakeReframer(counter),
        subtitle_renderer=_FakeSubtitleRenderer(counter),
    )
    return config, deps, counter


def _run_resume(config: PipelineConfig, deps: _ResumeDeps, *, resume: bool) -> PipelineResult:
    return run_long_video(
        _URL,
        config=config,
        model=deps.model,
        detector=deps.detector,
        transcriber=deps.transcriber,
        selector=deps.selector,
        reframer=deps.reframer,
        subtitle_renderer=deps.subtitle_renderer,
        resume=resume,
    )


def test_resume_single_to_multi_segment_uses_indexed_finals(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """--resume de 1 a 2 segmentos produce final_00/final_01 sin mezclar final.mp4."""
    config, deps, _ = _setup_resume(monkeypatch, tmp_path, _PayloadBox(b"source-bytes"))
    result1 = _run_resume(config, deps, resume=False)
    assert result1.final_video.name == "final.mp4"
    assert (config.output_dir / "final.mp4").is_file()

    selection2 = SegmentSelection(
        segments=(Segment(start_s=0.0, end_s=1.0), Segment(start_s=1.0, end_s=2.0)),
        rationale="r2",
    )

    def _parse_selection_2(raw: object) -> SegmentSelection:
        _ = raw
        return selection2

    monkeypatch.setattr(deps.selector, "parse_response", _parse_selection_2)
    _ = (config.output_dir / "selection.json").write_text(
        selection2.model_dump_json(indent=2), encoding="utf-8"
    )
    result2 = _run_resume(config, deps, resume=True)

    first = config.output_dir / "final_00.mp4"
    second = config.output_dir / "final_01.mp4"
    assert first.is_file()
    assert second.is_file()
    assert not (config.output_dir / "final.mp4").exists()
    assert result2.final_videos == (first, second)


def test_resume_multi_to_single_segment_removes_indexed_finals(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """--resume de 2 a 1 segmento produce final.mp4 sin huérfanos final_00/final_01."""
    config, deps, _ = _setup_resume(monkeypatch, tmp_path, _PayloadBox(b"source-bytes"))
    selection2 = SegmentSelection(
        segments=(Segment(start_s=0.0, end_s=1.0), Segment(start_s=1.0, end_s=2.0)),
        rationale="r2",
    )

    def _parse_selection_2(raw: object) -> SegmentSelection:
        _ = raw
        return selection2

    monkeypatch.setattr(deps.selector, "parse_response", _parse_selection_2)
    result1 = _run_resume(config, deps, resume=False)
    assert result1.final_videos == (
        config.output_dir / "final_00.mp4",
        config.output_dir / "final_01.mp4",
    )
    assert (config.output_dir / "final_00.mp4").is_file()
    assert (config.output_dir / "final_01.mp4").is_file()

    selection1 = SegmentSelection(
        segments=(Segment(start_s=0.0, end_s=1.0),),
        rationale="r1",
    )

    def _parse_selection_1(raw: object) -> SegmentSelection:
        _ = raw
        return selection1

    monkeypatch.setattr(deps.selector, "parse_response", _parse_selection_1)
    _ = (config.output_dir / "selection.json").write_text(
        selection1.model_dump_json(indent=2), encoding="utf-8"
    )
    result2 = _run_resume(config, deps, resume=True)

    final = config.output_dir / "final.mp4"
    assert final.is_file()
    assert not (config.output_dir / "final_00.mp4").exists()
    assert not (config.output_dir / "final_01.mp4").exists()
    assert result2.final_videos == (final,)


def test_resume_remote_source_revalidates_and_invalidates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """--resume re-descarga la URL y si los bytes cambian invalida etapas hijas."""
    box = _PayloadBox(b"server-bytes-v1")
    config, deps, counter = _setup_resume(monkeypatch, tmp_path, box)
    result1 = _run_resume(config, deps, resume=False)
    assert isinstance(result1, PipelineResult)
    assert counter.download_calls == 1
    assert counter.transcribe_calls == 1
    assert (config.output_dir / "source.mp4").read_bytes() == b"server-bytes-v1"

    box.data = b"server-bytes-v2-changed"
    result2 = _run_resume(config, deps, resume=True)
    assert isinstance(result2, PipelineResult)
    assert counter.download_calls == 2
    assert (config.output_dir / "source.mp4").read_bytes() == b"server-bytes-v2-changed"
    assert counter.transcribe_calls == 2
    assert counter.detect_calls == 2
    assert counter.reframe_calls == 2
    assert counter.subtitles_calls == 2


def test_resume_remote_source_unchanged_reuses_stages(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """--resume con los mismos bytes remotos no re-ejecuta las etapas hijas."""
    config, deps, counter = _setup_resume(monkeypatch, tmp_path, _PayloadBox(b"stable-bytes"))
    _ = _run_resume(config, deps, resume=False)
    _ = _run_resume(config, deps, resume=True)
    assert counter.download_calls == 2
    assert counter.transcribe_calls == 1
    assert counter.reframe_calls == 1
    assert counter.subtitles_calls == 1


def test_run_audio_locked_and_repost_full_pipeline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """run_audio_locked y run_repost ejecutan el pipeline con su modo activo."""
    config, deps, _ = _setup_resume(monkeypatch, tmp_path, _PayloadBox(b"source-bytes"))
    track = tmp_path / "track.mp3"
    _ = track.write_bytes(b"audio-bytes")
    audio_config = replace(config, audio_locked=True, audio_track_path=track)
    audio_result = run_audio_locked(
        _URL,
        model=deps.model,
        config=audio_config,
        detector=deps.detector,
        transcriber=deps.transcriber,
        selector=deps.selector,
        reframer=deps.reframer,
        subtitle_renderer=deps.subtitle_renderer,
    )
    assert isinstance(audio_result, PipelineResult)
    assert audio_result.final_video.is_file()

    repost_config = replace(config, repost_mode=True)
    repost_result = run_repost(
        _URL,
        model=deps.model,
        config=repost_config,
        detector=deps.detector,
        transcriber=deps.transcriber,
        selector=deps.selector,
        reframer=deps.reframer,
        subtitle_renderer=deps.subtitle_renderer,
    )
    assert isinstance(repost_result, PipelineResult)
    assert repost_result.transcript is None
    assert repost_result.final_video.is_file()


def test_resume_invalidates_on_timestamp_range_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """--resume invalida las etapas y vuelve a descargar si el contrato cambia timestamp_ranges."""
    config, deps, counter = _setup_resume(monkeypatch, tmp_path, _PayloadBox(b"stable-bytes"))
    _ = _run_resume(config, deps, resume=False)
    assert counter.download_calls == 1
    assert counter.transcribe_calls == 1

    new_contract = config.contract.model_copy(
        update={"timestamp_ranges": (TimestampRange(start_sec=10.0, end_sec=25.0),)}
    )
    config_changed = replace(config, contract=new_contract)
    _ = _run_resume(config_changed, deps, resume=True)
    assert counter.download_calls == 2
    assert counter.transcribe_calls == 2


def test_resume_skip_remote_revalidation_env_skips_revalidation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """KLIPTYCH_SKIP_REMOTE_REVALIDATION=1 omite _revalidate_remote_source en resume."""
    box = _PayloadBox(b"server-bytes-v1")
    config, deps, counter = _setup_resume(monkeypatch, tmp_path, box)
    _ = _run_resume(config, deps, resume=False)
    assert counter.download_calls == 1
    assert (config.output_dir / "source.mp4").read_bytes() == b"server-bytes-v1"

    box.data = b"server-bytes-v2-changed"
    monkeypatch.setenv("KLIPTYCH_SKIP_REMOTE_REVALIDATION", "1")
    revalidate_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def _spy_revalidate(*args: object, **kwargs: object) -> Path:
        revalidate_calls.append((args, kwargs))
        source_arg = args[1] if len(args) > 1 else None
        assert isinstance(source_arg, Path)
        return source_arg

    monkeypatch.setattr(
        "kliptych.orchestrator._revalidate_remote_source",
        _spy_revalidate,
    )
    result2 = _run_resume(config, deps, resume=True)

    assert isinstance(result2, PipelineResult)
    assert revalidate_calls == []
    assert counter.download_calls == 1
    assert (config.output_dir / "source.mp4").read_bytes() == b"server-bytes-v1"
    assert counter.transcribe_calls == 1
