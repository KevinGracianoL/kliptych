"""Fix 2: pista de audio externa con fingerprint y revalidación en --resume."""

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import cast, override

import pytest

from kliptych.contract import Contract, Segment
from kliptych.encoding import RenderConfig
from kliptych.moments import ChatMessage, Moment, MomentDetector, MomentSource
from kliptych.orchestrator import (
    LongVideoModel,
    PipelineConfig,
    PipelineResult,
    Reframer,
    SubtitleBurner,
    compute_long_video_fingerprint,
    run_long_video,
)
from kliptych.reframe import ReframeResult, ReframeTarget
from kliptych.segment import SegmentSelection, SegmentSelector
from kliptych.transcribe import Transcriber, Transcript, Word

_URL = "https://example.com/video"
_AUDIO_URL = "https://example.com/audio.mp3"


@dataclass
class _Counter:
    download_calls: int = 0
    audio_downloads: int = 0
    transcribe_calls: int = 0
    subtitles_calls: int = 0


@dataclass
class _Box:
    video: bytes
    audio: bytes


def _contract() -> Contract:
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


def _ffmpeg_run(argv: list[str], **kwargs: object) -> object:
    _ = kwargs
    if argv and argv[0] == "ffprobe":
        return SimpleNamespace(
            args=argv,
            returncode=0,
            stdout="width=320\nheight=240\nduration=2.0\n",
            stderr="",
        )
    out_path = Path(argv[-1])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _ = out_path.write_bytes(b"ffmpeg_output")
    return SimpleNamespace(
        args=argv,
        returncode=0,
        stdout="width=320\nheight=240\nduration=2.0\n",
        stderr="",
    )


class _Downloader:
    def __init__(self, counter: _Counter, box: _Box) -> None:
        self._counter: _Counter = counter
        self._box: _Box = box

    def download_video(
        self, *, url: str, destination: Path, format_selector: str | None = None
    ) -> Path:
        _ = format_selector
        if "audio" in url:
            self._counter.audio_downloads += 1
            _ = destination.write_bytes(self._box.audio)
        else:
            self._counter.download_calls += 1
            _ = destination.write_bytes(self._box.video)
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
    @override
    def detect(
        self,
        video: Path,
        *,
        transcript: Transcript | None = None,
        chat: Sequence[ChatMessage] | None = None,
    ) -> tuple[Moment, ...]:
        _ = (video, transcript, chat)
        return (Moment(start_s=0.0, end_s=1.0, score=0.8, source=MomentSource.FUSED),)


class _FakeModel:
    def select_segments(self, prompt: Mapping[str, object]) -> object:
        _ = (self, prompt)
        return {"segments": [{"start_s": 0.0, "end_s": 1.0}], "rationale": "resumen"}


class _FakeSelector(SegmentSelector):
    @override
    def build_prompt(
        self, transcript: Transcript, moments: tuple[Moment, ...], contract: Contract
    ) -> dict[str, object]:
        _ = (transcript, moments, contract)
        return {}

    @override
    def parse_response(self, raw: object) -> SegmentSelection:
        _ = raw
        return SegmentSelection(segments=(Segment(start_s=0.0, end_s=1.0),), rationale="r")


class _FakeReframer(Reframer):
    @override
    def analyze(self, video: Path) -> ReframeResult:
        _ = video
        return ReframeResult(
            targets=(ReframeTarget(x=0, y=0, width=134, height=240),),
            source_width=320,
            source_height=240,
        )

    @override
    def render(self, *, video: Path, destination: Path, result: ReframeResult) -> Path:
        _ = (video, result)
        _ = destination.write_bytes(b"reframed_bytes")
        return destination


class _FakeSubtitles(SubtitleBurner):
    def __init__(self, counter: _Counter) -> None:
        self._counter: _Counter = counter

    @override
    def write(self, words: Sequence[Word], destination: Path) -> Path:
        _ = words
        self._counter.subtitles_calls += 1
        _ = destination.write_text("Dialogue", encoding="utf-8")
        return destination

    @override
    def burn(
        self, *, video: Path, subtitles: Path, destination: Path, mute_audio: bool = False
    ) -> Path:
        _ = (video, subtitles, mute_audio)
        _ = destination.write_bytes(b"final_video_bytes")
        return destination


def _setup(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, box: _Box
) -> tuple[PipelineConfig, _Counter]:
    counter = _Counter()

    def _factory(*, timeout_s: float, max_size_bytes: int) -> _Downloader:
        _ = (timeout_s, max_size_bytes)
        return _Downloader(counter, box)

    monkeypatch.setattr("kliptych.orchestrator.subprocess.run", _ffmpeg_run)
    monkeypatch.setattr("kliptych.orchestrator.MediaDownloader", _factory)
    config = PipelineConfig(
        output_dir=tmp_path / "out",
        contract=_contract(),
        render=RenderConfig(),
        audio_locked=True,
        audio_track_url=_AUDIO_URL,
    )
    return config, counter


def _run(config: PipelineConfig, counter: _Counter) -> PipelineResult:
    return run_long_video(
        _URL,
        config=config,
        model=cast("LongVideoModel", _FakeModel()),
        detector=_FakeDetector(),
        transcriber=_FakeTranscriber(counter),
        selector=_FakeSelector(),
        reframer=_FakeReframer(),
        subtitle_renderer=_FakeSubtitles(counter),
        resume=False,
    )


def _run_resume(config: PipelineConfig, counter: _Counter) -> PipelineResult:
    return run_long_video(
        _URL,
        config=config,
        model=cast("LongVideoModel", _FakeModel()),
        detector=_FakeDetector(),
        transcriber=_FakeTranscriber(counter),
        selector=_FakeSelector(),
        reframer=_FakeReframer(),
        subtitle_renderer=_FakeSubtitles(counter),
        resume=True,
    )


def test_audio_remote_change_invalidates_downstream(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    box = _Box(video=b"video-v1", audio=b"audio-v1")
    config, counter = _setup(monkeypatch, tmp_path, box)
    result1 = _run(config, counter)
    assert isinstance(result1, PipelineResult)
    audio_path = config.output_dir / "audio_track.mp3"
    assert audio_path.read_bytes() == b"audio-v1"
    assert counter.audio_downloads == 1

    box.audio = b"audio-v2-changed"
    result2 = _run_resume(config, counter)
    assert isinstance(result2, PipelineResult)
    assert audio_path.read_bytes() == b"audio-v2-changed"
    assert counter.audio_downloads == 2
    assert counter.subtitles_calls == 2


def test_audio_remote_unchanged_reuses_without_reexecution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    box = _Box(video=b"video-v1", audio=b"stable-audio")
    config, counter = _setup(monkeypatch, tmp_path, box)
    _ = _run(config, counter)
    assert counter.subtitles_calls == 1
    _ = _run_resume(config, counter)
    assert counter.audio_downloads == 2
    assert counter.subtitles_calls == 1


def test_audio_fingerprint_includes_content_hash(tmp_path: Path) -> None:
    track = tmp_path / "track.mp3"
    _ = track.write_bytes(b"audio-v1")
    config = PipelineConfig(
        output_dir=tmp_path / "out",
        contract=_contract(),
        render=RenderConfig(),
        audio_locked=True,
        audio_track_path=track,
    )
    fp1 = compute_long_video_fingerprint(_URL, config=config)
    _ = track.write_bytes(b"audio-v2-different")
    fp2 = compute_long_video_fingerprint(_URL, config=config)
    assert fp1 != fp2


def _setup_local(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, audio_bytes: bytes
) -> tuple[PipelineConfig, _Counter]:
    counter = _Counter()
    box = _Box(video=b"video-v1", audio=b"unused-remote")

    def _factory(*, timeout_s: float, max_size_bytes: int) -> _Downloader:
        _ = (timeout_s, max_size_bytes)
        return _Downloader(counter, box)

    monkeypatch.setattr("kliptych.orchestrator.subprocess.run", _ffmpeg_run)
    monkeypatch.setattr("kliptych.orchestrator.MediaDownloader", _factory)
    track = tmp_path / "track.mp3"
    _ = track.write_bytes(audio_bytes)
    config = PipelineConfig(
        output_dir=tmp_path / "out",
        contract=_contract(),
        render=RenderConfig(),
        audio_locked=True,
        audio_track_path=track,
    )
    return config, counter


def test_local_audio_change_invalidates_on_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, counter = _setup_local(monkeypatch, tmp_path, b"audio-v1")
    result1 = _run(config, counter)
    assert isinstance(result1, PipelineResult)
    assert counter.subtitles_calls == 1
    work_copy = config.output_dir / "audio_track.mp3"
    assert work_copy.is_file()
    assert work_copy.read_bytes() == b"audio-v1"

    track = tmp_path / "track.mp3"
    _ = track.write_bytes(b"audio-v2-changed")
    result2 = _run_resume(config, counter)
    assert isinstance(result2, PipelineResult)
    assert work_copy.read_bytes() == b"audio-v2-changed"
    assert counter.subtitles_calls == 2


def test_local_audio_missing_hash_invalidates_on_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, counter = _setup_local(monkeypatch, tmp_path, b"stable-audio")
    _ = _run(config, counter)
    assert counter.subtitles_calls == 1
    checkpoint_path = config.output_dir / "checkpoint.json"
    payload = cast("dict[str, object]", json.loads(checkpoint_path.read_text(encoding="utf-8")))
    payload["audio_content_hash"] = None
    _ = checkpoint_path.write_text(json.dumps(payload), encoding="utf-8")
    _ = _run_resume(config, counter)
    assert counter.subtitles_calls == 2
