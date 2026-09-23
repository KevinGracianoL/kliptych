"""Tests unitarios de detección de momentos: ffmpeg simulado, cero red y cero subprocesos.

La salida de ffmpeg se inyecta como texto controlado; ningún test ejecuta el
binario real (eso vive en ``test_moments_integration.py``).
"""

import subprocess
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import cast

import pytest
from pydantic import ValidationError

from kliptych import moments
from kliptych.moments import (
    ChatMessage,
    DetectionConfig,
    FFmpegMomentDetector,
    Moment,
    MomentDetectionError,
    MomentSource,
)
from kliptych.transcribe import Transcript, Word


# Los helpers puros son privados del módulo; se acceden por ``getattr`` con un
# nombre variable para no ensanchar la superficie pública solo por testearlos.
def _private(name: str) -> object:
    return cast("object", getattr(moments, name))


def _method(detector: FFmpegMomentDetector, name: str) -> object:
    return cast("object", getattr(detector, name))


_parse_scene_scores = cast(
    "Callable[[str], tuple[tuple[float, float], ...]]", _private("_parse_scene_scores")
)
_parse_rms_levels = cast(
    "Callable[[str], tuple[tuple[float, float], ...]]", _private("_parse_rms_levels")
)
_db_to_linear = cast("Callable[[str], float]", _private("_db_to_linear"))
_normalize = cast("Callable[[Sequence[float]], tuple[float, ...]]", _private("_normalize"))
_energy_bins = cast(
    "Callable[[Sequence[tuple[float, float]], float], tuple[float, ...]]", _private("_energy_bins")
)
_chat_bins = cast(
    "Callable[[Sequence[ChatMessage], float], tuple[float, ...]]", _private("_chat_bins")
)
_window_moments = cast("Callable[..., tuple[Moment, ...]]", _private("_window_moments"))
_scene_moments = cast("Callable[..., tuple[Moment, ...]]", _private("_scene_moments"))
_fuse = cast("Callable[..., tuple[Moment, ...]]", _private("_fuse"))


def _energy_frames(detector: FFmpegMomentDetector, video: Path) -> tuple[tuple[float, float], ...]:
    method = cast(
        "Callable[[Path], tuple[tuple[float, float], ...]]",
        _method(detector, "_energy_frames"),
    )
    return method(video)


def _scene_events(detector: FFmpegMomentDetector, video: Path) -> tuple[tuple[float, float], ...]:
    method = cast(
        "Callable[[Path], tuple[tuple[float, float], ...]]",
        _method(detector, "_scene_events"),
    )
    return method(video)


_FFMPEG = "ffmpeg"

_SCENE_OUTPUT = (
    "frame:0    pts:10240   pts_time:1\n"
    "lavfi.scd.score=20.000\n"
    "frame:1    pts:20480   pts_time:2\n"
    "lavfi.scd.score=40.000\n"
)

_RMS_OUTPUT = (
    "frame:0    pts:0       pts_time:0\n"
    "lavfi.astats.Overall.RMS_level=-20.000000\n"
    "frame:1    pts:44100   pts_time:1\n"
    "lavfi.astats.Overall.RMS_level=-inf\n"
)

_Call = tuple[list[str], dict[str, object]]
_FakeRun = Callable[..., subprocess.CompletedProcess[str]]


def _completed(
    *,
    stdout: str = "",
    stderr: str = "",
    returncode: int = 0,
) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(
        args=[_FFMPEG], returncode=returncode, stdout=stdout, stderr=stderr
    )


def _fake_run(
    calls: list[_Call],
    *,
    scene: str = "",
    rms: str = "",
    returncode: int = 0,
    stderr: str = "",
) -> _FakeRun:
    def run(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append((argv, kwargs))
        if any("astats" in argument for argument in argv):
            return _completed(stdout=rms, stderr=stderr, returncode=returncode)
        if any("scdet" in argument for argument in argv):
            return _completed(stdout=scene, stderr=stderr, returncode=returncode)
        return _completed(stderr=stderr, returncode=returncode)

    return run


def _raising_run(exc: BaseException) -> _FakeRun:
    def run(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        _ = (argv, kwargs)
        raise exc

    return run


def _video(tmp_path: Path) -> Path:
    path = tmp_path / "video.mp4"
    _ = path.write_bytes(b"video")
    return path


def _word(start: float, end: float, text: str) -> Word:
    return Word(start_s=start, end_s=end, text=text, confidence=0.9, token_id=0)


def _transcript(duration_s: float) -> Transcript:
    return Transcript(
        words=(_word(0.0, 0.5, "hola"),),
        language="es",
        duration_s=duration_s,
        text="hola",
    )


def test_parse_scene_scores_pairs_time_and_score() -> None:
    assert _parse_scene_scores(_SCENE_OUTPUT) == ((1.0, 20.0), (2.0, 40.0))


def test_parse_scene_scores_ignores_dangling_lines() -> None:
    text = "lavfi.scd.score=5\npts_time:1\nsin score\n"
    assert _parse_scene_scores(text) == ()


def test_parse_rms_levels_converts_db_to_linear() -> None:
    frames = _parse_rms_levels(_RMS_OUTPUT)
    assert frames[0][0] == pytest.approx(0.0)
    assert frames[0][1] == pytest.approx(0.1)
    assert frames[1] == (pytest.approx(1.0), 0.0)


def test_parse_rms_levels_ignores_dangling_lines() -> None:
    assert _parse_rms_levels("pts_time:1\nsin nivel\n") == ()


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("-inf", 0.0), ("0", 1.0), ("-20", 0.1), ("nan", 0.0)],
)
def test_db_to_linear(raw: str, expected: float) -> None:
    assert _db_to_linear(raw) == pytest.approx(expected)


def test_normalize_scales_by_maximum() -> None:
    assert _normalize([0.0, 0.5, 1.0]) == (0.0, 0.5, 1.0)


def test_normalize_all_zero_is_zero() -> None:
    assert _normalize([0.0, 0.0]) == (0.0, 0.0)


def test_normalize_empty_is_empty() -> None:
    assert _normalize([]) == ()


def test_scene_moments_use_cut_scores_and_boundaries() -> None:
    events = ((1.0, 20.0), (2.0, 40.0))
    moments = _scene_moments(events, duration_s=3.0)
    assert [(m.start_s, m.end_s, m.score, m.source) for m in moments] == [
        (1.0, 2.0, 0.5, MomentSource.SCENE),
        (2.0, 3.0, 1.0, MomentSource.SCENE),
    ]


def test_scene_moments_empty_without_events_or_duration() -> None:
    assert _scene_moments((), duration_s=3.0) == ()
    assert _scene_moments(((1.0, 1.0),), duration_s=0.0) == ()


def test_scene_moments_ignore_cuts_at_video_edges() -> None:
    events = ((0.0, 10.0), (3.0, 10.0))
    assert _scene_moments(events, duration_s=3.0) == ()


def test_energy_bins_average_frames_per_window() -> None:
    frames = ((0.0, 0.1), (0.5, 0.3), (1.0, 0.0))
    assert _energy_bins(frames, 1.0) == (pytest.approx(0.2), 0.0)


def test_energy_bins_empty_is_empty() -> None:
    assert _energy_bins((), 1.0) == ()


def test_chat_bins_count_messages_per_window() -> None:
    messages = (
        ChatMessage(timestamp_s=0.0, text="a"),
        ChatMessage(timestamp_s=0.5, text="b"),
        ChatMessage(timestamp_s=1.0, text="c"),
        ChatMessage(timestamp_s=2.5, text="d"),
    )
    assert _chat_bins(messages, 2.0) == (1.5, 0.5)


def test_chat_bins_empty_is_empty() -> None:
    assert _chat_bins((), 2.0) == ()


def test_window_moments_merge_adjacent_bins() -> None:
    moments = _window_moments(
        (0.5, 0.5, 0.0, 1.0),
        window_s=1.0,
        duration_s=4.0,
        min_score=0.5,
        source=MomentSource.AUDIO_ENERGY,
    )
    assert [(m.start_s, m.end_s, m.score) for m in moments] == [(0.0, 2.0, 0.5), (3.0, 4.0, 1.0)]


def test_window_moments_clamp_to_duration() -> None:
    moments = _window_moments(
        (1.0,),
        window_s=2.0,
        duration_s=1.5,
        min_score=0.5,
        source=MomentSource.CHAT_DENSITY,
    )
    assert [(m.start_s, m.end_s) for m in moments] == [(0.0, 1.5)]


def test_window_moments_below_threshold_are_dropped() -> None:
    moments = _window_moments(
        (1.0, 0.1),
        window_s=1.0,
        duration_s=2.0,
        min_score=0.5,
        source=MomentSource.AUDIO_ENERGY,
    )
    assert [(m.start_s, m.end_s, m.score) for m in moments] == [(0.0, 1.0, 1.0)]


def test_fuse_combines_overlapping_sources() -> None:
    moments = (
        Moment(start_s=0.0, end_s=2.0, score=1.0, source=MomentSource.SCENE),
        Moment(start_s=0.0, end_s=2.0, score=0.5, source=MomentSource.AUDIO_ENERGY),
    )
    fused = _fuse(moments, weights={MomentSource.SCENE: 0.5, MomentSource.AUDIO_ENERGY: 0.5})
    assert len(fused) == 1
    assert fused[0].source is MomentSource.FUSED
    assert fused[0].score == pytest.approx(0.75)


def test_fuse_keeps_disjoint_moments_separate() -> None:
    moments = (
        Moment(start_s=0.0, end_s=1.0, score=1.0, source=MomentSource.SCENE),
        Moment(start_s=2.0, end_s=3.0, score=1.0, source=MomentSource.AUDIO_ENERGY),
    )
    fused = _fuse(moments, weights={MomentSource.SCENE: 0.5, MomentSource.AUDIO_ENERGY: 0.5})
    assert len(fused) == 2
    assert all(moment.score == pytest.approx(1.0) for moment in fused)


def test_fuse_sorts_by_score_descending() -> None:
    moments = (
        Moment(start_s=0.0, end_s=1.0, score=0.2, source=MomentSource.SCENE),
        Moment(start_s=2.0, end_s=3.0, score=0.9, source=MomentSource.SCENE),
    )
    fused = _fuse(moments, weights={MomentSource.SCENE: 1.0})
    assert [moment.score for moment in fused] == pytest.approx([0.9, 0.2])


def test_fuse_empty_is_empty() -> None:
    assert _fuse((), weights={MomentSource.SCENE: 1.0}) == ()


def test_missing_video_fails(tmp_path: Path) -> None:
    with pytest.raises(MomentDetectionError, match="no existe"):
        _ = FFmpegMomentDetector().detect(tmp_path / "nope.mp4")


def test_energy_frames_use_argv_without_shell(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[_Call] = []
    monkeypatch.setattr("kliptych.moments.subprocess.run", _fake_run(calls, rms=_RMS_OUTPUT))
    frames = _energy_frames(FFmpegMomentDetector(), _video(tmp_path))
    assert frames[0][1] == pytest.approx(0.1)
    argv, kwargs = calls[0]
    assert argv[0] == _FFMPEG
    assert any("astats" in argument for argument in argv)
    assert kwargs == {"capture_output": True, "text": True, "timeout": 300.0, "check": False}


def test_scene_events_use_argv_without_shell(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[_Call] = []
    monkeypatch.setattr("kliptych.moments.subprocess.run", _fake_run(calls, scene=_SCENE_OUTPUT))
    events = _scene_events(FFmpegMomentDetector(), _video(tmp_path))
    assert events == ((1.0, 20.0), (2.0, 40.0))
    argv, _ = calls[0]
    assert argv[0] == _FFMPEG
    assert any("scdet" in argument for argument in argv)


def test_detect_fuses_all_sources(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "kliptych.moments.subprocess.run",
        _fake_run([], scene=_SCENE_OUTPUT, rms=_RMS_OUTPUT),
    )
    chat = (
        ChatMessage(timestamp_s=0.0, text="a"),
        ChatMessage(timestamp_s=0.1, text="b"),
    )
    moments = FFmpegMomentDetector().detect(
        _video(tmp_path),
        transcript=_transcript(3.0),
        chat=chat,
    )
    assert moments
    assert all(moment.source is MomentSource.FUSED for moment in moments)
    assert all(0.0 <= moment.score <= 1.0 for moment in moments)
    assert [moment.score for moment in moments] == sorted(
        (moment.score for moment in moments), reverse=True
    )


def test_detect_without_inputs_returns_empty(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("kliptych.moments.subprocess.run", _fake_run([]))
    assert FFmpegMomentDetector().detect(_video(tmp_path)) == ()


def test_detect_without_transcript_uses_last_frame_time(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "kliptych.moments.subprocess.run",
        _fake_run([], scene=_SCENE_OUTPUT, rms=_RMS_OUTPUT),
    )
    moments = FFmpegMomentDetector().detect(_video(tmp_path))
    assert moments


def test_missing_ffmpeg_becomes_detection_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "kliptych.moments.subprocess.run",
        _raising_run(FileNotFoundError("ffmpeg")),
    )
    with pytest.raises(MomentDetectionError, match="no está disponible"):
        _ = FFmpegMomentDetector().detect(_video(tmp_path))


def test_timeout_becomes_detection_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "kliptych.moments.subprocess.run",
        _raising_run(subprocess.TimeoutExpired(cmd="ffmpeg", timeout=300.0)),
    )
    with pytest.raises(MomentDetectionError, match="timeout"):
        _ = FFmpegMomentDetector().detect(_video(tmp_path))


def test_os_error_becomes_detection_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "kliptych.moments.subprocess.run",
        _raising_run(OSError("permiso denegado")),
    )
    with pytest.raises(MomentDetectionError, match="no se pudo ejecutar"):
        _ = FFmpegMomentDetector().detect(_video(tmp_path))


def test_nonzero_returncode_becomes_detection_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "kliptych.moments.subprocess.run",
        _fake_run([], returncode=1, stderr="boom"),
    )
    with pytest.raises(MomentDetectionError, match="falló con código 1"):
        _ = FFmpegMomentDetector().detect(_video(tmp_path))


@pytest.mark.parametrize("timeout_s", [0.0, -1.0])
def test_invalid_timeout_rejected(timeout_s: float) -> None:
    with pytest.raises(ValueError, match="timeout"):
        _ = FFmpegMomentDetector(timeout_s=timeout_s)


def test_invalid_window_rejected() -> None:
    with pytest.raises(ValueError, match="ventana"):
        _ = DetectionConfig(window_s=0.0)


def test_invalid_scene_threshold_rejected() -> None:
    with pytest.raises(ValueError, match="umbral de escena"):
        _ = DetectionConfig(scene_threshold=-1.0)


def test_invalid_scene_change_rejected() -> None:
    with pytest.raises(ValueError, match="cambio de escena"):
        _ = DetectionConfig(scene_change=1.5)


def test_invalid_energy_min_score_rejected() -> None:
    with pytest.raises(ValueError, match="energía"):
        _ = DetectionConfig(energy_min_score=1.5)


def test_invalid_chat_min_score_rejected() -> None:
    with pytest.raises(ValueError, match="chat"):
        _ = DetectionConfig(chat_min_score=-0.1)


def test_negative_weight_rejected() -> None:
    with pytest.raises(ValueError, match="pesos negativos"):
        _ = DetectionConfig(weights={MomentSource.SCENE: -1.0})


def test_moment_rejects_score_above_one() -> None:
    with pytest.raises(ValidationError):
        _ = Moment(start_s=0.0, end_s=1.0, score=1.5, source=MomentSource.SCENE)


def test_moment_rejects_negative_start() -> None:
    with pytest.raises(ValidationError):
        _ = Moment(start_s=-0.1, end_s=1.0, score=0.5, source=MomentSource.SCENE)


def test_moment_rejects_unordered_bounds() -> None:
    with pytest.raises(ValidationError, match="no puede superar"):
        _ = Moment(start_s=2.0, end_s=1.0, score=0.5, source=MomentSource.SCENE)


def test_moment_rejects_non_finite_bounds() -> None:
    with pytest.raises(ValidationError, match="finitas"):
        _ = Moment(start_s=0.0, end_s=float("inf"), score=0.5, source=MomentSource.SCENE)


def test_moment_rejects_extra_fields() -> None:
    with pytest.raises(ValidationError):
        _ = Moment.model_validate(
            {"start_s": 0.0, "end_s": 1.0, "score": 0.5, "source": "scene", "extra": 1}
        )


def test_moment_is_frozen() -> None:
    moment = Moment(start_s=0.0, end_s=1.0, score=0.5, source=MomentSource.SCENE)
    with pytest.raises(ValidationError):
        moment.score = 0.9


def test_chat_message_rejects_negative_timestamp() -> None:
    with pytest.raises(ValidationError):
        _ = ChatMessage(timestamp_s=-0.1, text="hola")


def test_chat_message_rejects_empty_text() -> None:
    with pytest.raises(ValidationError):
        _ = ChatMessage(timestamp_s=0.0, text="")


def test_chat_message_rejects_extra_fields() -> None:
    with pytest.raises(ValidationError):
        _ = ChatMessage.model_validate({"timestamp_s": 0.0, "text": "hola", "author": "x"})
