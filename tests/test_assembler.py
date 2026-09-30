"""Tests del ensamblado given_clips: unitarios con subprocess falso e integración."""

import json
import shutil
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import cast

import pytest

from kliptych.assembler import AssembleError, FFmpegAssembler
from kliptych.gate.probe import FFprobeProbe

_FFMPEG = shutil.which("ffmpeg")
_FFPROBE = shutil.which("ffprobe")

_DEFAULT_TIMEOUT_S = 300.0
_SQUARE = "scale=trunc(iw*sar/2)*2:ih,setsar=1"
_SCALE = (
    f"{_SQUARE},scale=1080:1920:force_original_aspect_ratio=decrease,"
    "pad=1080:1920:(ow-iw)/2:(oh-ih)/2:color=black,setsar=1"
)

_Call = tuple[list[str], dict[str, object]]
_FakeRun = Callable[..., subprocess.CompletedProcess[str]]


def _completed(returncode: int = 0, stderr: str = "") -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(
        args=["ffmpeg"], returncode=returncode, stdout="", stderr=stderr
    )


def _file(tmp_path: Path, name: str) -> Path:
    path = tmp_path / name
    _ = path.write_bytes(b"contenido")
    return path


def _assert_invocation(
    kwargs: dict[str, object], *, timeout_s: float, expect_cwd: bool = False
) -> None:
    if expect_cwd:
        cwd = kwargs.pop("cwd", None)
        assert cwd is not None
        assert Path(str(cwd)).is_dir()
    assert kwargs == {
        "capture_output": True,
        "text": True,
        "timeout": timeout_s,
        "check": False,
    }


def _fake_run(
    calls: list[_Call],
    *,
    payload: bytes = b"artefacto",
    returncode: int = 0,
    stderr: str = "",
    timeout_s: float = _DEFAULT_TIMEOUT_S,
    expect_cwd: bool = False,
) -> _FakeRun:
    def run(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append((argv, kwargs))
        _assert_invocation(dict(kwargs), timeout_s=timeout_s, expect_cwd=expect_cwd)
        _ = Path(argv[-1]).write_bytes(payload)
        return _completed(returncode=returncode, stderr=stderr)

    return run


def _timeout_run(
    calls: list[_Call],
    *,
    payload: bytes = b"parcial",
    timeout_s: float = _DEFAULT_TIMEOUT_S,
) -> _FakeRun:
    def run(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append((argv, kwargs))
        _assert_invocation(kwargs, timeout_s=timeout_s)
        _ = Path(argv[-1]).write_bytes(payload)
        raise subprocess.TimeoutExpired(cmd="ffmpeg", timeout=timeout_s)

    return run


def _failing_run(exc: BaseException, *, timeout_s: float = _DEFAULT_TIMEOUT_S) -> _FakeRun:
    def run(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        _ = argv
        _assert_invocation(kwargs, timeout_s=timeout_s)
        raise exc

    return run


def _no_partials(destination: Path) -> None:
    assert list(destination.parent.glob(f".{destination.stem}.part-*")) == []


def test_missing_clip_fails(tmp_path: Path) -> None:
    with pytest.raises(AssembleError, match="clip no existe"):
        _ = FFmpegAssembler().assemble(clip=tmp_path / "nope.mp4", destination=tmp_path / "out.mp4")


def test_missing_watermark_fails(tmp_path: Path) -> None:
    clip = _file(tmp_path, "clip.mp4")
    with pytest.raises(AssembleError, match="watermark no existe"):
        _ = FFmpegAssembler().assemble(
            clip=clip,
            destination=tmp_path / "out.mp4",
            watermark=tmp_path / "nope.png",
        )


def test_invalid_canvas_fails(tmp_path: Path) -> None:
    clip = _file(tmp_path, "clip.mp4")
    with pytest.raises(AssembleError, match="dimensiones"):
        _ = FFmpegAssembler().assemble(clip=clip, destination=tmp_path / "out.mp4", width=0)


def test_unpreparable_destination_fails(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    blocker = _file(tmp_path, "blocker.txt")
    calls: list[_Call] = []
    monkeypatch.setattr("kliptych.assembler.subprocess.run", _fake_run(calls))
    clip = _file(tmp_path, "clip.mp4")
    with pytest.raises(AssembleError, match="directorio del destino"):
        _ = FFmpegAssembler().assemble(clip=clip, destination=blocker / "out.mp4")
    assert calls == []


def test_invokes_ffmpeg_with_exact_argv(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls: list[_Call] = []
    monkeypatch.setattr("kliptych.assembler.subprocess.run", _fake_run(calls))
    clip = _file(tmp_path, "clip.mp4")
    destination = tmp_path / "run" / "piece.mp4"
    result = FFmpegAssembler(ffmpeg="ffmpeg").assemble(clip=clip, destination=destination)
    assert result == destination
    assert destination.read_bytes() == b"artefacto"
    _no_partials(destination)
    assert len(calls) == 1
    argv, _ = calls[0]
    output = Path(argv[-1])
    assert output.parent == destination.parent
    assert output.name.startswith(".piece.part-")
    assert output.suffix == ".mp4"
    assert argv[:-1] == [
        "ffmpeg",
        "-hide_banner",
        "-nostdin",
        "-v",
        "error",
        "-y",
        "-i",
        str(clip),
        "-vf",
        _SCALE,
        "-map",
        "0:v:0",
        "-map",
        "0:a?",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "20",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-movflags",
        "+faststart",
    ]


def test_invocation_contract_pins_timeout(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls: list[_Call] = []
    monkeypatch.setattr("kliptych.assembler.subprocess.run", _fake_run(calls, timeout_s=7.5))
    clip = _file(tmp_path, "clip.mp4")
    _ = FFmpegAssembler(ffmpeg="ffmpeg", timeout_s=7.5).assemble(
        clip=clip, destination=tmp_path / "piece.mp4"
    )
    assert len(calls) == 1


def test_watermark_argv_binds_watermark_input(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[_Call] = []
    monkeypatch.setattr("kliptych.assembler.subprocess.run", _fake_run(calls))
    clip = _file(tmp_path, "clip.mp4")
    watermark = _file(tmp_path, "wm.png")
    destination = tmp_path / "piece.mp4"
    _ = FFmpegAssembler(ffmpeg="ffmpeg").assemble(
        clip=clip,
        destination=destination,
        watermark=watermark,
    )
    assert len(calls) == 1
    argv, _ = calls[0]
    inputs = [argv[index + 1] for index, arg in enumerate(argv) if arg == "-i"]
    assert inputs == [str(clip), str(watermark)]
    assert argv[argv.index("-filter_complex") + 1] == (
        f"[0:v]{_SCALE}[base];[1:v]format=rgba,scale=216:-2[wm];[base][wm]overlay=W-w-20:20[v]"
    )
    assert argv[argv.index("-map") + 1] == "[v]"
    assert "-vf" not in argv
    assert "movie=" not in argv[argv.index("-filter_complex") + 1]


def test_missing_binary_fails(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr("kliptych.assembler.subprocess.run", _failing_run(FileNotFoundError()))
    clip = _file(tmp_path, "clip.mp4")
    with pytest.raises(AssembleError, match="no está disponible"):
        _ = FFmpegAssembler().assemble(clip=clip, destination=tmp_path / "out.mp4")


def test_timeout_keeps_previous_artifact(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls: list[_Call] = []
    monkeypatch.setattr("kliptych.assembler.subprocess.run", _timeout_run(calls))
    clip = _file(tmp_path, "clip.mp4")
    destination = tmp_path / "piece.mp4"
    _ = destination.write_bytes(b"artefacto-previo")
    with pytest.raises(AssembleError, match="timeout"):
        _ = FFmpegAssembler().assemble(clip=clip, destination=destination)
    assert destination.read_bytes() == b"artefacto-previo"
    _no_partials(destination)


def test_unexpected_os_error_fails(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        "kliptych.assembler.subprocess.run",
        _failing_run(PermissionError(13, "permiso denegado")),
    )
    clip = _file(tmp_path, "clip.mp4")
    destination = tmp_path / "piece.mp4"
    with pytest.raises(AssembleError, match="no se pudo ejecutar ffmpeg"):
        _ = FFmpegAssembler().assemble(clip=clip, destination=destination)
    _no_partials(destination)


def test_nonzero_exit_reports_stderr_and_keeps_previous_artifact(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[_Call] = []
    monkeypatch.setattr(
        "kliptych.assembler.subprocess.run",
        _fake_run(calls, payload=b"parcial", returncode=1, stderr="  Invalid data found  "),
    )
    clip = _file(tmp_path, "clip.mp4")
    destination = tmp_path / "piece.mp4"
    _ = destination.write_bytes(b"artefacto-previo")
    with pytest.raises(AssembleError, match="código 1: Invalid data found"):
        _ = FFmpegAssembler().assemble(clip=clip, destination=destination)
    assert destination.read_bytes() == b"artefacto-previo"
    _no_partials(destination)


def test_failed_assemble_leaves_no_partial_artifact(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[_Call] = []
    monkeypatch.setattr(
        "kliptych.assembler.subprocess.run",
        _fake_run(calls, payload=b"parcial", returncode=1, stderr="boom"),
    )
    clip = _file(tmp_path, "clip.mp4")
    destination = tmp_path / "piece.mp4"
    with pytest.raises(AssembleError, match="código 1"):
        _ = FFmpegAssembler().assemble(clip=clip, destination=destination)
    assert not destination.exists()
    _no_partials(destination)


def test_render_arguments_match_executed_argv(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[_Call] = []
    monkeypatch.setattr("kliptych.assembler.subprocess.run", _fake_run(calls))
    clip = _file(tmp_path, "clip.mp4")
    destination = tmp_path / "piece.mp4"
    assembler = FFmpegAssembler(ffmpeg="ffmpeg")
    recipe = assembler.render_arguments(clip=clip, destination=destination, watermark=None)
    _ = assembler.assemble(clip=clip, destination=destination)
    assert len(calls) == 1
    executed, _ = calls[0]
    assert list(recipe[:-1]) == executed[:-1]
    assert recipe[-1] == str(destination)
    assert executed[-1] != str(destination)


def test_assemble_with_subtitles_burns_ass_relative_to_output(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[_Call] = []
    monkeypatch.setattr("kliptych.assembler.subprocess.run", _fake_run(calls, expect_cwd=True))
    clip = _file(tmp_path, "clip.mp4")
    subtitles = _file(tmp_path, "song.ass")
    destination = tmp_path / "run" / "piece.mp4"
    result = FFmpegAssembler(ffmpeg="ffmpeg").assemble(
        clip=clip, destination=destination, subtitles=subtitles
    )
    assert result == destination
    assert len(calls) == 1
    argv, kwargs = calls[0]
    assert kwargs["cwd"] == Path(argv[-1]).parent
    assert Path(argv[-1]).is_absolute()
    vf = argv[argv.index("-vf") + 1]
    assert vf == _SCALE + ",subtitles=filename='../song.ass'"


def test_render_arguments_with_watermark_and_subtitles_chain_burn_after_overlay(
    tmp_path: Path,
) -> None:
    clip = _file(tmp_path, "clip.mp4")
    watermark = _file(tmp_path, "mark.png")
    subtitles = _file(tmp_path, "song.ass")
    destination = tmp_path / "run" / "piece.mp4"
    argv = FFmpegAssembler(ffmpeg="ffmpeg").render_arguments(
        clip=clip, destination=destination, watermark=watermark, subtitles=subtitles
    )
    graph = argv[argv.index("-filter_complex") + 1]
    assert "[base][wm]overlay=" in graph
    assert ";[v]subtitles=filename='../song.ass'[vout]" in graph
    assert "[vout]" in argv


def test_missing_subtitles_fails(tmp_path: Path) -> None:
    clip = _file(tmp_path, "clip.mp4")
    with pytest.raises(AssembleError, match="no existe"):
        _ = FFmpegAssembler().assemble(
            clip=clip,
            destination=tmp_path / "out.mp4",
            subtitles=tmp_path / "nope.ass",
        )


def _generate_clip(
    path: Path,
    *,
    width: int,
    height: int,
    duration: float,
    color: str = "blue",
    setsar: str | None = None,
) -> Path:
    assert _FFMPEG is not None
    argv = [
        _FFMPEG,
        "-y",
        "-v",
        "error",
        "-f",
        "lavfi",
        "-i",
        f"color=c={color}:s={width}x{height}:d={duration}",
        "-f",
        "lavfi",
        "-i",
        f"sine=frequency=440:duration={duration}",
    ]
    if setsar is not None:
        argv += ["-vf", f"setsar={setsar}"]
    argv += [
        "-shortest",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        str(path),
    ]
    _ = subprocess.run(argv, capture_output=True, check=True, timeout=120)
    return path


def _generate_watermark(path: Path) -> Path:
    assert _FFMPEG is not None
    argv = [
        _FFMPEG,
        "-y",
        "-v",
        "error",
        "-f",
        "lavfi",
        "-i",
        "color=c=white:s=64x64",
        "-frames:v",
        "1",
        str(path),
    ]
    _ = subprocess.run(argv, capture_output=True, check=True, timeout=120)
    return path


def _gray_frame(path: Path, *, width: int, height: int) -> bytes:
    assert _FFMPEG is not None
    argv = [
        _FFMPEG,
        "-v",
        "error",
        "-i",
        str(path),
        "-frames:v",
        "1",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "gray",
        "-",
    ]
    completed = subprocess.run(argv, capture_output=True, check=True, timeout=120)
    frame = completed.stdout
    assert len(frame) == width * height
    return frame


def _band_height(frame: bytes, *, width: int, height: int, threshold: int = 40) -> int:
    rows = [y for y in range(height) if max(frame[y * width : (y + 1) * width]) > threshold]
    return rows[-1] - rows[0] + 1


def _box_mean(frame: bytes, *, width: int, x0: int, y0: int, x1: int, y1: int) -> float:
    total = 0
    count = 0
    for y in range(y0, y1):
        row = frame[y * width + x0 : y * width + x1]
        total += sum(row)
        count += len(row)
    return total / count


@pytest.mark.integration
@pytest.mark.skipif(
    _FFMPEG is None or _FFPROBE is None,
    reason="ffmpeg/ffprobe no disponibles",
)
def test_assembles_vertical_artifact(tmp_path: Path) -> None:
    assert _FFMPEG is not None
    assert _FFPROBE is not None
    clip = _generate_clip(tmp_path / "clip.mp4", width=320, height=240, duration=1.5)
    destination = tmp_path / "piece.mp4"
    result = FFmpegAssembler(ffmpeg=_FFMPEG).assemble(clip=clip, destination=destination)
    assert result == destination
    info = FFprobeProbe(ffprobe=_FFPROBE).probe(destination)
    assert info.width == 1080
    assert info.height == 1920
    assert info.has_audio is True
    assert info.duration_s == pytest.approx(1.5, abs=0.3)


@pytest.mark.integration
@pytest.mark.skipif(
    _FFMPEG is None or _FFPROBE is None,
    reason="ffmpeg/ffprobe no disponibles",
)
def test_watermark_visible_in_top_right_box(tmp_path: Path) -> None:
    assert _FFMPEG is not None
    assert _FFPROBE is not None
    clip = _generate_clip(tmp_path / "clip.mp4", width=320, height=240, duration=1.0)
    watermark = _generate_watermark(tmp_path / "wm.png")
    plain = tmp_path / "plain.mp4"
    marked = tmp_path / "marked.mp4"
    _ = FFmpegAssembler(ffmpeg=_FFMPEG).assemble(clip=clip, destination=plain)
    _ = FFmpegAssembler(ffmpeg=_FFMPEG).assemble(
        clip=clip,
        destination=marked,
        watermark=watermark,
    )
    # 64x64 escalado al 20% de 1080 = 216 px, arriba a la derecha con margen 20.
    box = {"width": 1080, "x0": 844, "y0": 20, "x1": 1060, "y1": 236}
    plain_frame = _gray_frame(plain, width=1080, height=1920)
    marked_frame = _gray_frame(marked, width=1080, height=1920)
    assert _box_mean(plain_frame, **box) < 50
    assert _box_mean(marked_frame, **box) > 200
    marked_info = FFprobeProbe(ffprobe=_FFPROBE).probe(marked)
    assert marked_info.has_audio is True


@pytest.mark.integration
@pytest.mark.skipif(
    _FFMPEG is None or _FFPROBE is None,
    reason="ffmpeg/ffprobe no disponibles",
)
def test_anamorphic_clip_preserves_display_aspect(tmp_path: Path) -> None:
    assert _FFMPEG is not None
    clip = _generate_clip(
        tmp_path / "clip.mp4",
        width=720,
        height=480,
        duration=1.0,
        color="white",
        setsar="32/27",
    )
    destination = tmp_path / "piece.mp4"
    _ = FFmpegAssembler(ffmpeg=_FFMPEG).assemble(clip=clip, destination=destination)
    frame = _gray_frame(destination, width=1080, height=1920)
    assert _band_height(frame, width=1080, height=1920) == pytest.approx(608, abs=4)


@pytest.mark.integration
@pytest.mark.skipif(
    _FFMPEG is None or _FFPROBE is None,
    reason="ffmpeg/ffprobe no disponibles",
)
def test_real_timeout_keeps_previous_artifact(tmp_path: Path) -> None:
    assert _FFMPEG is not None
    clip = _generate_clip(tmp_path / "clip.mp4", width=640, height=360, duration=8.0)
    destination = tmp_path / "piece.mp4"
    _ = destination.write_bytes(b"artefacto-previo")
    with pytest.raises(AssembleError, match="timeout"):
        _ = FFmpegAssembler(ffmpeg=_FFMPEG, timeout_s=0.5).assemble(
            clip=clip,
            destination=destination,
        )
    assert destination.read_bytes() == b"artefacto-previo"
    _no_partials(destination)


def test_cut_exact_rejects_missing_file_or_invalid_bounds(tmp_path: Path) -> None:
    assembler = FFmpegAssembler()
    missing = tmp_path / "missing.mp4"
    dst = tmp_path / "dst.mp4"
    with pytest.raises(AssembleError, match="no existe"):
        _ = assembler.cut_exact(source=missing, destination=dst, start_s=0.0, duration_s=10.0)

    existing = _file(tmp_path, "existing.mp4")
    with pytest.raises(AssembleError, match="intervalo de corte"):
        _ = assembler.cut_exact(source=existing, destination=dst, start_s=-1.0, duration_s=10.0)
    with pytest.raises(AssembleError, match="intervalo de corte"):
        _ = assembler.cut_exact(source=existing, destination=dst, start_s=0.0, duration_s=0.0)


@pytest.mark.integration
@pytest.mark.skipif(
    _FFMPEG is None or _FFPROBE is None,
    reason="ffmpeg/ffprobe no disponibles",
)
def test_exact_cut_real_ffmpeg_integration(tmp_path: Path) -> None:
    assert _FFMPEG is not None
    assert _FFPROBE is not None
    source = tmp_path / "source_30s.mp4"
    argv_gen = [
        _FFMPEG,
        "-hide_banner",
        "-nostdin",
        "-v",
        "error",
        "-y",
        "-f",
        "lavfi",
        "-i",
        "testsrc=duration=30:size=320x240:rate=30",
        "-f",
        "lavfi",
        "-i",
        "sine=frequency=1000:duration=30",
        "-c:v",
        "libx264",
        "-c:a",
        "aac",
        str(source),
    ]
    _ = subprocess.run(argv_gen, check=True)

    destination = tmp_path / "cut_exact.mp4"
    assembler = FFmpegAssembler(ffmpeg=_FFMPEG)
    out = assembler.cut_exact(
        source=source,
        destination=destination,
        start_s=5.0,
        duration_s=10.0,
    )
    assert out.is_file()

    probe_cmd = [
        _FFPROBE,
        "-v",
        "error",
        "-show_entries",
        "format=duration:stream=codec_type,start_time",
        "-of",
        "json",
        str(out),
    ]
    res = subprocess.run(probe_cmd, capture_output=True, text=True, check=True)
    data = cast("dict[str, object]", json.loads(res.stdout))
    fmt = cast("dict[str, object]", data["format"])
    fmt_duration = float(cast("str", fmt["duration"]))
    assert fmt_duration == pytest.approx(10.0, abs=0.1)

    streams = cast("list[dict[str, object]]", data.get("streams", []))
    for stream in streams:
        start_time = float(cast("str", stream.get("start_time", "0.0")))
        assert start_time == pytest.approx(0.0, abs=0.05)
