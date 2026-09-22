"""Tests del probe: unitarios con subprocess falso e integración con ffprobe."""

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from kliptych.gate import ProbeError
from kliptych.gate.probe import FFprobeProbe

_FFMPEG = shutil.which("ffmpeg")
_FFPROBE = shutil.which("ffprobe")


def _completed(stdout: str, returncode: int = 0) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(
        args=["ffprobe"],
        returncode=returncode,
        stdout=stdout,
        stderr="",
    )


def _media_file(tmp_path: Path) -> Path:
    target = tmp_path / "clip.mp4"
    _ = target.write_bytes(b"video")
    return target


def _fake_run(payload: str) -> object:
    def run(*_args: object, **_kwargs: object) -> subprocess.CompletedProcess[str]:
        return _completed(payload)

    return run


def test_probe_missing_file_fails(tmp_path: Path) -> None:
    with pytest.raises(ProbeError, match="no existe"):
        _ = FFprobeProbe().probe(tmp_path / "nope.mp4")


def test_parse_video_document(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    payload = json.dumps(
        {
            "streams": [
                {"codec_type": "video", "width": 1080, "height": 1920},
                {"codec_type": "audio"},
            ],
            "format": {"format_name": "mov,mp4", "duration": "12.5"},
        }
    )
    monkeypatch.setattr("kliptych.gate.probe.subprocess.run", _fake_run(payload))
    info = FFprobeProbe().probe(_media_file(tmp_path))
    assert info.has_video is True
    assert info.has_audio is True
    assert info.duration_s == pytest.approx(12.5)
    assert info.width == 1080
    assert info.height == 1920
    assert info.format_name == "mov,mp4"


def test_document_without_format_defaults_safely(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr("kliptych.gate.probe.subprocess.run", _fake_run(json.dumps({})))
    info = FFprobeProbe().probe(_media_file(tmp_path))
    assert info.has_video is False
    assert info.has_audio is False
    assert info.duration_s is None
    assert info.format_name == "desconocido"


def test_probe_invokes_ffprobe_with_exact_argv(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[list[str]] = []

    def run(argv: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(argv)
        return _completed(json.dumps({}))

    monkeypatch.setattr("kliptych.gate.probe.subprocess.run", run)
    target = _media_file(tmp_path)
    _ = FFprobeProbe(ffprobe="ffprobe").probe(target)
    assert calls == [
        [
            "ffprobe",
            "-v",
            "error",
            "-print_format",
            "json",
            "-show_format",
            "-show_streams",
            str(target),
        ]
    ]


def test_unexpected_os_error_becomes_probe_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def run(*_args: object, **_kwargs: object) -> subprocess.CompletedProcess[str]:
        raise PermissionError(13, "permiso denegado")

    monkeypatch.setattr("kliptych.gate.probe.subprocess.run", run)
    with pytest.raises(ProbeError, match="no se pudo ejecutar ffprobe"):
        _ = FFprobeProbe().probe(_media_file(tmp_path))


def test_invalid_json_fails(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr("kliptych.gate.probe.subprocess.run", _fake_run("no soy json"))
    with pytest.raises(ProbeError, match="inválida"):
        _ = FFprobeProbe().probe(_media_file(tmp_path))


def test_nonzero_exit_fails(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def run(*_args: object, **_kwargs: object) -> subprocess.CompletedProcess[str]:
        return _completed("", returncode=1)

    monkeypatch.setattr("kliptych.gate.probe.subprocess.run", run)
    with pytest.raises(ProbeError, match="falló"):
        _ = FFprobeProbe().probe(_media_file(tmp_path))


def test_missing_binary_fails(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def run(*_args: object, **_kwargs: object) -> subprocess.CompletedProcess[str]:
        raise FileNotFoundError

    monkeypatch.setattr("kliptych.gate.probe.subprocess.run", run)
    with pytest.raises(ProbeError, match="no está disponible"):
        _ = FFprobeProbe().probe(_media_file(tmp_path))


def test_timeout_fails(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def run(*_args: object, **_kwargs: object) -> subprocess.CompletedProcess[str]:
        raise subprocess.TimeoutExpired(cmd="ffprobe", timeout=1)

    monkeypatch.setattr("kliptych.gate.probe.subprocess.run", run)
    with pytest.raises(ProbeError, match="timeout"):
        _ = FFprobeProbe().probe(_media_file(tmp_path))


@pytest.mark.integration
@pytest.mark.skipif(_FFPROBE is None, reason="ffprobe no disponible")
def test_probe_non_media_file_fails(tmp_path: Path) -> None:
    assert _FFPROBE is not None
    target = tmp_path / "nota.txt"
    _ = target.write_text("no soy un video")
    with pytest.raises(ProbeError, match="ffprobe"):
        _ = FFprobeProbe(ffprobe=_FFPROBE).probe(target)


@pytest.mark.integration
@pytest.mark.skipif(
    _FFMPEG is None or _FFPROBE is None,
    reason="ffmpeg/ffprobe no disponibles",
)
def test_probe_reads_generated_video(tmp_path: Path) -> None:
    assert _FFMPEG is not None
    assert _FFPROBE is not None
    target = tmp_path / "clip.mp4"
    argv = [
        _FFMPEG,
        "-y",
        "-v",
        "error",
        "-f",
        "lavfi",
        "-i",
        "color=c=black:s=108x192:d=1.5",
        "-f",
        "lavfi",
        "-i",
        "anullsrc=r=44100:cl=mono",
        "-shortest",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        str(target),
    ]
    _ = subprocess.run(argv, capture_output=True, check=True, timeout=120)
    info = FFprobeProbe(ffprobe=_FFPROBE).probe(target)
    assert info.has_video is True
    assert info.has_audio is True
    assert info.duration_s == pytest.approx(1.5, abs=0.3)
    assert info.width == 108
    assert info.height == 192
