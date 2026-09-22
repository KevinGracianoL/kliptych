"""Tests de la detección de entorno."""

import locale
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

import pytest

from kliptych.environment import (
    CommandResult,
    GpuInfo,
    SubprocessRunner,
    detect_environment,
)

_FFMPEG_VERSION = ("ffmpeg", "-version")
_FFPROBE_VERSION = ("ffprobe", "-version")
_FFMPEG_ENCODERS = ("ffmpeg", "-hide_banner", "-encoders")
_NVIDIA_SMI = (
    "nvidia-smi",
    "--query-gpu=name,memory.total,driver_version",
    "--format=csv,noheader,nounits",
)

_FFMPEG_LINE = "ffmpeg version N-118380-gca3550948c Copyright (c) 2000-2025\n"
_FFPROBE_LINE = "ffprobe version N-118380-gca3550948c Copyright (c) 2007-2025\n"
_ENCODERS_OUTPUT = " V....D h264_nvenc NVIDIA NVENC H.264 encoder (codec h264)\n"
_GPU_LINE = "NVIDIA GeForce GTX 1650 Ti, 4096, 610.74\n"


class FakeRunner:
    def __init__(self, results: Mapping[tuple[str, ...], CommandResult]) -> None:
        self._results: dict[tuple[str, ...], CommandResult] = dict(results)
        self.calls: list[tuple[str, ...]] = []
        self.timeouts: list[float] = []

    def set_result(self, argv: tuple[str, ...], result: CommandResult) -> None:
        self._results[argv] = result

    def run(self, argv: Sequence[str], *, timeout_s: float = 15.0) -> CommandResult:
        call = tuple(argv)
        self.calls.append(call)
        self.timeouts.append(timeout_s)
        return self._results.get(call, CommandResult(ok=False, stderr="no configurado"))


def _full_runner() -> FakeRunner:
    return FakeRunner(
        {
            _FFMPEG_VERSION: CommandResult(ok=True, stdout=_FFMPEG_LINE),
            _FFPROBE_VERSION: CommandResult(ok=True, stdout=_FFPROBE_LINE),
            _FFMPEG_ENCODERS: CommandResult(ok=True, stdout=_ENCODERS_OUTPUT),
            _NVIDIA_SMI: CommandResult(ok=True, stdout=_GPU_LINE),
        }
    )


def test_detect_full_environment() -> None:
    report = detect_environment(_full_runner())
    assert report.ffmpeg_version == "N-118380-gca3550948c"
    assert report.ffprobe_version == "N-118380-gca3550948c"
    assert report.nvenc_available is True
    assert report.gpu == GpuInfo(
        name="NVIDIA GeForce GTX 1650 Ti",
        vram_mib=4096,
        driver_version="610.74",
    )
    assert report.degradations == ()


def test_missing_tools_produce_degradations() -> None:
    report = detect_environment(FakeRunner({}))
    assert report.ffmpeg_version is None
    assert report.ffprobe_version is None
    assert report.nvenc_available is False
    assert report.gpu is None
    joined = " | ".join(report.degradations)
    assert "ffmpeg" in joined
    assert "ffprobe" in joined
    assert "GPU" in joined


def test_ffmpeg_without_nvenc_degrades_render() -> None:
    runner = _full_runner()
    runner.set_result(_FFMPEG_ENCODERS, CommandResult(ok=True, stdout=" V..... libx264\n"))
    report = detect_environment(runner)
    assert report.nvenc_available is False
    assert any("NVENC" in degradation for degradation in report.degradations)


def test_encoder_listing_failure_is_not_nvenc() -> None:
    runner = _full_runner()
    runner.set_result(_FFMPEG_ENCODERS, CommandResult(ok=False, stderr="boom"))
    report = detect_environment(runner)
    assert report.nvenc_available is False


def test_malformed_gpu_line_is_ignored() -> None:
    runner = _full_runner()
    runner.set_result(_NVIDIA_SMI, CommandResult(ok=True, stdout="algo raro\n"))
    report = detect_environment(runner)
    assert report.gpu is None
    assert any("GPU" in degradation for degradation in report.degradations)


def test_subprocess_runner_reports_success() -> None:
    result = SubprocessRunner().run([sys.executable, "-c", "print('hola')"])
    assert result.ok is True
    assert result.stdout.strip() == "hola"


def test_subprocess_runner_missing_binary() -> None:
    result = SubprocessRunner().run(["binario-que-no-existe-kliptych"])
    assert result.ok is False
    assert result.stderr


def test_subprocess_runner_timeout() -> None:
    result = SubprocessRunner().run(
        [sys.executable, "-c", "import time; time.sleep(5)"],
        timeout_s=0.2,
    )
    assert result.ok is False


def _undecodable_byte() -> bytes | None:
    encoding = locale.getencoding()
    for value in range(256):
        try:
            _ = bytes([value]).decode(encoding)
        except UnicodeDecodeError:
            return bytes([value])
    return None


def test_subprocess_runner_survives_undecodable_output(tmp_path: Path) -> None:
    bad = _undecodable_byte()
    if bad is None:
        pytest.skip("el codec local decodifica todos los bytes")
    script = tmp_path / "emit_bad_byte.py"
    line = b"sys.stdout.buffer.write(b'hola \\x" + f"{bad[0]:02x}".encode() + b" fin')\n"
    _ = script.write_bytes(b"import sys\n" + line)
    result = SubprocessRunner().run([sys.executable, str(script)])
    assert result.ok is True
    assert "\ufffd" in result.stdout


@pytest.mark.integration
def test_real_environment_is_consistent() -> None:
    report = detect_environment(SubprocessRunner())
    if report.ffmpeg_version is None:
        assert any("ffmpeg" in degradation for degradation in report.degradations)
    if report.gpu is None:
        assert any("GPU" in degradation for degradation in report.degradations)
