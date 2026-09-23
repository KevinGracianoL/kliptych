"""Tests de descarga sin red.

Cubren la lógica de MediaDownloader con un runner falso y el
SubprocessDownloadRunner real ejecutando subprocesos locales.
"""

import subprocess
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

import pytest

from kliptych.download import (
    DownloadError,
    DownloadRunner,
    MediaDownloader,
    SubprocessDownloadRunner,
)
from kliptych.environment import CommandResult

_YTDLP = "yt-dlp"
_STREAMLINK = "streamlink"
_CHAT = "chat_downloader"

_DEFAULT_TIMEOUT_S = 3600.0
_DEFAULT_MAX_SIZE_BYTES = 2 * 1024**3
_PROBE_TIMEOUT_S = 15.0


def _output_path(argv: Sequence[str]) -> Path | None:
    if "--output" not in argv:
        return None
    return Path(argv[argv.index("--output") + 1])


def _write_partial_artifact(argv: Sequence[str]) -> None:
    destination = _output_path(argv)
    if destination is None:
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    _ = destination.write_bytes(b"parcial")


class FakeRunner:
    def __init__(
        self,
        *,
        available: Sequence[str] = (),
        on_download: Callable[[list[str]], None] | None = None,
        result: CommandResult | None = None,
        download_error: BaseException | None = None,
        probe_error: BaseException | None = None,
        write_before_error: bool = True,
    ) -> None:
        self._available: set[str] = set(available)
        self._on_download: Callable[[list[str]], None] | None = on_download
        self._result: CommandResult = result if result is not None else CommandResult(ok=True)
        self._download_error: BaseException | None = download_error
        self._probe_error: BaseException | None = probe_error
        self._write_before_error: bool = write_before_error
        self.calls: list[tuple[str, ...]] = []
        self.timeouts: list[float] = []

    def run(self, argv: Sequence[str], *, timeout_s: float) -> CommandResult:
        call = tuple(argv)
        self.calls.append(call)
        self.timeouts.append(timeout_s)
        if len(call) == 2 and call[1] == "--version":
            if self._probe_error is not None:
                raise self._probe_error
            return CommandResult(ok=call[0] in self._available)
        if self._on_download is not None:
            self._on_download(list(call))
        failed = self._download_error is not None or not self._result.ok
        if failed and self._write_before_error:
            _write_partial_artifact(call)
        if self._download_error is not None:
            raise self._download_error
        return self._result


def _writing_runner(
    destination: Path,
    *,
    payload: bytes = b"media",
    available: Sequence[str] = (_YTDLP,),
    sidecars: Sequence[Path] = (),
    download_error: BaseException | None = None,
) -> FakeRunner:
    def on_download(_argv: list[str]) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        _ = destination.write_bytes(payload)
        for sidecar in sidecars:
            _ = sidecar.write_bytes(b"parcial")

    return FakeRunner(
        available=available,
        on_download=on_download,
        download_error=download_error,
    )


def _download_calls(runner: FakeRunner) -> list[tuple[str, ...]]:
    return [call for call in runner.calls if not (len(call) == 2 and call[1] == "--version")]


def test_default_constructor_values(tmp_path: Path) -> None:
    destination = tmp_path / "video.mp4"
    runner = _writing_runner(destination)
    result = MediaDownloader(runner=runner).download_video(
        url="https://example.com/v",
        destination=destination,
    )
    assert result == destination
    assert runner.timeouts[-1] == _DEFAULT_TIMEOUT_S
    download = _download_calls(runner)[-1]
    assert download[download.index("--max-filesize") + 1] == str(_DEFAULT_MAX_SIZE_BYTES)


def test_invalid_timeout_rejected() -> None:
    with pytest.raises(ValueError, match="timeout"):
        _ = MediaDownloader(timeout_s=0)


def test_invalid_max_size_rejected() -> None:
    with pytest.raises(ValueError, match="tamaño máximo"):
        _ = MediaDownloader(max_size_bytes=-1)


def test_has_reports_available_tools() -> None:
    runner = FakeRunner(available=[_YTDLP])
    downloader = MediaDownloader(runner=runner)
    assert downloader.has(_YTDLP) is True
    assert downloader.has(_STREAMLINK) is False
    assert runner.timeouts == [_PROBE_TIMEOUT_S, _PROBE_TIMEOUT_S]


def test_has_survives_probe_error() -> None:
    runner = FakeRunner(probe_error=subprocess.TimeoutExpired(cmd=_YTDLP, timeout=1.0))
    assert MediaDownloader(runner=runner).has(_YTDLP) is False


def test_missing_tool_fails_before_running(tmp_path: Path) -> None:
    runner = FakeRunner(available=[])
    destination = tmp_path / "video.mp4"
    with pytest.raises(DownloadError, match="no está disponible"):
        _ = MediaDownloader(runner=runner).download_video(
            url="https://example.com/v",
            destination=destination,
        )
    assert _download_calls(runner) == []


def test_timeout_propagates_as_download_error(tmp_path: Path) -> None:
    destination = tmp_path / "video.mp4"
    runner = FakeRunner(
        available=[_YTDLP],
        download_error=subprocess.TimeoutExpired(cmd=_YTDLP, timeout=5.0),
    )
    with pytest.raises(DownloadError, match="timeout") as excinfo:
        _ = MediaDownloader(runner=runner, timeout_s=5.0).download_video(
            url="https://example.com/v",
            destination=destination,
        )
    assert isinstance(excinfo.value.__cause__, subprocess.TimeoutExpired)
    assert not destination.exists()


def test_unexpected_os_error_fails(tmp_path: Path) -> None:
    destination = tmp_path / "video.mp4"
    runner = FakeRunner(
        available=[_YTDLP],
        download_error=PermissionError(13, "permiso denegado"),
    )
    with pytest.raises(DownloadError, match="no se pudo ejecutar"):
        _ = MediaDownloader(runner=runner).download_video(
            url="https://example.com/v",
            destination=destination,
        )
    assert not destination.exists()


def test_unpreparable_destination_fails(tmp_path: Path) -> None:
    blocker = tmp_path / "blocker.txt"
    _ = blocker.write_bytes(b"no soy directorio")
    runner = FakeRunner(available=[_YTDLP])
    with pytest.raises(DownloadError, match="directorio del destino"):
        _ = MediaDownloader(runner=runner).download_video(
            url="https://example.com/v",
            destination=blocker / "video.mp4",
        )
    assert _download_calls(runner) == []


def test_nonzero_exit_reports_stderr(tmp_path: Path) -> None:
    destination = tmp_path / "video.mp4"
    runner = FakeRunner(
        available=[_YTDLP],
        result=CommandResult(ok=False, stderr="  ERROR: no se pudo descargar  "),
    )
    with pytest.raises(DownloadError, match="ERROR: no se pudo descargar"):
        _ = MediaDownloader(runner=runner).download_video(
            url="https://example.com/v",
            destination=destination,
        )
    assert not destination.exists()


def test_missing_artifact_after_success_fails(tmp_path: Path) -> None:
    runner = FakeRunner(available=[_YTDLP])
    destination = tmp_path / "video.mp4"
    with pytest.raises(DownloadError, match="no produjo el artefacto"):
        _ = MediaDownloader(runner=runner).download_video(
            url="https://example.com/v",
            destination=destination,
        )
    assert not destination.exists()


def test_oversized_download_fails_and_removes_artifact(tmp_path: Path) -> None:
    destination = tmp_path / "video.mp4"
    sidecar = tmp_path / "video.mp4.part"
    runner = _writing_runner(destination, payload=b"demasiado grande", sidecars=[sidecar])
    with pytest.raises(DownloadError, match="excede el tamaño máximo"):
        _ = MediaDownloader(runner=runner, max_size_bytes=4).download_video(
            url="https://example.com/v",
            destination=destination,
        )
    assert not destination.exists()
    assert not sidecar.exists()


def test_failure_removes_ytdlp_sidecars(tmp_path: Path) -> None:
    destination = tmp_path / "video.mp4"
    sidecars = [
        tmp_path / "video.mp4.part",
        tmp_path / "video.mp4.ytdl",
        tmp_path / "video.mp4.part-Frag0",
        tmp_path / "video.mp4.f137",
    ]
    runner = _writing_runner(
        destination,
        sidecars=sidecars,
        download_error=subprocess.TimeoutExpired(cmd=_YTDLP, timeout=5.0),
    )
    with pytest.raises(DownloadError, match="timeout"):
        _ = MediaDownloader(runner=runner, timeout_s=5.0).download_video(
            url="https://example.com/v",
            destination=destination,
        )
    assert not destination.exists()
    for sidecar in sidecars:
        assert not sidecar.exists()


def test_failure_preserves_preexisting_destination(tmp_path: Path) -> None:
    destination = tmp_path / "video.mp4"
    _ = destination.write_bytes(b"original")
    runner = FakeRunner(
        available=[_YTDLP],
        download_error=subprocess.TimeoutExpired(cmd=_YTDLP, timeout=5.0),
        write_before_error=False,
    )
    with pytest.raises(DownloadError, match="timeout"):
        _ = MediaDownloader(runner=runner, timeout_s=5.0).download_video(
            url="https://example.com/v",
            destination=destination,
        )
    assert destination.read_bytes() == b"original"


def test_artifact_verification_oserror_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "video.mp4"
    runner = _writing_runner(destination)

    def _raise_oserror(_self: Path) -> bool:
        raise PermissionError(13, "permiso denegado")

    monkeypatch.setattr(Path, "is_file", _raise_oserror)
    with pytest.raises(DownloadError, match="no se pudo verificar"):
        _ = MediaDownloader(runner=runner).download_video(
            url="https://example.com/v",
            destination=destination,
        )


def test_download_stream_uses_streamlink(tmp_path: Path) -> None:
    destination = tmp_path / "stream.ts"
    runner = _writing_runner(destination, available=[_STREAMLINK])
    result = MediaDownloader(runner=runner).download_stream(
        url="https://example.com/live",
        destination=destination,
    )
    assert result == destination
    assert _download_calls(runner)[-1][0] == _STREAMLINK


def test_download_chat_uses_chat_downloader(tmp_path: Path) -> None:
    destination = tmp_path / "chat.json"
    runner = _writing_runner(destination, available=[_CHAT])
    result = MediaDownloader(runner=runner).download_chat(
        url="https://example.com/v",
        destination=destination,
    )
    assert result == destination
    assert _download_calls(runner)[-1][0] == _CHAT


@pytest.mark.parametrize(
    "url",
    [
        "file:///etc/passwd",
        "ftp://internal/",
        "http://169.254.169.254/",
        "http://127.0.0.1:9999/",
        "http://10.0.0.1/",
        "http://172.16.0.1/",
        "http://192.168.1.1/",
        "http://[::1]/",
        "http://[fc00::1]/",
        "http://[::ffff:127.0.0.1]/",
        "http://[::ffff:192.168.1.1]/",
        "http://[::ffff:169.254.169.254]/",
        "http://",
        "http://[::1",
    ],
)
def test_download_video_rejects_unsafe_urls(url: str, tmp_path: Path) -> None:
    runner = FakeRunner(available=[_YTDLP])
    destination = tmp_path / "video.mp4"
    with pytest.raises(DownloadError):
        _ = MediaDownloader(runner=runner).download_video(url=url, destination=destination)
    assert _download_calls(runner) == []
    assert not destination.exists()


def test_download_video_accepts_public_https_url(tmp_path: Path) -> None:
    destination = tmp_path / "video.mp4"
    runner = _writing_runner(destination)
    result = MediaDownloader(runner=runner).download_video(
        url="https://example.com/video",
        destination=destination,
    )
    assert result == destination


def test_all_downloads_validate_url(tmp_path: Path) -> None:
    runner = FakeRunner(available=[_YTDLP, _STREAMLINK, _CHAT])
    downloader = MediaDownloader(runner=runner)
    destination = tmp_path / "artifact"
    with pytest.raises(DownloadError):
        _ = downloader.download_video(url="file:///etc/passwd", destination=destination)
    with pytest.raises(DownloadError):
        _ = downloader.download_stream(url="file:///etc/passwd", destination=destination)
    with pytest.raises(DownloadError):
        _ = downloader.download_chat(url="file:///etc/passwd", destination=destination)
    assert _download_calls(runner) == []


def test_build_ytdlp_argv_shape(tmp_path: Path) -> None:
    destination = tmp_path / "video.mp4"
    argv = MediaDownloader.build_ytdlp_argv(
        url="https://example.com/v",
        destination=destination,
        max_size_bytes=1024,
        format_selector="bestvideo+bestaudio",
    )
    assert isinstance(argv, list)
    assert argv == [
        _YTDLP,
        "--ignore-config",
        "--no-playlist",
        "--no-progress",
        "--max-filesize",
        "1024",
        "--output",
        str(destination),
        "--format",
        "bestvideo+bestaudio",
        "--",
        "https://example.com/v",
    ]
    assert argv[-2] == "--"
    assert argv[-1] == "https://example.com/v"


def test_build_ytdlp_argv_without_format(tmp_path: Path) -> None:
    argv = MediaDownloader.build_ytdlp_argv(
        url="https://example.com/v",
        destination=tmp_path / "video.mp4",
        max_size_bytes=2048,
    )
    assert "--format" not in argv
    assert "--ignore-config" in argv
    assert argv[-2] == "--"
    assert argv[-1] == "https://example.com/v"


def test_build_streamlink_argv_shape(tmp_path: Path) -> None:
    destination = tmp_path / "stream.ts"
    argv = MediaDownloader.build_streamlink_argv(
        url="https://example.com/live",
        destination=destination,
    )
    assert isinstance(argv, list)
    assert argv == [
        _STREAMLINK,
        "--no-config",
        "--output",
        str(destination),
        "--force",
        "--progress",
        "no",
        "--default-stream",
        "best",
        "--",
        "https://example.com/live",
    ]
    assert argv[-2] == "--"
    assert argv[-1] == "https://example.com/live"


def test_build_streamlink_argv_custom_stream(tmp_path: Path) -> None:
    argv = MediaDownloader.build_streamlink_argv(
        url="https://example.com/live",
        destination=tmp_path / "stream.ts",
        stream="720p",
    )
    assert argv[argv.index("--default-stream") + 1] == "720p"


def test_build_chat_argv_shape(tmp_path: Path) -> None:
    destination = tmp_path / "chat.json"
    argv = MediaDownloader.build_chat_argv(
        url="https://example.com/v",
        destination=destination,
    )
    assert isinstance(argv, list)
    assert argv == [
        _CHAT,
        "--quiet",
        "--overwrite",
        "--output",
        str(destination),
        "--",
        "https://example.com/v",
    ]
    assert argv[-2] == "--"
    assert argv[-1] == "https://example.com/v"


def test_builders_separate_url_with_double_dash(tmp_path: Path) -> None:
    destination = tmp_path / "artifact"
    builders = [
        MediaDownloader.build_ytdlp_argv(
            url="-x",
            destination=destination,
            max_size_bytes=1024,
        ),
        MediaDownloader.build_streamlink_argv(url="-x", destination=destination),
        MediaDownloader.build_chat_argv(url="-x", destination=destination),
    ]
    for argv in builders:
        assert argv[-2] == "--"
        assert argv[-1] == "-x"


def test_subprocess_download_runner_runs_local_command() -> None:
    runner: DownloadRunner = SubprocessDownloadRunner()
    result = runner.run([sys.executable, "-c", "print('hola')"], timeout_s=30.0)
    assert result.ok is True
    assert result.stdout.strip() == "hola"


def test_subprocess_download_runner_missing_binary() -> None:
    result = SubprocessDownloadRunner().run(
        ["binario-que-no-existe-kliptych-download"],
        timeout_s=5.0,
    )
    assert result.ok is False
    assert result.stderr


def test_subprocess_download_runner_lets_timeout_propagate() -> None:
    with pytest.raises(subprocess.TimeoutExpired):
        _ = SubprocessDownloadRunner().run(
            [sys.executable, "-c", "import time; time.sleep(5)"],
            timeout_s=0.2,
        )


def test_subprocess_download_runner_isolates_process_group(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    class _FakeProcess:
        pid: int = 4242
        returncode: int = 0

        @staticmethod
        def communicate(timeout: float | None = None) -> tuple[str, str]:
            _ = timeout
            return ("", "")

        @staticmethod
        def wait(timeout: float | None = None) -> int:
            _ = timeout
            return 0

    def _fake_popen(argv: list[str], **kwargs: object) -> _FakeProcess:
        captured["argv"] = argv
        captured.update(kwargs)
        return _FakeProcess()

    monkeypatch.setattr("kliptych.download.subprocess.Popen", _fake_popen)
    result = SubprocessDownloadRunner().run([_YTDLP, "--version"], timeout_s=1.0)
    assert result.ok is True
    if sys.platform == "win32":
        assert captured["creationflags"] == subprocess.CREATE_NEW_PROCESS_GROUP
        assert "start_new_session" not in captured
    else:
        assert captured["start_new_session"] is True
        assert "creationflags" not in captured
