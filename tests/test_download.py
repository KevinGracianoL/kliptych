"""Tests de la descarga local: unitarios con runner falso e integración real.

Ningún test unitario toca la red ni binarios externos: el runner falso emula
``subprocess`` (resultado, timeout, stderr y escritura del artefacto). Los tests
marcados ``integration`` requieren los binarios reales y se saltan si faltan;
sirven el video sintético desde un ``http.server`` en localhost, sin internet.
"""

import http.server
import shutil
import subprocess
import sys
import threading
from collections.abc import Generator, Iterable, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import ClassVar, override

import pytest

from kliptych.download import (
    DEFAULT_MAX_SIZE_BYTES,
    DEFAULT_TIMEOUT_S,
    DownloadError,
    MediaDownloader,
    SubprocessDownloadRunner,
)
from kliptych.environment import CommandResult

_FFMPEG = shutil.which("ffmpeg")
_YTDLP = shutil.which("yt-dlp")


class FakeRunner:
    """Runner falso que emula ``subprocess`` sin red ni binarios reales."""

    def __init__(
        self,
        *,
        available: Iterable[str] = (),
        payload: bytes = b"contenido",
        write_output: bool = True,
        download_timeout: bool = False,
        probe_timeout: bool = False,
        failure: CommandResult | None = None,
        os_error: bool = False,
    ) -> None:
        self._available: frozenset[str] = frozenset(available)
        self._payload: bytes = payload
        self._write_output: bool = write_output
        self._download_timeout: bool = download_timeout
        self._probe_timeout: bool = probe_timeout
        self._failure: CommandResult | None = failure
        self._os_error: bool = os_error
        self.calls: list[tuple[list[str], float]] = []

    def run(self, argv: Sequence[str], *, timeout_s: float) -> CommandResult:
        call = list(argv)
        self.calls.append((call, timeout_s))
        tool = call[0]
        if "--version" in call:
            if self._probe_timeout:
                raise subprocess.TimeoutExpired(cmd=tool, timeout=timeout_s)
            return CommandResult(ok=tool in self._available, stderr="")
        if self._os_error:
            raise OSError(2, "no se pudo ejecutar")
        if self._download_timeout:
            raise subprocess.TimeoutExpired(cmd=tool, timeout=timeout_s)
        if self._failure is not None:
            return self._failure
        if self._write_output:
            destination = Path(call[call.index("--output") + 1])
            destination.parent.mkdir(parents=True, exist_ok=True)
            _ = destination.write_bytes(self._payload)
        return CommandResult(ok=True)

    def download_call(self) -> list[str]:
        downloads = [call for call, _ in self.calls if "--version" not in call]
        assert len(downloads) == 1
        return downloads[0]

    def timeouts(self) -> list[float]:
        return [timeout for _, timeout in self.calls]


def test_has_detects_missing_tool() -> None:
    downloader = MediaDownloader(runner=FakeRunner())
    assert downloader.has("yt-dlp") is False


def test_has_detects_available_tool() -> None:
    downloader = MediaDownloader(runner=FakeRunner(available={"yt-dlp"}))
    assert downloader.has("yt-dlp") is True


def test_has_returns_false_on_probe_timeout() -> None:
    downloader = MediaDownloader(runner=FakeRunner(probe_timeout=True))
    assert downloader.has("yt-dlp") is False


def test_video_argv_never_uses_shell(tmp_path: Path) -> None:
    argv = MediaDownloader().build_ytdlp_argv(
        url="https://example.com/video",
        destination=tmp_path / "video.mp4",
    )
    assert isinstance(argv, list)
    assert all(isinstance(part, str) for part in argv)
    assert "-c" not in argv
    assert argv[0] == "yt-dlp"
    assert "--max-filesize" in argv
    assert argv[argv.index("--max-filesize") + 1] == str(DEFAULT_MAX_SIZE_BYTES)
    assert argv[-1] == "https://example.com/video"


def test_video_argv_binds_format_selector(tmp_path: Path) -> None:
    argv = MediaDownloader().build_ytdlp_argv(
        url="https://example.com/video",
        destination=tmp_path / "video.mp4",
        format_selector="bestvideo+bestaudio",
    )
    assert argv[argv.index("--format") + 1] == "bestvideo+bestaudio"


def test_chat_argv_never_uses_shell(tmp_path: Path) -> None:
    argv = MediaDownloader().build_chat_argv(
        url="https://example.com/stream",
        destination=tmp_path / "chat.json",
    )
    assert isinstance(argv, list)
    assert "-c" not in argv
    assert argv[0] == "chat_downloader"
    assert "--output" in argv
    assert argv[-1] == "https://example.com/stream"


def test_streamlink_argv_never_uses_shell(tmp_path: Path) -> None:
    argv = MediaDownloader().build_streamlink_argv(
        url="https://twitch.tv/channel",
        destination=tmp_path / "live.ts",
    )
    assert isinstance(argv, list)
    assert "-c" not in argv
    assert argv[0] == "streamlink"
    assert "--output" in argv
    assert "--stream-segment-timeout" in argv
    assert argv[-1] == "https://twitch.tv/channel"


def test_download_video_missing_tool_raises(tmp_path: Path) -> None:
    downloader = MediaDownloader(runner=FakeRunner())
    with pytest.raises(DownloadError, match="no está disponible"):
        _ = downloader.download_video(
            url="https://example.com/video",
            destination=tmp_path / "video.mp4",
        )


def test_download_stream_missing_tool_raises(tmp_path: Path) -> None:
    downloader = MediaDownloader(runner=FakeRunner())
    with pytest.raises(DownloadError, match="no está disponible"):
        _ = downloader.download_stream(
            url="https://twitch.tv/channel",
            destination=tmp_path / "live.ts",
        )


def test_download_chat_missing_tool_raises(tmp_path: Path) -> None:
    downloader = MediaDownloader(runner=FakeRunner())
    with pytest.raises(DownloadError, match="no está disponible"):
        _ = downloader.download_chat(
            url="https://example.com/stream",
            destination=tmp_path / "chat.json",
        )


def test_download_video_timeout_raises(tmp_path: Path) -> None:
    runner = FakeRunner(available={"yt-dlp"}, download_timeout=True)
    downloader = MediaDownloader(runner=runner, timeout_s=12.5)
    with pytest.raises(DownloadError, match="timeout") as excinfo:
        _ = downloader.download_video(
            url="https://example.com/video",
            destination=tmp_path / "video.mp4",
        )
    assert isinstance(excinfo.value.__cause__, subprocess.TimeoutExpired)
    assert 12.5 in runner.timeouts()


def test_download_chat_timeout_raises(tmp_path: Path) -> None:
    runner = FakeRunner(available={"chat_downloader"}, download_timeout=True)
    downloader = MediaDownloader(runner=runner, timeout_s=3.0)
    with pytest.raises(DownloadError, match="timeout") as excinfo:
        _ = downloader.download_chat(
            url="https://example.com/stream",
            destination=tmp_path / "chat.json",
        )
    assert isinstance(excinfo.value.__cause__, subprocess.TimeoutExpired)


def test_download_video_failure_raises(tmp_path: Path) -> None:
    runner = FakeRunner(
        available={"yt-dlp"},
        failure=CommandResult(ok=False, stderr="  Invalid data found  "),
    )
    downloader = MediaDownloader(runner=runner)
    with pytest.raises(DownloadError, match="Invalid data found"):
        _ = downloader.download_video(
            url="https://example.com/video",
            destination=tmp_path / "video.mp4",
        )


def test_download_video_os_error_raises(tmp_path: Path) -> None:
    runner = FakeRunner(available={"yt-dlp"}, os_error=True)
    downloader = MediaDownloader(runner=runner)
    with pytest.raises(DownloadError, match="no se pudo ejecutar"):
        _ = downloader.download_video(
            url="https://example.com/video",
            destination=tmp_path / "video.mp4",
        )


def test_download_video_writes_to_destination(tmp_path: Path) -> None:
    runner = FakeRunner(available={"yt-dlp"}, payload=b"video-bytes")
    destination = tmp_path / "nested" / "video.mp4"
    downloader = MediaDownloader(runner=runner)
    result = downloader.download_video(
        url="https://example.com/video",
        destination=destination,
    )
    assert result == destination
    assert destination.read_bytes() == b"video-bytes"
    argv = runner.download_call()
    assert argv[argv.index("--output") + 1] == str(destination)
    assert argv[-1] == "https://example.com/video"
    assert DEFAULT_TIMEOUT_S in runner.timeouts()


def test_download_chat_writes_to_destination(tmp_path: Path) -> None:
    runner = FakeRunner(available={"chat_downloader"}, payload=b"[]")
    destination = tmp_path / "chat.json"
    downloader = MediaDownloader(runner=runner)
    result = downloader.download_chat(
        url="https://example.com/stream",
        destination=destination,
    )
    assert result == destination
    assert destination.read_bytes() == b"[]"
    argv = runner.download_call()
    assert argv[0] == "chat_downloader"
    assert argv[-1] == "https://example.com/stream"


def test_download_stream_writes_to_destination(tmp_path: Path) -> None:
    runner = FakeRunner(available={"streamlink"}, payload=b"live")
    destination = tmp_path / "live.ts"
    downloader = MediaDownloader(runner=runner)
    result = downloader.download_stream(
        url="https://twitch.tv/channel",
        destination=destination,
    )
    assert result == destination
    assert destination.read_bytes() == b"live"
    assert runner.download_call()[0] == "streamlink"


def test_download_video_rejects_oversized_file(tmp_path: Path) -> None:
    runner = FakeRunner(available={"yt-dlp"}, payload=b"x" * 10)
    downloader = MediaDownloader(runner=runner, max_size_bytes=5)
    with pytest.raises(DownloadError, match="cota"):
        _ = downloader.download_video(
            url="https://example.com/video",
            destination=tmp_path / "video.mp4",
        )


def test_download_video_requires_artifact(tmp_path: Path) -> None:
    runner = FakeRunner(available={"yt-dlp"}, write_output=False)
    downloader = MediaDownloader(runner=runner)
    with pytest.raises(DownloadError, match="no generó"):
        _ = downloader.download_video(
            url="https://example.com/video",
            destination=tmp_path / "video.mp4",
        )


def test_unpreparable_destination_fails(tmp_path: Path) -> None:
    blocker = tmp_path / "blocker"
    _ = blocker.write_bytes(b"x")
    runner = FakeRunner(available={"yt-dlp"})
    downloader = MediaDownloader(runner=runner)
    with pytest.raises(DownloadError, match="directorio del destino"):
        _ = downloader.download_video(
            url="https://example.com/video",
            destination=blocker / "video.mp4",
        )


def test_invalid_max_size_rejected() -> None:
    with pytest.raises(ValueError, match="max_size_bytes"):
        _ = MediaDownloader(max_size_bytes=0)


def test_invalid_timeout_rejected() -> None:
    with pytest.raises(ValueError, match="timeout_s"):
        _ = MediaDownloader(timeout_s=-1.0)


def test_subprocess_runner_reports_success() -> None:
    result = SubprocessDownloadRunner().run(
        [sys.executable, "-c", "print('ok')"],
        timeout_s=30.0,
    )
    assert result.ok is True
    assert result.stdout.strip() == "ok"


def test_subprocess_runner_missing_binary() -> None:
    result = SubprocessDownloadRunner().run(
        ["binario-inexistente-kliptych"],
        timeout_s=30.0,
    )
    assert result.ok is False
    assert result.stderr


def test_subprocess_runner_propagates_timeout() -> None:
    with pytest.raises(subprocess.TimeoutExpired):
        _ = SubprocessDownloadRunner().run(
            [sys.executable, "-c", "import time; time.sleep(5)"],
            timeout_s=0.2,
        )


class _VideoHandler(http.server.BaseHTTPRequestHandler):
    payload: ClassVar[bytes] = b""

    def do_GET(self) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "video/mp4")
        self.send_header("Content-Length", str(len(self.payload)))
        self.end_headers()
        _ = self.wfile.write(self.payload)

    @override
    def log_message(self, format: str, *args: object) -> None:
        return


def _generate_clip(path: Path) -> None:
    assert _FFMPEG is not None
    argv = [
        _FFMPEG,
        "-y",
        "-v",
        "error",
        "-f",
        "lavfi",
        "-i",
        "color=c=blue:s=320x240:d=1",
        "-f",
        "lavfi",
        "-i",
        "sine=frequency=440:duration=1",
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


@contextmanager
def _serve(payload: bytes) -> Generator[str]:
    _VideoHandler.payload = payload
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _VideoHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address[:2]
    try:
        yield f"http://{host}:{port}/source.mp4"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=10)


@pytest.mark.integration
@pytest.mark.skipif(
    _FFMPEG is None or _YTDLP is None,
    reason="ffmpeg/yt-dlp no disponibles",
)
def test_real_ytdlp_downloads_local_video(tmp_path: Path) -> None:
    assert _FFMPEG is not None
    assert _YTDLP is not None
    source = tmp_path / "source.mp4"
    _generate_clip(source)
    destination = tmp_path / "out" / "video.mp4"
    with _serve(source.read_bytes()) as server_url:
        result = MediaDownloader().download_video(url=server_url, destination=destination)
    assert result == destination
    assert destination.is_file()
    assert destination.stat().st_size > 0


@pytest.mark.integration
def test_real_tool_probes_do_not_crash() -> None:
    downloader = MediaDownloader()
    for tool in ("yt-dlp", "streamlink", "chat_downloader"):
        assert isinstance(downloader.has(tool), bool)
