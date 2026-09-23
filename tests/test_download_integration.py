"""Integración de descarga con binarios reales y servidor HTTP local.

El módulo se salta entero si faltan ffmpeg o yt-dlp; streamlink y
chat-downloader son opcionales y solo se sondean sin romper. Los fixtures se
sirven en loopback, que la validación de URL (probada en unitarios) rechaza por
diseño, así que aquí se desactiva esa validación para ejercitar la descarga.
"""

import shutil
import subprocess
import threading
import time
from collections.abc import Generator
from contextlib import contextmanager
from functools import partial
from http.server import BaseHTTPRequestHandler, SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import override

import pytest

from kliptych.download import DownloadError, MediaDownloader, SubprocessDownloadRunner

_FFMPEG = shutil.which("ffmpeg")
_YTDLP = shutil.which("yt-dlp")

_SLOW_CONTENT_LENGTH = 1024**3
_SLOW_CHUNK_DELAY_S = 0.5
_SLOW_TIMEOUT_S = 3.0
_SLOW_TIMEOUT_MARGIN_S = 15.0

pytestmark = pytest.mark.skipif(
    _FFMPEG is None or _YTDLP is None,
    reason="ffmpeg/yt-dlp no disponibles",
)


def _noop_validate(_url: str) -> None:
    return


class _QuietHandler(SimpleHTTPRequestHandler):
    @override
    def log_message(self, format: str, *args: object) -> None:
        return


class _SlowHandler(BaseHTTPRequestHandler):
    def do_HEAD(self) -> None:
        self._send_headers()

    def do_GET(self) -> None:
        self._send_headers()
        try:
            while True:
                _ = self.wfile.write(b"\0")
                self.wfile.flush()
                time.sleep(_SLOW_CHUNK_DELAY_S)
        except OSError:
            return

    def _send_headers(self) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "video/mp4")
        self.send_header("Content-Length", str(_SLOW_CONTENT_LENGTH))
        self.end_headers()

    @override
    def log_message(self, format: str, *args: object) -> None:
        return


@contextmanager
def _serve(directory: Path) -> Generator[str]:
    handler = partial(_QuietHandler, directory=str(directory))
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address[:2]
    try:
        yield f"http://{host}:{port}"
    finally:
        server.shutdown()
        server.server_close()


@contextmanager
def _serve_slow() -> Generator[str]:
    server = ThreadingHTTPServer(("127.0.0.1", 0), _SlowHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address[:2]
    try:
        yield f"http://{host}:{port}"
    finally:
        server.shutdown()
        server.server_close()


def _generate_clip(path: Path) -> Path:
    assert _FFMPEG is not None
    argv = [
        _FFMPEG,
        "-y",
        "-v",
        "error",
        "-f",
        "lavfi",
        "-i",
        "color=c=blue:s=160x120:d=1.0",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        str(path),
    ]
    _ = subprocess.run(argv, capture_output=True, check=True, timeout=120)
    return path


@pytest.mark.integration
def test_real_ytdlp_downloads_local_video(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert _YTDLP is not None
    monkeypatch.setattr("kliptych.download._validate_url", _noop_validate)
    served = tmp_path / "served"
    served.mkdir()
    _ = _generate_clip(served / "clip.mp4")
    destination = tmp_path / "download" / "clip.mp4"
    with _serve(served) as base:
        result = MediaDownloader(runner=SubprocessDownloadRunner()).download_video(
            url=f"{base}/clip.mp4",
            destination=destination,
        )
    assert result == destination
    assert destination.is_file()
    assert destination.stat().st_size > 0


@pytest.mark.integration
def test_real_ytdlp_timeout_kills_slow_download(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert _YTDLP is not None
    monkeypatch.setattr("kliptych.download._validate_url", _noop_validate)
    destination = tmp_path / "slow" / "clip.mp4"
    with _serve_slow() as base:
        started = time.monotonic()
        with pytest.raises(DownloadError, match="timeout"):
            _ = MediaDownloader(
                runner=SubprocessDownloadRunner(),
                timeout_s=_SLOW_TIMEOUT_S,
            ).download_video(url=f"{base}/clip.mp4", destination=destination)
        elapsed = time.monotonic() - started
    assert elapsed < _SLOW_TIMEOUT_S + _SLOW_TIMEOUT_MARGIN_S
    assert not destination.exists()
    assert not destination.with_name(f"{destination.name}.part").exists()


@pytest.mark.integration
def test_real_tool_probes_do_not_crash() -> None:
    downloader = MediaDownloader(runner=SubprocessDownloadRunner())
    for tool in ("yt-dlp", "streamlink", "chat_downloader"):
        assert isinstance(downloader.has(tool), bool)
