"""Integración de descarga con binarios reales y servidor HTTP local.

El módulo se salta entero si faltan ffmpeg o yt-dlp; streamlink y
chat-downloader son opcionales y solo se sondean sin romper.
"""

import shutil
import subprocess
import threading
from collections.abc import Generator
from contextlib import contextmanager
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import override

import pytest

from kliptych.download import MediaDownloader, SubprocessDownloadRunner

_FFMPEG = shutil.which("ffmpeg")
_YTDLP = shutil.which("yt-dlp")

pytestmark = pytest.mark.skipif(
    _FFMPEG is None or _YTDLP is None,
    reason="ffmpeg/yt-dlp no disponibles",
)


class _QuietHandler(SimpleHTTPRequestHandler):
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
def test_real_ytdlp_downloads_local_video(tmp_path: Path) -> None:
    assert _YTDLP is not None
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
def test_real_tool_probes_do_not_crash() -> None:
    downloader = MediaDownloader(runner=SubprocessDownloadRunner())
    for tool in ("yt-dlp", "streamlink", "chat_downloader"):
        assert isinstance(downloader.has(tool), bool)
