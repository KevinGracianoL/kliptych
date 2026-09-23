"""Tests del transporte HTTP real contra un servidor local."""

import json
import threading
from collections.abc import Iterator
from contextlib import suppress
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import ClassVar, Protocol, override

import pytest

from kliptych.runtime import HttpError, UrllibTransport


class _Handler(BaseHTTPRequestHandler):
    body: ClassVar[bytes] = b'{"ok": true}'
    status: ClassVar[int] = 200

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        _ = self.rfile.read(length)
        self.send_response(self.status)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        with suppress(OSError):
            _ = self.wfile.write(self.body)

    @override
    def log_message(self, format: str, *args: object) -> None:
        return


class StartServer(Protocol):
    def __call__(self, body: bytes, *, status: int = 200) -> str: ...


@pytest.fixture
def start_server() -> Iterator[StartServer]:
    servers: list[ThreadingHTTPServer] = []

    def start(body: bytes, *, status: int = 200) -> str:
        _Handler.body = body
        _Handler.status = status
        server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        servers.append(server)
        host, port = server.server_address[:2]
        return f"http://{host}:{port}/v1/chat"

    yield start

    for server in servers:
        server.shutdown()


def test_post_json_returns_response(start_server: StartServer) -> None:
    url = start_server(b'{"ok": true}')
    response = UrllibTransport().post_json(
        url,
        headers={"Authorization": "Bearer x"},
        payload={"a": 1},
        timeout_s=5.0,
    )
    assert response.status == 200
    assert json.loads(response.body) == {"ok": True}


def test_post_json_maps_http_error_status(start_server: StartServer) -> None:
    url = start_server(b"", status=401)
    response = UrllibTransport().post_json(url, headers={}, payload={}, timeout_s=5.0)
    assert response.status == 401


def test_post_json_enforces_size_limit(start_server: StartServer) -> None:
    url = start_server(b"x" * 4096)
    with pytest.raises(HttpError, match="límite"):
        _ = UrllibTransport(max_response_bytes=128).post_json(
            url,
            headers={},
            payload={},
            timeout_s=5.0,
        )


def test_post_json_rejects_non_http_scheme() -> None:
    with pytest.raises(HttpError, match="esquema"):
        _ = UrllibTransport().post_json(
            "ftp://example.com/x",
            headers={},
            payload={},
            timeout_s=1.0,
        )


def test_post_json_wraps_connection_errors() -> None:
    with pytest.raises(HttpError, match="conexión"):
        _ = UrllibTransport().post_json(
            "http://127.0.0.1:1/v1/chat",
            headers={},
            payload={},
            timeout_s=1.0,
        )
