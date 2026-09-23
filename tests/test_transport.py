"""Tests del transporte HTTP real contra servidores locales."""

import json
import socket
import threading
import time
from collections.abc import Iterator
from contextlib import suppress
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import ClassVar, Protocol, cast, override

import pytest

from kliptych.runtime import HttpError, UrllibTransport


class _Handler(BaseHTTPRequestHandler):
    body: ClassVar[bytes] = b'{"ok": true}'
    status: ClassVar[int] = 200
    location: ClassVar[str | None] = None
    chunk_delay_s: ClassVar[float] = 0.0
    requests: ClassVar[list[tuple[str, str | None, bytes]]] = []

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        request_body = self.rfile.read(length)
        _Handler.requests.append((self.path, self.headers.get("Content-Type"), request_body))
        self.send_response(self.status)
        self.send_header("Content-Type", "application/json")
        if self.location is not None:
            self.send_header("Location", self.location)
        self.end_headers()
        if self.chunk_delay_s > 0:
            for index in range(len(self.body)):
                with suppress(OSError):
                    _ = self.wfile.write(self.body[index : index + 1])
                time.sleep(self.chunk_delay_s)
        else:
            with suppress(OSError):
                _ = self.wfile.write(self.body)

    @override
    def log_message(self, format: str, *args: object) -> None:
        return


class StartServer(Protocol):
    def __call__(
        self,
        body: bytes,
        *,
        status: int = 200,
        location: str | None = None,
        path: str = "/v1/chat",
        chunk_delay_s: float = 0.0,
    ) -> str: ...


@pytest.fixture
def start_server() -> Iterator[StartServer]:
    servers: list[ThreadingHTTPServer] = []
    _Handler.requests.clear()

    def start(
        body: bytes,
        *,
        status: int = 200,
        location: str | None = None,
        path: str = "/v1/chat",
        chunk_delay_s: float = 0.0,
    ) -> str:
        _Handler.body = body
        _Handler.status = status
        _Handler.location = location
        _Handler.chunk_delay_s = chunk_delay_s
        server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        servers.append(server)
        host, port = server.server_address[:2]
        return f"http://{host}:{port}{path}"

    yield start

    for server in servers:
        server.shutdown()
        server.server_close()


class StartRawServer(Protocol):
    def __call__(self, payload: bytes, *, shutdown_write: bool = False) -> str: ...


@pytest.fixture
def start_raw_server() -> Iterator[StartRawServer]:
    sockets: list[socket.socket] = []

    def start(payload: bytes, *, shutdown_write: bool = False) -> str:
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.bind(("127.0.0.1", 0))
        server.listen(1)
        sockets.append(server)

        def accept_once() -> None:
            with suppress(OSError):
                accepted = server.accept()
                connection = accepted[0]
                with connection:
                    _request: bytes = connection.recv(65536)
                    connection.sendall(payload)
                    if shutdown_write:
                        connection.shutdown(socket.SHUT_WR)

        threading.Thread(target=accept_once, daemon=True).start()
        host = cast("str", server.getsockname()[0])
        port = cast("int", server.getsockname()[1])
        return f"http://{host}:{port}/v1/chat"

    yield start

    for server in sockets:
        with suppress(OSError):
            server.close()


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
    path, content_type, body = _Handler.requests[-1]
    assert path == "/v1/chat"
    assert content_type == "application/json"
    assert json.loads(body) == {"a": 1}


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


def test_post_json_wraps_malformed_http_responses(start_raw_server: StartRawServer) -> None:
    url = start_raw_server(b"NOT-HTTP GARBAGE\r\n\r\n")
    with pytest.raises(HttpError, match="conexión"):
        _ = UrllibTransport().post_json(url, headers={}, payload={}, timeout_s=5.0)


def test_post_json_rejects_truncated_body(start_raw_server: StartRawServer) -> None:
    payload = (
        b"HTTP/1.1 200 OK\r\nContent-Length: 1000\r\nContent-Type: application/json\r\n\r\n"
        b'{"ok": true}'
    )
    url = start_raw_server(payload, shutdown_write=True)
    with pytest.raises(HttpError, match="incompleta"):
        _ = UrllibTransport().post_json(url, headers={}, payload={}, timeout_s=5.0)


def test_post_json_does_not_follow_redirects(start_server: StartServer) -> None:
    target = start_server(b"respuesta del host no configurado", path="/steal")
    source = start_server(b"", status=302, location=target)
    response = UrllibTransport().post_json(
        source,
        headers={"Authorization": "Bearer secreto"},
        payload={},
        timeout_s=5.0,
    )
    assert response.status == 302
    assert all(path != "/steal" for path, _, _ in _Handler.requests)


def test_post_json_bounds_total_time_on_slow_drip(start_server: StartServer) -> None:
    url = start_server(b"x" * 40, chunk_delay_s=0.1)
    started = time.monotonic()
    with pytest.raises(HttpError, match="timeout"):
        _ = UrllibTransport().post_json(url, headers={}, payload={}, timeout_s=0.3)
    assert time.monotonic() - started < 1.5
