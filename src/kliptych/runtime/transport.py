"""Transporte HTTP mínimo para el runtime LLM: stdlib, sin shell.

El endpoint es configuración del operador, nunca texto del LLM; aun así se
valida el esquema y se aplican timeout y límite de tamaño de respuesta.
"""

import http.client
import json
import urllib.error
import urllib.request
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol, cast
from urllib.parse import urlsplit

DEFAULT_MAX_RESPONSE_BYTES = 5 * 1024 * 1024


class HttpError(Exception):
    """El transporte HTTP no pudo completar la petición."""


@dataclass(frozen=True, slots=True)
class HttpResponse:
    """Respuesta HTTP cruda: estado y cuerpo acotado."""

    status: int
    body: bytes


class HttpTransport(Protocol):
    """Interfaz de transporte HTTP que consume el backend del modelo."""

    def post_json(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        payload: Mapping[str, object],
        timeout_s: float,
    ) -> HttpResponse:
        """Envía un JSON por POST y devuelve la respuesta.

        Args:
            url: URL absoluta http/https del endpoint.
            headers: Cabeceras adicionales de la petición.
            payload: Cuerpo JSON serializable.
            timeout_s: Timeout máximo de la petición, en segundos.

        Returns:
            La respuesta con su estado y cuerpo.

        Raises:
            HttpError: Si la conexión falla, el esquema no es http/https o la
                respuesta excede el límite de tamaño.
        """
        ...


class UrllibTransport:
    """Transporte real con ``urllib``, timeout explícito y tope de tamaño."""

    def __init__(self, *, max_response_bytes: int = DEFAULT_MAX_RESPONSE_BYTES) -> None:
        """Configura el tope de lectura de respuestas.

        Args:
            max_response_bytes: Máximo de bytes aceptados por respuesta.
        """
        self._max_response_bytes: int = max_response_bytes

    def post_json(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        payload: Mapping[str, object],
        timeout_s: float,
    ) -> HttpResponse:
        """Envía un JSON por POST y devuelve la respuesta acotada.

        Args:
            url: URL absoluta http/https del endpoint.
            headers: Cabeceras adicionales de la petición.
            payload: Cuerpo JSON serializable.
            timeout_s: Timeout máximo de la petición, en segundos.

        Returns:
            La respuesta con su estado y cuerpo.

        Raises:
            HttpError: Si la conexión falla, el esquema no es http/https o la
                respuesta excede el límite de tamaño.
        """
        scheme = urlsplit(url).scheme
        if scheme not in {"http", "https"}:
            msg = f"esquema de URL no soportado: {scheme!r}"
            raise HttpError(msg)
        request = urllib.request.Request(
            url,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json", **headers},
            method="POST",
        )
        try:
            with _open_connection(request, timeout_s) as response:
                status, body = _read_limited(response, self._max_response_bytes)
        except urllib.error.HTTPError as error:
            return HttpResponse(status=error.code, body=b"")
        except OSError as error:
            msg = f"falló la conexión HTTP: {error}"
            raise HttpError(msg) from error
        return HttpResponse(status=status, body=body)


def _open_connection(
    request: urllib.request.Request,
    timeout_s: float,
) -> http.client.HTTPResponse:
    response = cast("object", urllib.request.urlopen(request, timeout=timeout_s))
    if isinstance(response, http.client.HTTPResponse):
        return response
    msg = "respuesta HTTP inesperada del servidor"
    raise HttpError(msg)


def _read_limited(response: http.client.HTTPResponse, max_bytes: int) -> tuple[int, bytes]:
    body = response.read(max_bytes + 1)
    if len(body) > max_bytes:
        msg = f"la respuesta excede el límite de {max_bytes} bytes"
        raise HttpError(msg)
    return response.status, body
