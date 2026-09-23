"""Backend OpenAI-compatible para extracción de contrato.

Política de fallback: no hay fallback silencioso. Si el backend no responde
tras los reintentos, o rechaza la petición, se lanza ``ModelUnavailableError``
y el orquestador decide degradar y registrarlo en el manifiesto. Si la salida
no es un ``ContractDraft`` válido tras los reintentos, se lanza
``ModelOutputError``. El prompt está versionado (``PROMPT_VERSION``) y el
nombre del modelo se registra por separado en el manifiesto.
"""

import os
import time
from collections.abc import Mapping
from dataclasses import dataclass
from http import HTTPStatus
from typing import ClassVar

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from kliptych.contract import ContractDraft
from kliptych.runtime.model import ModelOutputError, ModelUnavailableError
from kliptych.runtime.transport import HttpError, HttpTransport, UrllibTransport

PROMPT_VERSION = "extract-v1"
ENV_BASE_URL = "KLIPTYCH_LLM_BASE_URL"
ENV_API_KEY = "KLIPTYCH_LLM_API_KEY"
ENV_MODEL = "KLIPTYCH_LLM_MODEL"

_SYSTEM_PROMPT = (
    "Eres el extractor de contratos de Kliptych. Recibes el brief de una "
    "campaña y devuelves ÚNICAMENTE un JSON válido con la forma del "
    "ContractDraft: cada campo es "
    '{"value": ..., "evidence": {"quote": str, "start": int, "end": int, '
    '"location": str}, "confidence": "explicit" | "inferred" | "missing" | '
    '"conflict"}. Reglas: si un campo no tiene cita textual, usa "missing" '
    'sin value ni evidence; si el brief se contradice, usa "conflict" con '
    "la cita; nunca inventes valores ni campos fuera del esquema."
)


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    """Política de timeout y reintentos con backoff exponencial."""

    max_attempts: int = 3
    backoff_s: float = 0.5
    timeout_s: float = 120.0


class _ChatMessage(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="ignore")

    content: str | None = None


class _ChatChoice(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="ignore")

    message: _ChatMessage


class _ChatCompletion(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="ignore")

    choices: list[_ChatChoice] = Field(default_factory=list)


class OpenAIChatModel:
    """Primer backend del runtime: chat completions compatible con OpenAI."""

    prompt_version: ClassVar[str] = PROMPT_VERSION

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        transport: HttpTransport | None = None,
        policy: RetryPolicy | None = None,
    ) -> None:
        """Configura el backend.

        Args:
            base_url: URL base del endpoint (configuración del operador).
            api_key: Credencial de acceso; nunca se registra en logs.
            model: Nombre del modelo solicitado.
            transport: Transporte HTTP a inyectar; por defecto el real.
            policy: Política de timeout y reintentos; por defecto la estándar.
        """
        self._base_url: str = base_url
        self._api_key: str = api_key
        self._model: str = model
        self._transport: HttpTransport = UrllibTransport() if transport is None else transport
        self._policy: RetryPolicy = RetryPolicy() if policy is None else policy

    @property
    def model_version(self) -> str:
        """Nombre del modelo solicitado al backend.

        Returns:
            El identificador de modelo configurado.
        """
        return self._model

    @classmethod
    def from_env(
        cls,
        environ: Mapping[str, str] | None = None,
        *,
        transport: HttpTransport | None = None,
        policy: RetryPolicy | None = None,
    ) -> "OpenAIChatModel":
        """Construye el backend desde las variables de entorno del operador.

        Args:
            environ: Mapeo de entorno; por defecto ``os.environ``.
            transport: Transporte HTTP a inyectar; por defecto el real.
            policy: Política de timeout y reintentos; por defecto la estándar.

        Returns:
            El backend configurado.

        Raises:
            ModelUnavailableError: Si falta alguna variable requerida.
        """
        source = os.environ if environ is None else environ
        missing = [name for name in (ENV_BASE_URL, ENV_API_KEY, ENV_MODEL) if not source.get(name)]
        if missing:
            msg = f"faltan variables de entorno: {', '.join(missing)}"
            raise ModelUnavailableError(msg)
        return cls(
            base_url=source[ENV_BASE_URL],
            api_key=source[ENV_API_KEY],
            model=source[ENV_MODEL],
            transport=transport,
            policy=policy,
        )

    def extract_contract(self, brief: str) -> ContractDraft:
        """Extrae el contrato del brief con evidencia por campo.

        Args:
            brief: Texto crudo del brief de campaña.

        Returns:
            El draft extraído y validado estructuralmente.

        Raises:
            ModelUnavailableError: Si el backend falla tras agotar los
                intentos, o rechaza la petición con un 4xx.
            ModelOutputError: Si la salida no es un draft válido tras agotar
                los intentos.
        """
        payload = self._request_payload(brief)
        url = f"{self._base_url.rstrip('/')}/chat/completions"
        last_transport_error: Exception | None = None
        last_output_error: Exception | None = None
        for attempt in range(1, self._policy.max_attempts + 1):
            if attempt > 1:
                self._backoff(attempt - 1)
            last_output_error = None
            try:
                response = self._transport.post_json(
                    url,
                    headers={"Authorization": f"Bearer {self._api_key}"},
                    payload=payload,
                    timeout_s=self._policy.timeout_s,
                )
            except HttpError as error:
                last_transport_error = error
                continue
            if (
                response.status >= HTTPStatus.INTERNAL_SERVER_ERROR
                or response.status == HTTPStatus.TOO_MANY_REQUESTS
            ):
                last_transport_error = ModelUnavailableError(
                    f"el backend devolvió HTTP {response.status}"
                )
                continue
            if response.status != HTTPStatus.OK:
                msg = f"el backend rechazó la petición (HTTP {response.status})"
                raise ModelUnavailableError(msg)
            try:
                return _parse_draft(response.body)
            except ModelOutputError as error:
                last_output_error = error
                continue
        if last_output_error is not None:
            msg = f"salida inválida tras {self._policy.max_attempts} intentos: {last_output_error}"
            raise ModelOutputError(msg) from last_output_error
        msg = f"el backend no respondió tras {self._policy.max_attempts} intentos"
        raise ModelUnavailableError(msg) from last_transport_error

    def _request_payload(self, brief: str) -> dict[str, object]:
        return {
            "model": self._model,
            "messages": [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": brief},
            ],
            "temperature": 0,
            "response_format": {"type": "json_object"},
        }

    def _backoff(self, step: int) -> None:
        delay = self._policy.backoff_s * (2.0 ** (step - 1))
        if delay > 0:
            time.sleep(delay)


def _parse_draft(body: bytes) -> ContractDraft:
    try:
        completion = _ChatCompletion.model_validate_json(body)
    except ValidationError as error:
        msg = "respuesta sin choices válidos"
        raise ModelOutputError(msg) from error
    if not completion.choices:
        msg = "respuesta sin choices"
        raise ModelOutputError(msg)
    content = completion.choices[0].message.content
    if not content:
        msg = "respuesta sin contenido de mensaje"
        raise ModelOutputError(msg)
    try:
        return ContractDraft.model_validate_json(_strip_code_fences(content))
    except ValidationError as error:
        msg = "el contenido no es un ContractDraft válido"
        raise ModelOutputError(msg) from error


def _strip_code_fences(content: str) -> str:
    stripped = content.strip()
    if stripped.startswith("```"):
        stripped = stripped.removeprefix("```").removeprefix("json").strip().removesuffix("```")
    return stripped.strip()
