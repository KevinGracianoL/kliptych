"""Backend OpenAI-compatible para extracción de contrato y redacción de captions.

Política de fallback: no hay fallback silencioso. Si el backend no responde
tras los reintentos, o rechaza la petición, se lanza ``ModelUnavailableError``
y el orquestador decide degradar y registrarlo en el manifiesto. Si la salida
no es válida tras los reintentos, se lanza ``ModelOutputError``. Los prompts
están versionados (``PROMPT_VERSION`` y ``CAPTION_PROMPT_VERSION``) y el nombre
del modelo se registra por separado en el manifiesto.
"""

import json
import os
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from http import HTTPStatus
from typing import ClassVar, cast

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from kliptych.contract import Contract, ContractDraft
from kliptych.runtime.model import (
    Caption,
    ModelOutputError,
    ModelUnavailableError,
    PieceContext,
    caption_prompt_payload,
)
from kliptych.runtime.transport import HttpError, HttpTransport, UrllibTransport

PROMPT_VERSION = "extract-v1"
CAPTION_PROMPT_VERSION = "caption-v1"
ENV_BASE_URL = "KLIPTYCH_LLM_BASE_URL"
ENV_API_KEY = "KLIPTYCH_LLM_API_KEY"
ENV_MODEL = "KLIPTYCH_LLM_MODEL"

_EXTRACT_SYSTEM_PROMPT = (
    "Eres el extractor de contratos de Kliptych. Recibes el brief de una "
    "campaña y devuelves ÚNICAMENTE un JSON válido con la forma del "
    "ContractDraft: cada campo es "
    '{"value": ..., "evidence": {"quote": str, "start": int, "end": int, '
    '"location": str}, "confidence": "explicit" | "inferred" | "missing" | '
    '"conflict"}. Reglas: si un campo no tiene cita textual, usa "missing" '
    'sin value ni evidence; si el brief se contradice, usa "conflict" con '
    "la cita; nunca inventes valores ni campos fuera del esquema."
)

_CAPTION_SYSTEM_PROMPT = (
    "Eres el redactor de captions de Kliptych. Recibes un JSON con el contrato "
    "de una pieza y devuelves ÚNICAMENTE un JSON con la forma "
    '{"caption": str, "hashtags": [str]}. Reglas: incluye las menciones '
    "obligatorias en el caption; los hashtags obligatorios van en el campo "
    "hashtags con el prefijo #; respeta el idioma del caption, las "
    "prohibiciones y los spelling locks; no inventes datos ni menciones."
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
    caption_prompt_version: ClassVar[str] = CAPTION_PROMPT_VERSION

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
        return self._complete(
            system_prompt=_EXTRACT_SYSTEM_PROMPT,
            user_content=brief,
            parse=_parse_draft,
        )

    def write_caption(self, contract: Contract, piece: PieceContext) -> Caption:
        """Redacta el caption de una pieza según el contrato.

        Args:
            contract: Contrato validado de la campaña.
            piece: Identidad de la pieza (id y plataforma).

        Returns:
            El caption con sus hashtags, validado estructuralmente.

        Raises:
            ModelInputError: Si el contrato no declara la plataforma de la pieza.
            ModelUnavailableError: Si el backend falla tras agotar los intentos.
            ModelOutputError: Si la salida no es un caption válido tras agotar
                los intentos.
        """
        return self._complete(
            system_prompt=_CAPTION_SYSTEM_PROMPT,
            user_content=_caption_user_content(contract, piece),
            parse=_parse_caption,
        )

    def chat_json(self, *, system_prompt: str, user_content: str) -> object:
        """Envía un prompt chat y devuelve el cuerpo JSON parseado.

        Args:
            system_prompt: Instrucciones del sistema para el backend.
            user_content: Contenido del usuario, serializado por el llamador.

        Returns:
            El valor JSON (objeto, lista o escalar) del primer choice.

        Raises:
            ModelUnavailableError: Si el backend no responde tras los intentos.
            ModelOutputError: Si la salida no es un JSON válido.
        """
        return self._complete(
            system_prompt=system_prompt,
            user_content=user_content,
            parse=_parse_json_value,
        )

    def _complete[T](
        self,
        *,
        system_prompt: str,
        user_content: str,
        parse: Callable[[bytes], T],
    ) -> T:
        payload = self._request_payload(system_prompt, user_content)
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
                return parse(response.body)
            except ModelOutputError as error:
                last_output_error = error
                continue
        if last_output_error is not None:
            msg = f"salida inválida tras {self._policy.max_attempts} intentos: {last_output_error}"
            raise ModelOutputError(msg) from last_output_error
        msg = f"el backend no respondió tras {self._policy.max_attempts} intentos"
        raise ModelUnavailableError(msg) from last_transport_error

    def _request_payload(self, system_prompt: str, user_content: str) -> dict[str, object]:
        return {
            "model": self._model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            "temperature": 0,
            "response_format": {"type": "json_object"},
        }

    def _backoff(self, step: int) -> None:
        delay = self._policy.backoff_s * (2.0 ** (step - 1))
        if delay > 0:
            time.sleep(delay)


def _caption_user_content(contract: Contract, piece: PieceContext) -> str:
    return json.dumps(caption_prompt_payload(contract, piece), ensure_ascii=False, sort_keys=True)


def _parse_json_value(body: bytes) -> object:
    """Parsea el completion del backend como un valor JSON genérico.

    Args:
        body: Cuerpo JSON del completion devuelto por el backend.

    Returns:
        El valor JSON (objeto, lista o escalar) del primer choice.

    Raises:
        ModelOutputError: Si la respuesta no contiene un JSON válido.
    """
    content = _completion_content(body)
    try:
        parsed = cast("object", json.loads(_strip_code_fences(content)))
    except ValueError:
        msg = "el contenido no es un JSON válido"
        raise ModelOutputError(msg) from None
    return parsed


def _parse_draft(body: bytes) -> ContractDraft:
    """Parsea el completion del backend sin encadenar contenido del modelo.

    Los errores de pydantic se lanzan con ``from None``: su representación
    puede incluir fragmentos del brief recibido y la política del repo evita
    filtrarlos a trazas o logs.

    Args:
        body: Cuerpo JSON del completion devuelto por el backend.

    Returns:
        El draft validado estructuralmente.

    Raises:
        ModelOutputError: Si la respuesta no contiene un draft válido.
    """
    content = _completion_content(body)
    try:
        return ContractDraft.model_validate_json(_strip_code_fences(content))
    except ValidationError:
        msg = "el contenido no es un ContractDraft válido"
        raise ModelOutputError(msg) from None


def _parse_caption(body: bytes) -> Caption:
    """Parsea el completion del backend sin encadenar contenido del modelo.

    Args:
        body: Cuerpo JSON del completion devuelto por el backend.

    Returns:
        El caption validado estructuralmente.

    Raises:
        ModelOutputError: Si la respuesta no contiene un caption válido.
    """
    content = _completion_content(body)
    try:
        return Caption.model_validate_json(_strip_code_fences(content))
    except ValidationError:
        msg = "el contenido no es un Caption válido"
        raise ModelOutputError(msg) from None


def _completion_content(body: bytes) -> str:
    try:
        completion = _ChatCompletion.model_validate_json(body)
    except ValidationError:
        msg = "respuesta sin choices válidos"
        raise ModelOutputError(msg) from None
    if not completion.choices:
        msg = "respuesta sin choices"
        raise ModelOutputError(msg)
    content = completion.choices[0].message.content
    if not content:
        msg = "respuesta sin contenido de mensaje"
        raise ModelOutputError(msg)
    return content


def _strip_code_fences(content: str) -> str:
    stripped = content.strip()
    if stripped.startswith("```"):
        stripped = stripped.removeprefix("```").removeprefix("json").strip().removesuffix("```")
    return stripped.strip()
