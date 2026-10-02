"""Inteligencia de campañas: clasificación de arquetipos (fase E).

Clasifica un brief y su contrato validado en ``KNOWN``, ``KNOWN_WITH_VARIATION``
o ``NEW_ARCHETYPE``. Es una capa analítica: no toca el pipeline ni modifica el
contrato. La clasificación asistida por LLM reutiliza el transporte HTTP del
runtime, exige timeout y falla explícitamente si el backend no responde o
devuelve una salida inválida; nunca inventa un arquetipo.
"""

import json
import time
from enum import StrEnum
from http import HTTPStatus
from typing import ClassVar, Protocol, Self

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from kliptych.contract import Contract
from kliptych.runtime.model import ModelOutputError, ModelUnavailableError
from kliptych.runtime.openai_compatible import RetryPolicy
from kliptych.runtime.transport import HttpError, HttpTransport, UrllibTransport

CLASSIFY_PROMPT_VERSION = "classify-v1"

_CLASSIFY_SYSTEM_PROMPT = (
    "Eres el clasificador de arquetipos de Kliptych. Recibes un JSON con el "
    "brief y el contrato extraído y devuelves ÚNICAMENTE un JSON con la forma "
    '{"archetype": "KNOWN" | "KNOWN_WITH_VARIATION" | "NEW_ARCHETYPE", '
    '"rationale": str, "variations": [str]}. Reglas: KNOWN si el brief encaja '
    "limpio en el contrato; KNOWN_WITH_VARIATION si hay un valor nuevo dentro "
    "de un campo existente, listando cada variación; NEW_ARCHETYPE si el brief "
    "exige algo que el contrato no puede representar. Nunca inventes campos. "
    "Las URLs de video o streaming (YouTube, Kick, Twitch, etc.) del brief son "
    "entradas operacionales del pipeline y no campos del contrato; "
    "clasifícalas como KNOWN si las reglas y plataformas coinciden."
)


class Archetype(StrEnum):
    """Clasificación de la campaña según su encaje en el contrato."""

    KNOWN = "KNOWN"
    KNOWN_WITH_VARIATION = "KNOWN_WITH_VARIATION"
    NEW_ARCHETYPE = "NEW_ARCHETYPE"


class ArchetypeClassification(BaseModel):
    """Resultado de la clasificación de una campaña."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", frozen=True)

    archetype: Archetype
    rationale: str = Field(min_length=1)
    variations: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _variations_match_archetype(self) -> Self:
        if self.archetype is Archetype.KNOWN_WITH_VARIATION and not self.variations:
            msg = "KNOWN_WITH_VARIATION exige al menos una variación registrada"
            raise ValueError(msg)
        if self.archetype is not Archetype.KNOWN_WITH_VARIATION and self.variations:
            msg = "solo KNOWN_WITH_VARIATION admite variaciones"
            raise ValueError(msg)
        return self


class CampaignClassifier(Protocol):
    """Interfaz del clasificador de campañas."""

    def classify(self, brief: str, contract: Contract) -> ArchetypeClassification:
        """Clasifica la campaña según su encaje en el contrato.

        Args:
            brief: Texto crudo del brief de campaña.
            contract: Contrato validado extraído del brief.

        Returns:
            La clasificación con su justificación y variaciones.

        Raises:
            ModelUnavailableError: Si el backend no está disponible.
            ModelOutputError: Si la salida no es una clasificación válida.
        """
        ...


class VanillaClassifier:
    """Clasificador del modo zero-contract: siempre ``KNOWN``, sin modelo.

    Sustituye a ``LLMCampaignClassifier`` cuando no hay brief. Sin brief no hay
    nada que clasificar: la pregunta "¿este brief encaja en el contrato?" no
    tiene sentido, y responderla exigiría una llamada al LLM que este modo
    prohíbe. La respuesta es ``KNOWN`` porque el contrato se sintetizó para la
    URL concreta y, por construcción, todo encaja.
    """

    @staticmethod
    def classify(brief: str, contract: Contract) -> ArchetypeClassification:
        """Clasifica como ``KNOWN`` sin contactar ningún modelo.

        Args:
            brief: Texto del brief; se ignora y puede estar vacío.
            contract: Contrato trampa del modo zero-contract.

        Returns:
            Una clasificación ``KNOWN`` sin variaciones.
        """
        _ = (brief, contract)
        return ArchetypeClassification(
            archetype=Archetype.KNOWN,
            rationale="modo zero-contract: sin brief no hay arquetipo que clasificar",
            variations=(),
        )


def classify_prompt_payload(brief: str, contract: Contract) -> dict[str, object]:
    """Construye el payload que recibe el clasificador.

    Es la fuente única del prompt: solo viajan el brief y el contrato
    representable, sin assets (rutas locales que no aportan a la clasificación).

    Args:
        brief: Texto crudo del brief de campaña.
        contract: Contrato validado extraído del brief.

    Returns:
        El payload serializable del prompt de clasificación.
    """
    return {
        "brief": brief,
        "contract": contract.model_dump(mode="json", exclude={"assets"}),
    }


class LLMCampaignClassifier:
    """Clasificación de campañas asistida por LLM."""

    prompt_version: ClassVar[str] = CLASSIFY_PROMPT_VERSION

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        transport: HttpTransport | None = None,
        policy: RetryPolicy | None = None,
    ) -> None:
        """Configura el clasificador.

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

    def classify(self, brief: str, contract: Contract) -> ArchetypeClassification:
        """Clasifica la campaña según su encaje en el contrato.

        Args:
            brief: Texto crudo del brief de campaña.
            contract: Contrato validado extraído del brief.

        Returns:
            La clasificación con su justificación y variaciones.

        Raises:
            ModelUnavailableError: Si el backend falla tras agotar los
                intentos, o rechaza la petición con un 4xx.
            ModelOutputError: Si la salida no es una clasificación válida tras
                agotar los intentos.
        """
        return self._complete(
            system_prompt=_CLASSIFY_SYSTEM_PROMPT,
            user_content=_classify_user_content(brief, contract),
        )

    def _complete(self, *, system_prompt: str, user_content: str) -> ArchetypeClassification:
        payload: dict[str, object] = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            "temperature": 0,
            "response_format": {"type": "json_object"},
        }
        url = f"{self._base_url.rstrip('/')}/chat/completions"
        last_transport_error: Exception | None = None
        last_output_error: Exception | None = None
        for attempt in range(1, self._policy.max_attempts + 1):
            if attempt > 1:
                delay = self._policy.backoff_s * (2.0 ** (attempt - 2))
                if delay > 0:
                    time.sleep(delay)
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
                return _parse_classification(response.body)
            except ModelOutputError as error:
                last_output_error = error
                continue
        if last_output_error is not None:
            msg = f"salida inválida tras {self._policy.max_attempts} intentos: {last_output_error}"
            raise ModelOutputError(msg) from last_output_error
        msg = f"el backend no respondió tras {self._policy.max_attempts} intentos"
        raise ModelUnavailableError(msg) from last_transport_error


class _ChatMessage(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="ignore")

    content: str | None = None


class _ChatChoice(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="ignore")

    message: _ChatMessage


class _ChatCompletion(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="ignore")

    choices: list[_ChatChoice] = Field(default_factory=list)


def _classify_user_content(brief: str, contract: Contract) -> str:
    return json.dumps(classify_prompt_payload(brief, contract), ensure_ascii=False, sort_keys=True)


def _parse_classification(body: bytes) -> ArchetypeClassification:
    """Parsea el completion sin encadenar contenido del modelo.

    Los errores de pydantic se lanzan con ``from None``: su representación puede
    incluir fragmentos del brief recibido y la política del repo evita
    filtrarlos a trazas o logs.

    Args:
        body: Cuerpo JSON del completion devuelto por el backend.

    Returns:
        La clasificación validada estructuralmente.

    Raises:
        ModelOutputError: Si la respuesta no contiene una clasificación válida.
    """
    content = _completion_content(body)
    try:
        return ArchetypeClassification.model_validate_json(_strip_code_fences(content))
    except ValidationError:
        msg = "el contenido no es una ArchetypeClassification válida"
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
