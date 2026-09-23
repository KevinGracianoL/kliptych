"""Runtime LLM de Kliptych: interfaz, backend OpenAI-compatible y grabaciones."""

from kliptych.runtime.model import (
    CampaignModel,
    ModelError,
    ModelOutputError,
    ModelUnavailableError,
)
from kliptych.runtime.openai_compatible import (
    ENV_API_KEY,
    ENV_BASE_URL,
    ENV_MODEL,
    PROMPT_VERSION,
    OpenAIChatModel,
    RetryPolicy,
)
from kliptych.runtime.recorded import (
    RecordedDocument,
    RecordedModel,
    brief_key,
    normalize_brief,
    record_response,
)
from kliptych.runtime.transport import (
    DEFAULT_MAX_RESPONSE_BYTES,
    HttpError,
    HttpResponse,
    HttpTransport,
    UrllibTransport,
)

__all__ = [
    "DEFAULT_MAX_RESPONSE_BYTES",
    "ENV_API_KEY",
    "ENV_BASE_URL",
    "ENV_MODEL",
    "PROMPT_VERSION",
    "CampaignModel",
    "HttpError",
    "HttpResponse",
    "HttpTransport",
    "ModelError",
    "ModelOutputError",
    "ModelUnavailableError",
    "OpenAIChatModel",
    "RecordedDocument",
    "RecordedModel",
    "RetryPolicy",
    "UrllibTransport",
    "brief_key",
    "normalize_brief",
    "record_response",
]
