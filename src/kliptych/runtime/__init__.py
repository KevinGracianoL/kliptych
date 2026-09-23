"""Runtime LLM de Kliptych: interfaz, backend OpenAI-compatible y grabaciones."""

from kliptych.hashing import brief_key, normalize_brief
from kliptych.runtime.model import (
    CampaignModel,
    Caption,
    ModelError,
    ModelInputError,
    ModelOutputError,
    ModelUnavailableError,
    PieceContext,
)
from kliptych.runtime.openai_compatible import (
    CAPTION_PROMPT_VERSION,
    ENV_API_KEY,
    ENV_BASE_URL,
    ENV_MODEL,
    PROMPT_VERSION,
    OpenAIChatModel,
    RetryPolicy,
)
from kliptych.runtime.recorded import (
    RecordedCaptionDocument,
    RecordedDocument,
    RecordedModel,
    record_caption,
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
    "CAPTION_PROMPT_VERSION",
    "DEFAULT_MAX_RESPONSE_BYTES",
    "ENV_API_KEY",
    "ENV_BASE_URL",
    "ENV_MODEL",
    "PROMPT_VERSION",
    "CampaignModel",
    "Caption",
    "HttpError",
    "HttpResponse",
    "HttpTransport",
    "ModelError",
    "ModelInputError",
    "ModelOutputError",
    "ModelUnavailableError",
    "OpenAIChatModel",
    "PieceContext",
    "RecordedCaptionDocument",
    "RecordedDocument",
    "RecordedModel",
    "RetryPolicy",
    "UrllibTransport",
    "brief_key",
    "normalize_brief",
    "record_caption",
    "record_response",
]
