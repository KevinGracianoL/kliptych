"""Tests de la clasificación de arquetipos de campaña (fase E).

La clasificación asistida por LLM se prueba con un transporte HTTP falso: cero
red y cero llamadas a modelos reales.
"""

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import cast

import pytest
from pydantic import ValidationError

from kliptych.campaign_types import Campaign, CampaignStatus, PendingCampaign, route_campaign
from kliptych.intelligence import (
    CLASSIFY_PROMPT_VERSION,
    Archetype,
    ArchetypeClassification,
    CampaignClassifier,
    LLMCampaignClassifier,
    classify_prompt_payload,
)
from kliptych.runtime import (
    HttpError,
    HttpResponse,
    ModelOutputError,
    ModelUnavailableError,
    RetryPolicy,
)
from tests.support import make_asset_ref, make_contract


@dataclass
class FakeTransport:
    responses: list[object] = field(default_factory=list)
    calls: list[dict[str, object]] = field(default_factory=list)

    def post_json(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        payload: Mapping[str, object],
        timeout_s: float,
    ) -> HttpResponse:
        self.calls.append(
            {
                "url": url,
                "headers": dict(headers),
                "payload": dict(payload),
                "timeout_s": timeout_s,
            }
        )
        assert self.responses, "el fake no tiene respuestas configuradas"
        response = self.responses.pop(0)
        if isinstance(response, HttpError):
            raise response
        assert isinstance(response, HttpResponse)
        return response


def _chat_response(content: str) -> HttpResponse:
    body = json.dumps({"choices": [{"message": {"content": content}}]}).encode("utf-8")
    return HttpResponse(status=200, body=body)


def _classification_json(
    archetype: str = "KNOWN",
    rationale: str = "encaja limpio en el contrato",
    variations: list[str] | None = None,
) -> str:
    return json.dumps(
        {
            "archetype": archetype,
            "rationale": rationale,
            "variations": [] if variations is None else variations,
        }
    )


def _classifier(transport: FakeTransport, *, max_attempts: int = 3) -> LLMCampaignClassifier:
    return LLMCampaignClassifier(
        base_url="https://llm.example/v1",
        api_key="secret-key",
        model="model-test",
        transport=transport,
        policy=RetryPolicy(max_attempts=max_attempts, backoff_s=0.0, timeout_s=5.0),
    )


def _sent_payload(transport: FakeTransport, index: int = 0) -> dict[str, object]:
    return cast("dict[str, object]", transport.calls[index]["payload"])


def _user_content(transport: FakeTransport, index: int = 0) -> str:
    payload = _sent_payload(transport, index)
    messages = cast("list[dict[str, str]]", payload["messages"])
    return messages[1]["content"]


def _rendered_chain_contains(error: BaseException, needle: str) -> bool:
    cause: BaseException | None = error
    while cause is not None:
        if needle in str(cause):
            return True
        cause = cause.__cause__
    return False


def test_archetype_enum_values() -> None:
    assert {archetype.value for archetype in Archetype} == {
        "KNOWN",
        "KNOWN_WITH_VARIATION",
        "NEW_ARCHETYPE",
    }


def test_classification_is_frozen() -> None:
    classification = ArchetypeClassification(archetype=Archetype.KNOWN, rationale="ok")
    with pytest.raises(ValidationError):
        classification.archetype = Archetype.NEW_ARCHETYPE


def test_classification_forbids_extra_fields() -> None:
    with pytest.raises(ValidationError, match="invented"):
        _ = ArchetypeClassification.model_validate(
            {"archetype": "KNOWN", "rationale": "ok", "invented": True}
        )


def test_classification_requires_rationale() -> None:
    with pytest.raises(ValidationError, match="rationale"):
        _ = ArchetypeClassification(archetype=Archetype.KNOWN, rationale="")


def test_known_with_variation_requires_variations() -> None:
    with pytest.raises(ValidationError, match="variación"):
        _ = ArchetypeClassification(
            archetype=Archetype.KNOWN_WITH_VARIATION, rationale="valor nuevo"
        )


@pytest.mark.parametrize("archetype", [Archetype.KNOWN, Archetype.NEW_ARCHETYPE])
def test_variations_only_allowed_for_known_with_variation(archetype: Archetype) -> None:
    with pytest.raises(ValidationError, match="KNOWN_WITH_VARIATION"):
        _ = ArchetypeClassification(
            archetype=archetype, rationale="ok", variations=("duracion 45s",)
        )


def test_known_with_variation_captures_variations() -> None:
    classification = ArchetypeClassification(
        archetype=Archetype.KNOWN_WITH_VARIATION,
        rationale="valor nuevo en duration.max",
        variations=("duration.max=45",),
    )
    assert classification.variations == ("duration.max=45",)


@pytest.mark.parametrize(
    ("archetype", "expected"),
    [
        (Archetype.KNOWN, CampaignStatus.CLASSIFIED),
        (Archetype.KNOWN_WITH_VARIATION, CampaignStatus.PENDING),
        (Archetype.NEW_ARCHETYPE, CampaignStatus.MANUAL_REVIEW),
    ],
)
def test_route_campaign(archetype: Archetype, expected: CampaignStatus) -> None:
    variations = ("variacion",) if archetype is Archetype.KNOWN_WITH_VARIATION else ()
    classification = ArchetypeClassification(
        archetype=archetype, rationale="ok", variations=variations
    )
    assert route_campaign(classification) is expected


def test_classify_returns_classification() -> None:
    transport = FakeTransport(responses=[_chat_response(_classification_json())])
    classification = _classifier(transport).classify("brief de prueba", make_contract())
    expected = ArchetypeClassification(
        archetype=Archetype.KNOWN, rationale="encaja limpio en el contrato"
    )
    assert classification == expected
    assert len(transport.calls) == 1
    call = transport.calls[0]
    assert call["url"] == "https://llm.example/v1/chat/completions"
    headers = cast("dict[str, str]", call["headers"])
    assert headers["Authorization"] == "Bearer secret-key"
    payload = _sent_payload(transport)
    assert payload["model"] == "model-test"
    assert payload["temperature"] == 0
    assert payload["response_format"] == {"type": "json_object"}
    assert cast("float", call["timeout_s"]) == pytest.approx(5.0)


def test_classify_prompt_contains_brief_and_contract() -> None:
    transport = FakeTransport(responses=[_chat_response(_classification_json())])
    _ = _classifier(transport).classify("brief de prueba", make_contract())
    content = _user_content(transport)
    assert "clasificador de arquetipos" in json.dumps(_sent_payload(transport)["messages"])
    assert "brief de prueba" in content
    sent = cast("dict[str, object]", json.loads(content))
    assert sent["brief"] == "brief de prueba"
    contract = cast("dict[str, object]", sent["contract"])
    assert contract["campaign_id"] == "camp-test"


def test_classify_prompt_excludes_asset_uris() -> None:
    contract = make_contract(required_assets=[make_asset_ref()])
    transport = FakeTransport(responses=[_chat_response(_classification_json())])
    _ = _classifier(transport).classify("brief", contract)
    content = _user_content(transport)
    assert "clip.mp4" not in content
    sent = cast("dict[str, object]", json.loads(content))
    sent_contract = cast("dict[str, object]", sent["contract"])
    assert "assets" not in sent_contract


def test_classify_prompt_payload_is_serializable() -> None:
    payload = classify_prompt_payload("brief", make_contract())
    assert payload["brief"] == "brief"
    assert json.dumps(payload)


def test_classify_strips_code_fences() -> None:
    content = f"```json\n{_classification_json()}\n```"
    transport = FakeTransport(responses=[_chat_response(content)])
    classification = _classifier(transport).classify("brief", make_contract())
    assert classification.archetype is Archetype.KNOWN


def test_classify_retries_invalid_output_then_success() -> None:
    transport = FakeTransport(
        responses=[
            _chat_response("no soy json"),
            _chat_response(_classification_json(archetype="NEW_ARCHETYPE")),
        ]
    )
    classification = _classifier(transport).classify("brief", make_contract())
    assert classification.archetype is Archetype.NEW_ARCHETYPE
    assert len(transport.calls) == 2


def test_classify_retries_on_server_error_then_success() -> None:
    transport = FakeTransport(
        responses=[HttpResponse(status=500, body=b""), _chat_response(_classification_json())]
    )
    classification = _classifier(transport).classify("brief", make_contract())
    assert classification.archetype is Archetype.KNOWN
    assert len(transport.calls) == 2


def test_classify_retries_on_transport_error_then_success() -> None:
    transport = FakeTransport(
        responses=[HttpError("timeout"), _chat_response(_classification_json())]
    )
    classification = _classifier(transport).classify("brief", make_contract())
    assert classification.archetype is Archetype.KNOWN
    assert len(transport.calls) == 2


def test_classify_backoff_grows_exponentially_between_attempts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sleeps: list[float] = []
    monkeypatch.setattr("kliptych.intelligence.time.sleep", sleeps.append)
    transport = FakeTransport(
        responses=[
            HttpResponse(status=500, body=b""),
            HttpResponse(status=500, body=b""),
            _chat_response(_classification_json()),
        ]
    )
    classifier = LLMCampaignClassifier(
        base_url="https://llm.example/v1",
        api_key="secret-key",
        model="model-test",
        transport=transport,
        policy=RetryPolicy(max_attempts=3, backoff_s=0.5, timeout_s=5.0),
    )
    assert classifier.classify("brief", make_contract()).archetype is Archetype.KNOWN
    assert sleeps == [0.5, 1.0]


def test_classify_exhausted_server_errors_raise_unavailable() -> None:
    transport = FakeTransport(
        responses=[
            HttpResponse(status=500, body=b""),
            HttpResponse(status=500, body=b""),
            HttpResponse(status=500, body=b""),
        ]
    )
    with pytest.raises(ModelUnavailableError, match="intentos"):
        _ = _classifier(transport).classify("brief", make_contract())
    assert len(transport.calls) == 3


def test_classify_client_error_is_not_retried() -> None:
    transport = FakeTransport(responses=[HttpResponse(status=401, body=b"")])
    with pytest.raises(ModelUnavailableError, match="401"):
        _ = _classifier(transport).classify("brief", make_contract())
    assert len(transport.calls) == 1


def test_classify_output_errors_do_not_leak_model_content() -> None:
    canary = "CANARIO123"
    content = _classification_json(archetype=canary)
    transport = FakeTransport(responses=[_chat_response(content)])
    with pytest.raises(ModelOutputError) as excinfo:
        _ = _classifier(transport, max_attempts=1).classify("brief", make_contract())
    assert not _rendered_chain_contains(excinfo.value, canary)


def test_classify_malformed_completion_body_raises_output_error() -> None:
    body = json.dumps({"choices": [{"message": "texto plano"}]}).encode("utf-8")
    transport = FakeTransport(responses=[HttpResponse(status=200, body=body)])
    with pytest.raises(ModelOutputError, match="choices válidos"):
        _ = _classifier(transport, max_attempts=1).classify("brief", make_contract())


def test_classify_response_without_choices_raises_output_error() -> None:
    body = json.dumps({"choices": []}).encode("utf-8")
    transport = FakeTransport(responses=[HttpResponse(status=200, body=body)])
    with pytest.raises(ModelOutputError, match="choices"):
        _ = _classifier(transport, max_attempts=1).classify("brief", make_contract())


def test_classify_response_without_content_raises_output_error() -> None:
    body = json.dumps({"choices": [{"message": {"content": None}}]}).encode("utf-8")
    transport = FakeTransport(responses=[HttpResponse(status=200, body=body)])
    with pytest.raises(ModelOutputError, match="contenido"):
        _ = _classifier(transport, max_attempts=1).classify("brief", make_contract())


def test_classify_prompt_version_is_exposed() -> None:
    classifier = _classifier(FakeTransport(responses=[]))
    assert classifier.prompt_version == CLASSIFY_PROMPT_VERSION
    assert classifier.model_version == "model-test"


def test_llm_classifier_satisfies_protocol() -> None:
    transport = FakeTransport(responses=[_chat_response(_classification_json())])
    classifier: CampaignClassifier = _classifier(transport)
    assert classifier.classify("brief", make_contract()).archetype is Archetype.KNOWN


def test_pending_campaign_holds_campaign_and_reason() -> None:
    campaign = Campaign(campaign_id="camp-01", brief="brief")
    pending = PendingCampaign(campaign=campaign, reason="falta un modo")
    assert pending.campaign is campaign
    assert pending.reason == "falta un modo"


def test_pending_campaign_rejects_blank_reason() -> None:
    campaign = Campaign(campaign_id="camp-01", brief="brief")
    with pytest.raises(ValueError, match="reason"):
        _ = PendingCampaign(campaign=campaign, reason="")
