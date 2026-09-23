"""Tests del runtime LLM: backend OpenAI-compatible, grabaciones y fallback."""

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

import pytest
from pydantic import ValidationError

from kliptych.assets import AssetRegistry
from kliptych.gate import CheckStatus, Gate, GateStatus
from kliptych.resolver import ResolutionStatus, resolve_contract
from kliptych.runtime import (
    PROMPT_VERSION,
    HttpError,
    HttpResponse,
    ModelOutputError,
    ModelUnavailableError,
    OpenAIChatModel,
    RecordedModel,
    RetryPolicy,
    record_response,
)
from tests.support import FakeProbe, make_draft, make_media, make_piece

_FIXTURES = Path(__file__).resolve().parents[1] / "campaigns" / "fixtures" / "given-clips"


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


def _draft_json() -> str:
    return make_draft().model_dump_json()


def _model(transport: FakeTransport, *, max_attempts: int = 3) -> OpenAIChatModel:
    return OpenAIChatModel(
        base_url="https://llm.example/v1",
        api_key="secret-key",
        model="model-test",
        transport=transport,
        policy=RetryPolicy(max_attempts=max_attempts, backoff_s=0.0, timeout_s=5.0),
    )


def test_extract_contract_returns_validated_draft() -> None:
    transport = FakeTransport(responses=[_chat_response(_draft_json())])
    draft = _model(transport).extract_contract("brief de prueba")
    assert draft == make_draft()
    assert len(transport.calls) == 1
    call = transport.calls[0]
    assert call["url"] == "https://llm.example/v1/chat/completions"
    headers = cast("dict[str, str]", call["headers"])
    assert headers["Authorization"] == "Bearer secret-key"
    payload = cast("dict[str, object]", call["payload"])
    assert payload["model"] == "model-test"
    assert payload["temperature"] == 0
    assert payload["response_format"] == {"type": "json_object"}
    assert "brief de prueba" in json.dumps(payload["messages"], ensure_ascii=False)
    assert cast("float", call["timeout_s"]) == pytest.approx(5.0)


def test_extract_contract_strips_code_fences() -> None:
    content = f"```json\n{_draft_json()}\n```"
    transport = FakeTransport(responses=[_chat_response(content)])
    assert _model(transport).extract_contract("brief") == make_draft()


def test_model_versions_are_exposed() -> None:
    transport = FakeTransport(responses=[_chat_response(_draft_json())])
    model = _model(transport)
    assert model.prompt_version == PROMPT_VERSION
    assert model.model_version == "model-test"


def test_retry_on_server_error_then_success() -> None:
    transport = FakeTransport(
        responses=[HttpResponse(status=500, body=b""), _chat_response(_draft_json())]
    )
    assert _model(transport).extract_contract("brief") == make_draft()
    assert len(transport.calls) == 2


def test_retry_on_transport_error_then_success() -> None:
    transport = FakeTransport(responses=[HttpError("timeout"), _chat_response(_draft_json())])
    assert _model(transport).extract_contract("brief") == make_draft()
    assert len(transport.calls) == 2


def test_retry_on_rate_limit_then_success() -> None:
    transport = FakeTransport(
        responses=[HttpResponse(status=429, body=b""), _chat_response(_draft_json())]
    )
    assert _model(transport).extract_contract("brief") == make_draft()
    assert len(transport.calls) == 2


def test_client_error_is_not_retried() -> None:
    transport = FakeTransport(responses=[HttpResponse(status=401, body=b"")])
    with pytest.raises(ModelUnavailableError, match="401"):
        _ = _model(transport).extract_contract("brief")
    assert len(transport.calls) == 1


def test_exhausted_server_errors_raise_unavailable() -> None:
    transport = FakeTransport(
        responses=[
            HttpResponse(status=500, body=b""),
            HttpResponse(status=500, body=b""),
            HttpResponse(status=500, body=b""),
        ]
    )
    with pytest.raises(ModelUnavailableError, match="intentos"):
        _ = _model(transport).extract_contract("brief")
    assert len(transport.calls) == 3


def test_mixed_failure_sequence_reports_last_attempt() -> None:
    transport = FakeTransport(responses=[_chat_response("no soy json"), HttpError("boom")])
    with pytest.raises(ModelUnavailableError, match="intentos") as excinfo:
        _ = _model(transport, max_attempts=2).extract_contract("brief")
    assert isinstance(excinfo.value.__cause__, HttpError)
    assert len(transport.calls) == 2


def test_output_error_keeps_cause_chain() -> None:
    transport = FakeTransport(responses=[_chat_response("no soy json")])
    with pytest.raises(ModelOutputError, match="intentos") as excinfo:
        _ = _model(transport, max_attempts=1).extract_contract("brief")
    assert isinstance(excinfo.value.__cause__, ModelOutputError)
    assert excinfo.value.__cause__.__cause__ is not None


def test_invalid_json_output_raises_output_error() -> None:
    transport = FakeTransport(
        responses=[
            _chat_response("no soy json"),
            _chat_response("no soy json"),
            _chat_response("no soy json"),
        ]
    )
    with pytest.raises(ModelOutputError, match="intentos"):
        _ = _model(transport).extract_contract("brief")
    assert len(transport.calls) == 3


def test_draft_without_evidence_raises_output_error() -> None:
    content = json.dumps({"campaign_id": {"value": "camp-01", "confidence": "explicit"}})
    transport = FakeTransport(responses=[_chat_response(content)])
    with pytest.raises(ModelOutputError, match="ContractDraft"):
        _ = _model(transport, max_attempts=1).extract_contract("brief")
    assert len(transport.calls) == 1


def test_from_env_requires_all_variables() -> None:
    with pytest.raises(ModelUnavailableError, match="KLIPTYCH_LLM_API_KEY"):
        _ = OpenAIChatModel.from_env({"KLIPTYCH_LLM_BASE_URL": "https://llm.example/v1"})


def test_from_env_builds_model() -> None:
    model = OpenAIChatModel.from_env(
        {
            "KLIPTYCH_LLM_BASE_URL": "https://llm.example/v1",
            "KLIPTYCH_LLM_API_KEY": "secret",
            "KLIPTYCH_LLM_MODEL": "model-test",
        }
    )
    assert model.model_version == "model-test"
    assert model.prompt_version == PROMPT_VERSION


def test_record_and_replay_round_trip(tmp_path: Path) -> None:
    directory = tmp_path / "recorded"
    path = record_response(
        "brief grabado",
        make_draft(),
        prompt_version=PROMPT_VERSION,
        directory=directory,
    )
    assert path.parent == directory
    model = RecordedModel.from_directory(directory)
    assert model.extract_contract("brief grabado") == make_draft()


def test_replay_normalizes_line_endings(tmp_path: Path) -> None:
    directory = tmp_path / "recorded"
    _ = record_response(
        "linea 1\nlinea 2\n",
        make_draft(),
        prompt_version=PROMPT_VERSION,
        directory=directory,
    )
    model = RecordedModel.from_directory(directory)
    assert model.extract_contract("linea 1\r\nlinea 2\r\n") == make_draft()


def test_replay_normalizes_lone_carriage_return(tmp_path: Path) -> None:
    directory = tmp_path / "recorded"
    _ = record_response(
        "linea 1\rlinea 2\r",
        make_draft(),
        prompt_version=PROMPT_VERSION,
        directory=directory,
    )
    model = RecordedModel.from_directory(directory)
    assert model.extract_contract("linea 1\nlinea 2\n") == make_draft()


def test_from_directory_rejects_stale_prompt_version(tmp_path: Path) -> None:
    directory = tmp_path / "recorded"
    _ = record_response(
        "brief viejo",
        make_draft(),
        prompt_version="extract-v0",
        directory=directory,
    )
    with pytest.raises(ModelUnavailableError, match="obsoleto"):
        _ = RecordedModel.from_directory(directory, expected_prompt_version=PROMPT_VERSION)


def test_replay_missing_brief_raises() -> None:
    model = RecordedModel()
    with pytest.raises(ModelUnavailableError, match="sin respuesta grabada"):
        _ = model.extract_contract("brief desconocido")


def test_all_fixture_recordings_use_current_prompt() -> None:
    fixture_root = Path(__file__).resolve().parents[1] / "campaigns" / "fixtures"
    directories = sorted(fixture_root.glob("*/recorded"))
    assert directories
    for directory in directories:
        _ = RecordedModel.from_directory(directory, expected_prompt_version=PROMPT_VERSION)


def _iter_evidence(payload: object) -> list[dict[str, object]]:
    found: list[dict[str, object]] = []
    if isinstance(payload, dict):
        mapping = cast("dict[str, object]", payload)
        evidence = mapping.get("evidence")
        if isinstance(evidence, dict):
            found.append(cast("dict[str, object]", evidence))
        for value in mapping.values():
            found.extend(_iter_evidence(value))
    elif isinstance(payload, list):
        for item in cast("list[object]", payload):
            found.extend(_iter_evidence(item))
    return found


def test_fixture_evidence_is_grounded_in_brief() -> None:
    brief = (_FIXTURES / "brief.md").read_text(encoding="utf-8")
    model = RecordedModel.from_directory(_FIXTURES / "recorded")
    draft = model.extract_contract(brief)
    payload = cast("dict[str, object]", draft.model_dump(mode="json"))
    entries = _iter_evidence(payload)
    assert entries
    for evidence in entries:
        quote = cast("str", evidence["quote"])
        start = cast("int", evidence["start"])
        end = cast("int", evidence["end"])
        assert brief[start:end] == quote


def test_fixture_brief_flows_to_the_gate(tmp_path: Path) -> None:
    brief = (_FIXTURES / "brief.md").read_text(encoding="utf-8")
    model = RecordedModel.from_directory(_FIXTURES / "recorded")
    draft = model.extract_contract(brief)
    registry = AssetRegistry(tmp_path)
    result = resolve_contract(draft, registry=registry)
    assert result.status is ResolutionStatus.RESOLVED
    assert result.contract is not None
    artifact = tmp_path / "piece.mp4"
    _ = artifact.write_bytes(b"video")
    gate = Gate(FakeProbe(info=make_media()))
    gate_result = gate.run(
        contract=result.contract,
        piece=make_piece(artifact, caption="mira @marca #marca"),
        assets=registry,
    )
    assert gate_result.status is GateStatus.PASSED
    assert all(check.status is CheckStatus.PASS for check in gate_result.checks)


def test_recorded_document_rejects_bad_hash(tmp_path: Path) -> None:
    directory = tmp_path / "recorded"
    directory.mkdir()
    path = directory / "malo.draft.json"
    _ = path.write_text(
        json.dumps(
            {
                "brief_sha256": "abc",
                "prompt_version": PROMPT_VERSION,
                "draft": json.loads(make_draft().model_dump_json()),
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValidationError, match="brief_sha256"):
        _ = RecordedModel.from_directory(directory)


def test_replay_rejects_duplicate_documents(tmp_path: Path) -> None:
    _ = record_response(
        "mismo brief",
        make_draft(),
        prompt_version=PROMPT_VERSION,
        directory=tmp_path,
    )
    document = next(iter(RecordedModel.from_directory(tmp_path).documents.values()))
    with pytest.raises(ValueError, match="documento duplicado"):
        _ = RecordedModel([document, document])


def test_response_without_choices_raises_output_error() -> None:
    body = json.dumps({"choices": []}).encode("utf-8")
    transport = FakeTransport(responses=[HttpResponse(status=200, body=body)])
    with pytest.raises(ModelOutputError, match="choices"):
        _ = _model(transport, max_attempts=1).extract_contract("brief")


def test_response_without_content_raises_output_error() -> None:
    body = json.dumps({"choices": [{"message": {"content": None}}]}).encode("utf-8")
    transport = FakeTransport(responses=[HttpResponse(status=200, body=body)])
    with pytest.raises(ModelOutputError, match="contenido"):
        _ = _model(transport, max_attempts=1).extract_contract("brief")


def test_malformed_completion_body_raises_output_error() -> None:
    body = json.dumps({"choices": [{"message": "texto plano"}]}).encode("utf-8")
    transport = FakeTransport(responses=[HttpResponse(status=200, body=body)])
    with pytest.raises(ModelOutputError, match="choices válidos"):
        _ = _model(transport, max_attempts=1).extract_contract("brief")


def test_backoff_grows_exponentially_between_attempts(monkeypatch: pytest.MonkeyPatch) -> None:
    sleeps: list[float] = []
    monkeypatch.setattr("kliptych.runtime.openai_compatible.time.sleep", sleeps.append)
    transport = FakeTransport(
        responses=[
            HttpResponse(status=500, body=b""),
            HttpResponse(status=500, body=b""),
            _chat_response(_draft_json()),
        ]
    )
    model = OpenAIChatModel(
        base_url="https://llm.example/v1",
        api_key="secret-key",
        model="model-test",
        transport=transport,
        policy=RetryPolicy(max_attempts=3, backoff_s=0.5, timeout_s=5.0),
    )
    assert model.extract_contract("brief") == make_draft()
    assert sleeps == [0.5, 1.0]


def test_from_env_wires_url_key_transport_and_policy() -> None:
    transport = FakeTransport(responses=[_chat_response(_draft_json())])
    policy = RetryPolicy(max_attempts=1, backoff_s=0.0, timeout_s=7.5)
    model = OpenAIChatModel.from_env(
        {
            "KLIPTYCH_LLM_BASE_URL": "https://llm.example/v1",
            "KLIPTYCH_LLM_API_KEY": "clave-canario",
            "KLIPTYCH_LLM_MODEL": "model-test",
        },
        transport=transport,
        policy=policy,
    )
    assert model.extract_contract("brief") == make_draft()
    call = transport.calls[0]
    assert call["url"] == "https://llm.example/v1/chat/completions"
    headers = cast("dict[str, str]", call["headers"])
    assert headers["Authorization"] == "Bearer clave-canario"
    assert cast("float", call["timeout_s"]) == pytest.approx(7.5)
