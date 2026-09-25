"""Tests del runtime LLM: backend OpenAI-compatible, grabaciones y fallback."""

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest
from pydantic import ValidationError

from kliptych.assets import AssetRegistry
from kliptych.contract import Platform
from kliptych.gate import CheckStatus, Gate, GateStatus
from kliptych.resolver import ResolutionStatus, resolve_contract
from kliptych.runtime import (
    CAPTION_PROMPT_VERSION,
    PROMPT_VERSION,
    Caption,
    HttpError,
    HttpResponse,
    ModelInputError,
    ModelOutputError,
    ModelUnavailableError,
    OpenAIChatModel,
    PieceContext,
    RecordedModel,
    RetryPolicy,
    record_caption,
    record_response,
)
from tests.support import (
    FakeProbe,
    make_asset_draft,
    make_asset_ref,
    make_contract,
    make_draft,
    make_media,
    make_piece,
    write_fixture_clip,
)

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


def test_output_errors_do_not_leak_model_content() -> None:
    canary = "CANARIO123"
    content = json.dumps({"platforms": {canary: {}}})
    transport = FakeTransport(responses=[_chat_response(content)])
    with pytest.raises(ModelOutputError) as excinfo:
        _ = _model(transport, max_attempts=1).extract_contract("brief")
    assert not _rendered_chain_contains(excinfo.value, canary)


def test_malformed_completion_does_not_leak_model_content() -> None:
    canary = "CANARIO123"
    body = ('{"choices": {"' + canary + '": {}}}').encode("utf-8")
    transport = FakeTransport(responses=[HttpResponse(status=200, body=body)])
    with pytest.raises(ModelOutputError) as excinfo:
        _ = _model(transport, max_attempts=1).extract_contract("brief")
    assert not _rendered_chain_contains(excinfo.value, canary)


def _rendered_chain_contains(error: BaseException, needle: str) -> bool:
    cause: BaseException | None = error
    while cause is not None:
        if needle in str(cause):
            return True
        cause = cause.__cause__
    return False


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
        model = RecordedModel.from_directory(
            directory,
            expected_prompt_version=PROMPT_VERSION,
            expected_caption_prompt_version=CAPTION_PROMPT_VERSION,
        )
        assert model.documents


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
    _ = write_fixture_clip(tmp_path)
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
    assert gate_result.status is GateStatus.PENDING_REVIEW
    audio_check = next(check for check in gate_result.checks if check.id == "audio.own_clip")
    assert audio_check.status is CheckStatus.MANUAL_REVIEW
    assert all(
        check.status is CheckStatus.PASS for check in gate_result.checks if check.id != "audio.own_clip"
    )


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


def _caption_json(
    caption: str = "Mira @marca #marca",
    hashtags: list[str] | None = None,
) -> str:
    return json.dumps(
        {"caption": caption, "hashtags": ["#marca"] if hashtags is None else hashtags}
    )


def _caption_piece(
    piece_id: str = "piece-01", platform: Platform = Platform.TIKTOK
) -> PieceContext:
    return PieceContext(piece_id=piece_id, platform=platform)


def _caption_user_content(transport: FakeTransport, index: int = 0) -> str:
    payload = cast("dict[str, object]", transport.calls[index]["payload"])
    return json.dumps(payload["messages"], ensure_ascii=False)


def test_caption_rejects_blank_text() -> None:
    with pytest.raises(ValidationError, match="caption"):
        _ = Caption(caption="")


@pytest.mark.parametrize(
    "hashtag", ["marca", "#marca extra", "#marca\n#otro", "# marca", "#marca "]
)
def test_caption_rejects_invalid_hashtag(hashtag: str) -> None:
    with pytest.raises(ValidationError, match="hashtags"):
        _ = Caption(caption="hola", hashtags=(hashtag,))


@pytest.mark.parametrize("hashtag", ["#marca", "#Marca", "#a", "#marca_2026"])
def test_caption_accepts_token_hashtags(hashtag: str) -> None:
    caption = Caption(caption="hola", hashtags=(hashtag,))
    assert caption.hashtags == (hashtag,)


def test_write_caption_returns_validated_caption() -> None:
    transport = FakeTransport(responses=[_chat_response(_caption_json())])
    caption = _model(transport).write_caption(make_contract(), _caption_piece())
    assert caption == Caption(caption="Mira @marca #marca", hashtags=("#marca",))
    assert len(transport.calls) == 1
    call = transport.calls[0]
    assert call["url"] == "https://llm.example/v1/chat/completions"
    payload = cast("dict[str, object]", call["payload"])
    assert payload["model"] == "model-test"
    assert payload["temperature"] == 0
    assert payload["response_format"] == {"type": "json_object"}
    content = _caption_user_content(transport)
    assert "redactor de captions" in content
    assert "@marca" in content
    assert "#marca" in content


def test_write_caption_does_not_send_asset_uris() -> None:
    contract = make_contract(required_assets=[make_asset_ref()])
    transport = FakeTransport(responses=[_chat_response(_caption_json())])
    _ = _model(transport).write_caption(contract, _caption_piece())
    content = _caption_user_content(transport)
    assert "clip.mp4" not in content
    assert "assets" not in content


def test_write_caption_pins_curated_payload_fields() -> None:
    contract = make_contract(
        required_mentions=("@marca",),
        required_hashtags=("#marca",),
        must_mention=("@marca",),
        forbidden=("estafa",),
        prohibitions=("no prometer resultados",),
        spelling_locks=("MarcaX",),
        first_line="Hola",
    )
    transport = FakeTransport(responses=[_chat_response(_caption_json())])
    _ = _model(transport).write_caption(contract, _caption_piece())
    payload = cast("dict[str, object]", transport.calls[0]["payload"])
    messages = cast("list[dict[str, str]]", payload["messages"])
    sent = cast("dict[str, object]", json.loads(messages[1]["content"]))
    assert sent == {
        "campaign_id": "camp-test",
        "platform": "tiktok",
        "piece_id": "piece-01",
        "languages": {"source": "es", "subtitles": None, "caption": "es", "voice": None},
        "caption_rules": {
            "must_mention": ["@marca"],
            "first_line": "Hola",
            "forbidden": ["estafa"],
        },
        "required_mentions": ["@marca"],
        "required_hashtags": ["#marca"],
        "prohibitions": ["no prometer resultados"],
        "spelling_locks": ["MarcaX"],
    }


def test_write_caption_strips_code_fences() -> None:
    content = f"```json\n{_caption_json()}\n```"
    transport = FakeTransport(responses=[_chat_response(content)])
    caption = _model(transport).write_caption(make_contract(), _caption_piece())
    assert caption.caption == "Mira @marca #marca"


def test_write_caption_retries_invalid_output_then_success() -> None:
    transport = FakeTransport(
        responses=[_chat_response("no soy json"), _chat_response(_caption_json())]
    )
    caption = _model(transport).write_caption(make_contract(), _caption_piece())
    assert caption.caption == "Mira @marca #marca"
    assert len(transport.calls) == 2


def test_write_caption_output_errors_do_not_leak_model_content() -> None:
    canary = "CANARIO123"
    content = json.dumps({"caption": "hola", "hashtags": [canary]})
    transport = FakeTransport(responses=[_chat_response(content)])
    with pytest.raises(ModelOutputError, match="Caption") as excinfo:
        _ = _model(transport, max_attempts=1).write_caption(make_contract(), _caption_piece())
    assert not _rendered_chain_contains(excinfo.value, canary)


def test_write_caption_rejects_undeclared_platform() -> None:
    piece = _caption_piece(platform=Platform.INSTAGRAM_REELS)
    transport = FakeTransport(responses=[])
    with pytest.raises(ModelInputError, match="instagram_reels"):
        _ = _model(transport).write_caption(make_contract(), piece)
    assert transport.calls == []


def test_recorded_write_caption_rejects_undeclared_platform() -> None:
    contract = make_contract()
    piece = _caption_piece(platform=Platform.INSTAGRAM_REELS)
    model = RecordedModel()
    with pytest.raises(ModelInputError, match="instagram_reels"):
        _ = model.write_caption(contract, piece)


def test_record_caption_rejects_undeclared_platform(tmp_path: Path) -> None:
    contract = make_contract()
    piece = _caption_piece(platform=Platform.INSTAGRAM_REELS)
    with pytest.raises(ModelInputError, match="instagram_reels"):
        _ = record_caption(
            contract,
            piece,
            Caption(caption="caption para x", hashtags=("#marca",)),
            prompt_version=CAPTION_PROMPT_VERSION,
            directory=tmp_path / "recorded",
        )


def test_model_exposes_caption_prompt_version() -> None:
    model = _model(FakeTransport(responses=[]))
    assert model.caption_prompt_version == CAPTION_PROMPT_VERSION


def test_record_and_replay_caption_round_trip(tmp_path: Path) -> None:
    contract = make_contract()
    piece = _caption_piece()
    expected = Caption(caption="Mira @marca", hashtags=("#marca",))
    directory = tmp_path / "recorded"
    path = record_caption(
        contract,
        piece,
        expected,
        prompt_version=CAPTION_PROMPT_VERSION,
        directory=directory,
    )
    assert path.parent == directory
    assert path.name.endswith(".caption.json")
    model = RecordedModel.from_directory(directory)
    assert model.write_caption(contract, piece) == expected


def test_caption_replay_is_stable_across_fresh_registries(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _ = (tmp_path / "clip.mp4").write_bytes(b"clip")
    draft = make_draft(assets={"required": [make_asset_draft()], "optional": []})
    first = resolve_contract(draft, registry=AssetRegistry(tmp_path))
    assert first.contract is not None

    def _frozen_now(_tz: object = None) -> datetime:
        return datetime(2026, 9, 22, tzinfo=UTC)

    monkeypatch.setattr("kliptych.assets.datetime", SimpleNamespace(now=_frozen_now))
    second = resolve_contract(draft, registry=AssetRegistry(tmp_path))
    assert second.contract is not None
    assert first.contract != second.contract
    piece = _caption_piece()
    expected = Caption(caption="Mira @marca", hashtags=("#marca",))
    directory = tmp_path / "recorded"
    _ = record_caption(
        first.contract,
        piece,
        expected,
        prompt_version=CAPTION_PROMPT_VERSION,
        directory=directory,
    )
    model = RecordedModel.from_directory(directory)
    assert model.write_caption(second.contract, piece) == expected


def test_caption_fixture_load_does_not_echo_content(tmp_path: Path) -> None:
    canary = "CANARIO-1234"
    directory = tmp_path / "recorded"
    directory.mkdir()
    payload = {
        "prompt_sha256": "a" * 64,
        "platform": "tiktok",
        "piece_id": "piece-01",
        "prompt_version": CAPTION_PROMPT_VERSION,
        "caption": {"caption": "hola", "hashtags": [canary]},
    }
    _ = (directory / "malo.caption.json").write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValidationError) as excinfo:
        _ = RecordedModel.from_directory(directory)
    assert not _rendered_chain_contains(excinfo.value, canary)


def test_replay_caption_missing_raises() -> None:
    model = RecordedModel()
    with pytest.raises(ModelUnavailableError, match="sin caption grabado"):
        _ = model.write_caption(make_contract(), _caption_piece())


def test_caption_replay_is_keyed_by_contract_and_piece(tmp_path: Path) -> None:
    contract = make_contract()
    piece = _caption_piece()
    other_contract = make_contract(required_hashtags=("#otra",))
    other_piece = _caption_piece(piece_id="piece-02")
    directory = tmp_path / "recorded"
    expected = Caption(caption="Mira @marca", hashtags=("#marca",))
    _ = record_caption(
        contract,
        piece,
        expected,
        prompt_version=CAPTION_PROMPT_VERSION,
        directory=directory,
    )
    model = RecordedModel.from_directory(directory)
    assert model.write_caption(contract, piece) == expected
    with pytest.raises(ModelUnavailableError, match="sin caption grabado"):
        _ = model.write_caption(other_contract, piece)
    with pytest.raises(ModelUnavailableError, match="sin caption grabado"):
        _ = model.write_caption(contract, other_piece)


def test_from_directory_rejects_stale_caption_prompt_version(tmp_path: Path) -> None:
    directory = tmp_path / "recorded"
    _ = record_caption(
        make_contract(),
        _caption_piece(),
        Caption(caption="Mira @marca", hashtags=("#marca",)),
        prompt_version="caption-v0",
        directory=directory,
    )
    with pytest.raises(ModelUnavailableError, match="captions con prompt obsoleto"):
        _ = RecordedModel.from_directory(
            directory,
            expected_caption_prompt_version=CAPTION_PROMPT_VERSION,
        )


def test_from_directory_rejects_unknown_json(tmp_path: Path) -> None:
    directory = tmp_path / "recorded"
    directory.mkdir()
    _ = (directory / "suelto.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ModelUnavailableError, match="no reconocidas"):
        _ = RecordedModel.from_directory(directory)


def test_replay_rejects_duplicate_captions(tmp_path: Path) -> None:
    directory = tmp_path / "recorded"
    _ = record_caption(
        make_contract(),
        _caption_piece(),
        Caption(caption="Mira @marca", hashtags=("#marca",)),
        prompt_version=CAPTION_PROMPT_VERSION,
        directory=directory,
    )
    document = next(iter(RecordedModel.from_directory(directory).captions.values()))
    with pytest.raises(ValueError, match="caption duplicado"):
        _ = RecordedModel(captions=[document, document])
