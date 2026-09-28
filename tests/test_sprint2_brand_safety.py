"""Sprint 2 (Objetivo 3): brand safety opt-in con LLM y fail-closed.

Solo se activa si el contrato prohíbe explícitamente la controversia o
exige brand safety (menciones en ``prohibitions``); en campañas estilo
CB20 el evaluador ni se invoca. Con la regla activa, riesgo del LLM o
cualquier fallo del evaluador (excepción, timeout, JSON inválido) exige
revisión humana: jamás un ``pass`` silencioso.
"""

from pathlib import Path

import pytest

from kliptych.assets import AssetRegistry
from kliptych.contract import Contract, Platform
from kliptych.gate import (
    DEFAULT_VALIDATORS,
    CheckStatus,
    Gate,
    GateStatus,
    check_brand_safety,
    make_brand_safety_validator,
)
from kliptych.gate.brand_safety import (
    Assessor,
    BrandSafetyAssessment,
    BrandSafetyError,
    make_model_assessor,
    parse_brand_safety_response,
)
from kliptych.gate.checks import GateContext
from tests.support import FakeProbe, make_contract, make_media, make_piece


def _artifact(tmp_path: Path) -> Path:
    path = tmp_path / "piece.mp4"
    _ = path.write_bytes(b"video")
    return path


def _safety_contract(*, prohibitions: tuple[str, ...]) -> Contract:
    return make_contract(
        hard=["artifact.integrity", "brand.safety"],
        audio_rule="any",
        min_s=None,
        max_s=None,
        required_mentions=(),
        required_hashtags=(),
        prohibitions=prohibitions,
    )


def _direct_context(tmp_path: Path, *, prohibitions: tuple[str, ...]) -> GateContext:
    contract = _safety_contract(prohibitions=prohibitions)
    return GateContext(
        contract=contract,
        rules=contract.platforms[Platform.TIKTOK],
        piece=make_piece(_artifact(tmp_path)),
        artifact_sha256="a" * 64,
        media=make_media(),
        assets=AssetRegistry(tmp_path),
    )


class _ExplodingAssessor:
    """Evaluador que registra si fue invocado y explota si lo fue."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def __call__(self, text: str) -> BrandSafetyAssessment:
        self.calls.append(text)
        msg = "el evaluador no debió invocarse"
        raise AssertionError(msg)


class _FakeModel:
    """Doble del backend chat_json con respuesta grabada."""

    def __init__(self, payload: object) -> None:
        self.payload: object = payload
        self.prompts: list[str] = []

    def chat_json(self, *, system_prompt: str, user_content: str) -> object:
        _ = system_prompt
        self.prompts.append(user_content)
        return self.payload


def _run_gate(
    tmp_path: Path, *, prohibitions: tuple[str, ...], assess: Assessor
) -> tuple[GateStatus, CheckStatus]:
    contract = _safety_contract(prohibitions=prohibitions)
    piece = make_piece(_artifact(tmp_path), subtitle_text="hola a todos")
    validators = {**DEFAULT_VALIDATORS, "brand.safety": make_brand_safety_validator(assess)}
    result = Gate(FakeProbe(info=make_media()), validators=validators).run(
        contract=contract, piece=piece, assets=AssetRegistry(tmp_path)
    )
    matches = [check for check in result.checks if check.id == "brand.safety"]
    assert len(matches) == 1
    return result.status, matches[0].status


def test_brand_safety_registered_by_default() -> None:
    assert DEFAULT_VALIDATORS["brand.safety"] is check_brand_safety


def test_brand_safety_inactive_for_cb20_style_campaign(tmp_path: Path) -> None:
    assessor = _ExplodingAssessor()
    contract = _safety_contract(prohibitions=("sorteo", "estafa"))
    piece = make_piece(_artifact(tmp_path))
    validators = {**DEFAULT_VALIDATORS, "brand.safety": make_brand_safety_validator(assessor)}
    result = Gate(FakeProbe(info=make_media()), validators=validators).run(
        contract=contract, piece=piece, assets=AssetRegistry(tmp_path)
    )
    matches = [check for check in result.checks if check.id == "brand.safety"]
    assert len(matches) == 1
    assert matches[0].status is CheckStatus.PASS
    assert assessor.calls == []


@pytest.mark.parametrize(
    "prohibition", ["sin controversia", "evitar polémica", "brand safety", "contenido tóxico"]
)
def test_brand_safety_active_markers_call_assessor(tmp_path: Path, prohibition: str) -> None:
    seen: list[str] = []

    def _clean(text: str) -> BrandSafetyAssessment:
        seen.append(text)
        return BrandSafetyAssessment(risk=False)

    status, check = _run_gate(tmp_path, prohibitions=(prohibition,), assess=_clean)
    assert check is CheckStatus.PASS
    assert status is GateStatus.PASSED
    assert len(seen) == 1


def test_brand_safety_risk_needs_review(tmp_path: Path) -> None:
    def _risky(text: str) -> BrandSafetyAssessment:
        _ = text
        return BrandSafetyAssessment(
            risk=True, categories=("toxicity",), reason="insultos en subtítulos"
        )

    status, check = _run_gate(tmp_path, prohibitions=("sin controversia",), assess=_risky)
    assert check is CheckStatus.MANUAL_REVIEW
    assert status is GateStatus.PENDING_REVIEW


@pytest.mark.parametrize("error", [TimeoutError("lento"), ConnectionError("caído")])
def test_brand_safety_assessor_error_needs_review(tmp_path: Path, error: Exception) -> None:
    def _boom(text: str) -> BrandSafetyAssessment:
        _ = text
        raise error

    _, check = _run_gate(tmp_path, prohibitions=("sin controversia",), assess=_boom)
    assert check is CheckStatus.MANUAL_REVIEW


def test_brand_safety_without_assessor_needs_review(tmp_path: Path) -> None:
    outcome = check_brand_safety(_direct_context(tmp_path, prohibitions=("sin controversia",)))
    assert outcome.status is CheckStatus.MANUAL_REVIEW


def test_brand_safety_invalid_json_needs_review(tmp_path: Path) -> None:
    model = _FakeModel({"veredicto": "peligroso"})
    _, check = _run_gate(
        tmp_path,
        prohibitions=("sin controversia",),
        assess=make_model_assessor(model),
    )
    assert check is CheckStatus.MANUAL_REVIEW


def test_model_assessor_parses_valid_response() -> None:
    model = _FakeModel({"risk": True, "categories": ["toxicity"], "reason": "x"})
    assessment = make_model_assessor(model)("caption hola")
    assert assessment == BrandSafetyAssessment(risk=True, categories=("toxicity",), reason="x")
    assert "caption hola" in model.prompts[0]


def test_model_assessor_rejects_malformed_payload() -> None:
    model = _FakeModel({"risk": "quizás"})
    with pytest.raises(BrandSafetyError):
        _ = make_model_assessor(model)("caption hola")


def test_parse_brand_safety_response_accepts_minimal() -> None:
    assert parse_brand_safety_response({"risk": False}) == BrandSafetyAssessment(risk=False)


@pytest.mark.parametrize(
    "payload",
    [
        {"categories": ["toxicity"]},
        {"risk": 1},
        {"risk": None},
        {"risk": True, "categories": "toxicity"},
        {"risk": True, "categories": [7]},
        {"risk": False, "reason": 7},
        ["risk"],
        "risk",
        None,
    ],
)
def test_parse_brand_safety_response_rejects_malformed(payload: object) -> None:
    with pytest.raises(BrandSafetyError):
        _ = parse_brand_safety_response(payload)
