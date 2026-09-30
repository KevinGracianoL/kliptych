"""Sprint 2 (Objetivo 3): brand safety opt-in con LLM y fail-closed.

Solo se activa si el contrato prohíbe explícitamente la controversia o
exige brand safety (menciones en ``prohibitions``); en campañas estilo
CB20 el evaluador ni se invoca. Con la regla activa, riesgo del LLM o
cualquier fallo del evaluador (excepción, timeout, JSON inválido) exige
revisión humana: jamás un ``pass`` silencioso.
"""

from pathlib import Path
from typing import cast, override

import pytest

from kliptych.assets import AssetRegistry
from kliptych.campaign_manager import CampaignManager
from kliptych.campaign_types import Campaign, CampaignStatus
from kliptych.config import Settings
from kliptych.contract import Contract, ContractDraft, Platform
from kliptych.environment import EnvironmentReport
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
from kliptych.git_proposals import ProposalEngine, PullRequest
from kliptych.intelligence import Archetype, ArchetypeClassification
from kliptych.orchestrator import PipelineResult, SlideshowResult
from kliptych.pipeline import RunOutcome, RunRequest, run_given_clips
from kliptych.resolver import resolve_contract
from kliptych.runtime import (
    CAPTION_PROMPT_VERSION,
    PROMPT_VERSION,
    CampaignModel,
    Caption,
    PieceContext,
    openai_compatible,
)
from kliptych.segment import SegmentSelection
from tests.support import (
    FakeProbe,
    candidate,
    make_asset_draft,
    make_contract,
    make_draft,
    make_media,
    make_piece,
)


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


@pytest.mark.parametrize(
    "phrase",
    [
        "sin groserias",
        "sin groserías",
        "contenido apto para marcas",
        "apto para todo publico",
        "apto para todo público",
        "family friendly",
        "nada de lenguaje ofensivo",
        "sin lenguaje ofensivo",
        "sin insultos",
        "no profanity",
    ],
)
def test_h5_expanded_brand_safety_phrases_activate_rule(tmp_path: Path, phrase: str) -> None:
    def _risky(text: str) -> BrandSafetyAssessment:
        _ = text
        return BrandSafetyAssessment(risk=True, categories=("profanity",), reason="inapropiado")

    status, check = _run_gate(tmp_path, prohibitions=(phrase,), assess=_risky)
    assert check is CheckStatus.MANUAL_REVIEW
    assert status is GateStatus.PENDING_REVIEW


class _ChatCampaignModel(CampaignModel):
    """Modelo de prueba para pipeline con chat_json."""

    model_version: str = "chat-test"

    def __init__(self, draft: ContractDraft, caption: Caption, chat_payload: object) -> None:
        self._draft: ContractDraft = draft
        self._caption: Caption = caption
        self._chat_payload: object = chat_payload
        self.chat_calls: list[str] = []

    @override
    def extract_contract(self, brief: str) -> ContractDraft:
        _ = brief
        return self._draft

    @override
    def write_caption(self, contract: Contract, piece: PieceContext) -> Caption:
        _ = (contract, piece)
        return self._caption

    def chat_json(self, *, system_prompt: str, user_content: str) -> object:
        _ = system_prompt
        self.chat_calls.append(user_content)
        return self._chat_payload


class _StubAssembler:
    """Ensamblador falso para tests sin ffmpeg."""

    def assemble(
        self,
        *,
        clip: Path,
        destination: Path,
        watermark: Path | None = None,
        watermark_config: object = None,
        width: int = 1080,
        height: int = 1920,
        subtitles: Path | None = None,
        mute_audio: bool = False,
    ) -> Path:
        _ = (self, clip, watermark, watermark_config, width, height, subtitles, mute_audio)
        destination.parent.mkdir(parents=True, exist_ok=True)
        _ = destination.write_bytes(b"assembled")
        return destination

    def render_arguments(self, **kwargs: object) -> tuple[str, ...]:
        _ = (self, kwargs)
        return ("ffmpeg", "assembled")


def test_h4_pipeline_injects_brand_safety_assessor_from_model(tmp_path: Path) -> None:
    _ = (tmp_path / "clip.mp4").write_bytes(b"clip")
    draft = make_draft(
        prohibitions=candidate(["sin groserias"]),
        assets={"required": [make_asset_draft()], "optional": []},
    )
    model = _ChatCampaignModel(
        draft,
        Caption(caption="mira @marca #marca", hashtags=("#marca",)),
        {"risk": False, "categories": [], "reason": None},
    )
    request = RunRequest(
        brief="cita del brief\nbrief sin groserias",
        destination=tmp_path / "delivery",
        environment=EnvironmentReport(),
        model_version="chat-test",
        prompt_version=PROMPT_VERSION,
        caption_prompt_version=CAPTION_PROMPT_VERSION,
        assembler=_StubAssembler(),
        gate=Gate(FakeProbe(info=make_media())),
    )
    result = run_given_clips(
        model=model,
        settings=Settings.from_root(tmp_path),
        request=request,
    )
    assert result.outcome is RunOutcome.EXPORTED
    assert len(model.chat_calls) >= 1
    assert result.delivery is not None
    assert result.delivery.exported[0].gate_status is GateStatus.PASSED


class _StubClassifier:
    def classify(self, brief: str, contract: Contract) -> ArchetypeClassification:
        _ = (self, brief, contract)
        return ArchetypeClassification(archetype=Archetype.KNOWN, rationale="t", variations=())


class _StubProvider:
    def create_branch(self, *, base: str, name: str) -> str:
        _ = (self, base, name)
        msg = "no git"
        raise AssertionError(msg)

    def read_file(self, *, branch: str, path: str) -> str | None:
        _ = (self, branch, path)
        return None

    def write_file(self, *, branch: str, path: str, content: str, message: str) -> str:
        _ = (self, branch, path, content, message)
        msg = "no git"
        raise AssertionError(msg)

    def open_pull_request(self, **kwargs: object) -> PullRequest:
        _ = (self, kwargs)
        msg = "no PR"
        raise AssertionError(msg)


class _StubVideoOrchestrator:
    def __init__(self, final: Path) -> None:
        self._final: Path = final

    def run_long_video(self, url: str, **kwargs: object) -> PipelineResult:
        _ = (url, kwargs)
        return PipelineResult(
            source=self._final,
            transcript=None,
            moments=(),
            selection=SegmentSelection(segments=(), rationale="t"),
            reframe=None,
            subtitles=None,
            final_video=self._final,
            cleaning=(),
        )

    def run_slideshow(self, images: object, **kwargs: object) -> SlideshowResult:
        _ = (self, images, kwargs)
        msg = "no slideshow"
        raise AssertionError(msg)


def test_h4_campaign_manager_injects_brand_safety_assessor_from_model(tmp_path: Path) -> None:
    final = tmp_path / "final.mp4"
    _ = final.write_bytes(b"final")
    chat_calls: list[str] = []

    class _ChatBackend:
        def chat_json(self, *, system_prompt: str, user_content: str) -> object:
            _ = (self, system_prompt)
            chat_calls.append(user_content)
            return {"risk": True, "categories": ["profanity"], "reason": "inapropiado"}

    manager = CampaignManager(
        classifier=_StubClassifier(),
        proposal_engine=ProposalEngine(provider=_StubProvider()),
        video_orchestrator=_StubVideoOrchestrator(final),
        gate=Gate(FakeProbe(info=make_media())),
        assets=AssetRegistry(tmp_path),
        destination=tmp_path / "delivery",
        model=_ChatBackend(),
    )
    campaign = Campaign(
        campaign_id="camp-01",
        brief="brief crudo",
        contract=make_contract(
            required_mentions=["@marca"],
            required_hashtags=["#marca"],
            audio_rule="any",
            prohibitions=["sin groserias"],
            hard=["artifact.integrity", "brand.safety"],
        ),
    )
    outcome = manager.process(campaign, mode="long_video", url="https://example.com/video")
    assert outcome.status is CampaignStatus.BLOCKED
    assert len(chat_calls) >= 1
    assert outcome.delivery_report is not None
    rejected = outcome.delivery_report.rejected[0]
    matched = [c for c in rejected.gate.checks if c.id == "brand.safety"]
    assert matched
    assert matched[0].status is CheckStatus.MANUAL_REVIEW


@pytest.mark.parametrize(
    "phrase",
    [
        "sin malas palabras",
        "malas palabras",
        "prohibido contenido nsfw",
        "nsfw",
        "sin temas sensibles",
        "temas sensibles",
        "no politica ni religion",
        "no contenido para adultos",
        "contenido para adultos",
        "no insultos",
        "sin insultos",
    ],
)
def test_h5_bis_hal_cases_activate_brand_safety(tmp_path: Path, phrase: str) -> None:
    def _risky(text: str) -> BrandSafetyAssessment:
        _ = text
        return BrandSafetyAssessment(risk=True, categories=("sensitive",), reason="inapropiado")

    status, check = _run_gate(tmp_path, prohibitions=(phrase,), assess=_risky)
    assert check is CheckStatus.MANUAL_REVIEW
    assert status is GateStatus.PENDING_REVIEW


def test_h5_bis_brand_safety_required_field_activates_rule(tmp_path: Path) -> None:
    def _risky(text: str) -> BrandSafetyAssessment:
        _ = text
        return BrandSafetyAssessment(risk=True, categories=("profanity",), reason="inapropiado")

    contract = make_contract(
        hard=["artifact.integrity", "brand.safety"],
        audio_rule="any",
        min_s=None,
        max_s=None,
        required_mentions=(),
        required_hashtags=(),
        prohibitions=(),
        brand_safety_required=True,
        brand_safety_citation="sin temas sensibles",
    )
    piece = make_piece(_artifact(tmp_path), subtitle_text="hola a todos")
    validators = {**DEFAULT_VALIDATORS, "brand.safety": make_brand_safety_validator(_risky)}
    result = Gate(FakeProbe(info=make_media()), validators=validators).run(
        contract=contract, piece=piece, assets=AssetRegistry(tmp_path)
    )
    matches = [check for check in result.checks if check.id == "brand.safety"]
    assert len(matches) == 1
    assert matches[0].status is CheckStatus.MANUAL_REVIEW
    assert result.status is GateStatus.PENDING_REVIEW


def _prompt_text(name: str) -> str:
    return cast("str", getattr(openai_compatible, name))


def test_h5_bis_extract_prompt_instructs_brand_safety_required() -> None:
    prompt = _prompt_text("_EXTRACT_SYSTEM_PROMPT")
    assert "brand_safety_required" in prompt
    assert "brand_safety_citation" in prompt


def test_h5_bis_resolver_resolves_brand_safety_from_draft(tmp_path: Path) -> None:
    draft = make_draft(
        brand_safety_required=candidate(value=True, quote="evitar contenido sensible"),
        brand_safety_citation=candidate("evitar contenido sensible", "evitar contenido sensible"),
    )
    result = resolve_contract(draft, registry=AssetRegistry(tmp_path))
    assert result is not None
    assert result.contract is not None
    contract = result.contract
    assert contract.brand_safety_required is True
    assert contract.brand_safety_citation == "evitar contenido sensible"
    assert "brand.safety" in contract.rules.hard
