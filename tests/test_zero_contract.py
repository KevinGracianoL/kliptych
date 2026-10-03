"""Tests del modo zero-contract: procesar con ``--url`` y sin llamar al LLM.

El modo zero-contract sustituye el brief de campaña por un contrato "vainilla"
sintetizado en memoria, de modo que ``--url`` basta para procesar un vídeo sin
ninguna llamada a la API del LLM ni credenciales ``KLIPTYCH_LLM_*``.
"""

import socket
from pathlib import Path
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

import pytest

from kliptych import __main__ as cli
from kliptych.campaign_manager import CampaignManager, CampaignOutcome
from kliptych.campaign_types import Campaign, CampaignStatus
from kliptych.contract import (
    AudioPolicy,
    Contract,
    Platform,
    contract_mutes_audio,
    vanilla_contract,
)
from kliptych.contract.schema import GlobalRestrictions, active_restriction_rules
from kliptych.gate.checks import DEFAULT_VALIDATORS
from kliptych.gate.engine import Gate
from kliptych.gate.models import GateStatus
from kliptych.git_proposals import ProposalEngine, PullRequest
from kliptych.hashing import brief_key
from kliptych.intelligence import (
    Archetype,
    ArchetypeClassification,
    CampaignClassifier,
    VanillaClassifier,
)
from kliptych.naming import is_safe_segment
from tests.support import FakeProbe, make_media, make_piece

if TYPE_CHECKING:
    from kliptych.orchestrator import PipelineResult, SlideshowResult

_AV_HARD_RULES = ("artifact.integrity", "artifact.video_stream")
_LLM_ENV_VARS = ("KLIPTYCH_LLM_BASE_URL", "KLIPTYCH_LLM_API_KEY", "KLIPTYCH_LLM_MODEL")


def _private(name: str) -> object:
    """Accede a un símbolo privado del módulo, como hace ``test_reframe``.

    Returns:
        El atributo pedido del módulo de la CLI.
    """
    return cast("object", getattr(cli, name))


_build_campaign_manager = cast(
    "Callable[..., CampaignManager]", _private("_build_campaign_manager")
)


@pytest.fixture
def _no_llm_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in _LLM_ENV_VARS:
        monkeypatch.delenv(name, raising=False)


class _ExplodingClassifier:
    """Clasificador que falla si el modo zero-contract llega a llamarlo."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def classify(self, brief: str, contract: Contract) -> ArchetypeClassification:
        _ = contract
        self.calls.append(brief)
        msg = "el modo zero-contract no debe clasificar"
        raise AssertionError(msg)


class _RecordingClassifier:
    def __init__(self, archetype: Archetype) -> None:
        self.calls: list[str] = []
        self._archetype: Archetype = archetype

    def classify(self, brief: str, contract: Contract) -> ArchetypeClassification:
        _ = contract
        self.calls.append(brief)
        return ArchetypeClassification(archetype=self._archetype, rationale="prueba", variations=())


class _StubOrchestrator:
    """Orquestador que registra los argumentos y se detiene a continuación."""

    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def run_long_video(self, url: str, **kwargs: object) -> "PipelineResult":
        self.calls.append({"url": url, **kwargs})
        msg = "parada deliberada tras inspeccionar los argumentos"
        raise RuntimeError(msg)

    @staticmethod
    def run_slideshow(images: "Sequence[Path]", **kwargs: object) -> "SlideshowResult":
        _ = (images, kwargs)
        msg = "el modo zero-contract nunca monta slideshows"
        raise AssertionError(msg)


class _StubGitProvider:
    """Proveedor de Git que falla si el modo zero-contract abre una PR.

    El camino zero-contract nunca debe proponer nada: no hay brief que clasificar
    y, por tanto, ningún arquetipo que pueda quedar fuera del catálogo.
    """

    @staticmethod
    def create_branch(*, base: str, name: str) -> str:
        _ = (base, name)
        msg = "el modo zero-contract no debe crear ramas"
        raise AssertionError(msg)

    @staticmethod
    def read_file(*, branch: str, path: str) -> str | None:
        _ = (branch, path)
        msg = "el modo zero-contract no debe leer del repositorio"
        raise AssertionError(msg)

    @staticmethod
    def write_file(*, branch: str, path: str, content: str, message: str) -> str:
        _ = (branch, path, content, message)
        msg = "el modo zero-contract no debe escribir en el repositorio"
        raise AssertionError(msg)

    @staticmethod
    def open_pull_request(
        *,
        branch: str,
        base: str,
        title: str,
        body: str,
        campaign_id: str,
        archetype: Archetype,
    ) -> PullRequest:
        _ = (branch, base, title, body, campaign_id, archetype)
        msg = "el modo zero-contract no debe abrir pull requests"
        raise AssertionError(msg)


def _manager(classifier: CampaignClassifier, orchestrator: _StubOrchestrator) -> CampaignManager:
    return CampaignManager(
        classifier=classifier,
        proposal_engine=ProposalEngine(provider=_StubGitProvider()),
        video_orchestrator=orchestrator,
    )


def _zero_contract_campaign() -> Campaign:
    return Campaign(
        campaign_id="abc123def456",
        brief="",
        contract=vanilla_contract(campaign_id="abc123def456"),
    )


# --- Contrato vainilla -------------------------------------------------------


def test_vanilla_contract_declares_only_av_integrity() -> None:
    """Exactamente las dos reglas de integridad A/V, y nada más.

    ``artifact.integrity`` verifica que el artefacto exista y sea legible;
    ``artifact.video_stream`` que tenga flujo de video. No hay una tercera.
    """
    contract = vanilla_contract(campaign_id="abc123def456")
    assert tuple(contract.rules.hard) == _AV_HARD_RULES
    assert contract.rules.recommended == []
    assert contract.rules.manual_review == []


def test_vanilla_contract_declares_no_audio_rule() -> None:
    """Ninguna regla de audio, en ninguna categoría.

    El contrato usa ``PlatformRules()``, cuyo ``audio_rule`` es ``ANY``: no hay
    audio que exigir ni que verificar. El resolver solo añade ``audio.present``
    cuando alguna plataforma exige audio (``resolver.py``), así que declararla
    aquí sería una aserción que el gate no puede sostener y que el propio
    pipeline nunca produciría.
    """
    contract = vanilla_contract(campaign_id="abc123def456")
    classified = [
        *contract.rules.hard,
        *contract.rules.recommended,
        *contract.rules.manual_review,
    ]
    assert "audio.present" not in classified
    assert not [rule for rule in classified if rule.startswith("audio.")]


def test_vanilla_contract_has_no_lexical_prohibitions() -> None:
    contract = vanilla_contract(campaign_id="abc123def456")
    assert contract.prohibitions == []
    assert contract.spelling_locks == []
    assert contract.hook_keyword is None


def test_vanilla_contract_disables_brand_safety() -> None:
    contract = vanilla_contract(campaign_id="abc123def456")
    assert contract.brand_safety_required is False
    assert contract.brand_safety_citation is None


def test_vanilla_contract_preserves_original_audio() -> None:
    contract = vanilla_contract(campaign_id="abc123def456")
    assert contract.audio_policy is AudioPolicy.ORIGINAL_AUDIO
    assert contract_mutes_audio(contract) is False


def test_vanilla_contract_requires_no_watermark() -> None:
    contract = vanilla_contract(campaign_id="abc123def456")
    assert contract.watermark.required is False
    assert contract.watermark.visible_full_video is False


def test_vanilla_contract_has_no_duration_or_caption_bounds() -> None:
    contract = vanilla_contract(campaign_id="abc123def456")
    rules = contract.platforms[Platform.TIKTOK]
    assert rules.duration.min_s is None
    assert rules.duration.max_s is None
    assert rules.required_hashtags == []
    assert rules.required_mentions == []
    assert rules.audio_rule.value == "any"


def test_vanilla_contract_activates_no_restriction_rules() -> None:
    """Las reglas A/V no son restricciones de campaña: no exigen clasificación.

    Si activaran alguna, el validador ``_declared_restrictions_are_classified``
    exigiría declararla y la construcción del contrato fallaría.
    """
    contract = vanilla_contract(campaign_id="abc123def456")
    restrictions = GlobalRestrictions(
        brand_safety_required=contract.brand_safety_required,
        audio_policy=contract.audio_policy,
        hook_keyword=contract.hook_keyword,
    )
    for rules in contract.platforms.values():
        assert active_restriction_rules(rules, restrictions) == []


def test_vanilla_contract_av_rules_all_have_a_registered_validator() -> None:
    """Cada regla declarada debe existir en el registro del gate."""
    for rule in _AV_HARD_RULES:
        assert rule in DEFAULT_VALIDATORS, rule


def test_vanilla_contract_rejects_empty_campaign_id() -> None:
    with pytest.raises(ValueError, match="campaign_id"):
        _ = vanilla_contract(campaign_id="")


@pytest.mark.usefixtures("_no_llm_env")
def test_vanilla_contract_is_deterministic() -> None:
    first = vanilla_contract(campaign_id="abc123def456")
    second = vanilla_contract(campaign_id="abc123def456")
    assert first.model_dump_json() == second.model_dump_json()


def test_vanilla_contract_accepts_another_platform() -> None:
    contract = vanilla_contract(campaign_id="abc123def456", platform=Platform.INSTAGRAM_REELS)
    assert tuple(contract.platforms) == (Platform.INSTAGRAM_REELS,)


# --- Gate --------------------------------------------------------------------


def _gate() -> Gate:
    return Gate(probe=FakeProbe(info=make_media()))


def test_gate_passes_vanilla_contract_on_av_integrity(tmp_path: Path) -> None:
    artifact = tmp_path / "final.mp4"
    _ = artifact.write_bytes(b"media-bytes")
    contract = vanilla_contract(campaign_id="abc123def456")
    result = _gate().evaluate_piece(make_piece(artifact), contract=contract)
    assert result.status is GateStatus.PASSED, result.checks
    assert {check.id for check in result.checks} == set(_AV_HARD_RULES)
    assert result.artifact_sha256 is not None


def test_gate_rejects_vanilla_contract_when_artifact_is_missing(tmp_path: Path) -> None:
    """``artifact.integrity`` es lo que evita el PASS en vacío sin pieza."""
    contract = vanilla_contract(campaign_id="abc123def456")
    missing = tmp_path / "no-existe.mp4"
    result = _gate().evaluate_piece(make_piece(missing), contract=contract)
    assert result.status is GateStatus.REJECTED


def test_gate_rejects_vanilla_contract_without_video_stream(tmp_path: Path) -> None:
    artifact = tmp_path / "final.mp4"
    _ = artifact.write_bytes(b"media-bytes")
    contract = vanilla_contract(campaign_id="abc123def456")
    gate = Gate(probe=FakeProbe(info=make_media(has_video=False)))
    result = gate.evaluate_piece(make_piece(artifact), contract=contract)
    assert result.status is GateStatus.REJECTED


def test_vanilla_contract_cannot_pass_without_an_artifact(tmp_path: Path) -> None:
    """La garantía real de A1: sin reglas, el gate aprobaría en vacío.

    ``_derive_status`` devuelve ``PASSED`` sobre cero checks, así que un
    contrato sin reglas aprobaría cualquier pieza, incluida una cuyo artefacto
    no existe. Estas dos reglas son lo que impide ese PASS, y es la razón por
    la que el contrato zero-contract no está realmente vacío.
    """
    contract = vanilla_contract(campaign_id="abc123def456")
    assert contract.rules.hard, "sin reglas el gate aprueba en vacío"
    missing = tmp_path / "no-existe.mp4"
    result = _gate().evaluate_piece(make_piece(missing), contract=contract)
    assert result.status is not GateStatus.PASSED
    assert result.status is GateStatus.REJECTED


def test_vanilla_contract_cannot_pass_without_a_video_stream(tmp_path: Path) -> None:
    """La segunda mitad de la garantía: el artefacto existe pero no tiene video."""
    artifact = tmp_path / "final.mp4"
    _ = artifact.write_bytes(b"media-bytes")
    contract = vanilla_contract(campaign_id="abc123def456")
    gate = Gate(probe=FakeProbe(info=make_media(has_video=False)))
    result = gate.evaluate_piece(make_piece(artifact), contract=contract)
    assert result.status is not GateStatus.PASSED
    assert result.status is GateStatus.REJECTED


# --- Clasificador y enrutado -------------------------------------------------


def test_vanilla_classifier_reports_known_without_a_model() -> None:
    contract = vanilla_contract(campaign_id="abc123def456")
    classification = VanillaClassifier().classify("brief", contract)
    assert classification.archetype is Archetype.KNOWN
    assert classification.variations == ()
    assert classification.rationale


def test_zero_contract_skips_the_classifier() -> None:
    classifier = _ExplodingClassifier()
    orchestrator = _StubOrchestrator()
    outcome = _manager(classifier, orchestrator).process(
        _zero_contract_campaign(), mode="long_video", url="https://example.com/v"
    )
    assert classifier.calls == []
    assert outcome.error is not None
    assert "parada deliberada" in outcome.error
    assert orchestrator.calls, "el motor de video debe recibir la llamada"
    assert orchestrator.calls[0]["url"] == "https://example.com/v"


def test_zero_contract_forces_repost_mode_to_avoid_the_llm() -> None:
    """Repost es el único camino que no selecciona segmentos con el LLM."""
    orchestrator = _StubOrchestrator()
    _ = _manager(_ExplodingClassifier(), orchestrator).process(
        _zero_contract_campaign(), mode="long_video", url="https://example.com/v"
    )
    assert orchestrator.calls[0]["repost_mode"] is True


def test_campaign_with_brief_still_classifies() -> None:
    """Regresión cero: con brief el clasificador se sigue usando."""
    classifier = _RecordingClassifier(Archetype.NEW_ARCHETYPE)
    orchestrator = _StubOrchestrator()
    campaign = Campaign(
        campaign_id="abc123def456",
        brief="brief real",
        contract=vanilla_contract(campaign_id="abc123def456"),
    )
    outcome = _manager(classifier, orchestrator).process(
        campaign, mode="repost", url="https://example.com/v"
    )
    assert classifier.calls == ["brief real"]
    assert orchestrator.calls == []
    assert outcome.archetype is Archetype.NEW_ARCHETYPE


def test_campaign_with_brief_keeps_known_archetype_rendering() -> None:
    classifier = _RecordingClassifier(Archetype.KNOWN)
    orchestrator = _StubOrchestrator()
    campaign = Campaign(
        campaign_id="abc123def456",
        brief="brief real",
        contract=vanilla_contract(campaign_id="abc123def456"),
    )
    _ = _manager(classifier, orchestrator).process(
        campaign, mode="repost", url="https://example.com/v"
    )
    assert classifier.calls == ["brief real"]
    assert len(orchestrator.calls) == 1


def test_campaign_with_brief_does_not_force_repost() -> None:
    """Con brief, un modo que necesita LLM se respeta tal cual."""
    classifier = _RecordingClassifier(Archetype.KNOWN)
    orchestrator = _StubOrchestrator()
    campaign = Campaign(
        campaign_id="abc123def456",
        brief="brief real",
        contract=vanilla_contract(campaign_id="abc123def456"),
    )
    _ = _manager(classifier, orchestrator).process(
        campaign, mode="long_video", url="https://example.com/v"
    )
    assert orchestrator.calls[0]["repost_mode"] is False


# --- CLI ---------------------------------------------------------------------


class _CapturingManager:
    """Manager que captura lo que el CLI le entrega, sin ejecutar el pipeline."""

    def __init__(self) -> None:
        self.campaign: Campaign | None = None
        self.kwargs: dict[str, object] = {}

    def process(
        self,
        campaign: Campaign,
        *,
        mode: str = "long_video",
        url: str | None = None,
        resume: bool = False,
        approve_manual_review: bool = False,
        approved_by: str | None = None,
        audio_track_path: Path | None = None,
        audio_track_url: str | None = None,
    ) -> CampaignOutcome:
        _ = (mode, url, resume, approve_manual_review, approved_by)
        _ = (audio_track_path, audio_track_url)
        self.campaign = campaign
        return CampaignOutcome(
            campaign_id=campaign.campaign_id,
            archetype=Archetype.KNOWN,
            status=CampaignStatus.COMPLETED,
        )


@pytest.mark.usefixtures("_no_llm_env")
def test_cli_url_only_builds_a_vanilla_contract() -> None:
    manager = _CapturingManager()
    exit_code = cli.main(
        ["campaign", "--url", "https://example.com/v", "--out", "delivery"],
        manager=manager,
    )
    campaign = manager.campaign
    assert campaign is not None
    assert exit_code == 0
    assert not campaign.brief
    assert campaign.contract is not None
    assert campaign.contract.brand_safety_required is False
    assert campaign.contract.prohibitions == []
    assert campaign.contract.audio_policy is AudioPolicy.ORIGINAL_AUDIO
    assert tuple(campaign.contract.rules.hard) == _AV_HARD_RULES


@pytest.mark.usefixtures("_no_llm_env")
def test_cli_url_only_campaign_id_is_derived_from_the_url() -> None:
    manager = _CapturingManager()
    url = "https://example.com/watch?v=abc"
    exit_code = cli.main(
        ["campaign", "--url", url, "--out", "delivery"],
        manager=manager,
    )
    campaign = manager.campaign
    assert campaign is not None
    assert exit_code == 0
    assert campaign.campaign_id == brief_key(url)[:12]
    assert campaign.campaign_id != brief_key("https://example.com/otra")[:12]


@pytest.mark.usefixtures("_no_llm_env")
def test_cli_url_only_campaign_id_is_a_safe_path_segment() -> None:
    """El campaign_id nombra el directorio de entrega: debe ser segmento seguro."""
    manager = _CapturingManager()
    _ = cli.main(
        ["campaign", "--url", "https://user:pw@host/path?a=1", "--out", "delivery"],
        manager=manager,
    )
    campaign = manager.campaign
    assert campaign is not None
    assert is_safe_segment(campaign.campaign_id)
    assert "pw" not in campaign.campaign_id


@pytest.mark.usefixtures("_no_llm_env")
def test_cli_defers_the_mode_decision_to_the_manager() -> None:
    """El CLI no reescribe ``--mode``: lo fuerza el gestor al ver la campaña.

    Forzarlo en el CLI dejaría fuera a cualquier otro llamador de
    ``CampaignManager.process``, que es quien garantiza el cero-LLM.
    """
    manager = _CapturingManager()
    exit_code = cli.main(
        ["campaign", "--url", "https://example.com/v", "--out", "delivery"],
        manager=manager,
    )
    campaign = manager.campaign
    assert campaign is not None
    assert exit_code == 0
    assert campaign.is_zero_contract is True


@pytest.mark.usefixtures("_no_llm_env")
def test_cli_without_brief_and_without_url_fails() -> None:
    exit_code = cli.main(["campaign", "--out", "delivery"])
    assert exit_code == 1


@pytest.mark.usefixtures("_no_llm_env")
def test_cli_with_brief_keeps_the_llm_flow(tmp_path: Path) -> None:
    """Con brief se conserva el camino tradicional, incluida la clasificación."""
    brief = tmp_path / "brief.md"
    _ = brief.write_text("brief de campana", encoding="utf-8")
    manager = _CapturingManager()
    exit_code = cli.main(
        ["campaign", str(brief), "--out", "delivery", "--url", "https://example.com/v"],
        manager=manager,
    )
    campaign = manager.campaign
    assert campaign is not None
    assert exit_code == 0
    assert campaign.brief == "brief de campana"
    assert campaign.campaign_id
    assert campaign.is_zero_contract is False


@pytest.mark.usefixtures("_no_llm_env")
def test_cli_campaign_manager_needs_no_llm_env_when_not_required() -> None:
    manager = _build_campaign_manager(require_llm=False)
    assert isinstance(manager, CampaignManager)


@pytest.mark.usefixtures("_no_llm_env")
def test_cli_campaign_manager_still_requires_llm_env_by_default() -> None:
    """Con brief el flujo tradicional sigue exigiendo credenciales LLM."""
    with pytest.raises(RuntimeError, match="KLIPTYCH_LLM"):
        _ = _build_campaign_manager()


@pytest.mark.usefixtures("_no_llm_env")
def test_zero_contract_makes_no_network_connection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """El modo promete cero llamadas al LLM: aquí se comprueba a nivel de socket.

    Es la garantía central del modo, y hasta ahora solo se probaba que el
    manager se construyera sin credenciales. Bloquear ``socket.connect``
    convierte esa promesa en algo verificable: cualquier intento de red, de
    cualquier destino, hace fallar el test.
    """
    attempts: list[object] = []

    def _blocked_connect(_self: socket.socket, address: object) -> None:
        attempts.append(address)
        msg = f"el modo zero-contract intentó conectar a {address}"
        raise AssertionError(msg)

    monkeypatch.setattr(socket.socket, "connect", _blocked_connect)
    manager = _CapturingManager()
    exit_code = cli.main(
        ["campaign", "--url", "https://example.com/v", "--out", "delivery"],
        manager=manager,
    )
    assert exit_code == 0
    assert attempts == []
