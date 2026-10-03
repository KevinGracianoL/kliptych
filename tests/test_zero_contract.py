"""Tests del modo zero-contract: procesar con ``--url`` y sin llamar al LLM.

El modo zero-contract sustituye el brief de campaña por un contrato "vainilla"
sintetizado en memoria, de modo que ``--url`` basta para procesar un vídeo sin
ninguna llamada a la API del LLM ni credenciales ``KLIPTYCH_LLM_*``.
"""

import argparse
import socket
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, cast

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence


import pytest

from kliptych import __main__ as cli
from kliptych import campaign_manager, orchestrator
from kliptych.campaign_manager import (
    CampaignManager,
    CampaignManagerError,
    CampaignOutcome,
)
from kliptych.campaign_types import Campaign, CampaignStatus
from kliptych.contract import (
    AudioPolicy,
    Contract,
    Platform,
    contract_mutes_audio,
    vanilla_contract,
)
from kliptych.contract.schema import GlobalRestrictions, active_restriction_rules
from kliptych.encoding import RenderConfig
from kliptych.gate.checks import DEFAULT_VALIDATORS
from kliptych.gate.engine import Gate
from kliptych.gate.models import GateStatus
from kliptych.git_proposals import ProposalEngine, PullRequest
from kliptych.hashing import brief_key
from kliptych.intelligence import (
    Archetype,
    ArchetypeClassification,
    CampaignClassifier,
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
_ignored_zero_contract_flags = cast(
    "Callable[[argparse.Namespace], list[str]]", _private("_ignored_zero_contract_flags")
)


@pytest.fixture
def _no_llm_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in _LLM_ENV_VARS:
        monkeypatch.delenv(name, raising=False)


def _module_private(module: object, name: str) -> object:
    """Accede a un símbolo privado de un módulo, como hace ``test_reframe``.

    Returns:
        El atributo pedido del módulo.
    """
    return cast("object", getattr(module, name))


# ``CampaignOutcome`` declara ``PipelineResult``, ``SlideshowResult`` y
# ``PullRequest``, que viven en el orquestador y en git_proposals; esa
# resolución la dispara el primer ``CampaignManager.process`` real y queda
# cacheada a nivel de módulo. Sin ella, los tests que usan el doble
# ``_CapturingManager`` dependen de que otro archivo se haya ejecutado antes y
# fallan cuando este corre aislado. Se resuelve al importar el módulo.
cast("Callable[[], None]", _module_private(campaign_manager, "_resolve_outcome_model"))()


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


def _manager(
    classifier: CampaignClassifier | None,
    orchestrator: _StubOrchestrator,
) -> CampaignManager:
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


def test_vanilla_contract_declares_only_artifact_rules() -> None:
    """Exactamente las dos reglas de integridad del artefacto, y nada más.

    ``artifact.integrity`` verifica que el artefacto exista y sea legible;
    ``artifact.video_stream`` que tenga flujo de video.

    No hay regla de audio: la plataforma usa ``PlatformRules()``, cuyo
    ``audio_rule`` es ``ANY``, y el resolver solo añade ``audio.present``
    cuando alguna plataforma exige audio. Declararla aquí sería una aserción
    que el gate no puede sostener.
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


def test_gate_passes_vanilla_contract_on_artifact_rules(tmp_path: Path) -> None:
    """El gate aprueba exactamente las dos reglas de artefacto, sin audio.

    ``artifact.integrity`` y ``artifact.video_stream`` cubren la integridad
    estándar del A/V: el artefacto existe, es legible y tiene video. No se
    comprueba la presencia de audio porque no hay regla que la exija: con
    ``audio_rule=ANY`` el resolver no añade ``audio.present``.
    """
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


def test_zero_contract_runs_with_no_classifier_configured() -> None:
    """La ausencia de clasificador es una configuración legítima, no un descuido.

    El modo zero-contract no tiene brief que clasificar, así que el gestor se
    construye sin clasificador. Antes ``classifier`` era obligatorio y la CLI
    cableaba un clasificador trivial que la ruta zero-contract nunca llamaba,
    porque el gestor cortocircuitaba antes: código que existía solo para
    satisfacer el tipo.
    """
    orchestrator = _StubOrchestrator()
    outcome = _manager(None, orchestrator).process(
        _zero_contract_campaign(), mode="long_video", url="https://example.com/v"
    )
    assert outcome.error is not None
    assert "parada deliberada" in outcome.error
    assert orchestrator.calls[0]["repost_mode"] is True


def test_campaign_with_brief_and_no_classifier_fails_closed() -> None:
    """Con brief y sin clasificador el gestor falla en cerrado, con un motivo.

    Sin este contrato, un clasificador ausente seIeakaba como
    ``'NoneType' object has no attribute 'classify'`` envuelto en el
    ``except Exception`` genérico: un error que no dice qué falta ni por qué.
    """
    orchestrator = _StubOrchestrator()
    campaign = Campaign(
        campaign_id="abc123def456",
        brief="brief real",
        contract=vanilla_contract(campaign_id="abc123def456"),
    )
    outcome = _manager(None, orchestrator).process(
        campaign, mode="repost", url="https://example.com/v"
    )
    assert outcome.error is not None
    assert "clasificador" in outcome.error
    assert "NoneType" not in outcome.error
    assert orchestrator.calls == []


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


_DOWNLOAD_SENTINEL = "SENTINEL_ZEROCONTRACT_DESCARGA"


class _DownloadAttemptedError(Exception):
    """Se lanza en lugar de descargar, para detener la corrida sin red."""


@pytest.mark.usefixtures("_no_llm_env")
def test_zero_contract_cli_opens_no_socket_before_building_the_manager(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Sin brief, la ruta del CLI no abre sockets antes de construir el gestor.

    Qué verifica: que ``cli.main`` con ``--url`` y sin ``--brief`` llega a
    construir el manager sin abrir un socket en el proceso de pytest.

    Qué NO verifica, y no pretende: el pipeline no se ejecuta, porque el
    manager inyectado devuelve sin procesar nada. Tampoco cubre el egress por
    subproceso: ``yt-dlp`` corre en otro proceso, donde este parche no llega. Y
    el parche solo cubre ``socket.socket.connect``: ``getaddrinfo`` y
    ``connect_ex`` lo esquivan. Para esas capas está
    ``test_zero_contract_runs_no_llm_client``, que intercepta las fronteras
    reales del LLM.
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


@pytest.mark.usefixtures("_no_llm_env")
def test_zero_contract_runs_no_llm_client(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Ninguna frontera de LLM se toca al construir y ejecutar el modo real.

    A diferencia del test del socket, aquí se construye el gestor de verdad con
    ``require_llm=False`` y se ejecuta ``CampaignManager.process`` sobre una
    campaña zero-contract. Las tres fronteras del LLM están parcheadas para
    explotar si se tocan:

    - ``OpenAIChatModel.__init__``: no debe construirse ningún backend de chat.
    - ``LLMCampaignClassifier.__init__``: no debe construirse ningún clasificador.
    - ``UrllibTransport.post_json``: no debe salir ninguna petición al LLM.

    La descarga se sustituye por una excepción para cortar la corrida antes de
    la red: el egress de ``yt-dlp`` es un subproceso y no es interceptable desde
    aquí, pero no es una llamada al LLM, y el modo zero-contract sí descarga el
    vídeo del usuario. Lo que este test cubre es exactamente la promesa: cero
    clientes de LLM y cero peticiones al LLM en el camino real.
    """
    llm_touched: list[str] = []

    def _boom_llm_backend(*args: object, **kwargs: object) -> object:
        _ = (args, kwargs)
        llm_touched.append("OpenAIChatModel.__init__")
        msg = "el modo zero-contract no debe construir un backend de chat"
        raise AssertionError(msg)

    def _boom_llm_classifier(*args: object, **kwargs: object) -> object:
        _ = (args, kwargs)
        llm_touched.append("LLMCampaignClassifier.__init__")
        msg = "el modo zero-contract no debe construir un clasificador"
        raise AssertionError(msg)

    def _boom_llm_post(*args: object, **kwargs: object) -> object:
        _ = (args, kwargs)
        llm_touched.append("UrllibTransport.post_json")
        msg = "el modo zero-contract no debe llamar al LLM"
        raise AssertionError(msg)

    def _stop_at_download(*args: object, **kwargs: object) -> object:
        _ = (args, kwargs)
        msg = _DOWNLOAD_SENTINEL
        raise _DownloadAttemptedError(msg)

    monkeypatch.setattr(
        "kliptych.runtime.openai_compatible.OpenAIChatModel.__init__", _boom_llm_backend
    )
    monkeypatch.setattr(
        "kliptych.intelligence.LLMCampaignClassifier.__init__", _boom_llm_classifier
    )
    monkeypatch.setattr("kliptych.runtime.transport.UrllibTransport.post_json", _boom_llm_post)
    monkeypatch.setattr("kliptych.download.MediaDownloader.download_video", _stop_at_download)

    manager = _build_campaign_manager(require_llm=False, destination=tmp_path / "delivery")
    campaign = Campaign(
        campaign_id="abc123def456",
        brief="",
        contract=vanilla_contract(campaign_id="abc123def456"),
    )
    outcome = manager.process(campaign, mode="long_video", url="https://example.com/v")

    # El token prueba que el pipeline REAL llego al downloader: no es un atajo
    # ni un doble de test. No se busca la palabra "descarga" porque el nombre de
    # la etapa va embebido en cualquier fallo de descarga, con lo que un
    # parche que dejara de aplicar pasaria igual haciendo egress real.
    assert outcome.error is not None
    assert _DOWNLOAD_SENTINEL in outcome.error, outcome.error
    assert llm_touched == []


class _VideoOrchestratorUnderTest(Protocol):
    """La parte del orquestador de video que ejercitan estos tests."""

    def run_long_video(self, url: str, **kwargs: object) -> object:
        """Lanza el pipeline long_video del orquestador.

        Returns:
            El resultado del pipeline.
        """
        ...


def _build_video_orchestrator(
    *,
    work_dir: Path,
    model: object | None,
) -> _VideoOrchestratorUnderTest:
    """Construye el orquestador real de la CLI con o sin modelo.

    Se accede al nombre privado por el mismo motivo y con el mismo patrón que
    usa ``test_audit_campaign_flow`` con ``_make_default_campaign_manager``:
    la clase no está exportada y el comportamiento que hay que cubrir es suyo.
    El tipo de retorno declarado es el Protocol de arriba, no la clase, para
    no importar un nombre privado.

    Returns:
        Un orquestador listo para ``run_long_video``.
    """
    factory = cast(
        "Callable[..., _VideoOrchestratorUnderTest]", _private("_DefaultVideoOrchestrator")
    )
    return factory(work_dir=work_dir, model=model, render=RenderConfig())


def test_video_orchestrator_without_model_rejects_non_repost_modes(tmp_path: Path) -> None:
    """Sin backend LLM, los modos que seleccionan segmentos fallan con nombre.

    Es alcanzable: ``_build_campaign_manager(require_llm=False)`` deja el
    orquestador sin modelo, y cualquier llamador que le pase una campaña CON
    brief y ``mode="long_video"`` llega aquí. Por CLI no ocurre, porque la ruta
    zero-contract fuerza repost y la ruta con brief exige credenciales; pero la
    clase se construye con ``model=None`` en esa configuración y no debe
    depender del cableado del CLI para fallar con un error entendible.
    """
    orchestrator = _build_video_orchestrator(work_dir=tmp_path, model=None)
    contract = vanilla_contract(campaign_id="abc123def456")
    with pytest.raises(CampaignManagerError, match="no hay backend LLM configurado"):
        _ = orchestrator.run_long_video(
            "https://example.com/v",
            contract=contract,
            mode="long_video",
        )


def test_video_orchestrator_without_model_lets_repost_through(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """La contraparte: repost no consulta el modelo, así que el guard no corta.

    Se sustituye ``run_repost`` por un doble para comprobar que la llamada
    llega, en vez de dejar que el pipeline real intente descargar.
    """
    seen: dict[str, object] = {}
    sentinel = object()

    def _fake_run_repost(url: str, **kwargs: object) -> object:
        seen["url"] = url
        seen["model"] = kwargs.get("model")
        return sentinel

    monkeypatch.setattr(orchestrator, "run_repost", _fake_run_repost)
    orchestrator_obj = _build_video_orchestrator(work_dir=tmp_path, model=None)
    contract = vanilla_contract(campaign_id="abc123def456")
    result = orchestrator_obj.run_long_video(
        "https://example.com/v",
        contract=contract,
        mode="repost_ugc",
    )
    assert result is sentinel
    assert seen["url"] == "https://example.com/v"
    assert seen["model"] is None


@pytest.mark.parametrize("mode", ["slideshow", "audio_locked"])
def test_zero_contract_warns_about_modes_it_cannot_run(
    mode: str, caplog: pytest.LogCaptureFixture
) -> None:
    """``slideshow`` y ``audio_locked`` se avisan: el modo efectivo es repost.

    Sin aviso, el operador recibía exit 0 con una entrega de repost cuando
    pidió un slideshow o una pista de audio externa.
    """
    _ = caplog
    namespace = argparse.Namespace(
        contract_draft=None,
        audio_track_path=None,
        audio_track_url=None,
        mode=mode,
    )
    assert f"--mode {mode}" in _ignored_zero_contract_flags(namespace)


@pytest.mark.parametrize("mode", ["long_video", "repost", "repost_ugc"])
def test_zero_contract_does_not_warn_about_modes_that_run(mode: str) -> None:
    """Ni el defecto del parser ni los modos que coinciden con el efectivo."""
    namespace = argparse.Namespace(
        contract_draft=None,
        audio_track_path=None,
        audio_track_url=None,
        mode=mode,
    )
    assert _ignored_zero_contract_flags(namespace) == []


_IGNORED_FLAG_ATTRS = cast("tuple[tuple[str, str], ...]", _private("_ZERO_CONTRACT_IGNORED_FLAGS"))
# Se escribe a mano en vez de derivarse de la constante de producción:
# parametrizar sobre la propia guarda sería circular, porque al quitarle una
# entrada el caso desaparece en lugar de fallar y la suite sigue en verde.
_EXPECTED_IGNORED_FLAGS = (
    ("contract_draft", "--contract-draft"),
    ("audio_track_path", "--audio-track-path"),
    ("audio_track_url", "--audio-track-url"),
)


def test_zero_contract_ignored_flags_match_the_documented_set() -> None:
    """La guarda de flags ignorados es exactamente esta lista.

    Ancla el conjunto de forma independiente del código de producción: sin esta
    aserción, borrar una entrada de la guarda se reduce el número de casos
    parametrizados y nada falla, dejando esa puerta sin protección.
    """
    assert _IGNORED_FLAG_ATTRS == _EXPECTED_IGNORED_FLAGS


@pytest.mark.parametrize(("attribute", "flag"), _EXPECTED_IGNORED_FLAGS)
def test_zero_contract_warns_about_every_ignored_flag(attribute: str, flag: str) -> None:
    """Cada flag que el modo zero-contract no puede honrar avisa.

    Antes solo ``--mode`` estaba protegido y dos de las tres puertas no tenían
    red de seguridad: quitar ``--contract-draft`` de la guarda no rompía nada.
    """
    values = {
        "contract_draft": None,
        "audio_track_path": None,
        "audio_track_url": None,
        "mode": "long_video",
    }
    values[attribute] = "valor-de-prueba"
    assert flag in _ignored_zero_contract_flags(argparse.Namespace(**values))


@pytest.mark.parametrize(("attribute", "flag"), _EXPECTED_IGNORED_FLAGS)
def test_zero_contract_stays_silent_when_no_flag_is_passed(attribute: str, flag: str) -> None:
    """Sin el flag presente no hay nada que avisar.

    Args:
        attribute: Nombre del atributo del flag en el namespace.
        flag: Nombre del flag, para la aserción del mensaje de fallo.
    """
    _ = (attribute, flag)
    namespace = argparse.Namespace(
        contract_draft=None,
        audio_track_path=None,
        audio_track_url=None,
        mode="long_video",
    )
    assert _ignored_zero_contract_flags(namespace) == []
