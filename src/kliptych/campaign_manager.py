"""Controlador maestro de campañas: conecta inteligencia, video y propuestas (fase E).

Es el punto de integración de la fase E: recibe una campaña con su contrato,
delega la clasificación de arquetipo en el clasificador y enruta según el
resultado. Un arquetipo ``KNOWN`` se renderiza con el motor de video; un
``KNOWN_WITH_VARIATION`` o un ``NEW_ARCHETYPE`` se materializan como una
propuesta de Pull Request y el motor de video NO se toca.

La invariante de seguridad es estructural: el motor de video solo se referencia
dentro de la rama ``KNOWN``. Importar este módulo no carga el orquestador (que
importa ``subprocess``); sus tipos de resultado se resuelven de forma diferida en
el primer uso.
"""

from __future__ import annotations

from functools import cache
from typing import TYPE_CHECKING, ClassVar, Protocol

from pydantic import BaseModel, ConfigDict, Field

from kliptych.campaign_types import CampaignStatus
from kliptych.intelligence import Archetype

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from kliptych.campaign_types import Campaign
    from kliptych.contract import Contract
    from kliptych.git_proposals import ProposalEngine, PullRequest
    from kliptych.intelligence import ArchetypeClassification, CampaignClassifier
    from kliptych.orchestrator import PipelineResult, SlideshowResult


class CampaignManagerError(Exception):
    """El manager de campaña no pudo procesar la petición."""


class VideoOrchestrator(Protocol):
    """Interfaz del motor de video (long_video y slideshow)."""

    def run_long_video(self, url: str, **kwargs: object) -> PipelineResult:
        """Renderiza un vídeo largo a partir de su URL.

        Args:
            url: URL http/https del vídeo fuente.
            **kwargs: Configuración adicional del motor.

        Returns:
            El resultado del pipeline long_video.
        """
        ...

    def run_slideshow(self, images: Sequence[Path], **kwargs: object) -> SlideshowResult:
        """Renderiza un slideshow a partir de imágenes.

        Args:
            images: Rutas locales de las imágenes, en orden de montaje.
            **kwargs: Configuración adicional del motor.

        Returns:
            El resultado del pipeline slideshow.
        """
        ...


class SlideshowOrchestrator(Protocol):
    """Interfaz del motor de slideshow."""

    def run(self, images: Sequence[Path], **kwargs: object) -> SlideshowResult:
        """Renderiza un slideshow a partir de imágenes.

        Args:
            images: Rutas locales de las imágenes, en orden de montaje.
            **kwargs: Configuración adicional del motor.

        Returns:
            El resultado del pipeline slideshow.
        """
        ...


class CampaignOutcome(BaseModel):
    """Resultado del procesamiento de una campaña."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", frozen=True)

    campaign_id: str = Field(min_length=1)
    archetype: Archetype
    status: CampaignStatus
    pull_request: PullRequest | None = None
    pipeline_result: PipelineResult | None = None
    slideshow_result: SlideshowResult | None = None
    error: str | None = None


@cache
def _resolve_outcome_model() -> None:
    """Resuelve las referencias diferidas de ``CampaignOutcome``.

    ``PipelineResult`` y ``SlideshowResult`` viven en el módulo del orquestador,
    que importa ``subprocess``. Se importan de forma diferida para que importar
    este controlador no cargue el motor de video; la resolución se cachea.
    """
    from kliptych.git_proposals import PullRequest
    from kliptych.orchestrator import PipelineResult, SlideshowResult

    _ = CampaignOutcome.model_rebuild(
        _types_namespace={
            "PullRequest": PullRequest,
            "PipelineResult": PipelineResult,
            "SlideshowResult": SlideshowResult,
        }
    )


class CampaignManager:
    """Controlador maestro que conecta inteligencia con ejecución."""

    def __init__(
        self,
        *,
        classifier: CampaignClassifier,
        proposal_engine: ProposalEngine,
        video_orchestrator: VideoOrchestrator | None = None,
        slideshow_orchestrator: SlideshowOrchestrator | None = None,
    ) -> None:
        """Configura el controlador y sus dependencias.

        Args:
            classifier: Clasificador de arquetipos de campaña.
            proposal_engine: Motor que materializa propuestas de Pull Request.
            video_orchestrator: Motor de video largo y slideshow; opcional.
            slideshow_orchestrator: Motor de slideshow alternativo; opcional.
        """
        self._classifier: CampaignClassifier = classifier
        self._proposal_engine: ProposalEngine = proposal_engine
        self._video_orchestrator: VideoOrchestrator | None = video_orchestrator
        self._slideshow_orchestrator: SlideshowOrchestrator | None = slideshow_orchestrator

    def process(
        self,
        campaign: Campaign,
        *,
        mode: str = "long_video",
        url: str | None = None,
        images: Sequence[Path] | None = None,
    ) -> CampaignOutcome:
        """Procesa la campaña según su arquetipo.

        Clasifica el brief y enruta: ``KNOWN`` ejecuta el motor de video; los
        arquetipos nuevos o con variación emiten una propuesta de Pull Request
        sin tocar el motor de video. Los fallos se devuelven como
        ``CampaignOutcome`` con ``error``, nunca como excepción.

        Args:
            campaign: Campaña con su brief y contrato validado.
            mode: Modo de video a ejecutar si el arquetipo es ``KNOWN``
                (``long_video`` o ``slideshow``).
            url: URL del vídeo fuente, requerida por el modo ``long_video``.
            images: Imágenes del slideshow, requeridas por el modo ``slideshow``.

        Returns:
            El resultado del procesamiento con su arquetipo, estado y artefactos.
        """
        _resolve_outcome_model()
        contract = campaign.contract
        if contract is None:
            return _error_outcome(
                campaign,
                f"la campaña {campaign.campaign_id} no tiene contrato validado",
                Archetype.NEW_ARCHETYPE,
            )
        try:
            classification = self._classifier.classify(campaign.brief, contract)
        except Exception as error:
            return _error_outcome(
                campaign,
                f"falló la clasificación de la campaña {campaign.campaign_id}: {error}",
                Archetype.NEW_ARCHETYPE,
            )
        if classification.archetype is Archetype.KNOWN:
            return self._process_known(campaign, mode=mode, url=url, images=images)
        return self._process_proposal(campaign, contract, classification)

    def _process_known(
        self,
        campaign: Campaign,
        *,
        mode: str,
        url: str | None,
        images: Sequence[Path] | None,
    ) -> CampaignOutcome:
        try:
            outcome = self._render_known(campaign, mode=mode, url=url, images=images)
        except Exception as error:
            return _error_outcome(
                campaign,
                f"falló el procesamiento de video de la campaña {campaign.campaign_id}: {error}",
                Archetype.KNOWN,
            )
        return outcome

    def _render_known(
        self,
        campaign: Campaign,
        *,
        mode: str,
        url: str | None,
        images: Sequence[Path] | None,
    ) -> CampaignOutcome:
        if mode == "long_video":
            return CampaignOutcome(
                campaign_id=campaign.campaign_id,
                archetype=Archetype.KNOWN,
                status=CampaignStatus.COMPLETED,
                pipeline_result=self._run_long_video(url),
            )
        if mode == "slideshow":
            return CampaignOutcome(
                campaign_id=campaign.campaign_id,
                archetype=Archetype.KNOWN,
                status=CampaignStatus.COMPLETED,
                slideshow_result=self._run_slideshow(images),
            )
        msg = f"modo de video no soportado: {mode!r}"
        raise CampaignManagerError(msg)

    def _process_proposal(
        self,
        campaign: Campaign,
        contract: Contract,
        classification: ArchetypeClassification,
    ) -> CampaignOutcome:
        archetype = classification.archetype
        status = (
            CampaignStatus.MANUAL_REVIEW
            if archetype is Archetype.NEW_ARCHETYPE
            else CampaignStatus.PENDING
        )
        enriched = campaign.model_copy(
            update={"archetype": archetype, "classification": classification}
        )
        try:
            pull_request = self._proposal_engine.propose(enriched, contract)
        except Exception as error:
            return _error_outcome(
                campaign,
                f"falló la propuesta de la campaña {campaign.campaign_id}: {error}",
                archetype,
            )
        return CampaignOutcome(
            campaign_id=campaign.campaign_id,
            archetype=archetype,
            status=status,
            pull_request=pull_request,
        )

    def _run_long_video(self, url: str | None) -> PipelineResult:
        orchestrator = self._video_orchestrator
        if orchestrator is None:
            msg = "no hay orquestador de video configurado para el modo long_video"
            raise CampaignManagerError(msg)
        if url is None:
            msg = "el modo long_video requiere la URL del video fuente"
            raise CampaignManagerError(msg)
        return orchestrator.run_long_video(url)

    def _run_slideshow(self, images: Sequence[Path] | None) -> SlideshowResult:
        if images is None:
            msg = "el modo slideshow requiere la secuencia de imágenes"
            raise CampaignManagerError(msg)
        slideshow_orchestrator = self._slideshow_orchestrator
        if slideshow_orchestrator is not None:
            return slideshow_orchestrator.run(images)
        video_orchestrator = self._video_orchestrator
        if video_orchestrator is None:
            msg = "no hay orquestador de slideshow configurado"
            raise CampaignManagerError(msg)
        return video_orchestrator.run_slideshow(images)


def _error_outcome(campaign: Campaign, message: str, archetype: Archetype) -> CampaignOutcome:
    """Construye un resultado de error sin tocar el motor de video.

    Args:
        campaign: Campaña que originó el error.
        message: Descripción del fallo.
        archetype: Arquetipo conocido, o ``NEW_ARCHETYPE`` si no se pudo
            clasificar (garantiza que no se renderice video).

    Returns:
        El resultado con ``error`` poblado y sin artefactos.
    """
    return CampaignOutcome(
        campaign_id=campaign.campaign_id,
        archetype=archetype,
        status=CampaignStatus.PENDING,
        error=message,
    )
