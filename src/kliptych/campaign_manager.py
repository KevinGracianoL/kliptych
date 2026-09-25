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
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, Protocol

from pydantic import BaseModel, ConfigDict, Field

from kliptych.campaign_types import CampaignStatus
from kliptych.intelligence import Archetype

if TYPE_CHECKING:
    from collections.abc import Sequence

    from kliptych.assets import AssetRegistry
    from kliptych.campaign_types import Campaign
    from kliptych.contract import Contract, Platform, PlatformRules
    from kliptych.exporter import DeliveryReport
    from kliptych.gate import Gate, Piece
    from kliptych.git_proposals import ProposalEngine, PullRequest
    from kliptych.intelligence import ArchetypeClassification, CampaignClassifier
    from kliptych.orchestrator import PipelineResult, SlideshowResult

_MAX_PIECE_ID_LENGTH: int = 64


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
    delivery_report: DeliveryReport | None = None
    error: str | None = None


@cache
def _resolve_outcome_model() -> None:
    """Resuelve las referencias diferidas de ``CampaignOutcome``.

    ``PipelineResult`` y ``SlideshowResult`` viven en el módulo del orquestador,
    que importa ``subprocess``. Se importan de forma diferida para que importar
    este controlador no cargue el motor de video; la resolución se cachea.
    """
    from kliptych.exporter import DeliveryReport
    from kliptych.git_proposals import PullRequest
    from kliptych.orchestrator import PipelineResult, SlideshowResult

    _ = CampaignOutcome.model_rebuild(
        _types_namespace={
            "PullRequest": PullRequest,
            "PipelineResult": PipelineResult,
            "SlideshowResult": SlideshowResult,
            "DeliveryReport": DeliveryReport,
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
        gate: Gate | None = None,
        assets: AssetRegistry | None = None,
        destination: Path | None = None,
    ) -> None:
        """Configura el controlador y sus dependencias.

        Args:
            classifier: Clasificador de arquetipos de campaña.
            proposal_engine: Motor que materializa propuestas de Pull Request.
            video_orchestrator: Motor de video largo y slideshow; opcional.
            slideshow_orchestrator: Motor de slideshow alternativo; opcional.
            gate: Gate configurado para verificar las entregas; opcional.
            assets: Registro de assets del workspace; opcional.
            destination: Directorio de destino del paquete de entrega; opcional.
        """
        self._classifier: CampaignClassifier = classifier
        self._proposal_engine: ProposalEngine = proposal_engine
        self._video_orchestrator: VideoOrchestrator | None = video_orchestrator
        self._slideshow_orchestrator: SlideshowOrchestrator | None = slideshow_orchestrator
        self._gate: Gate | None = gate
        self._assets: AssetRegistry | None = assets
        self._destination: Path | None = destination

    def process(
        self,
        campaign: Campaign,
        *,
        mode: str = "long_video",
        url: str | None = None,
        images: Sequence[Path] | None = None,
        resume: bool = False,
        gate: Gate | None = None,
        assets: AssetRegistry | None = None,
        destination: Path | None = None,
        caption: str | None = None,
        hashtags: Sequence[str] = (),
        platform: Platform | None = None,
        approve_manual_review: bool = False,
        approved_by: str | None = None,
        audio_track_path: Path | None = None,
        audio_track_url: str | None = None,
    ) -> CampaignOutcome:
        """Procesa la campaña según su arquetipo.

        Clasifica el brief y enruta: ``KNOWN`` ejecuta el motor de video; los
        arquetipos nuevos o con variación emiten una propuesta de Pull Request
        sin tocar el motor de video. Los fallos se devuelven como
        ``CampaignOutcome`` con ``error``, nunca como excepción.

        Args:
            campaign: Campaña con su brief y contrato validado.
            mode: Modo de video a ejecutar si el arquetipo es ``KNOWN``
                (``long_video``, ``audio_locked``, ``repost`` o ``slideshow``).
            url: URL del vídeo fuente, requerida por los modos de video.
            images: Imágenes del slideshow, requeridas por el modo ``slideshow``.
            resume: Si es True, reanuda la ejecución desde checkpoints previos.
            gate: Gate de validación; si se omite, usa el inyectado en el manager.
            assets: Registro de assets; si se omite, usa el inyectado.
            destination: Destino de la entrega; si se omite, usa el inyectado.
            caption: Texto del caption; si se omite, se infiere del contrato.
            hashtags: Hashtags de la pieza; si se omiten, se infieren del contrato.
            platform: Plataforma específica a entregar; si se omite, entrega todas las del contrato.
            approve_manual_review: Si es True, aprueba piezas en estado PENDING_REVIEW.
            approved_by: Identificador del operador o sistema que aprueba la revisión manual.
            audio_track_path: Pista de audio local para el modo ``audio_locked``.
            audio_track_url: URL de la pista de audio para el modo ``audio_locked``.

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
            return self._process_known(
                campaign,
                mode=mode,
                url=url,
                images=images,
                resume=resume,
                gate=gate,
                assets=assets,
                destination=destination,
                caption=caption,
                hashtags=hashtags,
                platform=platform,
                approve_manual_review=approve_manual_review,
                approved_by=approved_by,
                audio_track_path=audio_track_path,
                audio_track_url=audio_track_url,
            )
        return self._process_proposal(campaign, contract, classification)

    def _process_known(
        self,
        campaign: Campaign,
        *,
        mode: str,
        url: str | None,
        images: Sequence[Path] | None,
        resume: bool = False,
        gate: Gate | None = None,
        assets: AssetRegistry | None = None,
        destination: Path | None = None,
        caption: str | None = None,
        hashtags: Sequence[str] = (),
        platform: Platform | None = None,
        approve_manual_review: bool = False,
        approved_by: str | None = None,
        audio_track_path: Path | None = None,
        audio_track_url: str | None = None,
    ) -> CampaignOutcome:
        try:
            outcome = self._render_known(
                campaign,
                mode=mode,
                url=url,
                images=images,
                resume=resume,
                gate=gate,
                assets=assets,
                destination=destination,
                caption=caption,
                hashtags=hashtags,
                platform=platform,
                approve_manual_review=approve_manual_review,
                approved_by=approved_by,
                audio_track_path=audio_track_path,
                audio_track_url=audio_track_url,
            )
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
        resume: bool = False,
        gate: Gate | None = None,
        assets: AssetRegistry | None = None,
        destination: Path | None = None,
        caption: str | None = None,
        hashtags: Sequence[str] = (),
        platform: Platform | None = None,
        approve_manual_review: bool = False,
        approved_by: str | None = None,
        audio_track_path: Path | None = None,
        audio_track_url: str | None = None,
    ) -> CampaignOutcome:
        pipeline_result: PipelineResult | None = None
        slideshow_result: SlideshowResult | None = None
        contract = campaign.contract
        if mode == "long_video":
            pipeline_result = self._run_long_video(url, resume=resume, contract=contract)
            final_videos = getattr(pipeline_result, "final_videos", (pipeline_result.final_video,))
        elif mode == "audio_locked":
            pipeline_result = self._run_long_video(
                url,
                resume=resume,
                contract=contract,
                audio_locked=True,
                audio_track_path=audio_track_path,
                audio_track_url=audio_track_url,
            )
            final_videos = getattr(pipeline_result, "final_videos", (pipeline_result.final_video,))
        elif mode == "repost":
            pipeline_result = self._run_long_video(
                url, resume=resume, contract=contract, repost_mode=True
            )
            final_videos = getattr(pipeline_result, "final_videos", (pipeline_result.final_video,))
        elif mode == "slideshow":
            slideshow_result = self._run_slideshow(
                images,
                resume=resume,
                contract=contract,
                audio_track_path=audio_track_path,
                audio_track_url=audio_track_url,
            )
            final_videos = getattr(
                slideshow_result, "final_videos", (slideshow_result.final_video,)
            )
        else:
            msg = f"modo de video no soportado: {mode!r}"
            raise CampaignManagerError(msg)

        effective_gate = gate if gate is not None else self._gate
        effective_dest = destination if destination is not None else self._destination
        effective_assets = assets if assets is not None else self._assets

        if effective_gate is None:
            msg = f"la campaña {campaign.campaign_id} requiere un Gate configurado para su entrega"
            raise CampaignManagerError(msg)

        from kliptych.assets import AssetRegistry
        from kliptych.exporter import ExportStatus, export_delivery

        dest = effective_dest if effective_dest is not None else Path("delivery")
        reg = effective_assets if effective_assets is not None else AssetRegistry(dest.parent)
        contract = campaign.contract
        if contract is None:
            msg = f"la campaña {campaign.campaign_id} no tiene contrato validado"
            raise CampaignManagerError(msg)

        pieces = _build_pieces(
            campaign=campaign,
            contract=contract,
            final_videos=final_videos,
            caption=caption,
            hashtags=hashtags,
            platform=platform,
        )
        delivery_report = export_delivery(
            contract=contract,
            pieces=pieces,
            gate=effective_gate,
            assets=reg,
            destination=dest,
            approve_manual_review=approve_manual_review,
            approved_by=approved_by,
        )
        status = (
            CampaignStatus.COMPLETED
            if delivery_report.status is ExportStatus.EXPORTED
            else CampaignStatus.BLOCKED
        )
        return CampaignOutcome(
            campaign_id=campaign.campaign_id,
            archetype=Archetype.KNOWN,
            status=status,
            pipeline_result=pipeline_result,
            slideshow_result=slideshow_result,
            delivery_report=delivery_report,
        )

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

    def _run_long_video(
        self,
        url: str | None,
        *,
        resume: bool = False,
        contract: Contract | None = None,
        audio_locked: bool = False,
        repost_mode: bool = False,
        audio_track_path: Path | None = None,
        audio_track_url: str | None = None,
    ) -> PipelineResult:
        orchestrator = self._video_orchestrator
        if orchestrator is None:
            msg = "no hay orquestador de video configurado para el modo long_video"
            raise CampaignManagerError(msg)
        if url is None:
            msg = "el modo long_video requiere la URL del video fuente"
            raise CampaignManagerError(msg)
        return orchestrator.run_long_video(
            url,
            resume=resume,
            contract=contract,
            audio_locked=audio_locked,
            repost_mode=repost_mode,
            audio_track_path=audio_track_path,
            audio_track_url=audio_track_url,
        )

    def _run_slideshow(
        self,
        images: Sequence[Path] | None,
        *,
        resume: bool = False,
        contract: Contract | None = None,
        audio_track_path: Path | None = None,
        audio_track_url: str | None = None,
    ) -> SlideshowResult:
        if images is None:
            msg = "el modo slideshow requiere la secuencia de imágenes"
            raise CampaignManagerError(msg)
        slideshow_orchestrator = self._slideshow_orchestrator
        if slideshow_orchestrator is not None:
            return slideshow_orchestrator.run(
                images,
                resume=resume,
                contract=contract,
                audio_track_path=audio_track_path,
                audio_track_url=audio_track_url,
            )
        video_orchestrator = self._video_orchestrator
        if video_orchestrator is None:
            msg = "no hay orquestador de slideshow configurado"
            raise CampaignManagerError(msg)
        return video_orchestrator.run_slideshow(
            images,
            resume=resume,
            contract=contract,
            audio_track_path=audio_track_path,
            audio_track_url=audio_track_url,
        )


def _piece_caption(brief: str, rules: PlatformRules, caption: str | None) -> str:
    if caption is not None:
        return caption
    req_mentions = dict.fromkeys([*rules.required_mentions, *rules.caption_rules.must_mention])
    mentions_str = " ".join(req_mentions)
    tags_str = " ".join(dict.fromkeys(rules.required_hashtags))
    first = f"{rules.caption_rules.first_line}\n" if rules.caption_rules.first_line else ""
    composed = f"{first}{brief} {mentions_str} {tags_str}".strip()
    return composed or (brief or "video")


def _piece_id(
    campaign_id: str,
    platform: Platform,
    *,
    multiple_platforms: bool,
    index: int | None = None,
) -> str:
    base = f"{campaign_id}-{platform.value}" if multiple_platforms else campaign_id
    raw = f"{base}-{index:02d}" if index is not None else base
    return raw[:_MAX_PIECE_ID_LENGTH] if len(raw) > _MAX_PIECE_ID_LENGTH else raw


def _build_pieces(
    *,
    campaign: Campaign,
    contract: Contract,
    final_video: Path | None = None,
    final_videos: Sequence[Path] | None = None,
    caption: str | None,
    hashtags: Sequence[str],
    platform: Platform | None,
) -> list[Piece]:
    from kliptych.gate.models import Piece

    if final_videos is not None:
        videos = tuple(final_videos)
    elif final_video is not None:
        videos = (final_video,)
    else:
        msg = "se requiere al menos un video para construir las piezas"
        raise CampaignManagerError(msg)

    platforms = [platform] if platform is not None else list(contract.platforms.keys())
    multiple_platforms = len(platforms) > 1
    multiple_videos = len(videos) > 1
    pieces: list[Piece] = []
    for vid_idx, video in enumerate(videos):
        index_for_id = vid_idx if multiple_videos else None
        for plat in platforms:
            plat_rules = contract.platforms[plat]
            piece_caption = _piece_caption(campaign.brief, plat_rules, caption)
            piece_tags = tuple(hashtags) if hashtags else tuple(plat_rules.required_hashtags)
            pieces.append(
                Piece(
                    piece_id=_piece_id(
                        campaign.campaign_id,
                        plat,
                        multiple_platforms=multiple_platforms,
                        index=index_for_id,
                    ),
                    platform=plat,
                    caption=piece_caption,
                    hashtags=piece_tags,
                    subtitle_text=None,
                    artifact_path=video,
                )
            )
    return pieces


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
