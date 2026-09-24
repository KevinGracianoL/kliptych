"""Generación de configuración de campañas (fase E).

Serializa la decisión de campaña a los artefactos que el motor de propuestas
commitea en una rama: un JSON de configuración y una entrada Markdown para el
log de variaciones. Es una capa pura: no toca red, Git ni el sistema de
archivos.
"""

from datetime import UTC, datetime
from typing import ClassVar

from pydantic import BaseModel, ConfigDict, Field

from kliptych.campaign_types import Campaign
from kliptych.contract import Contract
from kliptych.intelligence import Archetype


class CampaignConfigError(Exception):
    """La configuración de campaña no se pudo generar."""


class CampaignConfig(BaseModel):
    """Configuración serializable de una campaña."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", frozen=True)

    campaign_id: str = Field(min_length=1)
    archetype: Archetype
    contract: dict[str, object]
    variations: tuple[str, ...] = ()
    created_at: str = Field(min_length=1)


def generate_campaign_config(campaign: Campaign, contract: Contract) -> str:
    """Genera la representación JSON de la configuración.

    Args:
        campaign: Campaña con su arquetipo y variaciones.
        contract: Contrato validado asociado a la campaña.

    Returns:
        El JSON de la configuración, con sangría de dos espacios.

    Raises:
        CampaignConfigError: Si la campaña no tiene arquetipo clasificado.
    """
    config = CampaignConfig(
        campaign_id=campaign.campaign_id,
        archetype=_archetype(campaign),
        contract=contract.model_dump(mode="json"),
        variations=_variations(campaign),
        created_at=datetime.now(UTC).isoformat(),
    )
    return config.model_dump_json(indent=2)


def variations_md_entry(campaign: Campaign) -> str:
    """Genera una entrada Markdown para el log de variaciones.

    Args:
        campaign: Campaña con su clasificación y variaciones.

    Returns:
        La entrada Markdown, terminada en salto de línea.

    Raises:
        CampaignConfigError: Si la campaña no tiene clasificación.
    """
    classification = campaign.classification
    if classification is None:
        msg = f"la campaña {campaign.campaign_id} no tiene clasificación"
        raise CampaignConfigError(msg)
    lines = [
        f"## {campaign.campaign_id}",
        "",
        f"- arquetipo: {classification.archetype.value}",
    ]
    lines.extend(f"- variación: {variation}" for variation in classification.variations)
    return "\n".join(lines) + "\n"


def _archetype(campaign: Campaign) -> Archetype:
    if campaign.archetype is not None:
        return campaign.archetype
    if campaign.classification is not None:
        return campaign.classification.archetype
    msg = f"la campaña {campaign.campaign_id} no tiene arquetipo clasificado"
    raise CampaignConfigError(msg)


def _variations(campaign: Campaign) -> tuple[str, ...]:
    classification = campaign.classification
    return () if classification is None else classification.variations
