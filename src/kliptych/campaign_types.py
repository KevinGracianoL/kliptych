"""Tipos del ciclo de vida de una campaña en Kliptych (fase E).

``Campaign`` asocia el brief con su clasificación y su contrato; ``route_campaign``
traduce el arquetipo al estado inicial de procesamiento. Un brief ``NEW_ARCHETYPE``
no se procesa: queda pendiente de revisión manual del owner.
"""

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import ClassVar

from pydantic import BaseModel, ConfigDict, Field

from kliptych.contract import Contract
from kliptych.intelligence import Archetype, ArchetypeClassification


class CampaignStatus(StrEnum):
    """Estado de la campaña en el sistema."""

    PENDING = "PENDING"
    CLASSIFIED = "CLASSIFIED"
    PROCESSING = "PROCESSING"
    COMPLETED = "COMPLETED"
    MANUAL_REVIEW = "MANUAL_REVIEW"
    BLOCKED = "BLOCKED"


class Campaign(BaseModel):
    """Campaña con su clasificación y estado."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", frozen=True)

    campaign_id: str = Field(min_length=1)
    brief: str = Field(min_length=1)
    status: CampaignStatus = CampaignStatus.PENDING
    archetype: Archetype | None = None
    classification: ArchetypeClassification | None = None
    contract: Contract | None = None
    variations_log: Path | None = None


@dataclass(frozen=True, slots=True)
class PendingCampaign:
    """Campaña pendiente de revisión manual (NEW_ARCHETYPE)."""

    campaign: Campaign
    reason: str

    def __post_init__(self) -> None:
        """Valida que el motivo de revisión no esté vacío.

        Raises:
            ValueError: Si ``reason`` está vacío.
        """
        if not self.reason:
            msg = "reason no puede estar vacío"
            raise ValueError(msg)


def route_campaign(classification: ArchetypeClassification) -> CampaignStatus:
    """Determina el estado de la campaña según su clasificación.

    Args:
        classification: Clasificación de arquetipo de la campaña.

    Returns:
        ``MANUAL_REVIEW`` para ``NEW_ARCHETYPE``, ``PENDING`` para
        ``KNOWN_WITH_VARIATION`` (la variación debe registrarse antes de
        procesar) y ``CLASSIFIED`` para ``KNOWN``.
    """
    if classification.archetype is Archetype.NEW_ARCHETYPE:
        return CampaignStatus.MANUAL_REVIEW
    if classification.archetype is Archetype.KNOWN_WITH_VARIATION:
        return CampaignStatus.PENDING
    return CampaignStatus.CLASSIFIED
