"""Base común de los modelos del contrato."""

from typing import ClassVar

from pydantic import BaseModel, ConfigDict


class ContractBase(BaseModel):
    """Configuración común: campos extra prohibidos y modelos inmutables."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", frozen=True)
