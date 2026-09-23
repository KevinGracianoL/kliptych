"""Base común de los modelos del contrato."""

from typing import ClassVar

from pydantic import BaseModel, ConfigDict


class ContractBase(BaseModel):
    """Configuración común: campos extra prohibidos, inmutables y sin input en errores."""

    model_config: ClassVar[ConfigDict] = ConfigDict(
        extra="forbid",
        frozen=True,
        hide_input_in_errors=True,
    )
