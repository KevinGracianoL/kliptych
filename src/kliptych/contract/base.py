"""Base común de los modelos del contrato."""

from typing import ClassVar

from pydantic import BaseModel, ConfigDict


class ContractBase(BaseModel):
    """Base del contrato: extra prohibido, inmutable e input oculto en errores.

    ``hide_input_in_errors`` oculta los valores en la representación de los
    ``ValidationError``; las claves de diccionario que aparecen en ``loc`` y
    el contenido de ``.errors()`` no se ocultan, por eso el runtime evita
    encadenar esos errores.
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(
        extra="forbid",
        frozen=True,
        hide_input_in_errors=True,
    )
