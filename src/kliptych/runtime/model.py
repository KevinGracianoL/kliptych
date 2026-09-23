"""Interfaz del modelo de runtime de Kliptych (capa 0).

La interfaz es independiente del agente de desarrollo (opencode, Claude Code,
Codex) y del proveedor concreto. ``select_segments`` y ``write_caption``
(brief §9) se incorporan en las fases C y B, cuando existan sus tipos.
"""

from typing import Protocol

from kliptych.contract import ContractDraft


class ModelError(Exception):
    """Base de los errores del runtime LLM."""


class ModelUnavailableError(ModelError):
    """El backend no respondió o rechazó la petición tras los reintentos."""


class ModelOutputError(ModelError):
    """El backend respondió algo que no es un ``ContractDraft`` válido."""


class CampaignModel(Protocol):
    """Interfaz del modelo de runtime que consume el pipeline."""

    def extract_contract(self, brief: str) -> ContractDraft:
        """Extrae el contrato del brief, con evidencia por campo.

        Args:
            brief: Texto crudo del brief de campaña.

        Returns:
            El draft extraído y validado estructuralmente.

        Raises:
            ModelUnavailableError: Si el backend no está disponible.
            ModelOutputError: Si la salida no es un draft válido.
        """
        ...
