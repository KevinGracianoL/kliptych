"""Interfaz del modelo de runtime de Kliptych (capa 0).

La interfaz es independiente del agente de desarrollo (opencode, Claude Code,
Codex) y del proveedor concreto. ``select_segments`` (brief §9) se incorpora en
la fase C, cuando exista su tipo.

``write_caption`` (brief §9) recibe ``Contract`` y un ``PieceContext`` en lugar
de la ``Piece`` del gate: la pieza final ya incluye el caption que se está
generando, así que el modelo recibe solo la identidad de la pieza (id y
plataforma). El pipeline no corrige la salida del modelo: el gate valida el
caption final contra el contrato y rechaza si falta una mención o un hashtag.
"""

from typing import Protocol

from pydantic import Field

from kliptych.contract import Contract, ContractDraft, Hashtag, Platform
from kliptych.contract.base import ContractBase


class ModelError(Exception):
    """Base de los errores del runtime LLM."""


class ModelUnavailableError(ModelError):
    """El backend no respondió o rechazó la petición tras los reintentos."""


class ModelOutputError(ModelError):
    """El backend respondió algo que no es un draft o caption válido."""


class ModelInputError(ModelError):
    """La petición no es válida para el contrato (p. ej. plataforma ausente)."""


class Caption(ContractBase):
    """Caption final de una pieza, con sus hashtags.

    Los hashtags usan la misma gramática de token que el contrato
    (``Hashtag``): un solo token con prefijo ``#``. Texto libre dentro de un
    hashtag no puede viajar por este campo, que termina en la metadata de
    entrega.
    """

    caption: str = Field(min_length=1)
    hashtags: tuple[Hashtag, ...] = ()


class PieceContext(ContractBase):
    """Identidad de la pieza cuyo caption se va a redactar."""

    piece_id: str = Field(min_length=1, max_length=64)
    platform: Platform


def caption_prompt_payload(contract: Contract, piece: PieceContext) -> dict[str, object]:
    """Construye el payload que el modelo recibe para redactar el caption.

    Es la fuente única del prompt y de la clave de replay: solo viaja lo que
    afecta al caption (reglas, idiomas y menciones/hashtags obligatorios), sin
    assets, rutas locales ni evidencia del brief.

    Args:
        contract: Contrato validado de la campaña.
        piece: Identidad de la pieza (id y plataforma).

    Returns:
        El payload serializable del prompt de caption.

    Raises:
        ModelInputError: Si el contrato no declara la plataforma de la pieza.
    """
    if piece.platform not in contract.platforms:
        msg = f"el contrato no declara la plataforma {piece.platform.value}"
        raise ModelInputError(msg)
    platform_rules = contract.platforms[piece.platform]
    return {
        "campaign_id": contract.campaign_id,
        "platform": piece.platform.value,
        "piece_id": piece.piece_id,
        "languages": contract.languages.model_dump(mode="json"),
        "caption_rules": platform_rules.caption_rules.model_dump(mode="json"),
        "required_mentions": list(platform_rules.required_mentions),
        "required_hashtags": list(platform_rules.required_hashtags),
        "prohibitions": list(contract.prohibitions),
        "spelling_locks": list(contract.spelling_locks),
    }


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

    def write_caption(self, contract: Contract, piece: PieceContext) -> Caption:
        """Redacta el caption de una pieza según el contrato.

        Args:
            contract: Contrato validado de la campaña.
            piece: Identidad de la pieza (id y plataforma).

        Returns:
            El caption con sus hashtags, validado estructuralmente.

        Raises:
            ModelInputError: Si el contrato no declara la plataforma de la pieza.
            ModelUnavailableError: Si el backend no está disponible.
            ModelOutputError: Si la salida no es un caption válido.
        """
        ...
