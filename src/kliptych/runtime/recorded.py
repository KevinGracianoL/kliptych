"""Backend de repetición con respuestas grabadas.

Los tests y las demos nunca llaman modelos reales: reproducen respuestas
grabadas (fixtures golden) desde disco. Los drafts se indexan por el sha256 del
brief normalizado a saltos de línea LF; los captions, por el hash canónico del
payload exacto del prompt de caption, de modo que el replay es estable entre
plataformas y no depende de datos que no viajan al modelo.
"""

from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import ClassVar

from pydantic import Field

from kliptych.contract import Contract, ContractDraft, Platform
from kliptych.contract.base import ContractBase
from kliptych.hashing import brief_key, sha256_canonical_json, sha256_text
from kliptych.runtime.model import (
    Caption,
    ModelUnavailableError,
    PieceContext,
    caption_prompt_payload,
)

_SHA256_PATTERN = r"^[0-9a-f]{64}$"


class RecordedDocument(ContractBase):
    """Respuesta grabada de un extractor para un brief concreto."""

    brief_sha256: str = Field(pattern=_SHA256_PATTERN)
    prompt_version: str = Field(min_length=1)
    draft: ContractDraft


class RecordedCaptionDocument(ContractBase):
    """Caption grabado para el prompt de una pieza concreta.

    ``prompt_sha256`` es el hash canónico del payload exacto que recibe el
    modelo (``caption_prompt_payload``): la clave no depende de datos que no
    viajan al prompt, como los hashes de los assets.
    """

    prompt_sha256: str = Field(pattern=_SHA256_PATTERN)
    platform: Platform
    piece_id: str = Field(min_length=1, max_length=64)
    prompt_version: str = Field(min_length=1)
    caption: Caption


def _caption_key(contract: Contract, piece: PieceContext) -> str:
    return sha256_canonical_json(caption_prompt_payload(contract, piece))


class RecordedModel:
    """Reproduce drafts y captions grabados, indexados por su clave estable."""

    model_version: ClassVar[str] = "recorded"

    def __init__(
        self,
        documents: Sequence[RecordedDocument] = (),
        captions: Sequence[RecordedCaptionDocument] = (),
    ) -> None:
        """Carga los documentos grabados en memoria.

        Args:
            documents: Drafts a indexar por hash de brief.
            captions: Captions a indexar por contrato, plataforma y pieza.

        Raises:
            ValueError: Si hay dos grabaciones con la misma clave.
        """
        self._documents: dict[str, RecordedDocument] = {}
        for document in documents:
            if document.brief_sha256 in self._documents:
                msg = f"documento duplicado para el brief {document.brief_sha256[:12]}"
                raise ValueError(msg)
            self._documents[document.brief_sha256] = document
        self._captions: dict[str, RecordedCaptionDocument] = {}
        for caption in captions:
            key = self._caption_document_key(caption)
            if key in self._captions:
                msg = f"caption duplicado para la pieza {caption.platform.value}/{caption.piece_id}"
                raise ValueError(msg)
            self._captions[key] = caption

    @staticmethod
    def _caption_document_key(document: RecordedCaptionDocument) -> str:
        return document.prompt_sha256

    @property
    def documents(self) -> Mapping[str, RecordedDocument]:
        """Drafts cargados, indexados por hash de brief.

        Returns:
            Vista de solo lectura del índice.
        """
        return self._documents

    @property
    def captions(self) -> Mapping[str, RecordedCaptionDocument]:
        """Captions cargados, indexados por contrato, plataforma y pieza.

        Returns:
            Vista de solo lectura del índice.
        """
        return self._captions

    @classmethod
    def from_directory(
        cls,
        directory: Path,
        *,
        expected_prompt_version: str | None = None,
        expected_caption_prompt_version: str | None = None,
    ) -> "RecordedModel":
        """Carga los documentos ``*.draft.json`` y ``*.caption.json`` del directorio.

        Args:
            directory: Directorio con las respuestas grabadas.
            expected_prompt_version: Si se indica, exige que los drafts se
                hayan grabado con esa versión del prompt.
            expected_caption_prompt_version: Si se indica, exige que los
                captions se hayan grabado con esa versión del prompt.

        Returns:
            El modelo con las grabaciones cargadas.

        Raises:
            ModelUnavailableError: Si hay archivos JSON no reconocidos, o
                alguna grabación usa otra versión del prompt que la esperada.
        """
        draft_paths = sorted(directory.glob("*.draft.json"))
        caption_paths = sorted(directory.glob("*.caption.json"))
        recognized = {path.name for path in (*draft_paths, *caption_paths)}
        unknown = sorted(
            path.name for path in directory.glob("*.json") if path.name not in recognized
        )
        if unknown:
            msg = f"grabaciones no reconocidas en {directory}: {', '.join(unknown)}"
            raise ModelUnavailableError(msg)
        documents = [
            RecordedDocument.model_validate_json(path.read_text(encoding="utf-8"))
            for path in draft_paths
        ]
        captions = [
            RecordedCaptionDocument.model_validate_json(path.read_text(encoding="utf-8"))
            for path in caption_paths
        ]
        _require_prompt_version(
            (document.prompt_version for document in documents),
            expected=expected_prompt_version,
            what="grabaciones",
        )
        _require_prompt_version(
            (caption.prompt_version for caption in captions),
            expected=expected_caption_prompt_version,
            what="captions",
        )
        return cls(documents, captions)

    def extract_contract(self, brief: str) -> ContractDraft:
        """Devuelve el draft grabado para el brief.

        Args:
            brief: Texto crudo del brief de campaña.

        Returns:
            El draft grabado.

        Raises:
            ModelUnavailableError: Si no hay respuesta grabada para el brief.
        """
        key = brief_key(brief)
        document = self._documents.get(key)
        if document is None:
            msg = f"sin respuesta grabada para el brief {key[:12]}"
            raise ModelUnavailableError(msg)
        return document.draft

    def write_caption(self, contract: Contract, piece: PieceContext) -> Caption:
        """Devuelve el caption grabado para el contrato, plataforma y pieza.

        Args:
            contract: Contrato validado de la campaña.
            piece: Identidad de la pieza (id y plataforma).

        Returns:
            El caption grabado.

        Raises:
            ModelInputError: Si el contrato no declara la plataforma de la pieza.
            ModelUnavailableError: Si no hay caption grabado para esa pieza.
        """
        key = _caption_key(contract, piece)
        document = self._captions.get(key)
        if document is None:
            msg = f"sin caption grabado para {piece.platform.value}/{piece.piece_id}"
            raise ModelUnavailableError(msg)
        return document.caption


def _require_prompt_version(
    versions: Iterable[str],
    *,
    expected: str | None,
    what: str,
) -> None:
    if expected is None:
        return
    stale = sorted({version for version in versions if version != expected})
    if stale:
        msg = f"{what} con prompt obsoleto: {', '.join(stale)}"
        raise ModelUnavailableError(msg)


def record_response(
    brief: str,
    draft: ContractDraft,
    *,
    prompt_version: str,
    directory: Path,
) -> Path:
    """Escribe una respuesta grabada como fixture JSON.

    Args:
        brief: Brief al que corresponde la respuesta.
        draft: Draft devuelto por el extractor.
        prompt_version: Versión del prompt usada al grabar.
        directory: Directorio destino; se crea si no existe.

    Returns:
        La ruta del archivo escrito.
    """
    directory.mkdir(parents=True, exist_ok=True)
    document = RecordedDocument(
        brief_sha256=brief_key(brief),
        prompt_version=prompt_version,
        draft=draft,
    )
    path = directory / f"{document.brief_sha256[:12]}.draft.json"
    _ = path.write_text(document.model_dump_json(indent=2), encoding="utf-8")
    return path


def record_caption(
    contract: Contract,
    piece: PieceContext,
    caption: Caption,
    *,
    prompt_version: str,
    directory: Path,
) -> Path:
    """Escribe un caption grabado como fixture JSON.

    Args:
        contract: Contrato al que corresponde el caption.
        piece: Identidad de la pieza grabada.
        caption: Caption devuelto por el modelo.
        prompt_version: Versión del prompt usada al grabar.
        directory: Directorio destino; se crea si no existe.

    Returns:
        La ruta del archivo escrito.
    """
    directory.mkdir(parents=True, exist_ok=True)
    key = _caption_key(contract, piece)
    document = RecordedCaptionDocument(
        prompt_sha256=key,
        platform=piece.platform,
        piece_id=piece.piece_id,
        prompt_version=prompt_version,
        caption=caption,
    )
    path = directory / f"{sha256_text(key)[:12]}.caption.json"
    _ = path.write_text(document.model_dump_json(indent=2), encoding="utf-8")
    return path
