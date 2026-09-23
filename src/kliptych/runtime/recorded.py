"""Backend de repetición con respuestas grabadas.

Los tests y las demos nunca llaman modelos reales: reproducen respuestas
grabadas (fixtures golden) desde disco. La clave es el sha256 del brief
normalizado a saltos de línea LF, de modo que el replay es estable entre
plataformas.
"""

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import ClassVar

from pydantic import Field

from kliptych.contract import ContractDraft
from kliptych.contract.base import ContractBase
from kliptych.hashing import sha256_text
from kliptych.runtime.model import ModelUnavailableError

_SHA256_PATTERN = r"^[0-9a-f]{64}$"


def normalize_brief(brief: str) -> str:
    """Normaliza un brief para hashearlo de forma estable.

    Args:
        brief: Texto crudo del brief.

    Returns:
        El brief con saltos de línea LF (CRLF y CR incluidos).
    """
    return brief.replace("\r\n", "\n").replace("\r", "\n")


def brief_key(brief: str) -> str:
    """Calcula la clave sha256 de un brief normalizado.

    Args:
        brief: Texto crudo del brief.

    Returns:
        El digest sha256 en hexadecimal.
    """
    return sha256_text(normalize_brief(brief))


class RecordedDocument(ContractBase):
    """Respuesta grabada de un extractor para un brief concreto."""

    brief_sha256: str = Field(pattern=_SHA256_PATTERN)
    prompt_version: str = Field(min_length=1)
    draft: ContractDraft


class RecordedModel:
    """Reproduce drafts grabados, indexados por hash del brief."""

    model_version: ClassVar[str] = "recorded"

    def __init__(self, documents: Sequence[RecordedDocument] = ()) -> None:
        """Carga los documentos grabados en memoria.

        Args:
            documents: Documentos a indexar por hash de brief.

        Raises:
            ValueError: Si hay dos documentos para el mismo brief.
        """
        self._documents: dict[str, RecordedDocument] = {}
        for document in documents:
            if document.brief_sha256 in self._documents:
                msg = f"documento duplicado para el brief {document.brief_sha256[:12]}"
                raise ValueError(msg)
            self._documents[document.brief_sha256] = document

    @property
    def documents(self) -> Mapping[str, RecordedDocument]:
        """Documentos cargados, indexados por hash de brief.

        Returns:
            Vista de solo lectura del índice.
        """
        return self._documents

    @classmethod
    def from_directory(
        cls,
        directory: Path,
        *,
        expected_prompt_version: str | None = None,
    ) -> "RecordedModel":
        """Carga todos los documentos ``*.json`` de un directorio.

        Args:
            directory: Directorio con las respuestas grabadas.
            expected_prompt_version: Si se indica, exige que todas las
                grabaciones se hayan hecho con esa versión del prompt.

        Returns:
            El modelo con los documentos cargados.

        Raises:
            ModelUnavailableError: Si alguna grabación usa otra versión del
                prompt que la esperada.
        """
        documents = [
            RecordedDocument.model_validate_json(path.read_text(encoding="utf-8"))
            for path in sorted(directory.glob("*.json"))
        ]
        if expected_prompt_version is not None:
            stale = sorted(
                {
                    document.prompt_version
                    for document in documents
                    if document.prompt_version != expected_prompt_version
                }
            )
            if stale:
                msg = f"grabaciones con prompt obsoleto: {', '.join(stale)}"
                raise ModelUnavailableError(msg)
        return cls(documents)

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
