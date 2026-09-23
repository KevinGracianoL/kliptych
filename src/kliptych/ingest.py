"""Ingesta de briefs locales a texto normalizado.

Formatos soportados: texto/markdown (UTF-8, con o sin BOM), PDF y DOCX. El
texto se normaliza a saltos de línea LF y se identifica con el mismo hash que
usa el runtime para las respuestas grabadas, de modo que un brief ingerido se
puede reproducir con ``RecordedModel``.

Google Docs y Notion requieren una decisión de autenticación del owner
(credenciales de API o export de enlaces compartidos); se incorporan cuando
esa decisión exista.
"""

import zipfile
from pathlib import Path
from typing import ClassVar
from xml.etree import ElementTree as ET

from pydantic import BaseModel, ConfigDict, Field
from pypdf import PdfReader
from pypdf.errors import PyPdfError

from kliptych.hashing import brief_key, normalize_brief

_TEXT_SUFFIXES = frozenset({".txt", ".md", ".markdown"})
_MARKDOWN_TYPES = frozenset({".md", ".markdown"})
_WORD_NAMESPACE = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
_DOCX_MEDIA_TYPE = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"


class IngestError(Exception):
    """El brief no se pudo ingerir."""


class IngestedBrief(BaseModel):
    """Brief ingerido: texto normalizado con procedencia y hash."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", frozen=True)

    source: str = Field(min_length=1)
    media_type: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    text: str = Field(min_length=1)


def ingest_file(path: Path) -> IngestedBrief:
    """Ingiere un brief local y devuelve su texto normalizado.

    Args:
        path: Ruta del archivo (texto/markdown, PDF o DOCX).

    Returns:
        El brief ingerido con procedencia y hash del texto normalizado.

    Raises:
        IngestError: Si el archivo no existe, el formato no está soportado, la
            codificación no es UTF-8 o el contenido no produce texto.
    """
    if not path.is_file():
        msg = f"el brief no existe en disco: {path}"
        raise IngestError(msg)
    suffix = path.suffix.lower()
    if suffix in _TEXT_SUFFIXES:
        media_type = "text/markdown" if suffix in _MARKDOWN_TYPES else "text/plain"
        text = _read_text(path)
    elif suffix == ".pdf":
        media_type = "application/pdf"
        text = _read_pdf(path)
    elif suffix == ".docx":
        media_type = _DOCX_MEDIA_TYPE
        text = _read_docx(path)
    else:
        msg = f"formato de brief no soportado: {suffix or path.name}"
        raise IngestError(msg)
    normalized = normalize_brief(text)
    if not normalized.strip():
        msg = f"el brief no contiene texto utilizable: {path}"
        raise IngestError(msg)
    return IngestedBrief(
        source=str(path),
        media_type=media_type,
        sha256=brief_key(normalized),
        text=normalized,
    )


def _read_text(path: Path) -> str:
    try:
        data = path.read_bytes()
    except OSError as error:
        msg = f"no se pudo leer el brief: {path}"
        raise IngestError(msg) from error
    try:
        return data.decode("utf-8-sig")
    except UnicodeDecodeError as error:
        msg = f"el brief no está codificado en UTF-8: {path}"
        raise IngestError(msg) from error


def _read_pdf(path: Path) -> str:
    try:
        reader = PdfReader(path)
    except (PyPdfError, OSError) as error:
        msg = f"no se pudo leer el PDF: {path}"
        raise IngestError(msg) from error
    if reader.is_encrypted:
        msg = f"el PDF está cifrado: {path}"
        raise IngestError(msg)
    try:
        pages = [page.extract_text() or "" for page in reader.pages]
    except (PyPdfError, OSError) as error:
        msg = f"no se pudo leer el PDF: {path}"
        raise IngestError(msg) from error
    return "\n\n".join(pages)


def _read_docx(path: Path) -> str:
    try:
        with zipfile.ZipFile(path) as archive:
            document = archive.read("word/document.xml")
    except (OSError, zipfile.BadZipFile, KeyError) as error:
        msg = f"no se pudo leer el DOCX: {path}"
        raise IngestError(msg) from error
    try:
        root = ET.fromstring(document)
    except ET.ParseError as error:
        msg = f"el DOCX tiene XML inválido: {path}"
        raise IngestError(msg) from error
    paragraphs: list[str] = []
    for paragraph in root.iter(f"{_WORD_NAMESPACE}p"):
        runs = [node.text or "" for node in paragraph.iter(f"{_WORD_NAMESPACE}t")]
        if runs:
            paragraphs.append("".join(runs))
    return "\n".join(paragraphs)
