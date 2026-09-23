"""Ingesta de briefs locales a texto normalizado.

Todo el sistema es local, offline y determinista: no hay OAuth, tokens ni
llamadas a APIs en la nube. Los briefs de Google Docs o Notion entran como
archivo exportado (texto/markdown, PDF, DOCX) o como texto pegado
(``ingest_text`` / ``kliptych ingest -``).

El texto se normaliza a saltos de línea LF y se identifica con el mismo hash
que usa el runtime para las respuestas grabadas, de modo que un brief ingerido
se puede reproducir con ``RecordedModel``. Toda entrada se trata como dato no
confiable: cotas de tamaño, límites de descompresión y traducción de errores
de los parsers a ``IngestError`` en el borde.
"""

import io
import logging
import zipfile
from collections.abc import Iterator
from pathlib import Path
from typing import ClassVar, cast
from xml.etree import ElementTree as ET

from pydantic import BaseModel, ConfigDict, Field
from pypdf import PdfReader

from kliptych.hashing import brief_key, normalize_brief

MAX_BRIEF_BYTES = 16 * 1024 * 1024
MAX_DOCUMENT_XML_BYTES = 32 * 1024 * 1024
_TEXT_SUFFIXES = frozenset({".txt", ".md", ".markdown"})
_MARKDOWN_TYPES = frozenset({".md", ".markdown"})
_WORD_NAMESPACE = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
_P_TAG = f"{_WORD_NAMESPACE}p"
_T_TAG = f"{_WORD_NAMESPACE}t"
_BR_TAG = f"{_WORD_NAMESPACE}br"
_CR_TAG = f"{_WORD_NAMESPACE}cr"
_TAB_TAG = f"{_WORD_NAMESPACE}tab"
_INLINE_TEXT = {_BR_TAG: "\n", _CR_TAG: "\n", _TAB_TAG: "\t"}
_DOCX_MEDIA_TYPE = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"

# pypdf emite warnings de parseo a stderr; la ingesta traduce esos fallos a
# IngestError y no debe contaminar la salida de la CLI.
_PYPDF_LOGGER = logging.getLogger("pypdf")
_PYPDF_LOGGER.addHandler(logging.NullHandler())
_PYPDF_LOGGER.propagate = False


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
        IngestError: Si el archivo no existe, excede el tamaño máximo, el
            formato no está soportado, la codificación no es UTF-8 o el
            contenido no produce texto.
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
    return _build(source=str(path), media_type=media_type, text=text)


def ingest_text(text: str, *, source: str = "texto pegado") -> IngestedBrief:
    """Ingiere un brief pegado como texto plano.

    Falla con ``IngestError`` si el texto no produce contenido utilizable.

    Args:
        text: Texto crudo del brief (por ejemplo pegado desde Google Docs o
            Notion tras la exportación manual).
        source: Procedencia declarada para el reporte.

    Returns:
        El brief ingerido con procedencia y hash del texto normalizado.
    """
    return _build(source=source, media_type="text/plain", text=text)


def ingest_bytes(data: bytes, *, source: str) -> IngestedBrief:
    """Ingiere un brief desde bytes UTF-8 (por ejemplo stdin).

    Args:
        data: Contenido crudo del brief.
        source: Procedencia declarada para el reporte.

    Returns:
        El brief ingerido con procedencia y hash del texto normalizado.

    Raises:
        IngestError: Si el contenido excede el tamaño máximo o no es UTF-8.
    """
    if len(data) > MAX_BRIEF_BYTES:
        msg = f"el brief excede el tamaño máximo de {MAX_BRIEF_BYTES} bytes: {source}"
        raise IngestError(msg)
    return _build(source=source, media_type="text/plain", text=_decode_utf8(data, source))


def _build(*, source: str, media_type: str, text: str) -> IngestedBrief:
    normalized = normalize_brief(text.removeprefix("\ufeff"))
    if not normalized.strip():
        msg = f"el brief no contiene texto utilizable: {source}"
        raise IngestError(msg)
    return IngestedBrief(
        source=source,
        media_type=media_type,
        sha256=brief_key(normalized),
        text=normalized,
    )


def _check_size(path: Path) -> None:
    try:
        size = path.stat().st_size
    except OSError as error:
        msg = f"no se pudo leer el brief: {path}"
        raise IngestError(msg) from error
    if size > MAX_BRIEF_BYTES:
        msg = f"el brief excede el tamaño máximo de {MAX_BRIEF_BYTES} bytes: {path}"
        raise IngestError(msg)


def _read_limited(path: Path) -> bytes:
    _check_size(path)
    try:
        return path.read_bytes()
    except OSError as error:
        msg = f"no se pudo leer el brief: {path}"
        raise IngestError(msg) from error


def _read_text(path: Path) -> str:
    return _decode_utf8(_read_limited(path), path)


def _decode_utf8(data: bytes, source: str | Path) -> str:
    try:
        return data.decode("utf-8-sig")
    except UnicodeDecodeError as error:
        msg = f"el brief no está codificado en UTF-8: {source}"
        raise IngestError(msg) from error


def _read_pdf(path: Path) -> str:
    _check_size(path)
    # Frontera de parser externo: toda falla de pypdf se traduce a IngestError
    # conservando la causa (los PDF con contraseña de usuario fallan aquí).
    try:
        reader = PdfReader(path)
        pages = [page.extract_text() or "" for page in reader.pages]
    except Exception as error:
        msg = f"no se pudo leer el PDF: {path}"
        raise IngestError(msg) from error
    return "\n\n".join(pages)


def _read_docx(path: Path) -> str:
    _check_size(path)
    try:
        document = _read_document_xml(path)
    except IngestError:
        raise
    except Exception as error:
        msg = f"no se pudo leer el DOCX: {path}"
        raise IngestError(msg) from error
    try:
        paragraphs = _extract_paragraph_text(document)
    except ET.ParseError as error:
        msg = f"el DOCX tiene XML inválido: {path}"
        raise IngestError(msg) from error
    return "\n".join(paragraphs)


def _read_document_xml(path: Path) -> bytes:
    with zipfile.ZipFile(path) as archive:
        info = archive.getinfo("word/document.xml")
        if info.file_size > MAX_DOCUMENT_XML_BYTES:
            msg = (
                "el DOCX excede el tamaño máximo de documento descomprimido "
                f"({MAX_DOCUMENT_XML_BYTES} bytes): {path}"
            )
            raise IngestError(msg)
        return archive.read(info)


def _document_events(document: bytes) -> Iterator[tuple[str, ET.Element]]:
    # typeshed devuelve Any cuando se pasan `events` porque la forma del evento
    # depende de la lista pedida; aquí siempre se piden ("start", "end").
    return cast(
        "Iterator[tuple[str, ET.Element]]",
        ET.iterparse(io.BytesIO(document), events=("start", "end")),
    )


def _extract_paragraph_text(document: bytes) -> list[str]:
    """Extrae párrafos del cuerpo del DOCX en una sola pasada O(n).

    Mapea ``w:t`` a texto, ``w:br``/``w:cr`` a salto de línea y ``w:tab`` a
    tabulación; los párrafos sin texto se omiten. La recursión se evita para
    tolerar anidamiento profundo sin agotar la pila.

    Returns:
        El texto de cada párrafo, en orden de documento.
    """
    paragraphs: list[str] = []
    buffer: list[str] = []
    depth = 0
    for event, element in _document_events(document):
        if event == "start":
            if element.tag == _P_TAG:
                depth += 1
                if depth == 1:
                    buffer = []
            continue
        if element.tag == _P_TAG:
            depth -= 1
            if depth == 0 and buffer:
                paragraphs.append("".join(buffer))
        elif depth > 0:
            inline = _INLINE_TEXT.get(element.tag)
            text = (element.text or "") if element.tag == _T_TAG else inline
            if text:
                buffer.append(text)
        element.clear()
    return paragraphs
