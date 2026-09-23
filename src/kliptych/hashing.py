"""Hashes sha256 deterministas para archivos y texto canónico."""

import json
from collections.abc import Mapping
from hashlib import sha256
from pathlib import Path

_CHUNK_SIZE = 1024 * 1024


def sha256_file(path: Path) -> str:
    """Calcula el sha256 de un archivo leyéndolo por bloques.

    Args:
        path: Ruta del archivo a hashear.

    Returns:
        El digest sha256 en hexadecimal.
    """
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(_CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(data: bytes) -> str:
    """Calcula el sha256 de un bloque de bytes.

    Args:
        data: Contenido a hashear.

    Returns:
        El digest sha256 en hexadecimal.
    """
    return sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    """Calcula el sha256 de un texto codificado en UTF-8.

    Args:
        text: Texto a hashear.

    Returns:
        El digest sha256 en hexadecimal.
    """
    return sha256(text.encode("utf-8")).hexdigest()


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


def sha256_canonical_json(payload: Mapping[str, object]) -> str:
    """Calcula el sha256 de un JSON canónico: claves ordenadas y sin espacios.

    Args:
        payload: Mapeo serializable a JSON.

    Returns:
        El digest sha256 en hexadecimal, estable ante el orden de las claves.
    """
    canonical = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return sha256_text(canonical)
