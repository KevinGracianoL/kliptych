"""Hashes sha256 deterministas para archivos y texto canónico."""

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


def sha256_text(text: str) -> str:
    """Calcula el sha256 de un texto codificado en UTF-8.

    Args:
        text: Texto a hashear.

    Returns:
        El digest sha256 en hexadecimal.
    """
    return sha256(text.encode("utf-8")).hexdigest()
