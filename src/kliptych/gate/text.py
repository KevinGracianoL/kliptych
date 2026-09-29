"""Normalización de texto y coincidencia con fronteras para validadores del gate.

Preserva la 'ñ' al remover diacríticos (NFKD) para que 'campeón' case con
'campeon' pero 'ano' no case con 'año', ni 'pena' con 'peña'.
Colapsa espacios múltiples y saltos de línea a un único espacio, y trata el
guion bajo '_' como separador de frontera no alfanumérica.
"""

from __future__ import annotations

import re
import unicodedata

__all__ = ["contains_phrase", "normalize_text"]

_ENIE_SENTINEL = "\x00enie\x00"


def normalize_text(text: str) -> str:
    """Normaliza un texto preservando la 'ñ' y colapsando espacios en blanco.

    Descompone con NFKD eliminando marcas combinatorias, pero protegiendo
    la 'ñ' para evitar colisiones semánticas ('ano' vs 'año'). Colapsa
    múltiples espacios y saltos de línea a un solo espacio.

    Args:
        text: Texto de entrada.

    Returns:
        Texto normalizado en minúsculas y sin acentos salvo 'ñ'.
    """
    nfc = unicodedata.normalize("NFC", text)
    folded = nfc.casefold()
    protected = folded.replace("ñ", _ENIE_SENTINEL)
    decomposed = unicodedata.normalize("NFKD", protected)
    filtered = "".join(char for char in decomposed if not unicodedata.combining(char))
    restored = filtered.replace(_ENIE_SENTINEL, "ñ")
    return re.sub(r"\s+", " ", restored).strip()


def contains_phrase(haystack: str, needle: str) -> bool:
    """Verifica si needle aparece en haystack con frontera de palabra completa.

    Trata el guion bajo '_' y los espacios como separadores de frontera y
    admite espacios o guiones bajos entre palabras de frases múltiples.

    Args:
        haystack: Texto donde buscar.
        needle: Frase o palabra buscada.

    Returns:
        True si la frase normalizada aparece con frontera completa.
    """
    norm_needle = normalize_text(needle)
    if not norm_needle:
        return False
    norm_haystack = normalize_text(haystack)
    if not norm_haystack:
        return False
    words = [w for w in re.split(r"[\s_]+", norm_needle) if w]
    if not words:
        return False
    escaped_words = [re.escape(w) for w in words]
    phrase_pattern = r"[\s_]+".join(escaped_words)
    pattern = rf"(?<![^\W_]){phrase_pattern}(?![^\W_])"
    return re.search(pattern, norm_haystack) is not None
