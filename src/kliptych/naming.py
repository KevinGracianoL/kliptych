"""Identificadores seguros para segmentos de ruta del pipeline.

Fuente única de la política: los ids que se usan como nombre de archivo o
directorio (``asset_id``, ``piece_id``, ``campaign_id``) no pueden contener
separadores, traversal, nombres reservados de Windows ni exceder la longitud
máxima.
"""

import re

MAX_SEGMENT_LENGTH = 64
_SAFE_SEGMENT = re.compile(r"[A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?")
_RESERVED_NAMES = frozenset(
    {
        "aux",
        "com1",
        "com2",
        "com3",
        "com4",
        "com5",
        "com6",
        "com7",
        "com8",
        "com9",
        "con",
        "lpt1",
        "lpt2",
        "lpt3",
        "lpt4",
        "lpt5",
        "lpt6",
        "lpt7",
        "lpt8",
        "lpt9",
        "nul",
        "prn",
    }
)


def is_safe_segment(value: str) -> bool:
    """Indica si el valor es un segmento de ruta seguro y portable.

    Args:
        value: Identificador a validar.

    Returns:
        ``True`` si no es vacío, no excede la longitud máxima, no contiene
        separadores ni traversal y no es un nombre reservado.
    """
    return not (
        len(value) > MAX_SEGMENT_LENGTH
        or value in {".", ".."}
        or not _SAFE_SEGMENT.fullmatch(value)
        or value.split(".", 1)[0].casefold() in _RESERVED_NAMES
    )
