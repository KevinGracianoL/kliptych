"""Validador del hook de apertura (CB22), indexado por rule_id.

- ``hook.keyword``: la palabra clave de apertura debe aparecer en los
  subtítulos o en el texto en pantalla con inicio <= 3.0 s. La comparación
  normaliza (NFKD sin diacríticos + casefold). Sin la palabra en ventana
  el resultado es ``fail``; sin segmentos con tiempos (solo
  ``subtitle_text`` plano) la ventana no es verificable y también es
  ``fail`` (fail-closed).

El volumen de los primeros 3 s se mide con ``volumedetect`` y se adjunta
como nota informativa: jamás modifica el veredicto.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from operator import itemgetter
from typing import TYPE_CHECKING

from kliptych.gate.models import CheckOutcome, CheckStatus, GateContext
from kliptych.gate.text import contains_phrase, normalize_text

if TYPE_CHECKING:
    from pathlib import Path

    from kliptych.gate.models import Piece

_RULE = "hook.keyword"
_HOOK_WINDOW_S = 3.0
_VOLUME_TIMEOUT_S = 30.0
_MAX_VOLUME_RE = re.compile(r"max_volume:\s*(-inf|inf|-?\d+(?:\.\d+)?)\s*dB")
_VOLUME_NOTE = "informativo: el volumen de los primeros 3 s no modifica el veredicto"


def check_hook_keyword(context: GateContext) -> CheckOutcome:
    """Verifica la palabra clave de apertura en la ventana de 3 s.

    Args:
        context: Contexto resuelto del gate.

    Returns:
        PASS si la campaña no exige hook o la palabra aparece en
        subtítulos o texto en pantalla con inicio <= 3.0 s; FAIL si falta,
        llega tarde o no hay segmentos con tiempos que verificar. En todos
        los casos activos la evidencia trae el volumen medido de los
        primeros 3 s como nota informativa.
    """
    keyword = context.contract.hook_keyword
    if keyword is None:
        return CheckOutcome(
            status=CheckStatus.PASS,
            evidence={"rule": _RULE, "active": False},
        )
    if not normalize_text(keyword):
        return CheckOutcome(
            status=CheckStatus.FAIL,
            evidence={
                "rule": _RULE,
                "keyword": keyword,
                "reason": "la palabra clave quedó vacía tras normalizar",
            },
        )
    hits = _hook_hits(keyword, context.piece)
    volume: dict[str, object] = {
        "volume_first_3s_db": _measure_first_3s_volume(context.piece.artifact_path),
        "volume_note": _VOLUME_NOTE,
    }
    if hits:
        return CheckOutcome(
            status=CheckStatus.PASS,
            evidence={
                "rule": _RULE,
                "keyword": keyword,
                "window_s": _HOOK_WINDOW_S,
                "matched_at_s": [round(start, 3) for _, start in hits],
                "matched_in": sorted({source for source, _ in hits}),
                **volume,
            },
        )
    return CheckOutcome(
        status=CheckStatus.FAIL,
        evidence={
            "rule": _RULE,
            "keyword": keyword,
            "window_s": _HOOK_WINDOW_S,
            "reason": (
                "la palabra clave no aparece en subtítulos ni texto en pantalla "
                f"con inicio <= {_HOOK_WINDOW_S} s"
            ),
            **volume,
        },
    )


def _hook_hits(keyword: str, piece: Piece) -> list[tuple[str, float]]:
    """Busca la palabra clave en ventana en ambas fuentes temporizadas.

    Args:
        keyword: Palabra clave requerida por el contrato.
        piece: Pieza con segmentos de subtítulos y texto en pantalla.

    Returns:
        Los (origen, inicio) con la palabra en ventana, en orden temporal.
    """
    sources = (
        ("subtitles", piece.subtitle_segments),
        ("screen", piece.screen_text_segments),
    )
    hits: list[tuple[str, float]] = []
    for source, segments in sources:
        window_segments = [s for s in segments if s.start_s <= _HOOK_WINDOW_S]
        source_hits: list[float] = [
            segment.start_s for segment in window_segments if contains_phrase(segment.text, keyword)
        ]
        if not source_hits and window_segments:
            for i, seg in enumerate(window_segments):
                sub_concat = " ".join(s.text for s in window_segments[i:])
                if contains_phrase(sub_concat, keyword):
                    source_hits = [seg.start_s]
        hits.extend((source, start_s) for start_s in source_hits)
    deduped = list(dict.fromkeys(hits))
    return sorted(deduped, key=itemgetter(1))


def _measure_first_3s_volume(artifact: Path) -> float | None:
    """Mide el ``max_volume`` de los primeros 3 s (solo informativo).

    Args:
        artifact: Ruta del artefacto final a medir.

    Returns:
        El ``max_volume`` en dB, o ``None`` si no se pudo medir (sin
        archivo, sin ffmpeg, sin pista de audio o fallo de medición). Un
        ``None`` nunca modifica el veredicto: el volumen es nota, no regla.
    """
    if not artifact.is_file():
        return None
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        return None
    argv = [
        ffmpeg,
        "-hide_banner",
        "-nostdin",
        "-v",
        "info",
        "-t",
        f"{_HOOK_WINDOW_S}",
        "-i",
        str(artifact),
        "-af",
        "volumedetect",
        "-f",
        "null",
        "-",
    ]
    try:
        completed = subprocess.run(
            argv,
            capture_output=True,
            text=True,
            timeout=_VOLUME_TIMEOUT_S,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return None
    match = _MAX_VOLUME_RE.search(completed.stderr if completed.returncode == 0 else "")
    if match is None:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None
