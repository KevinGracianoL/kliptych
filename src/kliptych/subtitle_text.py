"""Texto de subtítulos por pieza para el gate.

El validador ``subtitles.spelling_lock`` verifica ``Piece.subtitle_text``: este
módulo construye ese texto desde la transcripción recortada al rango temporal
del segmento de la pieza (nunca la transcripción completa del VOD) o lo
hidrata desde artefactos persistidos en el directorio de trabajo
(``transcript.json`` o el ``.ass`` de la pieza) para que ``--resume`` nunca
evalúe el gate con ``subtitle_text=None`` cuando el texto existe en disco.

No toca ffmpeg ni faster-whisper: solo ``pathlib``, ``json`` y los modelos de
transcripción y contrato. Es seguro importarlo desde el controlador de
campañas, que no puede cargar el motor de video.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

from kliptych.gate.models import SubtitleSegment
from kliptych.transcribe import Transcript

if TYPE_CHECKING:
    from pathlib import Path

    from kliptych.contract import Segment

_TAG_PATTERN = re.compile(r"\{[^}]*\}")
_DIALOGUE_PREFIX = "Dialogue:"
_DIALOGUE_FIELD_COUNT = 10
_DIALOGUE_TEXT_INDEX = 9


@dataclass(frozen=True, slots=True)
class PieceSubtitleSources:
    """Fuentes para derivar el ``subtitle_text`` de una pieza.

    Attributes:
        transcript: Transcripción en memoria del pipeline (``None`` en modo
            repost o al reanudar sin ella).
        segments: Segmentos seleccionados, alineados por índice con los videos
            finales del lote.
        subtitles_path: ``.ass`` primario del pipeline (``None`` si no existe).
        work_dir: Directorio de trabajo con ``transcript.json`` y
            ``subtitles[.ass|_NN.ass]`` persistidos para ``--resume``.
    """

    transcript: Transcript | None = None
    segments: tuple[Segment, ...] = ()
    subtitles_path: Path | None = None
    work_dir: Path | None = None


def segment_subtitle_text(transcript: Transcript, segment: Segment) -> str | None:
    """Recorta el texto de la transcripción al rango temporal del segmento.

    Solo entran las palabras que se solapan con ``[start_s, end_s)`` (la misma
    regla de solape que usa el render de subtítulos): una pieza jamás recibe
    la transcripción completa del VOD sin recortar.

    Args:
        transcript: Transcripción word-level del video fuente.
        segment: Rango temporal de la pieza dentro del video fuente.

    Returns:
        Las palabras del rango unidas con espacios, o ``None`` si el rango no
        contiene palabras (nada que verificar).
    """
    words = [
        word.text
        for word in transcript.words
        if word.end_s > segment.start_s and word.start_s < segment.end_s
    ]
    if not words:
        return None
    return " ".join(words)


def subtitle_text_from_ass(path: Path) -> str | None:
    r"""Extrae el texto de diálogo de un archivo ``.ass``.

    Lee las líneas ``Dialogue:`` (el texto es el décimo campo), elimina las
    etiquetas de override ``{...}`` (karaoke ``\k`` incluido), convierte
    ``\N`` en espacios y une los diálogos con un espacio.

    Args:
        path: Ruta del archivo ``.ass`` de la pieza.

    Returns:
        El texto de diálogo, o ``None`` si el archivo no existe, no se puede
        leer o no trae diálogo aprovechable.
    """
    try:
        content = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None
    texts: list[str] = []
    for line in content.splitlines():
        if not line.startswith(_DIALOGUE_PREFIX):
            continue
        fields = line[len(_DIALOGUE_PREFIX) :].split(",", _DIALOGUE_TEXT_INDEX)
        if len(fields) != _DIALOGUE_FIELD_COUNT:
            continue
        cleaned = _TAG_PATTERN.sub("", fields[_DIALOGUE_TEXT_INDEX]).replace("\\N", " ").strip()
        if cleaned:
            texts.append(cleaned)
    if not texts:
        return None
    return " ".join(texts)


def hydrate_piece_subtitle_text(
    *,
    transcript: Transcript | None,
    segment: Segment | None,
    subtitles_path: Path | None = None,
    work_dir: Path | None = None,
    suffix: str = "",
) -> str | None:
    """Deriva el ``subtitle_text`` de una pieza con cadena de respaldo a disco.

    Orden de prioridad: transcripción en memoria recortada al segmento (fuente
    de verdad cuando está completa), ``transcript.json`` persistido recortado
    al segmento (``--resume`` sin transcripción en memoria) y líneas de
    diálogo del ``.ass`` de la pieza.

    Args:
        transcript: Transcripción en memoria, o ``None`` si no está disponible.
        segment: Rango temporal de la pieza, o ``None`` si no se conoce.
        subtitles_path: ``.ass`` primario del pipeline, o ``None``.
        work_dir: Directorio de trabajo con artefactos persistidos, o ``None``.
        suffix: Sufijo del lote (``""`` o ``"_NN"``) para localizar el
            ``.ass`` indexado.

    Returns:
        El texto de subtítulos de la pieza, o ``None`` si ninguna fuente lo
        aporta (el gate lo reporta como no verificable, jamás como PASS).
    """
    if transcript is not None and segment is not None:
        return segment_subtitle_text(transcript, segment)
    if segment is not None and work_dir is not None:
        saved = _load_transcript(work_dir / "transcript.json")
        if saved is not None:
            text = segment_subtitle_text(saved, segment)
            if text is not None:
                return text
    for candidate in _ass_candidates(subtitles_path, work_dir, suffix):
        text = subtitle_text_from_ass(candidate)
        if text is not None:
            return text
    return None


def piece_subtitle_text(
    sources: PieceSubtitleSources | None, *, index: int, total: int
) -> str | None:
    """Deriva el ``subtitle_text`` de la pieza ``index`` de un lote.

    Alinea el índice del video con el segmento seleccionado y resuelve el
    sufijo del ``.ass`` indexado (``"_NN"`` en lotes múltiples).

    Args:
        sources: Fuentes de subtítulos del lote, o ``None`` si la pieza no
            tiene de dónde derivar texto.
        index: Índice del video final dentro del lote.
        total: Número de videos finales del lote.

    Returns:
        El texto de subtítulos de la pieza, o ``None`` sin fuentes o sin
        texto para ese índice.
    """
    if sources is None:
        return None
    segment = sources.segments[index] if 0 <= index < len(sources.segments) else None
    suffix = f"_{index:02d}" if total > 1 else ""
    return hydrate_piece_subtitle_text(
        transcript=sources.transcript,
        segment=segment,
        subtitles_path=sources.subtitles_path,
        work_dir=sources.work_dir,
        suffix=suffix,
    )


_TIME_PARTS_HMS = 3
_TIME_PARTS_MS = 2


def _parse_ass_time(time_str: str) -> float | None:
    parts = time_str.strip().split(":")
    try:
        if len(parts) == _TIME_PARTS_HMS:
            return float(parts[0]) * 3600.0 + float(parts[1]) * 60.0 + float(parts[2])
        if len(parts) == _TIME_PARTS_MS:
            return float(parts[0]) * 60.0 + float(parts[1])
        return float(time_str)
    except ValueError:
        return None


def segment_subtitle_segments(
    transcript: Transcript, segment: Segment | None = None
) -> tuple[SubtitleSegment, ...]:
    """Deriva los segmentos de subtítulos con tiempos relativos a la pieza.

    Args:
        transcript: Transcripción completa.
        segment: Rango temporal del segmento dentro de la fuente, o None si la
            pieza abarca todo el audio.

    Returns:
        Segmentos con marcas de tiempo relativas al inicio de la pieza (0.0s).
    """
    segments: list[SubtitleSegment] = []
    if segment is not None:
        duration_s = segment.end_s - segment.start_s
        for word in transcript.words:
            if word.end_s <= segment.start_s or word.start_s >= segment.end_s:
                continue
            start_s = max(0.0, word.start_s - segment.start_s)
            end_s = min(duration_s, word.end_s - segment.start_s)
            if end_s <= start_s:
                continue
            clean = word.text.strip()
            if clean:
                segments.append(SubtitleSegment(text=clean, start_s=start_s, end_s=end_s))
    else:
        for word in transcript.words:
            if word.end_s <= word.start_s:
                continue
            clean = word.text.strip()
            if clean:
                segments.append(SubtitleSegment(text=clean, start_s=word.start_s, end_s=word.end_s))
    return tuple(segments)


def subtitle_segments_from_ass(path: Path) -> tuple[SubtitleSegment, ...]:
    r"""Extrae los segmentos de diálogo con marcas de tiempo de un archivo ``.ass``.

    Args:
        path: Ruta del archivo ``.ass``.

    Returns:
        Tupla de SubtitleSegment con tiempos relativos y texto limpio.
    """
    try:
        content = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return ()
    segments: list[SubtitleSegment] = []
    for line in content.splitlines():
        if not line.startswith(_DIALOGUE_PREFIX):
            continue
        fields = line[len(_DIALOGUE_PREFIX) :].split(",", _DIALOGUE_TEXT_INDEX)
        if len(fields) != _DIALOGUE_FIELD_COUNT:
            continue
        start_s = _parse_ass_time(fields[1])
        end_s = _parse_ass_time(fields[2])
        if start_s is None or end_s is None or end_s <= start_s:
            continue
        cleaned = _TAG_PATTERN.sub("", fields[_DIALOGUE_TEXT_INDEX]).replace("\\N", " ").strip()
        if cleaned:
            segments.append(SubtitleSegment(text=cleaned, start_s=start_s, end_s=end_s))
    return tuple(segments)


def _segments_from_sources(
    transcript: Transcript | None,
    segment: Segment | None,
    work_dir: Path | None,
) -> tuple[SubtitleSegment, ...]:
    if transcript is not None:
        return segment_subtitle_segments(transcript, segment)
    if work_dir is not None:
        saved = _load_transcript(work_dir / "transcript.json")
        if saved is not None:
            return segment_subtitle_segments(saved, segment)
    return ()


def hydrate_piece_subtitle_segments(
    *,
    transcript: Transcript | None,
    segment: Segment | None,
    subtitles_path: Path | None = None,
    work_dir: Path | None = None,
    suffix: str = "",
) -> tuple[SubtitleSegment, ...]:
    """Deriva los ``subtitle_segments`` de una pieza con cadena de respaldo a disco.

    Args:
        transcript: Transcripción en memoria, o None.
        segment: Rango temporal de la pieza, o None.
        subtitles_path: Archivo .ass primario, o None.
        work_dir: Directorio de trabajo con artefactos persistidos, o None.
        suffix: Sufijo de lote ("" o "_NN").

    Returns:
        Tupla de SubtitleSegment de la pieza (vacía si no hay fuentes disponibles).
    """
    scoped = _segments_from_sources(transcript, segment, work_dir)
    if scoped:
        return scoped
    for candidate in _ass_candidates(subtitles_path, work_dir, suffix):
        segments = subtitle_segments_from_ass(candidate)
        if segments:
            return segments
    return _segments_from_sources(transcript, None, work_dir)


def piece_subtitle_segments(
    sources: PieceSubtitleSources | None, *, index: int, total: int
) -> tuple[SubtitleSegment, ...]:
    """Deriva los ``subtitle_segments`` de la pieza ``index`` de un lote.

    Args:
        sources: Fuentes de subtítulos del lote, o None.
        index: Índice del video final dentro del lote.
        total: Número de videos finales del lote.

    Returns:
        Tupla de SubtitleSegment de la pieza.
    """
    if sources is None:
        return ()
    segment = sources.segments[index] if 0 <= index < len(sources.segments) else None
    suffix = f"_{index:02d}" if total > 1 else ""
    return hydrate_piece_subtitle_segments(
        transcript=sources.transcript,
        segment=segment,
        subtitles_path=sources.subtitles_path,
        work_dir=sources.work_dir,
        suffix=suffix,
    )


def _load_transcript(path: Path) -> Transcript | None:
    """Carga un ``transcript.json`` persistido sin propagar fallos.

    Args:
        path: Ruta del ``transcript.json`` del directorio de trabajo.

    Returns:
        La transcripción validada, o ``None`` si falta o es ilegible.
    """
    try:
        return Transcript.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, UnicodeDecodeError):
        return None


def _ass_candidates(
    subtitles_path: Path | None, work_dir: Path | None, suffix: str
) -> tuple[Path, ...]:
    """Ordena los ``.ass`` candidatos para una pieza sin duplicados.

    Args:
        subtitles_path: ``.ass`` primario del pipeline, o ``None``.
        work_dir: Directorio de trabajo con ``.ass`` persistidos, o ``None``.
        suffix: Sufijo del lote (``""`` o ``"_NN"``).

    Returns:
        Los candidatos en orden de prioridad.
    """
    candidates: list[Path] = []
    if subtitles_path is not None:
        candidates.append(subtitles_path)
    if work_dir is not None:
        indexed = work_dir / f"subtitles{suffix}.ass"
        if indexed not in candidates:
            candidates.append(indexed)
    return tuple(candidates)
