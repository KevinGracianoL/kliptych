"""Selección de segmentos asistida por LLM a partir del transcript y los momentos.

Construye el payload del prompt de ``select_segments`` (brief §9) y valida la
respuesta contra las restricciones del contrato v1.1 y las cotas del vídeo
fuente. El alcance termina en ``SegmentSelection``: no corta vídeo, no reframea
ni renderiza subtítulos.

``parse_response`` usa las cotas derivadas por el ``build_prompt`` previo porque
el contrato y la duración del vídeo solo se conocen al construir el prompt; la
validación es explícita y falla si no hubo ``build_prompt``.
"""

import contextlib
from collections.abc import Mapping
from dataclasses import dataclass
from typing import ClassVar, Protocol, cast

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from kliptych.contract import Contract, Segment, TimestampRange, prompt_languages
from kliptych.moments import Moment
from kliptych.transcribe import Transcript

SEGMENT_SYSTEM_PROMPT = (
    "Eres el selector de segmentos de Kliptych. Recibes un JSON con el "
    "transcript con marcas de tiempo, los momentos candidatos y las reglas del "
    "contrato, y devuelves ÚNICAMENTE un JSON con la forma "
    '{"segments": [{"start_s": float, "end_s": float}], "rationale": str}. '
    "Reglas: elige los mejores cortes dentro de las cotas del vídeo fuente; "
    "si el contrato especifica mandatory_timestamp_ranges, respétalos exactamente; "
    "respeta la duración mínima y máxima; los segmentos deben empezar antes de "
    "terminar y no inventes datos fuera del transcript."
)


class SegmentSelectionError(Exception):
    """La selección de segmentos no se pudo construir o validar."""


class SegmentSelection(BaseModel):
    """Segmentos elegidos por el modelo, con su justificación."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", frozen=True)

    segments: tuple[Segment, ...]
    rationale: str = Field(min_length=1)


@dataclass(frozen=True, slots=True)
class _SelectionBounds:
    duration_s: float
    min_s: float
    max_s: float | None


class SegmentSelector(Protocol):
    """Interfaz del selector de segmentos que consume el pipeline."""

    def build_prompt(
        self,
        transcript: Transcript,
        moments: tuple[Moment, ...],
        contract: Contract,
    ) -> dict[str, object]:
        """Construye el payload serializable del prompt de selección.

        Args:
            transcript: Transcripción con palabras y duración del vídeo fuente.
            moments: Momentos candidatos detectados, ordenados por puntuación.
            contract: Contrato validado con las reglas de duración e idioma.

        Returns:
            El payload del prompt de selección de segmentos.

        Raises:
            SegmentSelectionError: Si los rangos de duración de las plataformas
                son incompatibles entre sí.
        """
        ...

    def parse_response(self, raw: object) -> SegmentSelection:
        """Valida la respuesta del modelo contra las cotas del vídeo fuente.

        Args:
            raw: Respuesta del modelo: objeto, JSON en texto o bytes.

        Returns:
            La selección validada estructuralmente y dentro de las cotas.

        Raises:
            SegmentSelectionError: Si no hubo ``build_prompt`` previo, la
                respuesta no es una ``SegmentSelection`` válida o algún
                segmento queda fuera del vídeo o de las duraciones permitidas.
        """
        ...


class LLMSegmentSelector:
    """Construye el prompt y valida la selección de segmentos del modelo."""

    def __init__(self) -> None:
        """Inicializa el selector sin cotas hasta el primer ``build_prompt``."""
        self._bounds: _SelectionBounds | None = None
        self._mandatory_ranges: tuple[TimestampRange, ...] = ()

    def build_prompt(
        self,
        transcript: Transcript,
        moments: tuple[Moment, ...],
        contract: Contract,
    ) -> dict[str, object]:
        """Construye el payload serializable del prompt de selección.

        Args:
            transcript: Transcripción con palabras y duración del vídeo fuente.
            moments: Momentos candidatos detectados, ordenados por puntuación.
            contract: Contrato validado con las reglas de duración e idioma.

        Returns:
            El payload del prompt de selección de segmentos.

        Raises:
            SegmentSelectionError: Si los rangos de duración de las plataformas
                son incompatibles entre sí.
        """
        bounds = _bounds_from(contract, transcript)
        self._bounds = bounds
        self._mandatory_ranges = contract.timestamp_ranges
        return {
            "campaign_id": contract.campaign_id,
            "contract": {
                "mode": contract.mode.value,
                "format": contract.format.value,
                "languages": prompt_languages(contract),
                "duration": {
                    "source_s": bounds.duration_s,
                    "min_s": bounds.min_s,
                    "max_s": bounds.max_s,
                },
                "prohibitions": list(contract.prohibitions),
                "spelling_locks": list(contract.spelling_locks),
                "declared_segments": [
                    segment.model_dump(mode="json") for segment in contract.segments
                ],
                "mandatory_timestamp_ranges": [
                    {"start_sec": tr.start_sec, "end_sec": tr.end_sec}
                    for tr in contract.timestamp_ranges
                ],
            },
            "transcript": {
                "language": transcript.language,
                "duration_s": transcript.duration_s,
                "text": transcript.text,
                "words": [
                    {"start_s": word.start_s, "end_s": word.end_s, "text": word.text}
                    for word in transcript.words
                ],
            },
            "moments": [
                {
                    "start_s": moment.start_s,
                    "end_s": moment.end_s,
                    "score": moment.score,
                    "source": moment.source.value,
                }
                for moment in moments
            ],
        }

    def parse_response(self, raw: object) -> SegmentSelection:
        """Valida la respuesta del modelo contra las cotas del vídeo fuente.

        Args:
            raw: Respuesta del modelo: objeto, JSON en texto o bytes.

        Returns:
            La selección validada estructuralmente y dentro de las cotas.

        Raises:
            SegmentSelectionError: Si no hubo ``build_prompt`` previo, la
                respuesta no es una ``SegmentSelection`` válida o algún
                segmento queda fuera del vídeo o de las duraciones permitidas.
        """
        bounds = self._bounds
        if bounds is None:
            msg = "parse_response exige un build_prompt previo para conocer las cotas"
            raise SegmentSelectionError(msg)
        if self._mandatory_ranges:
            forced_segments = tuple(
                Segment(start_s=tr.start_sec, end_s=tr.end_sec) for tr in self._mandatory_ranges
            )
            rationale = "mandatory timestamp ranges from contract"
            with contextlib.suppress(Exception):
                parsed = _parse_selection(raw)
                rationale = parsed.rationale
            selection = SegmentSelection(segments=forced_segments, rationale=rationale)
            _validate_bounds(selection, bounds)
            return selection
        selection = _parse_selection(raw)
        _validate_bounds(selection, bounds)
        return selection


def _bounds_from(contract: Contract, transcript: Transcript) -> _SelectionBounds:
    minimums = [
        rules.duration.min_s
        for rules in contract.platforms.values()
        if rules.duration.min_s is not None
    ]
    maximums = [
        rules.duration.max_s
        for rules in contract.platforms.values()
        if rules.duration.max_s is not None
    ]
    min_s = float(max(minimums, default=0))
    max_s = float(min(maximums)) if maximums else None
    if max_s is not None and min_s > max_s:
        msg = f"rangos de duración incompatibles entre plataformas: min {min_s} > max {max_s}"
        raise SegmentSelectionError(msg)
    return _SelectionBounds(duration_s=transcript.duration_s, min_s=min_s, max_s=max_s)


def _parse_selection(raw: object) -> SegmentSelection:
    if isinstance(raw, bytes | str):
        return _selection_from_json(raw)
    if isinstance(raw, Mapping):
        return _selection_from_object(cast("object", raw))
    msg = "la respuesta del modelo no es un objeto JSON"
    raise SegmentSelectionError(msg)


def _selection_from_json(raw: bytes | str) -> SegmentSelection:
    try:
        return SegmentSelection.model_validate_json(raw)
    except ValidationError:
        msg = "la respuesta del modelo no es una SegmentSelection válida"
        raise SegmentSelectionError(msg) from None


def _selection_from_object(raw: object) -> SegmentSelection:
    try:
        return SegmentSelection.model_validate(raw)
    except ValidationError:
        msg = "la respuesta del modelo no es una SegmentSelection válida"
        raise SegmentSelectionError(msg) from None


def _validate_bounds(selection: SegmentSelection, bounds: _SelectionBounds) -> None:
    if not selection.segments:
        msg = "la selección no contiene segmentos"
        raise SegmentSelectionError(msg)
    for segment in selection.segments:
        length_s = segment.end_s - segment.start_s
        if bounds.duration_s > 0.0 and segment.end_s > bounds.duration_s:
            msg = (
                f"el segmento [{segment.start_s}, {segment.end_s}] excede el vídeo "
                f"fuente de {bounds.duration_s} s"
            )
            raise SegmentSelectionError(msg)
        if length_s < bounds.min_s:
            msg = f"el segmento dura {length_s} s y el mínimo es {bounds.min_s} s"
            raise SegmentSelectionError(msg)
        if bounds.max_s is not None and length_s > bounds.max_s:
            msg = f"el segmento dura {length_s} s y el máximo es {bounds.max_s} s"
            raise SegmentSelectionError(msg)
