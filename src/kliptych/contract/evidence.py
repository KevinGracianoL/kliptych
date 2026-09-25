"""Evidencia textual obligatoria por campo extraído del brief."""

from enum import StrEnum
from typing import Self

from pydantic import AliasChoices, Field, model_validator

from kliptych.contract.base import ContractBase


class Confidence(StrEnum):
    """Nivel de respaldo textual de un campo extraído."""

    EXPLICIT = "explicit"
    INFERRED = "inferred"
    MISSING = "missing"
    CONFLICT = "conflict"


class SourceEvidence(ContractBase):
    """Cita textual del brief que respalda un valor extraído."""

    quote: str = Field(min_length=1)
    start: int = Field(ge=0, validation_alias=AliasChoices("start", "start_char"))
    end: int = Field(ge=0, validation_alias=AliasChoices("end", "end_char"))
    location: str = Field(min_length=1)

    @property
    def start_char(self) -> int:
        """Offset inicial de la cita textual en el brief."""
        return self.start

    @property
    def end_char(self) -> int:
        """Offset final de la cita textual en el brief."""
        return self.end

    @model_validator(mode="after")
    def _span_is_ordered(self) -> Self:
        if self.end < self.start:
            msg = f"end ({self.end}) no puede ser menor que start ({self.start})"
            raise ValueError(msg)
        return self


class FieldCandidate[T](ContractBase):
    """Valor extraído con su evidencia; nunca inventa valores sin cita.

    Invariantes:

    - ``explicit`` e ``inferred`` exigen ``value`` y ``evidence``.
    - ``conflict`` exige ``evidence`` y queda sin resolver (``value`` nulo).
    - ``missing`` no admite ni ``value`` ni ``evidence``.
    """

    value: T | None = None
    evidence: SourceEvidence | None = None
    confidence: Confidence = Confidence.MISSING

    @model_validator(mode="after")
    def _evidence_matches_confidence(self) -> Self:
        if self.confidence is Confidence.MISSING:
            if self.value is not None or self.evidence is not None:
                msg = "un campo 'missing' no puede llevar value ni evidence"
                raise ValueError(msg)
            return self
        if self.evidence is None:
            msg = f"un campo '{self.confidence}' exige evidencia textual"
            raise ValueError(msg)
        if self.confidence is Confidence.CONFLICT:
            if self.value is not None:
                msg = "un campo 'conflict' queda sin resolver: no admite value"
                raise ValueError(msg)
            return self
        if self.value is None:
            msg = f"un campo '{self.confidence}' exige un value"
            raise ValueError(msg)
        return self

    @property
    def status(self) -> Confidence:
        """Estado de resolución del campo.

        Returns:
            El mismo nivel de confianza, expuesto como estado del campo.
        """
        return self.confidence
