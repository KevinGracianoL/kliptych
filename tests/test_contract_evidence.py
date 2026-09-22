"""Tests del modelo de evidencia textual por campo."""

import pytest
from pydantic import ValidationError

from kliptych.contract import Confidence, FieldCandidate, SourceEvidence


def _evidence(**overrides: object) -> SourceEvidence:
    data: dict[str, object] = {
        "quote": "El video debe durar como mínimo ocho segundos",
        "start": 421,
        "end": 472,
        "location": "brief.md#l12",
    }
    data.update(overrides)
    return SourceEvidence.model_validate(data)


def test_explicit_candidate_keeps_value_and_evidence() -> None:
    candidate = FieldCandidate[int](
        value=8,
        evidence=_evidence(),
        confidence=Confidence.EXPLICIT,
    )
    assert candidate.value == 8
    assert candidate.evidence is not None
    assert candidate.evidence.quote.startswith("El video")
    assert candidate.evidence.location == "brief.md#l12"


def test_explicit_candidate_without_evidence_is_rejected() -> None:
    with pytest.raises(ValidationError, match="evidencia"):
        _ = FieldCandidate[int](value=8, confidence=Confidence.EXPLICIT)


def test_explicit_candidate_without_value_is_rejected() -> None:
    with pytest.raises(ValidationError, match="value"):
        _ = FieldCandidate[int](evidence=_evidence(), confidence=Confidence.EXPLICIT)


def test_inferred_candidate_requires_evidence_too() -> None:
    with pytest.raises(ValidationError, match="evidencia"):
        _ = FieldCandidate[str](value="es", confidence=Confidence.INFERRED)


def test_inferred_candidate_with_evidence_is_valid() -> None:
    candidate = FieldCandidate[str](
        value="es",
        evidence=_evidence(quote="todo el material es en español"),
        confidence=Confidence.INFERRED,
    )
    assert candidate.value == "es"


def test_missing_candidate_is_valid_and_empty() -> None:
    candidate = FieldCandidate[int]()
    assert candidate.value is None
    assert candidate.evidence is None
    assert candidate.confidence is Confidence.MISSING
    assert candidate.status is Confidence.MISSING


def test_missing_candidate_cannot_carry_value() -> None:
    with pytest.raises(ValidationError, match="missing"):
        _ = FieldCandidate[int](value=8, confidence=Confidence.MISSING)


def test_conflict_candidate_requires_evidence() -> None:
    with pytest.raises(ValidationError, match="evidencia"):
        _ = FieldCandidate[int](confidence=Confidence.CONFLICT)


def test_conflict_candidate_stays_unresolved() -> None:
    candidate = FieldCandidate[int](
        evidence=_evidence(quote="mínimo 8 s; en otra sección pide 15 s"),
        confidence=Confidence.CONFLICT,
    )
    assert candidate.value is None
    assert candidate.status is Confidence.CONFLICT
    with pytest.raises(ValidationError, match="conflict"):
        _ = FieldCandidate[int](
            value=8,
            evidence=_evidence(),
            confidence=Confidence.CONFLICT,
        )


def test_evidence_span_must_be_ordered() -> None:
    with pytest.raises(ValidationError, match="end"):
        _ = _evidence(start=472, end=421)


def test_evidence_quote_cannot_be_empty() -> None:
    with pytest.raises(ValidationError):
        _ = _evidence(quote="")


def test_evidence_location_cannot_be_empty() -> None:
    with pytest.raises(ValidationError):
        _ = _evidence(location="")


def test_candidate_forbids_unknown_fields() -> None:
    with pytest.raises(ValidationError, match="Extra inputs"):
        _ = FieldCandidate[int].model_validate(
            {
                "value": 8,
                "evidence": _evidence(),
                "confidence": Confidence.EXPLICIT,
                "source": "inventado",
            }
        )
