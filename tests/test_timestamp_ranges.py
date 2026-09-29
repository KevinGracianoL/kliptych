"""Tests para marcas temporales mandatorias (Sprint 3 Objetivo 2).

Valida el modelo TimestampRange, su inclusión en Contract y ContractDraft,
la instrucción en el system prompt y la resolución determinista de formatos
de tiempo ("MM:SS", "HH:MM:SS", numéricos) tanto desde draft como desde brief.
"""

import math
from pathlib import Path
from typing import cast

import pytest
from pydantic import ValidationError

from kliptych.assets import AssetRegistry
from kliptych.contract import TimestampRange, contract_digest
from kliptych.contract.draft import TimestampRangeDraft
from kliptych.resolver import parse_timestamp_seconds, resolve_contract
from kliptych.runtime import openai_compatible
from tests.support import candidate, make_contract, make_draft


def _prompt_text(name: str) -> str:
    return cast("str", getattr(openai_compatible, name))


def test_timestamp_range_valid_bounds() -> None:
    tr = TimestampRange(start_sec=10.0, end_sec=25.5)
    assert math.isclose(tr.start_sec, 10.0)
    assert math.isclose(tr.end_sec, 25.5)


def test_timestamp_range_rejects_negative_start() -> None:
    with pytest.raises(ValidationError):
        _ = TimestampRange(start_sec=-1.0, end_sec=10.0)


def test_timestamp_range_rejects_inverted_or_equal_bounds() -> None:
    with pytest.raises(ValidationError):
        _ = TimestampRange(start_sec=10.0, end_sec=10.0)
    with pytest.raises(ValidationError):
        _ = TimestampRange(start_sec=20.0, end_sec=10.0)


def test_timestamp_range_rejects_nan_or_inf() -> None:
    with pytest.raises(ValidationError):
        _ = TimestampRange(start_sec=float("nan"), end_sec=10.0)
    with pytest.raises(ValidationError):
        _ = TimestampRange(start_sec=0.0, end_sec=float("inf"))


def test_contract_includes_timestamp_ranges() -> None:
    contract = make_contract(
        timestamp_ranges=((10.0, 20.0),),
    )
    assert len(contract.timestamp_ranges) == 1
    assert math.isclose(contract.timestamp_ranges[0].start_sec, 10.0)
    assert math.isclose(contract.timestamp_ranges[0].end_sec, 20.0)


def test_contract_digest_empty_ranges_does_not_mutate_base() -> None:
    base = make_contract()
    with_empty = make_contract(timestamp_ranges=())
    assert contract_digest(base) == contract_digest(with_empty)


def test_contract_digest_changes_when_timestamp_ranges_declared() -> None:
    base = make_contract()
    with_range = make_contract(timestamp_ranges=((5.0, 15.0),))
    assert contract_digest(base) != contract_digest(with_range)


def test_parse_timestamp_seconds_formats() -> None:
    assert math.isclose(cast("float", parse_timestamp_seconds(45.0)), 45.0)
    assert math.isclose(cast("float", parse_timestamp_seconds(45.5)), 45.5)
    assert math.isclose(cast("float", parse_timestamp_seconds("45")), 45.0)
    assert math.isclose(cast("float", parse_timestamp_seconds("45s")), 45.0)
    assert math.isclose(cast("float", parse_timestamp_seconds("01:30")), 90.0)
    assert math.isclose(cast("float", parse_timestamp_seconds("1:30")), 90.0)
    assert math.isclose(cast("float", parse_timestamp_seconds("01:15:30")), 4530.0)
    assert math.isclose(cast("float", parse_timestamp_seconds("00:01:30.500")), 90.5)


def test_extract_prompt_instructs_timestamp_ranges() -> None:
    prompt = _prompt_text("_EXTRACT_SYSTEM_PROMPT")
    assert "timestamp_ranges" in prompt
    assert "MM:SS" in prompt
    assert "HH:MM:SS" in prompt


def test_resolver_resolves_timestamp_ranges_from_draft(tmp_path: Path) -> None:
    draft = make_draft(
        timestamp_ranges=[
            TimestampRangeDraft(
                start_sec=candidate("01:30", "corte del 01:30 al 02:45"),
                end_sec=candidate("02:45", "corte del 01:30 al 02:45"),
            )
        ]
    )
    result = resolve_contract(draft, registry=AssetRegistry(tmp_path))
    assert result is not None
    assert result.contract is not None
    assert len(result.contract.timestamp_ranges) == 1
    assert math.isclose(result.contract.timestamp_ranges[0].start_sec, 90.0)
    assert math.isclose(result.contract.timestamp_ranges[0].end_sec, 165.0)


@pytest.mark.parametrize(
    ("brief", "expected_start", "expected_end"),
    [
        ("usar el fragmento 01:30 - 02:45 del directo", 90.0, 165.0),
        ("cortar del minuto 1:30 al 2:45 para el clip", 90.0, 165.0),
        ("segmento: 00:01:00 - 00:02:30", 60.0, 150.0),
        ("timestamps: 10s - 45s", 10.0, 45.0),
    ],
)
def test_resolver_resolves_timestamp_ranges_from_brief_text(
    tmp_path: Path, brief: str, expected_start: float, expected_end: float
) -> None:
    full_brief = f"cita del brief. {brief}"
    draft = make_draft()
    result = resolve_contract(draft, registry=AssetRegistry(tmp_path), brief_text=full_brief)
    assert result is not None
    assert result.contract is not None
    assert len(result.contract.timestamp_ranges) == 1
    assert math.isclose(result.contract.timestamp_ranges[0].start_sec, expected_start)
    assert math.isclose(result.contract.timestamp_ranges[0].end_sec, expected_end)
