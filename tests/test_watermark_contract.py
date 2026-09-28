"""Sprint 2 (Objetivo 2): contrato de watermark con posición y tamaño.

El `Watermark` del contrato incorpora la posición (`WatermarkPosition`),
el tamaño relativo (`scale_ratio`), la opacidad y el ancho mínimo
(`min_width_ratio`) que exigen el render dinámico y el validador OpenCV.
El extractor debe instruir su extracción con evidencia y el resolutor debe
propagar los campos del draft sin inventarlos.
"""

from pathlib import Path
from typing import cast

import pytest
from pydantic import ValidationError

from kliptych.assets import AssetRegistry
from kliptych.contract import Watermark, WatermarkPosition, contract_digest
from kliptych.resolver import ResolutionStatus, resolve_contract
from kliptych.runtime import openai_compatible
from tests.support import candidate, make_contract, make_draft


def _prompt_text(name: str) -> str:
    return cast("str", getattr(openai_compatible, name))


def test_watermark_position_values() -> None:
    assert {position.value for position in WatermarkPosition} == {
        "center_bottom",
        "center_top",
        "center",
        "top_right",
        "top_left",
        "bottom_right",
        "bottom_left",
    }


def test_watermark_defaults_match_legacy_top_right_render() -> None:
    watermark = Watermark(required=True, asset_id="wm-marca", visible_full_video=True)
    assert watermark.position is WatermarkPosition.TOP_RIGHT
    assert watermark.scale_ratio == pytest.approx(0.20)
    assert watermark.opacity == pytest.approx(1.0)


def test_watermark_rejects_out_of_range_ratios() -> None:
    with pytest.raises(ValidationError, match="scale_ratio"):
        _ = Watermark(
            required=True,
            asset_id="wm-marca",
            visible_full_video=True,
            scale_ratio=1.5,
        )
    with pytest.raises(ValidationError, match="opacity"):
        _ = Watermark(
            required=True,
            asset_id="wm-marca",
            visible_full_video=True,
            opacity=-0.1,
        )
    with pytest.raises(ValidationError, match="min_width_ratio"):
        _ = Watermark(
            required=True,
            asset_id="wm-marca",
            visible_full_video=True,
            min_width_ratio=2.0,
        )


def test_watermark_rejects_scale_below_minimum_width() -> None:
    with pytest.raises(ValidationError, match="min_width_ratio"):
        _ = Watermark(
            required=True,
            asset_id="wm-marca",
            visible_full_video=True,
            scale_ratio=0.05,
            min_width_ratio=0.10,
        )


@pytest.mark.parametrize("opacity", [0.0, 0.05, 0.149])
def test_watermark_rejects_near_invisible_opacity(opacity: float) -> None:
    with pytest.raises(ValidationError, match="opacity"):
        _ = Watermark(
            required=True,
            asset_id="wm-marca",
            visible_full_video=True,
            opacity=opacity,
        )


def test_watermark_accepts_opacity_floor() -> None:
    watermark = Watermark(
        required=True,
        asset_id="wm-marca",
        visible_full_video=True,
        opacity=0.15,
    )
    assert watermark.opacity == pytest.approx(0.15)


def test_default_watermark_config_keeps_digest() -> None:
    assert contract_digest(make_contract()) == contract_digest(
        make_contract(watermark_position="top_right")
    )


def test_positioned_watermark_changes_digest() -> None:
    assert contract_digest(make_contract()) != contract_digest(
        make_contract(watermark_position="center_bottom")
    )


def test_extract_prompt_instructs_watermark_with_evidence() -> None:
    prompt = _prompt_text("_EXTRACT_SYSTEM_PROMPT")
    assert "watermark" in prompt
    for position in ("center_bottom", "center_top", "top_right", "bottom_left"):
        assert position in prompt
    assert "scale_ratio" in prompt
    assert "evidence" in prompt


def test_resolver_propagates_watermark_position_and_size(tmp_path: Path) -> None:
    draft = make_draft(
        watermark={
            "required": candidate(value=True),
            "asset_id": candidate("wm-marca"),
            "visible_full_video": candidate(value=True),
            "position": candidate("center_bottom"),
            "scale_ratio": candidate(0.25),
            "opacity": candidate(0.9),
            "min_width_ratio": candidate(0.10),
        },
    )
    result = resolve_contract(draft, registry=AssetRegistry(tmp_path))
    assert result.status is ResolutionStatus.RESOLVED
    assert result.contract is not None
    assert result.contract.watermark.position is WatermarkPosition.CENTER_BOTTOM
    assert result.contract.watermark.scale_ratio == pytest.approx(0.25)
    assert result.contract.watermark.opacity == pytest.approx(0.9)
    assert result.contract.watermark.min_width_ratio == pytest.approx(0.10)
    assert "watermark.full_video" in result.contract.rules.hard


def test_resolver_defaults_undeclared_watermark_fields(tmp_path: Path) -> None:
    draft = make_draft(
        watermark={
            "required": candidate(value=True),
            "asset_id": candidate("wm-marca"),
            "visible_full_video": candidate(value=False),
        },
    )
    result = resolve_contract(draft, registry=AssetRegistry(tmp_path))
    assert result.status is ResolutionStatus.RESOLVED
    assert result.contract is not None
    assert result.contract.watermark.position is WatermarkPosition.TOP_RIGHT
    assert "watermark.present" in result.contract.rules.hard
