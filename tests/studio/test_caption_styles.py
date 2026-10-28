"""Tests for B2.5 caption style resolution to SubtitleStyle."""

import pytest

from kliptych.contract.enums import CaptionCapitalization
from kliptych.studio.caption_styles import (
    NOTO_SANS_FONTNAME,
    resolve_studio_caption_style,
)
from kliptych.studio.project_models import (
    CaptionHorizontalAlignment,
    CaptionOutline,
    CaptionShadow,
    CaptionStyle,
    CaptionStylePreset,
    CaptionStyleSize,
    CaptionVerticalPosition,
)


def _base_style(preset: CaptionStylePreset) -> CaptionStyle:
    return CaptionStyle(
        preset=preset,
        size=CaptionStyleSize.MEDIUM,
        vertical_position=CaptionVerticalPosition.BOTTOM,
        horizontal_alignment=CaptionHorizontalAlignment.CENTER,
        outline=CaptionOutline.THIN,
        shadow=CaptionShadow.NONE,
    )


def test_resolve_clean_preset_defaults() -> None:
    style = resolve_studio_caption_style(
        _base_style(CaptionStylePreset.CLEAN),
        highlight_color="&H00FF0000",
        capitalization=CaptionCapitalization.UPPERCASE,
    )
    assert style.fontname == NOTO_SANS_FONTNAME
    assert style.fontsize == 54
    assert style.primary_colour == "&H00FFFFFF"
    assert style.secondary_colour == "&H00FF0000"
    assert style.outline_colour == "&H00000000"
    assert style.back_colour == "&H00000000"
    assert style.border_style == 1
    assert style.outline == 2
    assert style.shadow == 0
    assert style.alignment == 2
    assert style.margin_v == 80
    assert style.capitalization is CaptionCapitalization.UPPERCASE


def test_resolve_bold_preset_defaults() -> None:
    style = resolve_studio_caption_style(_base_style(CaptionStylePreset.BOLD))
    assert style.fontname == NOTO_SANS_FONTNAME
    assert style.fontsize == 62
    assert style.border_style == 1
    assert style.back_colour == "&H80000000"
    assert style.outline == 2  # outline="thin" override on base style
    assert style.shadow == 0  # shadow="none" override on base style


def test_resolve_boxed_preset_defaults() -> None:
    style = resolve_studio_caption_style(
        CaptionStyle(
            preset=CaptionStylePreset.BOXED,
            size=CaptionStyleSize.MEDIUM,
            vertical_position=CaptionVerticalPosition.BOTTOM,
            horizontal_alignment=CaptionHorizontalAlignment.CENTER,
            outline=CaptionOutline.THICK,
            shadow=CaptionShadow.NONE,
        )
    )
    assert style.fontname == NOTO_SANS_FONTNAME
    assert style.fontsize == 54
    assert style.border_style == 3
    assert style.back_colour == "&H99000000"
    assert style.outline == 4
    assert style.shadow == 0


@pytest.mark.parametrize(
    ("size_enum", "expected_font_size"),
    [
        (CaptionStyleSize.SMALL, 46),
        (CaptionStyleSize.MEDIUM, 54),
        (CaptionStyleSize.LARGE, 70),
    ],
)
def test_size_overrides(size_enum: CaptionStyleSize, expected_font_size: int) -> None:
    style = resolve_studio_caption_style(
        CaptionStyle(
            preset=CaptionStylePreset.CLEAN,
            size=size_enum,
            vertical_position=CaptionVerticalPosition.BOTTOM,
            horizontal_alignment=CaptionHorizontalAlignment.CENTER,
            outline=CaptionOutline.THIN,
            shadow=CaptionShadow.NONE,
        )
    )
    assert style.fontsize == expected_font_size


@pytest.mark.parametrize(
    ("outline_enum", "expected_outline"),
    [
        (CaptionOutline.NONE, 0),
        (CaptionOutline.THIN, 2),
        (CaptionOutline.THICK, 4),
    ],
)
def test_outline_overrides(outline_enum: CaptionOutline, expected_outline: int) -> None:
    style = resolve_studio_caption_style(
        CaptionStyle(
            preset=CaptionStylePreset.CLEAN,
            size=CaptionStyleSize.MEDIUM,
            vertical_position=CaptionVerticalPosition.BOTTOM,
            horizontal_alignment=CaptionHorizontalAlignment.CENTER,
            outline=outline_enum,
            shadow=CaptionShadow.NONE,
        )
    )
    assert style.outline == expected_outline


@pytest.mark.parametrize(
    ("shadow_enum", "expected_shadow"),
    [
        (CaptionShadow.NONE, 0),
        (CaptionShadow.SOFT, 1),
        (CaptionShadow.STRONG, 3),
    ],
)
def test_shadow_overrides(shadow_enum: CaptionShadow, expected_shadow: int) -> None:
    style = resolve_studio_caption_style(
        CaptionStyle(
            preset=CaptionStylePreset.CLEAN,
            size=CaptionStyleSize.MEDIUM,
            vertical_position=CaptionVerticalPosition.BOTTOM,
            horizontal_alignment=CaptionHorizontalAlignment.CENTER,
            outline=CaptionOutline.THIN,
            shadow=shadow_enum,
        )
    )
    assert style.shadow == expected_shadow


@pytest.mark.parametrize(
    ("vert", "horiz", "expected_align", "expected_margin_v"),
    [
        (CaptionVerticalPosition.TOP, CaptionHorizontalAlignment.LEFT, 7, 60),
        (CaptionVerticalPosition.TOP, CaptionHorizontalAlignment.CENTER, 8, 60),
        (CaptionVerticalPosition.TOP, CaptionHorizontalAlignment.RIGHT, 9, 60),
        (CaptionVerticalPosition.MIDDLE, CaptionHorizontalAlignment.LEFT, 4, 0),
        (CaptionVerticalPosition.MIDDLE, CaptionHorizontalAlignment.CENTER, 5, 0),
        (CaptionVerticalPosition.MIDDLE, CaptionHorizontalAlignment.RIGHT, 6, 0),
        (CaptionVerticalPosition.BOTTOM, CaptionHorizontalAlignment.LEFT, 1, 80),
        (CaptionVerticalPosition.BOTTOM, CaptionHorizontalAlignment.CENTER, 2, 80),
        (CaptionVerticalPosition.BOTTOM, CaptionHorizontalAlignment.RIGHT, 3, 80),
    ],
)
def test_nine_cell_alignment_and_margin_mapping(
    vert: CaptionVerticalPosition,
    horiz: CaptionHorizontalAlignment,
    expected_align: int,
    expected_margin_v: int,
) -> None:
    style = resolve_studio_caption_style(
        CaptionStyle(
            preset=CaptionStylePreset.CLEAN,
            size=CaptionStyleSize.MEDIUM,
            vertical_position=vert,
            horizontal_alignment=horiz,
            outline=CaptionOutline.THIN,
            shadow=CaptionShadow.NONE,
        )
    )
    assert style.alignment == expected_align
    assert style.margin_v == expected_margin_v
