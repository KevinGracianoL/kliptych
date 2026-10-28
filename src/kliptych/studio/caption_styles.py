"""Deterministic resolution of B2.5 caption styles to SubtitleStyle."""

from kliptych.contract.enums import CaptionCapitalization
from kliptych.studio.project_models import (
    CaptionHorizontalAlignment,
    CaptionOutline,
    CaptionShadow,
    CaptionStyle,
    CaptionStylePreset,
    CaptionStyleSize,
    CaptionVerticalPosition,
)
from kliptych.subtitles import SubtitleStyle

NOTO_SANS_FONTNAME = "Noto Sans"
_PRIMARY_COLOUR = "&H00FFFFFF"
_OUTLINE_COLOUR = "&H00000000"

_PRESET_BASE_CONFIG: dict[CaptionStylePreset, tuple[int, str, int]] = {
    # preset -> (base_fontsize, back_colour, border_style)
    CaptionStylePreset.CLEAN: (54, "&H00000000", 1),
    CaptionStylePreset.BOLD: (62, "&H80000000", 1),
    CaptionStylePreset.BOXED: (54, "&H99000000", 3),
}

_OUTLINE_WIDTH: dict[CaptionOutline, int] = {
    CaptionOutline.NONE: 0,
    CaptionOutline.THIN: 2,
    CaptionOutline.THICK: 4,
}

_SHADOW_DEPTH: dict[CaptionShadow, int] = {
    CaptionShadow.NONE: 0,
    CaptionShadow.SOFT: 1,
    CaptionShadow.STRONG: 3,
}

_ALIGNMENT_MATRIX: dict[
    tuple[CaptionVerticalPosition, CaptionHorizontalAlignment],
    tuple[int, int],
] = {
    # (vertical, horizontal) -> (alignment, margin_v)
    (CaptionVerticalPosition.TOP, CaptionHorizontalAlignment.LEFT): (7, 60),
    (CaptionVerticalPosition.TOP, CaptionHorizontalAlignment.CENTER): (8, 60),
    (CaptionVerticalPosition.TOP, CaptionHorizontalAlignment.RIGHT): (9, 60),
    (CaptionVerticalPosition.MIDDLE, CaptionHorizontalAlignment.LEFT): (4, 0),
    (CaptionVerticalPosition.MIDDLE, CaptionHorizontalAlignment.CENTER): (5, 0),
    (CaptionVerticalPosition.MIDDLE, CaptionHorizontalAlignment.RIGHT): (6, 0),
    (CaptionVerticalPosition.BOTTOM, CaptionHorizontalAlignment.LEFT): (1, 80),
    (CaptionVerticalPosition.BOTTOM, CaptionHorizontalAlignment.CENTER): (2, 80),
    (CaptionVerticalPosition.BOTTOM, CaptionHorizontalAlignment.RIGHT): (3, 80),
}


def _resolve_font_size(preset: CaptionStylePreset, size: CaptionStyleSize) -> int:
    match size:
        case CaptionStyleSize.SMALL:
            return 46
        case CaptionStyleSize.MEDIUM:
            return _PRESET_BASE_CONFIG[preset][0]
        case CaptionStyleSize.LARGE:
            return 70


def resolve_studio_caption_style(
    style: CaptionStyle,
    *,
    highlight_color: str = "&H000000FF",
    capitalization: CaptionCapitalization = CaptionCapitalization.ORIGINAL,
) -> SubtitleStyle:
    """Resolve a validated B2.5 CaptionStyle into an immutable SubtitleStyle.

    Args:
        style: The validated complete B2.5 caption style.
        highlight_color: Secondary sweep colour (&HAABBGGRR).
        capitalization: Text capitalization mode.

    Returns:
        A fully configured SubtitleStyle instance.
    """
    _, back_colour, border_style = _PRESET_BASE_CONFIG[style.preset]
    fontsize = _resolve_font_size(style.preset, style.size)
    outline = _OUTLINE_WIDTH[style.outline]
    shadow = _SHADOW_DEPTH[style.shadow]
    alignment, margin_v = _ALIGNMENT_MATRIX[style.vertical_position, style.horizontal_alignment]

    return SubtitleStyle(
        fontname=NOTO_SANS_FONTNAME,
        fontsize=fontsize,
        primary_colour=_PRIMARY_COLOUR,
        secondary_colour=highlight_color,
        outline_colour=_OUTLINE_COLOUR,
        back_colour=back_colour,
        border_style=border_style,
        outline=outline,
        shadow=shadow,
        margin_v=margin_v,
        alignment=alignment,
        capitalization=capitalization,
    )
