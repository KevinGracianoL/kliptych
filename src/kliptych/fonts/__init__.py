"""Packaged font assets for deterministic caption rendering."""

from kliptych.fonts.packaged import (
    NOTO_SANS_REGULAR_FILENAME,
    NOTO_SANS_REGULAR_SHA256,
    FontError,
    packaged_font_path,
    packaged_fonts_dir,
    validate_packaged_font,
    validate_packaged_font_dir,
)

__all__ = [
    "NOTO_SANS_REGULAR_FILENAME",
    "NOTO_SANS_REGULAR_SHA256",
    "FontError",
    "packaged_font_path",
    "packaged_fonts_dir",
    "validate_packaged_font",
    "validate_packaged_font_dir",
]
