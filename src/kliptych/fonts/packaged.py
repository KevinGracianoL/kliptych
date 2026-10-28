"""Packaged font assets for deterministic caption rendering."""

import hashlib
from importlib import resources
from pathlib import Path

NOTO_SANS_REGULAR_FILENAME = "NotoSans-Regular.ttf"
NOTO_SANS_REGULAR_SHA256 = "b85c38ecea8a7cfb39c24e395a4007474fa5a4fc864f6ee33309eb4948d232d5"


class FontError(Exception):
    """The packaged font is missing, modified, or invalid."""


def packaged_fonts_dir() -> Path:
    """Return and validate the directory containing the packaged font.

    Returns:
        The validated Path to the packaged fonts directory.
    """
    directory = Path(str(resources.files("kliptych").joinpath("fonts")))
    validate_packaged_font_dir(directory)
    return directory


def packaged_font_path() -> Path:
    """Return and validate the path to NotoSans-Regular.ttf.

    Returns:
        The validated Path to the packaged font file.
    """
    path = packaged_fonts_dir() / NOTO_SANS_REGULAR_FILENAME
    validate_packaged_font(path)
    return path


def validate_packaged_font(path: Path | str) -> None:
    """Validate that path is the unmodified packaged font.

    Args:
        path: Path to the font file.

    Raises:
        FontError: If the file is missing, a symlink, or fails SHA-256 validation.
    """
    candidate = Path(path)
    if candidate.is_symlink():
        msg = f"packaged font is a symlink: {candidate}"
        raise FontError(msg)
    if not candidate.is_file():
        msg = f"packaged font is missing or not a regular file: {candidate}"
        raise FontError(msg)

    hasher = hashlib.sha256()
    with candidate.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            hasher.update(chunk)
    digest = hasher.hexdigest()

    if digest != NOTO_SANS_REGULAR_SHA256:
        msg = f"packaged font hash mismatch: expected {NOTO_SANS_REGULAR_SHA256}, got {digest}"
        raise FontError(msg)


def validate_packaged_font_dir(directory: Path | str) -> None:
    """Validate that directory is a regular directory containing the valid font.

    Args:
        directory: Path to the fonts directory.

    Raises:
        FontError: If the directory or the contained font is invalid.
    """
    candidate = Path(directory)
    if candidate.is_symlink() or not candidate.is_dir():
        msg = f"packaged font directory is missing or invalid: {candidate}"
        raise FontError(msg)
    validate_packaged_font(candidate / NOTO_SANS_REGULAR_FILENAME)
