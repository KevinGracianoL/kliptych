"""Tests for packaged font assets and verification."""

from pathlib import Path

import pytest

from kliptych.fonts import (
    FontError,
    packaged_font_path,
    packaged_fonts_dir,
    validate_packaged_font,
    validate_packaged_font_dir,
)


def test_packaged_fonts_dir_exists_and_contains_expected_files() -> None:
    directory = packaged_fonts_dir()
    assert directory.is_dir()
    assert (directory / "NotoSans-Regular.ttf").is_file()
    assert (directory / "OFL.txt").is_file()
    assert (directory / "PROVENANCE.txt").is_file()


def test_packaged_font_path_is_regular_and_valid() -> None:
    font_path = packaged_font_path()
    assert font_path.is_file()
    assert not font_path.is_symlink()
    validate_packaged_font(font_path)


def test_validate_packaged_font_rejects_missing_file(tmp_path: Path) -> None:
    missing = tmp_path / "NonExistent.ttf"
    with pytest.raises(FontError, match="missing"):
        validate_packaged_font(missing)


def test_validate_packaged_font_rejects_symlink(tmp_path: Path) -> None:
    target = packaged_font_path()
    link = tmp_path / "link.ttf"
    link.symlink_to(target)
    with pytest.raises(FontError, match="symlink"):
        validate_packaged_font(link)


def test_validate_packaged_font_rejects_corrupted_hash(tmp_path: Path) -> None:
    corrupted = tmp_path / "NotoSans-Regular.ttf"
    _ = corrupted.write_bytes(b"tampered content")
    with pytest.raises(FontError, match="hash mismatch"):
        validate_packaged_font(corrupted)


def test_validate_packaged_font_dir_rejects_invalid_dir(tmp_path: Path) -> None:
    invalid_dir = tmp_path / "fonts"
    invalid_dir.mkdir()
    # Missing font inside directory
    with pytest.raises(FontError):
        validate_packaged_font_dir(invalid_dir)
