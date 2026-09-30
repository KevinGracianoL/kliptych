"""Tests para recorte de ventana temporal .lrc, generación .ass y validador spelling_lock.

Valida:
1. Recorte de .lrc a ventana [start_sec, end_sec] y desplazamiento a 0.0s.
2. Fallo fail-closed cuando no hay líneas en la ventana (nunca lyric video mudo).
3. Generación de eventos Dialogue en .ass y SubtitleSegment para la pieza.
4. spelling_lock para Format.LYRIC_VIDEO:
   - Coincidente -> PASS
   - Palabra alterada -> FAIL
   - Sin subtítulos sincronizados verificados -> FAIL (nunca PASSED)
   - Normalización NFC + preservación de 'ñ'
"""

import math
from pathlib import Path

import pytest

from kliptych.assets import AssetRegistry
from kliptych.contract import Contract, Format, LyricConfig
from kliptych.gate import CheckStatus, Piece, SubtitleSegment
from kliptych.gate.checks import check_spelling_locks
from kliptych.gate.models import GateContext
from kliptych.lyrics import (
    LrcEmptyWindowError,
    LyricLine,
    cut_lyric_window,
    lyric_lines_to_ass,
    lyric_lines_to_subtitle_segments,
    lyric_lines_to_subtitle_text,
    parse_lrc,
)
from tests.support import make_contract, make_media, make_piece


def test_cut_lyric_window_shifts_to_zero() -> None:
    content = (
        "[00:05.00]Línea previa\n"
        "[00:10.00]Primera en ventana\n"
        "[00:15.00]Segunda en ventana\n"
        "[00:20.00]Tercera en ventana\n"
        "[00:30.00]Línea posterior\n"
    )
    lines = parse_lrc(content)
    window = cut_lyric_window(lines, start_sec=10.0, end_sec=25.0)

    assert len(window) == 3
    assert math.isclose(window[0].start_sec, 0.0)
    assert window[0].text == "Primera en ventana"
    assert window[0].end_sec is not None
    assert math.isclose(window[0].end_sec, 5.0)

    assert math.isclose(window[1].start_sec, 5.0)
    assert window[1].text == "Segunda en ventana"
    assert window[1].end_sec is not None
    assert math.isclose(window[1].end_sec, 10.0)

    assert math.isclose(window[2].start_sec, 10.0)
    assert window[2].text == "Tercera en ventana"
    assert window[2].end_sec is not None
    assert math.isclose(window[2].end_sec, 15.0)


def test_cut_lyric_window_fails_closed_when_empty() -> None:
    content = "[00:05.00]Línea muy temprana\n"
    lines = parse_lrc(content)
    with pytest.raises(LrcEmptyWindowError, match=r"no hay líneas"):
        _ = cut_lyric_window(lines, start_sec=20.0, end_sec=40.0)


def test_lyric_lines_to_ass_generates_dialogue_events() -> None:
    lines = (
        LyricLine(start_sec=0.0, end_sec=5.0, text="Hola mundo"),
        LyricLine(start_sec=5.0, end_sec=10.0, text="Segunda línea"),
    )
    ass_content = lyric_lines_to_ass(lines, duration_s=10.0)
    assert "[Script Info]" in ass_content
    assert "[V4+ Styles]" in ass_content
    assert "[Events]" in ass_content
    assert "Dialogue: 0,0:00:00.00,0:00:05.00,Default,,0,0,0,,Hola mundo" in ass_content
    assert "Dialogue: 0,0:00:05.00,0:00:10.00,Default,,0,0,0,,Segunda línea" in ass_content


def test_lyric_lines_to_subtitle_segments_and_text() -> None:
    lines = (
        LyricLine(start_sec=0.0, end_sec=4.0, text="Primera"),
        LyricLine(start_sec=4.0, end_sec=8.0, text="Segunda"),
    )
    segments = lyric_lines_to_subtitle_segments(lines)
    assert len(segments) == 2
    assert segments[0] == SubtitleSegment(text="Primera", start_s=0.0, end_s=4.0)
    assert segments[1] == SubtitleSegment(text="Segunda", start_s=4.0, end_s=8.0)

    text = lyric_lines_to_subtitle_text(lines)
    assert text == "Primera Segunda"


def _make_context(
    tmp_path: Path,
    contract: Contract,
    piece: Piece,
) -> GateContext:
    registry = AssetRegistry(tmp_path)
    platform = next(iter(contract.platforms.keys()))
    return GateContext(
        contract=contract,
        rules=contract.platforms[platform],
        piece=piece,
        artifact_sha256="f" * 64,
        media=make_media(),
        assets=registry,
    )


def test_gate_spelling_lock_lyric_video_matching_passes(tmp_path: Path) -> None:
    contract = make_contract(
        format_=Format.LYRIC_VIDEO,
        spelling_locks=["canción", "corazón"],
    )
    piece = make_piece(
        tmp_path / "clip.mp4",
        subtitle_text="esta es mi cancion del corazon",
        subtitle_segments=[SubtitleSegment(text="cancion", start_s=0.0, end_s=2.0)],
    )
    ctx = _make_context(tmp_path, contract, piece)
    outcome = check_spelling_locks(ctx)
    assert outcome.status is CheckStatus.PASS


def test_gate_spelling_lock_lyric_video_altered_word_fails(tmp_path: Path) -> None:
    contract = make_contract(
        format_=Format.LYRIC_VIDEO,
        spelling_locks=["canción", "corazón"],
    )
    piece = make_piece(
        tmp_path / "clip.mp4",
        subtitle_text="esta es mi cancion alterada sin la otra",
        subtitle_segments=[SubtitleSegment(text="cancion", start_s=0.0, end_s=2.0)],
    )
    ctx = _make_context(tmp_path, contract, piece)
    outcome = check_spelling_locks(ctx)
    assert outcome.status is CheckStatus.FAIL
    assert outcome.evidence["missing"] == ["corazón"]


def test_gate_spelling_lock_lyric_video_unverified_subtitles_fails(tmp_path: Path) -> None:
    contract = make_contract(
        format_=Format.LYRIC_VIDEO,
        spelling_locks=[],
    )
    piece = make_piece(
        tmp_path / "clip.mp4",
        subtitle_text=None,
        subtitle_segments=(),
    )
    ctx = _make_context(tmp_path, contract, piece)
    outcome = check_spelling_locks(ctx)
    assert outcome.status is CheckStatus.FAIL


def test_gate_spelling_lock_preserves_enie_normalization(tmp_path: Path) -> None:
    contract = make_contract(
        format_=Format.LYRIC_VIDEO,
        spelling_locks=["año"],
    )
    bad_piece = make_piece(
        tmp_path / "clip.mp4",
        subtitle_text="feliz ano nuevo",
        subtitle_segments=[SubtitleSegment(text="feliz ano", start_s=0.0, end_s=2.0)],
    )
    ctx_bad = _make_context(tmp_path, contract, bad_piece)
    assert check_spelling_locks(ctx_bad).status is CheckStatus.FAIL

    good_piece = make_piece(
        tmp_path / "clip.mp4",
        subtitle_text="feliz año nuevo",
        subtitle_segments=[SubtitleSegment(text="feliz año", start_s=0.0, end_s=2.0)],
    )
    ctx_good = _make_context(tmp_path, contract, good_piece)
    assert check_spelling_locks(ctx_good).status is CheckStatus.PASS


def test_gate_spelling_lock_ground_truth_lrc_asset_detects_altered_word(tmp_path: Path) -> None:
    lrc_file = tmp_path / "song.lrc"
    _ = lrc_file.write_text("[00:00.00]palabra exacta del tema\n", encoding="utf-8")

    registry = AssetRegistry(tmp_path)
    ref = registry.register(
        asset_id="song_lrc",
        kind="lyrics",
        uri="song.lrc",
        origin="test",
    )

    contract = make_contract(
        format_=Format.LYRIC_VIDEO,
        lyric_video=LyricConfig(lrc_asset_id="song_lrc"),
        required_assets=[ref],
    )

    bad_piece = make_piece(
        tmp_path / "clip.mp4",
        subtitle_text="palabra cambiada del tema",
        subtitle_segments=[SubtitleSegment(text="palabra cambiada", start_s=0.0, end_s=2.0)],
    )
    platform = next(iter(contract.platforms.keys()))
    ctx_bad = GateContext(
        contract=contract,
        rules=contract.platforms[platform],
        piece=bad_piece,
        artifact_sha256="f" * 64,
        media=make_media(),
        assets=registry,
    )
    outcome_bad = check_spelling_locks(ctx_bad)
    assert outcome_bad.status is CheckStatus.FAIL

    good_piece = make_piece(
        tmp_path / "clip.mp4",
        subtitle_text="palabra exacta del tema",
        subtitle_segments=[SubtitleSegment(text="palabra exacta", start_s=0.0, end_s=2.0)],
    )
    ctx_good = GateContext(
        contract=contract,
        rules=contract.platforms[platform],
        piece=good_piece,
        artifact_sha256="f" * 64,
        media=make_media(),
        assets=registry,
    )
    outcome_good = check_spelling_locks(ctx_good)
    assert outcome_good.status is CheckStatus.PASS
