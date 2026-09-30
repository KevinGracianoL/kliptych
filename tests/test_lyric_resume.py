"""Tests para fingerprint de checkpoint y resume con Format.LYRIC_VIDEO (Objetivo 4).

Valida:
1. Inclusión en el fingerprint de long_video:
   - Formato LYRIC_VIDEO
   - Configuración lyric_video
   - SHA-256 / firma del archivo .lrc
2. Invalidación ante cambios de contenido de .lrc
3. Hidratación de subtitle_segments y subtitle_text desde .ass en --resume.
"""

from pathlib import Path

from kliptych.contract import Format, LyricConfig
from kliptych.encoding import RenderConfig
from kliptych.gate.models import SubtitleSegment
from kliptych.orchestrator import (
    PipelineConfig,
    compute_long_video_fingerprint,
)
from kliptych.subtitle_text import (
    hydrate_piece_subtitle_segments,
    hydrate_piece_subtitle_text,
)
from tests.support import make_contract

_URL = "https://example.com/stream.mp4"


def _make_config(
    tmp_path: Path,
    lrc_name: str,
    lrc_content: str,
    *,
    format_: Format = Format.LYRIC_VIDEO,
    track_name: str = "Mi Cancion",
    artist_name: str = "Mi Artista",
) -> PipelineConfig:
    lrc_file = tmp_path / lrc_name
    _ = lrc_file.write_text(lrc_content, encoding="utf-8")
    contract = make_contract(
        format_=format_,
        lyric_video=LyricConfig(track_name=track_name, artist_name=artist_name),
    )
    return PipelineConfig(
        output_dir=tmp_path / "out",
        contract=contract,
        render=RenderConfig(),
        lrc_path=lrc_file,
    )


def test_same_lyric_inputs_keep_fingerprint(tmp_path: Path) -> None:
    first = _make_config(tmp_path, "song1.lrc", "[00:10.00]Letra A\n")
    second = _make_config(tmp_path, "song1.lrc", "[00:10.00]Letra A\n")
    assert compute_long_video_fingerprint(_URL, config=first) == (
        compute_long_video_fingerprint(_URL, config=second)
    )


def test_lyric_content_change_invalidates_fingerprint(tmp_path: Path) -> None:
    before = _make_config(tmp_path, "song_a.lrc", "[00:10.00]Letra A\n")
    after = _make_config(tmp_path, "song_b.lrc", "[00:10.00]Letra Modificada B\n")
    assert compute_long_video_fingerprint(_URL, config=before) != (
        compute_long_video_fingerprint(_URL, config=after)
    )


def test_lyric_format_change_invalidates_fingerprint(tmp_path: Path) -> None:
    before = _make_config(tmp_path, "song.lrc", "[00:10.00]Letra\n", format_=Format.LYRIC_VIDEO)
    after = _make_config(tmp_path, "song.lrc", "[00:10.00]Letra\n", format_=Format.VIDEO)
    assert compute_long_video_fingerprint(_URL, config=before) != (
        compute_long_video_fingerprint(_URL, config=after)
    )


def test_lyric_config_track_change_invalidates_fingerprint(tmp_path: Path) -> None:
    before = _make_config(tmp_path, "song.lrc", "[00:10.00]Letra\n", track_name="Track 1")
    after = _make_config(tmp_path, "song.lrc", "[00:10.00]Letra\n", track_name="Track 2")
    assert compute_long_video_fingerprint(_URL, config=before) != (
        compute_long_video_fingerprint(_URL, config=after)
    )


def test_resume_hydrates_subtitle_segments_and_text_from_persisted_ass(tmp_path: Path) -> None:
    work_dir = tmp_path / "work"
    work_dir.mkdir(parents=True, exist_ok=True)
    ass_path = work_dir / "subtitles.ass"
    ass_content = (
        "[Script Info]\nTitle: Test\nScriptType: v4.00+\n\n"
        "[Events]\n"
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
        "Dialogue: 0,0:00:01.00,0:00:04.00,Default,,0,0,0,,Primera línea\n"
        "Dialogue: 0,0:00:04.00,0:00:08.00,Default,,0,0,0,,Segunda línea\n"
    )
    _ = ass_path.write_text(ass_content, encoding="utf-8")

    text = hydrate_piece_subtitle_text(
        transcript=None,
        segment=None,
        subtitles_path=ass_path,
        work_dir=work_dir,
    )
    assert text == "Primera línea Segunda línea"

    segments = hydrate_piece_subtitle_segments(
        transcript=None,
        segment=None,
        subtitles_path=ass_path,
        work_dir=work_dir,
    )
    assert len(segments) == 2
    assert segments[0] == SubtitleSegment(text="Primera línea", start_s=1.0, end_s=4.0)
    assert segments[1] == SubtitleSegment(text="Segunda línea", start_s=4.0, end_s=8.0)
