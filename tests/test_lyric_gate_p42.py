"""Regresión P42-2a/2b/4: tiempos exactos, ASS fail-closed y ventanas por pieza.

P42-2a: el gate comparaba solo la secuencia de tokens, sin intervalos de
tiempo. Segmentos o eventos Dialogue desplazados pasaban el gate.
P42-2b: ass_path inexistente se sustituía en silencio por un sidecar válido.
P42-4: _build_pieces no rellenaba start_sec/end_sec/ass_path y el gate
comparaba cada pieza multi-clip contra el .lrc completo.
"""

from collections.abc import Sequence
from pathlib import Path

from kliptych.assets import AssetRegistry
from kliptych.contract import Format, LyricConfig
from kliptych.gate import CheckStatus, GateContext, SubtitleSegment
from kliptych.gate.checks import check_spelling_locks
from tests.support import make_contract, make_media, make_piece

_LRC_TWO_LINES = "[00:00.00]alpha beta\n[00:02.00]gamma delta\n"


def _register_lrc(tmp_path: Path, content: str = _LRC_TWO_LINES) -> AssetRegistry:
    _ = (tmp_path / "song.lrc").write_text(content, encoding="utf-8")
    registry = AssetRegistry(tmp_path)
    _ = registry.register(
        asset_id="song_lrc",
        kind="lyrics",
        uri="song.lrc",
        origin="test",
    )
    return registry


def _lyric_context(
    registry: AssetRegistry,
    artifact: Path,
    *,
    start_sec: float | None = None,
    end_sec: float | None = None,
    subtitle_text: str | None = None,
    subtitle_segments: Sequence[SubtitleSegment] = (),
    ass_path: Path | None = None,
) -> GateContext:
    contract = make_contract(
        format_=Format.LYRIC_VIDEO,
        lyric_video=LyricConfig(lrc_asset_id="song_lrc"),
    )
    piece = make_piece(
        artifact,
        start_sec=start_sec,
        end_sec=end_sec,
        subtitle_text=subtitle_text,
        subtitle_segments=subtitle_segments,
        ass_path=ass_path,
    )
    platform = next(iter(contract.platforms.keys()))
    return GateContext(
        contract=contract,
        rules=contract.platforms[platform],
        piece=piece,
        artifact_sha256="f" * 64,
        media=make_media(duration_s=10.0),
        assets=registry,
    )


def _write_ass(path: Path, events: list[tuple[str, str, str]]) -> Path:
    lines = [
        "[Script Info]",
        "Title: Test",
        "",
        "[Events]",
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
    ]
    for start, end, text in events:
        lines.append(f"Dialogue: 0,{start},{end},Default,,0,0,0,,{text}")
    _ = path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_p42_2a_segments_with_wrong_times_fail(tmp_path: Path) -> None:
    """Tokens correctos pero intervalos desplazados (4.0-4.5) deben fallar."""
    registry = _register_lrc(tmp_path)
    ctx = _lyric_context(
        registry,
        tmp_path / "clip.mp4",
        start_sec=0.0,
        end_sec=5.0,
        subtitle_text="alpha beta gamma delta",
        subtitle_segments=[
            SubtitleSegment(text="alpha beta", start_s=4.0, end_s=4.5),
            SubtitleSegment(text="gamma delta", start_s=4.5, end_s=5.0),
        ],
    )
    outcome = check_spelling_locks(ctx)
    assert outcome.status is CheckStatus.FAIL


def test_p42_2a_ass_dialogue_with_wrong_times_fails(tmp_path: Path) -> None:
    """Eventos Dialogue desplazados (8-9 s) con texto correcto deben fallar."""
    registry = _register_lrc(tmp_path)
    ass_path = _write_ass(
        tmp_path / "clip.ass",
        [
            ("0:00:08.00", "0:00:08.50", "alpha beta"),
            ("0:00:08.50", "0:00:09.00", "gamma delta"),
        ],
    )
    ctx = _lyric_context(
        registry,
        tmp_path / "clip.mp4",
        start_sec=0.0,
        end_sec=5.0,
        subtitle_text="alpha beta gamma delta",
        subtitle_segments=[
            SubtitleSegment(text="alpha beta", start_s=0.0, end_s=2.0),
            SubtitleSegment(text="gamma delta", start_s=2.0, end_s=5.0),
        ],
        ass_path=ass_path,
    )
    outcome = check_spelling_locks(ctx)
    assert outcome.status is CheckStatus.FAIL
