"""Regresión P42-2a/2b/4: tiempos exactos, ASS fail-closed y ventanas por pieza.

P42-2a: el gate comparaba solo la secuencia de tokens, sin intervalos de
tiempo. Segmentos o eventos Dialogue desplazados pasaban el gate.
P42-2b: ass_path inexistente se sustituía en silencio por un sidecar válido.
P42-4: _build_pieces no rellenaba start_sec/end_sec/ass_path y el gate
comparaba cada pieza multi-clip contra el .lrc completo.
"""

import math
from collections.abc import Sequence
from pathlib import Path
from typing import NoReturn, override

from kliptych.assets import AssetRegistry
from kliptych.campaign_manager import CampaignManager
from kliptych.campaign_types import Campaign, CampaignStatus
from kliptych.contract import Contract, Format, LyricConfig, Segment
from kliptych.exporter import ExportStatus
from kliptych.gate import (
    CheckStatus,
    Gate,
    GateContext,
    GateResult,
    Piece,
    SubtitleSegment,
)
from kliptych.gate.checks import check_spelling_locks
from kliptych.git_proposals import ProposalEngine, PullRequest
from kliptych.intelligence import Archetype, ArchetypeClassification
from kliptych.lyrics import LyricLine, lyric_lines_to_ass
from kliptych.orchestrator import PipelineResult, SlideshowResult
from kliptych.segment import SegmentSelection
from kliptych.transcribe import Transcript, Word
from tests.support import FakeProbe, make_contract, make_media, make_piece

_LRC_TWO_LINES = "[00:00.00]alpha beta\n[00:02.00]gamma delta\n"
_LRC_TWO_CLIPS = "[00:00.00]first half\n[00:05.00]second half\n"


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
    timestamp_ranges: Sequence[tuple[float, float]] = (),
) -> GateContext:
    contract = make_contract(
        format_=Format.LYRIC_VIDEO,
        lyric_video=LyricConfig(lrc_asset_id="song_lrc"),
        timestamp_ranges=list(timestamp_ranges),
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


def test_p42_2b_declared_missing_ass_path_fails(tmp_path: Path) -> None:
    """ass_path inexistente falla aunque exista un sidecar válido en el dir."""
    registry = _register_lrc(tmp_path)
    _ = _write_ass(
        tmp_path / "subtitles.ass",
        [
            ("0:00:00.00", "0:00:02.00", "alpha beta"),
            ("0:00:02.00", "0:00:05.00", "gamma delta"),
        ],
    )
    ctx = _lyric_context(
        registry,
        tmp_path / "clip.mp4",
        subtitle_text="alpha beta gamma delta",
        subtitle_segments=[
            SubtitleSegment(text="alpha beta", start_s=0.0, end_s=2.0),
            SubtitleSegment(text="gamma delta", start_s=2.0, end_s=5.0),
        ],
        ass_path=tmp_path / "missing.ass",
    )
    outcome = check_spelling_locks(ctx)
    assert outcome.status is CheckStatus.FAIL


def test_p42_2b_missing_sidecar_fails_closed(tmp_path: Path) -> None:
    """Lyric video sin .ass verificable falla en cerrado, nunca pasa."""
    registry = _register_lrc(tmp_path)
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
    )
    outcome = check_spelling_locks(ctx)
    assert outcome.status is CheckStatus.FAIL


_RANGES_TWO_CLIPS: list[tuple[float, float]] = [(0.0, 5.0), (5.0, 10.0)]


def _two_clip_contexts(tmp_path: Path, *, swapped: bool = False) -> list[GateContext]:
    """Arma dos piezas [0,5]+[5,10] con subtítulos y .ass por ventana.

    Args:
        tmp_path: Directorio temporal del test.
        swapped: Si es True, cruza los contenidos entre ventanas.

    Returns:
        Los dos contextos listos para el gate.
    """
    registry = _register_lrc(tmp_path, _LRC_TWO_CLIPS)
    first = ("first half", 0.0, 5.0, "0:00:00.00", "0:00:05.00")
    second = ("second half", 0.0, 4.0, "0:00:00.00", "0:00:04.00")
    windows = [(0.0, 5.0), (5.0, 10.0)]
    contents = [second, first] if swapped else [first, second]
    contexts: list[GateContext] = []
    for i, ((text, seg_start, seg_end, ass_start, ass_end), (win_start, win_end)) in enumerate(
        zip(contents, windows, strict=True)
    ):
        ass_path = tmp_path / f"clip_{i}.ass"
        _ = _write_ass(ass_path, [(ass_start, ass_end, text)])
        contexts.append(
            _lyric_context(
                registry,
                tmp_path / f"clip_{i}.mp4",
                start_sec=win_start,
                end_sec=win_end,
                subtitle_text=text,
                subtitle_segments=[SubtitleSegment(text=text, start_s=seg_start, end_s=seg_end)],
                ass_path=ass_path,
                timestamp_ranges=_RANGES_TWO_CLIPS,
            )
        )
    return contexts


def test_p42_4_two_clip_windows_pass(tmp_path: Path) -> None:
    """Campaña [0,5]+[5,10] con ventanas correctas: ambas piezas pasan."""
    for ctx in _two_clip_contexts(tmp_path):
        assert check_spelling_locks(ctx).status is CheckStatus.PASS


def test_p42_4_swapped_windows_fail(tmp_path: Path) -> None:
    """Ventanas cruzadas ([0,5] con contenido de [5,10]): ambas fallan."""
    for ctx in _two_clip_contexts(tmp_path, swapped=True):
        assert check_spelling_locks(ctx).status is CheckStatus.FAIL


def test_p42_4_multi_range_missing_window_fails(tmp_path: Path) -> None:
    """Multi-rango sin start_sec/end_sec en la pieza: falla explícito."""
    registry = _register_lrc(tmp_path, _LRC_TWO_CLIPS)
    _ = _write_ass(
        tmp_path / "subtitles.ass",
        [
            ("0:00:00.00", "0:00:05.00", "first half"),
            ("0:00:05.00", "0:00:09.00", "second half"),
        ],
    )
    ctx = _lyric_context(
        registry,
        tmp_path / "clip.mp4",
        subtitle_text="first half second half",
        subtitle_segments=[
            SubtitleSegment(text="first half", start_s=0.0, end_s=5.0),
            SubtitleSegment(text="second half", start_s=5.0, end_s=9.0),
        ],
        timestamp_ranges=_RANGES_TWO_CLIPS,
    )
    outcome = check_spelling_locks(ctx)
    assert outcome.status is CheckStatus.FAIL


def test_p42_4_build_pieces_preserves_windows_and_ass(tmp_path: Path) -> None:
    """El manager propaga ventana/.ass por pieza y ambas pasan el gate."""
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    _ = (tmp_path / "song.lrc").write_text(_LRC_TWO_CLIPS, encoding="utf-8")
    registry = AssetRegistry(tmp_path)
    _ = registry.register(asset_id="song_lrc", kind="lyrics", uri="song.lrc", origin="test")

    ass_00 = work_dir / "subtitles_00.ass"
    _ = ass_00.write_text(
        lyric_lines_to_ass((LyricLine(start_sec=0.0, text="first half", end_sec=5.0),)),
        encoding="utf-8",
    )
    ass_01 = work_dir / "subtitles_01.ass"
    _ = ass_01.write_text(
        lyric_lines_to_ass((LyricLine(start_sec=0.0, text="second half", end_sec=4.0),)),
        encoding="utf-8",
    )
    final_00 = work_dir / "final_00.mp4"
    _ = final_00.write_bytes(b"video zero")
    final_01 = work_dir / "final_01.mp4"
    _ = final_01.write_bytes(b"video one")
    _ = (work_dir / "source.mp4").write_bytes(b"source")

    transcript = Transcript(
        words=(
            Word(start_s=0.0, end_s=1.0, text="whisper", confidence=0.9, token_id=0),
            Word(start_s=6.0, end_s=7.0, text="words", confidence=0.9, token_id=1),
        ),
        language="es",
        duration_s=12.0,
        text="whisper words",
    )
    result = PipelineResult(
        source=work_dir / "source.mp4",
        transcript=transcript,
        moments=(),
        selection=SegmentSelection(
            segments=(Segment(start_s=0.0, end_s=5.0), Segment(start_s=5.0, end_s=10.0)),
            rationale="dos cortes",
        ),
        reframe=None,
        subtitles=ass_00,
        final_video=final_00,
        cleaning=(),
        final_videos=(final_00, final_01),
    )
    contract = make_contract(
        format_=Format.LYRIC_VIDEO,
        lyric_video=LyricConfig(lrc_asset_id="song_lrc"),
        timestamp_ranges=_RANGES_TWO_CLIPS,
        audio_rule="any",
    )
    recording_gate = _RecordingGate(FakeProbe(info=make_media()))
    manager = CampaignManager(
        classifier=_KnownClassifier(),
        proposal_engine=ProposalEngine(provider=_NeverProposeProvider()),
        video_orchestrator=_StubVideoOrchestrator(result),
        gate=recording_gate,
        assets=registry,
        destination=tmp_path / "delivery",
    )
    outcome = manager.process(
        Campaign(campaign_id="camp-01", brief="brief crudo", contract=contract),
        mode="long_video",
        url="https://example.com/video",
        caption="mira @marca #marca",
    )

    assert len(recording_gate.pieces) == 2
    first, second = recording_gate.pieces
    assert first.start_sec is not None
    assert math.isclose(first.start_sec, 0.0)
    assert first.end_sec is not None
    assert math.isclose(first.end_sec, 5.0)
    assert first.ass_path == ass_00
    assert first.subtitle_text == "first half"
    assert second.start_sec is not None
    assert math.isclose(second.start_sec, 5.0)
    assert second.end_sec is not None
    assert math.isclose(second.end_sec, 10.0)
    assert second.ass_path == ass_01
    assert second.subtitle_text == "second half"
    assert outcome.status is CampaignStatus.COMPLETED
    assert outcome.delivery_report is not None
    assert outcome.delivery_report.status is ExportStatus.EXPORTED


class _KnownClassifier:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def classify(self, brief: str, contract: Contract) -> ArchetypeClassification:
        self.calls.append(brief)
        _ = contract
        return ArchetypeClassification(archetype=Archetype.KNOWN, rationale="test")


class _NeverProposeProvider:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def _reject(self, call: str) -> NoReturn:
        self.calls.append(call)
        msg = "no debe proponer en modo KNOWN"
        raise AssertionError(msg)

    def create_branch(self, *, base: str, name: str) -> str:
        _ = (base, name)
        self._reject("create_branch")

    def read_file(self, *, branch: str, path: str) -> str | None:
        _ = (branch, path)
        self._reject("read_file")

    def write_file(self, *, branch: str, path: str, content: str, message: str) -> str:
        _ = (branch, path, content, message)
        self._reject("write_file")

    def open_pull_request(
        self,
        *,
        branch: str,
        base: str,
        title: str,
        body: str,
        campaign_id: str,
        archetype: Archetype,
    ) -> PullRequest:
        _ = (branch, base, title, body, campaign_id, archetype)
        self._reject("open_pull_request")


class _StubVideoOrchestrator:
    _result: PipelineResult

    def __init__(self, result: PipelineResult) -> None:
        self._result = result
        self.slideshow_calls: list[tuple[Path, ...]] = []

    def run_long_video(self, url: str, **kwargs: object) -> PipelineResult:
        _ = kwargs
        _ = url
        return self._result

    def run_slideshow(self, images: Sequence[Path], **kwargs: object) -> SlideshowResult:
        _ = kwargs
        self.slideshow_calls.append(tuple(images))
        msg = "no debe usar slideshow en este test"
        raise AssertionError(msg)


class _RecordingGate(Gate):
    def __init__(self, probe: FakeProbe) -> None:
        super().__init__(probe)
        self.pieces: list[Piece] = []

    @override
    def run(self, *, contract: Contract, piece: Piece, assets: AssetRegistry) -> GateResult:
        self.pieces.append(piece)
        return super().run(contract=contract, piece=piece, assets=assets)
