"""Regresión P42-4a: coordenadas absolutas para el recorte .lrc con descarga quirúrgica.

Con rangos quirúrgicos (p. ej. [2,7] y [7,12]) el vídeo descargado arranca en
0.0 s y los segmentos de la selección son relativos ([0,5] y [5,10]). El
recorte del .lrc debe usar coordenadas absolutas (relativo + source_offset_sec);
los tiempos quemados en el .ass y los subtitle_segments siguen relativos al
inicio del clip (0.0 s), pero el contenido corresponde a la ventana absoluta.
"""

import math
import shutil
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import NoReturn, override

import pytest

from kliptych.assets import AssetRegistry
from kliptych.campaign_manager import CampaignManager
from kliptych.campaign_types import Campaign, CampaignStatus
from kliptych.contract import Contract, Format, LyricConfig, Segment
from kliptych.encoding import RenderConfig
from kliptych.exporter import ExportStatus
from kliptych.gate import (
    CheckStatus,
    Gate,
    GateContext,
    GateResult,
    Piece,
)
from kliptych.gate.checks import check_spelling_locks
from kliptych.git_proposals import ProposalEngine, PullRequest
from kliptych.intelligence import Archetype, ArchetypeClassification
from kliptych.lyrics import cut_lyric_window, lyric_lines_to_ass, parse_lrc
from kliptych.moments import Moment
from kliptych.orchestrator import PipelineConfig, PipelineResult, SlideshowResult, run_long_video
from kliptych.reframe import ReframeResult
from kliptych.segment import SegmentSelection
from kliptych.transcribe import Transcript
from tests.support import FakeProbe, make_contract, make_media

_FFMPEG = shutil.which("ffmpeg")

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(_FFMPEG is None, reason="ffmpeg no instalado"),
]

_LRC_OFFSET = "[00:00.00]PRELUDE\n[00:02.00]ACTUAL FIRST\n[00:07.00]ACTUAL SECOND\n[00:12.00]OUTRO\n"
_RANGES_ABSOLUTE: list[tuple[float, float]] = [(2.0, 7.0), (7.0, 12.0)]
_VIDEO_URL = "https://example.com/source.mp4"
_FIXTURE_TIMEOUT_S = 120


def _generate_source(path: Path, duration: int = 12) -> Path:
    assert _FFMPEG is not None
    argv = [
        _FFMPEG,
        "-hide_banner",
        "-v",
        "error",
        "-y",
        "-f",
        "lavfi",
        "-i",
        f"testsrc=size=360x640:rate=10:duration={duration}",
        "-f",
        "lavfi",
        "-i",
        f"sine=frequency=440:duration={duration}",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-shortest",
        str(path),
    ]
    completed = subprocess.run(
        argv, capture_output=True, text=True, timeout=_FIXTURE_TIMEOUT_S, check=False
    )
    assert completed.returncode == 0, completed.stderr
    return path


def _install_section_downloader(monkeypatch: pytest.MonkeyPatch, fixture: Path) -> None:
    """Downloader que ignora la sección quirúrgica y copia la fixture completa.

    El corte a la ventana descargada lo hace el pipeline real con ffmpeg
    (``_cut_exact_ffmpeg``), igual que en producción.
    """

    class _Downloader:
        def __init__(self, *, timeout_s: float, max_size_bytes: int) -> None:
            _ = (timeout_s, max_size_bytes)

        def download_video(
            self,
            *,
            url: str,
            destination: Path,
            format_selector: str | None = None,
            section: tuple[float, float] | None = None,
        ) -> Path:
            _ = (url, format_selector, section)
            destination.parent.mkdir(parents=True, exist_ok=True)
            _ = shutil.copyfile(fixture, destination)
            return destination

    monkeypatch.setattr("kliptych.orchestrator.MediaDownloader", _Downloader)


class _EmptyTranscriber:
    def transcribe(self, audio: Path) -> Transcript:
        _ = audio
        return Transcript(words=(), language="es", duration_s=10.0, text="sin palabras")


class _NoMoments:
    def detect(
        self, video: Path, *, transcript: Transcript | None = None
    ) -> tuple[Moment, ...]:
        _ = (video, transcript)
        return ()


class _RelativeSelector:
    """Selector que devuelve los segmentos relativos al vídeo descargado."""

    def build_prompt(
        self,
        transcript: Transcript,
        moments: tuple[Moment, ...],
        contract: Contract,
    ) -> dict[str, object]:
        _ = (transcript, moments, contract)
        return {}

    def parse_response(self, raw: object) -> SegmentSelection:
        _ = raw
        return SegmentSelection(
            segments=(Segment(start_s=0.0, end_s=5.0), Segment(start_s=5.0, end_s=10.0)),
            rationale="dos cortes relativos al vídeo descargado",
        )


class _CopyReframer:
    def analyze(self, video: Path) -> ReframeResult:
        _ = video
        return ReframeResult(
            targets=(), source_width=360, source_height=640, target_aspect="9:16"
        )

    def render(self, *, video: Path, destination: Path, result: ReframeResult) -> Path:
        _ = result
        destination.parent.mkdir(parents=True, exist_ok=True)
        _ = shutil.copyfile(video, destination)
        return destination


class _DummyModel:
    @staticmethod
    def select_segments(prompt: Mapping[str, object]) -> object:
        _ = prompt
        return {}


def _relative_selection() -> SegmentSelection:
    return SegmentSelection(
        segments=(Segment(start_s=0.0, end_s=5.0), Segment(start_s=5.0, end_s=10.0)),
        rationale="dos cortes relativos",
    )


def test_p42_4a_surgical_ranges_burn_absolute_lyric_window(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Rangos [2,7]+[7,12]: el .ass usa contenido absoluto con tiempos relativos."""
    source = _generate_source(tmp_path / "source.mp4")
    _install_section_downloader(monkeypatch, source)
    lrc_path = tmp_path / "lyrics.lrc"
    _ = lrc_path.write_text(_LRC_OFFSET, encoding="utf-8")

    out_dir = tmp_path / "out"
    contract = make_contract(
        format_=Format.LYRIC_VIDEO,
        timestamp_ranges=_RANGES_ABSOLUTE,
    )
    config = PipelineConfig(
        output_dir=out_dir,
        contract=contract,
        render=RenderConfig(nvenc_available=False),
        lrc_path=lrc_path,
    )
    result = run_long_video(
        _VIDEO_URL,
        model=_DummyModel(),
        config=config,
        detector=_NoMoments(),
        transcriber=_EmptyTranscriber(),
        selector=_RelativeSelector(),
        reframer=_CopyReframer(),
        resume=False,
    )

    assert math.isclose(result.source_offset_sec, 2.0)
    assert [(s.start_s, s.end_s) for s in result.selection.segments] == [
        (0.0, 5.0),
        (5.0, 10.0),
    ]
    ass_00 = (out_dir / "subtitles_00.ass").read_text(encoding="utf-8")
    ass_01 = (out_dir / "subtitles_01.ass").read_text(encoding="utf-8")
    assert "ACTUAL FIRST" in ass_00
    assert "PRELUDE" not in ass_00
    assert "ACTUAL SECOND" not in ass_00
    assert "ACTUAL SECOND" in ass_01
    assert "ACTUAL FIRST" not in ass_01
    assert "PRELUDE" not in ass_01
    assert "Dialogue: 0,0:00:00.00," in ass_00
    assert "Dialogue: 0,0:00:00.00," in ass_01

    resumed = run_long_video(
        _VIDEO_URL,
        model=_DummyModel(),
        config=config,
        detector=_NoMoments(),
        transcriber=_EmptyTranscriber(),
        selector=_RelativeSelector(),
        reframer=_CopyReframer(),
        resume=True,
    )
    assert math.isclose(resumed.source_offset_sec, 2.0)
    assert (out_dir / "subtitles_00.ass").read_text(encoding="utf-8") == ass_00
    assert (out_dir / "subtitles_01.ass").read_text(encoding="utf-8") == ass_01


def _register_offset_lrc(tmp_path: Path) -> AssetRegistry:
    _ = (tmp_path / "song.lrc").write_text(_LRC_OFFSET, encoding="utf-8")
    registry = AssetRegistry(tmp_path)
    _ = registry.register(asset_id="song_lrc", kind="lyrics", uri="song.lrc", origin="test")
    return registry


def test_p42_4a_pieces_carry_absolute_windows_with_relative_segments(
    tmp_path: Path,
) -> None:
    """El manager propaga ventanas absolutas y letras relativas que pasan el gate."""
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    registry = _register_offset_lrc(tmp_path)
    lines = parse_lrc(_LRC_OFFSET)
    ass_00 = work_dir / "subtitles_00.ass"
    _ = ass_00.write_text(
        lyric_lines_to_ass(cut_lyric_window(lines, start_sec=2.0, end_sec=7.0)),
        encoding="utf-8",
    )
    ass_01 = work_dir / "subtitles_01.ass"
    _ = ass_01.write_text(
        lyric_lines_to_ass(cut_lyric_window(lines, start_sec=7.0, end_sec=12.0)),
        encoding="utf-8",
    )
    final_00 = work_dir / "final_00.mp4"
    _ = final_00.write_bytes(b"video zero")
    final_01 = work_dir / "final_01.mp4"
    _ = final_01.write_bytes(b"video one")
    _ = (work_dir / "source.mp4").write_bytes(b"source")

    result = PipelineResult(
        source=work_dir / "source.mp4",
        transcript=Transcript(
            words=(), language="es", duration_s=10.0, text="sin palabras"
        ),
        moments=(),
        selection=_relative_selection(),
        reframe=None,
        subtitles=ass_00,
        final_video=final_00,
        cleaning=(),
        final_videos=(final_00, final_01),
        source_offset_sec=2.0,
    )
    contract = make_contract(
        format_=Format.LYRIC_VIDEO,
        lyric_video=LyricConfig(lrc_asset_id="song_lrc"),
        timestamp_ranges=_RANGES_ABSOLUTE,
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
    assert math.isclose(first.start_sec, 2.0)
    assert first.end_sec is not None
    assert math.isclose(first.end_sec, 7.0)
    assert first.subtitle_text == "ACTUAL FIRST"
    assert len(first.subtitle_segments) == 1
    assert math.isclose(first.subtitle_segments[0].start_s, 0.0)
    assert second.start_sec is not None
    assert math.isclose(second.start_sec, 7.0)
    assert second.end_sec is not None
    assert math.isclose(second.end_sec, 12.0)
    assert second.subtitle_text == "ACTUAL SECOND"
    assert len(second.subtitle_segments) == 1
    assert math.isclose(second.subtitle_segments[0].start_s, 0.0)
    assert outcome.status is CampaignStatus.COMPLETED
    assert outcome.delivery_report is not None
    assert outcome.delivery_report.status is ExportStatus.EXPORTED

    platform = next(iter(contract.platforms.keys()))
    for piece in recording_gate.pieces:
        ctx = GateContext(
            contract=contract,
            rules=contract.platforms[platform],
            piece=piece,
            artifact_sha256="f" * 64,
            media=make_media(duration_s=10.0),
            assets=registry,
        )
        assert check_spelling_locks(ctx).status is CheckStatus.PASS


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

    def run_long_video(self, url: str, **kwargs: object) -> PipelineResult:
        _ = kwargs
        _ = url
        return self._result

    def run_slideshow(self, images: Sequence[Path], **kwargs: object) -> SlideshowResult:
        _ = kwargs
        _ = images
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
