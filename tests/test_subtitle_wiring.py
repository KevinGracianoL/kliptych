"""Tests del cableado real de subtitle_text hacia el gate (Fix 1 Sprint 1).

El validador ``subtitles.spelling_lock`` recibía siempre ``subtitle_text=None``
desde ``pipeline.py`` y ``campaign_manager.py``: el gate nunca podía verificar
los locks. Estos tests exigen texto real por pieza/segmento (recortado al rango
del segmento, jamás la transcripción completa) e hidratación desde artefactos
persistidos en modo ``--resume``.
"""

import json
from collections.abc import Sequence
from pathlib import Path
from typing import NoReturn, cast, override

import pytest

from kliptych.assembler import RenderSpec
from kliptych.assets import AssetRegistry
from kliptych.campaign_manager import CampaignManager
from kliptych.campaign_types import Campaign, CampaignStatus
from kliptych.config import Settings
from kliptych.contract import Contract, ContractDraft, Segment
from kliptych.environment import EnvironmentReport
from kliptych.exporter import ExportStatus
from kliptych.gate import (
    CheckStatus,
    Gate,
    GateResult,
    GateStatus,
    Piece,
    SubtitleSegment,
)
from kliptych.git_proposals import ProposalEngine, PullRequest
from kliptych.intelligence import Archetype, ArchetypeClassification
from kliptych.orchestrator import PipelineResult, SlideshowResult
from kliptych.pipeline import RunOutcome, RunRequest, run_given_clips
from kliptych.runtime import (
    CAPTION_PROMPT_VERSION,
    PROMPT_VERSION,
    CampaignModel,
    Caption,
    PieceContext,
)
from kliptych.segment import SegmentSelection
from kliptych.subtitle_text import (
    hydrate_piece_subtitle_segments,
    hydrate_piece_subtitle_text,
    segment_subtitle_text,
    subtitle_text_from_ass,
)
from kliptych.transcribe import Transcript, Word
from tests.support import (
    FakeProbe,
    candidate,
    make_asset_draft,
    make_contract,
    make_draft,
    make_media,
)

_VIDEO_URL = "https://example.com/video"


def _word(start: float, end: float, text: str) -> Word:
    return Word(start_s=start, end_s=end, text=text, confidence=0.9, token_id=0)


def _transcript() -> Transcript:
    words = (
        _word(0.0, 1.0, "hola"),
        _word(1.0, 2.0, "maracax"),
        _word(10.0, 11.0, "lejano"),
    )
    return Transcript(words=words, language="es", duration_s=12.0, text="hola maracax lejano")


def _ass_content() -> str:
    return (
        "[Script Info]\n"
        "Title: Kliptych\n"
        "ScriptType: v4.00+\n"
        "\n"
        "[V4+ Styles]\n"
        "Format: Name, Fontname\n"
        "Style: Default,Arial\n"
        "\n"
        "[Events]\n"
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
        "Dialogue: 0,0:00:00.00,0:00:01.00,Default,,0,0,0,,{\\k100}hola maracax\n"
    )


# ---------------------------------------------------------------------------
# Helpers puros: recorte por segmento y parseo de .ass
# ---------------------------------------------------------------------------


def test_segment_text_trims_to_segment_range() -> None:
    text = segment_subtitle_text(_transcript(), Segment(start_s=0.0, end_s=5.0))
    assert text == "hola maracax"


def test_segment_text_excludes_words_outside_range() -> None:
    text = segment_subtitle_text(_transcript(), Segment(start_s=9.0, end_s=12.0))
    assert text == "lejano"


def test_segment_text_without_words_in_range_is_none() -> None:
    assert segment_subtitle_text(_transcript(), Segment(start_s=5.0, end_s=9.0)) is None


def test_subtitle_text_from_ass_strips_karaoke_tags(tmp_path: Path) -> None:
    ass = tmp_path / "subtitles.ass"
    _ = ass.write_text(_ass_content(), encoding="utf-8")
    assert subtitle_text_from_ass(ass) == "hola maracax"


def test_subtitle_text_from_missing_ass_is_none(tmp_path: Path) -> None:
    assert subtitle_text_from_ass(tmp_path / "no-existe.ass") is None


def test_subtitle_text_from_ass_skips_malformed_dialogue(tmp_path: Path) -> None:
    ass = tmp_path / "mal.ass"
    lines = [
        "[Events]",
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
        "Dialogue: 0,0:00:00.00",
        "Dialogue: 0,0:00:00.00,0:00:01.00,Default,,0,0,0,,   ",
        "Dialogue: 0,0:00:00.00,0:00:01.00,Default,,0,0,0,,hola\\NMarcaX",
    ]
    _ = ass.write_text("\n".join(lines) + "\n", encoding="utf-8")
    assert subtitle_text_from_ass(ass) == "hola MarcaX"


def test_subtitle_text_from_ass_without_dialogue_is_none(tmp_path: Path) -> None:
    ass = tmp_path / "vacio.ass"
    _ = ass.write_text("[Events]\nFormat: Layer, Start\n", encoding="utf-8")
    assert subtitle_text_from_ass(ass) is None


def test_hydrate_prefers_transcript_over_disk(tmp_path: Path) -> None:
    disk = tmp_path / "transcript.json"
    _ = disk.write_text(
        Transcript(
            words=(_word(0.0, 1.0, "otro"),),
            language="es",
            duration_s=1.0,
            text="otro",
        ).model_dump_json(),
        encoding="utf-8",
    )
    text = hydrate_piece_subtitle_text(
        transcript=_transcript(),
        segment=Segment(start_s=0.0, end_s=5.0),
        work_dir=tmp_path,
    )
    assert text == "hola maracax"


def test_hydrate_from_saved_transcript_json_when_memory_is_none(tmp_path: Path) -> None:
    _ = (tmp_path / "transcript.json").write_text(_transcript().model_dump_json(), encoding="utf-8")
    text = hydrate_piece_subtitle_text(
        transcript=None,
        segment=Segment(start_s=0.0, end_s=5.0),
        work_dir=tmp_path,
    )
    assert text == "hola maracax"


def test_hydrate_from_ass_when_no_transcript_anywhere(tmp_path: Path) -> None:
    ass = tmp_path / "subtitles.ass"
    _ = ass.write_text(_ass_content(), encoding="utf-8")
    text = hydrate_piece_subtitle_text(
        transcript=None,
        segment=Segment(start_s=0.0, end_s=5.0),
        subtitles_path=ass,
        work_dir=tmp_path,
    )
    assert text == "hola maracax"
    segments = hydrate_piece_subtitle_segments(
        transcript=None,
        segment=Segment(start_s=0.0, end_s=5.0),
        subtitles_path=ass,
        work_dir=tmp_path,
    )
    assert len(segments) == 1
    assert segments[0].text == "hola maracax"
    assert segments[0].start_s == pytest.approx(0.0)
    assert segments[0].end_s == pytest.approx(1.0)


def test_hydrate_without_sources_is_none(tmp_path: Path) -> None:
    assert (
        hydrate_piece_subtitle_text(
            transcript=None,
            segment=Segment(start_s=0.0, end_s=5.0),
            work_dir=tmp_path,
        )
        is None
    )


def test_hydrate_trusts_memory_transcript_over_disk(tmp_path: Path) -> None:
    _ = (tmp_path / "transcript.json").write_text(_transcript().model_dump_json(), encoding="utf-8")
    assert (
        hydrate_piece_subtitle_text(
            transcript=_transcript(),
            segment=Segment(start_s=5.0, end_s=9.0),
            work_dir=tmp_path,
        )
        is None
    )


def test_hydrate_falls_back_to_ass_when_saved_transcript_is_corrupt(
    tmp_path: Path,
) -> None:
    _ = (tmp_path / "transcript.json").write_text("{no-json", encoding="utf-8")
    ass = tmp_path / "subtitles.ass"
    _ = ass.write_text(_ass_content(), encoding="utf-8")
    text = hydrate_piece_subtitle_text(
        transcript=None,
        segment=Segment(start_s=0.0, end_s=5.0),
        work_dir=tmp_path,
    )
    assert text == "hola maracax"


def test_hydrate_dedupes_primary_and_indexed_ass(tmp_path: Path) -> None:
    ass = tmp_path / "subtitles.ass"
    _ = ass.write_text(_ass_content(), encoding="utf-8")
    text = hydrate_piece_subtitle_text(
        transcript=None,
        segment=Segment(start_s=0.0, end_s=5.0),
        subtitles_path=ass,
        work_dir=tmp_path,
    )
    assert text == "hola maracax"


# ---------------------------------------------------------------------------
# Flujo pipeline.py (given_clips): subtítulos reales por asset
# ---------------------------------------------------------------------------


class _StaticModel(CampaignModel):
    """Modelo de prueba con draft fijo y caption fijo."""

    model_version: str = "static"

    def __init__(self, draft: ContractDraft, caption: Caption) -> None:
        self._draft: ContractDraft = draft
        self._caption: Caption = caption

    @override
    def extract_contract(self, brief: str) -> ContractDraft:
        _ = brief
        return self._draft

    @override
    def write_caption(self, contract: Contract, piece: PieceContext) -> Caption:
        _ = (contract, piece)
        return self._caption


class _StubAssembler:
    """Ensamblador de prueba: escribe bytes sin ffmpeg."""

    def __init__(self) -> None:
        self.assembled: list[Path] = []
        self.rendered: list[Path] = []

    def assemble(self, spec: RenderSpec) -> Path:
        _ = (spec.watermark, spec.watermark_config, spec.subtitles, spec.mute_audio)
        spec.destination.parent.mkdir(parents=True, exist_ok=True)
        _ = spec.destination.write_bytes(b"video")
        self.assembled.append(spec.destination)
        return spec.destination

    def render_arguments(self, spec: RenderSpec) -> tuple[str, ...]:
        _ = (spec.watermark, spec.watermark_config, spec.subtitles, spec.mute_audio)
        self.rendered.append(spec.destination)
        return ("ffmpeg", str(spec.destination))


def _caption() -> Caption:
    return Caption(caption="mira @marca #marca", hashtags=("#marca",))


def test_pipeline_spelling_lock_rejects_misspelled_real_subtitles(tmp_path: Path) -> None:
    _ = (tmp_path / "clip.mp4").write_bytes(b"clip")
    draft = make_draft(
        spelling_locks=candidate(["MarcaX"]),
        assets={"required": [make_asset_draft()], "optional": []},
    )
    request = RunRequest(
        brief="cita del brief\nbrief con spelling lock",
        destination=tmp_path / "delivery",
        environment=EnvironmentReport(),
        model_version="static",
        prompt_version=PROMPT_VERSION,
        caption_prompt_version=CAPTION_PROMPT_VERSION,
        assembler=_StubAssembler(),
        gate=Gate(FakeProbe(info=make_media())),
        subtitle_texts={"clip-01": "hablamos de maracax en el video"},
    )
    result = run_given_clips(
        model=_StaticModel(draft, _caption()),
        settings=Settings.from_root(tmp_path),
        request=request,
    )
    assert result.outcome is RunOutcome.BLOCKED
    assert result.delivery is not None
    rejected = result.delivery.rejected[0]
    assert "subtitles.spelling_lock" in rejected.reason
    assert rejected.gate.status.value == "rejected"
    matched = [c for c in rejected.gate.checks if c.id == "subtitles.spelling_lock"]
    assert matched
    assert matched[0].status is CheckStatus.FAIL


# ---------------------------------------------------------------------------
# Flujo campaign_manager.py: recorte por segmento + resume desde disco
# ---------------------------------------------------------------------------


class _FakeClassifier:
    def __init__(self) -> None:
        self.calls: list[tuple[str, Contract]] = []

    def classify(self, brief: str, contract: Contract) -> ArchetypeClassification:
        self.calls.append((brief, contract))
        return ArchetypeClassification(archetype=Archetype.KNOWN, rationale="t", variations=())


class _FakeProvider:
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


class _FakeVideoOrchestrator:
    def __init__(self, result: PipelineResult) -> None:
        self._result: PipelineResult = result
        self.calls: list[str] = []
        self.slideshow_calls: list[tuple[Path, ...]] = []

    def run_long_video(self, url: str, **kwargs: object) -> PipelineResult:
        _ = kwargs
        self.calls.append(url)
        return self._result

    def run_slideshow(self, images: Sequence[Path], **kwargs: object) -> SlideshowResult:
        _ = kwargs
        self.slideshow_calls.append(tuple(images))
        msg = "no debe usar slideshow en este test"
        raise AssertionError(msg)


class _RecordingGate(Gate):
    """Gate que registra las piezas evaluadas para inspeccionar subtitle_text."""

    def __init__(self, probe: FakeProbe) -> None:
        super().__init__(probe)
        self.pieces: list[Piece] = []

    @override
    def run(self, *, contract: Contract, piece: Piece, assets: AssetRegistry) -> GateResult:
        self.pieces.append(piece)
        return super().run(contract=contract, piece=piece, assets=assets)


def _campaign() -> Campaign:
    return Campaign(
        campaign_id="camp-01",
        brief="brief crudo",
        contract=make_contract(
            required_mentions=["@marca"],
            required_hashtags=["#marca"],
            audio_rule="any",
            spelling_locks=["MarcaX"],
        ),
    )


def _pipeline_result(work_dir: Path, *, transcript: Transcript | None) -> PipelineResult:
    final = work_dir / "final.mp4"
    _ = final.write_bytes(b"video content")
    return PipelineResult(
        source=work_dir / "source.mp4",
        transcript=transcript,
        moments=(),
        selection=SegmentSelection(
            segments=(Segment(start_s=0.0, end_s=5.0),), rationale="recorte"
        ),
        reframe=None,
        subtitles=None,
        final_video=final,
        cleaning=(),
    )


def _manager(
    video: _FakeVideoOrchestrator, gate: Gate, destination: Path, assets: AssetRegistry
) -> CampaignManager:
    return CampaignManager(
        classifier=_FakeClassifier(),
        proposal_engine=ProposalEngine(provider=_FakeProvider()),
        video_orchestrator=video,
        gate=gate,
        assets=assets,
        destination=destination,
    )


def test_campaign_spelling_lock_rejects_misspelled_segment_subtitles(tmp_path: Path) -> None:
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    _ = (work_dir / "source.mp4").write_bytes(b"source")
    video = _FakeVideoOrchestrator(_pipeline_result(work_dir, transcript=_transcript()))
    gate = _RecordingGate(FakeProbe(info=make_media(duration_s=10.0)))
    manager = _manager(video, gate, tmp_path / "delivery", AssetRegistry(tmp_path))

    outcome = manager.process(_campaign(), mode="long_video", url=_VIDEO_URL)

    assert outcome.status is CampaignStatus.BLOCKED
    assert outcome.delivery_report is not None
    assert outcome.delivery_report.status is ExportStatus.BLOCKED
    rejected = outcome.delivery_report.rejected[0]
    matched = [check for check in rejected.gate.checks if check.id == "subtitles.spelling_lock"]
    assert matched
    assert matched[0].status is CheckStatus.FAIL
    assert gate.pieces
    assert gate.pieces[0].subtitle_text == "hola maracax"


def test_campaign_spelling_lock_ignores_misspelling_outside_segment(tmp_path: Path) -> None:
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    _ = (work_dir / "source.mp4").write_bytes(b"source")
    words = (
        _word(0.0, 1.0, "hola"),
        _word(1.0, 2.0, "MarcaX"),
        _word(10.0, 11.0, "maracax"),
    )
    transcript = Transcript(words=words, language="es", duration_s=12.0, text="t")
    video = _FakeVideoOrchestrator(_pipeline_result(work_dir, transcript=transcript))
    gate = _RecordingGate(FakeProbe(info=make_media(duration_s=10.0)))
    manager = _manager(video, gate, tmp_path / "delivery", AssetRegistry(tmp_path))

    outcome = manager.process(_campaign(), mode="long_video", url=_VIDEO_URL)

    assert outcome.status is CampaignStatus.COMPLETED
    assert outcome.delivery_report is not None
    assert outcome.delivery_report.status is ExportStatus.EXPORTED


def test_campaign_resume_hydrates_subtitles_from_saved_transcript(tmp_path: Path) -> None:
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    _ = (work_dir / "source.mp4").write_bytes(b"source")
    _ = (work_dir / "transcript.json").write_text(_transcript().model_dump_json(), encoding="utf-8")
    video = _FakeVideoOrchestrator(_pipeline_result(work_dir, transcript=None))
    gate = _RecordingGate(FakeProbe(info=make_media(duration_s=10.0)))
    manager = _manager(video, gate, tmp_path / "delivery", AssetRegistry(tmp_path))

    outcome = manager.process(_campaign(), mode="long_video", url=_VIDEO_URL, resume=True)

    assert gate.pieces
    assert gate.pieces[0].subtitle_text == "hola maracax"
    assert outcome.status is CampaignStatus.BLOCKED
    assert outcome.delivery_report is not None
    rejected = outcome.delivery_report.rejected[0]
    matched = [check for check in rejected.gate.checks if check.id == "subtitles.spelling_lock"]
    assert matched
    assert matched[0].status is CheckStatus.FAIL


def test_h3_campaign_passes_subtitle_segments_and_evaluates_hook(tmp_path: Path) -> None:
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    _ = (work_dir / "source.mp4").write_bytes(b"source")
    video = _FakeVideoOrchestrator(_pipeline_result(work_dir, transcript=_transcript()))
    gate = _RecordingGate(FakeProbe(info=make_media(duration_s=10.0)))
    manager = _manager(video, gate, tmp_path / "delivery", AssetRegistry(tmp_path))

    campaign = Campaign(
        campaign_id="camp-01",
        brief="brief crudo",
        contract=make_contract(
            required_mentions=["@marca"],
            required_hashtags=["#marca"],
            audio_rule="any",
            hook_keyword="hola",
            hard=["artifact.integrity", "hook.keyword"],
        ),
    )
    _ = manager.process(campaign, mode="long_video", url=_VIDEO_URL)

    assert gate.pieces
    piece = gate.pieces[0]
    assert len(piece.subtitle_segments) >= 2
    assert piece.subtitle_segments[0].text == "hola"
    assert piece.subtitle_segments[0].start_s == pytest.approx(0.0)


def test_h3_campaign_resume_hydrates_subtitle_segments(tmp_path: Path) -> None:
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    _ = (work_dir / "source.mp4").write_bytes(b"source")
    _ = (work_dir / "transcript.json").write_text(_transcript().model_dump_json(), encoding="utf-8")
    video = _FakeVideoOrchestrator(_pipeline_result(work_dir, transcript=None))
    gate = _RecordingGate(FakeProbe(info=make_media(duration_s=10.0)))
    manager = _manager(video, gate, tmp_path / "delivery", AssetRegistry(tmp_path))

    campaign = Campaign(
        campaign_id="camp-01",
        brief="brief crudo",
        contract=make_contract(
            required_mentions=["@marca"],
            required_hashtags=["#marca"],
            audio_rule="any",
            hook_keyword="hola",
            hard=["artifact.integrity", "hook.keyword"],
        ),
    )
    _ = manager.process(campaign, mode="long_video", url=_VIDEO_URL, resume=True)

    assert gate.pieces
    piece = gate.pieces[0]
    assert len(piece.subtitle_segments) >= 2
    assert piece.subtitle_segments[0].text == "hola"
    assert piece.subtitle_segments[0].start_s == pytest.approx(0.0)


def test_h3_pipeline_passes_subtitle_segments_and_evaluates_hook(tmp_path: Path) -> None:
    _ = (tmp_path / "clip.mp4").write_bytes(b"clip")
    draft = make_draft(
        hook_keyword=candidate("mira"),
        assets={"required": [make_asset_draft()], "optional": []},
    )
    request = RunRequest(
        brief="cita del brief\nbrief con hook",
        destination=tmp_path / "delivery",
        environment=EnvironmentReport(),
        model_version="static",
        prompt_version=PROMPT_VERSION,
        caption_prompt_version=CAPTION_PROMPT_VERSION,
        assembler=_StubAssembler(),
        gate=Gate(FakeProbe(info=make_media())),
        subtitle_segments={"clip-01": (SubtitleSegment(text="mira esto", start_s=0.5, end_s=1.5),)},
    )
    result = run_given_clips(
        model=_StaticModel(draft, _caption()),
        settings=Settings.from_root(tmp_path),
        request=request,
    )
    assert result.delivery is not None
    assert result.outcome is RunOutcome.EXPORTED
    exported = result.delivery.exported[0]
    assert exported.gate_status is GateStatus.PASSED
    gate_data = cast(
        "dict[str, object]",
        json.loads((tmp_path / "delivery" / exported.gate_path).read_text(encoding="utf-8")),
    )
    checks = cast("list[dict[str, object]]", gate_data["checks"])
    matched = [c for c in checks if c["id"] == "hook.keyword"]
    assert matched
    assert matched[0]["status"] == "pass"
