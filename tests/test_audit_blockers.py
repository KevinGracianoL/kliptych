"""Tests adversariales para los 5 blockers de auditoría.

1. Mandatory Gate in Default Path: no retorna COMPLETED sin evaluar Gate.
2. CLI Exit Code & --approve-manual-review: CLI retorna 1 en BLOCKED.
3. Proyección de obligaciones huérfanas: disparan PENDING_REVIEW.
4. Lote atómico en PARTIAL: no publica si no es EXPORTED.
5. Procedencia real brief_text vs ContractDraft: rechazo de citas alteradas.
"""

import subprocess
from collections.abc import Sequence
from datetime import UTC, datetime
from io import StringIO
from pathlib import Path
from typing import override
from unittest.mock import patch

import pytest

from kliptych.__main__ import main
from kliptych.assets import AssetRegistry
from kliptych.campaign_manager import (
    CampaignManager,
    CampaignOutcome,
)
from kliptych.campaign_types import Campaign, CampaignStatus
from kliptych.config import Settings
from kliptych.contract import (
    Attribution,
    AttributionType,
    AudioRule,
    Contract,
    FieldCandidate,
    Format,
    LinkRules,
    Mode,
    OfficialAudio,
    Platform,
    PlatformRules,
)
from kliptych.contract.draft import ContractDraft, DurationDraft, PlatformDraft
from kliptych.contract.evidence import Confidence, SourceEvidence
from kliptych.contract.schema import Segment
from kliptych.encoding import RenderConfig, run_ffmpeg_with_fallback
from kliptych.environment import SubprocessRunner, detect_environment
from kliptych.exporter import DeliveryReport, ExportStatus, export_delivery
from kliptych.gate import CheckStatus, Gate, GateStatus, Piece
from kliptych.gate.checks import DEFAULT_VALIDATORS, CheckOutcome, GateContext
from kliptych.git_proposals import GitHubCliProvider, ProposalEngine, PullRequest
from kliptych.intelligence import Archetype, ArchetypeClassification, CampaignClassifier
from kliptych.manifest import RunManifest, read_manifest, write_manifest
from kliptych.orchestrator import PipelineResult, SlideshowResult
from kliptych.pipeline import PipelineError, RunRequest, run_given_clips
from kliptych.reframe import ReframeError
from kliptych.resolver import ProvenanceError, resolve_contract
from kliptych.runtime import CampaignModel, Caption, PieceContext
from kliptych.segment import SegmentSelection
from kliptych.subtitles import SubtitleError
from tests.support import FakeProbe, make_contract, make_media, make_piece


def _ensure_outcome_resolved() -> None:
    _ = CampaignOutcome.model_rebuild(
        _types_namespace={
            "PullRequest": PullRequest,
            "PipelineResult": PipelineResult,
            "DeliveryReport": DeliveryReport,
        }
    )


def _fake_probe() -> FakeProbe:
    return FakeProbe(info=make_media(duration_s=15.0, has_video=True, has_audio=True))


def _valid_piece(tmp_path: Path, caption: str = "mira @marca #marca") -> Piece:
    artifact = tmp_path / "valid.mp4"
    if not artifact.exists():
        _ = artifact.write_bytes(b"valid video payload")
    return make_piece(artifact, caption=caption, hashtags=["#marca"])


class _DummyClassifier(CampaignClassifier):
    @override
    def classify(self, brief: str, contract: Contract) -> ArchetypeClassification:
        _ = (brief, contract)
        return ArchetypeClassification(
            archetype=Archetype.KNOWN,
            rationale="test",
        )


# ---------------------------------------------------------------------------
# 1. Gate obligatorio en CampaignManager
# ---------------------------------------------------------------------------


def test_campaign_manager_without_gate_fails(tmp_path: Path) -> None:
    """CampaignManager no debe retornar COMPLETED si no hay gate configurado."""
    video_path = tmp_path / "rendered.mp4"
    _ = video_path.write_bytes(b"video payload")

    class FakeVideo:
        @staticmethod
        def run_long_video(url: str, **kwargs: object) -> PipelineResult:
            _ = (url, kwargs)
            return PipelineResult(
                source=video_path,
                transcript=None,
                moments=(),
                selection=SegmentSelection(
                    segments=(Segment(start_s=0.0, end_s=1.0),), rationale="test"
                ),
                reframe=None,
                subtitles=None,
                final_video=video_path,
                cleaning=(),
            )

        @staticmethod
        def run_slideshow(images: Sequence[Path], **kwargs: object) -> SlideshowResult:
            _ = (images, kwargs)
            raise NotImplementedError

    manager = CampaignManager(
        classifier=_DummyClassifier(),
        proposal_engine=ProposalEngine(provider=GitHubCliProvider(workdir=tmp_path, repo="o/r")),
        video_orchestrator=FakeVideo(),
        gate=None,
    )
    campaign = Campaign(
        campaign_id="camp-no-gate",
        brief="brief",
        contract=make_contract(),
    )
    outcome = manager.process(
        campaign,
        mode="long_video",
        url="http://fake.mp4",
        gate=None,
    )
    assert outcome.status is not CampaignStatus.COMPLETED
    assert outcome.delivery_report is None
    assert outcome.error is not None
    assert "Gate" in outcome.error


def test_cli_campaign_exits_one_on_blocked_status_despite_pipeline_success(tmp_path: Path) -> None:
    """Kliptych campaign retorna 1 en BLOCKED aunque haya pipeline_result."""
    _ensure_outcome_resolved()
    video_path = tmp_path / "final.mp4"
    _ = video_path.write_bytes(b"video payload")
    blocked_outcome = CampaignOutcome(
        campaign_id="camp-test",
        archetype=Archetype.KNOWN,
        status=CampaignStatus.BLOCKED,
        pipeline_result=PipelineResult(
            source=video_path,
            transcript=None,
            moments=(),
            selection=SegmentSelection(
                segments=(Segment(start_s=0.0, end_s=1.0),), rationale="test"
            ),
            reframe=None,
            subtitles=None,
            final_video=video_path,
            cleaning=(),
        ),
    )

    class MockManager:
        @staticmethod
        def process(
            campaign: Campaign,
            *,
            mode: str = "long_video",
            url: str | None = None,
            resume: bool = False,
            approve_manual_review: bool = False,
            approved_by: str | None = None,
        ) -> CampaignOutcome:
            _ = (campaign, mode, url, resume, approve_manual_review, approved_by)
            return blocked_outcome

    brief_file = tmp_path / "brief.txt"
    _ = brief_file.write_text("brief content", encoding="utf-8")
    code = main(
        ["campaign", str(brief_file), "--out", str(tmp_path / "out"), "--url", "http://fake.mp4"],
        manager=MockManager(),
    )
    assert code == 1


# ---------------------------------------------------------------------------
# 2. CLI --help expone --approve-manual-review
# ---------------------------------------------------------------------------


def test_cli_campaign_help_exposes_approve_manual_review() -> None:
    stream = StringIO()
    with pytest.raises(SystemExit) as exc_info, patch("sys.stdout", stream):
        _ = main(["campaign", "--help"])
    assert exc_info.value.code == 0
    help_text = stream.getvalue()
    assert "--approve-manual-review" in help_text
    assert "--approved-by" in help_text


def test_cli_run_help_exposes_approve_manual_review() -> None:
    stream = StringIO()
    with pytest.raises(SystemExit) as exc_info, patch("sys.stdout", stream):
        _ = main(["run", "--help"])
    assert exc_info.value.code == 0
    help_text = stream.getvalue()
    assert "--approve-manual-review" in help_text
    assert "--approved-by" in help_text


def test_cli_campaign_approved_by_defaults_and_explicit(tmp_path: Path) -> None:
    """Verifica que --approve-manual-review asigne cli-operator o el valor explícito."""

    class RecordingManager:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        def process(
            self,
            campaign: Campaign,
            *,
            mode: str = "long_video",
            url: str | None = None,
            resume: bool = False,
            approve_manual_review: bool = False,
            approved_by: str | None = None,
        ) -> CampaignOutcome:
            _ = (mode, url, resume)
            self.calls.append(
                {
                    "approve_manual_review": approve_manual_review,
                    "approved_by": approved_by,
                }
            )
            return CampaignOutcome(
                campaign_id=campaign.campaign_id,
                archetype=Archetype.KNOWN,
                status=CampaignStatus.COMPLETED,
            )

    brief_file = tmp_path / "brief.txt"
    _ = brief_file.write_text("texto del brief", encoding="utf-8")

    # 1. Con --approve-manual-review sin --approved-by -> default "cli-operator"
    mgr1 = RecordingManager()
    code1 = main(
        [
            "campaign",
            str(brief_file),
            "--out",
            str(tmp_path / "out1"),
            "--approve-manual-review",
        ],
        manager=mgr1,
    )
    assert code1 == 0
    assert mgr1.calls[0]["approve_manual_review"] is True
    assert mgr1.calls[0]["approved_by"] == "cli-operator"

    # 2. Con --approve-manual-review y --approved-by "auditor-x"
    mgr2 = RecordingManager()
    code2 = main(
        [
            "campaign",
            str(brief_file),
            "--out",
            str(tmp_path / "out2"),
            "--approve-manual-review",
            "--approved-by",
            "auditor-x",
        ],
        manager=mgr2,
    )
    assert code2 == 0
    assert mgr2.calls[0]["approve_manual_review"] is True
    assert mgr2.calls[0]["approved_by"] == "auditor-x"


# ---------------------------------------------------------------------------
# 3. Obligaciones huérfanas proyectadas a PENDING_REVIEW
# ---------------------------------------------------------------------------


def test_contract_with_official_audio_triggers_pending_review(tmp_path: Path) -> None:
    contract = make_contract(
        manual_review=["audio.official_track"],
    ).model_copy(
        update={
            "official_audio": OfficialAudio(tiktok_url="https://tiktok.com/music/1"),
            "platforms": {
                Platform.TIKTOK: PlatformRules(
                    audio_rule=AudioRule.OFFICIAL_REQUIRED,
                    required_mentions=["@marca"],
                    required_hashtags=["#marca"],
                )
            },
        }
    )
    piece = _valid_piece(tmp_path, caption="mira @marca #marca")
    gate = Gate(probe=_fake_probe())
    result = gate.evaluate_piece(piece, contract=contract, assets=AssetRegistry(tmp_path))
    assert result.status is GateStatus.PENDING_REVIEW


def test_contract_with_attribution_triggers_pending_review(tmp_path: Path) -> None:
    contract = make_contract(
        manual_review=["attribution.required"],
    ).model_copy(
        update={
            "platforms": {
                Platform.TIKTOK: PlatformRules(
                    attribution=Attribution(type=AttributionType.TAG, value="@creador"),
                    required_mentions=["@marca"],
                    required_hashtags=["#marca"],
                )
            },
        }
    )
    piece = _valid_piece(tmp_path, caption="mira @marca #marca")
    gate = Gate(probe=_fake_probe())
    result = gate.evaluate_piece(piece, contract=contract, assets=AssetRegistry(tmp_path))
    assert result.status is GateStatus.PENDING_REVIEW


def test_contract_with_link_in_bio_triggers_pending_review(tmp_path: Path) -> None:
    contract = make_contract(
        manual_review=["link.in_bio"],
    ).model_copy(
        update={
            "platforms": {
                Platform.TIKTOK: PlatformRules(
                    link_rules=LinkRules(link_in_bio=True),
                    required_mentions=["@marca"],
                    required_hashtags=["#marca"],
                )
            },
        }
    )
    piece = _valid_piece(tmp_path, caption="mira @marca #marca")
    gate = Gate(probe=_fake_probe())
    result = gate.evaluate_piece(piece, contract=contract, assets=AssetRegistry(tmp_path))
    assert result.status is GateStatus.PENDING_REVIEW


# ---------------------------------------------------------------------------
# 4. Lote atómico en PARTIAL
# ---------------------------------------------------------------------------


def test_export_delivery_partial_batch_does_not_publish(tmp_path: Path) -> None:
    contract = make_contract(audio_rule="any")
    good = _valid_piece(tmp_path, caption="mira @marca #marca")
    bad_artifact = tmp_path / "bad.mp4"
    _ = bad_artifact.write_bytes(b"bad video")
    bad = Piece(
        piece_id="piece-02",
        platform=Platform.TIKTOK,
        caption="sin menciones ni nada",
        hashtags=(),
        subtitle_text=None,
        artifact_path=bad_artifact,
    )
    destination = tmp_path / "delivery"
    gate = Gate(probe=_fake_probe())
    assets = AssetRegistry(tmp_path)

    report = export_delivery(
        contract=contract,
        pieces=[good, bad],
        gate=gate,
        assets=assets,
        destination=destination,
    )

    assert report.status is ExportStatus.PARTIAL
    assert report.published_dir is None
    assert not destination.exists()


# ---------------------------------------------------------------------------
# 5. Procedencia real brief_text vs ContractDraft
# ---------------------------------------------------------------------------


def _cand[T](val: T, q: str, s: int) -> FieldCandidate[T]:
    return FieldCandidate(
        value=val,
        evidence=SourceEvidence(
            quote=q,
            start=s,
            end=s + len(q),
            location="§1",
        ),
        confidence=Confidence.EXPLICIT,
    )


def test_pipeline_and_cli_reject_altered_quotes_against_real_brief(tmp_path: Path) -> None:
    real_brief = "# Campaña Test\nDuración permitida: entre 10 y 30 segundos."
    brief_file = tmp_path / "brief.md"
    _ = brief_file.write_text(real_brief, encoding="utf-8")

    # Cita alterada: "entre 15 y 45 segundos" no existe en real_brief
    draft = ContractDraft(
        mode=_cand(Mode.LONG_VIDEO, "Campaña Test", 2),
        format=_cand(Format.VIDEO, "Campaña Test", 2),
        campaign_id=_cand("camp-test", "Campaña Test", 2),
        platforms={
            Platform.TIKTOK: PlatformDraft(
                duration=DurationDraft(
                    min_s=_cand(15, "entre 15 y 45 segundos", 15),
                    max_s=_cand(45, "entre 15 y 45 segundos", 15),
                )
            )
        },
    )

    # 1. resolve_contract directo con brief_text real debe fallar por cita alterada
    with pytest.raises(ProvenanceError):
        _ = resolve_contract(draft, registry=AssetRegistry(tmp_path), brief_text=real_brief)

    # 2. run_given_clips con contract_draft con cita alterada debe fallar con PipelineError
    class DummyModel(CampaignModel):
        model_version: str = "v1"

        @override
        def extract_contract(self, brief: str) -> ContractDraft:
            _ = brief
            return draft

        @override
        def write_caption(self, contract: Contract, piece: PieceContext) -> Caption:
            _ = (contract, piece)
            return Caption(caption="test", hashtags=())

    request = RunRequest(
        brief=real_brief,
        destination=tmp_path / "out",
        environment=detect_environment(SubprocessRunner()),
        model_version="v1",
        prompt_version="v1",
        caption_prompt_version="v1",
        contract_draft=draft,
    )
    with pytest.raises(PipelineError, match="procedencia"):
        _ = run_given_clips(
            model=DummyModel(),
            settings=Settings.from_root(tmp_path),
            request=request,
        )

    # 3. CLI campaign con --contract-draft y --brief con cita alterada debe retornar 1
    draft_file = tmp_path / "draft.json"
    _ = draft_file.write_text(draft.model_dump_json(), encoding="utf-8")

    code = main(
        [
            "campaign",
            "--brief",
            str(brief_file),
            "--contract-draft",
            str(draft_file),
            "--out",
            str(tmp_path / "out"),
        ]
    )
    assert code == 1


def test_vector_5_json_brief_rejected_in_run_and_campaign(tmp_path: Path) -> None:
    """Verifica que un brief JSON sea rechazado para evitar auto-validación."""
    json_brief = tmp_path / "brief.txt"
    _ = json_brief.write_text('{"campaign_id": "test", "format": "video"}', encoding="utf-8")

    # En campaign debe retornar 1
    campaign_code = main(["campaign", "--brief", str(json_brief), "--out", str(tmp_path / "out1")])
    assert campaign_code == 1

    # En run debe retornar 1
    run_code = main(["run", "--brief", str(json_brief), "--out", str(tmp_path / "out2")])
    assert run_code == 1


def test_audit_10_traceability_in_manual_approval(tmp_path: Path) -> None:
    """Verifica que al aprobar una pieza PENDING_REVIEW se registre trazabilidad auditable."""

    def manual_review(_context: GateContext) -> CheckOutcome:
        return CheckOutcome(status=CheckStatus.MANUAL_REVIEW, evidence={"reason": "visual"})

    probe = FakeProbe(info=make_media(duration_s=15.0, has_video=True, has_audio=True))
    gate = Gate(
        probe,
        validators={**DEFAULT_VALIDATORS, "watermark.full_video": manual_review},
    )
    contract = make_contract(
        hard=[
            "duration.min",
            "caption.required_hashtag",
            "caption.required_mention",
            "watermark.full_video",
        ],
        audio_rule="any",
    )

    piece = _valid_piece(tmp_path)
    dest = tmp_path / "delivery_trace"

    report = export_delivery(
        contract=contract,
        pieces=[piece],
        gate=gate,
        assets=AssetRegistry(tmp_path),
        destination=dest,
        approve_manual_review=True,
        approved_by="auditor-jane",
    )

    assert report.status is ExportStatus.EXPORTED
    assert "watermark.full_video" in report.manually_approved_rules
    assert report.approved_by == "auditor-jane"
    assert report.approved_at_utc is not None
    _ = datetime.fromisoformat(report.approved_at_utc)

    assert len(report.exported) == 1
    exp = report.exported[0]
    assert "watermark.full_video" in exp.manually_approved_rules
    assert exp.approved_by == "auditor-jane"
    assert exp.approved_at_utc == report.approved_at_utc

    # Verificar delivery_report.json persistido
    report_file = dest / "delivery_report.json"
    data = report_file.read_text(encoding="utf-8")
    assert '"manually_approved_rules"' in data
    assert '"watermark.full_video"' in data
    assert '"auditor-jane"' in data

    # Verificar también que write_manifest / RunManifest registre estos campos
    run_dir = tmp_path / "run_test"
    manifest = RunManifest(
        run_id="run-1",
        started_at=datetime.now(UTC),
        environment=detect_environment(SubprocessRunner()),
        manually_approved_rules=report.manually_approved_rules,
        approved_by=report.approved_by,
        approved_at_utc=report.approved_at_utc,
    )
    m_path = write_manifest(manifest, run_dir)
    loaded = read_manifest(m_path)
    assert loaded.manually_approved_rules == ("watermark.full_video",)
    assert loaded.approved_by == "auditor-jane"
    assert loaded.approved_at_utc == report.approved_at_utc


def test_audit_14_nvenc_fallback_in_reframe_and_subtitles(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verifica que reframe y subtitles usen fallback h264_nvenc -> libx264 ante fallos de NVENC."""
    calls: list[list[str]] = []

    def mock_subprocess_run(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        _ = kwargs
        calls.append(list(cmd))
        if "h264_nvenc" in cmd:
            return subprocess.CompletedProcess(
                cmd, returncode=1, stdout="", stderr="CUDA out of memory error"
            )
        return subprocess.CompletedProcess(cmd, returncode=0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", mock_subprocess_run)

    render = RenderConfig(nvenc_available=True)
    temp_reframe = tmp_path / "temp_reframe.mp4"
    _ = temp_reframe.write_bytes(b"temp")

    # 1. Probar fallback con ReframeError
    reframe_cmd = [
        "ffmpeg",
        "-i",
        "in.mp4",
        "-c:v",
        "h264_nvenc",
        "-preset",
        "p5",
        "-cq",
        "23",
        "-pix_fmt",
        "yuv420p",
        "out.mp4",
    ]
    run_ffmpeg_with_fallback(
        reframe_cmd, temporary=temp_reframe, render=render, error_cls=ReframeError
    )
    assert len(calls) == 2
    assert "h264_nvenc" in calls[0]
    assert "libx264" in calls[1]

    # 2. Probar fallback con SubtitleError
    calls.clear()
    temp_subtitles = tmp_path / "temp_sub.mp4"
    _ = temp_subtitles.write_bytes(b"temp")
    subtitles_cmd = [
        "ffmpeg",
        "-i",
        "in.mp4",
        "-c:v",
        "h264_nvenc",
        "-preset",
        "p5",
        "-cq",
        "23",
        "-pix_fmt",
        "yuv420p",
        "out.mp4",
    ]
    run_ffmpeg_with_fallback(
        subtitles_cmd, temporary=temp_subtitles, render=render, error_cls=SubtitleError
    )
    assert len(calls) == 2
    assert "h264_nvenc" in calls[0]
    assert "libx264" in calls[1]
