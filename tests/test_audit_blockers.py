"""Tests adversariales para los 5 blockers de auditoría.

1. Mandatory Gate in Default Path: no retorna COMPLETED sin evaluar Gate.
2. CLI Exit Code & --approve-manual-review: CLI retorna 1 en BLOCKED.
3. Proyección de obligaciones huérfanas: disparan PENDING_REVIEW.
4. Lote atómico en PARTIAL: no publica si no es EXPORTED.
5. Procedencia real brief_text vs ContractDraft: rechazo de citas alteradas.
"""

from collections.abc import Sequence
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
from kliptych.environment import SubprocessRunner, detect_environment
from kliptych.exporter import DeliveryReport, ExportStatus, export_delivery
from kliptych.gate import Gate, GateStatus, Piece
from kliptych.git_proposals import GitHubCliProvider, ProposalEngine, PullRequest
from kliptych.intelligence import Archetype, ArchetypeClassification, CampaignClassifier
from kliptych.orchestrator import PipelineResult, SlideshowResult
from kliptych.pipeline import PipelineError, RunRequest, run_given_clips
from kliptych.resolver import ProvenanceError, resolve_contract
from kliptych.runtime import CampaignModel, Caption, PieceContext
from kliptych.segment import SegmentSelection
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
        ) -> CampaignOutcome:
            _ = (campaign, mode, url, resume, approve_manual_review)
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


def test_cli_run_help_exposes_approve_manual_review() -> None:
    stream = StringIO()
    with pytest.raises(SystemExit) as exc_info, patch("sys.stdout", stream):
        _ = main(["run", "--help"])
    assert exc_info.value.code == 0
    help_text = stream.getvalue()
    assert "--approve-manual-review" in help_text


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
    contract = make_contract()
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
