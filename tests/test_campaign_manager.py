"""Tests del controlador maestro de campañas (E3-PR3).

Clasificador, motor de propuestas y orquestadores se inyectan como fakes: cero
red, cero ffmpeg y cero subproceso real. Se verifica el enrutamiento por
arquetipo, la invariante de seguridad (el motor de video nunca se invoca si el
arquetipo no es ``KNOWN``), la propagación de errores y que importar el
controlador no carga el motor de video.
"""

import ast
import subprocess
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

import pytest
from pydantic import ValidationError

from kliptych import campaign_manager
from kliptych.campaign_manager import (
    CampaignManager,
    CampaignOutcome,
    SlideshowOrchestrator,
    VideoOrchestrator,
)
from kliptych.campaign_types import Campaign, CampaignStatus
from kliptych.contract import Contract, Segment
from kliptych.git_proposals import GitError, ProposalEngine, PullRequest
from kliptych.intelligence import Archetype, ArchetypeClassification
from kliptych.orchestrator import PipelineError, PipelineResult, SlideshowResult
from kliptych.runtime import ModelUnavailableError
from kliptych.segment import SegmentSelection
from tests.support import make_contract

_URL = "https://github.com/owner/repo/pull/7"
_VIDEO_URL = "https://example.com/video"
_IMAGE = Path("slide-01.jpg")

_FORBIDDEN_MODULES = (
    "subprocess",
    "kliptych.orchestrator",
    "kliptych.reframe",
    "kliptych.transcribe",
    "kliptych.moments",
    "kliptych.segment",
    "kliptych.subtitles",
    "kliptych.download",
    "kliptych.encoding",
    "kliptych.assembler",
)


def _private(name: str) -> object:
    return cast("object", getattr(campaign_manager, name))


_resolve_outcome_model = cast("Callable[[], None]", _private("_resolve_outcome_model"))


def _outcome(**fields: object) -> CampaignOutcome:
    _resolve_outcome_model()
    return CampaignOutcome.model_validate(fields)


def _classification(
    archetype: Archetype,
    *,
    variations: tuple[str, ...] = (),
) -> ArchetypeClassification:
    return ArchetypeClassification(
        archetype=archetype,
        rationale="justificación",
        variations=variations,
    )


@dataclass
class FakeClassifier:
    classification: ArchetypeClassification | None = None
    error: Exception | None = None
    calls: list[tuple[str, Contract]] = field(default_factory=list)

    def classify(self, brief: str, contract: Contract) -> ArchetypeClassification:
        self.calls.append((brief, contract))
        if self.error is not None:
            raise self.error
        assert self.classification is not None
        return self.classification


@dataclass
class FakeGitProvider:
    existing_files: dict[str, str] = field(default_factory=dict)
    fail_with: GitError | None = None
    branches: list[tuple[str, str]] = field(default_factory=list)
    written: list[dict[str, str]] = field(default_factory=list)
    opened: list[dict[str, object]] = field(default_factory=list)

    def _maybe_fail(self) -> None:
        if self.fail_with is not None:
            raise self.fail_with

    def create_branch(self, *, base: str, name: str) -> str:
        self._maybe_fail()
        self.branches.append((base, name))
        return name

    def read_file(self, *, branch: str, path: str) -> str | None:
        self._maybe_fail()
        _ = branch
        return self.existing_files.get(path)

    def write_file(self, *, branch: str, path: str, content: str, message: str) -> str:
        self._maybe_fail()
        self.written.append(
            {"branch": branch, "path": path, "content": content, "message": message}
        )
        return path

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
        self._maybe_fail()
        self.opened.append(
            {
                "branch": branch,
                "base": base,
                "title": title,
                "body": body,
                "campaign_id": campaign_id,
                "archetype": archetype,
            }
        )
        return PullRequest(
            url=_URL,
            branch=branch,
            title=title,
            body=body,
            campaign_id=campaign_id,
            archetype=archetype,
        )


@dataclass
class FakeVideoOrchestrator:
    pipeline_result: PipelineResult | None = None
    slideshow_result: SlideshowResult | None = None
    error: Exception | None = None
    long_video_calls: list[str] = field(default_factory=list)
    slideshow_calls: list[tuple[Path, ...]] = field(default_factory=list)

    def run_long_video(self, url: str, **kwargs: object) -> PipelineResult:
        _ = kwargs
        self.long_video_calls.append(url)
        if self.error is not None:
            raise self.error
        assert self.pipeline_result is not None
        return self.pipeline_result

    def run_slideshow(self, images: Sequence[Path], **kwargs: object) -> SlideshowResult:
        _ = kwargs
        self.slideshow_calls.append(tuple(images))
        if self.error is not None:
            raise self.error
        assert self.slideshow_result is not None
        return self.slideshow_result


@dataclass
class FakeSlideshowOrchestrator:
    slideshow_result: SlideshowResult | None = None
    error: Exception | None = None
    calls: list[tuple[Path, ...]] = field(default_factory=list)

    def run(self, images: Sequence[Path], **kwargs: object) -> SlideshowResult:
        _ = kwargs
        self.calls.append(tuple(images))
        if self.error is not None:
            raise self.error
        assert self.slideshow_result is not None
        return self.slideshow_result


def _selection() -> SegmentSelection:
    return SegmentSelection(segments=(Segment(start_s=0.0, end_s=1.0),), rationale="recorte")


def _pipeline_result() -> PipelineResult:
    return PipelineResult(
        source=Path("source.mp4"),
        transcript=None,
        moments=(),
        selection=_selection(),
        reframe=None,
        subtitles=None,
        final_video=Path("final.mp4"),
        cleaning=(),
    )


def _slideshow_result() -> SlideshowResult:
    return SlideshowResult(
        images=(_IMAGE,),
        slideshow_video=Path("slideshow.mp4"),
        final_video=Path("final.mp4"),
        subtitles=None,
        cleaning=(),
    )


def _campaign(*, with_contract: bool = True) -> Campaign:
    return Campaign(
        campaign_id="camp-01",
        brief="brief crudo",
        contract=make_contract() if with_contract else None,
    )


def _manager(
    *,
    classifier: FakeClassifier,
    provider: FakeGitProvider | None = None,
    video: FakeVideoOrchestrator | None = None,
    slideshow: FakeSlideshowOrchestrator | None = None,
) -> CampaignManager:
    return CampaignManager(
        classifier=classifier,
        proposal_engine=ProposalEngine(
            provider=FakeGitProvider() if provider is None else provider,
        ),
        video_orchestrator=video,
        slideshow_orchestrator=slideshow,
    )


def test_fakes_satisfy_protocols() -> None:
    video: VideoOrchestrator = FakeVideoOrchestrator()
    slideshow: SlideshowOrchestrator = FakeSlideshowOrchestrator()
    assert video is not None
    assert slideshow is not None


def test_outcome_is_frozen() -> None:
    outcome = _outcome(
        campaign_id="camp-01",
        archetype=Archetype.KNOWN,
        status=CampaignStatus.COMPLETED,
    )
    with pytest.raises(ValidationError):
        outcome.status = CampaignStatus.PENDING


def test_outcome_forbids_extra_fields() -> None:
    with pytest.raises(ValidationError, match="invented"):
        _ = _outcome(
            campaign_id="camp-01",
            archetype="KNOWN",
            status="COMPLETED",
            invented=True,
        )


def test_outcome_requires_non_empty_campaign_id() -> None:
    with pytest.raises(ValidationError, match="campaign_id"):
        _ = _outcome(campaign_id="", archetype="KNOWN", status="COMPLETED")


def test_known_routes_to_long_video_and_creates_no_proposal() -> None:
    classifier = FakeClassifier(classification=_classification(Archetype.KNOWN))
    video = FakeVideoOrchestrator(pipeline_result=_pipeline_result())
    provider = FakeGitProvider()
    manager = _manager(classifier=classifier, provider=provider, video=video)

    outcome = manager.process(_campaign(), mode="long_video", url=_VIDEO_URL)

    assert video.long_video_calls == [_VIDEO_URL]
    assert outcome.archetype is Archetype.KNOWN
    assert outcome.status is CampaignStatus.COMPLETED
    assert outcome.pipeline_result == _pipeline_result()
    assert outcome.slideshow_result is None
    assert outcome.pull_request is None
    assert outcome.error is None
    assert provider.branches == []
    assert provider.opened == []


def test_classifier_receives_brief_and_contract() -> None:
    campaign = _campaign()
    classifier = FakeClassifier(classification=_classification(Archetype.KNOWN))
    video = FakeVideoOrchestrator(pipeline_result=_pipeline_result())
    manager = _manager(classifier=classifier, video=video)

    _ = manager.process(campaign, mode="long_video", url=_VIDEO_URL)

    assert classifier.calls[0][0] == "brief crudo"
    assert classifier.calls[0][1] is campaign.contract


def test_known_with_variation_creates_pr_and_skips_video() -> None:
    classifier = FakeClassifier(
        classification=_classification(
            Archetype.KNOWN_WITH_VARIATION,
            variations=("duration.max=45",),
        )
    )
    video = FakeVideoOrchestrator(pipeline_result=_pipeline_result())
    provider = FakeGitProvider()
    manager = _manager(classifier=classifier, provider=provider, video=video)

    outcome = manager.process(_campaign(), mode="long_video", url=_VIDEO_URL)

    assert video.long_video_calls == []
    assert len(provider.opened) == 1
    assert provider.opened[0]["archetype"] is Archetype.KNOWN_WITH_VARIATION
    assert outcome.archetype is Archetype.KNOWN_WITH_VARIATION
    assert outcome.status is CampaignStatus.PENDING
    assert outcome.pull_request is not None
    assert outcome.pull_request.url == _URL
    assert outcome.pipeline_result is None
    assert outcome.slideshow_result is None
    assert outcome.error is None


def test_known_with_variation_registers_variation() -> None:
    classifier = FakeClassifier(
        classification=_classification(
            Archetype.KNOWN_WITH_VARIATION,
            variations=("duration.max=45",),
        )
    )
    video = FakeVideoOrchestrator(pipeline_result=_pipeline_result())
    provider = FakeGitProvider()
    manager = _manager(classifier=classifier, provider=provider, video=video)

    _ = manager.process(_campaign(), mode="long_video", url=_VIDEO_URL)

    assert provider.written[0]["path"] == "campaigns/variations.md"
    assert "duration.max=45" in provider.written[0]["content"]


def test_new_archetype_creates_pr_and_skips_video() -> None:
    classifier = FakeClassifier(classification=_classification(Archetype.NEW_ARCHETYPE))
    video = FakeVideoOrchestrator(pipeline_result=_pipeline_result())
    provider = FakeGitProvider()
    manager = _manager(classifier=classifier, provider=provider, video=video)

    outcome = manager.process(_campaign(), mode="long_video", url=_VIDEO_URL)

    assert video.long_video_calls == []
    assert len(provider.opened) == 1
    assert provider.opened[0]["archetype"] is Archetype.NEW_ARCHETYPE
    assert outcome.archetype is Archetype.NEW_ARCHETYPE
    assert outcome.status is CampaignStatus.MANUAL_REVIEW
    assert outcome.pull_request is not None
    assert outcome.pull_request.url == _URL
    assert outcome.pipeline_result is None
    assert outcome.error is None


def test_classification_failure_returns_error_without_video_or_proposal() -> None:
    classifier = FakeClassifier(error=ModelUnavailableError("sin backend"))
    video = FakeVideoOrchestrator(pipeline_result=_pipeline_result())
    provider = FakeGitProvider()
    manager = _manager(classifier=classifier, provider=provider, video=video)

    outcome = manager.process(_campaign(), mode="long_video", url=_VIDEO_URL)

    assert outcome.error is not None
    assert "clasificación" in outcome.error
    assert outcome.pipeline_result is None
    assert outcome.pull_request is None
    assert video.long_video_calls == []
    assert provider.opened == []


def test_video_failure_returns_error() -> None:
    classifier = FakeClassifier(classification=_classification(Archetype.KNOWN))
    video = FakeVideoOrchestrator(error=PipelineError("boom"))
    provider = FakeGitProvider()
    manager = _manager(classifier=classifier, provider=provider, video=video)

    outcome = manager.process(_campaign(), mode="long_video", url=_VIDEO_URL)

    assert outcome.error is not None
    assert "video" in outcome.error
    assert outcome.archetype is Archetype.KNOWN
    assert outcome.pipeline_result is None
    assert outcome.pull_request is None
    assert provider.opened == []


def test_proposal_failure_returns_error() -> None:
    classifier = FakeClassifier(classification=_classification(Archetype.NEW_ARCHETYPE))
    video = FakeVideoOrchestrator(pipeline_result=_pipeline_result())
    provider = FakeGitProvider(fail_with=GitError("boom"))
    manager = _manager(classifier=classifier, provider=provider, video=video)

    outcome = manager.process(_campaign(), mode="long_video", url=_VIDEO_URL)

    assert outcome.error is not None
    assert "propuesta" in outcome.error
    assert outcome.pull_request is None
    assert outcome.pipeline_result is None
    assert video.long_video_calls == []


def test_missing_contract_returns_error_without_calls() -> None:
    classifier = FakeClassifier(classification=_classification(Archetype.KNOWN))
    video = FakeVideoOrchestrator(pipeline_result=_pipeline_result())
    provider = FakeGitProvider()
    manager = _manager(classifier=classifier, provider=provider, video=video)

    outcome = manager.process(_campaign(with_contract=False), mode="long_video", url=_VIDEO_URL)

    assert outcome.error is not None
    assert "contrato" in outcome.error
    assert classifier.calls == []
    assert video.long_video_calls == []
    assert provider.opened == []


def test_missing_video_orchestrator_returns_error() -> None:
    classifier = FakeClassifier(classification=_classification(Archetype.KNOWN))
    manager = _manager(classifier=classifier)

    outcome = manager.process(_campaign(), mode="long_video", url=_VIDEO_URL)

    assert outcome.error is not None
    assert "orquestador" in outcome.error
    assert outcome.pipeline_result is None


def test_unsupported_mode_returns_error_without_video_call() -> None:
    classifier = FakeClassifier(classification=_classification(Archetype.KNOWN))
    video = FakeVideoOrchestrator(pipeline_result=_pipeline_result())
    manager = _manager(classifier=classifier, video=video)

    outcome = manager.process(_campaign(), mode="podcast", url=_VIDEO_URL)

    assert outcome.error is not None
    assert "modo" in outcome.error
    assert video.long_video_calls == []
    assert video.slideshow_calls == []


def test_slideshow_mode_uses_slideshow_orchestrator() -> None:
    classifier = FakeClassifier(classification=_classification(Archetype.KNOWN))
    video = FakeVideoOrchestrator(pipeline_result=_pipeline_result())
    slideshow = FakeSlideshowOrchestrator(slideshow_result=_slideshow_result())
    manager = _manager(classifier=classifier, video=video, slideshow=slideshow)

    outcome = manager.process(_campaign(), mode="slideshow", images=(_IMAGE,))

    assert slideshow.calls == [(_IMAGE,)]
    assert video.long_video_calls == []
    assert video.slideshow_calls == []
    assert outcome.slideshow_result == _slideshow_result()
    assert outcome.pipeline_result is None
    assert outcome.status is CampaignStatus.COMPLETED


def test_slideshow_mode_falls_back_to_video_orchestrator() -> None:
    classifier = FakeClassifier(classification=_classification(Archetype.KNOWN))
    video = FakeVideoOrchestrator(slideshow_result=_slideshow_result())
    manager = _manager(classifier=classifier, video=video)

    outcome = manager.process(_campaign(), mode="slideshow", images=(_IMAGE,))

    assert video.slideshow_calls == [(_IMAGE,)]
    assert video.long_video_calls == []
    assert outcome.slideshow_result == _slideshow_result()


def _module_level_imports() -> list[str]:
    source_path = campaign_manager.__file__
    assert source_path is not None
    tree = ast.parse(Path(source_path).read_text(encoding="utf-8"))
    modules: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            modules.append(node.module)
    return modules


def test_no_video_engine_imported_at_module_level() -> None:
    modules = _module_level_imports()
    forbidden = [
        module
        for module in modules
        if module in _FORBIDDEN_MODULES
        or module.startswith("kliptych.orchestrator")
        or "ffmpeg" in module.lower()
    ]
    assert forbidden == []


def test_importing_manager_does_not_load_video_engine() -> None:
    code = (
        "import sys; import kliptych.campaign_manager; "
        "print('kliptych.orchestrator' in sys.modules)"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "False"
