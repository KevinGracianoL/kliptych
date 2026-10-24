"""Behavioral tests for the Track B1 Studio project store."""

from pathlib import Path
import uuid
from typing import Protocol, cast

import pytest

from kliptych.contract.enums import Layout
from kliptych.contract.schema import Band, LayoutSpec, NormalizedRect
from kliptych.gate.models import MediaInfo
from kliptych.hashing import sha256_bytes, sha256_file
from kliptych.studio import project_store as _project_store
from kliptych.studio.models import (
    AnalysisCandidate,
    AnalysisDocument,
    AnalysisProvenance,
    AnalysisSource,
)
from kliptych.studio.project_models import (
    LayoutPosition,
    ProjectDocument,
    Variant,
    VariantOrigin,
    VariantRange,
    create_initial_variant,
)
from kliptych.studio.project_store import (
    ProjectStore,
    ProjectStoreError,
    open_project_store,
)

_SOURCE_SHA256 = sha256_bytes(b"source")


def _analysis(
    *, project_id: str = "a" * 32, source_sha256: str = _SOURCE_SHA256
) -> AnalysisDocument:
    return AnalysisDocument(
        schema_version="1",
        project_id=project_id,
        project_name="My clip",
        source=AnalysisSource(url="https://example.com/video", sha256=source_sha256),
        provenance=AnalysisProvenance(transcription_version="1", segmenter_version="1"),
        candidates=(),
    )


def _candidate() -> AnalysisCandidate:
    return AnalysisCandidate(
        candidate_id="a" * 64,
        recommended_start_ms=10_000,
        recommended_end_ms=30_000,
        score=0.9,
        excerpt="Candidate",
    )


def _private(name: str) -> object:
    return cast("object", getattr(_project_store, name))


class _BackupNameMatcher(Protocol):
    def __call__(self, name: str, *, prefix: str) -> bool:
        """Return whether a backup name belongs to the requested project."""
        ...


def _write_then_fail(store: ProjectStore) -> None:
    message = "boom"
    with store.stage() as stage:
        stage.write_analysis(_analysis(project_id="c" * 32))
        stage.write_text("source.mp4", "second source")
        raise RuntimeError(message)


def _commit_incomplete_stage(store: ProjectStore) -> None:
    with store.stage() as stage:
        stage.write_analysis(_analysis())


def _commit_reanalysis(store: ProjectStore, *, source: str = "second source") -> None:
    with store.stage() as stage:
        stage.write_analysis(
            _analysis(project_id="c" * 32, source_sha256=sha256_bytes(source.encode()))
        )
        stage.write_text("source.mp4", source)
        stage.write_text("transcript.json", "second transcript")
        stage.write_text("candidates.json", "second candidates")


def _publish_changed_source_reanalysis(store: ProjectStore, analysis: AnalysisDocument) -> None:
    with store.stage() as stage:
        stage.write_analysis(analysis)
        stage.write_text("source.mp4", "second source")
        stage.write_text("transcript.json", "second transcript")
        stage.write_text("candidates.json", "second candidates")


def _write_editable_state(project_dir: Path) -> None:
    _ = (project_dir / "project.json").write_text('{"schema_version":"1"}', encoding="utf-8")
    watermark_dir = project_dir / "assets" / "watermark"
    watermark_dir.mkdir(parents=True)
    _ = (watermark_dir / "asset.png").write_bytes(b"png")
    preview_dir = project_dir / "previews" / "variant" / "position"
    preview_dir.mkdir(parents=True)
    _ = (preview_dir / "preview.jpg").write_bytes(b"jpeg")


def _project_document(project_dir: Path, analysis: AnalysisDocument) -> ProjectDocument:
    return ProjectDocument(
        schema_version="1",
        project_id=analysis.project_id,
        analysis_sha256=sha256_file(project_dir / "analysis.json"),
        transcript_sha256=sha256_file(project_dir / "transcript.json"),
        source_sha256=analysis.source.sha256,
        source_duration_ms=60_000,
        variants=(),
    )


class _Probe:
    def __init__(self, duration_s: float | None) -> None:
        self.duration_s: float | None = duration_s

    def probe(self, path: Path) -> MediaInfo:
        assert path.name == "source.mp4"
        return MediaInfo(
            format_name="mp4",
            has_video=True,
            has_audio=True,
            duration_s=self.duration_s,
            width=1920,
            height=1080,
        )


def test_new_project_publishes_complete_analysis_atomically(tmp_path: Path) -> None:
    project_dir = tmp_path / "project"

    store = open_project_store(project_dir)
    with store.stage() as stage:
        stage.write_analysis(_analysis())
        stage.write_text("source.mp4", "source")
        stage.write_text("transcript.json", "transcript")
        stage.write_text("candidates.json", "candidates")

    published = AnalysisDocument.model_validate_json(
        (project_dir / "analysis.json").read_text(encoding="utf-8")
    )
    assert published == _analysis()
    assert (project_dir / "source.mp4").read_text(encoding="utf-8") == "source"
    assert not list(tmp_path.glob(".analysis-staging-*"))


def test_new_project_creates_missing_parent_directories(tmp_path: Path) -> None:
    project_dir = tmp_path / "missing" / "parents" / "project"

    store = open_project_store(project_dir)
    with store.stage() as stage:
        stage.write_analysis(_analysis())
        stage.write_text("source.mp4", "source")
        stage.write_text("transcript.json", "transcript")
        stage.write_text("candidates.json", "candidates")

    assert (project_dir / "analysis.json").is_file()


def test_new_project_with_file_parent_returns_domain_error(tmp_path: Path) -> None:
    parent = tmp_path / "not-a-directory"
    _ = parent.write_text("file", encoding="utf-8")

    with pytest.raises(ProjectStoreError, match="project_directory_invalid"):
        _ = open_project_store(parent / "project")


def test_store_writes_and_reads_validated_project_document_atomically(tmp_path: Path) -> None:
    project_dir = tmp_path / "project"
    store = open_project_store(project_dir)
    analysis = _analysis()
    with store.stage() as stage:
        stage.write_analysis(analysis)
        stage.write_text("source.mp4", "source")
        stage.write_text("transcript.json", "transcript")
        stage.write_text("candidates.json", "candidates")
    store = open_project_store(project_dir)
    document = store.initialize_project(probe=_Probe(60.0))

    store.write_project(document)

    store.write_project(document.model_copy(update={"variants": ()}))

    assert store.read_project() == document
    assert (
        ProjectDocument.model_validate_json(
            (project_dir / "project.json").read_text(encoding="utf-8")
        )
        == document
    )


def test_store_returns_none_when_project_document_does_not_exist(tmp_path: Path) -> None:
    store = open_project_store(tmp_path / "project")

    assert store.read_project() is None


def test_store_rejects_link_like_missing_project_document_without_replacing_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project_dir = tmp_path / "project"
    store = open_project_store(project_dir)
    analysis = _analysis()
    with store.stage() as stage:
        stage.write_analysis(analysis)
        stage.write_text("source.mp4", "source")
        stage.write_text("transcript.json", "transcript")
        stage.write_text("candidates.json", "candidates")
    project_path = project_dir / "project.json"

    def fake_link_like(path: Path) -> bool:
        return path == project_path

    monkeypatch.setattr("kliptych.studio.project_store._is_link_like", fake_link_like)

    with pytest.raises(ProjectStoreError, match="regular file"):
        _ = open_project_store(project_dir).initialize_project(probe=_Probe(60.0))

    assert not project_path.exists()


def test_store_rejects_project_document_without_completed_analysis(tmp_path: Path) -> None:
    store = open_project_store(tmp_path / "project")
    document = ProjectDocument(
        schema_version="1",
        project_id="unbound",
        analysis_sha256="a" * 64,
        transcript_sha256="b" * 64,
        source_sha256="c" * 64,
        source_duration_ms=60_000,
        variants=(),
    )

    with pytest.raises(ProjectStoreError, match="requires a completed analysis"):
        store.write_project(document)


def test_store_rejects_normal_write_that_changes_measured_source_duration(tmp_path: Path) -> None:
    project_dir = tmp_path / "project"
    store = open_project_store(project_dir)
    analysis = _analysis()
    with store.stage() as stage:
        stage.write_analysis(analysis)
        stage.write_text("source.mp4", "source")
        stage.write_text("transcript.json", "transcript")
        stage.write_text("candidates.json", "candidates")
    initialized = open_project_store(project_dir).initialize_project(probe=_Probe(60.0))
    previous = (project_dir / "project.json").read_bytes()

    with pytest.raises(ProjectStoreError, match="evidence fields"):
        open_project_store(project_dir).write_project(
            initialized.model_copy(update={"source_duration_ms": 1})
        )

    assert (project_dir / "project.json").read_bytes() == previous


def test_store_rejects_normal_write_when_source_artifact_becomes_link_like(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project_dir = tmp_path / "project"
    store = open_project_store(project_dir)
    analysis = _analysis()
    with store.stage() as stage:
        stage.write_analysis(analysis)
        stage.write_text("source.mp4", "source")
        stage.write_text("transcript.json", "transcript")
        stage.write_text("candidates.json", "candidates")
    initialized = open_project_store(project_dir).initialize_project(probe=_Probe(60.0))
    previous = (project_dir / "project.json").read_bytes()
    source_path = project_dir / "source.mp4"

    def fake_link_like(path: Path) -> bool:
        return path == source_path

    monkeypatch.setattr("kliptych.studio.project_store._is_link_like", fake_link_like)

    with pytest.raises(ProjectStoreError, match="regular source"):
        open_project_store(project_dir).write_project(initialized)

    assert (project_dir / "project.json").read_bytes() == previous


def test_store_rejects_unvalidated_model_copy_without_writing(tmp_path: Path) -> None:
    project_dir = tmp_path / "project"
    store = open_project_store(project_dir)
    analysis = _analysis()
    with store.stage() as stage:
        stage.write_analysis(analysis)
        stage.write_text("source.mp4", "source")
        stage.write_text("transcript.json", "transcript")
        stage.write_text("candidates.json", "candidates")
    initialized = open_project_store(project_dir).initialize_project(probe=_Probe(60.0))
    previous = (project_dir / "project.json").read_bytes()
    corrupted = initialized.model_copy(update={"variants": ("not-a-variant",)})

    with pytest.raises(ProjectStoreError, match="document is invalid"):
        open_project_store(project_dir).write_project(corrupted)

    assert (project_dir / "project.json").read_bytes() == previous


def test_store_rejects_active_variant_with_forged_b1_origin_without_writing(
    tmp_path: Path,
) -> None:
    project_dir = tmp_path / "project"
    candidate = _candidate()
    analysis = _analysis().model_copy(update={"candidates": (candidate,)})
    store = open_project_store(project_dir)
    with store.stage() as stage:
        stage.write_analysis(analysis)
        stage.write_text("source.mp4", "source")
        stage.write_text("transcript.json", "transcript")
        stage.write_text("candidates.json", "candidates")
    store = open_project_store(project_dir)
    project = store.initialize_project(probe=_Probe(60.0))
    valid = store.create_variant(candidate_id=candidate.candidate_id, name="Variant 1")
    previous = (project_dir / "project.json").read_bytes()
    forged_origin = VariantOrigin(
        source_candidate_id="b" * 64,
        analysis_sha256="c" * 64,
        transcript_sha256="d" * 64,
        source_sha256="e" * 64,
    )
    forged_variant = valid.variants[0].model_copy(update={"origin": forged_origin})
    forged = project.model_copy(update={"variants": (forged_variant,)})

    with pytest.raises(ProjectStoreError, match=r"(?i)active variant origin"):
        store.write_project(forged)

    assert (project_dir / "project.json").read_bytes() == previous


@pytest.mark.parametrize(
    ("field", "value"),
    [("origin", "invalid-origin"), ("layout_timeline", ("invalid-position",))],
)
def test_store_rejects_nested_unvalidated_model_copy_without_writing(
    tmp_path: Path, field: str, value: object
) -> None:
    project_dir = tmp_path / "project"
    candidate = _candidate()
    analysis = _analysis().model_copy(update={"candidates": (candidate,)})
    store = open_project_store(project_dir)
    with store.stage() as stage:
        stage.write_analysis(analysis)
        stage.write_text("source.mp4", "source")
        stage.write_text("transcript.json", "transcript")
        stage.write_text("candidates.json", "candidates")
    store = open_project_store(project_dir)
    _ = store.initialize_project(probe=_Probe(60.0))
    project = store.create_variant(candidate_id=candidate.candidate_id, name="Variant 1")
    previous = (project_dir / "project.json").read_bytes()
    variant = project.variants[0].model_copy(update={field: value})
    corrupted = project.model_copy(update={"variants": (variant,)})

    with pytest.raises(ProjectStoreError, match="document is invalid"):
        store.write_project(corrupted)

    assert (project_dir / "project.json").read_bytes() == previous


def test_store_rejects_normal_write_before_measured_initialization(tmp_path: Path) -> None:
    project_dir = tmp_path / "project"
    store = open_project_store(project_dir)
    analysis = _analysis()
    with store.stage() as stage:
        stage.write_analysis(analysis)
        stage.write_text("source.mp4", "source")
        stage.write_text("transcript.json", "transcript")
        stage.write_text("candidates.json", "candidates")
    document = _project_document(project_dir, analysis)

    with pytest.raises(ProjectStoreError, match="must be initialized"):
        open_project_store(project_dir).write_project(document)

    assert not (project_dir / "project.json").exists()


def test_store_rejects_analysis_with_source_hash_not_matching_source_file(tmp_path: Path) -> None:
    project_dir = tmp_path / "project"
    store = open_project_store(project_dir)
    with store.stage() as stage:
        stage.write_analysis(_analysis(source_sha256="a" * 64))
        stage.write_text("source.mp4", "source")
        stage.write_text("transcript.json", "transcript")
        stage.write_text("candidates.json", "candidates")

    with pytest.raises(ProjectStoreError, match="source hash"):
        _ = open_project_store(project_dir).initialize_project(probe=_Probe(60.0))


def test_store_initializes_project_from_b1_artifacts_and_measured_duration(tmp_path: Path) -> None:
    project_dir = tmp_path / "project"
    store = open_project_store(project_dir)
    analysis = _analysis()
    with store.stage() as stage:
        stage.write_analysis(analysis)
        stage.write_text("source.mp4", "source")
        stage.write_text("transcript.json", "transcript")
        stage.write_text("candidates.json", "candidates")
    store = open_project_store(project_dir)

    document = store.initialize_project(probe=_Probe(21.7836))

    assert document.project_id == analysis.project_id
    assert document.source_duration_ms == 21_784
    assert document.analysis_sha256 == sha256_file(project_dir / "analysis.json")
    assert document.transcript_sha256 == sha256_file(project_dir / "transcript.json")
    assert store.read_project() == document


@pytest.mark.parametrize("duration_s", [None, 0.0, 0.0004, -1.0, float("inf")])
def test_store_rejects_unusable_measured_source_duration(
    tmp_path: Path, duration_s: float | None
) -> None:
    project_dir = tmp_path / "project"
    store = open_project_store(project_dir)
    with store.stage() as stage:
        stage.write_analysis(_analysis())
        stage.write_text("source.mp4", "source")
        stage.write_text("transcript.json", "transcript")
        stage.write_text("candidates.json", "candidates")

    with pytest.raises(ProjectStoreError, match="source duration"):
        _ = open_project_store(project_dir).initialize_project(probe=_Probe(duration_s))


def test_store_rebinds_current_b1_evidence_and_marks_variants_for_review(tmp_path: Path) -> None:
    project_dir = tmp_path / "project"
    store = open_project_store(project_dir)
    first_analysis = _analysis(source_sha256=sha256_bytes(b"first source")).model_copy(
        update={"candidates": (_candidate(),)}
    )
    with store.stage() as stage:
        stage.write_analysis(first_analysis)
        stage.write_text("source.mp4", "first source")
        stage.write_text("transcript.json", "first transcript")
        stage.write_text("candidates.json", "first candidates")
    project = open_project_store(project_dir).initialize_project(probe=_Probe(60.0))
    variant = create_initial_variant(
        name="Variant 1",
        selection=VariantRange(start_ms=10_000, end_ms=30_000, source_duration_ms=60_000),
        origin=VariantOrigin(
            source_candidate_id="a" * 64,
            analysis_sha256=project.analysis_sha256,
            transcript_sha256=project.transcript_sha256,
            source_sha256=project.source_sha256,
        ),
    )
    open_project_store(project_dir).write_project(
        project.model_copy(update={"variants": (variant,)})
    )

    reanalysis = open_project_store(project_dir)
    with reanalysis.stage() as stage:
        stage.write_analysis(
            _analysis(
                project_id=first_analysis.project_id,
                source_sha256=sha256_bytes(b"first source"),
            ).model_copy(update={"candidates": (_candidate(),)})
        )
        stage.write_text("source.mp4", "first source")
        stage.write_text("transcript.json", "second transcript")
        stage.write_text("candidates.json", "second candidates")

    rebound = open_project_store(project_dir).rebind_project(probe=_Probe(70.0))

    assert rebound.source_duration_ms == 70_000
    assert rebound.variants[0].status.value == "requires_review"
    assert rebound.variants[0].origin.analysis_sha256 == project.analysis_sha256
    assert rebound.transcript_sha256 != project.transcript_sha256


def test_reanalysis_with_changed_source_requires_explicit_user_action(
    tmp_path: Path,
) -> None:
    project_dir = tmp_path / "project"
    first_analysis = _analysis(source_sha256=sha256_bytes(b"first source")).model_copy(
        update={"candidates": (_candidate(),)}
    )
    store = open_project_store(project_dir)
    with store.stage() as stage:
        stage.write_analysis(first_analysis)
        stage.write_text("source.mp4", "first source")
        stage.write_text("transcript.json", "first transcript")
        stage.write_text("candidates.json", "first candidates")
    project = open_project_store(project_dir).initialize_project(probe=_Probe(60.0))
    variant = create_initial_variant(
        name="Variant 1",
        selection=VariantRange(start_ms=10_000, end_ms=30_000, source_duration_ms=60_000),
        origin=VariantOrigin(
            source_candidate_id="a" * 64,
            analysis_sha256=project.analysis_sha256,
            transcript_sha256=project.transcript_sha256,
            source_sha256=project.source_sha256,
        ),
    )
    open_project_store(project_dir).write_project(
        project.model_copy(update={"variants": (variant,)})
    )
    changed_analysis = _analysis(
        project_id=first_analysis.project_id,
        source_sha256=sha256_bytes(b"second source"),
    )
    with pytest.raises(ProjectStoreError, match="source changed"):
        _publish_changed_source_reanalysis(open_project_store(project_dir), changed_analysis)

    assert (project_dir / "source.mp4").read_text(encoding="utf-8") == "first source"
    assert open_project_store(project_dir).read_project() == project.model_copy(
        update={"variants": (variant,)}
    )
    assert not list(tmp_path.glob(".analysis-backup-*"))
    assert not list(tmp_path.glob(".analysis-staging-*"))


def test_changed_source_does_not_migrate_corrupt_editable_state(tmp_path: Path) -> None:
    project_dir = tmp_path / "project"
    first_analysis = _analysis(source_sha256=sha256_bytes(b"first source"))
    store = open_project_store(project_dir)
    with store.stage() as stage:
        stage.write_analysis(first_analysis)
        stage.write_text("source.mp4", "first source")
        stage.write_text("transcript.json", "first transcript")
        stage.write_text("candidates.json", "first candidates")
    _ = (project_dir / "project.json").write_text("{not valid json}", encoding="utf-8")
    assets = project_dir / "assets"
    assets.mkdir()
    _ = (assets / "asset.txt").write_text("asset", encoding="utf-8")
    previews = project_dir / "previews"
    previews.mkdir()
    _ = (previews / "preview.txt").write_text("preview", encoding="utf-8")
    changed_analysis = _analysis(
        project_id=first_analysis.project_id,
        source_sha256=sha256_bytes(b"second source"),
    )

    with pytest.raises(ProjectStoreError, match="source changed"):
        _publish_changed_source_reanalysis(open_project_store(project_dir), changed_analysis)

    assert (project_dir / "source.mp4").read_text(encoding="utf-8") == "first source"
    assert (project_dir / "project.json").read_text(encoding="utf-8") == "{not valid json}"
    assert (assets / "asset.txt").read_text(encoding="utf-8") == "asset"
    assert (previews / "preview.txt").read_text(encoding="utf-8") == "preview"
    assert not list(tmp_path.glob(".analysis-backup-*"))
    assert not list(tmp_path.glob(".analysis-staging-*"))


def test_changed_source_does_not_migrate_editable_state_when_active_source_is_missing(
    tmp_path: Path,
) -> None:
    project_dir = tmp_path / "project"
    first_analysis = _analysis(source_sha256=sha256_bytes(b"first source"))
    store = open_project_store(project_dir)
    with store.stage() as stage:
        stage.write_analysis(first_analysis)
        stage.write_text("source.mp4", "first source")
        stage.write_text("transcript.json", "first transcript")
        stage.write_text("candidates.json", "first candidates")
    _ = (project_dir / "project.json").write_text("{not valid json}", encoding="utf-8")
    assets = project_dir / "assets"
    assets.mkdir()
    _ = (assets / "asset.txt").write_text("asset", encoding="utf-8")
    (project_dir / "source.mp4").unlink()
    changed_analysis = _analysis(
        project_id=first_analysis.project_id,
        source_sha256=sha256_bytes(b"second source"),
    )

    with pytest.raises(ProjectStoreError, match="source is unavailable"):
        _publish_changed_source_reanalysis(open_project_store(project_dir), changed_analysis)

    assert not (project_dir / "source.mp4").exists()
    assert (project_dir / "project.json").read_text(encoding="utf-8") == "{not valid json}"
    assert (assets / "asset.txt").read_text(encoding="utf-8") == "asset"
    assert not list(tmp_path.glob(".analysis-backup-*"))
    assert not list(tmp_path.glob(".analysis-staging-*"))


def test_store_rebind_rejects_changed_b1_project_id_without_overwriting_project(
    tmp_path: Path,
) -> None:
    project_dir = tmp_path / "project"
    store = open_project_store(project_dir)
    first_analysis = _analysis(source_sha256=sha256_bytes(b"first source"))
    with store.stage() as stage:
        stage.write_analysis(first_analysis)
        stage.write_text("source.mp4", "first source")
        stage.write_text("transcript.json", "first transcript")
        stage.write_text("candidates.json", "first candidates")
    original = open_project_store(project_dir).initialize_project(probe=_Probe(60.0))
    original_bytes = (project_dir / "project.json").read_bytes()
    with open_project_store(project_dir).stage() as stage:
        stage.write_analysis(
            _analysis(project_id="different-project", source_sha256=sha256_bytes(b"first source"))
        )
        stage.write_text("source.mp4", "first source")
        stage.write_text("transcript.json", "second transcript")
        stage.write_text("candidates.json", "second candidates")

    with pytest.raises(ProjectStoreError, match="different analysis project"):
        _ = open_project_store(project_dir).rebind_project(probe=_Probe(60.0))

    assert (project_dir / "project.json").read_bytes() == original_bytes
    assert original.project_id != "different-project"


def test_store_rebind_rejects_inconsistent_current_source_evidence_without_overwriting(
    tmp_path: Path,
) -> None:
    project_dir = tmp_path / "project"
    first_analysis = _analysis(source_sha256=sha256_bytes(b"first source"))
    store = open_project_store(project_dir)
    with store.stage() as stage:
        stage.write_analysis(first_analysis)
        stage.write_text("source.mp4", "first source")
        stage.write_text("transcript.json", "first transcript")
        stage.write_text("candidates.json", "first candidates")
    original = open_project_store(project_dir).initialize_project(probe=_Probe(60.0))
    original_bytes = (project_dir / "project.json").read_bytes()
    with open_project_store(project_dir).stage() as stage:
        stage.write_analysis(
            _analysis(project_id=first_analysis.project_id, source_sha256="a" * 64)
        )
        stage.write_text("source.mp4", "first source")
        stage.write_text("transcript.json", "second transcript")
        stage.write_text("candidates.json", "second candidates")

    with pytest.raises(ProjectStoreError, match="source hash"):
        _ = open_project_store(project_dir).rebind_project(probe=_Probe(60.0))

    assert (project_dir / "project.json").read_bytes() == original_bytes
    assert original.source_sha256 == sha256_bytes(b"first source")


def test_store_rebind_rejects_variant_that_exceeds_new_source_duration(tmp_path: Path) -> None:
    project_dir = tmp_path / "project"
    store = open_project_store(project_dir)
    analysis = _analysis(source_sha256=sha256_bytes(b"first source")).model_copy(
        update={"candidates": (_candidate(),)}
    )
    with store.stage() as stage:
        stage.write_analysis(analysis)
        stage.write_text("source.mp4", "first source")
        stage.write_text("transcript.json", "first transcript")
        stage.write_text("candidates.json", "first candidates")
    project = open_project_store(project_dir).initialize_project(probe=_Probe(60.0))
    variant = create_initial_variant(
        name="Long variant",
        selection=VariantRange(start_ms=10_000, end_ms=50_000, source_duration_ms=60_000),
        origin=VariantOrigin(
            source_candidate_id="a" * 64,
            analysis_sha256=project.analysis_sha256,
            transcript_sha256=project.transcript_sha256,
            source_sha256=project.source_sha256,
        ),
    )
    open_project_store(project_dir).write_project(
        project.model_copy(update={"variants": (variant,)})
    )

    rebound = open_project_store(project_dir).rebind_project(probe=_Probe(40.0))

    assert rebound.source_duration_ms == 40_000
    assert rebound.variants[0].status.value == "requires_review"
    assert rebound.variants[0].end_ms == 50_000


def test_store_creates_variant_from_active_analysis_candidate(tmp_path: Path) -> None:
    project_dir = tmp_path / "project"
    store = open_project_store(project_dir)
    candidate = AnalysisCandidate(
        candidate_id="a" * 64,
        recommended_start_ms=10_000,
        recommended_end_ms=30_000,
        score=0.9,
        excerpt="Candidate text",
    )
    analysis = _analysis().model_copy(update={"candidates": (candidate,)})
    with store.stage() as stage:
        stage.write_analysis(analysis)
        stage.write_text("source.mp4", "source")
        stage.write_text("transcript.json", "transcript")
        stage.write_text("candidates.json", "candidates")
    store = open_project_store(project_dir)
    _ = store.initialize_project(probe=_Probe(60.0))

    project = store.create_variant(candidate_id=candidate.candidate_id, name="Variant 1")

    assert len(project.variants) == 1
    variant = project.variants[0]
    assert variant.origin.source_candidate_id == candidate.candidate_id
    assert (variant.start_ms, variant.end_ms) == (10_000, 30_000)
    assert store.read_project() == project


def test_reused_store_reads_current_analysis_after_reanalysis(tmp_path: Path) -> None:
    project_dir = tmp_path / "project"
    store = open_project_store(project_dir)
    first_candidate = AnalysisCandidate(
        candidate_id="a" * 64,
        recommended_start_ms=10_000,
        recommended_end_ms=30_000,
        score=0.9,
        excerpt="First candidate",
    )
    first_analysis = _analysis(source_sha256=sha256_bytes(b"first source")).model_copy(
        update={"candidates": (first_candidate,)}
    )
    with store.stage() as stage:
        stage.write_analysis(first_analysis)
        stage.write_text("source.mp4", "first source")
        stage.write_text("transcript.json", "first transcript")
        stage.write_text("candidates.json", "first candidates")
    _ = store.initialize_project(probe=_Probe(60.0))
    second_candidate = AnalysisCandidate(
        candidate_id="b" * 64,
        recommended_start_ms=20_000,
        recommended_end_ms=40_000,
        score=0.9,
        excerpt="Second candidate",
    )
    second_analysis = _analysis(
        project_id=first_analysis.project_id,
        source_sha256=sha256_bytes(b"first source"),
    ).model_copy(update={"candidates": (second_candidate,)})
    with store.stage() as stage:
        stage.write_analysis(second_analysis)
        stage.write_text("source.mp4", "first source")
        stage.write_text("transcript.json", "second transcript")
        stage.write_text("candidates.json", "second candidates")

    rebound = store.rebind_project(probe=_Probe(60.0))

    with pytest.raises(ProjectStoreError, match="active analysis"):
        _ = store.create_variant(candidate_id=first_candidate.candidate_id, name="Old candidate")
    current = store.create_variant(
        candidate_id=second_candidate.candidate_id, name="Current candidate"
    )

    assert rebound.variants == ()
    assert current.variants[0].origin.source_candidate_id == second_candidate.candidate_id


def test_store_rejects_variant_for_absent_active_candidate(tmp_path: Path) -> None:
    project_dir = tmp_path / "project"
    store = open_project_store(project_dir)
    with store.stage() as stage:
        stage.write_analysis(_analysis())
        stage.write_text("source.mp4", "source")
        stage.write_text("transcript.json", "transcript")
        stage.write_text("candidates.json", "candidates")
    store = open_project_store(project_dir)
    _ = store.initialize_project(probe=_Probe(60.0))

    with pytest.raises(ProjectStoreError, match="active analysis"):
        _ = store.create_variant(candidate_id="a" * 64, name="Variant 1")


def test_store_rejects_candidate_range_outside_measured_duration_without_writing(
    tmp_path: Path,
) -> None:
    project_dir = tmp_path / "project"
    candidate = AnalysisCandidate(
        candidate_id="a" * 64,
        recommended_start_ms=50_000,
        recommended_end_ms=70_000,
        score=0.9,
        excerpt="Too long",
    )
    analysis = _analysis().model_copy(update={"candidates": (candidate,)})
    store = open_project_store(project_dir)
    with store.stage() as stage:
        stage.write_analysis(analysis)
        stage.write_text("source.mp4", "source")
        stage.write_text("transcript.json", "transcript")
        stage.write_text("candidates.json", "candidates")
    store = open_project_store(project_dir)
    _ = store.initialize_project(probe=_Probe(60.0))
    previous = (project_dir / "project.json").read_bytes()

    with pytest.raises(ProjectStoreError, match="does not fit"):
        _ = store.create_variant(candidate_id=candidate.candidate_id, name="Too long")

    assert (project_dir / "project.json").read_bytes() == previous


def test_existing_non_project_directory_is_rejected_without_writes(tmp_path: Path) -> None:
    project_dir = tmp_path / "not-a-project"
    project_dir.mkdir()
    marker = project_dir / "keep.txt"
    _ = marker.write_text("do not touch", encoding="utf-8")

    with pytest.raises(ProjectStoreError, match="project_directory_invalid"):
        _ = open_project_store(project_dir)

    assert marker.read_text(encoding="utf-8") == "do not touch"


def test_symlink_to_existing_project_is_rejected_without_following_link(tmp_path: Path) -> None:
    target = tmp_path / "target"
    initial = open_project_store(target)
    with initial.stage() as stage:
        stage.write_analysis(_analysis())
        stage.write_text("source.mp4", "source")
        stage.write_text("transcript.json", "transcript")
        stage.write_text("candidates.json", "candidates")
    project_link = tmp_path / "project-link"
    project_link.symlink_to(target, target_is_directory=True)

    with pytest.raises(ProjectStoreError, match="project_directory_invalid"):
        _ = open_project_store(project_link)

    assert project_link.is_symlink()
    assert (target / "analysis.json").is_file()
    assert not list(tmp_path.glob(".analysis-staging-*"))
    assert not list(tmp_path.glob(".analysis-backup-*"))


def test_failed_reanalysis_preserves_active_analysis_and_removes_staging(tmp_path: Path) -> None:
    project_dir = tmp_path / "project"
    store = open_project_store(project_dir)
    with store.stage() as stage:
        stage.write_analysis(_analysis())
        stage.write_text("source.mp4", "first source")
        stage.write_text("transcript.json", "first transcript")
        stage.write_text("candidates.json", "first candidates")

    previous = (project_dir / "analysis.json").read_bytes()
    reopened = open_project_store(project_dir)

    with pytest.raises(RuntimeError, match="boom"):
        _write_then_fail(reopened)

    assert (project_dir / "analysis.json").read_bytes() == previous
    assert (project_dir / "source.mp4").read_text(encoding="utf-8") == "first source"
    assert not list(tmp_path.glob(".analysis-staging-*"))


def test_successful_reanalysis_preserves_editable_state_atomically(tmp_path: Path) -> None:
    project_dir = tmp_path / "project"
    initial = open_project_store(project_dir)
    with initial.stage() as stage:
        stage.write_analysis(_analysis())
        stage.write_text("source.mp4", "first source")
        stage.write_text("transcript.json", "first transcript")
        stage.write_text("candidates.json", "first candidates")
    _write_editable_state(project_dir)

    reanalysis = open_project_store(project_dir)
    _commit_reanalysis(reanalysis, source="first source")

    assert (project_dir / "source.mp4").read_text(encoding="utf-8") == "first source"
    assert (project_dir / "project.json").read_text(encoding="utf-8") == '{"schema_version":"1"}'
    assert (project_dir / "assets" / "watermark" / "asset.png").read_bytes() == b"png"
    preview = project_dir / "previews" / "variant" / "position" / "preview.jpg"
    assert preview.read_bytes() == b"jpeg"


def test_incomplete_new_project_stage_is_rejected_and_project_is_removed(tmp_path: Path) -> None:
    project_dir = tmp_path / "project"
    store = open_project_store(project_dir)

    with pytest.raises(ProjectStoreError, match="analysis_stage_invalid"):
        _commit_incomplete_stage(store)

    assert not project_dir.exists()
    assert not list(tmp_path.glob(".analysis-staging-*"))


def test_publish_failure_restores_existing_project(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project_dir = tmp_path / "project"
    initial = open_project_store(project_dir)
    with initial.stage() as stage:
        stage.write_analysis(_analysis())
        stage.write_text("source.mp4", "first source")
        stage.write_text("transcript.json", "first transcript")
        stage.write_text("candidates.json", "first candidates")
    _write_editable_state(project_dir)
    previous = (project_dir / "analysis.json").read_bytes()

    def fail_stage_publish(source: Path, destination: Path) -> Path:
        if source.name.startswith(".analysis-staging-"):
            msg = "simulated publish failure"
            raise OSError(msg)
        return source.replace(destination)

    monkeypatch.setattr("kliptych.studio.project_store._replace", fail_stage_publish)
    reopened = open_project_store(project_dir)

    with pytest.raises(ProjectStoreError, match="analysis_publish_failed"):
        _commit_reanalysis(reopened)

    assert (project_dir / "analysis.json").read_bytes() == previous
    assert (project_dir / "source.mp4").read_text(encoding="utf-8") == "first source"
    assert (project_dir / "project.json").read_text(encoding="utf-8") == '{"schema_version":"1"}'
    asset = project_dir / "assets" / "watermark" / "asset.png"
    assert asset.read_bytes() == b"png"
    assert not list(tmp_path.glob(".analysis-backup-*"))
    assert not list(tmp_path.glob(".analysis-staging-*"))


def test_reanalysis_rejects_editable_state_symlink_and_restores_project(tmp_path: Path) -> None:
    project_dir = tmp_path / "project"
    initial = open_project_store(project_dir)
    with initial.stage() as stage:
        stage.write_analysis(_analysis())
        stage.write_text("source.mp4", "first source")
        stage.write_text("transcript.json", "first transcript")
        stage.write_text("candidates.json", "first candidates")
    _write_editable_state(project_dir)
    external = tmp_path / "outside.png"
    _ = external.write_bytes(b"external")
    linked_asset = project_dir / "assets" / "watermark" / "linked.png"
    linked_asset.symlink_to(external)

    with pytest.raises(ProjectStoreError, match="analysis_publish_failed"):
        _commit_reanalysis(open_project_store(project_dir), source="first source")

    assert (project_dir / "source.mp4").read_text(encoding="utf-8") == "first source"
    assert linked_asset.is_symlink()
    assert external.read_bytes() == b"external"
    assert not list(tmp_path.glob(".analysis-backup-*"))
    assert not list(tmp_path.glob(".analysis-staging-*"))


def test_open_project_store_removes_stale_backup_for_same_project(tmp_path: Path) -> None:
    project_dir = tmp_path / "project"
    initial = open_project_store(project_dir)
    with initial.stage() as stage:
        stage.write_analysis(_analysis())
        stage.write_text("source.mp4", "source")
        stage.write_text("transcript.json", "transcript")
        stage.write_text("candidates.json", "candidates")
    stale_backup = tmp_path / ".analysis-backup-project-00000000-0000-0000-0000-000000000001"
    stale_backup.mkdir()
    _ = (stale_backup / "orphan.txt").write_text("old", encoding="utf-8")
    other_backup = tmp_path / ".analysis-backup-other-stale"
    other_backup.mkdir()

    _ = open_project_store(project_dir)

    assert not stale_backup.exists()
    assert other_backup.exists()


@pytest.mark.parametrize("project_name", ["*", "?", "[ab]"])
def test_backup_name_parser_treats_glob_metacharacters_as_literals(project_name: str) -> None:
    prefix = f".analysis-backup-{project_name}-"
    own = f"{prefix}00000000-0000-0000-0000-000000000001"
    unrelated = ".analysis-backup-unrelated-00000000-0000-0000-0000-000000000001"

    matcher = cast("_BackupNameMatcher", _private("_is_backup_for_project"))
    assert matcher(own, prefix=prefix) is True
    assert matcher(unrelated, prefix=prefix) is False


def test_link_like_path_in_editable_state_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project_dir = tmp_path / "project"
    initial = open_project_store(project_dir)
    with initial.stage() as stage:
        stage.write_analysis(_analysis())
        stage.write_text("source.mp4", "first source")
        stage.write_text("transcript.json", "first transcript")
        stage.write_text("candidates.json", "first candidates")
    _write_editable_state(project_dir)
    linked_asset = project_dir / "assets" / "watermark" / "asset.png"

    def fake_link_like(path: Path) -> bool:
        return path.name == linked_asset.name

    monkeypatch.setattr("kliptych.studio.project_store._is_link_like", fake_link_like)

    with pytest.raises(ProjectStoreError, match="analysis_publish_failed"):
        _commit_reanalysis(open_project_store(project_dir))

    assert (project_dir / "source.mp4").read_text(encoding="utf-8") == "first source"


def test_reanalysis_interrupt_during_editable_state_copy_restores_project(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project_dir = tmp_path / "project"
    initial = open_project_store(project_dir)
    with initial.stage() as stage:
        stage.write_analysis(_analysis())
        stage.write_text("source.mp4", "first source")
        stage.write_text("transcript.json", "first transcript")
        stage.write_text("candidates.json", "first candidates")
    _write_editable_state(project_dir)

    def interrupt_copy(*_args: object, **_kwargs: object) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr("kliptych.studio.project_store.shutil.copytree", interrupt_copy)

    with pytest.raises(KeyboardInterrupt):
        _commit_reanalysis(open_project_store(project_dir), source="first source")

    assert (project_dir / "source.mp4").read_text(encoding="utf-8") == "first source"
    assert (project_dir / "project.json").read_text(encoding="utf-8") == '{"schema_version":"1"}'
    assert (project_dir / "assets" / "watermark" / "asset.png").read_bytes() == b"png"
    assert not list(tmp_path.glob(".analysis-backup-*"))
    assert not list(tmp_path.glob(".analysis-staging-*"))


def test_store_update_position_persists_validated_layout(
    tmp_path: Path,
) -> None:
    """A validated crop update persists to project.json without changing evidence."""
    candidate = _candidate()
    analysis = _analysis().model_copy(update={"candidates": (candidate,)})
    project_dir = tmp_path / "project"
    store = open_project_store(project_dir)
    with store.stage() as stage:
        stage.write_analysis(analysis)
        stage.write_text("source.mp4", "source")
        stage.write_text("transcript.json", "transcript")
        stage.write_text("candidates.json", "candidates")
    store = open_project_store(project_dir)
    _ = store.initialize_project(probe=_Probe(60.0))
    created = store.create_variant(candidate_id=candidate.candidate_id, name="Variant 1")
    variant = created.variants[0]
    position = variant.layout_timeline[0]

    updated_spec = LayoutSpec(
        layout=Layout.VERTICAL,
        bands=(
            Band(
                source="source.mp4",
                source_rect=NormalizedRect(x=0.1, y=0.2, width=0.5, height=0.6),
            ),
        ),
    )
    updated = store.update_position(
        variant_id=variant.variant_id,
        position_id=position.position_id,
        layout_spec=updated_spec,
    )

    persisted = open_project_store(project_dir).read_project()
    assert persisted is not None
    persisted_position = persisted.variants[0].layout_timeline[0]
    assert persisted_position.layout_spec == updated_spec
    assert persisted.variants[0].variant_id == variant.variant_id
    assert updated.variants[0].layout_timeline[0].layout_spec == updated_spec


def _setup_project_with_variant(
    tmp_path: Path,
) -> tuple[Path, ProjectStore, Variant, LayoutPosition]:
    candidate = _candidate()
    analysis = _analysis().model_copy(update={"candidates": (candidate,)})
    project_dir = tmp_path / "project"
    store = open_project_store(project_dir)
    with store.stage() as stage:
        stage.write_analysis(analysis)
        stage.write_text("source.mp4", "source")
        stage.write_text("transcript.json", "transcript")
        stage.write_text("candidates.json", "candidates")
    store = open_project_store(project_dir)
    _ = store.initialize_project(probe=_Probe(60.0))
    created = store.create_variant(candidate_id=candidate.candidate_id, name="Variant 1")
    variant = created.variants[0]
    position = variant.layout_timeline[0]
    return project_dir, store, variant, position


def test_store_split_position_persists(tmp_path: Path) -> None:
    _, store, variant, position = _setup_project_with_variant(tmp_path)
    at_ms = (position.start_ms + position.end_ms) // 2
    updated = store.split_position(
        variant_id=variant.variant_id, position_id=position.position_id, at_ms=at_ms
    )
    assert len(updated.variants[0].layout_timeline) == 2


def test_store_split_position_rejects_boundary(tmp_path: Path) -> None:
    _, store, variant, position = _setup_project_with_variant(tmp_path)
    with pytest.raises(ProjectStoreError):
        _ = store.split_position(
            variant_id=variant.variant_id,
            position_id=position.position_id,
            at_ms=position.start_ms,
        )


def test_store_delete_position_persists(tmp_path: Path) -> None:
    project_dir, store, variant, position = _setup_project_with_variant(tmp_path)
    at_ms = (position.start_ms + position.end_ms) // 2
    _ = store.split_position(
        variant_id=variant.variant_id, position_id=position.position_id, at_ms=at_ms
    )
    refreshed = open_project_store(project_dir).read_project()
    assert refreshed is not None
    variant = refreshed.variants[0]
    pos_to_delete = variant.layout_timeline[1]
    updated = store.delete_position(
        variant_id=variant.variant_id,
        position_id=pos_to_delete.position_id,
        absorb_neighbor=None,
    )
    assert len(updated.variants[0].layout_timeline) == 1


def test_store_duplicate_variant_persists(tmp_path: Path) -> None:
    _project_dir, store, variant, _ = _setup_project_with_variant(tmp_path)
    updated = store.duplicate_variant(variant_id=variant.variant_id, name="Copy")
    assert len(updated.variants) == 2
    assert updated.variants[1].name == "Copy"


def test_store_import_watermark_persists(tmp_path: Path) -> None:
    project_dir, store, variant, _ = _setup_project_with_variant(tmp_path)
    png_path = tmp_path / "logo.png"
    _ = png_path.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 8)
    updated = store.import_watermark(
        variant_id=variant.variant_id, source_path=png_path, position=None
    )
    wm = updated.variants[0].watermark
    assert wm.enabled is True
    assert wm.asset_id is not None
    assert (project_dir / "assets" / "watermark" / f"{wm.asset_id}.png").is_file()


def test_store_import_watermark_rejects_non_png(tmp_path: Path) -> None:
    _, store, variant, _ = _setup_project_with_variant(tmp_path)
    bad = tmp_path / "logo.txt"
    _ = bad.write_text("not png")
    with pytest.raises(ProjectStoreError):
        _ = store.import_watermark(variant_id=variant.variant_id, source_path=bad, position=None)


def test_store_set_watermark_persists(tmp_path: Path) -> None:
    _project_dir, store, variant, _ = _setup_project_with_variant(tmp_path)
    png_path = tmp_path / "logo.png"
    _ = png_path.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 8)
    _ = store.import_watermark(variant_id=variant.variant_id, source_path=png_path, position=None)
    updated = store.set_watermark(variant_id=variant.variant_id, enabled=False, position=None)
    assert updated.variants[0].watermark.enabled is False


def test_store_set_captions_persists(tmp_path: Path) -> None:
    _project_dir, store, variant, _ = _setup_project_with_variant(tmp_path)
    updated = store.set_captions(
        variant_id=variant.variant_id,
        enabled=True,
        highlight_color="&H00FF0000",
        capitalization=None,
    )
    assert updated.variants[0].captions_enabled is True
    assert updated.variants[0].caption_highlight_color == "&H00FF0000"


def test_store_set_caption_correction_persists(tmp_path: Path) -> None:
    _, store, variant, position = _setup_project_with_variant(tmp_path)
    updated = store.set_caption_correction(
        variant_id=variant.variant_id,
        interval=(position.start_ms, position.end_ms),
        corrected_text="Texto corregido",
        author="user",
        created_at_ms=1000,
    )
    corrections = updated.variants[0].caption_corrections
    assert len(corrections) == 1
    assert corrections[0].corrected_text == "Texto corregido"


def test_store_set_caption_correction_rejects_invalid_range(tmp_path: Path) -> None:
    _, store, variant, position = _setup_project_with_variant(tmp_path)
    with pytest.raises(ProjectStoreError):
        _ = store.set_caption_correction(
            variant_id=variant.variant_id,
            interval=(position.end_ms, position.start_ms),
            corrected_text="Texto",
            author="user",
            created_at_ms=1000,
        )


def test_store_clear_caption_correction_persists(tmp_path: Path) -> None:
    _, store, variant, position = _setup_project_with_variant(tmp_path)
    updated = store.set_caption_correction(
        variant_id=variant.variant_id,
        interval=(position.start_ms, position.end_ms),
        corrected_text="Texto corregido",
        author="user",
        created_at_ms=1000,
    )
    correction_id = updated.variants[0].caption_corrections[0].correction_id
    cleared = store.clear_caption_correction(
        variant_id=variant.variant_id, correction_id=correction_id
    )
    assert cleared.variants[0].caption_corrections == ()


def test_store_set_caption_correction_requires_project(tmp_path: Path) -> None:
    project_dir = tmp_path / "project"
    store = open_project_store(project_dir)
    with store.stage() as stage:
        stage.write_analysis(_analysis())
        stage.write_text("source.mp4", "source")
        stage.write_text("transcript.json", "transcript")
        stage.write_text("candidates.json", "candidates")
    store = open_project_store(project_dir)
    _ = store.initialize_project(probe=_Probe(60.0))
    with pytest.raises(ProjectStoreError):
        _ = store.set_caption_correction(
            variant_id=uuid.uuid4(),
            interval=(0, 1000),
            corrected_text="Texto",
            author="user",
            created_at_ms=1,
        )


def test_store_clear_caption_correction_requires_project(tmp_path: Path) -> None:
    project_dir = tmp_path / "project"
    store = open_project_store(project_dir)
    with store.stage() as stage:
        stage.write_analysis(_analysis())
        stage.write_text("source.mp4", "source")
        stage.write_text("transcript.json", "transcript")
        stage.write_text("candidates.json", "candidates")
    store = open_project_store(project_dir)
    _ = store.initialize_project(probe=_Probe(60.0))
    with pytest.raises(ProjectStoreError):
        _ = store.clear_caption_correction(
            variant_id=uuid.uuid4(), correction_id=uuid.uuid4()
        )
