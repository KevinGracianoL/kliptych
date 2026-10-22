"""Behavioral tests for the Track B1 Studio project store."""

from pathlib import Path

import pytest

from kliptych.studio.models import AnalysisDocument, AnalysisProvenance, AnalysisSource
from kliptych.studio.project_store import (
    ProjectStore,
    ProjectStoreError,
    open_project_store,
)


def _analysis(*, project_id: str = "a" * 32) -> AnalysisDocument:
    return AnalysisDocument(
        schema_version="1",
        project_id=project_id,
        project_name="My clip",
        source=AnalysisSource(url="https://example.com/video", sha256="b" * 64),
        provenance=AnalysisProvenance(transcription_version="1", segmenter_version="1"),
        candidates=(),
    )


def _write_then_fail(store: ProjectStore) -> None:
    message = "boom"
    with store.stage() as stage:
        stage.write_analysis(_analysis(project_id="c" * 32))
        stage.write_text("source.mp4", "second source")
        raise RuntimeError(message)


def _commit_incomplete_stage(store: ProjectStore) -> None:
    with store.stage() as stage:
        stage.write_analysis(_analysis())


def _commit_reanalysis(store: ProjectStore) -> None:
    with store.stage() as stage:
        stage.write_analysis(_analysis(project_id="c" * 32))
        stage.write_text("source.mp4", "second source")
        stage.write_text("transcript.json", "second transcript")
        stage.write_text("candidates.json", "second candidates")


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
    assert not list(tmp_path.glob(".analysis-backup-*"))
    assert not list(tmp_path.glob(".analysis-staging-*"))
