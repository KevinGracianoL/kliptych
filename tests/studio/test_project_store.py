"""Behavioral tests for the Track B1 Studio project store."""

from pathlib import Path
from typing import Protocol, cast

import pytest

from kliptych.studio import project_store as _project_store
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


def _commit_reanalysis(store: ProjectStore) -> None:
    with store.stage() as stage:
        stage.write_analysis(_analysis(project_id="c" * 32))
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
    _commit_reanalysis(reanalysis)

    assert (project_dir / "source.mp4").read_text(encoding="utf-8") == "second source"
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
        _commit_reanalysis(open_project_store(project_dir))

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
        _commit_reanalysis(open_project_store(project_dir))

    assert (project_dir / "source.mp4").read_text(encoding="utf-8") == "first source"
    assert (project_dir / "project.json").read_text(encoding="utf-8") == '{"schema_version":"1"}'
    assert (project_dir / "assets" / "watermark" / "asset.png").read_bytes() == b"png"
    assert not list(tmp_path.glob(".analysis-backup-*"))
    assert not list(tmp_path.glob(".analysis-staging-*"))
