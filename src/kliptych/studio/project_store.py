"""Transactional local storage for one private, single-user Studio project.

The project root is controlled by its Studio user. Existing symbolic links in
editable state are rejected, but this is not a multi-writer filesystem sandbox:
an untrusted concurrent process modifying the same project during a transaction
is outside the local desktop threat model.
"""

import os
import shutil
import tempfile
import uuid
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Self

from pydantic import ValidationError

from kliptych.studio.models import AnalysisDocument

_ANALYSIS_NAME = "analysis.json"
_REQUIRED_ARTIFACTS = frozenset(
    {_ANALYSIS_NAME, "source.mp4", "transcript.json", "candidates.json"}
)
_STAGING_PREFIX = ".analysis-staging-"
_PROJECT_DIRECTORY_INVALID = "project_directory_invalid"
_ARTIFACT_NAME_INVALID = "artifact_name_invalid"
_ANALYSIS_PUBLISH_FAILED = "analysis_publish_failed"
_ANALYSIS_STAGE_INVALID = "analysis_stage_invalid"
_EDITABLE_STATE_NAMES = ("project.json", "assets", "previews")


class ProjectStoreError(Exception):
    """A project directory cannot safely host a Studio analysis."""

    def __init__(self, code: str, message: str) -> None:
        """Create a typed storage failure."""
        super().__init__(f"{code}: {message}")
        self.code: str = code
        self.message: str = message


@dataclass(frozen=True, slots=True)
class ProjectStore:
    """A validated project directory with a transaction-based staging API."""

    project_dir: Path
    existing_analysis: AnalysisDocument | None
    owns_empty_project_dir: bool

    def stage(self) -> "AnalysisStage":
        """Create an uncommitted stage owned by this store.

        Returns:
            A context-managed staging directory. A normal context exit publishes
            the staged result; an exception removes it without changing active
            artifacts.

        Raises:
            ProjectStoreError: If the staging directory cannot be created.
        """
        directory = self.project_dir.parent / f"{_STAGING_PREFIX}{uuid.uuid4().hex}"
        try:
            directory.mkdir()
        except OSError as error:
            msg = "Could not create analysis staging directory."
            raise ProjectStoreError(_PROJECT_DIRECTORY_INVALID, msg) from error
        return AnalysisStage(store=self, directory=directory)


@dataclass(slots=True)
class AnalysisStage:
    """One unpublished set of analysis artifacts."""

    store: ProjectStore
    directory: Path
    _committed: bool = False

    def __enter__(self) -> Self:
        """Enter the analysis staging transaction.

        Returns:
            This transaction.
        """
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> Literal[False]:
        """Publish on success or remove only this stage on failure.

        Returns:
            False so exceptions always propagate.
        """
        if exc_type is None:
            try:
                self.commit()
            except BaseException:
                _remove_tree(self.directory)
                self._discard_new_project()
                raise
        else:
            _remove_tree(self.directory)
            self._discard_new_project()
        return False

    def _discard_new_project(self) -> None:
        """Remove a new project after a failed or cancelled transaction."""
        if self.store.owns_empty_project_dir:
            _remove_tree(self.store.project_dir)

    def write_analysis(self, document: AnalysisDocument) -> None:
        """Write the canonical analysis document to this stage."""
        _atomic_write_text(self.directory / _ANALYSIS_NAME, document.model_dump_json(indent=2))

    def write_text(self, name: str, content: str) -> None:
        """Write a testable text artifact using an allowlisted filename.

        Raises:
            ProjectStoreError: If the artifact filename is not allowed.
        """
        if name not in _REQUIRED_ARTIFACTS - {_ANALYSIS_NAME}:
            msg = "Artifact name is not allowed in an analysis stage."
            raise ProjectStoreError(_ARTIFACT_NAME_INVALID, msg)
        _atomic_write_text(self.directory / name, content)

    def write_bytes(self, name: str, content: bytes) -> None:
        """Write a binary allowlisted artifact to this stage.

        Raises:
            ProjectStoreError: If the artifact filename is not allowed.
        """
        if name not in _REQUIRED_ARTIFACTS - {_ANALYSIS_NAME}:
            msg = "Artifact name is not allowed in an analysis stage."
            raise ProjectStoreError(_ARTIFACT_NAME_INVALID, msg)
        _atomic_write_bytes(self.directory / name, content)

    def commit(self) -> None:
        """Validate and publish this complete stage exactly once.

        Raises:
            ProjectStoreError: If validation or publication fails.
        """
        if self._committed:
            return
        _validate_stage(self.directory)
        backup = self.store.project_dir.parent / (
            f".analysis-backup-{self.store.project_dir.name}-{uuid.uuid4().hex}"
        )
        try:
            self._prepare_publish(backup)
            _ = _replace(self.directory, self.store.project_dir)
        except BaseException as error:
            _restore_backup(project_dir=self.store.project_dir, backup=backup)
            if not isinstance(error, OSError):
                raise
            msg = "Could not publish analysis transaction."
            raise ProjectStoreError(_ANALYSIS_PUBLISH_FAILED, msg) from error
        else:
            self._committed = True
            _remove_tree_best_effort(backup)

    def _prepare_publish(self, backup: Path) -> None:
        """Move the active project aside and preserve its editable state."""
        if self.store.owns_empty_project_dir:
            _ = self.store.project_dir.rmdir()
            return
        _ = _replace(self.store.project_dir, backup)
        _migrate_editable_state(source=backup, destination=self.directory)


def open_project_store(project_dir: Path) -> ProjectStore:
    """Open a new or existing Studio project directory safely.

    Args:
        project_dir: A non-existent directory for a new project, or a directory
            containing a valid completed analysis for reanalysis.

    Returns:
        A store whose stage transactions can publish complete analysis artifacts.

    Raises:
        ProjectStoreError: If the path is unsafe or not a valid Studio project.

    """
    _cleanup_stale_backups(project_dir)
    if _is_link_like(project_dir):
        msg = "Project directory must not be a symbolic link."
        raise ProjectStoreError(_PROJECT_DIRECTORY_INVALID, msg)
    if not project_dir.exists():
        _create_new_project(project_dir)
        return ProjectStore(
            project_dir=project_dir,
            existing_analysis=None,
            owns_empty_project_dir=True,
        )
    analysis = _load_active_analysis(project_dir)
    return ProjectStore(
        project_dir=project_dir,
        existing_analysis=analysis,
        owns_empty_project_dir=False,
    )


def _create_new_project(project_dir: Path) -> None:
    try:
        project_dir.mkdir(parents=True, exist_ok=False)
    except OSError as error:
        msg = "Project directory could not be created."
        raise ProjectStoreError(_PROJECT_DIRECTORY_INVALID, msg) from error


def _load_active_analysis(project_dir: Path) -> AnalysisDocument:
    analysis_path = project_dir / _ANALYSIS_NAME
    if not project_dir.is_dir() or not analysis_path.is_file():
        msg = "Project directory is not a completed Kliptych Studio project."
        raise ProjectStoreError(_PROJECT_DIRECTORY_INVALID, msg)
    try:
        return AnalysisDocument.model_validate_json(analysis_path.read_text(encoding="utf-8"))
    except (OSError, ValidationError, ValueError) as error:
        msg = "Project analysis document is invalid."
        raise ProjectStoreError(_PROJECT_DIRECTORY_INVALID, msg) from error


def _validate_stage(directory: Path) -> None:
    actual = frozenset(path.name for path in directory.iterdir() if path.is_file())
    if actual != _REQUIRED_ARTIFACTS:
        msg = "Analysis stage does not contain the complete required artifact set."
        raise ProjectStoreError(_ANALYSIS_STAGE_INVALID, msg)
    try:
        _ = AnalysisDocument.model_validate_json(
            (directory / _ANALYSIS_NAME).read_text(encoding="utf-8")
        )
    except (OSError, ValidationError, ValueError) as error:
        msg = "Staged analysis document is invalid."
        raise ProjectStoreError(_ANALYSIS_STAGE_INVALID, msg) from error


def _restore_backup(*, project_dir: Path, backup: Path) -> None:
    if not backup.exists() or project_dir.exists():
        return
    with suppress(OSError):
        _ = _replace(backup, project_dir)


def _migrate_editable_state(*, source: Path, destination: Path) -> None:
    """Copy editable state from a private local project into its next stage.

    Raises:
        OSError: If editable state is unsafe or cannot be copied.
    """
    for name in _EDITABLE_STATE_NAMES:
        original = source / name
        if _is_link_like(original):
            msg = "Editable project state must not contain symbolic links or junctions."
            raise OSError(msg)
        if original.is_file():
            _ = shutil.copy2(original, destination / name, follow_symlinks=False)
        elif original.is_dir():
            _reject_symlinks(original)
            _ = shutil.copytree(original, destination / name, symlinks=False)
        elif original.exists() or _is_link_like(original):
            msg = "Editable project state must be a regular file or directory."
            raise OSError(msg)


def _reject_symlinks(path: Path) -> None:
    """Reject links from state created under Studio's controlled project root.

    Raises:
        OSError: If a managed state path contains a symbolic link or junction.
    """
    if _is_link_like(path) or any(_is_link_like(child) for child in path.rglob("*")):
        msg = "Editable project state must not contain symbolic links or junctions."
        raise OSError(msg)


def _is_link_like(path: Path) -> bool:
    """Return whether a path is a symbolic link or a Windows directory junction."""
    return path.is_symlink() or path.is_junction()


def _replace(source: Path, destination: Path) -> Path:
    """Rename one sibling path, allowing tests to prove rollback behavior.

    Returns:
        The destination path after the rename.
    """
    return source.replace(destination)


def _atomic_write_text(destination: Path, content: str) -> None:
    fd, temporary = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.stem}-",
        suffix=".tmp",
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            _ = handle.write(content)
        _ = Path(temporary).replace(destination)
    except BaseException:
        with suppress(OSError):
            Path(temporary).unlink()
        raise


def _atomic_write_bytes(destination: Path, content: bytes) -> None:
    fd, temporary = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.stem}-",
        suffix=".tmp",
    )
    try:
        with os.fdopen(fd, "wb") as handle:
            _ = handle.write(content)
        _ = Path(temporary).replace(destination)
    except BaseException:
        with suppress(OSError):
            Path(temporary).unlink()
        raise


def _remove_tree(path: Path) -> None:
    with suppress(FileNotFoundError):
        shutil.rmtree(path)


def _remove_tree_best_effort(path: Path) -> None:
    """Try to remove a published transaction backup without affecting success."""
    with suppress(BaseException):
        _remove_tree(path)


def _cleanup_stale_backups(project_dir: Path) -> None:
    """Retry cleanup of backups left by an interrupted post-publication cleanup."""
    prefix = f".analysis-backup-{project_dir.name}-"
    try:
        siblings = tuple(project_dir.parent.iterdir())
    except OSError:
        return
    for backup in siblings:
        if _is_backup_for_project(backup.name, prefix=prefix):
            _remove_tree_best_effort(backup)


def _is_backup_for_project(name: str, *, prefix: str) -> bool:
    """Return whether a sibling name has this project's exact UUID backup form."""
    if not name.startswith(prefix):
        return False
    try:
        _ = uuid.UUID(name.removeprefix(prefix))
    except ValueError:
        return False
    return True
