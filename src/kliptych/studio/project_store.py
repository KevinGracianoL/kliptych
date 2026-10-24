"""Transactional local storage for one private, single-user Studio project.

The project root is controlled by its Studio user. Existing symbolic links in
editable state are rejected, but this is not a multi-writer filesystem sandbox:
an untrusted concurrent process modifying the same project during a transaction
is outside the local desktop threat model.
"""

import math
import os
import shutil
import tempfile
import uuid
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Self

from pydantic import ValidationError
from pydantic_core import PydanticSerializationError

from kliptych.gate.probe import MediaProbe
from kliptych.hashing import sha256_file
from kliptych.studio.models import AnalysisDocument
from kliptych.contract.schema import LayoutSpec
from kliptych.studio.project_models import (
    ProjectDocument,
    ProjectStatus,
    VariantOrigin,
    VariantRange,
    create_initial_variant,
)

_ANALYSIS_NAME = "analysis.json"
_PROJECT_NAME = "project.json"
_TRANSCRIPT_NAME = "transcript.json"
_SOURCE_NAME = "source.mp4"
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

    def read_project(self) -> ProjectDocument | None:
        """Read the optional validated B2 editable project document.

        Returns:
            The project document, or None when it has not been created.

        Raises:
            ProjectStoreError: If an existing project document is invalid.
        """
        path = self.project_dir / _PROJECT_NAME
        if _is_link_like(path) or (path.exists() and not path.is_file()):
            msg = "Editable project document is not a regular file."
            raise ProjectStoreError(_PROJECT_DIRECTORY_INVALID, msg)
        if not path.exists():
            return None
        try:
            return ProjectDocument.model_validate_json(path.read_text(encoding="utf-8"))
        except (OSError, ValidationError, ValueError) as error:
            msg = "Editable project document is invalid."
            raise ProjectStoreError(_PROJECT_DIRECTORY_INVALID, msg) from error

    def write_project(self, document: ProjectDocument) -> None:
        """Atomically persist a validated editable project document.

        Raises:
            ProjectStoreError: If the document belongs to another project or
                cannot be stored safely.
        """
        try:
            payload = document.model_dump_json(warnings="error")
            validated = ProjectDocument.model_validate_json(payload)
        except (AttributeError, PydanticSerializationError, ValidationError, ValueError) as error:
            msg = "Editable project document is invalid."
            raise ProjectStoreError(_PROJECT_DIRECTORY_INVALID, msg) from error
        self._validate_project_binding(validated)
        previous = self.read_project()
        if previous is None:
            msg = "Editable project state must be initialized from measured evidence."
            raise ProjectStoreError(_PROJECT_DIRECTORY_INVALID, msg)
        if not _has_same_evidence(previous, validated):
            msg = "Normal project writes cannot change measured evidence fields."
            raise ProjectStoreError(_PROJECT_DIRECTORY_INVALID, msg)
        try:
            _atomic_write_text(
                self.project_dir / _PROJECT_NAME, validated.model_dump_json(indent=2)
            )
        except OSError as error:
            msg = "Editable project document could not be written."
            raise ProjectStoreError(_PROJECT_DIRECTORY_INVALID, msg) from error

    def initialize_project(self, *, probe: MediaProbe) -> ProjectDocument:
        """Create project.json from current B1 artifacts and measured source duration.

        Returns:
            The atomically persisted, empty editable project document.

        Raises:
            ProjectStoreError: If B1 artifacts or measured source duration are
                unavailable or invalid.
        """
        if self.read_project() is not None:
            msg = "Editable project state has already been initialized."
            raise ProjectStoreError(_PROJECT_DIRECTORY_INVALID, msg)
        document = self._build_project_document(probe)
        self._write_initialized_project(document)
        return document

    def _write_initialized_project(self, document: ProjectDocument) -> None:
        """Persist the first project document after measured evidence validation.

        Raises:
            ProjectStoreError: If the initialized document cannot be written.
        """
        try:
            _atomic_write_text(self.project_dir / _PROJECT_NAME, document.model_dump_json(indent=2))
        except OSError as error:
            msg = "Initialized editable project document could not be written."
            raise ProjectStoreError(_PROJECT_DIRECTORY_INVALID, msg) from error

    def rebind_project(self, *, probe: MediaProbe) -> ProjectDocument:
        """Bind existing editable state to current B1 evidence explicitly.

        Returns:
            The rebound project with variants requiring review when evidence
            changed.

        Raises:
            ProjectStoreError: If no editable project exists or B1 artifacts
                cannot produce valid current evidence.
        """
        project = self.read_project()
        if project is None:
            msg = "Editable project state has not been initialized."
            raise ProjectStoreError(_PROJECT_DIRECTORY_INVALID, msg)
        current = self._current_project_evidence(probe)
        if current.project_id != project.project_id:
            msg = "Current analysis belongs to a different analysis project."
            raise ProjectStoreError(_PROJECT_DIRECTORY_INVALID, msg)
        rebound = project.require_review_if_evidence_changed(
            analysis_sha256=current.analysis_sha256,
            transcript_sha256=current.transcript_sha256,
            source_sha256=current.source_sha256,
        )
        variants = tuple(
            variant.model_copy(update={"status": ProjectStatus.REQUIRES_REVIEW})
            if variant.end_ms > current.source_duration_ms
            else variant
            for variant in rebound.variants
        )
        rebound = rebound.model_copy(
            update={"source_duration_ms": current.source_duration_ms, "variants": variants}
        )
        self._write_rebound_project(rebound)
        return rebound

    def create_variant(self, *, candidate_id: str, name: str) -> ProjectDocument:
        """Create and persist an initial variant from an active analysis candidate.

        Returns:
            The project document containing the new active variant.

        Raises:
            ProjectStoreError: If editable state is absent, stale, or the
                candidate is not present in the active analysis.
        """
        project = self.read_project()
        analysis = self._read_current_analysis()
        if project is None or analysis is None:
            msg = "Creating a variant requires active analysis and editable project state."
            raise ProjectStoreError(_PROJECT_DIRECTORY_INVALID, msg)
        self._validate_project_binding(project)
        candidate = next(
            (item for item in analysis.candidates if item.candidate_id == candidate_id), None
        )
        if candidate is None:
            msg = "Candidate is not part of the active analysis."
            raise ProjectStoreError(_PROJECT_DIRECTORY_INVALID, msg)
        try:
            variant = create_initial_variant(
                name=name,
                selection=VariantRange(
                    start_ms=candidate.recommended_start_ms,
                    end_ms=candidate.recommended_end_ms,
                    source_duration_ms=project.source_duration_ms,
                ),
                origin=VariantOrigin(
                    source_candidate_id=candidate.candidate_id,
                    analysis_sha256=project.analysis_sha256,
                    transcript_sha256=project.transcript_sha256,
                    source_sha256=project.source_sha256,
                ),
            )
        except ValueError as error:
            msg = "Candidate range does not fit measured source duration."
            raise ProjectStoreError(_PROJECT_DIRECTORY_INVALID, msg) from error
        updated = project.model_copy(update={"variants": (*project.variants, variant)})
        self.write_project(updated)
        return updated

    def update_position(
        self,
        *,
        variant_id: uuid.UUID,
        position_id: uuid.UUID,
        layout_spec: LayoutSpec,
    ) -> ProjectDocument:
        """Persist a validated layout update for one position.

        Returns:
            The project document containing the updated position.

        Raises:
            ProjectStoreError: If editable state, variant, or position is absent,
                or the supplied layout is invalid.
        """
        project = self.read_project()
        if project is None:
            msg = "Updating a position requires editable project state."
            raise ProjectStoreError(_PROJECT_DIRECTORY_INVALID, msg)
        self._validate_project_binding(project)
        variant_index = next(
            (i for i, item in enumerate(project.variants) if item.variant_id == variant_id),
            None,
        )
        if variant_index is None:
            msg = "Variant is not part of the editable project."
            raise ProjectStoreError(_PROJECT_DIRECTORY_INVALID, msg)
        variant = project.variants[variant_index]
        position_index = next(
            (i for i, item in enumerate(variant.layout_timeline) if item.position_id == position_id),
            None,
        )
        if position_index is None:
            msg = "Position is not part of the variant timeline."
            raise ProjectStoreError(_PROJECT_DIRECTORY_INVALID, msg)
        if not layout_spec.bands:
            msg = "Layout position requires explicit bands."
            raise ProjectStoreError(_PROJECT_DIRECTORY_INVALID, msg)
        if any(band.source != _SOURCE_NAME for band in layout_spec.bands):
            msg = "B2 layout bands must reference source.mp4."
            raise ProjectStoreError(_PROJECT_DIRECTORY_INVALID, msg)
        for band in layout_spec.bands:
            rect = band.source_rect
            if rect.width < 0.05 or rect.height < 0.05:
                msg = "Layout crop must be at least 0.05 in each dimension."
                raise ProjectStoreError(_PROJECT_DIRECTORY_INVALID, msg)
        positions = list(variant.layout_timeline)
        positions[position_index] = positions[position_index].model_copy(
            update={"layout_spec": layout_spec}
        )
        updated_variant = variant.model_copy(update={"layout_timeline": tuple(positions)})
        variants = list(project.variants)
        variants[variant_index] = updated_variant
        updated = project.model_copy(update={"variants": tuple(variants)})
        self.write_project(updated)
        return updated

    def _write_rebound_project(self, document: ProjectDocument) -> None:
        """Persist an explicit evidence rebind without normal hash equality checks.

        Raises:
            ProjectStoreError: If the rebound document cannot be written.
        """
        try:
            _atomic_write_text(self.project_dir / _PROJECT_NAME, document.model_dump_json(indent=2))
        except OSError as error:
            msg = "Rebound editable project document could not be written."
            raise ProjectStoreError(_PROJECT_DIRECTORY_INVALID, msg) from error

    def _current_project_evidence(self, probe: MediaProbe) -> ProjectDocument:
        """Measure and hash current B1 artifacts into an empty project document.

        Returns:
            A document containing current B1 evidence and no human variants.
        """
        return self._build_project_document(probe)

    def _validate_project_binding(self, document: ProjectDocument) -> None:
        """Require editable state to bind exactly to current B1 evidence.

        Raises:
            ProjectStoreError: If B1 analysis is absent or any evidence hash
                differs from the persisted project artifacts.
        """
        analysis = self._read_current_analysis()
        if analysis is None:
            msg = "Editable project state requires a completed analysis."
            raise ProjectStoreError(_PROJECT_DIRECTORY_INVALID, msg)
        analysis_path = self.project_dir / _ANALYSIS_NAME
        source_path = self.project_dir / _SOURCE_NAME
        transcript_path = self.project_dir / _TRANSCRIPT_NAME
        if not source_path.is_file() or _is_link_like(source_path):
            msg = "Editable project state requires a regular source artifact."
            raise ProjectStoreError(_PROJECT_DIRECTORY_INVALID, msg)
        if not transcript_path.is_file() or _is_link_like(transcript_path):
            msg = "Editable project state requires a regular transcript artifact."
            raise ProjectStoreError(_PROJECT_DIRECTORY_INVALID, msg)
        if document.project_id != analysis.project_id:
            msg = "Editable project document belongs to a different analysis project."
            raise ProjectStoreError(_PROJECT_DIRECTORY_INVALID, msg)
        source_hash = sha256_file(source_path)
        if analysis.source.sha256 != source_hash:
            msg = "Analysis source hash does not match source artifact."
            raise ProjectStoreError(_PROJECT_DIRECTORY_INVALID, msg)
        if document.source_sha256 != source_hash:
            msg = "Editable project document source hash does not match analysis."
            raise ProjectStoreError(_PROJECT_DIRECTORY_INVALID, msg)
        if document.analysis_sha256 != sha256_file(analysis_path):
            msg = "Editable project document analysis hash does not match analysis artifact."
            raise ProjectStoreError(_PROJECT_DIRECTORY_INVALID, msg)
        if document.transcript_sha256 != sha256_file(transcript_path):
            msg = "Editable project document transcript hash does not match transcript artifact."
            raise ProjectStoreError(_PROJECT_DIRECTORY_INVALID, msg)
        self._validate_active_variant_origins(document, analysis)

    @staticmethod
    def _validate_active_variant_origins(
        document: ProjectDocument, analysis: AnalysisDocument
    ) -> None:
        """Require active variants to retain verifiable origins in active analysis.

        Raises:
            ProjectStoreError: If an active variant origin is stale or absent.
        """
        candidate_ids = {candidate.candidate_id for candidate in analysis.candidates}
        for variant in document.variants:
            if variant.status is not ProjectStatus.ACTIVE:
                continue
            if (
                variant.origin.analysis_sha256 != document.analysis_sha256
                or variant.origin.transcript_sha256 != document.transcript_sha256
                or variant.origin.source_sha256 != document.source_sha256
                or variant.origin.source_candidate_id not in candidate_ids
            ):
                msg = "Active variant origin does not match current B1 evidence."
                raise ProjectStoreError(_PROJECT_DIRECTORY_INVALID, msg)

    def _build_project_document(self, probe: MediaProbe) -> ProjectDocument:
        """Build an empty project bound to current B1 artifacts.

        Returns:
            An empty editable document bound to measured current evidence.

        Raises:
            ProjectStoreError: If required artifacts or duration are invalid.
        """
        analysis = self._read_current_analysis()
        if analysis is None:
            msg = "Editable project state requires a completed analysis."
            raise ProjectStoreError(_PROJECT_DIRECTORY_INVALID, msg)
        source_path = self.project_dir / _SOURCE_NAME
        transcript_path = self.project_dir / _TRANSCRIPT_NAME
        if not source_path.is_file() or _is_link_like(source_path):
            msg = "Editable project state requires a regular source artifact."
            raise ProjectStoreError(_PROJECT_DIRECTORY_INVALID, msg)
        if not transcript_path.is_file() or _is_link_like(transcript_path):
            msg = "Editable project state requires a regular transcript artifact."
            raise ProjectStoreError(_PROJECT_DIRECTORY_INVALID, msg)
        try:
            duration_s = probe.probe(source_path).duration_s
        except Exception as error:
            msg = "Source duration could not be measured."
            raise ProjectStoreError(_PROJECT_DIRECTORY_INVALID, msg) from error
        duration_ms = _duration_ms(duration_s)
        source_sha256 = sha256_file(source_path)
        if analysis.source.sha256 != source_sha256:
            msg = "Analysis source hash does not match source artifact."
            raise ProjectStoreError(_PROJECT_DIRECTORY_INVALID, msg)
        return ProjectDocument(
            schema_version="1",
            project_id=analysis.project_id,
            analysis_sha256=sha256_file(self.project_dir / _ANALYSIS_NAME),
            transcript_sha256=sha256_file(transcript_path),
            source_sha256=source_sha256,
            source_duration_ms=duration_ms,
            variants=(),
        )

    def _read_current_analysis(self) -> AnalysisDocument | None:
        """Read current analysis.json instead of retaining a stale store snapshot.

        Returns:
            The current valid analysis, or None when no analysis is published.

        Raises:
            ProjectStoreError: If current analysis is unsafe or invalid.
        """
        path = self.project_dir / _ANALYSIS_NAME
        if not path.exists():
            return None
        if not path.is_file() or _is_link_like(path):
            msg = "Analysis artifact is not a regular file."
            raise ProjectStoreError(_PROJECT_DIRECTORY_INVALID, msg)
        try:
            return AnalysisDocument.model_validate_json(path.read_text(encoding="utf-8"))
        except (OSError, ValidationError, ValueError) as error:
            msg = "Analysis artifact is invalid."
            raise ProjectStoreError(_PROJECT_DIRECTORY_INVALID, msg) from error

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
        self._reject_automatic_source_migration()
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

    def _reject_automatic_source_migration(self) -> None:
        """Require user action before replacing a source bound to editable state.

        Raises:
            ProjectStoreError: If a staged source differs from a project that
                already owns editable B2 state.
        """
        has_editable_state = any(
            (self.store.project_dir / name).exists() or _is_link_like(self.store.project_dir / name)
            for name in _EDITABLE_STATE_NAMES
        )
        if not has_editable_state:
            return
        active_source = self.store.project_dir / _SOURCE_NAME
        staged_source = self.directory / _SOURCE_NAME
        if not active_source.is_file() or _is_link_like(active_source):
            msg = "Reanalysis source is unavailable and requires explicit user action."
            raise ProjectStoreError(_ANALYSIS_PUBLISH_FAILED, msg)
        if not staged_source.is_file() or _is_link_like(staged_source):
            msg = "Staged reanalysis source is unavailable."
            raise ProjectStoreError(_ANALYSIS_PUBLISH_FAILED, msg)
        if sha256_file(active_source) != sha256_file(staged_source):
            msg = "Reanalysis source changed and requires explicit user action."
            raise ProjectStoreError(_ANALYSIS_PUBLISH_FAILED, msg)

    def _prepare_publish(self, backup: Path) -> None:
        """Move the active project aside and preserve its editable state."""
        if self.store.owns_empty_project_dir and not any(self.store.project_dir.iterdir()):
            _ = self.store.project_dir.rmdir()
            return
        _ = _replace(self.store.project_dir, backup)
        _migrate_editable_state(source=backup, destination=self.directory)
        _mark_migrated_project_requires_review(destination=self.directory)


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


def _mark_migrated_project_requires_review(*, destination: Path) -> None:
    """Block preserved variants when staged B1 evidence differs from their origin."""
    project_path = destination / _PROJECT_NAME
    if not project_path.is_file():
        return
    try:
        project = ProjectDocument.model_validate_json(project_path.read_text(encoding="utf-8"))
    except (OSError, ValidationError, ValueError):
        return
    analysis_path = destination / _ANALYSIS_NAME
    transcript_path = destination / _TRANSCRIPT_NAME
    source_path = destination / _SOURCE_NAME
    if not analysis_path.is_file() or not transcript_path.is_file() or not source_path.is_file():
        return
    try:
        analysis = AnalysisDocument.model_validate_json(analysis_path.read_text(encoding="utf-8"))
    except (OSError, ValidationError, ValueError):
        return
    analysis_sha256 = sha256_file(analysis_path)
    transcript_sha256 = sha256_file(transcript_path)
    source_sha256 = sha256_file(source_path)
    if (
        project.analysis_sha256 != analysis_sha256
        or project.transcript_sha256 != transcript_sha256
        or project.source_sha256 != source_sha256
        or analysis.source.sha256 != source_sha256
    ):
        _atomic_write_text(
            project_path, project.mark_variants_requires_review().model_dump_json(indent=2)
        )


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


def _duration_ms(duration_s: float | None) -> int:
    """Round a finite positive ffprobe duration into a positive millisecond bound.

    Returns:
        The nearest positive integer millisecond duration.

    Raises:
        ProjectStoreError: If the duration cannot define a positive millisecond
            range for editable Studio state.
    """
    if duration_s is None or not math.isfinite(duration_s) or duration_s <= 0:
        msg = "Measured source duration is unusable."
        raise ProjectStoreError(_PROJECT_DIRECTORY_INVALID, msg)
    duration_ms = round(duration_s * 1000)
    if duration_ms <= 0:
        msg = "Measured source duration rounds to zero milliseconds."
        raise ProjectStoreError(_PROJECT_DIRECTORY_INVALID, msg)
    return duration_ms


def _has_same_evidence(previous: ProjectDocument, current: ProjectDocument) -> bool:
    """Return whether normal persistence preserves the measured evidence snapshot."""
    return (
        previous.project_id == current.project_id
        and previous.analysis_sha256 == current.analysis_sha256
        and previous.transcript_sha256 == current.transcript_sha256
        and previous.source_sha256 == current.source_sha256
        and previous.source_duration_ms == current.source_duration_ms
    )
