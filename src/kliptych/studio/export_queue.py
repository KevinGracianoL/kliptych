"""Persistent export queue for Kliptych Studio (ADR 0008).

A project-local ordered collection of explicitly selected export jobs. Each
job records the variant identity, a canonical revision digest, explicit
audio normalization and a controlled unique relative output path. Only one
worker runs at a time; state mutations are serialized separately so
enqueue, stop and status keep working while a render runs.
"""

import contextlib
import os
import re
import sqlite3
import tempfile
import uuid
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path, PurePosixPath
from typing import ClassVar, Literal, Protocol, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from kliptych.hashing import sha256_file
from kliptych.studio.final_render import (
    FinalRenderError,
    FinalRenderRequest,
    FinalRenderResult,
)
from kliptych.studio.models import AnalysisDocument
from kliptych.studio.project_models import ProjectDocument, Variant, variant_revision
from kliptych.studio.project_store import ProjectStoreError, open_project_store

EXPORT_QUEUE_SCHEMA_VERSION = "1"
_EXPORT_QUEUE_FILENAME = "export_queue.json"
_STATE_LOCK_FILENAME = ".export_queue_state.db"
_WORKER_LOCK_FILENAME = ".export_worker.db"
_MIN_PATH_PARTS = 2
_STATE_TIMEOUT_S = 5.0
_WORKER_TIMEOUT_S = 1.0
_JOB_HEX_LENGTH = 8
_SHA256_PATTERN = r"^[0-9a-f]{64}$"

_EXPORT_QUEUE_INVALID = "export_queue_invalid"
_EXPORT_QUEUE_IO = "export_queue_io"
_EXPORT_QUEUE_BUSY = "export_queue_busy"
_WORKER_BUSY = "worker_busy"
_JOB_NOT_FOUND = "job_not_found"
_JOB_INVALID = "job_invalid"
_VARIANT_NOT_FOUND = "variant_not_found"
_VARIANT_NOT_ACTIVE = "variant_not_active"
_ORIGIN_MISMATCH = "origin_mismatch"
_REVISION_MISMATCH = "revision_mismatch"


class ExportJobState(StrEnum):
    """Lifecycle state of one persisted export job."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    INTERRUPTED = "interrupted"


class ExportJob(BaseModel):
    """One persisted request to export a variant revision."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", frozen=True)

    job_id: uuid.UUID
    variant_id: uuid.UUID
    variant_name: str = Field(min_length=1)
    revision: str = Field(pattern=_SHA256_PATTERN)
    output_path: str = Field(min_length=1)
    normalize_audio: bool = True
    state: ExportJobState = ExportJobState.PENDING
    error: str | None = None
    output_sha256: str | None = Field(default=None, pattern=_SHA256_PATTERN)
    width: int | None = Field(default=None, gt=0)
    height: int | None = Field(default=None, gt=0)

    @field_validator("output_path")
    @classmethod
    def _output_path_is_controlled(cls, value: str) -> str:
        """Validate that an output path stays inside the controlled delivery prefix.

        Args:
            value: The relative output path to validate.

        Returns:
            The validated output path.

        Raises:
            ValueError: If the path is absolute, escapes, or leaves delivery.
        """
        if "\\" in value:
            msg = f"export output path must use posix separators: {value!r}"
            raise ValueError(msg)
        parsed = PurePosixPath(value)
        if parsed.is_absolute():
            msg = f"export output path must be relative: {value!r}"
            raise ValueError(msg)
        if ".." in parsed.parts:
            msg = f"export output path must not escape the project: {value!r}"
            raise ValueError(msg)
        if len(parsed.parts) < _MIN_PATH_PARTS or parsed.parts[0] != "delivery":
            msg = f"export output path must live under delivery/: {value!r}"
            raise ValueError(msg)
        if parsed.suffix.lower() != ".mp4":
            msg = f"export output path must end in .mp4: {value!r}"
            raise ValueError(msg)
        return value

    @model_validator(mode="after")
    def _state_and_error_agree(self) -> Self:
        """Validate that terminal states carry actionable errors.

        Returns:
            The validated job.

        Raises:
            ValueError: If state and error disagree.
        """
        if self.state is ExportJobState.COMPLETED and self.error is not None:
            msg = "completed jobs must not carry an error"
            raise ValueError(msg)
        pending_running = {ExportJobState.PENDING, ExportJobState.RUNNING}
        if self.state in pending_running and self.error is not None:
            msg = "pending/running jobs must not carry an error"
            raise ValueError(msg)
        terminal = {ExportJobState.FAILED, ExportJobState.INTERRUPTED}
        if self.state in terminal and not self.error:
            msg = "failed/interrupted jobs must carry an error"
            raise ValueError(msg)
        return self


class ExportQueueDocument(BaseModel):
    """Versioned project-local export queue."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["1"] = EXPORT_QUEUE_SCHEMA_VERSION
    project_id: str = Field(min_length=1)
    stop_requested: bool = False
    jobs: tuple[ExportJob, ...] = ()

    @model_validator(mode="after")
    def _jobs_have_unique_identity(self) -> Self:
        """Validate that jobs and destinations are unique.

        Returns:
            The validated queue.

        Raises:
            ValueError: If job IDs or output paths collide.
        """
        if len({job.job_id for job in self.jobs}) != len(self.jobs):
            msg = "export job IDs must be unique"
            raise ValueError(msg)
        if len({job.output_path for job in self.jobs}) != len(self.jobs):
            msg = "export output paths must be unique"
            raise ValueError(msg)
        return self


class ExportQueueError(Exception):
    """A persisted export queue cannot satisfy the requested operation."""

    def __init__(self, code: str, message: str) -> None:
        """Create a typed queue failure.

        Args:
            code: Machine-readable failure code.
            message: Human-actionable message.
        """
        super().__init__(f"{code}: {message}")
        self.code: str = code
        self.message: str = message


class ExportRenderer(Protocol):
    """Renders one queued job to its unique destination."""

    def __call__(self, request: FinalRenderRequest) -> FinalRenderResult:
        """Render one export request through the existing final-render service.

        Args:
            request: The render request with expected revision and no overwrite.

        Returns:
            The validated final render result.
        """
        ...


@dataclass(frozen=True, slots=True)
class QueueRunSummary:
    """Aggregate outcome of one explicit queue run."""

    completed: int
    failed: int
    stopped: bool


@dataclass(frozen=True, slots=True)
class JobOutcome:
    """Terminal outcome recorded for one job."""

    state: ExportJobState
    error: str | None
    output_sha256: str | None = None
    width: int | None = None
    height: int | None = None


def _is_link_like(path: Path) -> bool:
    """Return whether a path is a symlink or a Windows junction.

    Args:
        path: The path to inspect.

    Returns:
        True when the path is link-like.
    """
    return path.is_symlink() or path.is_junction()


def _queue_path(project_dir: Path) -> Path:
    """Return the controlled queue document path.

    Args:
        project_dir: The Studio project directory.

    Returns:
        The queue JSON path.
    """
    return project_dir / _EXPORT_QUEUE_FILENAME


def _state_lock_path(project_dir: Path) -> Path:
    """Return the state-serialization lock database path.

    Args:
        project_dir: The Studio project directory.

    Returns:
        The state lock database path.
    """
    return project_dir / _STATE_LOCK_FILENAME


def _worker_lock_path(project_dir: Path) -> Path:
    """Return the worker-lease lock database path.

    Args:
        project_dir: The Studio project directory.

    Returns:
        The worker lock database path.
    """
    return project_dir / _WORKER_LOCK_FILENAME


def _sanitize_stem(name: str) -> str:
    """Return a controlled filename stem for one variant name.

    Args:
        name: The human variant name.

    Returns:
        The sanitized stem.
    """
    cleaned = re.sub(r"[^\w\-. ]", "_", name.strip())
    cleaned = re.sub(r"^[.\s]+", "", cleaned)
    return cleaned.strip()


def _unique_output_path(variant_name: str, job_id: uuid.UUID) -> str:
    """Return a controlled unique relative delivery path for one job.

    Args:
        variant_name: The human variant name.
        job_id: The fresh job identity.

    Returns:
        The relative delivery path incorporating the job identity.
    """
    stem = _sanitize_stem(variant_name) or f"variant_{job_id.hex[:_JOB_HEX_LENGTH]}"
    return f"delivery/{stem}-{job_id.hex[:_JOB_HEX_LENGTH]}.mp4"


def _atomic_write_text(destination: Path, content: str) -> None:
    """Atomically write text, cleaning the temp file on any failure.

    Args:
        destination: The final file path.
        content: The text content to persist.
    """
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
        with contextlib.suppress(OSError):
            Path(temporary).unlink()
        raise


def _read_queue_unlocked(project_dir: Path) -> ExportQueueDocument | None:
    """Read and validate the queue without taking any lock.

    Args:
        project_dir: The Studio project directory.

    Returns:
        The queue document, or None when it has not been created.

    Raises:
        ExportQueueError: If the queue is link-like, malformed, or versioned wrong.
    """
    path = _queue_path(project_dir)
    if _is_link_like(path) or (path.exists() and not path.is_file()):
        msg = "Export queue must be a regular file."
        raise ExportQueueError(_EXPORT_QUEUE_INVALID, msg)
    if not path.exists():
        return None
    try:
        return ExportQueueDocument.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        msg = f"Export queue is invalid: {error}"
        raise ExportQueueError(_EXPORT_QUEUE_INVALID, msg) from error


def _write_queue_unlocked(project_dir: Path, document: ExportQueueDocument) -> None:
    """Validate and atomically persist the queue without taking any lock.

    Args:
        project_dir: The Studio project directory.
        document: The queue document to persist.

    Raises:
        ExportQueueError: If the queue is invalid or cannot be written.
    """
    path = _queue_path(project_dir)
    if _is_link_like(path):
        msg = "Export queue must not be a symbolic link."
        raise ExportQueueError(_EXPORT_QUEUE_INVALID, msg)
    try:
        payload = document.model_dump_json()
        validated = ExportQueueDocument.model_validate_json(payload)
    except ValueError as error:
        msg = f"Export queue is invalid: {error}"
        raise ExportQueueError(_EXPORT_QUEUE_INVALID, msg) from error
    try:
        project_dir.mkdir(parents=True, exist_ok=True)
        _atomic_write_text(path, validated.model_dump_json(indent=2))
    except OSError as error:
        msg = f"Export queue could not be written: {error}"
        raise ExportQueueError(_EXPORT_QUEUE_IO, msg) from error


def _open_lock_db(path: Path, *, timeout_s: float) -> sqlite3.Connection:
    """Open one lock database with a bounded busy wait.

    Args:
        path: The lock database path.
        timeout_s: The bounded SQLite busy timeout.

    Returns:
        The open SQLite connection.

    Raises:
        ExportQueueError: If the lock database is link-like or cannot open.
    """
    if _is_link_like(path):
        msg = "Export lock database must not be a symbolic link."
        raise ExportQueueError(_EXPORT_QUEUE_INVALID, msg)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        msg = f"Export lock database could not be prepared: {error}"
        raise ExportQueueError(_EXPORT_QUEUE_IO, msg) from error
    try:
        return sqlite3.connect(str(path), timeout=timeout_s, isolation_level=None)
    except sqlite3.Error as error:
        msg = f"Export lock database could not be opened: {error}"
        raise ExportQueueError(_EXPORT_QUEUE_IO, msg) from error


@contextmanager
def _state_locked(project_dir: Path, *, timeout_s: float = _STATE_TIMEOUT_S) -> Generator[None]:
    """Hold the short state-serialization lock for one read-modify-write.

    Args:
        project_dir: The Studio project directory.
        timeout_s: The bounded busy wait.

    Yields:
        Nothing; holds the exclusive state lock.

    Raises:
        ExportQueueError: If the queue is busy.
    """
    conn = _open_lock_db(_state_lock_path(project_dir), timeout_s=timeout_s)
    try:
        _ = conn.execute(
            "CREATE TABLE IF NOT EXISTS state_lock (id INTEGER PRIMARY KEY CHECK(id=1))"
        )
        _ = conn.execute("BEGIN EXCLUSIVE")
        yield
        _ = conn.execute("COMMIT")
    except sqlite3.OperationalError as error:
        with contextlib.suppress(sqlite3.Error):
            _ = conn.execute("ROLLBACK")
        msg = f"Export queue is busy: {error}"
        raise ExportQueueError(_EXPORT_QUEUE_BUSY, msg) from error
    finally:
        conn.close()


@contextmanager
def _worker_held(project_dir: Path, *, timeout_s: float = _WORKER_TIMEOUT_S) -> Generator[None]:
    """Hold the worker lease for the duration of one explicit run.

    Args:
        project_dir: The Studio project directory.
        timeout_s: The bounded busy wait.

    Yields:
        Nothing; holds the exclusive worker lease.

    Raises:
        ExportQueueError: If another worker already owns the project.
    """
    conn = _open_lock_db(_worker_lock_path(project_dir), timeout_s=timeout_s)
    try:
        _ = conn.execute(
            "CREATE TABLE IF NOT EXISTS worker_lease (id INTEGER PRIMARY KEY CHECK(id=1))"
        )
        _ = conn.execute("BEGIN EXCLUSIVE")
        yield
        _ = conn.execute("COMMIT")
    except sqlite3.OperationalError as error:
        with contextlib.suppress(sqlite3.Error):
            _ = conn.execute("ROLLBACK")
        msg = "An export worker already owns this project."
        raise ExportQueueError(_WORKER_BUSY, msg) from error
    finally:
        conn.close()


def is_worker_active(project_dir: Path) -> bool:
    """Return whether a worker lease is currently held, without side effects.

    Args:
        project_dir: The Studio project directory.

    Returns:
        True while a worker holds the lease.

    Raises:
        ExportQueueError: If the worker lease cannot be safely inspected.
    """
    path = _worker_lock_path(project_dir)
    if _is_link_like(path):
        msg = "Export worker lease must not be a symbolic link or junction."
        raise ExportQueueError(_EXPORT_QUEUE_INVALID, msg)
    if not path.exists():
        return False
    try:
        conn = sqlite3.connect(str(path), timeout=0.0, isolation_level=None)
    except sqlite3.Error as error:
        msg = f"Export worker lease could not be inspected: {error}"
        raise ExportQueueError(_EXPORT_QUEUE_IO, msg) from error
    try:
        try:
            _ = conn.execute("BEGIN EXCLUSIVE")
        except sqlite3.Error as error:
            if error.sqlite_errorcode in {sqlite3.SQLITE_BUSY, sqlite3.SQLITE_LOCKED}:
                return True
            msg = f"Export worker lease could not be inspected: {error}"
            raise ExportQueueError(_EXPORT_QUEUE_IO, msg) from error
        with contextlib.suppress(sqlite3.Error):
            _ = conn.execute("ROLLBACK")
        return False
    finally:
        conn.close()


def _require_project(project_dir: Path) -> ProjectDocument:
    """Read the current editable project or raise an explicit queue error.

    Args:
        project_dir: The Studio project directory.

    Returns:
        The current project document.

    Raises:
        ExportQueueError: If the project is missing or unsafe.
    """
    if not project_dir.is_dir():
        msg = "Export queue requires an existing project directory."
        raise ExportQueueError(_EXPORT_QUEUE_INVALID, msg)
    try:
        store = open_project_store(project_dir)
        project = store.read_project()
    except ProjectStoreError as error:
        raise ExportQueueError(error.code, error.message) from error
    if project is None:
        msg = "Export queue requires initialized editable project state."
        raise ExportQueueError(_EXPORT_QUEUE_INVALID, msg)
    return project


def _read_analysis_candidate_ids(project_dir: Path) -> set[str]:
    """Read current analysis candidate IDs for origin validation.

    Args:
        project_dir: The Studio project directory.

    Returns:
        The current candidate ID set.

    Raises:
        ExportQueueError: If the analysis artifact is invalid.
    """
    try:
        text = (project_dir / "analysis.json").read_text(encoding="utf-8")
        analysis = AnalysisDocument.model_validate_json(text)
    except (OSError, ValueError) as error:
        msg = f"Export queue analysis artifact is invalid: {error}"
        raise ExportQueueError(_EXPORT_QUEUE_INVALID, msg) from error
    return {candidate.candidate_id for candidate in analysis.candidates}


def _find_variant(project: ProjectDocument, variant_id: uuid.UUID) -> Variant | None:
    """Return one variant by ID or None when it was removed.

    Args:
        project: The current project document.
        variant_id: The variant identity.

    Returns:
        The variant, or None when removed.
    """
    return next((item for item in project.variants if item.variant_id == variant_id), None)


def _validate_job_ready(
    project: ProjectDocument, job: ExportJob, *, candidate_ids: set[str]
) -> Variant:
    """Validate one job against current evidence without rendering.

    Args:
        project: The current project document.
        job: The queued job.
        candidate_ids: The current analysis candidate IDs.

    Returns:
        The validated variant.

    Raises:
        ExportQueueError: If the variant is removed, stale, or needs review.
    """
    variant = _find_variant(project, job.variant_id)
    if variant is None:
        msg = f"Variant {job.variant_id} was removed; enqueue afresh."
        raise ExportQueueError(_VARIANT_NOT_FOUND, msg)
    if variant.status.value != "active":
        msg = f"Variant {variant.variant_id} requires review before export."
        raise ExportQueueError(_VARIANT_NOT_ACTIVE, msg)
    if (
        variant.origin.analysis_sha256 != project.analysis_sha256
        or variant.origin.transcript_sha256 != project.transcript_sha256
        or variant.origin.source_sha256 != project.source_sha256
        or variant.origin.source_candidate_id not in candidate_ids
    ):
        msg = "Variant origin does not match current B1 evidence; enqueue afresh."
        raise ExportQueueError(_ORIGIN_MISMATCH, msg)
    current = variant_revision(variant, project=project)
    if current != job.revision:
        msg = "Variant revision changed; enqueue afresh instead of rebounding."
        raise ExportQueueError(_REVISION_MISMATCH, msg)
    return variant


def _ensure_queue_for_project(
    project: ProjectDocument, existing: ExportQueueDocument | None
) -> ExportQueueDocument:
    """Return the existing queue or a fresh empty queue bound to the project.

    Args:
        project: The current project document.
        existing: The persisted queue, if any.

    Returns:
        The bound queue document.

    Raises:
        ExportQueueError: If the queue belongs to another project.
    """
    if existing is None:
        return ExportQueueDocument(project_id=project.project_id)
    if existing.project_id != project.project_id:
        msg = "Export queue belongs to a different analysis project."
        raise ExportQueueError(_EXPORT_QUEUE_INVALID, msg)
    return existing


def _build_new_jobs(
    project: ProjectDocument,
    variant_ids: list[uuid.UUID],
    *,
    candidate_ids: set[str],
    normalize_audio: bool,
) -> list[ExportJob]:
    """Build validated new jobs for one enqueue request.

    Args:
        project: The current project document.
        variant_ids: The requested variant identities in order.
        candidate_ids: The current analysis candidate IDs.
        normalize_audio: The explicit audio setting.

    Returns:
        The new pending jobs.

    Raises:
        ExportQueueError: If any variant is missing, stale, or needs review.
    """
    new_jobs: list[ExportJob] = []
    for variant_id in variant_ids:
        variant = _find_variant(project, variant_id)
        if variant is None:
            msg = f"Variant {variant_id} is not part of the editable project."
            raise ExportQueueError(_VARIANT_NOT_FOUND, msg)
        if variant.status.value != "active":
            msg = f"Variant {variant_id} requires review before export."
            raise ExportQueueError(_VARIANT_NOT_ACTIVE, msg)
        if (
            variant.origin.analysis_sha256 != project.analysis_sha256
            or variant.origin.transcript_sha256 != project.transcript_sha256
            or variant.origin.source_sha256 != project.source_sha256
            or variant.origin.source_candidate_id not in candidate_ids
        ):
            msg = "Variant origin does not match current B1 evidence."
            raise ExportQueueError(_ORIGIN_MISMATCH, msg)
        revision = variant_revision(variant, project=project)
        job_id = uuid.uuid4()
        new_jobs.append(
            ExportJob(
                job_id=job_id,
                variant_id=variant.variant_id,
                variant_name=variant.name,
                revision=revision,
                output_path=_unique_output_path(variant.name, job_id),
                normalize_audio=normalize_audio,
                state=ExportJobState.PENDING,
            )
        )
    return new_jobs


def enqueue_variants(
    project_dir: Path,
    *,
    variant_ids: list[uuid.UUID],
    normalize_audio: bool = True,
    timeout_s: float = _STATE_TIMEOUT_S,
) -> ExportQueueDocument:
    """Append one job per active variant revision in order.

    Args:
        project_dir: The Studio project directory.
        variant_ids: The requested variant identities in order.
        normalize_audio: The explicit audio setting.
        timeout_s: The bounded state-lock wait.

    Returns:
        The persisted queue with the new pending jobs.

    Raises:
        ExportQueueError: If any variant is missing, stale, or needs review.
    """
    if not variant_ids:
        msg = "Enqueue requires at least one variant ID."
        raise ExportQueueError(_EXPORT_QUEUE_INVALID, msg)
    project = _require_project(project_dir)
    candidate_ids = _read_analysis_candidate_ids(project_dir)
    new_jobs = _build_new_jobs(
        project,
        variant_ids,
        candidate_ids=candidate_ids,
        normalize_audio=normalize_audio,
    )
    with _state_locked(project_dir, timeout_s=timeout_s):
        existing = _read_queue_unlocked(project_dir)
        queue = _ensure_queue_for_project(project, existing)
        updated = queue.model_copy(update={"jobs": (*queue.jobs, *new_jobs)})
        _write_queue_unlocked(project_dir, updated)
        return updated


def request_stop(project_dir: Path, *, timeout_s: float = _STATE_TIMEOUT_S) -> ExportQueueDocument:
    """Persist a cooperative stop-after-current request.

    Args:
        project_dir: The Studio project directory.
        timeout_s: The bounded state-lock wait.

    Returns:
        The persisted queue with stop requested.
    """
    project = _require_project(project_dir)
    with _state_locked(project_dir, timeout_s=timeout_s):
        existing = _read_queue_unlocked(project_dir)
        queue = _ensure_queue_for_project(project, existing)
        updated = queue.model_copy(update={"stop_requested": True})
        _write_queue_unlocked(project_dir, updated)
        return updated


def _recover_running(queue: ExportQueueDocument) -> ExportQueueDocument:
    """Return the queue with abandoned running jobs marked interrupted.

    Args:
        queue: The queue snapshot to recover.

    Returns:
        The recovered queue.
    """
    recovered = tuple(
        job.model_copy(
            update={
                "state": ExportJobState.INTERRUPTED,
                "error": ("Worker ended without a terminal result; explicit retry is required."),
            }
        )
        if job.state is ExportJobState.RUNNING
        else job
        for job in queue.jobs
    )
    return queue.model_copy(update={"jobs": recovered})


def get_status(
    project_dir: Path, *, timeout_s: float = _STATE_TIMEOUT_S
) -> tuple[ExportQueueDocument, bool]:
    """Return the queue with abandoned running jobs recovered, plus worker state.

    Args:
        project_dir: The Studio project directory.
        timeout_s: The bounded state-lock wait.

    Returns:
        The queue snapshot and whether a worker is active.
    """
    project = _require_project(project_dir)
    with _state_locked(project_dir, timeout_s=timeout_s):
        active = is_worker_active(project_dir)
        existing = _read_queue_unlocked(project_dir)
        queue = _ensure_queue_for_project(project, existing)
        if not active and any(job.state is ExportJobState.RUNNING for job in queue.jobs):
            queue = _recover_running(queue)
            _write_queue_unlocked(project_dir, queue)
        return queue, active


def retry_job(
    project_dir: Path, *, job_id: uuid.UUID, timeout_s: float = _STATE_TIMEOUT_S
) -> ExportQueueDocument:
    """Return one failed/interrupted job to pending after revalidation.

    Args:
        project_dir: The Studio project directory.
        job_id: The job identity to retry.
        timeout_s: The bounded state-lock wait.

    Returns:
        The persisted queue with the job pending.

    Raises:
        ExportQueueError: If the job cannot be retried or is stale.
    """
    project = _require_project(project_dir)
    candidate_ids = _read_analysis_candidate_ids(project_dir)
    with _state_locked(project_dir, timeout_s=timeout_s):
        existing = _read_queue_unlocked(project_dir)
        queue = _ensure_queue_for_project(project, existing)
        index = next(
            (i for i, job in enumerate(queue.jobs) if job.job_id == job_id),
            None,
        )
        if index is None:
            msg = f"Job {job_id} is not part of the export queue."
            raise ExportQueueError(_JOB_NOT_FOUND, msg)
        job = queue.jobs[index]
        retryable = {ExportJobState.FAILED, ExportJobState.INTERRUPTED}
        if job.state not in retryable:
            msg = f"Only failed/interrupted jobs can be retried, not {job.state.value}."
            raise ExportQueueError(_JOB_INVALID, msg)
        _ = _validate_job_ready(project, job, candidate_ids=candidate_ids)
        retried = job.model_copy(update={"state": ExportJobState.PENDING, "error": None})
        jobs = list(queue.jobs)
        jobs[index] = retried
        updated = queue.model_copy(update={"jobs": tuple(jobs)})
        _write_queue_unlocked(project_dir, updated)
        return updated


def _finish_job(
    project_dir: Path, job: ExportJob, outcome: JobOutcome, *, timeout_s: float
) -> None:
    """Record one terminal job outcome without losing concurrent updates.

    Args:
        project_dir: The Studio project directory.
        job: The job to finish.
        outcome: The terminal outcome.
        timeout_s: The bounded state-lock wait.

    Raises:
        ExportQueueError: If the job is missing.
    """
    project = _require_project(project_dir)
    with _state_locked(project_dir, timeout_s=timeout_s):
        existing = _read_queue_unlocked(project_dir)
        queue = _ensure_queue_for_project(project, existing)
        jobs = list(queue.jobs)
        index = next((i for i, item in enumerate(jobs) if item.job_id == job.job_id), None)
        if index is None:
            msg = f"Job {job.job_id} is not part of the export queue."
            raise ExportQueueError(_JOB_NOT_FOUND, msg)
        finished = jobs[index].model_copy(
            update={
                "state": outcome.state,
                "error": outcome.error,
                "output_sha256": outcome.output_sha256,
                "width": outcome.width,
                "height": outcome.height,
            }
        )
        jobs[index] = finished
        updated = queue.model_copy(update={"jobs": tuple(jobs)})
        _write_queue_unlocked(project_dir, updated)


def _failure_outcome(
    error: FinalRenderError | ProjectStoreError | ExportQueueError | OSError | ValueError,
) -> JobOutcome:
    """Map one explicit render failure to a failed job outcome.

    Args:
        error: The explicit failure.

    Returns:
        The failed outcome.
    """
    if isinstance(error, FinalRenderError):
        state = (
            ExportJobState.INTERRUPTED
            if error.error_code == "render_cancelled"
            else ExportJobState.FAILED
        )
        return JobOutcome(state=state, error=f"{error.error_code}: {error.message}")
    if isinstance(error, (ProjectStoreError, ExportQueueError)):
        return JobOutcome(state=ExportJobState.FAILED, error=f"{error.code}: {error.message}")
    if isinstance(error, OSError):
        return JobOutcome(state=ExportJobState.FAILED, error=f"{_EXPORT_QUEUE_IO}: {error}")
    return JobOutcome(state=ExportJobState.FAILED, error=f"{_EXPORT_QUEUE_INVALID}: {error}")


def _next_pending(queue: ExportQueueDocument) -> ExportJob | None:
    """Return the next pending job in order, if any.

    Args:
        queue: The queue snapshot.

    Returns:
        The next pending job, or None when the queue is drained.
    """
    return next((job for job in queue.jobs if job.state is ExportJobState.PENDING), None)


def _prepare_run(project_dir: Path) -> None:
    """Recover jobs and reset stop while the caller owns state and worker locks.

    Args:
        project_dir: The Studio project directory.
    """
    project = _require_project(project_dir)
    existing = _read_queue_unlocked(project_dir)
    queue = _ensure_queue_for_project(project, existing)
    if any(job.state is ExportJobState.RUNNING for job in queue.jobs):
        queue = _recover_running(queue)
    queue = queue.model_copy(update={"stop_requested": False})
    _write_queue_unlocked(project_dir, queue)


def _take_next_job(
    project_dir: Path, *, state_timeout_s: float
) -> tuple[ExportQueueDocument, ExportJob | None, bool]:
    """Take the next pending job unless a stop was requested.

    Args:
        project_dir: The Studio project directory.
        state_timeout_s: The bounded state-lock wait.

    Returns:
        The queue snapshot, the next job (if any), and whether a stop was seen.
    """
    with _state_locked(project_dir, timeout_s=state_timeout_s):
        existing = _read_queue_unlocked(project_dir)
        current_project = _require_project(project_dir)
        queue = _ensure_queue_for_project(current_project, existing)
        if queue.stop_requested:
            return queue, None, True
        job = _next_pending(queue)
        if job is None:
            return queue, None, False
        running = job.model_copy(update={"state": ExportJobState.RUNNING})
        jobs = tuple(running if item.job_id == job.job_id else item for item in queue.jobs)
        updated = queue.model_copy(update={"jobs": jobs})
        _write_queue_unlocked(project_dir, updated)
        return updated, running, False


def _render_one_job(
    project_dir: Path,
    job: ExportJob,
    *,
    renderer: ExportRenderer,
    state_timeout_s: float,
) -> bool:
    """Render one job and record its terminal outcome; return success.

    Args:
        project_dir: The Studio project directory.
        job: The pending job snapshot.
        renderer: The existing final-render service.
        state_timeout_s: The bounded state-lock wait.

    Returns:
        True when the job completed.
    """
    try:
        fresh_project = _require_project(project_dir)
        candidate_ids = _read_analysis_candidate_ids(project_dir)
        _ = _validate_job_ready(fresh_project, job, candidate_ids=candidate_ids)
    except ExportQueueError as error:
        _finish_job(
            project_dir,
            job,
            JobOutcome(state=ExportJobState.FAILED, error=f"{error.code}: {error.message}"),
            timeout_s=state_timeout_s,
        )
        return False
    try:
        destination = project_dir / PurePosixPath(job.output_path).as_posix()
        result = renderer(
            FinalRenderRequest(
                project_dir=project_dir,
                variant_id=job.variant_id,
                destination=destination,
                normalize_audio=job.normalize_audio,
                expected_revision=job.revision,
                overwrite=False,
            )
        )
    except (FinalRenderError, ProjectStoreError, ExportQueueError, OSError, ValueError) as error:
        outcome = _failure_outcome(error)
        if outcome.state is ExportJobState.INTERRUPTED:
            _ = request_stop(project_dir, timeout_s=state_timeout_s)
        _finish_job(project_dir, job, outcome, timeout_s=state_timeout_s)
        return False
    try:
        output_hash = sha256_file(result.output_path)
    except OSError as error:
        _finish_job(
            project_dir,
            job,
            JobOutcome(state=ExportJobState.FAILED, error=f"{_EXPORT_QUEUE_IO}: {error}"),
            timeout_s=state_timeout_s,
        )
        return False
    _finish_job(
        project_dir,
        job,
        JobOutcome(
            state=ExportJobState.COMPLETED,
            error=None,
            output_sha256=output_hash,
            width=result.width,
            height=result.height,
        ),
        timeout_s=state_timeout_s,
    )
    return True


def run_pending(
    project_dir: Path,
    *,
    renderer: ExportRenderer,
    timeout_s: float = _WORKER_TIMEOUT_S,
    state_timeout_s: float = _STATE_TIMEOUT_S,
) -> tuple[ExportQueueDocument, QueueRunSummary]:
    """Run pending jobs in order, continuing after per-job failures.

    Args:
        project_dir: The Studio project directory.
        renderer: The existing final-render service.
        timeout_s: The bounded worker-lease wait.
        state_timeout_s: The bounded state-lock wait.

    Returns:
        The final queue snapshot and the aggregate summary.
    """
    _ = _require_project(project_dir)
    with contextlib.ExitStack() as lease:
        with _state_locked(project_dir, timeout_s=state_timeout_s):
            _ = lease.enter_context(_worker_held(project_dir, timeout_s=timeout_s))
            _prepare_run(project_dir)
        completed = 0
        failed = 0
        stopped = False
        while True:
            _, nxt, saw_stop = _take_next_job(project_dir, state_timeout_s=state_timeout_s)
            if saw_stop:
                stopped = True
                break
            if nxt is None:
                break
            if _render_one_job(
                project_dir, nxt, renderer=renderer, state_timeout_s=state_timeout_s
            ):
                completed += 1
            else:
                failed += 1
        with _state_locked(project_dir, timeout_s=state_timeout_s):
            existing = _read_queue_unlocked(project_dir)
            final_project = _require_project(project_dir)
            final_queue = _ensure_queue_for_project(final_project, existing)
        return final_queue, QueueRunSummary(completed=completed, failed=failed, stopped=stopped)


def queue_event_payload(queue: ExportQueueDocument, *, worker_active: bool) -> dict[str, object]:
    """Return the controlled JSON payload for one queue snapshot.

    Args:
        queue: The queue snapshot.
        worker_active: Whether a worker currently owns the project.

    Returns:
        The export_queue event payload.
    """
    return {
        "type": "export_queue",
        "queue": queue.model_dump(mode="json"),
        "worker_active": worker_active,
    }
