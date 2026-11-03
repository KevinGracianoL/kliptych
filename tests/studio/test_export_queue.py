"""Behavioral tests for Studio persistent export queue."""

import json
import signal
import sqlite3
from pathlib import Path
from typing import Literal, cast
from uuid import uuid4

import pytest
from pydantic import ValidationError

from kliptych.__main__ import main
from kliptych.gate.models import MediaInfo
from kliptych.hashing import sha256_bytes, sha256_file
from kliptych.studio import export_queue as eq
from kliptych.studio import final_render as frmod
from kliptych.studio.export_queue import (
    ExportJob,
    ExportJobState,
    ExportQueueDocument,
    ExportQueueError,
)
from kliptych.studio.final_render import FinalRenderError, FinalRenderRequest, FinalRenderResult
from kliptych.studio.models import (
    AnalysisCandidate,
    AnalysisDocument,
    AnalysisProvenance,
    AnalysisSource,
)
from kliptych.studio.project_models import ProjectStatus, Variant, variant_revision
from kliptych.studio.project_store import ProjectStore, open_project_store

_SOURCE_SHA = sha256_bytes(b"source")


def _analysis_doc(*, candidates: tuple[AnalysisCandidate, ...] = ()) -> AnalysisDocument:
    return AnalysisDocument(
        schema_version="1",
        project_id="a" * 32,
        project_name="clip",
        source=AnalysisSource(url="https://example.com/v", sha256=_SOURCE_SHA),
        provenance=AnalysisProvenance(transcription_version="1", segmenter_version="1"),
        candidates=candidates,
    )


def _candidate(*, candidate_id: str = "a" * 64) -> AnalysisCandidate:
    return AnalysisCandidate(
        candidate_id=candidate_id,
        recommended_start_ms=10_000,
        recommended_end_ms=30_000,
        score=0.9,
        excerpt="c",
    )


class _InitProbe:
    def __init__(self, width: int = 1920, height: int = 1080) -> None:
        self._width: int = width
        self._height: int = height

    def probe(self, path: Path) -> MediaInfo:
        _ = path
        return MediaInfo(
            format_name="mp4",
            duration_s=60.0,
            has_video=True,
            has_audio=True,
            width=self._width,
            height=self._height,
        )


class _FixedProbe:
    def __init__(self, width: int, height: int) -> None:
        self._width: int = width
        self._height: int = height

    def probe(self, path: Path) -> MediaInfo:
        _ = path
        return MediaInfo(
            format_name="mp4",
            duration_s=20.0,
            has_video=True,
            has_audio=True,
            width=self._width,
            height=self._height,
        )


class _StubRunner:
    def __init__(self) -> None:
        self.calls: int = 0

    def run(self, argv: tuple[str, ...], *, timeout_s: float, cwd: Path | None = None) -> object:
        self.calls += 1
        _ = timeout_s
        _ = cwd
        dest = Path(argv[-1])
        dest.parent.mkdir(parents=True, exist_ok=True)
        _ = dest.write_bytes(b"video-bytes-%d" % self.calls)
        return None


def _setup_project(tmp_path: Path) -> tuple[Path, ProjectStore, Variant]:
    candidate = _candidate()
    analysis = _analysis_doc(candidates=(candidate,))
    project_dir = tmp_path / "project"
    store = open_project_store(project_dir)
    transcript_json = (
        '{"words":[{"start_s":12.0,"end_s":14.0,"text":"hello world",'
        '"confidence":0.95,"token_id":1}],'
        '"language":"en","duration_s":60.0,"text":"hello world"}'
    )
    with store.stage() as stage:
        stage.write_analysis(analysis)
        stage.write_text("source.mp4", "source")
        stage.write_text("transcript.json", transcript_json)
        stage.write_text("candidates.json", "candidates")
    store = open_project_store(project_dir)
    _ = store.initialize_project(probe=_InitProbe())
    created = store.create_variant(candidate_id=candidate.candidate_id, name="V1")
    return project_dir, store, created.variants[0]


def _parse_cli_out(text: str) -> dict[str, object]:
    return cast("dict[str, object]", json.loads(text.strip().splitlines()[-1]))


def _disable_logging(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_setup_logging(*, verbose: bool = False, quiet: bool = False) -> None:
        _ = verbose
        _ = quiet

    monkeypatch.setattr("kliptych.__main__.setup_logging", fake_setup_logging)


class _OkRenderer:
    def __init__(self) -> None:
        self.calls: int = 0
        self.seen: list[FinalRenderRequest] = []

    def __call__(self, request: FinalRenderRequest) -> FinalRenderResult:
        self.calls += 1
        self.seen.append(request)
        assert request.destination is not None
        assert request.overwrite is False
        assert request.expected_revision is not None
        request.destination.parent.mkdir(parents=True, exist_ok=True)
        _ = request.destination.write_bytes(b"ok-mp4-%d" % self.calls)
        return FinalRenderResult(
            output_path=request.destination,
            variant_id=request.variant_id,
            duration_ms=20000,
            width=1080,
            height=1920,
            gate_verdict="pass",
            encoder_used="libx264",
            audio_normalized=request.normalize_audio,
        )


class _FailRenderer:
    def __init__(self, code: str = "gate_rejected") -> None:
        self._code: str = code
        self.calls: int = 0

    def __call__(self, request: FinalRenderRequest) -> FinalRenderResult:
        self.calls += 1
        _ = request
        msg = f"renderer failed {self.calls}"
        raise FinalRenderError(msg, error_code=self._code)


def _force_running_state(project_dir: Path, job_id_str: str) -> None:
    """Simulate a crash by marking one job running via public document IO."""
    path = project_dir / "export_queue.json"
    doc = ExportQueueDocument.model_validate_json(path.read_text(encoding="utf-8"))
    jobs = list(doc.jobs)
    for i, job in enumerate(jobs):
        if str(job.job_id) == job_id_str:
            jobs[i] = job.model_copy(
                update={
                    "state": ExportJobState.RUNNING,
                    "error": None,
                }
            )
    updated = doc.model_copy(update={"jobs": tuple(jobs)})
    _ = path.write_text(updated.model_dump_json(indent=2), encoding="utf-8")


def test_status_empty_project_returns_empty_queue(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _disable_logging(monkeypatch)
    project_dir, _, _ = _setup_project(tmp_path)
    code = main(
        [
            "--quiet",
            "studio",
            "export-queue",
            "--project-dir",
            str(project_dir),
            "--action",
            "status",
        ]
    )
    assert code == 0
    event = _parse_cli_out(capsys.readouterr().out)
    assert event["type"] == "export_queue"
    queue = cast("dict[str, object]", event["queue"])
    assert queue["schema_version"] == "1"
    assert queue["stop_requested"] is False
    assert queue["jobs"] == []
    assert event["worker_active"] is False
    assert event["schema_version"] == "1"


def test_enqueue_and_run_sequential_success(tmp_path: Path) -> None:
    project_dir, _, variant = _setup_project(tmp_path)
    doc = eq.enqueue_variants(project_dir, variant_ids=[variant.variant_id])
    assert len(doc.jobs) == 1
    assert doc.jobs[0].state is ExportJobState.PENDING
    assert doc.jobs[0].output_path.startswith("delivery/")
    assert doc.jobs[0].normalize_audio is True
    renderer = _OkRenderer()
    final, summary = eq.run_pending(project_dir, renderer=renderer)
    assert summary.completed == 1
    assert summary.failed == 0
    assert summary.stopped is False
    assert renderer.calls == 1
    assert final.jobs[0].state is ExportJobState.COMPLETED
    assert final.jobs[0].error is None
    assert final.jobs[0].output_sha256 is not None
    assert Path(project_dir / final.jobs[0].output_path).is_file()


def test_run_continues_after_failure(tmp_path: Path) -> None:
    project_dir, store, variant = _setup_project(tmp_path)
    duped = store.duplicate_variant(variant_id=variant.variant_id, name="Copy")
    ids = [v.variant_id for v in duped.variants]
    _ = eq.enqueue_variants(project_dir, variant_ids=ids)

    class _MixedRenderer:
        def __init__(self) -> None:
            self.calls: int = 0

        def __call__(self, request: FinalRenderRequest) -> FinalRenderResult:
            self.calls += 1
            assert request.destination is not None
            if self.calls == 1:
                msg = "gate bad"
                raise FinalRenderError(msg, error_code="gate_rejected")
            request.destination.parent.mkdir(parents=True, exist_ok=True)
            _ = request.destination.write_bytes(b"second-ok")
            return FinalRenderResult(
                output_path=request.destination,
                variant_id=request.variant_id,
                duration_ms=20000,
                width=1080,
                height=1920,
                gate_verdict="pass",
                encoder_used="libx264",
                audio_normalized=True,
            )

    renderer = _MixedRenderer()
    final, summary = eq.run_pending(project_dir, renderer=renderer)
    assert summary.completed == 1
    assert summary.failed == 1
    assert renderer.calls == 2
    states = [j.state for j in final.jobs]
    assert states == [ExportJobState.FAILED, ExportJobState.COMPLETED]


def test_stop_during_first_job(tmp_path: Path) -> None:
    project_dir, store, variant = _setup_project(tmp_path)
    duped = store.duplicate_variant(variant_id=variant.variant_id, name="Copy")
    ids = [v.variant_id for v in duped.variants]
    _ = eq.enqueue_variants(project_dir, variant_ids=ids)

    class _StopRenderer:
        def __init__(self, project_dir: Path) -> None:
            self._project_dir: Path = project_dir
            self.calls: int = 0

        def __call__(self, request: FinalRenderRequest) -> FinalRenderResult:
            self.calls += 1
            _ = eq.request_stop(self._project_dir)
            assert request.destination is not None
            request.destination.parent.mkdir(parents=True, exist_ok=True)
            _ = request.destination.write_bytes(b"first-done")
            return FinalRenderResult(
                output_path=request.destination,
                variant_id=request.variant_id,
                duration_ms=20000,
                width=1080,
                height=1920,
                gate_verdict="pass",
                encoder_used="libx264",
                audio_normalized=True,
            )

    renderer = _StopRenderer(project_dir)
    final, summary = eq.run_pending(project_dir, renderer=renderer)
    assert renderer.calls == 1
    assert summary.completed == 1
    assert summary.stopped is True
    assert final.stop_requested is True
    assert final.jobs[1].state is ExportJobState.PENDING


def test_status_active_worker_not_marked_interrupted(tmp_path: Path) -> None:
    project_dir, _, variant = _setup_project(tmp_path)
    _ = eq.enqueue_variants(project_dir, variant_ids=[variant.variant_id])
    seen: dict[str, object] = {}

    class _CheckRenderer:
        def __call__(self, request: FinalRenderRequest) -> FinalRenderResult:
            queue, active = eq.get_status(project_dir)
            seen["active"] = active
            seen["running"] = [j.state.value for j in queue.jobs]
            assert request.destination is not None
            request.destination.parent.mkdir(parents=True, exist_ok=True)
            _ = request.destination.write_bytes(b"active-ok")
            return FinalRenderResult(
                output_path=request.destination,
                variant_id=request.variant_id,
                duration_ms=20000,
                width=1080,
                height=1920,
                gate_verdict="pass",
                encoder_used="libx264",
                audio_normalized=True,
            )

    renderer = _CheckRenderer()
    final, _ = eq.run_pending(project_dir, renderer=renderer)
    assert seen["active"] is True
    running = cast("list[str]", seen["running"])
    assert "running" in running
    assert final.jobs[0].state is ExportJobState.COMPLETED


def test_worker_contention_rejects_second_run(tmp_path: Path) -> None:
    project_dir, _, variant = _setup_project(tmp_path)
    _ = eq.enqueue_variants(project_dir, variant_ids=[variant.variant_id])
    seen: dict[str, object] = {"nested": "missing"}

    class _ContendingRenderer:
        def __call__(self, request: FinalRenderRequest) -> FinalRenderResult:
            _ = request
            try:
                _ = eq.run_pending(project_dir, renderer=_OkRenderer(), timeout_s=0.2)
            except ExportQueueError as error:
                seen["nested"] = error.code
            assert request.destination is not None
            request.destination.parent.mkdir(parents=True, exist_ok=True)
            _ = request.destination.write_bytes(b"outer-ok")
            return FinalRenderResult(
                output_path=request.destination,
                variant_id=request.variant_id,
                duration_ms=20000,
                width=1080,
                height=1920,
                gate_verdict="pass",
                encoder_used="libx264",
                audio_normalized=True,
            )

    renderer = _ContendingRenderer()
    _, summary = eq.run_pending(project_dir, renderer=renderer)
    assert summary.completed == 1
    assert seen["nested"] == "worker_busy"


def test_restart_marks_abandoned_interrupted(tmp_path: Path) -> None:
    project_dir, _, variant = _setup_project(tmp_path)
    queue = eq.enqueue_variants(project_dir, variant_ids=[variant.variant_id])
    job = queue.jobs[0]
    _force_running_state(project_dir, str(job.job_id))
    recovered, active = eq.get_status(project_dir)
    assert active is False
    assert recovered.jobs[0].state is ExportJobState.INTERRUPTED
    retried = eq.retry_job(project_dir, job_id=job.job_id)
    assert retried.jobs[0].state is ExportJobState.PENDING


def test_retry_rejects_stale_revision(tmp_path: Path) -> None:
    project_dir, store, variant = _setup_project(tmp_path)
    queue = eq.enqueue_variants(project_dir, variant_ids=[variant.variant_id])
    job = queue.jobs[0]
    _ = eq.run_pending(project_dir, renderer=_FailRenderer())
    _ = store.set_captions(variant_id=variant.variant_id, enabled=True)
    with pytest.raises(ExportQueueError, match="revision_mismatch"):
        _ = eq.retry_job(project_dir, job_id=job.job_id)


def test_stale_job_fails_without_render(tmp_path: Path) -> None:
    project_dir, store, variant = _setup_project(tmp_path)
    _ = eq.enqueue_variants(project_dir, variant_ids=[variant.variant_id])
    _ = store.set_captions(variant_id=variant.variant_id, enabled=True)

    class _NoCallRenderer:
        def __init__(self) -> None:
            self.calls: int = 0

        def __call__(self, request: FinalRenderRequest) -> FinalRenderResult:
            self.calls += 1
            _ = request
            msg = "renderer must not be called for stale job"
            raise AssertionError(msg)

    renderer = _NoCallRenderer()
    final, summary = eq.run_pending(project_dir, renderer=renderer)
    assert renderer.calls == 0
    assert summary.failed == 1
    assert summary.completed == 0
    assert final.jobs[0].state is ExportJobState.FAILED
    assert "revision_mismatch" in (final.jobs[0].error or "")


def test_removed_variant_fails_without_render(tmp_path: Path) -> None:
    project_dir, _, variant = _setup_project(tmp_path)
    _ = eq.enqueue_variants(project_dir, variant_ids=[variant.variant_id])
    project = open_project_store(project_dir).read_project()
    assert project is not None
    pruned = project.model_copy(update={"variants": ()})
    open_project_store(project_dir).write_project(pruned)

    class _NeverRenderer:
        def __init__(self) -> None:
            self.calls: int = 0

        def __call__(self, request: FinalRenderRequest) -> FinalRenderResult:
            self.calls += 1
            _ = request
            msg = "must not render removed variant"
            raise AssertionError(msg)

    renderer = _NeverRenderer()
    final, summary = eq.run_pending(project_dir, renderer=renderer)
    assert renderer.calls == 0
    assert summary.failed == 1
    assert "not_found" in (final.jobs[0].error or "")


def test_malformed_and_link_rejected(tmp_path: Path) -> None:
    project_dir, _, _ = _setup_project(tmp_path)
    qpath = project_dir / "export_queue.json"
    _ = qpath.write_text("{not-json", encoding="utf-8")
    with pytest.raises(ExportQueueError, match="export_queue_invalid"):
        _ = eq.get_status(project_dir)
    _ = qpath.write_text('{"schema_version":"99","project_id":"x","jobs":[]}', encoding="utf-8")
    with pytest.raises(ExportQueueError, match="export_queue_invalid"):
        _ = eq.get_status(project_dir)
    _ = qpath.unlink()
    _ = qpath.write_text('{"schema_version":"1","project_id":"x","jobs":[]}', encoding="utf-8")
    link_target = tmp_path / "outside.json"
    _ = link_target.write_text("{}", encoding="utf-8")
    _ = qpath.unlink()
    qpath.symlink_to(link_target)
    with pytest.raises(ExportQueueError, match="export_queue_invalid"):
        _ = eq.get_status(project_dir)


def test_bad_output_path_rejected() -> None:
    with pytest.raises(ValidationError):
        _ = ExportJob(
            job_id=uuid4(),
            variant_id=uuid4(),
            variant_name="V",
            revision="b" * 64,
            output_path="/abs/path.mp4",
            normalize_audio=True,
            state=ExportJobState.PENDING,
        )


def test_no_overwrite_atomic(tmp_path: Path) -> None:
    project_dir, _, variant = _setup_project(tmp_path)
    probe = _FixedProbe(width=1080, height=1920)
    runner = _StubRunner()
    first = frmod.render_final_variant(
        frmod.FinalRenderRequest(project_dir=project_dir, variant_id=variant.variant_id),
        probe=probe,
        runner=runner,
    )
    assert first.output_path.is_file()
    before = first.output_path.read_bytes()
    dest = first.output_path
    with pytest.raises(frmod.FinalRenderError) as exc:
        _ = frmod.render_final_variant(
            frmod.FinalRenderRequest(
                project_dir=project_dir,
                variant_id=variant.variant_id,
                destination=dest,
                overwrite=False,
            ),
            probe=_FixedProbe(width=1080, height=1920),
            runner=_StubRunner(),
        )
    assert exc.value.error_code == "destination_exists"
    assert dest.read_bytes() == before


def test_ambiguous_recovery_never_overwrites(tmp_path: Path) -> None:
    project_dir, _, variant = _setup_project(tmp_path)
    queue = eq.enqueue_variants(project_dir, variant_ids=[variant.variant_id])
    job = queue.jobs[0]
    dest = project_dir / job.output_path
    dest.parent.mkdir(parents=True, exist_ok=True)
    _ = dest.write_bytes(b"published-before-crash")
    before = dest.read_bytes()
    _force_running_state(project_dir, str(job.job_id))
    recovered, active = eq.get_status(project_dir)
    assert active is False
    assert recovered.jobs[0].state is ExportJobState.INTERRUPTED
    _ = eq.retry_job(project_dir, job_id=job.job_id)
    with pytest.raises(frmod.FinalRenderError) as exc_info:
        _ = frmod.render_final_variant(
            frmod.FinalRenderRequest(
                project_dir=project_dir,
                variant_id=variant.variant_id,
                destination=dest,
                overwrite=False,
            ),
            probe=_FixedProbe(width=1080, height=1920),
            runner=_StubRunner(),
        )
    assert exc_info.value.error_code == "destination_exists"
    assert dest.read_bytes() == before


def test_revision_helper_canonical(tmp_path: Path) -> None:
    project_dir, _, variant = _setup_project(tmp_path)
    project = open_project_store(project_dir).read_project()
    assert project is not None
    first = variant_revision(variant, project=project)
    second = variant_revision(variant, project=project)
    assert first == second
    assert len(first) == 64
    other = variant.model_copy(update={"name": "Other"})
    assert variant_revision(other, project=project) != first


def test_cli_smoke_and_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _disable_logging(monkeypatch)
    project_dir, _, variant = _setup_project(tmp_path)

    def _fake_render(
        request: FinalRenderRequest,
        *,
        probe: object = None,
        runner: object = None,
        ffmpeg: str = "ffmpeg",
        nvenc_available: bool = False,
        gate: object | None = None,
    ) -> FinalRenderResult:
        _ = (probe, runner, ffmpeg, nvenc_available, gate)
        assert request.destination is not None
        assert request.overwrite is False
        assert request.expected_revision is not None
        request.destination.parent.mkdir(parents=True, exist_ok=True)
        _ = request.destination.write_bytes(b"cli-ok")
        return FinalRenderResult(
            output_path=request.destination,
            variant_id=request.variant_id,
            duration_ms=20000,
            width=1080,
            height=1920,
            gate_verdict="pass",
            encoder_used="libx264",
            audio_normalized=request.normalize_audio,
        )

    monkeypatch.setattr("kliptych.__main__.render_final_variant", _fake_render)
    code = main(
        [
            "--quiet",
            "studio",
            "export-queue",
            "--project-dir",
            str(project_dir),
            "--action",
            "enqueue",
            "--variant-id",
            str(variant.variant_id),
        ]
    )
    assert code == 0
    event = _parse_cli_out(capsys.readouterr().out)
    assert event["type"] == "export_queue"
    queue = cast("dict[str, object]", event["queue"])
    jobs = cast("list[dict[str, object]]", queue["jobs"])
    assert len(jobs) == 1
    assert jobs[0]["state"] == "pending"
    assert jobs[0]["error"] is None
    job_id = cast("str", jobs[0]["job_id"])
    code = main(
        ["--quiet", "studio", "export-queue", "--project-dir", str(project_dir), "--action", "run"]
    )
    assert code == 0
    event = _parse_cli_out(capsys.readouterr().out)
    queue = cast("dict[str, object]", event["queue"])
    jobs = cast("list[dict[str, object]]", queue["jobs"])
    assert jobs[0]["state"] == "completed"
    code = main(
        [
            "--quiet",
            "studio",
            "export-queue",
            "--project-dir",
            str(project_dir),
            "--action",
            "retry",
            "--job-id",
            job_id,
        ]
    )
    assert code == 1
    event = _parse_cli_out(capsys.readouterr().out)
    assert event["type"] == "error"
    code = main(
        ["--quiet", "studio", "export-queue", "--project-dir", str(project_dir), "--action", "stop"]
    )
    assert code == 0


def test_queue_preserved_across_reanalysis(tmp_path: Path) -> None:
    project_dir, _, variant = _setup_project(tmp_path)
    before = eq.enqueue_variants(project_dir, variant_ids=[variant.variant_id])
    assert len(before.jobs) == 1
    store = open_project_store(project_dir)
    analysis_text = (project_dir / "analysis.json").read_text(encoding="utf-8")
    analysis = AnalysisDocument.model_validate_json(analysis_text)
    with store.stage() as stage:
        stage.write_analysis(analysis)
        stage.write_text("source.mp4", "source")
        stage.write_text("transcript.json", (project_dir / "transcript.json").read_text())
        stage.write_text("candidates.json", "candidates")
    after_path = project_dir / "export_queue.json"
    assert after_path.is_file()
    after = eq.get_status(project_dir)[0]
    assert len(after.jobs) == 1
    assert after.jobs[0].job_id == before.jobs[0].job_id
    _ = sha256_file(after_path)


def test_structural_guards() -> None:
    src = Path(str(eq.__file__)).read_text(encoding="utf-8")
    assert "sqlite3" in src
    assert "getpid" not in src
    assert "O_EXCL" not in src
    assert "time.sleep" not in src
    fr_src = Path(str(frmod.__file__)).read_text(encoding="utf-8")
    assert "os.link" in fr_src
    store_src = (Path(str(eq.__file__)).parent / "project_store.py").read_text(encoding="utf-8")
    assert "export_queue.json" in store_src


def test_output_path_validator_branches() -> None:
    for bad in [
        "delivery/../evil.mp4",
        "clip.mp4",
        "delivery/clip.mkv",
        "delivery\\clip.mp4",
    ]:
        with pytest.raises(ValidationError):
            _ = ExportJob(
                job_id=uuid4(),
                variant_id=uuid4(),
                variant_name="V",
                revision="b" * 64,
                output_path=bad,
                normalize_audio=True,
                state=ExportJobState.PENDING,
            )


def test_job_state_error_agreement() -> None:
    _ = ExportJob(
        job_id=uuid4(),
        variant_id=uuid4(),
        variant_name="V",
        revision="b" * 64,
        output_path="delivery/v-abc12345.mp4",
        normalize_audio=True,
        state=ExportJobState.COMPLETED,
        error=None,
    )
    cases: list[tuple[ExportJobState, str | None]] = [
        (ExportJobState.COMPLETED, "oops"),
        (ExportJobState.PENDING, "oops"),
        (ExportJobState.FAILED, None),
        (ExportJobState.INTERRUPTED, None),
    ]
    for state, error in cases:
        with pytest.raises(ValidationError):
            _ = ExportJob(
                job_id=uuid4(),
                variant_id=uuid4(),
                variant_name="V",
                revision="b" * 64,
                output_path="delivery/v-abc12345.mp4",
                normalize_audio=True,
                state=state,
                error=error,
            )


def test_lock_and_project_error_paths(tmp_path: Path) -> None:
    project_dir, _, variant = _setup_project(tmp_path)
    _ = eq.enqueue_variants(project_dir, variant_ids=[variant.variant_id])
    state_db = project_dir / ".export_queue_state.db"
    if not state_db.exists():
        conn = sqlite3.connect(str(state_db))
        _ = conn.execute(
            "CREATE TABLE IF NOT EXISTS state_lock (id INTEGER PRIMARY KEY CHECK(id=1))"
        )
        conn.close()
    link_target = tmp_path / "outside.db"
    _ = link_target.write_bytes(b"x")
    worker_db = project_dir / ".export_worker.db"
    if worker_db.exists():
        _ = worker_db.unlink()
    worker_db.symlink_to(link_target)
    with pytest.raises(ExportQueueError, match="symbolic link"):
        _ = eq.is_worker_active(project_dir)
    _ = worker_db.unlink()
    with pytest.raises(ExportQueueError, match="existing project directory"):
        _ = eq.get_status(tmp_path / "missing-project")
    bad_dir = tmp_path / "bad-project"
    _ = bad_dir.mkdir()
    _ = (bad_dir / "analysis.json").write_text("{}", encoding="utf-8")
    _ = (bad_dir / "source.mp4").write_text("x", encoding="utf-8")
    _ = (bad_dir / "transcript.json").write_text("x", encoding="utf-8")
    with pytest.raises(ExportQueueError, match="project_directory_invalid"):
        _ = eq.enqueue_variants(bad_dir, variant_ids=[uuid4()])


def test_state_busy_and_retry_invalid(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    project_dir, _, variant = _setup_project(tmp_path)
    queue = eq.enqueue_variants(project_dir, variant_ids=[variant.variant_id])
    job = queue.jobs[0]
    with pytest.raises(ExportQueueError, match="job_invalid"):
        _ = eq.retry_job(project_dir, job_id=job.job_id)
    with pytest.raises(ExportQueueError, match="job_not_found"):
        _ = eq.retry_job(project_dir, job_id=uuid4())

    def _boom(*_args: object, **_kwargs: object) -> object:
        msg = "locked"
        raise sqlite3.OperationalError(msg)

    class _BusyConn:
        def __init__(self) -> None:
            self.calls: int = 0

        def execute(self, _sql: object, *_args: object, **_kwargs: object) -> object:
            self.calls += 1
            msg = "locked"
            raise sqlite3.OperationalError(msg)

        def close(self) -> None:
            self.calls += 1

    def _busy_connect(*_args: object, **_kwargs: object) -> object:
        return _BusyConn()

    _ = _boom
    monkeypatch.setattr("sqlite3.connect", _busy_connect)
    with pytest.raises(ExportQueueError, match=r"export_queue_busy|worker_busy"):
        _ = eq.get_status(project_dir)


def test_run_recovers_running_at_start(tmp_path: Path) -> None:
    project_dir, _, variant = _setup_project(tmp_path)
    queue = eq.enqueue_variants(project_dir, variant_ids=[variant.variant_id])
    _force_running_state(project_dir, str(queue.jobs[0].job_id))
    final, summary = eq.run_pending(project_dir, renderer=_OkRenderer())
    assert summary.failed == 0
    assert summary.completed == 0
    assert any(j.state is ExportJobState.INTERRUPTED for j in final.jobs)


def test_status_does_not_interrupt_worker_starting_before_state_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project_dir, _, variant = _setup_project(tmp_path)
    queue = eq.enqueue_variants(project_dir, variant_ids=[variant.variant_id])
    original_connect = sqlite3.connect
    workers: list[sqlite3.Connection] = []

    def connect(
        database: str,
        *,
        timeout: float,
        isolation_level: Literal["DEFERRED", "EXCLUSIVE", "IMMEDIATE"] | None,
    ) -> sqlite3.Connection:
        if database.endswith(".export_queue_state.db") and not workers:
            worker = original_connect(str(project_dir / ".export_worker.db"), isolation_level=None)
            _ = worker.execute("CREATE TABLE worker_lease (id INTEGER PRIMARY KEY)")
            _ = worker.execute("BEGIN EXCLUSIVE")
            workers.append(worker)
            running = queue.jobs[0].model_copy(update={"state": ExportJobState.RUNNING})
            _ = (project_dir / "export_queue.json").write_text(
                queue.model_copy(update={"jobs": (running,)}).model_dump_json(), encoding="utf-8"
            )
        return original_connect(database, timeout=timeout, isolation_level=isolation_level)

    monkeypatch.setattr(sqlite3, "connect", connect)
    try:
        status, active = eq.get_status(project_dir)
        assert status.jobs[0].state is ExportJobState.RUNNING
        assert active is True
    finally:
        for worker in workers:
            worker.close()


def test_queue_cannot_render_through_linked_delivery(tmp_path: Path) -> None:
    project_dir, _, variant = _setup_project(tmp_path)
    queue = eq.enqueue_variants(project_dir, variant_ids=[variant.variant_id])
    external = tmp_path / "external"
    external.mkdir()
    (project_dir / "delivery").symlink_to(external, target_is_directory=True)
    runner = _StubRunner()
    request = FinalRenderRequest(
        project_dir,
        variant.variant_id,
        destination=project_dir / queue.jobs[0].output_path,
        overwrite=False,
    )
    with pytest.raises(FinalRenderError, match=r"destination|directory"):
        _ = frmod.render_final_variant(request, probe=_FixedProbe(1080, 1920), runner=runner)
    assert runner.calls == 0
    assert list(external.iterdir()) == []


@pytest.mark.parametrize("linked", [False, True])
def test_worker_status_does_not_report_idle_when_lease_cannot_be_verified(
    tmp_path: Path, *, linked: bool
) -> None:
    project_dir, _, _ = _setup_project(tmp_path)
    lease = project_dir / ".export_worker.db"
    if linked:
        lease.symlink_to(tmp_path / "missing-lease")
    else:
        _ = lease.write_bytes(b"not a sqlite database")
    with pytest.raises(ExportQueueError):
        _ = eq.is_worker_active(project_dir)


def test_review_project_ancestor_alias_does_not_block_export(tmp_path: Path) -> None:
    real_parent = tmp_path / "real-parent"
    real_parent.mkdir()
    project_dir, _, variant = _setup_project(real_parent)
    alias = tmp_path / "alias"
    alias.symlink_to(real_parent, target_is_directory=True)
    linked_project = alias / project_dir.name
    result = frmod.render_final_variant(
        FinalRenderRequest(
            linked_project,
            variant.variant_id,
            destination=linked_project / "delivery" / "queued.mp4",
            overwrite=False,
        ),
        probe=_FixedProbe(1080, 1920),
        runner=_StubRunner(),
    )
    assert result.output_path.is_file()
    assert result.output_path.is_relative_to(project_dir.resolve())


def test_review_reanalysis_preserves_completed_export(tmp_path: Path) -> None:
    project_dir, store, variant = _setup_project(tmp_path)
    _ = eq.enqueue_variants(project_dir, variant_ids=[variant.variant_id])
    before, _ = eq.run_pending(project_dir, renderer=_OkRenderer())
    output = project_dir / before.jobs[0].output_path
    original = output.read_bytes()
    analysis = AnalysisDocument.model_validate_json(
        (project_dir / "analysis.json").read_text(encoding="utf-8")
    )
    with store.stage() as stage:
        stage.write_analysis(analysis)
        stage.write_text("source.mp4", "source")
        stage.write_text("transcript.json", (project_dir / "transcript.json").read_text())
        stage.write_text("candidates.json", "candidates")
    after, _ = eq.get_status(project_dir)
    assert after.jobs[0].state is ExportJobState.COMPLETED
    assert output.read_bytes() == original


def test_review_termination_interrupts_job_and_stops_worker(tmp_path: Path) -> None:
    project_dir, _, variant = _setup_project(tmp_path)
    _ = eq.enqueue_variants(project_dir, variant_ids=[variant.variant_id, variant.variant_id])
    calls: list[FinalRenderRequest] = []

    def render(request: FinalRenderRequest) -> FinalRenderResult:
        calls.append(request)
        if len(calls) == 1:
            previous = signal.signal(signal.SIGTERM, frmod.signal_cleanup_handler)
            try:
                signal.raise_signal(signal.SIGTERM)
            finally:
                _ = signal.signal(signal.SIGTERM, previous)
        return _OkRenderer()(request)

    queue, summary = eq.run_pending(project_dir, renderer=render)
    assert len(calls) == 1
    assert summary.stopped
    assert queue.jobs[0].state is ExportJobState.INTERRUPTED
    assert queue.jobs[1].state is ExportJobState.PENDING


def test_review_status_does_not_create_a_missing_project(tmp_path: Path) -> None:
    missing = tmp_path / "missing"
    with pytest.raises(ExportQueueError):
        _ = eq.get_status(missing)
    assert not missing.exists()


def test_stop_after_worker_becomes_visible_is_not_lost_during_startup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project_dir, _, variant = _setup_project(tmp_path)
    _ = eq.enqueue_variants(project_dir, variant_ids=[variant.variant_id])
    original_connect = sqlite3.connect
    requested = False

    def connect(
        database: str,
        *,
        timeout: float,
        isolation_level: Literal["DEFERRED", "EXCLUSIVE", "IMMEDIATE"] | None,
    ) -> sqlite3.Connection:
        nonlocal requested
        if (
            database.endswith(".export_queue_state.db")
            and not requested
            and eq.is_worker_active(project_dir)
        ):
            requested = True
            _ = eq.request_stop(project_dir)
        return original_connect(database, timeout=timeout, isolation_level=isolation_level)

    monkeypatch.setattr(sqlite3, "connect", connect)
    renderer = _OkRenderer()
    queue, summary = eq.run_pending(project_dir, renderer=renderer)
    assert requested
    assert summary.stopped
    assert renderer.calls == 0
    assert queue.jobs[0].state is ExportJobState.PENDING


def test_enqueue_rejects_origin_mismatch_after_reanalysis(tmp_path: Path) -> None:
    project_dir, _, variant = _setup_project(tmp_path)
    project = open_project_store(project_dir).read_project()
    assert project is not None
    bad_origin = variant.origin.model_copy(update={"source_candidate_id": "c" * 64})
    bad_variant = variant.model_copy(update={"origin": bad_origin})
    bad_project = project.model_copy(update={"variants": (bad_variant,)})
    _ = (project_dir / "project.json").write_text(
        bad_project.model_dump_json(indent=2), encoding="utf-8"
    )
    with pytest.raises(ExportQueueError, match="origin_mismatch"):
        _ = eq.enqueue_variants(project_dir, variant_ids=[variant.variant_id])


def test_queue_project_mismatch_rejected(tmp_path: Path) -> None:
    project_dir, _, variant = _setup_project(tmp_path)
    _ = eq.enqueue_variants(project_dir, variant_ids=[variant.variant_id])
    path = project_dir / "export_queue.json"
    doc = eq.ExportQueueDocument.model_validate_json(path.read_text(encoding="utf-8"))
    bad = doc.model_copy(update={"project_id": "different"})
    _ = path.write_text(bad.model_dump_json(indent=2), encoding="utf-8")
    with pytest.raises(ExportQueueError, match="different analysis project"):
        _ = eq.get_status(project_dir)


def test_queue_unique_identity_rejected() -> None:
    job_id = uuid4()
    variant_id = uuid4()
    job = ExportJob(
        job_id=job_id,
        variant_id=variant_id,
        variant_name="V",
        revision="b" * 64,
        output_path="delivery/v-abc12345.mp4",
        normalize_audio=True,
        state=ExportJobState.PENDING,
    )
    other = job.model_copy(update={"job_id": uuid4()})
    with pytest.raises(ValidationError):
        _ = eq.ExportQueueDocument(
            project_id="p",
            jobs=(job, other),
        )
    dup_id = job.model_copy(update={"output_path": "delivery/other-abc12345.mp4"})
    with pytest.raises(ValidationError):
        _ = eq.ExportQueueDocument(project_id="p", jobs=(job, dup_id))


def test_io_failure_returns_explicit_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    project_dir, _, variant = _setup_project(tmp_path)

    def boom(_destination: Path, _content: str) -> None:
        msg = "disk full"
        raise OSError(msg)

    monkeypatch.setattr("kliptych.studio.export_queue._atomic_write_text", boom)
    with pytest.raises(ExportQueueError, match="export_queue_io"):
        _ = eq.enqueue_variants(project_dir, variant_ids=[variant.variant_id])


def test_enqueue_rejects_bad_variant_and_review(tmp_path: Path) -> None:
    project_dir, _, variant = _setup_project(tmp_path)
    with pytest.raises(ExportQueueError, match="variant_not_found"):
        _ = eq.enqueue_variants(project_dir, variant_ids=[uuid4()])
    project = open_project_store(project_dir).read_project()
    assert project is not None
    reviewed_variant = variant.model_copy(update={"status": ProjectStatus.REQUIRES_REVIEW})
    reviewed_doc = project.model_copy(update={"variants": (reviewed_variant,)})
    open_project_store(project_dir).write_project(reviewed_doc)
    with pytest.raises(ExportQueueError, match="variant_not_active"):
        _ = eq.enqueue_variants(project_dir, variant_ids=[variant.variant_id])


def test_final_render_revision_mismatch_and_overwrite_default(tmp_path: Path) -> None:
    project_dir, _, variant = _setup_project(tmp_path)
    req = frmod.FinalRenderRequest(project_dir=project_dir, variant_id=variant.variant_id)
    assert req.overwrite is True
    assert req.expected_revision is None
    with pytest.raises(frmod.FinalRenderError) as exc_info:
        _ = frmod.render_final_variant(
            frmod.FinalRenderRequest(
                project_dir=project_dir,
                variant_id=variant.variant_id,
                expected_revision="0" * 64,
            ),
            probe=_FixedProbe(width=1080, height=1920),
            runner=_StubRunner(),
        )
    assert exc_info.value.error_code == "revision_mismatch"
