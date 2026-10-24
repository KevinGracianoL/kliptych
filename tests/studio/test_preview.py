"""Behavioral tests for exact-frame B2.2 JPEG composition previews."""

import os
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import cast, override
from uuid import UUID

import pytest

from kliptych.environment import CommandResult, SubprocessRunner
from kliptych.gate.models import MediaInfo
from kliptych.gate.probe import FFprobeProbe, MediaProbe, ProbeError
from kliptych.hashing import sha256_file
from kliptych.studio import preview as _preview_module
from kliptych.studio.models import (
    AnalysisCandidate,
    AnalysisDocument,
    AnalysisProvenance,
    AnalysisSource,
)
from kliptych.studio.preview import (
    PreviewError,
    PreviewRequest,
    generate_preview,
)
from kliptych.studio.project_models import (
    ProjectStatus,
    VariantOrigin,
    VariantRange,
    create_initial_variant,
)
from kliptych.studio.project_store import open_project_store


def _private(name: str) -> object:
    return cast("object", getattr(_preview_module, name))


class _Probe(MediaProbe):
    @override
    def probe(self, path: Path) -> MediaInfo:
        if path.suffix == ".jpg":
            if path.read_bytes() == b"png-cache":
                return MediaInfo(
                    format_name="png_pipe",
                    has_video=True,
                    has_audio=False,
                    duration_s=None,
                    width=720,
                    height=1280,
                )
            if path.read_bytes() != b"\xff\xd8preview\xff\xd9":
                msg = "invalid jpeg fixture"
                raise ProbeError(msg)
            is_source = "-source" in path.name
            return MediaInfo(
                format_name="image2",
                has_video=True,
                has_audio=False,
                duration_s=None,
                width=1280 if is_source else 720,
                height=720 if is_source else 1280,
            )
        assert path.name == "source.mp4"
        return MediaInfo(
            format_name="mp4",
            has_video=True,
            has_audio=True,
            duration_s=60.0,
            width=1920,
            height=1080,
        )


class _NarrowProbe(_Probe):
    @override
    def probe(self, path: Path) -> MediaInfo:
        media = super().probe(path)
        return media.model_copy(update={"width": 1})


class _Runner:
    def __init__(self) -> None:
        self.argv: tuple[str, ...] | None = None
        self.invocations: list[tuple[str, ...]] = []

    def run(self, argv: Sequence[str], *, timeout_s: float = 30.0) -> CommandResult:
        _ = timeout_s
        self.argv = tuple(argv)
        self.invocations.append(self.argv)
        destination = Path(argv[-1])
        _ = destination.write_bytes(b"\xff\xd8preview\xff\xd9")
        return CommandResult(ok=True)


class _FailsSecondRunner(_Runner):
    @override
    def run(self, argv: Sequence[str], *, timeout_s: float = 30.0) -> CommandResult:
        if len(self.invocations) == 1:
            self.invocations.append(tuple(argv))
            return CommandResult(ok=False)
        return super().run(argv, timeout_s=timeout_s)


def _analysis() -> AnalysisDocument:
    candidate = AnalysisCandidate(
        candidate_id="a" * 64,
        recommended_start_ms=10_000,
        recommended_end_ms=30_000,
        score=0.9,
        excerpt="Candidate",
    )
    return AnalysisDocument(
        schema_version="1",
        project_id="opaque-project-id",
        project_name="My clip",
        source=AnalysisSource(
            url="https://example.com/video",
            sha256="41cf6794ba4200b839c53531555f0f3998df4cbb01a4d5cb0b94e3ca5e23947d",
        ),
        provenance=AnalysisProvenance(transcription_version="1", segmenter_version="1"),
        candidates=(candidate,),
    )


def _initialized_project(tmp_path: Path) -> tuple[Path, UUID, UUID]:
    project_dir = tmp_path / "project"
    store = open_project_store(project_dir)
    analysis = _analysis()
    with store.stage() as stage:
        stage.write_analysis(analysis)
        stage.write_text("source.mp4", "source")
        stage.write_text("transcript.json", "transcript")
        stage.write_text("candidates.json", "candidates")
    store = open_project_store(project_dir)
    project = store.initialize_project(probe=_Probe())
    variant = create_initial_variant(
        name="Variant 1",
        selection=VariantRange(start_ms=10_000, end_ms=30_000, source_duration_ms=60_000),
        origin=VariantOrigin(
            source_candidate_id=analysis.candidates[0].candidate_id,
            analysis_sha256=project.analysis_sha256,
            transcript_sha256=project.transcript_sha256,
            source_sha256=project.source_sha256,
        ),
    )
    store.write_project(project.model_copy(update={"variants": (variant,)}))
    return project_dir, variant.variant_id, variant.layout_timeline[0].position_id


def test_preview_generates_fingerprinted_exact_frame_jpeg(tmp_path: Path) -> None:
    project_dir, variant_id, position_id = _initialized_project(tmp_path)
    runner = _Runner()

    result = generate_preview(
        PreviewRequest(
            project_dir=project_dir,
            variant_id=variant_id,
            position_id=position_id,
            at_ms=15_000,
        ),
        probe=_Probe(),
        runner=runner,
    )

    assert result.preview_path.is_file()
    assert result.preview_path.read_bytes() == b"\xff\xd8preview\xff\xd9"
    assert result.preview_path.parent.name == str(variant_id)
    assert result.width == 720
    assert result.height == 1280
    assert result.frame_time_ms == 15_000
    assert runner.argv is not None
    composed = runner.invocations[0]
    assert "trim=start=15" in composed[composed.index("-filter_complex") + 1]
    assert "scale=720:1280" in composed[composed.index("-filter_complex") + 1]
    assert "-q:v" in composed
    assert composed[composed.index("-q:v") + 1] == "3"
    source = runner.invocations[1]
    assert "min(1\\,min(1280/iw\\,1280/ih))" in source[source.index("-vf") + 1]
    assert source[source.index("-q:v") + 1] == "3"


def test_preview_rejects_playhead_outside_position(tmp_path: Path) -> None:
    project_dir, variant_id, position_id = _initialized_project(tmp_path)

    with pytest.raises(PreviewError, match="inside the selected position"):
        _ = generate_preview(
            PreviewRequest(
                project_dir=project_dir,
                variant_id=variant_id,
                position_id=position_id,
                at_ms=30_000,
            ),
            probe=_Probe(),
            runner=_Runner(),
        )


def test_preview_rejects_variant_requiring_review(tmp_path: Path) -> None:
    project_dir, variant_id, position_id = _initialized_project(tmp_path)
    store = open_project_store(project_dir)
    project = store.read_project()
    assert project is not None
    reviewed = project.model_copy(
        update={
            "variants": (
                project.variants[0].model_copy(update={"status": ProjectStatus.REQUIRES_REVIEW}),
            )
        }
    )
    store.write_project(reviewed)

    with pytest.raises(PreviewError, match="active variant"):
        _ = generate_preview(
            PreviewRequest(
                project_dir=project_dir,
                variant_id=variant_id,
                position_id=position_id,
                at_ms=15_000,
            ),
            probe=_Probe(),
            runner=_Runner(),
        )


def test_preview_cache_avoids_second_runner_invocation(tmp_path: Path) -> None:
    project_dir, variant_id, position_id = _initialized_project(tmp_path)
    first_runner = _Runner()
    first = generate_preview(
        PreviewRequest(
            project_dir=project_dir,
            variant_id=variant_id,
            position_id=position_id,
            at_ms=15_000,
        ),
        probe=_Probe(),
        runner=first_runner,
    )
    second_runner = _Runner()

    second = generate_preview(
        PreviewRequest(
            project_dir=project_dir,
            variant_id=variant_id,
            position_id=position_id,
            at_ms=15_000,
        ),
        probe=_Probe(),
        runner=second_runner,
    )

    assert second.preview_path == first.preview_path
    assert second.preview_fingerprint == first.preview_fingerprint
    assert second_runner.argv is None


def test_preview_regenerates_corrupt_fingerprint_cache(tmp_path: Path) -> None:
    project_dir, variant_id, position_id = _initialized_project(tmp_path)
    first = generate_preview(
        PreviewRequest(
            project_dir=project_dir, variant_id=variant_id, position_id=position_id, at_ms=15_000
        ),
        probe=_Probe(),
        runner=_Runner(),
    )
    _ = first.preview_path.write_bytes(b"")
    runner = _Runner()

    refreshed = generate_preview(
        PreviewRequest(
            project_dir=project_dir, variant_id=variant_id, position_id=position_id, at_ms=15_000
        ),
        probe=_Probe(),
        runner=runner,
    )

    assert runner.argv is not None
    assert refreshed.preview_path.read_bytes() == b"\xff\xd8preview\xff\xd9"


def test_preview_regenerates_marker_only_corrupt_jpeg_cache(tmp_path: Path) -> None:
    project_dir, variant_id, position_id = _initialized_project(tmp_path)
    first = generate_preview(
        PreviewRequest(
            project_dir=project_dir, variant_id=variant_id, position_id=position_id, at_ms=15_000
        ),
        probe=_Probe(),
        runner=_Runner(),
    )
    _ = first.preview_path.write_bytes(b"\xff\xd8not-a-real-image\xff\xd9")
    runner = _Runner()

    refreshed = generate_preview(
        PreviewRequest(
            project_dir=project_dir, variant_id=variant_id, position_id=position_id, at_ms=15_000
        ),
        probe=_Probe(),
        runner=runner,
    )

    assert runner.argv is not None
    assert refreshed.preview_path.read_bytes() == b"\xff\xd8preview\xff\xd9"


def test_preview_regenerates_wrong_image_format_cache(tmp_path: Path) -> None:
    project_dir, variant_id, position_id = _initialized_project(tmp_path)
    first = generate_preview(
        PreviewRequest(
            project_dir=project_dir, variant_id=variant_id, position_id=position_id, at_ms=15_000
        ),
        probe=_Probe(),
        runner=_Runner(),
    )
    _ = first.preview_path.write_bytes(b"png-cache")
    runner = _Runner()

    refreshed = generate_preview(
        PreviewRequest(
            project_dir=project_dir, variant_id=variant_id, position_id=position_id, at_ms=15_000
        ),
        probe=_Probe(),
        runner=runner,
    )

    assert runner.argv is not None
    assert refreshed.preview_path.read_bytes() == b"\xff\xd8preview\xff\xd9"


def test_preview_rejects_directory_at_fingerprint_cache_path(tmp_path: Path) -> None:
    project_dir, variant_id, position_id = _initialized_project(tmp_path)
    first = generate_preview(
        PreviewRequest(
            project_dir=project_dir,
            variant_id=variant_id,
            position_id=position_id,
            at_ms=15_000,
        ),
        probe=_Probe(),
        runner=_Runner(),
    )
    first.preview_path.unlink()
    first.preview_path.mkdir()

    with pytest.raises(PreviewError, match="cache entry"):
        _ = generate_preview(
            PreviewRequest(
                project_dir=project_dir,
                variant_id=variant_id,
                position_id=position_id,
                at_ms=15_000,
            ),
            probe=_Probe(),
            runner=_Runner(),
        )


def test_preview_rejects_directory_at_source_frame_cache_path(tmp_path: Path) -> None:
    project_dir, variant_id, position_id = _initialized_project(tmp_path)
    first = generate_preview(
        PreviewRequest(
            project_dir=project_dir,
            variant_id=variant_id,
            position_id=position_id,
            at_ms=15_000,
        ),
        probe=_Probe(),
        runner=_Runner(),
    )
    first.source_frame_path.unlink()
    first.source_frame_path.mkdir()
    previous_composed = first.preview_path.read_bytes()

    with pytest.raises(PreviewError, match="cache entry"):
        _ = generate_preview(
            PreviewRequest(
                project_dir=project_dir,
                variant_id=variant_id,
                position_id=position_id,
                at_ms=15_000,
            ),
            probe=_Probe(),
            runner=_Runner(),
        )

    assert first.preview_path.read_bytes() == previous_composed
    assert first.source_frame_path.is_dir()


def test_source_frame_failure_leaves_no_partial_preview_pair(tmp_path: Path) -> None:
    project_dir, variant_id, position_id = _initialized_project(tmp_path)
    first = generate_preview(
        PreviewRequest(
            project_dir=project_dir,
            variant_id=variant_id,
            position_id=position_id,
            at_ms=15_000,
        ),
        probe=_Probe(),
        runner=_Runner(),
    )
    first.source_frame_path.unlink()

    with pytest.raises(PreviewError, match="source frame renderer failed"):
        _ = generate_preview(
            PreviewRequest(
                project_dir=project_dir,
                variant_id=variant_id,
                position_id=position_id,
                at_ms=15_000,
            ),
            probe=_Probe(),
            runner=_FailsSecondRunner(),
        )

    assert not first.preview_path.exists()
    assert not first.source_frame_path.exists()
    assert not list(first.preview_path.parent.glob(".*.tmp.jpg"))


def test_preview_temp_reservation_failure_cleans_reserved_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project_dir, variant_id, position_id = _initialized_project(tmp_path)
    calls = 0
    real_reserve = cast("Callable[[Path], Path]", _private("_temporary_preview_path"))

    def fail_second_reservation(destination: Path) -> Path:
        nonlocal calls
        calls += 1
        if calls == 2:
            message = "disk full"
            raise OSError(message)
        return real_reserve(destination)

    monkeypatch.setattr("kliptych.studio.preview._temporary_preview_path", fail_second_reservation)

    with pytest.raises(PreviewError, match="temporary outputs"):
        _ = generate_preview(
            PreviewRequest(
                project_dir=project_dir,
                variant_id=variant_id,
                position_id=position_id,
                at_ms=15_000,
            ),
            probe=_Probe(),
            runner=_Runner(),
        )

    preview_dir = project_dir / "previews" / str(variant_id)
    assert not list(preview_dir.glob(".*.tmp*"))


@pytest.mark.skipif(
    os.name == "nt", reason="Windows retains a descriptor when close is injected to fail"
)
def test_preview_temp_close_failure_cleans_reserved_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project_dir, variant_id, position_id = _initialized_project(tmp_path)
    calls = 0
    real_close = os.close

    def fail_first_close(fd: int) -> None:
        nonlocal calls
        calls += 1
        if calls == 1:
            message = "close failed"
            raise OSError(message)
        real_close(fd)

    monkeypatch.setattr("kliptych.studio.preview.os.close", fail_first_close)

    with pytest.raises(PreviewError, match="temporary outputs"):
        _ = generate_preview(
            PreviewRequest(
                project_dir=project_dir,
                variant_id=variant_id,
                position_id=position_id,
                at_ms=15_000,
            ),
            probe=_Probe(),
            runner=_Runner(),
        )

    preview_dir = project_dir / "previews" / str(variant_id)
    assert not list(preview_dir.glob(".*.tmp*"))


def test_preview_source_hash_write_failure_is_structured_and_cleans_temps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project_dir, variant_id, position_id = _initialized_project(tmp_path)

    def fail_write(*_args: object, **_kwargs: object) -> None:
        message = "disk full"
        raise OSError(message)

    monkeypatch.setattr("kliptych.studio.preview._write_text", fail_write)

    with pytest.raises(PreviewError, match="source hash"):
        _ = generate_preview(
            PreviewRequest(
                project_dir=project_dir,
                variant_id=variant_id,
                position_id=position_id,
                at_ms=15_000,
            ),
            probe=_Probe(),
            runner=_Runner(),
        )

    preview_dir = project_dir / "previews" / str(variant_id)
    assert not list(preview_dir.glob(".*.tmp*"))
    assert not list(preview_dir.glob("*.jpg"))


def test_preview_rejects_source_hash_mismatch(tmp_path: Path) -> None:
    project_dir, variant_id, position_id = _initialized_project(tmp_path)
    _ = (project_dir / "source.mp4").write_text("changed", encoding="utf-8")

    with pytest.raises(PreviewError, match="source hash"):
        _ = generate_preview(
            PreviewRequest(
                project_dir=project_dir,
                variant_id=variant_id,
                position_id=position_id,
                at_ms=15_000,
            ),
            probe=_Probe(),
            runner=_Runner(),
        )


def test_preview_translates_unsupported_source_dimensions_to_preview_error(tmp_path: Path) -> None:
    project_dir, variant_id, position_id = _initialized_project(tmp_path)

    with pytest.raises(PreviewError, match="at least two"):
        _ = generate_preview(
            PreviewRequest(
                project_dir=project_dir,
                variant_id=variant_id,
                position_id=position_id,
                at_ms=15_000,
            ),
            probe=_NarrowProbe(),
            runner=_Runner(),
        )


def test_preview_rejects_changed_transcript_evidence(tmp_path: Path) -> None:
    project_dir, variant_id, position_id = _initialized_project(tmp_path)
    _ = (project_dir / "transcript.json").write_text("changed", encoding="utf-8")

    with pytest.raises(PreviewError, match="transcript evidence"):
        _ = generate_preview(
            PreviewRequest(
                project_dir=project_dir,
                variant_id=variant_id,
                position_id=position_id,
                at_ms=15_000,
            ),
            probe=_Probe(),
            runner=_Runner(),
        )


def test_preview_rejects_link_like_preview_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project_dir, variant_id, position_id = _initialized_project(tmp_path)
    root = project_dir / "previews"

    def fake_link_like(path: Path) -> bool:
        return path == root

    monkeypatch.setattr("kliptych.studio.preview.Path.is_symlink", fake_link_like)

    with pytest.raises(PreviewError, match="Preview root"):
        _ = generate_preview(
            PreviewRequest(
                project_dir=project_dir,
                variant_id=variant_id,
                position_id=position_id,
                at_ms=15_000,
            ),
            probe=_Probe(),
            runner=_Runner(),
        )


@pytest.mark.skipif(not Path("/usr/bin/ffmpeg").is_file(), reason="ffmpeg no instalado")
def test_preview_subprocess_generates_real_720_by_1280_jpeg(tmp_path: Path) -> None:
    project_dir = tmp_path / "project"
    source = tmp_path / "fixture.mp4"
    completed = SubprocessRunner.run(
        (
            "/usr/bin/ffmpeg",
            "-hide_banner",
            "-nostdin",
            "-v",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=1280x720:rate=30",
            "-t",
            "2",
            str(source),
        ),
        timeout_s=60.0,
    )
    assert completed.ok, completed.stderr
    source_hash = sha256_file(source)
    candidate = AnalysisCandidate(
        candidate_id="a" * 64,
        recommended_start_ms=0,
        recommended_end_ms=1_000,
        score=0.9,
        excerpt="Candidate",
    )
    analysis = AnalysisDocument(
        schema_version="1",
        project_id="opaque-project-id",
        project_name="My clip",
        source=AnalysisSource(url="https://example.com/video", sha256=source_hash),
        provenance=AnalysisProvenance(transcription_version="1", segmenter_version="1"),
        candidates=(candidate,),
    )
    store = open_project_store(project_dir)
    with store.stage() as stage:
        stage.write_analysis(analysis)
        stage.write_bytes("source.mp4", source.read_bytes())
        stage.write_text("transcript.json", "transcript")
        stage.write_text("candidates.json", "candidates")
    store = open_project_store(project_dir)
    project = store.initialize_project(probe=FFprobeProbe())
    variant = create_initial_variant(
        name="Variant 1",
        selection=VariantRange(
            start_ms=0, end_ms=1_000, source_duration_ms=project.source_duration_ms
        ),
        origin=VariantOrigin(
            source_candidate_id=candidate.candidate_id,
            analysis_sha256=project.analysis_sha256,
            transcript_sha256=project.transcript_sha256,
            source_sha256=project.source_sha256,
        ),
    )
    store.write_project(project.model_copy(update={"variants": (variant,)}))

    result = generate_preview(
        PreviewRequest(
            project_dir=project_dir,
            variant_id=variant.variant_id,
            position_id=variant.layout_timeline[0].position_id,
            at_ms=500,
        ),
        probe=FFprobeProbe(),
        runner=SubprocessRunner(),
    )
    dimensions = FFprobeProbe().probe(result.preview_path)
    source_dimensions = FFprobeProbe().probe(result.source_frame_path)

    assert dimensions.width == 720
    assert dimensions.height == 1280
    assert source_dimensions.width == 1280
    assert source_dimensions.height == 720


@pytest.mark.skipif(not Path("/usr/bin/ffmpeg").is_file(), reason="ffmpeg no instalado")
def test_preview_source_frame_does_not_upscale_small_source(tmp_path: Path) -> None:
    project_dir = tmp_path / "project"
    source = tmp_path / "small.mp4"
    completed = SubprocessRunner.run(
        (
            "/usr/bin/ffmpeg",
            "-hide_banner",
            "-nostdin",
            "-v",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=100x100:rate=30",
            "-t",
            "2",
            str(source),
        ),
        timeout_s=60.0,
    )
    assert completed.ok, completed.stderr
    source_hash = sha256_file(source)
    candidate = AnalysisCandidate(
        candidate_id="a" * 64,
        recommended_start_ms=0,
        recommended_end_ms=1_000,
        score=0.9,
        excerpt="Candidate",
    )
    analysis = AnalysisDocument(
        schema_version="1",
        project_id="opaque-project-id",
        project_name="My clip",
        source=AnalysisSource(url="https://example.com/video", sha256=source_hash),
        provenance=AnalysisProvenance(transcription_version="1", segmenter_version="1"),
        candidates=(candidate,),
    )
    store = open_project_store(project_dir)
    with store.stage() as stage:
        stage.write_analysis(analysis)
        stage.write_bytes("source.mp4", source.read_bytes())
        stage.write_text("transcript.json", "transcript")
        stage.write_text("candidates.json", "candidates")
    store = open_project_store(project_dir)
    project = store.initialize_project(probe=FFprobeProbe())
    variant = create_initial_variant(
        name="Variant 1",
        selection=VariantRange(
            start_ms=0, end_ms=1_000, source_duration_ms=project.source_duration_ms
        ),
        origin=VariantOrigin(
            source_candidate_id=candidate.candidate_id,
            analysis_sha256=project.analysis_sha256,
            transcript_sha256=project.transcript_sha256,
            source_sha256=project.source_sha256,
        ),
    )
    store.write_project(project.model_copy(update={"variants": (variant,)}))

    result = generate_preview(
        PreviewRequest(
            project_dir=project_dir,
            variant_id=variant.variant_id,
            position_id=variant.layout_timeline[0].position_id,
            at_ms=500,
        ),
        probe=FFprobeProbe(),
        runner=SubprocessRunner(),
    )
    dimensions = FFprobeProbe().probe(result.source_frame_path)

    assert dimensions.width == 100
    assert dimensions.height == 100
