"""Tests for the temporal motion preview service."""

from pathlib import Path
from uuid import uuid4

import pytest

from kliptych.contract.enums import Layout
from kliptych.contract.schema import Band, LayoutSpec, NormalizedRect
from kliptych.gate.models import MediaInfo
from kliptych.hashing import sha256_bytes
from kliptych.studio import motion_preview as mp
from kliptych.studio.models import (
    AnalysisCandidate,
    AnalysisDocument,
    AnalysisProvenance,
    AnalysisSource,
)
from kliptych.studio.preview import PreviewError
from kliptych.studio.project_models import LayoutPosition, ProjectStatus, Variant
from kliptych.studio.project_store import ProjectStore, open_project_store

_SOURCE_SHA = sha256_bytes(b"source")


def _analysis() -> AnalysisDocument:
    return AnalysisDocument(
        schema_version="1",
        project_id="a" * 32,
        project_name="clip",
        source=AnalysisSource(url="https://example.com/v", sha256=_SOURCE_SHA),
        provenance=AnalysisProvenance(transcription_version="1", segmenter_version="1"),
        candidates=(),
    )


def _candidate() -> AnalysisCandidate:
    return AnalysisCandidate(
        candidate_id="a" * 64,
        recommended_start_ms=10_000,
        recommended_end_ms=30_000,
        score=0.9,
        excerpt="c",
    )


def _setup(tmp_path: Path) -> tuple[Path, ProjectStore, Variant, LayoutPosition]:
    candidate = _candidate()
    analysis = _analysis().model_copy(update={"candidates": (candidate,)})
    project_dir = tmp_path / "project"
    store = open_project_store(project_dir)
    with store.stage() as stage:
        stage.write_analysis(analysis)
        stage.write_text("source.mp4", "source")
        stage.write_text(
            "transcript.json",
            '{"words":[{"start_s":15.0,"end_s":16.0,"text":"hello","confidence":0.9,"token_id":1}],"language":"en","duration_s":60.0,"text":"hello"}',
        )
        stage.write_text("candidates.json", "candidates")
    store = open_project_store(project_dir)
    _ = store.initialize_project(probe=_Probe(60.0))
    created = store.create_variant(candidate_id=candidate.candidate_id, name="V1")
    variant = created.variants[0]
    position = variant.layout_timeline[0]
    return project_dir, store, variant, position


class _Probe:
    def __init__(self, duration_s: float) -> None:
        self._duration: float = duration_s

    def probe(self, path: object) -> MediaInfo:
        _ = path
        return MediaInfo(
            format_name="mp4",
            duration_s=self._duration,
            has_video=True,
            has_audio=False,
            width=720,
            height=1280,
        )


class _Runner:
    def __init__(self) -> None:
        self.ran: bool = False

    def run(self, argv: tuple[str, ...], *, timeout_s: float) -> object:
        _ = timeout_s
        self.ran = True
        dest = Path(argv[-1])
        _ = dest.write_bytes(b"fake-mp4")
        return None


def test_motion_preview_happy_path(tmp_path: Path) -> None:
    project_dir, _store, variant, position = _setup(tmp_path)
    probe = _Probe(60.0)
    runner = _Runner()
    result = mp.generate_motion_preview(
        mp.MotionPreviewRequest(
            project_dir=project_dir,
            variant_id=variant.variant_id,
            position_id=position.position_id,
            at_ms=position.start_ms + 1000,
            duration_ms=2000,
        ),
        probe=probe,
        runner=runner,
    )
    assert result.preview_path.is_file()
    assert runner.ran is True


def test_motion_preview_outside_position_raises(tmp_path: Path) -> None:
    project_dir, _store, variant, position = _setup(tmp_path)
    probe = _Probe(60.0)
    runner = _Runner()
    with pytest.raises(PreviewError):
        _ = mp.generate_motion_preview(
            mp.MotionPreviewRequest(
                project_dir=project_dir,
                variant_id=variant.variant_id,
                position_id=position.position_id,
                at_ms=position.end_ms + 1000,
                duration_ms=1000,
            ),
            probe=probe,
            runner=runner,
        )


def test_motion_preview_requires_project(tmp_path: Path) -> None:
    project_dir = tmp_path / "missing"
    probe = _Probe(60.0)
    runner = _Runner()
    with pytest.raises(PreviewError):
        _ = mp.generate_motion_preview(
            mp.MotionPreviewRequest(
                project_dir=project_dir,
                variant_id=uuid4(),
                position_id=uuid4(),
                at_ms=0,
            ),
            probe=probe,
            runner=runner,
        )


def test_motion_preview_caches(tmp_path: Path) -> None:
    project_dir, _store, variant, position = _setup(tmp_path)
    probe = _Probe(60.0)
    runner = _Runner()
    r1 = mp.generate_motion_preview(
        mp.MotionPreviewRequest(
            project_dir=project_dir,
            variant_id=variant.variant_id,
            position_id=position.position_id,
            at_ms=position.start_ms + 1000,
            duration_ms=2000,
        ),
        probe=probe,
        runner=runner,
    )
    runner2 = _Runner()
    r2 = mp.generate_motion_preview(
        mp.MotionPreviewRequest(
            project_dir=project_dir,
            variant_id=variant.variant_id,
            position_id=position.position_id,
            at_ms=position.start_ms + 1000,
            duration_ms=2000,
        ),
        probe=probe,
        runner=runner2,
    )
    assert r1.fingerprint == r2.fingerprint
    assert runner2.ran is False


class _FailingRunner:
    def run(self, argv: tuple[str, ...], *, timeout_s: float) -> object:  # ruff: ignore[no-self-use]
        _ = (argv, timeout_s)
        msg = "boom"
        raise RuntimeError(msg)


def test_motion_preview_runner_failure_raises(tmp_path: Path) -> None:
    project_dir, _store, variant, position = _setup(tmp_path)
    probe = _Probe(60.0)
    with pytest.raises(PreviewError):
        _ = mp.generate_motion_preview(
            mp.MotionPreviewRequest(
                project_dir=project_dir,
                variant_id=variant.variant_id,
                position_id=position.position_id,
                at_ms=position.start_ms + 1000,
                duration_ms=2000,
            ),
            probe=probe,
            runner=_FailingRunner(),
        )


def test_motion_preview_with_captions(tmp_path: Path) -> None:
    project_dir, store, variant, position = _setup(tmp_path)
    _ = store.set_captions(
        variant_id=variant.variant_id,
        enabled=True,
        highlight_color="&H00FF0000",
        capitalization=None,
    )
    refreshed = open_project_store(project_dir).read_project()
    assert refreshed is not None
    variant = refreshed.variants[0]
    position = variant.layout_timeline[0]
    probe = _Probe(60.0)
    runner = _Runner()
    result = mp.generate_motion_preview(
        mp.MotionPreviewRequest(
            project_dir=project_dir,
            variant_id=variant.variant_id,
            position_id=position.position_id,
            at_ms=14000,
            duration_ms=3000,
        ),
        probe=probe,
        runner=runner,
    )
    assert result.preview_path.is_file()
    assert runner.ran is True


def test_motion_preview_with_watermark(tmp_path: Path) -> None:
    project_dir, store, variant, position = _setup(tmp_path)
    png = tmp_path / "logo.png"
    _ = png.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 8)
    _ = store.import_watermark(variant_id=variant.variant_id, source_path=png, position=None)
    refreshed = open_project_store(project_dir).read_project()
    assert refreshed is not None
    variant = refreshed.variants[0]
    position = variant.layout_timeline[0]
    probe = _Probe(60.0)
    runner = _Runner()
    result = mp.generate_motion_preview(
        mp.MotionPreviewRequest(
            project_dir=project_dir,
            variant_id=variant.variant_id,
            position_id=position.position_id,
            at_ms=position.start_ms + 1000,
            duration_ms=2000,
        ),
        probe=probe,
        runner=runner,
    )
    assert result.preview_path.is_file()
    assert runner.ran is True


def test_motion_preview_with_caption_correction(tmp_path: Path) -> None:
    project_dir, store, variant, position = _setup(tmp_path)
    _ = store.set_captions(
        variant_id=variant.variant_id,
        enabled=True,
        highlight_color="&H00FF0000",
        capitalization=None,
    )
    _ = store.set_caption_correction(
        variant_id=variant.variant_id,
        interval=(14500, 15500),
        corrected_text="Corregido",
        author="user",
        created_at_ms=1,
    )
    refreshed = open_project_store(project_dir).read_project()
    assert refreshed is not None
    variant = refreshed.variants[0]
    position = variant.layout_timeline[0]
    probe = _Probe(60.0)
    runner = _Runner()
    result = mp.generate_motion_preview(
        mp.MotionPreviewRequest(
            project_dir=project_dir,
            variant_id=variant.variant_id,
            position_id=position.position_id,
            at_ms=position.start_ms + 1000,
            duration_ms=2000,
        ),
        probe=probe,
        runner=runner,
    )
    assert result.preview_path.is_file()


class _FailingProbe:
    def probe(self, path: object) -> MediaInfo:  # ruff: ignore[no-self-use]
        _ = path
        msg = "probe boom"
        raise RuntimeError(msg)


def test_motion_preview_probe_failure_raises(tmp_path: Path) -> None:
    project_dir, _store, variant, position = _setup(tmp_path)
    with pytest.raises(PreviewError):
        _ = mp.generate_motion_preview(
            mp.MotionPreviewRequest(
                project_dir=project_dir,
                variant_id=variant.variant_id,
                position_id=position.position_id,
                at_ms=position.start_ms + 1000,
                duration_ms=2000,
            ),
            probe=_FailingProbe(),
            runner=_Runner(),
        )


def test_motion_preview_variant_requires_review_raises(tmp_path: Path) -> None:
    project_dir, _store, variant, position = _setup(tmp_path)
    # mark variant requires_review by changing source hash

    project = open_project_store(project_dir).read_project()
    assert project is not None
    v = project.variants[0].model_copy(update={"status": ProjectStatus.REQUIRES_REVIEW})
    project = project.model_copy(update={"variants": (v,)})
    # write directly
    _ = (project_dir / "project.json").write_text(project.model_dump_json(indent=2))
    probe = _Probe(60.0)
    runner = _Runner()
    with pytest.raises(PreviewError):
        _ = mp.generate_motion_preview(
            mp.MotionPreviewRequest(
                project_dir=project_dir,
                variant_id=variant.variant_id,
                position_id=position.position_id,
                at_ms=position.start_ms + 1000,
                duration_ms=2000,
            ),
            probe=probe,
            runner=runner,
        )


def test_motion_preview_source_hash_mismatch_raises(tmp_path: Path) -> None:
    project_dir, _store, variant, position = _setup(tmp_path)
    # corrupt source
    _ = (project_dir / "source.mp4").write_text("tampered")
    probe = _Probe(60.0)
    runner = _Runner()
    with pytest.raises(PreviewError):
        _ = mp.generate_motion_preview(
            mp.MotionPreviewRequest(
                project_dir=project_dir,
                variant_id=variant.variant_id,
                position_id=position.position_id,
                at_ms=position.start_ms + 1000,
                duration_ms=2000,
            ),
            probe=probe,
            runner=runner,
        )


def test_motion_preview_transcript_hash_mismatch_raises(tmp_path: Path) -> None:
    project_dir, _store, variant, position = _setup(tmp_path)
    _ = (project_dir / "transcript.json").write_text(
        '{"words":[],"language":"en","duration_s":60.0,"text":"x"}'
    )
    probe = _Probe(60.0)
    runner = _Runner()
    with pytest.raises(PreviewError):
        _ = mp.generate_motion_preview(
            mp.MotionPreviewRequest(
                project_dir=project_dir,
                variant_id=variant.variant_id,
                position_id=position.position_id,
                at_ms=position.start_ms + 1000,
                duration_ms=2000,
            ),
            probe=probe,
            runner=runner,
        )


def test_motion_preview_symlink_cache_rejected(tmp_path: Path) -> None:
    project_dir, _store, variant, position = _setup(tmp_path)
    probe = _Probe(60.0)
    runner = _Runner()
    result = mp.generate_motion_preview(
        mp.MotionPreviewRequest(
            project_dir=project_dir,
            variant_id=variant.variant_id,
            position_id=position.position_id,
            at_ms=position.start_ms + 1000,
            duration_ms=2000,
        ),
        probe=probe,
        runner=runner,
    )
    result.preview_path.unlink()
    link = tmp_path / "evil.mp4"
    _ = link.write_bytes(b"x")
    result.preview_path.symlink_to(link)
    runner2 = _Runner()
    _ = mp.generate_motion_preview(
        mp.MotionPreviewRequest(
            project_dir=project_dir,
            variant_id=variant.variant_id,
            position_id=position.position_id,
            at_ms=position.start_ms + 1000,
            duration_ms=2000,
        ),
        probe=probe,
        runner=runner2,
    )
    assert runner2.ran is True


def test_motion_preview_analysis_hash_mismatch_raises(tmp_path: Path) -> None:
    project_dir, _store, variant, position = _setup(tmp_path)
    # tamper analysis.json
    _ = (project_dir / "analysis.json").write_text(
        (project_dir / "analysis.json").read_text().replace("a" * 32, "b" * 32)
    )
    probe = _Probe(60.0)
    runner = _Runner()
    with pytest.raises(PreviewError):
        _ = mp.generate_motion_preview(
            mp.MotionPreviewRequest(
                project_dir=project_dir,
                variant_id=variant.variant_id,
                position_id=position.position_id,
                at_ms=position.start_ms + 1000,
                duration_ms=2000,
            ),
            probe=probe,
            runner=runner,
        )


def test_motion_preview_variant_not_found_raises(tmp_path: Path) -> None:
    project_dir, _store, _variant, position = _setup(tmp_path)
    probe = _Probe(60.0)
    runner = _Runner()
    with pytest.raises(PreviewError):
        _ = mp.generate_motion_preview(
            mp.MotionPreviewRequest(
                project_dir=project_dir,
                variant_id=uuid4(),
                position_id=position.position_id,
                at_ms=position.start_ms + 1000,
                duration_ms=2000,
            ),
            probe=probe,
            runner=runner,
        )


def test_motion_preview_position_not_found_raises(tmp_path: Path) -> None:
    project_dir, _store, variant, position = _setup(tmp_path)
    probe = _Probe(60.0)
    runner = _Runner()
    with pytest.raises(PreviewError):
        _ = mp.generate_motion_preview(
            mp.MotionPreviewRequest(
                project_dir=project_dir,
                variant_id=variant.variant_id,
                position_id=uuid4(),
                at_ms=position.start_ms + 1000,
                duration_ms=2000,
            ),
            probe=probe,
            runner=runner,
        )


def test_motion_preview_watermark_sha_mismatch_skips_watermark(tmp_path: Path) -> None:
    project_dir, store, variant, position = _setup(tmp_path)
    png = tmp_path / "logo.png"
    _ = png.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 8)
    _ = store.import_watermark(variant_id=variant.variant_id, source_path=png, position=None)
    # tamper the stored asset
    wm_dir = project_dir / "assets" / "watermark"
    for f in wm_dir.iterdir():
        _ = f.write_bytes(b"tampered")
    refreshed = open_project_store(project_dir).read_project()
    assert refreshed is not None
    variant = refreshed.variants[0]
    position = variant.layout_timeline[0]
    probe = _Probe(60.0)
    runner = _Runner()
    result = mp.generate_motion_preview(
        mp.MotionPreviewRequest(
            project_dir=project_dir,
            variant_id=variant.variant_id,
            position_id=position.position_id,
            at_ms=position.start_ms + 1000,
            duration_ms=2000,
        ),
        probe=probe,
        runner=runner,
    )
    assert result.preview_path.is_file()


def test_motion_preview_caption_correction_non_overlapping(tmp_path: Path) -> None:
    project_dir, store, variant, position = _setup(tmp_path)
    _ = store.set_captions(
        variant_id=variant.variant_id,
        enabled=True,
        highlight_color="&H00FF0000",
        capitalization=None,
    )
    _ = store.set_caption_correction(
        variant_id=variant.variant_id,
        interval=(position.start_ms + 5000, position.start_ms + 6000),
        corrected_text="Corregido",
        author="user",
        created_at_ms=1,
    )
    refreshed = open_project_store(project_dir).read_project()
    assert refreshed is not None
    variant = refreshed.variants[0]
    position = variant.layout_timeline[0]
    probe = _Probe(60.0)
    runner = _Runner()
    result = mp.generate_motion_preview(
        mp.MotionPreviewRequest(
            project_dir=project_dir,
            variant_id=variant.variant_id,
            position_id=position.position_id,
            at_ms=position.start_ms + 1000,
            duration_ms=2000,
        ),
        probe=probe,
        runner=runner,
    )
    assert result.preview_path.is_file()


def test_motion_preview_multi_band_layout(tmp_path: Path) -> None:
    candidate = _candidate()
    analysis = _analysis().model_copy(update={"candidates": (candidate,)})
    project_dir = tmp_path / "project"
    store = open_project_store(project_dir)
    with store.stage() as stage:
        stage.write_analysis(analysis)
        stage.write_text("source.mp4", "source")
        stage.write_text(
            "transcript.json",
            '{"words":[{"start_s":15.0,"end_s":16.0,"text":"hello","confidence":0.9,"token_id":1}],"language":"en","duration_s":60.0,"text":"hello"}',
        )
        stage.write_text("candidates.json", "candidates")
    store = open_project_store(project_dir)
    _ = store.initialize_project(probe=_Probe(60.0))
    created = store.create_variant(candidate_id=candidate.candidate_id, name="V1")
    variant = created.variants[0]
    position = variant.layout_timeline[0]
    # change layout to dividido (2 bands)

    spec = LayoutSpec(
        layout=Layout.DIVIDIDO,
        bands=(
            Band(source="source.mp4", source_rect=NormalizedRect()),
            Band(source="source.mp4", source_rect=NormalizedRect()),
        ),
    )
    _ = store.update_position(
        variant_id=variant.variant_id, position_id=position.position_id, layout_spec=spec
    )
    refreshed = open_project_store(project_dir).read_project()
    assert refreshed is not None
    variant = refreshed.variants[0]
    position = variant.layout_timeline[0]
    probe = _Probe(60.0)
    runner = _Runner()
    result = mp.generate_motion_preview(
        mp.MotionPreviewRequest(
            project_dir=project_dir,
            variant_id=variant.variant_id,
            position_id=position.position_id,
            at_ms=position.start_ms + 1000,
            duration_ms=2000,
        ),
        probe=probe,
        runner=runner,
    )
    assert result.preview_path.is_file()
    assert runner.ran is True


def test_motion_preview_probe_failure_raises2(tmp_path: Path) -> None:
    project_dir, _store, variant, position = _setup(tmp_path)

    class _FailingProbe:
        def probe(self, path: object) -> MediaInfo:  # ruff: ignore[no-self-use]
            _ = path
            msg = "probe boom"
            raise RuntimeError(msg)

    with pytest.raises(PreviewError):
        _ = mp.generate_motion_preview(
            mp.MotionPreviewRequest(
                project_dir=project_dir,
                variant_id=variant.variant_id,
                position_id=position.position_id,
                at_ms=position.start_ms + 1000,
                duration_ms=2000,
            ),
            probe=_FailingProbe(),
            runner=_Runner(),
        )


def test_motion_preview_symlink_cache_rejected2(tmp_path: Path) -> None:
    project_dir, _store, variant, position = _setup(tmp_path)
    probe = _Probe(60.0)
    runner = _Runner()
    result = mp.generate_motion_preview(
        mp.MotionPreviewRequest(
            project_dir=project_dir,
            variant_id=variant.variant_id,
            position_id=position.position_id,
            at_ms=position.start_ms + 1000,
            duration_ms=2000,
        ),
        probe=probe,
        runner=runner,
    )
    result.preview_path.unlink()
    link = tmp_path / "evil.mp4"
    _ = link.write_bytes(b"x")
    result.preview_path.symlink_to(link)
    runner2 = _Runner()
    _ = mp.generate_motion_preview(
        mp.MotionPreviewRequest(
            project_dir=project_dir,
            variant_id=variant.variant_id,
            position_id=position.position_id,
            at_ms=position.start_ms + 1000,
            duration_ms=2000,
        ),
        probe=probe,
        runner=runner2,
    )
    assert runner2.ran is True
