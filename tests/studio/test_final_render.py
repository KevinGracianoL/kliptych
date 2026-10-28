"""Tests for Track B3 studio final render and export service."""

from pathlib import Path
from uuid import uuid4

import pytest

from kliptych.contract.enums import WatermarkPosition
from kliptych.gate.models import MediaInfo
from kliptych.hashing import sha256_bytes
from kliptych.studio import final_render as fr
from kliptych.studio.models import (
    AnalysisCandidate,
    AnalysisDocument,
    AnalysisProvenance,
    AnalysisSource,
)
from kliptych.studio.project_models import (
    CaptionStyle,
    CaptionStylePreset,
    ProjectStatus,
    Variant,
)
from kliptych.studio.project_store import ProjectStore, open_project_store

_SOURCE_SHA = sha256_bytes(b"source")


def _analysis() -> AnalysisDocument:
    return AnalysisDocument(
        schema_version="1",
        project_id="b" * 32,
        project_name="clip_final",
        source=AnalysisSource(url="https://example.com/v", sha256=_SOURCE_SHA),
        provenance=AnalysisProvenance(transcription_version="1", segmenter_version="1"),
        candidates=(),
    )


def _candidate() -> AnalysisCandidate:
    return AnalysisCandidate(
        candidate_id="b" * 64,
        recommended_start_ms=10_000,
        recommended_end_ms=30_000,
        score=0.95,
        excerpt="c_final",
    )


def _setup_project(tmp_path: Path) -> tuple[Path, ProjectStore, Variant]:
    candidate = _candidate()
    analysis = _analysis().model_copy(update={"candidates": (candidate,)})
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
    _ = store.initialize_project(probe=_Probe(duration_s=60.0, width=1920, height=1080))
    created = store.create_variant(candidate_id=candidate.candidate_id, name="Variante 1")
    variant = created.variants[0]
    return project_dir, store, variant


class _Probe:
    def __init__(
        self,
        duration_s: float,
        *,
        width: int = 1080,
        height: int = 1920,
        has_video: bool = True,
        has_audio: bool = True,
    ) -> None:
        self._duration: float = duration_s
        self._width: int = width
        self._height: int = height
        self._has_video: bool = has_video
        self._has_audio: bool = has_audio

    def probe(self, path: object) -> MediaInfo:
        _ = path
        return MediaInfo(
            format_name="mp4",
            duration_s=self._duration,
            has_video=self._has_video,
            has_audio=self._has_audio,
            width=self._width,
            height=self._height,
        )


class _Runner:
    def __init__(self, *, fail_nvenc: bool = False, fail_all: bool = False) -> None:
        self.invocations: list[tuple[tuple[str, ...], Path | None]] = []
        self._fail_nvenc: bool = fail_nvenc
        self._fail_all: bool = fail_all

    def run(self, argv: tuple[str, ...], *, timeout_s: float, cwd: Path | None = None) -> object:
        _ = timeout_s
        self.invocations.append((argv, cwd))
        if self._fail_all:
            msg = "ffmpeg execution failed"
            raise RuntimeError(msg)
        if self._fail_nvenc and "h264_nvenc" in argv:
            msg = "Driver does not support NVENC"
            raise RuntimeError(msg)
        dest = Path(argv[-1])
        dest.parent.mkdir(parents=True, exist_ok=True)
        _ = dest.write_bytes(b"final-mp4-data")
        return None


def test_render_final_variant_happy_path(tmp_path: Path) -> None:
    project_dir, _store, variant = _setup_project(tmp_path)
    probe = _Probe(duration_s=20.0, width=1080, height=1920)
    runner = _Runner()

    result = fr.render_final_variant(
        fr.FinalRenderRequest(
            project_dir=project_dir,
            variant_id=variant.variant_id,
        ),
        probe=probe,
        runner=runner,
    )

    assert result.output_path.is_file()
    assert result.output_path.name == "Variante 1.mp4"
    assert result.variant_id == variant.variant_id
    assert result.gate_verdict == "pass"
    assert result.width == 1080
    assert result.height == 1920
    assert result.encoder_used == "libx264"
    assert len(runner.invocations) == 1


def test_render_final_custom_destination(tmp_path: Path) -> None:
    project_dir, _store, variant = _setup_project(tmp_path)
    probe = _Probe(duration_s=20.0, width=1080, height=1920)
    runner = _Runner()
    custom_out = tmp_path / "exports" / "custom_clip.mp4"

    result = fr.render_final_variant(
        fr.FinalRenderRequest(
            project_dir=project_dir,
            variant_id=variant.variant_id,
            destination=custom_out,
        ),
        probe=probe,
        runner=runner,
    )

    assert result.output_path == custom_out.resolve()
    assert custom_out.is_file()


def test_render_final_nvenc_success(tmp_path: Path) -> None:
    project_dir, _store, variant = _setup_project(tmp_path)
    probe = _Probe(duration_s=20.0, width=1080, height=1920)
    runner = _Runner()

    result = fr.render_final_variant(
        fr.FinalRenderRequest(
            project_dir=project_dir,
            variant_id=variant.variant_id,
        ),
        probe=probe,
        runner=runner,
        nvenc_available=True,
    )

    assert result.encoder_used == "h264_nvenc"
    assert len(runner.invocations) == 1
    assert "h264_nvenc" in runner.invocations[0][0]


def test_render_final_nvenc_fallback_to_libx264(tmp_path: Path) -> None:
    project_dir, _store, variant = _setup_project(tmp_path)
    probe = _Probe(duration_s=20.0, width=1080, height=1920)
    runner = _Runner(fail_nvenc=True)

    result = fr.render_final_variant(
        fr.FinalRenderRequest(
            project_dir=project_dir,
            variant_id=variant.variant_id,
        ),
        probe=probe,
        runner=runner,
        nvenc_available=True,
    )

    assert result.encoder_used == "libx264"
    assert len(runner.invocations) == 2
    assert "h264_nvenc" in runner.invocations[0][0]
    assert "libx264" in runner.invocations[1][0]


def test_render_final_multi_position_timeline(tmp_path: Path) -> None:
    project_dir, store, variant = _setup_project(tmp_path)
    # Split position at 20_000 ms
    pos_id = variant.layout_timeline[0].position_id
    updated = store.split_position(variant_id=variant.variant_id, position_id=pos_id, at_ms=20_000)
    assert len(updated.variants[0].layout_timeline) == 2

    probe = _Probe(duration_s=20.0, width=1080, height=1920)
    runner = _Runner()

    result = fr.render_final_variant(
        fr.FinalRenderRequest(
            project_dir=project_dir,
            variant_id=variant.variant_id,
        ),
        probe=probe,
        runner=runner,
    )

    assert result.output_path.is_file()
    argv = runner.invocations[0][0]
    fc_idx = argv.index("-filter_complex")
    filter_complex = argv[fc_idx + 1]
    assert "concat=n=2:v=1:a=0" in filter_complex


def test_render_final_with_captions_and_corrections(tmp_path: Path) -> None:
    project_dir, store, variant = _setup_project(tmp_path)
    style = CaptionStyle.default().model_copy(update={"preset": CaptionStylePreset.BOLD})
    _ = store.set_captions(
        variant_id=variant.variant_id,
        enabled=True,
        caption_style=style,
    )
    _ = store.set_caption_correction(
        variant_id=variant.variant_id,
        interval=(12_000, 14_000),
        corrected_text="corrected caption text",
        author="reviewer",
        created_at_ms=12345,
    )

    probe = _Probe(duration_s=20.0, width=1080, height=1920)
    runner = _Runner()

    result = fr.render_final_variant(
        fr.FinalRenderRequest(
            project_dir=project_dir,
            variant_id=variant.variant_id,
        ),
        probe=probe,
        runner=runner,
    )

    assert result.output_path.is_file()
    argv = runner.invocations[0][0]
    fc_idx = argv.index("-filter_complex")
    filter_complex = argv[fc_idx + 1]
    assert "subtitles=" in filter_complex
    assert "fontsdir=" in filter_complex


def test_render_final_with_watermark(tmp_path: Path) -> None:
    project_dir, store, variant = _setup_project(tmp_path)
    png_bytes = b"\x89PNG\r\n\x1a\nfake-watermark-png-bytes"
    wm_file = tmp_path / "watermark.png"
    _ = wm_file.write_bytes(png_bytes)

    _ = store.import_watermark(
        variant_id=variant.variant_id,
        source_path=wm_file,
        position=WatermarkPosition.TOP_LEFT,
    )

    probe = _Probe(duration_s=20.0, width=1080, height=1920)
    runner = _Runner()

    result = fr.render_final_variant(
        fr.FinalRenderRequest(
            project_dir=project_dir,
            variant_id=variant.variant_id,
        ),
        probe=probe,
        runner=runner,
    )

    assert result.output_path.is_file()
    argv = runner.invocations[0][0]
    assert "-i" in argv
    fc_idx = argv.index("-filter_complex")
    filter_complex = argv[fc_idx + 1]
    assert "overlay=" in filter_complex


def test_render_final_gate_rejection_fail_closed(tmp_path: Path) -> None:
    project_dir, _store, variant = _setup_project(tmp_path)
    # Aspect 16:9 instead of 9:16 should trigger gate rejection
    bad_probe = _Probe(duration_s=20.0, width=1920, height=1080)
    runner = _Runner()

    with pytest.raises(fr.FinalRenderError) as exc_info:
        _ = fr.render_final_variant(
            fr.FinalRenderRequest(
                project_dir=project_dir,
                variant_id=variant.variant_id,
            ),
            probe=bad_probe,
            runner=runner,
        )

    assert "Gate check failed" in exc_info.value.message
    assert exc_info.value.error_code == "gate_rejected"
    # Verify temporary file cleaned up
    delivery_dir = project_dir / "delivery"
    assert not any(delivery_dir.glob("*.tmp.mp4")) if delivery_dir.exists() else True


def test_render_final_evidence_mismatch_raises(tmp_path: Path) -> None:
    project_dir, _store, variant = _setup_project(tmp_path)
    # Corrupt source.mp4 evidence
    _ = (project_dir / "source.mp4").write_text("corrupted source")

    probe = _Probe(duration_s=20.0, width=1080, height=1920)
    runner = _Runner()

    with pytest.raises(fr.FinalRenderError) as exc_info:
        _ = fr.render_final_variant(
            fr.FinalRenderRequest(
                project_dir=project_dir,
                variant_id=variant.variant_id,
            ),
            probe=probe,
            runner=runner,
        )

    assert "Source evidence does not match" in exc_info.value.message
    assert exc_info.value.error_code == "evidence_mismatch"


def test_render_final_ffmpeg_failure_cleans_up(tmp_path: Path) -> None:
    project_dir, _store, variant = _setup_project(tmp_path)
    probe = _Probe(duration_s=20.0, width=1080, height=1920)
    runner = _Runner(fail_all=True)

    with pytest.raises(fr.FinalRenderError) as exc_info:
        _ = fr.render_final_variant(
            fr.FinalRenderRequest(
                project_dir=project_dir,
                variant_id=variant.variant_id,
            ),
            probe=probe,
            runner=runner,
        )

    assert exc_info.value.error_code == "render_failed"
    delivery_dir = project_dir / "delivery"
    assert not any(delivery_dir.glob("*.tmp.mp4")) if delivery_dir.exists() else True
    assert not any(delivery_dir.glob("*.ass")) if delivery_dir.exists() else True


def test_render_final_uninitialized_project_raises(tmp_path: Path) -> None:
    project_dir = tmp_path / "uninit_project"
    store = open_project_store(project_dir)
    with store.stage() as stage:
        stage.write_analysis(_analysis())
        stage.write_text("source.mp4", "source")
        stage.write_text("transcript.json", "{}")
        stage.write_text("candidates.json", "candidates")
    probe = _Probe(duration_s=20.0, width=1080, height=1920)
    runner = _Runner()

    with pytest.raises(fr.FinalRenderError) as exc_info:
        _ = fr.render_final_variant(
            fr.FinalRenderRequest(
                project_dir=project_dir,
                variant_id=uuid4(),
            ),
            probe=probe,
            runner=runner,
        )

    assert exc_info.value.error_code == "uninitialized_project"


def test_render_final_variant_not_found_raises(tmp_path: Path) -> None:
    project_dir, _store, _variant = _setup_project(tmp_path)
    probe = _Probe(duration_s=20.0, width=1080, height=1920)
    runner = _Runner()

    with pytest.raises(fr.FinalRenderError) as exc_info:
        _ = fr.render_final_variant(
            fr.FinalRenderRequest(
                project_dir=project_dir,
                variant_id=uuid4(),
            ),
            probe=probe,
            runner=runner,
        )

    assert exc_info.value.error_code == "variant_not_found"


def test_render_final_variant_not_active_raises(tmp_path: Path) -> None:
    project_dir, store, variant = _setup_project(tmp_path)
    project = store.read_project()
    assert project is not None
    # Mark variant as pending review
    updated_variant = variant.model_copy(update={"status": ProjectStatus.REQUIRES_REVIEW})
    updated_project = project.model_copy(update={"variants": (updated_variant,)})
    store.write_project(updated_project)

    probe = _Probe(duration_s=20.0, width=1080, height=1920)
    runner = _Runner()

    with pytest.raises(fr.FinalRenderError) as exc_info:
        _ = fr.render_final_variant(
            fr.FinalRenderRequest(
                project_dir=project_dir,
                variant_id=variant.variant_id,
            ),
            probe=probe,
            runner=runner,
        )

    assert exc_info.value.error_code == "variant_not_active"


def test_render_final_invalid_dimensions_raises(tmp_path: Path) -> None:
    project_dir, _store, variant = _setup_project(tmp_path)
    # Dimensions None
    probe = _Probe(duration_s=20.0, width=0, height=0)
    runner = _Runner()

    with pytest.raises(fr.FinalRenderError) as exc_info:
        _ = fr.render_final_variant(
            fr.FinalRenderRequest(
                project_dir=project_dir,
                variant_id=variant.variant_id,
            ),
            probe=probe,
            runner=runner,
        )

    assert exc_info.value.error_code == "invalid_dimensions"


def test_render_final_symlink_destination_raises(tmp_path: Path) -> None:
    project_dir, _store, variant = _setup_project(tmp_path)
    probe = _Probe(duration_s=20.0, width=1080, height=1920)
    runner = _Runner()

    target_file = tmp_path / "real_file.mp4"
    target_file.touch()
    symlink_dest = tmp_path / "symlink_clip.mp4"
    symlink_dest.symlink_to(target_file)

    with pytest.raises(fr.FinalRenderError) as exc_info:
        _ = fr.render_final_variant(
            fr.FinalRenderRequest(
                project_dir=project_dir,
                variant_id=variant.variant_id,
                destination=symlink_dest,
            ),
            probe=probe,
            runner=runner,
        )

    assert exc_info.value.error_code == "invalid_destination"


def test_render_final_symlink_delivery_dir_raises(tmp_path: Path) -> None:
    project_dir, _store, variant = _setup_project(tmp_path)
    probe = _Probe(duration_s=20.0, width=1080, height=1920)
    runner = _Runner()

    real_dir = tmp_path / "real_delivery"
    real_dir.mkdir()
    symlink_delivery = project_dir / "delivery"
    symlink_delivery.symlink_to(real_dir)

    with pytest.raises(fr.FinalRenderError) as exc_info:
        _ = fr.render_final_variant(
            fr.FinalRenderRequest(
                project_dir=project_dir,
                variant_id=variant.variant_id,
            ),
            probe=probe,
            runner=runner,
        )

    assert exc_info.value.error_code == "invalid_directory"


class _NoOutputRunner:
    @staticmethod
    def run(argv: tuple[str, ...], *, timeout_s: float, cwd: Path | None = None) -> object:
        _ = argv
        _ = timeout_s
        _ = cwd
        # Returns success but does not create the output file
        return None


def test_render_final_runner_produces_no_output_raises(tmp_path: Path) -> None:
    project_dir, _store, variant = _setup_project(tmp_path)
    probe = _Probe(duration_s=20.0, width=1080, height=1920)
    runner = _NoOutputRunner()

    with pytest.raises(fr.FinalRenderError) as exc_info:
        _ = fr.render_final_variant(
            fr.FinalRenderRequest(
                project_dir=project_dir,
                variant_id=variant.variant_id,
            ),
            probe=probe,
            runner=runner,
        )

    assert exc_info.value.error_code == "encoder_failed"


def test_render_final_cancellation_by_signal_cleans_up(tmp_path: Path) -> None:
    project_dir, _store, variant = _setup_project(tmp_path)
    probe = _Probe(duration_s=20.0, width=1080, height=1920)

    class _CancellingRunner:
        @staticmethod
        def run(argv: tuple[str, ...], *, timeout_s: float, cwd: Path | None = None) -> object:
            _ = argv
            _ = timeout_s
            _ = cwd
            # Simulate SIGTERM signal handler
            fr.signal_cleanup_handler(15, None)
            return None

    runner = _CancellingRunner()

    with pytest.raises(fr.FinalRenderError) as exc_info:
        _ = fr.render_final_variant(
            fr.FinalRenderRequest(
                project_dir=project_dir,
                variant_id=variant.variant_id,
            ),
            probe=probe,
            runner=runner,
        )

    assert exc_info.value.error_code == "render_cancelled"
    delivery_dir = project_dir / "delivery"
    assert not any(delivery_dir.glob("*.tmp.mp4")) if delivery_dir.exists() else True
