"""Behavioral tests for Studio output aspect (vertical/horizontal)."""

import json
from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import cast
from uuid import UUID, uuid4

import pytest
from pydantic import ValidationError

from kliptych.__main__ import main
from kliptych.contract.enums import Aspect
from kliptych.environment import CommandResult, SubprocessRunner
from kliptych.gate.models import MediaInfo
from kliptych.gate.probe import FFprobeProbe, ProbeError
from kliptych.hashing import sha256_bytes, sha256_canonical_json, sha256_file
from kliptych.studio import final_render as fr
from kliptych.studio.export_queue import ExportQueueDocument
from kliptych.studio.models import (
    AnalysisCandidate,
    AnalysisDocument,
    AnalysisProvenance,
    AnalysisSource,
)
from kliptych.studio.motion_preview import MotionPreviewRequest, generate_motion_preview
from kliptych.studio.preview import PreviewRequest, generate_preview
from kliptych.studio.project_models import (
    ProjectDocument,
    Variant,
    VariantOrigin,
    VariantRange,
    create_initial_variant,
    duplicate_variant,
    final_dimensions,
    preview_dimensions,
)
from kliptych.studio.project_store import ProjectStore, open_project_store

_HASH = "b" * 64
_CANDIDATE = "a" * 64


def _origin() -> VariantOrigin:
    return VariantOrigin(
        source_candidate_id=_CANDIDATE,
        analysis_sha256=_HASH,
        transcript_sha256=_HASH,
        source_sha256=_HASH,
    )


@pytest.mark.integration
@pytest.mark.parametrize("aspect", [Aspect.VERTICAL, Aspect.HORIZONTAL])
def test_real_media_preview_and_queued_export_match_aspect(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], aspect: Aspect
) -> None:
    source = tmp_path / "synthetic.mp4"
    generated = SubprocessRunner().run(
        (
            "ffmpeg",
            "-hide_banner",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=640x360:rate=8:duration=4",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:sample_rate=48000:duration=4",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-c:a",
            "aac",
            "-shortest",
            str(source),
        ),
        timeout_s=60.0,
    )
    assert generated.ok
    project_dir = tmp_path / "project"
    candidate = _candidate().model_copy(
        update={
            "recommended_start_ms": 0,
            "recommended_end_ms": 4000,
        }
    )
    analysis = _analysis_doc(candidates=(candidate,))
    analysis = analysis.model_copy(
        update={
            "source": analysis.source.model_copy(update={"sha256": sha256_file(source)}),
        }
    )
    store = open_project_store(project_dir)
    with store.stage() as stage:
        stage.write_analysis(analysis)
        stage.write_bytes("source.mp4", source.read_bytes())
        stage.write_text("candidates.json", "[]")
        stage.write_text(
            "transcript.json",
            json.dumps(
                {
                    "words": [
                        {
                            "start_s": 0.5,
                            "end_s": 3.5,
                            "text": "Synthetic caption",
                            "confidence": 1.0,
                            "token_id": 1,
                        }
                    ],
                    "language": "en",
                    "duration_s": 4.0,
                    "text": "Synthetic caption",
                }
            ),
        )
    store = open_project_store(project_dir)
    _ = store.initialize_project(probe=FFprobeProbe())
    created = store.create_variant(candidate_id=candidate.candidate_id, name="Synthetic")
    copied = store.duplicate_variant(
        variant_id=created.variants[0].variant_id,
        name="Format copy",
        output_aspect=aspect,
    ).variants[-1]
    _ = store.split_position(
        variant_id=copied.variant_id,
        position_id=copied.layout_timeline[0].position_id,
        at_ms=2000,
    )
    _ = store.set_captions(variant_id=copied.variant_id, enabled=True)
    preview = generate_preview(
        PreviewRequest(project_dir, copied.variant_id, copied.layout_timeline[0].position_id, 1000),
        probe=FFprobeProbe(),
        runner=SubprocessRunner(),
    )
    assert (preview.width, preview.height) == preview_dimensions(aspect)
    assert (
        main(
            [
                "--quiet",
                "studio",
                "preview-motion",
                "--project-dir",
                str(project_dir),
                "--variant-id",
                str(copied.variant_id),
                "--position-id",
                str(copied.layout_timeline[0].position_id),
                "--at-ms",
                "500",
                "--duration-ms",
                "1000",
            ]
        )
        == 0
    )
    motion_info = FFprobeProbe().probe(
        project_dir / cast("str", _parse_cli_out(capsys.readouterr().out)["preview_path"])
    )
    assert (motion_info.width, motion_info.height) == preview_dimensions(aspect)
    for action in ("enqueue", "run"):
        args = [
            "--quiet",
            "studio",
            "export-queue",
            "--project-dir",
            str(project_dir),
            "--action",
            action,
        ]
        if action == "enqueue":
            args.extend(["--variant-id", str(copied.variant_id)])
        assert main(args) == 0
        _ = capsys.readouterr()
    queue = ExportQueueDocument.model_validate_json(
        (project_dir / "export_queue.json").read_text(encoding="utf-8")
    )
    job = queue.jobs[0]
    assert job.state == "completed", job.error
    output = FFprobeProbe().probe(project_dir / job.output_path)
    assert (output.width, output.height) == final_dimensions(aspect)
    assert output.has_audio
    assert output.duration_s == pytest.approx(4.0, abs=0.1)


def _variant(*, start_ms: int = 10_000, end_ms: int = 30_000) -> Variant:
    return create_initial_variant(
        name="Variant 1",
        selection=VariantRange(
            start_ms=start_ms,
            end_ms=end_ms,
            source_duration_ms=60_000,
        ),
        origin=_origin(),
    )


def _analysis_doc(*, candidates: tuple[AnalysisCandidate, ...] = ()) -> AnalysisDocument:
    return AnalysisDocument(
        schema_version="1",
        project_id="a" * 32,
        project_name="clip",
        source=AnalysisSource(url="https://example.com/v", sha256=sha256_bytes(b"source")),
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


class _JpegProbe:
    def __init__(self, width: int, height: int) -> None:
        self._width: int = width
        self._height: int = height

    def probe(self, path: Path) -> MediaInfo:
        if path.suffix == ".jpg":
            if b"preview" not in path.read_bytes():
                msg = "invalid jpeg fixture"
                raise ProbeError(msg)
            return MediaInfo(
                format_name="image2",
                has_video=True,
                has_audio=False,
                duration_s=None,
                width=self._width,
                height=self._height,
            )
        return MediaInfo(
            format_name="mp4",
            has_video=True,
            has_audio=True,
            duration_s=60.0,
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


class _PreviewRunner:
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


class _MotionRunner:
    def __init__(self) -> None:
        self.invocations: list[tuple[str, ...]] = []
        self.seen_ass: list[str] = []

    def run(self, argv: tuple[str, ...], *, timeout_s: float, cwd: Path | None = None) -> object:
        _ = timeout_s
        _ = cwd
        self.invocations.append(tuple(argv))
        items = list(argv)
        idx = items.index("-filter_complex")
        self.seen_ass.append(str(items[idx + 1]))
        dest = Path(argv[-1])
        _ = dest.write_bytes(b"fake-mp4")
        return None


class _FinalRunner:
    def __init__(self) -> None:
        self.invocations: list[tuple[tuple[str, ...], Path | None]] = []

    def run(self, argv: tuple[str, ...], *, timeout_s: float, cwd: Path | None = None) -> object:
        _ = timeout_s
        self.invocations.append((tuple(argv), cwd))
        dest = Path(argv[-1])
        dest.parent.mkdir(parents=True, exist_ok=True)
        _ = dest.write_bytes(b"final-mp4-data")
        return None


class _VerticalJpegProbe:
    def __init__(self) -> None:
        self.calls: int = 0

    def probe(self, path: Path) -> MediaInfo:
        self.calls += 1
        if path.suffix == ".jpg" and "-source" not in path.name and ".tmp" not in path.name:
            return MediaInfo(
                format_name="image2",
                has_video=True,
                has_audio=False,
                duration_s=None,
                width=720,
                height=1280,
            )
        if path.suffix == ".jpg":
            return MediaInfo(
                format_name="image2",
                has_video=True,
                has_audio=False,
                duration_s=None,
                width=1280,
                height=720,
            )
        return MediaInfo(
            format_name="mp4",
            has_video=True,
            has_audio=True,
            duration_s=60.0,
            width=1920,
            height=1080,
        )


class _AssCapturingRunner:
    def __init__(self) -> None:
        self.ass_text: str = ""
        self.filter_complex: str = ""

    def run(self, argv: tuple[str, ...], *, timeout_s: float, cwd: Path | None = None) -> object:
        _ = timeout_s
        _ = cwd
        items = list(argv)
        idx = items.index("-filter_complex")
        self.filter_complex = str(items[idx + 1])
        start = self.filter_complex.find("filename='")
        end = self.filter_complex.find("'", start + len("filename='"))
        ass_name = self.filter_complex[start + len("filename='") : end]
        ass_path = Path(cwd) / ass_name if cwd is not None else Path(ass_name)
        if ass_path.is_file():
            self.ass_text = ass_path.read_text(encoding="utf-8")
        dest = Path(argv[-1])
        dest.parent.mkdir(parents=True, exist_ok=True)
        _ = dest.write_bytes(b"fake-mp4")
        return None


def _setup_project(tmp_path: Path) -> tuple[Path, ProjectStore, Variant]:
    candidate = _candidate(candidate_id="b" * 64)
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


def test_legacy_document_without_aspect_means_vertical() -> None:
    variant = _variant()
    assert variant.output_aspect is Aspect.VERTICAL


def test_schema1_rejects_horizontal_variant() -> None:
    variant = _variant()
    horizontal = variant.model_copy(update={"output_aspect": Aspect.HORIZONTAL})
    with pytest.raises(ValidationError):
        _ = ProjectDocument(
            schema_version="1",
            project_id="p1",
            analysis_sha256=_HASH,
            transcript_sha256=_HASH,
            source_sha256=_HASH,
            source_duration_ms=60_000,
            variants=(horizontal,),
        )


def test_schema2_accepts_horizontal_variant() -> None:
    variant = _variant()
    horizontal = variant.model_copy(update={"output_aspect": Aspect.HORIZONTAL})
    document = ProjectDocument(
        schema_version="2",
        project_id="p1",
        analysis_sha256=_HASH,
        transcript_sha256=_HASH,
        source_sha256=_HASH,
        source_duration_ms=60_000,
        variants=(horizontal,),
    )
    assert document.variants[0].output_aspect is Aspect.HORIZONTAL


def test_rejects_unsupported_aspect() -> None:
    variant = _variant()
    payload = variant.model_dump(mode="json")
    payload["output_aspect"] = Aspect.SQUARE.value
    with pytest.raises(ValidationError):
        _ = Variant.model_validate(payload)
    with pytest.raises(ValueError, match="unsupported Studio output aspect"):
        _ = duplicate_variant(variant, name="Bad", output_aspect=Aspect.SQUARE)


def test_duplicate_preserves_aspect_by_default() -> None:
    variant = _variant()
    horizontal = variant.model_copy(update={"output_aspect": Aspect.HORIZONTAL})
    duplicate = duplicate_variant(horizontal, name="Copy")
    assert duplicate.output_aspect is Aspect.HORIZONTAL
    assert duplicate.variant_id != horizontal.variant_id


def test_duplicate_with_output_aspect_changes_only_copy() -> None:
    variant = _variant()
    assert variant.output_aspect is Aspect.VERTICAL
    duplicate = duplicate_variant(variant, name="Horizontal copy", output_aspect=Aspect.HORIZONTAL)
    assert duplicate.output_aspect is Aspect.HORIZONTAL
    assert variant.output_aspect is Aspect.VERTICAL
    assert duplicate.variant_id != variant.variant_id
    assert duplicate.start_ms == variant.start_ms
    assert duplicate.end_ms == variant.end_ms


def test_duplicate_preserves_all_human_decisions_and_fresh_ids() -> None:
    variant = _variant()
    enriched = variant.model_copy(
        update={"captions_enabled": True, "caption_highlight_color": "&H00000001"}
    )
    duplicate = duplicate_variant(enriched, name="Copy", output_aspect=Aspect.HORIZONTAL)
    assert duplicate.origin == enriched.origin
    assert duplicate.start_ms == enriched.start_ms
    assert duplicate.end_ms == enriched.end_ms
    assert duplicate.status == enriched.status
    assert duplicate.captions_enabled is True
    assert duplicate.caption_highlight_color == "&H00000001"
    assert duplicate.watermark == enriched.watermark
    assert duplicate.caption_corrections == enriched.caption_corrections
    assert duplicate.layout_timeline[0].layout_spec == enriched.layout_timeline[0].layout_spec
    assert duplicate.layout_timeline[0].start_ms == enriched.layout_timeline[0].start_ms
    assert duplicate.layout_timeline[0].end_ms == enriched.layout_timeline[0].end_ms
    assert duplicate.variant_id != enriched.variant_id
    assert duplicate.layout_timeline[0].position_id != enriched.layout_timeline[0].position_id
    assert enriched.output_aspect is Aspect.VERTICAL


def test_legacy_read_is_readonly_and_write_upgrades(tmp_path: Path) -> None:
    project_dir, store, variant = _setup_project(tmp_path)
    persisted = open_project_store(project_dir).read_project()
    assert persisted is not None
    assert persisted.schema_version == "2"
    raw_before = (project_dir / "project.json").read_text(encoding="utf-8")
    assert '"schema_version"' in raw_before
    _ = (variant, store)


def test_legacy_schema1_file_reads_vertical_and_stays_readonly(tmp_path: Path) -> None:
    candidate = _candidate()
    analysis = _analysis_doc(candidates=(candidate,))
    project_dir = tmp_path / "project"
    store = open_project_store(project_dir)
    with store.stage() as stage:
        stage.write_analysis(analysis)
        stage.write_text("source.mp4", "source")
        stage.write_text("transcript.json", "transcript")
        stage.write_text("candidates.json", "candidates")
    store = open_project_store(project_dir)
    _ = store.initialize_project(probe=_InitProbe())
    created = store.create_variant(candidate_id=candidate.candidate_id, name="V1")
    assert created.variants[0].output_aspect is Aspect.VERTICAL
    raw = created.model_dump_json(indent=2)
    legacy_payload = raw.replace('"schema_version": "2"', '"schema_version": "1"')
    if '"schema_version": "1"' not in legacy_payload:
        legacy_payload = raw.replace('"2"', '"1"', 1)
    _ = (project_dir / "project.json").write_text(legacy_payload, encoding="utf-8")
    before = (project_dir / "project.json").read_text(encoding="utf-8")
    reread = open_project_store(project_dir).read_project()
    assert reread is not None
    assert reread.schema_version == "1"
    assert reread.variants[0].output_aspect is Aspect.VERTICAL
    after = (project_dir / "project.json").read_text(encoding="utf-8")
    assert after == before
    upgraded = open_project_store(project_dir).duplicate_variant(
        variant_id=reread.variants[0].variant_id, name="Copy"
    )
    assert upgraded.schema_version == "2"
    assert upgraded.variants[1].output_aspect is Aspect.VERTICAL


def test_legacy_schema1_duplicate_horizontal_from_disk_upgrades(tmp_path: Path) -> None:
    candidate = _candidate()
    analysis = _analysis_doc(candidates=(candidate,))
    project_dir = tmp_path / "project"
    store = open_project_store(project_dir)
    with store.stage() as stage:
        stage.write_analysis(analysis)
        stage.write_text("source.mp4", "source")
        stage.write_text("transcript.json", "transcript")
        stage.write_text("candidates.json", "candidates")
    store = open_project_store(project_dir)
    _ = store.initialize_project(probe=_InitProbe())
    created = store.create_variant(candidate_id=candidate.candidate_id, name="V1")
    raw = created.model_dump_json(indent=2)
    legacy_payload = raw.replace('"schema_version": "2"', '"schema_version": "1"')
    if '"schema_version": "1"' not in legacy_payload:
        legacy_payload = raw.replace('"2"', '"1"', 1)
    _ = (project_dir / "project.json").write_text(legacy_payload, encoding="utf-8")
    reread = open_project_store(project_dir).read_project()
    assert reread is not None
    assert reread.schema_version == "1"
    upgraded = open_project_store(project_dir).duplicate_variant(
        variant_id=reread.variants[0].variant_id,
        name="Horizontal",
        output_aspect=Aspect.HORIZONTAL,
    )
    assert upgraded.schema_version == "2"
    assert upgraded.variants[0].output_aspect is Aspect.VERTICAL
    assert upgraded.variants[1].output_aspect is Aspect.HORIZONTAL
    assert upgraded.variants[1].name == "Horizontal"
    assert upgraded.variants[1].variant_id != reread.variants[0].variant_id


def test_canvas_sizes_are_single_source_of_truth() -> None:
    assert preview_dimensions(Aspect.VERTICAL) == (720, 1280)
    assert preview_dimensions(Aspect.HORIZONTAL) == (1280, 720)
    assert final_dimensions(Aspect.VERTICAL) == (1080, 1920)
    assert final_dimensions(Aspect.HORIZONTAL) == (1920, 1080)


def test_initialize_emits_schema2(tmp_path: Path) -> None:
    project_dir = tmp_path / "project"
    store = open_project_store(project_dir)
    with store.stage() as stage:
        stage.write_analysis(_analysis_doc())
        stage.write_text("source.mp4", "source")
        stage.write_text("transcript.json", "transcript")
        stage.write_text("candidates.json", "candidates")
    document = open_project_store(project_dir).initialize_project(probe=_InitProbe())
    assert document.schema_version == "2"


def test_preview_fingerprint_is_aspect_specific(tmp_path: Path) -> None:
    project_dir, store, variant = _setup_project(tmp_path)
    persisted = store.read_project()
    assert persisted is not None
    horizontal_same_ids = variant.model_copy(update={"output_aspect": Aspect.HORIZONTAL})
    _ = store.write_project(persisted.model_copy(update={"variants": (variant,)}))
    first = generate_preview(
        PreviewRequest(
            project_dir=project_dir,
            variant_id=variant.variant_id,
            position_id=variant.layout_timeline[0].position_id,
            at_ms=15_000,
        ),
        probe=_JpegProbe(width=720, height=1280),
        runner=_PreviewRunner(),
    )
    _ = store.write_project(persisted.model_copy(update={"variants": (horizontal_same_ids,)}))
    second = generate_preview(
        PreviewRequest(
            project_dir=project_dir,
            variant_id=variant.variant_id,
            position_id=variant.layout_timeline[0].position_id,
            at_ms=15_000,
        ),
        probe=_JpegProbe(width=1280, height=720),
        runner=_PreviewRunner(),
    )
    assert first.preview_fingerprint != second.preview_fingerprint
    assert first.preview_path != second.preview_path


def test_motion_fingerprint_is_aspect_specific(tmp_path: Path) -> None:
    project_dir, store, variant = _setup_project(tmp_path)
    persisted = store.read_project()
    assert persisted is not None
    horizontal_same_ids = variant.model_copy(update={"output_aspect": Aspect.HORIZONTAL})
    _ = store.write_project(persisted.model_copy(update={"variants": (variant,)}))
    first = generate_motion_preview(
        MotionPreviewRequest(
            project_dir=project_dir,
            variant_id=variant.variant_id,
            position_id=variant.layout_timeline[0].position_id,
            at_ms=variant.start_ms + 1000,
            duration_ms=2000,
        ),
        probe=_FixedProbe(width=720, height=1280),
        runner=_MotionRunner(),
    )
    _ = store.write_project(persisted.model_copy(update={"variants": (horizontal_same_ids,)}))
    second = generate_motion_preview(
        MotionPreviewRequest(
            project_dir=project_dir,
            variant_id=variant.variant_id,
            position_id=variant.layout_timeline[0].position_id,
            at_ms=variant.start_ms + 1000,
            duration_ms=2000,
        ),
        probe=_FixedProbe(width=1280, height=720),
        runner=_MotionRunner(),
    )
    assert first.fingerprint != second.fingerprint
    assert first.preview_path != second.preview_path


def test_preview_expected_fingerprint_includes_canvas_and_aspect(tmp_path: Path) -> None:
    project_dir, store, variant = _setup_project(tmp_path)
    horizontal = variant.model_copy(update={"output_aspect": Aspect.HORIZONTAL})
    persisted = store.read_project()
    assert persisted is not None
    _ = store.write_project(persisted.model_copy(update={"variants": (horizontal,)}))
    runner = _PreviewRunner()
    result = generate_preview(
        PreviewRequest(
            project_dir=project_dir,
            variant_id=horizontal.variant_id,
            position_id=horizontal.layout_timeline[0].position_id,
            at_ms=15_000,
        ),
        probe=_JpegProbe(width=1280, height=720),
        runner=runner,
    )
    position = horizontal.layout_timeline[0]
    expected = sha256_canonical_json(
        {
            "implementation_version": "1",
            "source_sha256": persisted.source_sha256,
            "variant_id": str(horizontal.variant_id),
            "position_id": str(position.position_id),
            "at_ms": 15_000,
            "layout_spec": position.layout_spec.model_dump(mode="json"),
            "watermark": horizontal.watermark.model_dump(mode="json"),
            "captions_enabled": horizontal.captions_enabled,
            "caption_highlight_color": horizontal.caption_highlight_color,
            "caption_capitalization": horizontal.caption_capitalization.value,
            "caption_style": None,
            "caption_corrections": [],
            "canvas_width": 1280,
            "canvas_height": 720,
            "output_aspect": "16:9",
            "jpeg_qscale": 3,
        }
    )
    assert result.preview_fingerprint == expected


def test_horizontal_preview_uses_horizontal_canvas(tmp_path: Path) -> None:
    project_dir, store, variant = _setup_project(tmp_path)
    horizontal = variant.model_copy(update={"output_aspect": Aspect.HORIZONTAL})
    persisted = store.read_project()
    assert persisted is not None
    _ = store.write_project(persisted.model_copy(update={"variants": (horizontal,)}))
    runner = _PreviewRunner()
    result = generate_preview(
        PreviewRequest(
            project_dir=project_dir,
            variant_id=horizontal.variant_id,
            position_id=horizontal.layout_timeline[0].position_id,
            at_ms=15_000,
        ),
        probe=_JpegProbe(width=1280, height=720),
        runner=runner,
    )
    assert (result.width, result.height) == (1280, 720)
    assert runner.argv is not None
    assert len(runner.invocations) == 2
    composed = " ".join(runner.invocations[0])
    assert "1280" in composed
    assert "720" in composed


def test_preview_cache_rejects_wrong_dimensions(tmp_path: Path) -> None:
    project_dir, store, variant = _setup_project(tmp_path)
    horizontal = variant.model_copy(update={"output_aspect": Aspect.HORIZONTAL})
    persisted = store.read_project()
    assert persisted is not None
    _ = store.write_project(persisted.model_copy(update={"variants": (horizontal,)}))
    first_runner = _PreviewRunner()
    first = generate_preview(
        PreviewRequest(
            project_dir=project_dir,
            variant_id=horizontal.variant_id,
            position_id=horizontal.layout_timeline[0].position_id,
            at_ms=15_000,
        ),
        probe=_JpegProbe(width=1280, height=720),
        runner=first_runner,
    )
    assert first.preview_path.is_file()
    _ = first.preview_path.write_bytes(b"\xff\xd8preview\xff\xd9")
    second_runner = _PreviewRunner()
    probe = _VerticalJpegProbe()
    second = generate_preview(
        PreviewRequest(
            project_dir=project_dir,
            variant_id=horizontal.variant_id,
            position_id=horizontal.layout_timeline[0].position_id,
            at_ms=15_000,
        ),
        probe=probe,
        runner=second_runner,
    )
    assert (second.width, second.height) == (1280, 720)
    assert len(second_runner.invocations) == 2
    assert probe.calls > 0


def test_motion_preview_uses_horizontal_canvas(tmp_path: Path) -> None:
    project_dir, store, variant = _setup_project(tmp_path)
    horizontal = variant.model_copy(update={"output_aspect": Aspect.HORIZONTAL})
    persisted = store.read_project()
    assert persisted is not None
    _ = store.write_project(persisted.model_copy(update={"variants": (horizontal,)}))
    runner = _MotionRunner()
    result = generate_motion_preview(
        MotionPreviewRequest(
            project_dir=project_dir,
            variant_id=horizontal.variant_id,
            position_id=horizontal.layout_timeline[0].position_id,
            at_ms=horizontal.start_ms + 1000,
            duration_ms=2000,
        ),
        probe=_FixedProbe(width=1920, height=1080),
        runner=runner,
    )
    assert (result.width, result.height) == (1280, 720)
    assert len(runner.invocations) == 1
    assert "1280" in runner.seen_ass[0]
    assert "720" in runner.seen_ass[0]


def test_motion_ass_uses_final_horizontal_dimensions(tmp_path: Path) -> None:
    project_dir, store, variant = _setup_project(tmp_path)
    horizontal = variant.model_copy(update={"output_aspect": Aspect.HORIZONTAL})
    persisted = store.read_project()
    assert persisted is not None
    updated_variant = horizontal.model_copy(update={"captions_enabled": True})
    _ = store.write_project(persisted.model_copy(update={"variants": (updated_variant,)}))
    runner = _AssCapturingRunner()
    result = generate_motion_preview(
        MotionPreviewRequest(
            project_dir=project_dir,
            variant_id=updated_variant.variant_id,
            position_id=updated_variant.layout_timeline[0].position_id,
            at_ms=updated_variant.start_ms + 1000,
            duration_ms=2000,
        ),
        probe=_FixedProbe(width=1920, height=1080),
        runner=runner,
    )
    assert (result.width, result.height) == (1280, 720)
    assert "PlayResX: 1920" in runner.ass_text
    assert "PlayResY: 1080" in runner.ass_text


def test_final_render_horizontal_passes_gate_and_rejects_mismatch(tmp_path: Path) -> None:
    project_dir, store, variant = _setup_project(tmp_path)
    horizontal = variant.model_copy(update={"output_aspect": Aspect.HORIZONTAL})
    persisted = store.read_project()
    assert persisted is not None
    _ = store.write_project(persisted.model_copy(update={"variants": (horizontal,)}))
    runner = _FinalRunner()
    ok = fr.render_final_variant(
        fr.FinalRenderRequest(project_dir=project_dir, variant_id=horizontal.variant_id),
        probe=_FixedProbe(width=1920, height=1080),
        runner=runner,
    )
    assert (ok.width, ok.height) == (1920, 1080)
    assert ok.gate_verdict == "pass"
    with pytest.raises(fr.FinalRenderError) as exc_info:
        _ = fr.render_final_variant(
            fr.FinalRenderRequest(project_dir=project_dir, variant_id=horizontal.variant_id),
            probe=_FixedProbe(width=1080, height=1920),
            runner=_FinalRunner(),
        )
    assert exc_info.value.error_code == "gate_rejected"


def test_final_render_uses_horizontal_canvas_and_ass(tmp_path: Path) -> None:
    project_dir, store, variant = _setup_project(tmp_path)
    horizontal = variant.model_copy(
        update={"output_aspect": Aspect.HORIZONTAL, "captions_enabled": True}
    )
    persisted = store.read_project()
    assert persisted is not None
    _ = store.write_project(persisted.model_copy(update={"variants": (horizontal,)}))
    runner = _FinalRunner()
    result = fr.render_final_variant(
        fr.FinalRenderRequest(project_dir=project_dir, variant_id=horizontal.variant_id),
        probe=_FixedProbe(width=1920, height=1080),
        runner=runner,
    )
    assert (result.width, result.height) == (1920, 1080)
    assert len(runner.invocations) == 1
    argv, _cwd = runner.invocations[0]
    items = list(argv)
    idx = items.index("-filter_complex")
    filter_complex = str(items[idx + 1])
    assert "1920" in filter_complex
    assert "1080" in filter_complex
    assert "subtitles=" in filter_complex


def test_cli_duplicate_variant_accepts_output_aspect(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _disable_logging(monkeypatch)
    project_dir, store, variant = _setup_project(tmp_path)
    _ = store

    def open_store(_path: Path) -> ProjectStore:
        return open_project_store(project_dir)

    monkeypatch.setattr("kliptych.__main__.open_project_store", open_store)
    code = main(
        [
            "--quiet",
            "studio",
            "duplicate-variant",
            "--project-dir",
            str(project_dir),
            "--variant-id",
            str(variant.variant_id),
            "--name",
            "Horizontal",
            "--output-aspect",
            "16:9",
        ]
    )
    assert code == 0
    event = _parse_cli_out(capsys.readouterr().out)
    assert event["type"] == "variant_duplicated"
    dumped = cast("dict[str, object]", event["variant"])
    assert dumped["output_aspect"] == "16:9"
    assert dumped["name"] == "Horizontal"


def test_cli_duplicate_variant_omitted_stays_backward_compatible(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _disable_logging(monkeypatch)

    seen: dict[str, object] = {}

    class _Variant:
        variant_id: str = "11111111-1111-1111-1111-111111111111"

        @staticmethod
        def model_dump(**_kwargs: object) -> dict[str, str]:
            return {"variant_id": "id"}

    variant = _Variant()
    project_variants: tuple[object, ...] = (variant,)

    class _Store:
        @staticmethod
        def duplicate_variant(*, variant_id: UUID, name: str) -> object:
            seen["variant_id"] = variant_id
            seen["name"] = name
            return SimpleNamespace(variants=project_variants)

    def open_store(_path: Path) -> _Store:
        return _Store()

    monkeypatch.setattr("kliptych.__main__.open_project_store", open_store)
    code = main(
        [
            "--quiet",
            "studio",
            "duplicate-variant",
            "--project-dir",
            str(tmp_path),
            "--variant-id",
            "11111111-1111-1111-1111-111111111111",
            "--name",
            "Copia",
        ]
    )
    assert code == 0
    event = _parse_cli_out(capsys.readouterr().out)
    assert event["type"] == "variant_duplicated"
    assert seen["name"] == "Copia"
    assert str(seen["variant_id"]) == "11111111-1111-1111-1111-111111111111"
    _ = uuid4()
