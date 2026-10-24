"""Exact-frame JPEG composition previews for the active B2.2 layout position."""

import os
import tempfile
from collections.abc import Callable, Sequence
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol
from uuid import UUID

from pydantic import ValidationError

from kliptych.environment import CommandRunner
from kliptych.gate.probe import MediaProbe, ProbeError
from kliptych.hashing import sha256_canonical_json, sha256_file
from kliptych.layout_geometry import build_layout_filter_chains
from kliptych.studio.models import AnalysisDocument
from kliptych.studio.project_models import LayoutPosition, ProjectDocument, ProjectStatus, Variant
from kliptych.studio.project_store import ProjectStoreError, open_project_store

_CANVAS_WIDTH = 720
_CANVAS_HEIGHT = 1280
_JPEG_QSCALE = 3
_PREVIEW_IMPLEMENTATION_VERSION = "1"
_PREVIEW_DIR = "previews"
_SOURCE_NAME = "source.mp4"
_JPEG_MARKER_BYTES = 4
_MIN_SOURCE_DIMENSION = 2
_EVEN_DIVISOR = 2
_SOURCE_FRAME_MAX_SIDE = 1280


class PreviewError(Exception):
    """A preview request cannot produce an exact composed JPEG."""

    def __init__(self, message: str) -> None:
        """Create a safe preview boundary error."""
        super().__init__(message)
        self.message: str = message


class PreviewRunner(Protocol):
    """Executes the prepared FFmpeg JPEG preview invocation."""

    def run(self, argv: tuple[str, ...], *, timeout_s: float) -> object:
        """Run FFmpeg through the project's approved command boundary."""
        ...


@dataclass(frozen=True, slots=True)
class PreviewRequest:
    """An exact source timestamp inside one active variant position."""

    project_dir: Path
    variant_id: UUID
    position_id: UUID
    at_ms: int


@dataclass(frozen=True, slots=True)
class PreviewResult:
    """A fingerprinted composed JPEG and its validated source state."""

    preview_path: Path
    source_frame_path: Path
    preview_fingerprint: str
    frame_time_ms: int
    width: int
    height: int
    layout_position: LayoutPosition


@dataclass(frozen=True, slots=True)
class _RenderInput:
    """Validated data needed to render one exact composition JPEG."""

    source: Path
    position: LayoutPosition
    dimensions: tuple[int, int]
    at_ms: int
    destination: Path
    ffmpeg: str


@dataclass(frozen=True, slots=True)
class _SourceFrameInput:
    """Inputs needed to render the exact uncropped source canvas frame."""

    source: Path
    at_ms: int
    destination: Path
    ffmpeg: str


@dataclass(frozen=True, slots=True)
class _PreviewPairPaths:
    """Destination and temporary paths for one composed/source cache pair."""

    composed_destination: Path
    source_destination: Path
    source_hash_destination: Path
    composed_temp: Path
    source_temp: Path
    source_hash_temp: Path


def generate_preview(
    request: PreviewRequest,
    *,
    probe: MediaProbe,
    runner: CommandRunner,
    ffmpeg: str = "ffmpeg",
) -> PreviewResult:
    """Compose one exact source frame through the active layout position.

    Returns:
        The fingerprinted JPEG result.

    Raises:
        PreviewError: If project evidence, selection, timestamp, source, or
            preview output is invalid.
    """
    try:
        store = open_project_store(request.project_dir)
        project = store.read_project()
    except ProjectStoreError as error:
        raise PreviewError(error.message) from error
    if project is None:
        msg = "Preview requires initialized editable project state."
        raise PreviewError(msg)
    analysis = _read_analysis(request.project_dir)
    _validate_current_evidence(request.project_dir, project)
    variant = _active_variant(
        project.variants, request.variant_id, analysis=analysis, project=project
    )
    position = _position(variant, request.position_id)
    if not position.start_ms <= request.at_ms < position.end_ms:
        msg = "Preview timestamp must be inside the selected position."
        raise PreviewError(msg)
    source = _validated_source(request.project_dir, project.source_sha256, analysis=analysis)
    dimensions = _dimensions(probe, source)
    fingerprint = _fingerprint(project.source_sha256, variant, position, request.at_ms)
    destination = _preview_path(request.project_dir, request.variant_id, fingerprint)
    source_destination = _source_preview_path(destination)
    source_hash_destination = _source_hash_path(source_destination)
    _require_regular_cache_path(source_destination)
    _require_regular_cache_path(source_hash_destination)
    if (
        _is_valid_jpeg(destination, probe=probe)
        and _is_valid_source_frame(source_destination, source_dimensions=dimensions, probe=probe)
        and _has_matching_source_hash(source_destination, source_hash_destination)
    ):
        return PreviewResult(
            preview_path=destination,
            source_frame_path=source_destination,
            preview_fingerprint=fingerprint,
            frame_time_ms=request.at_ms,
            width=_CANVAS_WIDTH,
            height=_CANVAS_HEIGHT,
            layout_position=position,
        )
    _invalidate_preview_pair(destination, source_destination, source_hash_destination)
    paths = _reserve_preview_pair_paths(
        composed_destination=destination,
        source_destination=source_destination,
        source_hash_destination=source_hash_destination,
    )
    try:
        _render_preview(
            _RenderInput(
                source=source,
                position=position,
                dimensions=dimensions,
                at_ms=request.at_ms,
                destination=paths.composed_temp,
                ffmpeg=ffmpeg,
            ),
            runner=runner,
            probe=probe,
        )
        _render_source_frame(
            _SourceFrameInput(
                source=source,
                at_ms=request.at_ms,
                destination=paths.source_temp,
                ffmpeg=ffmpeg,
            ),
            runner=runner,
            probe=probe,
        )
        _write_source_hash(paths.source_temp, destination=paths.source_hash_temp)
        _publish_preview_pair(paths)
    finally:
        _remove_preview_temp(paths.composed_temp)
        _remove_preview_temp(paths.source_temp)
        _remove_preview_temp(paths.source_hash_temp)
    return PreviewResult(
        preview_path=destination,
        source_frame_path=source_destination,
        preview_fingerprint=fingerprint,
        frame_time_ms=request.at_ms,
        width=_CANVAS_WIDTH,
        height=_CANVAS_HEIGHT,
        layout_position=position,
    )


def _active_variant(
    variants: Sequence[Variant],
    variant_id: UUID,
    *,
    analysis: AnalysisDocument,
    project: ProjectDocument,
) -> Variant:
    variant = next((item for item in variants if item.variant_id == variant_id), None)
    if variant is None:
        msg = "Preview variant is not part of the editable project."
        raise PreviewError(msg)
    if variant.status is not ProjectStatus.ACTIVE:
        msg = "Preview requires an active variant."
        raise PreviewError(msg)
    if (
        variant.origin.analysis_sha256 != project.analysis_sha256
        or variant.origin.transcript_sha256 != project.transcript_sha256
        or variant.origin.source_sha256 != project.source_sha256
        or variant.origin.source_candidate_id
        not in {candidate.candidate_id for candidate in analysis.candidates}
    ):
        msg = "Preview active variant origin does not match current B1 evidence."
        raise PreviewError(msg)
    return variant


def _position(variant: Variant, position_id: UUID) -> LayoutPosition:
    position = next(
        (item for item in variant.layout_timeline if item.position_id == position_id), None
    )
    if position is None:
        msg = "Preview position is not part of the selected variant."
        raise PreviewError(msg)
    return position


def _validated_source(
    project_dir: Path, expected_sha256: str, *, analysis: AnalysisDocument
) -> Path:
    source = project_dir / _SOURCE_NAME
    if not source.is_file() or source.is_symlink() or source.is_junction():
        msg = "Preview requires a regular source artifact."
        raise PreviewError(msg)
    if sha256_file(source) != expected_sha256:
        msg = "Preview source hash does not match editable project evidence."
        raise PreviewError(msg)
    if analysis.source.sha256 != expected_sha256:
        msg = "Preview analysis source evidence does not match editable project evidence."
        raise PreviewError(msg)
    return source


def _read_analysis(project_dir: Path) -> AnalysisDocument:
    path = project_dir / "analysis.json"
    if not path.is_file() or path.is_symlink() or path.is_junction():
        msg = "Preview requires a regular analysis artifact."
        raise PreviewError(msg)
    try:
        return AnalysisDocument.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValidationError, ValueError) as error:
        msg = "Preview analysis artifact is invalid."
        raise PreviewError(msg) from error


def _validate_current_evidence(project_dir: Path, project: ProjectDocument) -> None:
    transcript = project_dir / "transcript.json"
    if not transcript.is_file() or transcript.is_symlink() or transcript.is_junction():
        msg = "Preview requires a regular transcript artifact."
        raise PreviewError(msg)
    if sha256_file(project_dir / "analysis.json") != project.analysis_sha256:
        msg = "Preview analysis evidence does not match editable project evidence."
        raise PreviewError(msg)
    if sha256_file(transcript) != project.transcript_sha256:
        msg = "Preview transcript evidence does not match editable project evidence."
        raise PreviewError(msg)


def _dimensions(probe: MediaProbe, source: Path) -> tuple[int, int]:
    try:
        media = probe.probe(source)
    except Exception as error:
        msg = "Preview source dimensions could not be measured."
        raise PreviewError(msg) from error
    if media.width is None or media.height is None:
        msg = "Preview source must have measured video dimensions."
        raise PreviewError(msg)
    if media.width < _MIN_SOURCE_DIMENSION or media.height < _MIN_SOURCE_DIMENSION:
        msg = "Preview source dimensions must be at least two pixels."
        raise PreviewError(msg)
    return (media.width, media.height)


def _fingerprint(
    source_sha256: str,
    variant: Variant,
    position: LayoutPosition,
    at_ms: int,
) -> str:
    return sha256_canonical_json(
        {
            "implementation_version": _PREVIEW_IMPLEMENTATION_VERSION,
            "source_sha256": source_sha256,
            "variant_id": str(variant.variant_id),
            "position_id": str(position.position_id),
            "at_ms": at_ms,
            "layout_spec": position.layout_spec.model_dump(mode="json"),
            "watermark": variant.watermark.model_dump(mode="json"),
            "canvas_width": _CANVAS_WIDTH,
            "canvas_height": _CANVAS_HEIGHT,
            "jpeg_qscale": _JPEG_QSCALE,
        }
    )


def _preview_path(project_dir: Path, variant_id: UUID, fingerprint: str) -> Path:
    root = project_dir / _PREVIEW_DIR
    if root.is_symlink() or root.is_junction() or (root.exists() and not root.is_dir()):
        msg = "Preview root is not a regular directory."
        raise PreviewError(msg)
    root.mkdir(exist_ok=True)
    directory = root / str(variant_id)
    if (
        directory.is_symlink()
        or directory.is_junction()
        or (directory.exists() and not directory.is_dir())
    ):
        msg = "Preview directory is not a regular directory."
        raise PreviewError(msg)
    directory.mkdir(exist_ok=True)
    destination = directory / f"{fingerprint}.jpg"
    _require_regular_cache_path(destination)
    return destination


def _require_regular_cache_path(path: Path) -> None:
    if path.exists() and (not path.is_file() or path.is_symlink() or path.is_junction()):
        msg = "Preview cache entry is not a regular file."
        raise PreviewError(msg)


def _source_preview_path(composed_path: Path) -> Path:
    return composed_path.with_name(f"{composed_path.stem}-source.jpg")


def _source_hash_path(source_path: Path) -> Path:
    return source_path.with_suffix(".sha256")


def _temporary_preview_path(destination: Path) -> Path:
    fd, temporary = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.stem}-",
        suffix=".tmp.jpg",
    )
    path = Path(temporary)
    try:
        os.close(fd)
    except OSError:
        _remove_preview_temp(path)
        raise
    return path


def _temporary_hash_path(destination: Path) -> Path:
    fd, temporary = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.stem}-",
        suffix=".tmp",
    )
    path = Path(temporary)
    try:
        os.close(fd)
    except OSError:
        _remove_preview_temp(path)
        raise
    return path


def _reserve_preview_pair_paths(
    *,
    composed_destination: Path,
    source_destination: Path,
    source_hash_destination: Path,
) -> _PreviewPairPaths:
    reserved: list[Path] = []
    try:
        composed_temp = _reserve_temp(reserved, composed_destination, _temporary_preview_path)
        source_temp = _reserve_temp(reserved, source_destination, _temporary_preview_path)
        source_hash_temp = _reserve_temp(reserved, source_hash_destination, _temporary_hash_path)
    except OSError as error:
        for path in reserved:
            _remove_preview_temp(path)
        msg = "Preview temporary outputs could not be reserved."
        raise PreviewError(msg) from error
    return _PreviewPairPaths(
        composed_destination=composed_destination,
        source_destination=source_destination,
        source_hash_destination=source_hash_destination,
        composed_temp=composed_temp,
        source_temp=source_temp,
        source_hash_temp=source_hash_temp,
    )


def _reserve_temp(reserved: list[Path], destination: Path, factory: Callable[[Path], Path]) -> Path:
    path = factory(destination)
    reserved.append(path)
    return path


def _publish_preview_pair(paths: _PreviewPairPaths) -> None:
    try:
        _ = paths.source_temp.replace(paths.source_destination)
        _ = paths.source_hash_temp.replace(paths.source_hash_destination)
        _ = paths.composed_temp.replace(paths.composed_destination)
    except OSError as error:
        _remove_preview_temp(paths.source_destination)
        _remove_preview_temp(paths.source_hash_destination)
        _remove_preview_temp(paths.composed_destination)
        msg = "Preview outputs could not be published."
        raise PreviewError(msg) from error


def _remove_preview_temp(path: Path) -> None:
    with suppress(OSError):
        path.unlink()


def _invalidate_preview_pair(
    composed_path: Path, source_path: Path, source_hash_path: Path
) -> None:
    """Remove an incomplete or invalid cache pair before regenerating it."""
    _remove_preview_temp(composed_path)
    _remove_preview_temp(source_path)
    _remove_preview_temp(source_hash_path)


def _render_preview(
    render_input: _RenderInput, *, runner: CommandRunner, probe: MediaProbe
) -> None:
    labels = tuple(
        f"[frame{index}]" for index in range(len(render_input.position.layout_spec.bands))
    )
    split = (
        f"[0:v]trim=start={_seconds(render_input.at_ms)},setpts=PTS-STARTPTS,split={len(labels)}"
        + "".join(labels)
    )
    chains = list(
        build_layout_filter_chains(
            layout_spec=render_input.position.layout_spec,
            canvas_width=_CANVAS_WIDTH,
            canvas_height=_CANVAS_HEIGHT,
            source_dimensions=tuple(render_input.dimensions for _ in labels),
            input_labels=labels,
        )
    )
    output = "[vout]"
    if len(labels) == 1:
        chains.append(f"[b0]null{output}")
    else:
        chains.append(
            "".join(f"[b{index}]" for index in range(len(labels)))
            + f"vstack=inputs={len(labels)}{output}"
        )
    argv = _preview_argv(
        render_input,
        filter_graph=";".join((split, *chains)),
        output=output,
        temporary=render_input.destination,
    )
    try:
        result = runner.run(argv, timeout_s=60.0)
    except Exception as error:
        msg = "Preview renderer failed."
        raise PreviewError(msg) from error
    if not result.ok:
        msg = "Preview renderer failed."
        raise PreviewError(msg)
    _require_jpeg(render_input.destination, probe=probe)


def _preview_argv(
    render_input: _RenderInput, *, filter_graph: str, output: str, temporary: Path
) -> tuple[str, ...]:
    return (
        render_input.ffmpeg,
        "-hide_banner",
        "-nostdin",
        "-v",
        "error",
        "-y",
        "-i",
        str(render_input.source),
        "-filter_complex",
        filter_graph,
        "-map",
        output,
        "-frames:v",
        "1",
        "-c:v",
        "mjpeg",
        "-q:v",
        str(_JPEG_QSCALE),
        str(temporary),
    )


def _require_jpeg(path: Path, *, probe: MediaProbe) -> None:
    if not _is_valid_jpeg(path, probe=probe):
        msg = "Preview renderer did not produce a JPEG."
        raise PreviewError(msg)


def _render_source_frame(
    frame_input: _SourceFrameInput, *, runner: CommandRunner, probe: MediaProbe
) -> None:
    argv = (
        frame_input.ffmpeg,
        "-hide_banner",
        "-nostdin",
        "-v",
        "error",
        "-y",
        "-i",
        str(frame_input.source),
        "-vf",
        "".join(
            (
                f"trim=start={_seconds(frame_input.at_ms)},setpts=PTS-STARTPTS,scale=",
                _source_scale_expression("iw"),
                ":",
                _source_scale_expression("ih"),
            )
        ),
        "-frames:v",
        "1",
        "-c:v",
        "mjpeg",
        "-q:v",
        str(_JPEG_QSCALE),
        str(frame_input.destination),
    )
    try:
        result = runner.run(argv, timeout_s=60.0)
    except Exception as error:
        msg = "Preview source frame renderer failed."
        raise PreviewError(msg) from error
    if not result.ok:
        msg = "Preview source frame renderer failed."
        raise PreviewError(msg)
    _require_source_frame(
        frame_input.destination,
        source_dimensions=_dimensions(probe, frame_input.source),
        probe=probe,
    )


def _is_valid_jpeg(path: Path, *, probe: MediaProbe) -> bool:
    if not path.is_file() or path.is_symlink() or path.is_junction():
        return False
    try:
        media = probe.probe(path)
    except ProbeError:
        return False
    try:
        with path.open("rb") as preview:
            start = preview.read(2)
            _ = preview.seek(-2, os.SEEK_END)
            end = preview.read(2)
    except OSError:
        return False
    return (
        path.stat().st_size >= _JPEG_MARKER_BYTES
        and start == b"\xff\xd8"
        and end == b"\xff\xd9"
        and media.width == _CANVAS_WIDTH
        and media.height == _CANVAS_HEIGHT
    )


def _is_valid_source_frame(
    path: Path, *, source_dimensions: tuple[int, int], probe: MediaProbe
) -> bool:
    if not path.is_file() or path.is_symlink() or path.is_junction():
        return False
    try:
        media = probe.probe(path)
    except ProbeError:
        return False
    return (
        _has_jpeg_markers(path)
        and media.width is not None
        and media.height is not None
        and (media.width, media.height) == _source_frame_dimensions(source_dimensions)
    )


def _require_source_frame(
    path: Path, *, source_dimensions: tuple[int, int], probe: MediaProbe
) -> None:
    if not _is_valid_source_frame(path, source_dimensions=source_dimensions, probe=probe):
        msg = "Preview renderer did not produce a source JPEG."
        raise PreviewError(msg)


def _has_jpeg_markers(path: Path) -> bool:
    if path.stat().st_size < _JPEG_MARKER_BYTES:
        return False
    try:
        with path.open("rb") as preview:
            start = preview.read(2)
            _ = preview.seek(-2, os.SEEK_END)
            end = preview.read(2)
    except OSError:
        return False
    return start == b"\xff\xd8" and end == b"\xff\xd9"


def _source_frame_dimensions(source_dimensions: tuple[int, int]) -> tuple[int, int]:
    source_width, source_height = source_dimensions
    scale = min(
        _SOURCE_FRAME_MAX_SIDE / source_width,
        _SOURCE_FRAME_MAX_SIDE / source_height,
        1,
    )
    width = _floor_even(int(source_width * scale))
    height = _floor_even(int(source_height * scale))
    return (max(_MIN_SOURCE_DIMENSION, width), max(_MIN_SOURCE_DIMENSION, height))


def _source_scale_expression(dimension: str) -> str:
    """Return FFmpeg scale math matching the no-upscale source-frame contract."""
    factor = f"min(1\\,min({_SOURCE_FRAME_MAX_SIDE}/iw\\,{_SOURCE_FRAME_MAX_SIDE}/ih))"
    return f"trunc({dimension}*{factor}/{_EVEN_DIVISOR})*{_EVEN_DIVISOR}"


def _floor_even(value: int) -> int:
    return value // _EVEN_DIVISOR * _EVEN_DIVISOR


def _write_source_hash(source_path: Path, *, destination: Path) -> None:
    try:
        _write_text(destination, sha256_file(source_path))
    except OSError as error:
        msg = "Preview source hash could not be written."
        raise PreviewError(msg) from error


def _has_matching_source_hash(source_path: Path, hash_path: Path) -> bool:
    if not hash_path.is_file() or hash_path.is_symlink() or hash_path.is_junction():
        return False
    try:
        return hash_path.read_text(encoding="utf-8") == sha256_file(source_path)
    except OSError:
        return False


def _write_text(destination: Path, content: str) -> None:
    with destination.open("w", encoding="utf-8") as handle:
        _ = handle.write(content)


def _seconds(at_ms: int) -> str:
    return f"{at_ms / 1000:.3f}"
