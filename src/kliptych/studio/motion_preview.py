"""Short temporal (motion) previews for one B2.4 position."""

import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from kliptych.gate.probe import MediaProbe
from kliptych.hashing import sha256_canonical_json
from kliptych.layout_geometry import build_layout_filter_chains
from kliptych.studio.preview import PreviewError  # type: ignore[attr-defined]
from kliptych.studio.project_models import LayoutPosition, ProjectDocument, Variant
from kliptych.studio.project_store import ProjectStoreError, open_project_store

_MOTION_PREVIEW_IMPLEMENTATION_VERSION = "1"
_MOTION_PREVIEW_DIR = "previews"
_MOTION_CANVAS_WIDTH = 720
_MOTION_CANVAS_HEIGHT = 1280
_DEFAULT_DURATION_MS = 3000
_MIN_DURATION_MS = 500
_MAX_DURATION_MS = 10000


class MotionPreviewRunner(Protocol):
    """Runs FFmpeg for a motion preview."""

    def run(self, argv: tuple[str, ...], *, timeout_s: float) -> object:
        """Run FFmpeg."""
        ...


@dataclass(frozen=True, slots=True)
class MotionPreviewRequest:
    """Request for a motion preview."""

    project_dir: Path
    variant_id: uuid.UUID
    position_id: uuid.UUID
    at_ms: int
    duration_ms: int = _DEFAULT_DURATION_MS


@dataclass(frozen=True, slots=True)
class MotionPreviewResult:
    """Result of a motion preview."""

    preview_path: Path
    fingerprint: str
    frame_start_ms: int
    frame_end_ms: int
    width: int
    height: int


def generate_motion_preview(
    request: MotionPreviewRequest,
    *,
    probe: MediaProbe,
    runner: MotionPreviewRunner,
    ffmpeg: str = "ffmpeg",
) -> MotionPreviewResult:
    """Generate a short motion preview.

    Returns:
        The rendered motion preview result.

    Raises:
        PreviewError: If the request cannot be rendered.
    """
    try:
        store = open_project_store(request.project_dir)
        project = store.read_project()
    except ProjectStoreError as error:
        raise PreviewError(error.message) from error
    if project is None:
        msg = "Motion preview requires initialized editable project state."
        raise PreviewError(msg)
    _validate_evidence(request.project_dir)
    variant = _find_variant(project, request.variant_id)
    position = _find_position(variant, request.position_id)
    if not (position.start_ms <= request.at_ms < position.end_ms):
        msg = "Motion preview timestamp must be inside the selected position."
        raise PreviewError(msg)
    duration_ms = max(_MIN_DURATION_MS, min(_MAX_DURATION_MS, request.duration_ms))
    start_ms = request.at_ms
    end_ms = min(start_ms + duration_ms, position.end_ms)
    if end_ms - start_ms < _MIN_DURATION_MS:
        end_ms = min(position.end_ms, start_ms + _MIN_DURATION_MS)
    source = request.project_dir / "source.mp4"

    dimensions = _probe_dimensions(probe, source)
    fingerprint = _motion_fingerprint(project.source_sha256, variant, position, start_ms, end_ms)
    destination = _motion_preview_path(request.project_dir, request.variant_id, fingerprint)
    if destination.is_file():
        return MotionPreviewResult(
            preview_path=destination,
            fingerprint=fingerprint,
            frame_start_ms=start_ms,
            frame_end_ms=end_ms,
            width=_MOTION_CANVAS_WIDTH,
            height=_MOTION_CANVAS_HEIGHT,
        )
    _render_motion_preview(
        source=source,
        position=position,
        start_ms=start_ms,
        end_ms=end_ms,
        dimensions=dimensions,
        destination=destination,
        ffmpeg=ffmpeg,
        runner=runner,
    )
    return MotionPreviewResult(
        preview_path=destination,
        fingerprint=fingerprint,
        frame_start_ms=start_ms,
        frame_end_ms=end_ms,
        width=_MOTION_CANVAS_WIDTH,
        height=_MOTION_CANVAS_HEIGHT,
    )


def _validate_evidence(project_dir: Path) -> None:
    transcript = project_dir / "transcript.json"
    if not transcript.is_file():
        msg = "Preview requires a regular transcript artifact."
        raise PreviewError(msg)


def _find_variant(project: ProjectDocument, variant_id: uuid.UUID) -> Variant:
    for v in project.variants:
        if v.variant_id == variant_id:
            if v.status.value != "active":
                msg = "Variant requires review before preview."
                raise PreviewError(msg)
            return v
    msg = "Variant not found."
    raise PreviewError(msg)


def _find_position(variant: Variant, position_id: uuid.UUID) -> LayoutPosition:
    for p in variant.layout_timeline:
        if p.position_id == position_id:
            return p
    msg = "Position not found."
    raise PreviewError(msg)


def _probe_dimensions(probe: MediaProbe, source: Path) -> tuple[int, int]:
    info = probe.probe(source)
    width = info.width if info.width is not None else 0
    height = info.height if info.height is not None else 0
    return int(width), int(height)


def _motion_fingerprint(
    source_sha256: str, variant: Variant, position: LayoutPosition, start_ms: int, end_ms: int
) -> str:
    return sha256_canonical_json(
        {
            "implementation_version": _MOTION_PREVIEW_IMPLEMENTATION_VERSION,
            "source_sha256": source_sha256,
            "variant_id": str(variant.variant_id),
            "position_id": str(position.position_id),
            "start_ms": start_ms,
            "end_ms": end_ms,
            "layout_spec": position.layout_spec.model_dump(mode="json"),
            "watermark": variant.watermark.model_dump(mode="json"),
            "captions_enabled": variant.captions_enabled,
            "caption_highlight_color": variant.caption_highlight_color,
            "caption_capitalization": variant.caption_capitalization.value,
            "caption_corrections": [c.model_dump(mode="json") for c in variant.caption_corrections],
            "canvas_width": _MOTION_CANVAS_WIDTH,
            "canvas_height": _MOTION_CANVAS_HEIGHT,
        }
    )


def _motion_preview_path(project_dir: Path, variant_id: uuid.UUID, fingerprint: str) -> Path:
    root = project_dir / _MOTION_PREVIEW_DIR
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
    return directory / f"motion_{fingerprint}.mp4"


def _render_motion_preview(  # ruff: ignore[too-many-arguments]
    *,
    source: Path,
    position: LayoutPosition,
    start_ms: int,
    end_ms: int,
    dimensions: tuple[int, int],
    destination: Path,
    ffmpeg: str,
    runner: MotionPreviewRunner,
) -> None:

    layout_spec = position.layout_spec
    n_bands = len(layout_spec.bands)
    source_dimensions = tuple((dimensions[0], dimensions[1]) for _ in range(n_bands))
    start_s = start_ms / 1000.0
    dur_s = (end_ms - start_ms) / 1000.0

    filter_parts: list[str] = []
    if n_bands > 1:
        filter_parts.append(
            "[0:v]split=" + str(n_bands) + "".join(f"[v{i}]" for i in range(n_bands))
        )
    labels: list[str] = []
    for i in range(n_bands):
        src_label = f"[v{i}]" if n_bands > 1 else "[0:v]"
        filter_parts.append(
            f"{src_label}trim=start={start_s}:duration={dur_s},setpts=PTS-STARTPTS[t{i}]"
        )
        labels.append(f"[t{i}]")
    chains = build_layout_filter_chains(
        layout_spec=layout_spec,
        canvas_width=_MOTION_CANVAS_WIDTH,
        canvas_height=_MOTION_CANVAS_HEIGHT,
        source_dimensions=source_dimensions,
        input_labels=labels,
    )
    filter_parts.extend(chains)
    if n_bands > 1:
        filter_parts.append(
            "".join(f"[b{i}]" for i in range(n_bands)) + f"vstack=inputs={n_bands}[outv]"
        )
    else:
        filter_parts.append("[b0]copy[outv]")

    filter_complex = ";".join(filter_parts)
    argv = [
        ffmpeg,
        "-hide_banner",
        "-nostdin",
        "-v",
        "error",
        "-y",
        "-ss",
        str(start_s),
        "-t",
        str(dur_s),
        "-i",
        str(source),
        "-filter_complex",
        filter_complex,
        "-map",
        "[outv]",
        "-t",
        str(dur_s),
        "-c:v",
        "libx264",
        "-preset",
        "ultrafast",
        "-crf",
        "28",
        "-pix_fmt",
        "yuv420p",
        "-an",
        str(destination),
    ]
    _ = runner.run(tuple(argv), timeout_s=120.0)
    if not destination.is_file():
        msg = "Motion preview render failed."
        raise PreviewError(msg)
