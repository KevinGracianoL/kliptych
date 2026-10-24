"""Short temporal (motion) previews for one B2.4 position."""

import contextlib
import os
import tempfile
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from kliptych.contract import CaptionStyleConfig
from kliptych.contract.enums import CaptionCapitalization
from kliptych.gate.probe import MediaProbe
from kliptych.hashing import sha256_canonical_json
from kliptych.layout_geometry import build_layout_filter_chains
from kliptych.studio.preview import PreviewError  # type: ignore[attr-defined]
from kliptych.studio.project_models import (
    LayoutPosition,
    ProjectDocument,
    Variant,
    VariantCaptionCorrection,
)
from kliptych.studio.project_store import ProjectStoreError, open_project_store
from kliptych.subtitles import SubtitleError, SubtitleRenderer, SubtitleStyle
from kliptych.transcribe import Transcript, Word

_MOTION_PREVIEW_IMPLEMENTATION_VERSION = "1"
_MOTION_PREVIEW_DIR = "previews"
_MOTION_CANVAS_WIDTH = 720
_MOTION_CANVAS_HEIGHT = 1280
_DEFAULT_DURATION_MS = 3000
_MIN_DURATION_MS = 500
_MIN_SOURCE_DIMENSION = 2
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
    if destination.is_file() and not destination.is_symlink() and not destination.is_junction():
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
        project_dir=request.project_dir,
        variant=variant,
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
    try:
        info = probe.probe(source)
    except Exception as error:
        msg = "Source dimensions could not be measured."
        raise PreviewError(msg) from error
    width = info.width
    height = info.height
    if (
        width is None
        or height is None
        or width < _MIN_SOURCE_DIMENSION
        or height < _MIN_SOURCE_DIMENSION
    ):
        msg = "Source dimensions are unusable."
        raise PreviewError(msg)
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


def _read_transcript(project_dir: Path) -> Transcript:
    path = project_dir / "transcript.json"
    return Transcript.model_validate_json(path.read_text(encoding="utf-8"))


def _words_in_range(transcript: Transcript, start_ms: int, end_ms: int) -> list[Word]:
    out: list[Word] = []
    for w in transcript.words:
        ws = int(w.start_s * 1000)
        we = int(w.end_s * 1000)
        if we <= start_ms or ws >= end_ms:
            continue
        out.append(w)
    return out


def _apply_caption_corrections(
    words: list[Word],
    corrections: Sequence[VariantCaptionCorrection],
    *,
    start_ms: int,
    end_ms: int,
) -> list[Word]:
    if not corrections:
        return words
    result: list[Word] = []
    for w in words:
        skip = False
        for c in corrections:
            if c.start_ms <= int(w.start_s * 1000) < c.end_ms:
                skip = True
                break
        if not skip:
            result.append(w)
    for c in corrections:
        if c.end_ms <= start_ms or c.start_ms >= end_ms:
            continue
        result.append(
            Word(
                start_s=max(c.start_ms, start_ms) / 1000.0,
                end_s=min(c.end_ms, end_ms) / 1000.0,
                text=c.corrected_text,
                confidence=1.0,
                token_id=0,
            )
        )
    result.sort(key=lambda w: w.start_s)
    return result


def _try_render_ass(  # ruff: ignore[too-many-arguments]
    *,
    project_dir: Path,
    variant: Variant,
    start_ms: int,
    end_ms: int,
    filter_parts: list[str],
    final_label: str,
) -> tuple[Path | None, str]:
    try:
        transcript = _read_transcript(project_dir)
    except (OSError, ValueError, SubtitleError) as error:
        _ = error
        return None, "[outv]"
    words = _words_in_range(transcript, start_ms, end_ms)
    words = _apply_caption_corrections(
        words, variant.caption_corrections, start_ms=start_ms, end_ms=end_ms
    )
    if not words:
        return None, final_label
    try:
        style = SubtitleStyle.from_caption_style(
            CaptionStyleConfig(
                capitalization=CaptionCapitalization(variant.caption_capitalization.value),
                highlight_color=variant.caption_highlight_color,
            )
        )
        renderer = SubtitleRenderer(style=style)
        ass_content = renderer.build(words)
    except (OSError, ValueError, SubtitleError) as error:
        _ = error
        return None, "[outv]"
    try:
        fd, tmp = tempfile.mkstemp(suffix=".ass", prefix="motion_")
        _ = os.close(fd)
        ass_path = Path(tmp)
        _ = ass_path.write_text(ass_content, encoding="utf-8")
    except (OSError, ValueError, SubtitleError) as error:
        _ = error
        return None, "[outv]"
    filter_parts.append(f"[outv]subtitles=filename='{ass_path}'[outv_sub]")
    return ass_path, "[outv_sub]"


def _watermark_xy(position: object) -> tuple[str, str]:
    margin = 20
    name = getattr(position, "value", str(position))
    table = {
        "top_left": (str(margin), str(margin)),
        "top_right": (f"W-w-{margin}", str(margin)),
        "bottom_left": (str(margin), f"H-h-{margin}"),
        "bottom_right": (f"W-w-{margin}", f"H-h-{margin}"),
        "center": ("(W-w)/2", "(H-h)/2"),
        "center_top": ("(W-w)/2", str(margin)),
        "center_bottom": ("(W-w)/2", f"H-h-{margin}"),
    }
    return table.get(name, (f"W-w-{margin}", f"H-h-{margin}"))


def _render_motion_preview(  # ruff: ignore[too-many-arguments,too-many-locals,too-many-branches,too-many-statements,complex-structure]
    *,
    source: Path,
    project_dir: Path,
    variant: Variant,
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

    # Optional watermark PNG overlay.
    extra_inputs: list[str] = []
    wm_enabled = variant.watermark.enabled and variant.watermark.asset_id is not None
    wm_asset_path: Path | None = None
    if wm_enabled:
        candidate = project_dir / "assets" / "watermark" / f"{variant.watermark.asset_id}.png"
        if candidate.is_file():
            wm_asset_path = candidate
            extra_inputs.extend(["-i", str(candidate)])

    # Optional captions (ASS) burn-in.
    ass_path: Path | None = None
    final_label = "[outv]"
    if variant.captions_enabled:
        ass_path, final_label = _try_render_ass(
            project_dir=project_dir,
            variant=variant,
            start_ms=start_ms,
            end_ms=end_ms,
            filter_parts=filter_parts,
            final_label=final_label,
        )

    # Watermark overlay filter.
    if wm_asset_path is not None:
        x_expr, y_expr = _watermark_xy(variant.watermark.position)
        target_w = int(720 * 0.15 / 2) * 2
        filter_parts.append(
            f"{final_label}[1:v]format=rgba,scale={target_w}:-2[wm];{final_label}[wm]overlay={x_expr}:{y_expr}[outv_wm]"
        )
        final_label = "[outv_wm]"

    filter_complex = ";".join(filter_parts)
    argv = [
        ffmpeg,
        "-hide_banner",
        "-nostdin",
        "-v",
        "error",
        "-y",
        "-i",
        str(source),
    ]
    argv += extra_inputs
    argv += [
        "-filter_complex",
        filter_complex,
        "-map",
        final_label,
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
    temp_destination = destination.with_suffix(".tmp.mp4")
    argv = [a if a != str(destination) else str(temp_destination) for a in argv]
    try:
        _ = runner.run(tuple(argv), timeout_s=120.0)
    finally:
        if ass_path is not None:
            with contextlib.suppress(OSError):
                ass_path.unlink()
    if not temp_destination.is_file():
        with contextlib.suppress(OSError):
            temp_destination.unlink()
        msg = "Motion preview render failed."
        raise PreviewError(msg)
    try:
        _ = temp_destination.replace(destination)
    except OSError as error:
        with contextlib.suppress(OSError):
            temp_destination.unlink()
        msg = "Motion preview could not be published."
        raise PreviewError(msg) from error
    if destination.is_symlink() or destination.is_junction():
        with contextlib.suppress(OSError):
            destination.unlink()
        msg = "Motion preview destination is not a regular file."
        raise PreviewError(msg)
