"""Final render and lazy export service for Kliptych Studio variants (Track B3)."""

import contextlib
import logging
import os
import re
import signal
import tempfile
import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import FrameType
from typing import Protocol, cast

from kliptych.contract import CaptionStyleConfig
from kliptych.contract.enums import CaptionCapitalization, Platform
from kliptych.contract.schema import vanilla_contract
from kliptych.encoding import (
    RenderConfig,
    audio_and_container_arguments,
    video_encoder_arguments,
)
from kliptych.fonts import FontError, packaged_fonts_dir
from kliptych.gate.engine import Gate
from kliptych.gate.models import GateResult, MediaInfo, Piece
from kliptych.gate.probe import MediaProbe, ProbeError
from kliptych.hashing import sha256_file
from kliptych.layout_geometry import build_layout_filter_chains
from kliptych.studio.caption_styles import resolve_studio_caption_style
from kliptych.studio.models import AnalysisDocument
from kliptych.studio.project_models import (
    LayoutPosition,
    ProjectDocument,
    Variant,
    VariantCaptionCorrection,
)
from kliptych.studio.project_store import ProjectStoreError, open_project_store
from kliptych.subtitles import (
    SubtitleError,
    SubtitleRenderer,
    SubtitleStyle,
    escape_filter_path,
)
from kliptych.transcribe import Transcript, Word

_CANVAS_WIDTH = 1080
_CANVAS_HEIGHT = 1920
_DELIVERY_DIR = "delivery"
_MIN_SOURCE_DIMENSION = 2
_DEFAULT_TIMEOUT_S = 600.0

type _SignalHandler = Callable[[int, FrameType | None], object] | int | None

logger = logging.getLogger(__name__)


class FinalRenderError(Exception):
    """Raised when a final render cannot be completed or validated."""

    message: str
    error_code: str

    def __init__(self, message: str, *, error_code: str = "final_render_failed") -> None:
        """Initialize the final render error with a message and machine-readable code."""
        super().__init__(message)
        self.message = message
        self.error_code = error_code


class FinalRenderRunner(Protocol):
    """Runs FFmpeg for a final render."""

    def run(self, argv: tuple[str, ...], *, timeout_s: float, cwd: Path | None = None) -> object:
        """Run FFmpeg."""
        ...


@dataclass(frozen=True, slots=True)
class FinalRenderRequest:
    """Request to render and export one variant from a Studio project."""

    project_dir: Path
    variant_id: uuid.UUID
    destination: Path | None = None


@dataclass(frozen=True, slots=True)
class FinalRenderResult:
    """Outcome of a successfully gate-evaluated final render."""

    output_path: Path
    variant_id: uuid.UUID
    duration_ms: int
    width: int
    height: int
    gate_verdict: str
    encoder_used: str


@dataclass(frozen=True, slots=True)
class _RenderContext:
    variant: Variant
    source: Path
    destination: Path
    output_dir: Path
    src_dimensions: tuple[int, int]
    source_has_audio: bool


def _prepare_context(request: FinalRenderRequest, probe: MediaProbe) -> _RenderContext:
    try:
        store = open_project_store(request.project_dir)
        project = store.read_project()
    except ProjectStoreError as error:
        raise FinalRenderError(error.message, error_code="project_error") from error
    if project is None:
        msg = "Final render requires an initialized editable project state."
        raise FinalRenderError(msg, error_code="uninitialized_project")

    _validate_evidence(request.project_dir, project)
    analysis_path = request.project_dir / "analysis.json"
    analysis = AnalysisDocument.model_validate_json(analysis_path.read_text(encoding="utf-8"))
    variant = _find_variant(project, analysis, request.variant_id)
    source = request.project_dir / "source.mp4"

    source_info = _probe_source(probe, source)
    dimensions = _require_dimensions(source_info)
    destination = _resolve_destination(request, variant)
    output_dir = destination.parent
    _require_regular_directory(output_dir)
    resolved_output_dir = output_dir.resolve()

    return _RenderContext(
        variant=variant,
        source=source,
        destination=destination,
        output_dir=resolved_output_dir,
        src_dimensions=dimensions,
        source_has_audio=source_info.has_audio,
    )


def signal_cleanup_handler(signum: int, _frame: object) -> None:
    """Handle termination signals cooperatively during final rendering.

    Raises:
        FinalRenderError: Always raised to trigger deterministic cleanup.
    """
    msg = f"Final render process terminated by signal {signum}."
    raise FinalRenderError(msg, error_code="render_cancelled")


def render_final_variant(  # ruff: ignore[too-many-arguments]
    request: FinalRenderRequest,
    *,
    probe: MediaProbe,
    runner: FinalRenderRunner,
    ffmpeg: str = "ffmpeg",
    nvenc_available: bool = False,
    gate: Gate | None = None,
) -> FinalRenderResult:
    """Render and lazily export one variant to its compliant MP4 deliverable.

    Returns:
        The validated final render result.
    """
    ctx = _prepare_context(request, probe)
    temp_destination = ctx.destination.with_suffix(".tmp.mp4").resolve()

    ass_path, filter_complex, extra_inputs, final_v_label, has_audio = _build_render_plan(
        project_dir=request.project_dir,
        variant=ctx.variant,
        src_dimensions=ctx.src_dimensions,
        source_has_audio=ctx.source_has_audio,
        output_dir=ctx.output_dir,
    )

    render_config = RenderConfig(target_aspect=9 / 16, nvenc_available=nvenc_available)
    effective_gate = gate if gate is not None else Gate(probe=probe, render=render_config)

    previous_signals: dict[int, object] = {}
    for sig in (signal.SIGTERM, signal.SIGINT):
        with contextlib.suppress(ValueError, OSError):
            previous_signals[sig] = signal.signal(sig, signal_cleanup_handler)

    try:
        encoder_used = _execute_ffmpeg_with_fallback(
            ffmpeg=ffmpeg,
            source=ctx.source,
            extra_inputs=extra_inputs,
            filter_complex=filter_complex,
            final_v_label=final_v_label,
            has_audio=has_audio,
            temp_destination=temp_destination,
            output_dir=ctx.output_dir,
            runner=runner,
            nvenc_available=nvenc_available,
        )

        _evaluate_gate(
            gate=effective_gate,
            variant=ctx.variant,
            temp_destination=temp_destination,
        )

        _publish_artifact(temp_destination, ctx.destination)
    except Exception:
        with contextlib.suppress(OSError):
            if temp_destination.exists():
                temp_destination.unlink()
        raise
    finally:
        for sig, prev in previous_signals.items():
            with contextlib.suppress(ValueError, OSError):
                _ = signal.signal(sig, cast("_SignalHandler", prev))
        if ass_path is not None:
            with contextlib.suppress(OSError):
                if ass_path.exists():
                    ass_path.unlink()

    duration_ms, width, height = _measure_output(probe, ctx.destination, ctx.variant)

    return FinalRenderResult(
        output_path=ctx.destination,
        variant_id=ctx.variant.variant_id,
        duration_ms=duration_ms,
        width=width,
        height=height,
        gate_verdict="pass",
        encoder_used=encoder_used,
    )


def _measure_output(probe: MediaProbe, destination: Path, variant: Variant) -> tuple[int, int, int]:
    info = _probe_source(probe, destination)
    w, h = _require_dimensions(info)
    duration_ms = (
        round(info.duration_s * 1000)
        if info.duration_s is not None
        else (variant.end_ms - variant.start_ms)
    )
    return duration_ms, w, h


def _sanitize_filename(name: str) -> str:
    cleaned = re.sub(r"[^\w\-. ]", "_", name.strip())
    cleaned = re.sub(r"^[.\s]+", "", cleaned)
    return cleaned.strip()


def _resolve_destination(request: FinalRenderRequest, variant: Variant) -> Path:
    if request.destination is not None:
        if request.destination.is_symlink() or request.destination.is_junction():
            msg = "Custom destination must not be a symlink or junction."
            raise FinalRenderError(msg, error_code="invalid_destination")
        dest = request.destination.resolve()
        if dest.is_symlink() or dest.is_junction():
            msg = "Custom destination must not be a symlink or junction."
            raise FinalRenderError(msg, error_code="invalid_destination")
        return dest
    sanitized = _sanitize_filename(variant.name) or f"variant_{variant.variant_id}"
    delivery_dir = request.project_dir / _DELIVERY_DIR
    return delivery_dir / f"{sanitized}.mp4"


def _require_regular_directory(path: Path) -> None:
    if path.is_symlink() or path.is_junction():
        msg = f"Delivery directory {path} must not be a symlink or junction."
        raise FinalRenderError(msg, error_code="invalid_directory")
    try:
        path.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        msg = f"Failed to create delivery directory: {error}"
        raise FinalRenderError(msg, error_code="io_error") from error


def _validate_evidence(project_dir: Path, project: ProjectDocument) -> None:
    analysis_path = project_dir / "analysis.json"
    transcript = project_dir / "transcript.json"
    source = project_dir / "source.mp4"
    for file_path, name in (
        (analysis_path, "analysis"),
        (transcript, "transcript"),
        (source, "source"),
    ):
        if not file_path.is_file() or file_path.is_symlink() or file_path.is_junction():
            msg = f"Final render requires a regular {name} artifact."
            raise FinalRenderError(msg, error_code="invalid_evidence")
    if sha256_file(analysis_path) != project.analysis_sha256:
        msg = "Analysis evidence does not match editable project evidence."
        raise FinalRenderError(msg, error_code="evidence_mismatch")
    if sha256_file(transcript) != project.transcript_sha256:
        msg = "Transcript evidence does not match editable project evidence."
        raise FinalRenderError(msg, error_code="evidence_mismatch")
    if sha256_file(source) != project.source_sha256:
        msg = "Source evidence does not match editable project evidence."
        raise FinalRenderError(msg, error_code="evidence_mismatch")
    try:
        analysis = AnalysisDocument.model_validate_json(analysis_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        msg = "Analysis artifact is invalid."
        raise FinalRenderError(msg, error_code="invalid_analysis") from error
    if analysis.source.sha256 != project.source_sha256:
        msg = "Analysis source hash does not match editable project evidence."
        raise FinalRenderError(msg, error_code="evidence_mismatch")


def _find_variant(
    project: ProjectDocument, analysis: AnalysisDocument, variant_id: uuid.UUID
) -> Variant:
    candidate_ids = {c.candidate_id for c in analysis.candidates}
    for v in project.variants:
        if v.variant_id == variant_id:
            if v.status.value != "active":
                msg = "Variant requires review before final render."
                raise FinalRenderError(msg, error_code="variant_not_active")
            if (
                v.origin.analysis_sha256 != project.analysis_sha256
                or v.origin.transcript_sha256 != project.transcript_sha256
                or v.origin.source_sha256 != project.source_sha256
                or v.origin.source_candidate_id not in candidate_ids
            ):
                msg = "Variant origin does not match current evidence."
                raise FinalRenderError(msg, error_code="origin_mismatch")
            return v
    msg = f"Variant {variant_id} not found."
    raise FinalRenderError(msg, error_code="variant_not_found")


def _probe_source(probe: MediaProbe, source: Path) -> MediaInfo:
    try:
        return probe.probe(source)
    except (OSError, ValueError, ProbeError) as error:
        msg = f"Media probe failed for {source.name}: {error}"
        raise FinalRenderError(msg, error_code="probe_error") from error


def _require_dimensions(info: MediaInfo) -> tuple[int, int]:
    width = info.width
    height = info.height
    if (
        width is None
        or height is None
        or width < _MIN_SOURCE_DIMENSION
        or height < _MIN_SOURCE_DIMENSION
    ):
        msg = "Source video dimensions are unusable."
        raise FinalRenderError(msg, error_code="invalid_dimensions")
    return int(width), int(height)


def _watermark_xy(position: object) -> tuple[str, str]:
    margin = 30
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


def _read_transcript(project_dir: Path) -> Transcript:
    path = project_dir / "transcript.json"
    return Transcript.model_validate_json(path.read_text(encoding="utf-8"))


def _words_in_range(transcript: Transcript, start_ms: int, end_ms: int) -> list[Word]:
    out: list[Word] = []
    for w in transcript.words:
        ws = round(w.start_s * 1000)
        we = round(w.end_s * 1000)
        if we <= start_ms or ws >= end_ms:
            continue
        clipped_start_ms = max(ws, start_ms)
        clipped_end_ms = min(we, end_ms)
        if clipped_end_ms <= clipped_start_ms:
            continue
        out.append(
            Word(
                start_s=(clipped_start_ms - start_ms) / 1000.0,
                end_s=(clipped_end_ms - start_ms) / 1000.0,
                text=w.text,
                confidence=w.confidence,
                token_id=w.token_id,
            )
        )
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
        ws = round(w.start_s * 1000)
        we = round(w.end_s * 1000)
        for c in corrections:
            c_start_rel = c.start_ms - start_ms
            c_end_rel = c.end_ms - start_ms
            if ws < c_end_rel and we > c_start_rel:
                skip = True
                break
        if not skip:
            result.append(w)
    for c in corrections:
        if c.end_ms <= start_ms or c.start_ms >= end_ms:
            continue
        c_start = max(c.start_ms, start_ms) - start_ms
        c_end = min(c.end_ms, end_ms) - start_ms
        result.append(
            Word(
                start_s=c_start / 1000.0,
                end_s=c_end / 1000.0,
                text=c.corrected_text,
                confidence=1.0,
                token_id=0,
            )
        )
    result.sort(key=lambda w: w.start_s)
    return result


def _resolve_caption_style_and_fonts(
    variant: Variant,
) -> tuple[SubtitleStyle, Path | None]:
    if variant.caption_style is not None:
        style = resolve_studio_caption_style(
            variant.caption_style,
            highlight_color=variant.caption_highlight_color,
            capitalization=variant.caption_capitalization,
        )
        return style, packaged_fonts_dir()
    style = SubtitleStyle.from_caption_style(
        CaptionStyleConfig(
            capitalization=CaptionCapitalization(variant.caption_capitalization.value),
            highlight_color=variant.caption_highlight_color,
        )
    )
    return style, None


def _build_ass_text(variant: Variant, words: list[Word]) -> tuple[str, Path | None] | None:
    try:
        style, fonts_dir = _resolve_caption_style_and_fonts(variant)
        return SubtitleRenderer(style=style).build(words), fonts_dir
    except (ValueError, SubtitleError, FontError):
        return None


def _try_render_ass(
    *,
    project_dir: Path,
    variant: Variant,
    output_dir: Path,
    filter_parts: list[str],
    current_label: str,
) -> tuple[Path | None, str]:
    try:
        transcript = _read_transcript(project_dir)
    except (OSError, ValueError, SubtitleError):
        return None, current_label
    words = _words_in_range(transcript, variant.start_ms, variant.end_ms)
    words = _apply_caption_corrections(
        words, variant.caption_corrections, start_ms=variant.start_ms, end_ms=variant.end_ms
    )
    if not words:
        return None, current_label
    built = _build_ass_text(variant, words)
    if built is None:
        return None, current_label
    ass_content, fonts_dir = built

    ass_path: Path | None = None
    try:
        fd, tmp = tempfile.mkstemp(suffix=".ass", prefix="render_final_", dir=output_dir)
        ass_path = Path(tmp)
        _ = os.close(fd)
        _ = ass_path.write_text(ass_content, encoding="utf-8")
    except (OSError, ValueError) as error:
        if ass_path is not None:
            with contextlib.suppress(OSError):
                ass_path.unlink()
        msg = f"Failed to write ASS subtitles: {error}"
        raise FinalRenderError(msg, error_code="ass_write_failed") from error

    subtitles_opts = f"filename='{ass_path.relative_to(output_dir).as_posix()}'"
    if fonts_dir is not None:
        subtitles_opts += f":fontsdir='{escape_filter_path(fonts_dir)}'"
    next_label = "[outv_sub]"
    filter_parts.append(f"{current_label}subtitles={subtitles_opts}{next_label}")
    return ass_path, next_label


def _build_position_filter_chain(
    pos_idx: int,
    pos: LayoutPosition,
    src_dimensions: tuple[int, int],
    branch_start: int,
    total_bands: int,
) -> tuple[list[str], str]:
    parts: list[str] = []
    dur_s = (pos.end_ms - pos.start_ms) / 1000.0
    start_s = pos.start_ms / 1000.0
    bands_count = len(pos.layout_spec.bands)
    source_dims = tuple(src_dimensions for _ in range(bands_count))

    band_labels: list[str] = []
    for band_idx in range(bands_count):
        in_label = f"[s{branch_start + band_idx}]" if total_bands > 1 else "[0:v]"
        t_label = f"[p{pos_idx}_t{band_idx}]"
        parts.append(
            f"{in_label}trim=start={start_s}:duration={dur_s},setpts=PTS-STARTPTS{t_label}"
        )
        band_labels.append(t_label)

    pos_band_labels: list[str] = [f"[p{pos_idx}_b{i}]" for i in range(bands_count)]
    raw_chains = build_layout_filter_chains(
        layout_spec=pos.layout_spec,
        canvas_width=_CANVAS_WIDTH,
        canvas_height=_CANVAS_HEIGHT,
        source_dimensions=source_dims,
        input_labels=band_labels,
    )
    for raw_chain in raw_chains:
        mapped_chain = raw_chain
        for b_idx in range(bands_count):
            suffix = f"[b{b_idx}]"
            if raw_chain.endswith(suffix):
                mapped_chain = raw_chain[: -len(suffix)] + f"[p{pos_idx}_b{b_idx}]"
                break
        parts.append(mapped_chain)

    pos_out = f"[pos_{pos_idx}_v]"
    if bands_count > 1:
        inputs = "".join(pos_band_labels)
        parts.append(f"{inputs}vstack=inputs={bands_count}{pos_out}")
    else:
        parts.append(f"{pos_band_labels[0]}copy{pos_out}")
    return parts, pos_out


def _build_timeline_video_chains(
    positions: Sequence[LayoutPosition],
    src_dimensions: tuple[int, int],
) -> tuple[list[str], str]:
    total_bands = sum(len(p.layout_spec.bands) for p in positions)
    filter_parts: list[str] = []

    if total_bands > 1:
        split_outputs = "".join(f"[s{i}]" for i in range(total_bands))
        filter_parts.append(f"[0:v]split={total_bands}{split_outputs}")

    branch_idx = 0
    pos_labels: list[str] = []

    for pos_idx, pos in enumerate(positions):
        pos_parts, pos_out = _build_position_filter_chain(
            pos_idx=pos_idx,
            pos=pos,
            src_dimensions=src_dimensions,
            branch_start=branch_idx,
            total_bands=total_bands,
        )
        filter_parts.extend(pos_parts)
        pos_labels.append(pos_out)
        branch_idx += len(pos.layout_spec.bands)

    stitched_label = "[stitched_v]"
    if len(positions) > 1:
        inputs = "".join(pos_labels)
        filter_parts.append(f"{inputs}concat=n={len(positions)}:v=1:a=0{stitched_label}")
    else:
        filter_parts.append(f"{pos_labels[0]}copy{stitched_label}")

    return filter_parts, stitched_label


def _build_watermark_chain(
    project_dir: Path,
    variant: Variant,
    current_label: str,
) -> tuple[list[str], list[str], str]:
    filter_parts: list[str] = []
    extra_inputs: list[str] = []
    wm = variant.watermark
    if not (wm.enabled and wm.asset_id is not None):
        return filter_parts, extra_inputs, current_label

    candidate = project_dir / "assets" / "watermark" / f"{wm.asset_id}.png"
    if not (
        candidate.is_file()
        and not candidate.is_symlink()
        and not candidate.is_junction()
        and wm.asset_sha256 is not None
        and sha256_file(candidate) == wm.asset_sha256
    ):
        return filter_parts, extra_inputs, current_label

    extra_inputs.extend(["-i", str(candidate.resolve())])
    x_expr, y_expr = _watermark_xy(wm.position)
    target_w = int(_CANVAS_WIDTH * 0.15 / 2) * 2
    next_label = "[outv_wm]"
    filter_parts.append(
        f"[1:v]format=rgba,scale={target_w}:-2[wm];{current_label}[wm]overlay={x_expr}:{y_expr}{next_label}"
    )
    return filter_parts, extra_inputs, next_label


def _build_render_plan(
    *,
    project_dir: Path,
    variant: Variant,
    src_dimensions: tuple[int, int],
    source_has_audio: bool,
    output_dir: Path,
) -> tuple[Path | None, str, list[str], str, bool]:
    filter_parts, current_label = _build_timeline_video_chains(
        variant.layout_timeline, src_dimensions
    )

    if source_has_audio:
        var_start_s = variant.start_ms / 1000.0
        var_dur_s = (variant.end_ms - variant.start_ms) / 1000.0
        filter_parts.append(
            f"[0:a]atrim=start={var_start_s}:duration={var_dur_s},asetpts=PTS-STARTPTS[outa]"
        )

    ass_path: Path | None = None
    if variant.captions_enabled:
        ass_path, current_label = _try_render_ass(
            project_dir=project_dir,
            variant=variant,
            output_dir=output_dir,
            filter_parts=filter_parts,
            current_label=current_label,
        )

    wm_parts, extra_inputs, current_label = _build_watermark_chain(
        project_dir, variant, current_label
    )
    filter_parts.extend(wm_parts)

    return ass_path, ";".join(filter_parts), extra_inputs, current_label, source_has_audio


def _execute_ffmpeg_with_fallback(  # ruff: ignore[too-many-arguments]
    *,
    ffmpeg: str,
    source: Path,
    extra_inputs: list[str],
    filter_complex: str,
    final_v_label: str,
    has_audio: bool,
    temp_destination: Path,
    output_dir: Path,
    runner: FinalRenderRunner,
    nvenc_available: bool,
) -> str:
    candidates = ["h264_nvenc", "libx264"] if nvenc_available else ["libx264"]
    last_error: Exception | None = None

    for encoder in candidates:
        use_nvenc = encoder == "h264_nvenc"
        argv = [
            ffmpeg,
            "-hide_banner",
            "-nostdin",
            "-v",
            "error",
            "-y",
            "-i",
            str(source.resolve()),
        ]
        argv.extend(extra_inputs)
        argv.extend(["-filter_complex", filter_complex])
        argv.extend(["-map", final_v_label])
        if has_audio:
            argv.extend(["-map", "[outa]"])
            argv.extend(list(audio_and_container_arguments()))
        else:
            argv.append("-an")
        argv.extend(list(video_encoder_arguments(nvenc_available=use_nvenc)))
        argv.append(str(temp_destination.resolve()))

        try:
            _ = runner.run(tuple(argv), timeout_s=_DEFAULT_TIMEOUT_S, cwd=output_dir)
        except (OSError, RuntimeError) as error:
            last_error = error
            with contextlib.suppress(OSError):
                if temp_destination.exists():
                    temp_destination.unlink()
            if use_nvenc and "libx264" in candidates:
                logger.warning(
                    "NVENC encoding failed (%s); falling back to CPU libx264.",
                    error,
                )
                continue
            break
        else:
            if not temp_destination.is_file():
                msg = f"FFmpeg exited without generating output file using {encoder}."
                raise FinalRenderError(msg, error_code="encoder_failed")
            return encoder

    msg = str(last_error) if last_error else "FFmpeg final render failed."
    raise FinalRenderError(msg, error_code="render_failed") from last_error


def _evaluate_gate(
    *,
    gate: Gate,
    variant: Variant,
    temp_destination: Path,
) -> None:
    piece = Piece(
        piece_id=str(variant.variant_id),
        platform=Platform.TIKTOK,
        caption=variant.name,
        artifact_path=temp_destination,
        start_sec=variant.start_ms / 1000.0,
        end_sec=variant.end_ms / 1000.0,
    )
    contract = vanilla_contract(campaign_id="studio", platform=Platform.TIKTOK)
    gate_result: GateResult = gate.evaluate_piece(piece, contract=contract)

    if not gate_result.passed:
        reasons = [
            f"{check.id} ({check.status.value}): {check.evidence.get('reason', check.evidence)}"
            for check in gate_result.checks
            if check.status.value in {"fail", "unsupported"}
        ]
        summary = "; ".join(reasons) or gate_result.status.value
        msg = f"Gate check failed: {summary}"
        raise FinalRenderError(msg, error_code="gate_rejected")


def _publish_artifact(temp_destination: Path, destination: Path) -> None:
    try:
        _ = temp_destination.replace(destination)
    except OSError as error:
        msg = f"Failed to publish final deliverable to {destination}: {error}"
        raise FinalRenderError(msg, error_code="publish_failed") from error
    if destination.is_symlink() or destination.is_junction():
        with contextlib.suppress(OSError):
            destination.unlink()
        msg = f"Final destination {destination} is a symlink or junction."
        raise FinalRenderError(msg, error_code="invalid_destination")
