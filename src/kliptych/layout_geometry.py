"""Shared deterministic FFmpeg layout geometry for render and Studio preview."""

from collections.abc import Sequence
from dataclasses import dataclass

from kliptych.contract import Band, Layout, LayoutSpec

_EVEN_DIVISOR = 2
_HORIZONTAL_NUMERATOR = 9
_HORIZONTAL_DENOMINATOR = 16


@dataclass(frozen=True, slots=True)
class BandGeometry:
    """Measured input and target geometry for one layout band."""

    index: int
    band: Band
    band_height: int
    source_dimensions: tuple[int, int]
    gap: int


def build_layout_filter_chains(
    *,
    layout_spec: LayoutSpec,
    canvas_width: int,
    canvas_height: int,
    source_dimensions: Sequence[tuple[int, int]],
) -> tuple[str, ...]:
    """Build per-band FFmpeg chains from declared geometry and measured sources.

    Args:
        layout_spec: Explicit composition and normalized crop rectangles.
        canvas_width: Even output canvas width.
        canvas_height: Even output canvas height.
        source_dimensions: One measured width/height pair per declared band.

    Returns:
        Filter chains labeled `[b0]`, `[b1]`, and so on.

    Raises:
        ValueError: If dimensions are not positive/even or do not correspond to
            the explicit layout bands.
    """
    _require_even_positive(canvas_width, name="canvas_width")
    _require_even_positive(canvas_height, name="canvas_height")
    bands = layout_spec.bands
    if len(bands) != len(source_dimensions):
        msg = "source_dimensions must have one entry per explicit layout band"
        raise ValueError(msg)
    if not bands:
        msg = "shared layout geometry requires explicit bands"
        raise ValueError(msg)
    if layout_spec.layout is Layout.HORIZONTAL:
        return (_horizontal_filter(bands[0], source_dimensions[0], canvas_width, canvas_height),)
    band_heights = layout_spec.band_heights(canvas_height)
    gap = layout_spec.gap if len(bands) > 1 else 0
    return tuple(
        _band_filter(
            BandGeometry(
                index=index,
                band=band,
                band_height=band_height,
                source_dimensions=dimensions,
                gap=gap,
            ),
            canvas_width=canvas_width,
        )
        for index, (band, band_height, dimensions) in enumerate(
            zip(bands, band_heights, source_dimensions, strict=True)
        )
    )


def _band_filter(geometry: BandGeometry, *, canvas_width: int) -> str:
    source_w, source_h = geometry.source_dimensions
    _require_source_dimension(source_w, name="source_width")
    _require_source_dimension(source_h, name="source_height")
    crop_w, crop_h, crop_x, crop_y = _crop_box(geometry.band, source_w, source_h)
    content_h = geometry.band_height - geometry.gap
    if content_h <= 0:
        msg = "band content height must be positive after gap"
        raise ValueError(msg)
    chain = (
        f"[{geometry.index}:v]crop={crop_w}:{crop_h}:{crop_x}:{crop_y},"
        f"scale={canvas_width}:{content_h}:force_original_aspect_ratio=increase,"
        f"crop={canvas_width}:{content_h},setsar=1"
    )
    if geometry.gap > 0:
        chain += f",pad={canvas_width}:{geometry.band_height}:0:0:color=black"
    return f"{chain}[b{geometry.index}]"


def _horizontal_filter(
    band: Band,
    source_dimensions: tuple[int, int],
    canvas_width: int,
    canvas_height: int,
) -> str:
    source_w, source_h = source_dimensions
    _require_source_dimension(source_w, name="source_width")
    _require_source_dimension(source_h, name="source_height")
    crop_w, crop_h, crop_x, crop_y = _crop_box(band, source_w, source_h)
    content_h = _even_floor(canvas_width * _HORIZONTAL_NUMERATOR // _HORIZONTAL_DENOMINATOR)
    top_padding = (canvas_height - content_h) // _EVEN_DIVISOR
    bottom_padding = canvas_height - content_h - top_padding
    if top_padding != bottom_padding:
        msg = "horizontal padding must be symmetric"
        raise ValueError(msg)
    return (
        f"[0:v]crop={crop_w}:{crop_h}:{crop_x}:{crop_y},"
        f"scale={canvas_width}:{content_h}:force_original_aspect_ratio=decrease,"
        f"pad={canvas_width}:{canvas_height}:0:{top_padding}:color=black,setsar=1[b0]"
    )


def _crop_box(band: Band, source_w: int, source_h: int) -> tuple[int, int, int, int]:
    rect = band.source_rect
    crop_w = _even_round(rect.width * source_w)
    crop_h = _even_round(rect.height * source_h)
    crop_w = max(_EVEN_DIVISOR, min(crop_w, source_w))
    crop_h = max(_EVEN_DIVISOR, min(crop_h, source_h))
    crop_x = max(0, min(_even_round(rect.x * source_w), source_w - crop_w))
    crop_y = max(0, min(_even_round(rect.y * source_h), source_h - crop_h))
    return crop_w, crop_h, crop_x, crop_y


def _even_round(value: float) -> int:
    return round(value / _EVEN_DIVISOR) * _EVEN_DIVISOR


def _even_floor(value: int) -> int:
    return value // _EVEN_DIVISOR * _EVEN_DIVISOR


def _require_even_positive(value: int, *, name: str) -> None:
    if value <= 0 or value % _EVEN_DIVISOR != 0:
        msg = f"{name} must be positive and even: {value}"
        raise ValueError(msg)


def _require_source_dimension(value: int, *, name: str) -> None:
    if value < _EVEN_DIVISOR:
        msg = f"{name} must be at least two pixels: {value}"
        raise ValueError(msg)
