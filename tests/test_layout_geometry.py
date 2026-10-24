"""Shared FFmpeg geometry tests for final render and Studio preview."""

import pytest

from kliptych.contract import Band, Layout, LayoutSpec, NormalizedRect
from kliptych.layout_geometry import build_layout_filter_chains


def test_horizontal_layout_preserves_even_16_by_9_frame_in_vertical_canvas() -> None:
    layout = LayoutSpec(
        layout=Layout.HORIZONTAL,
        bands=(Band(source="source.mp4", source_rect=NormalizedRect()),),
    )

    chains = build_layout_filter_chains(
        layout_spec=layout,
        canvas_width=720,
        canvas_height=1280,
        source_dimensions=((1920, 1080),),
    )

    assert chains == (
        (
            "[0:v]crop=1920:1080:0:0,"
            "scale=720:404:force_original_aspect_ratio=decrease,"
            "pad=720:1280:0:438:color=black,setsar=1[b0]"
        ),
    )


def test_vertical_layout_keeps_existing_fill_and_crop_geometry() -> None:
    layout = LayoutSpec(
        layout=Layout.VERTICAL,
        bands=(
            Band(
                source="source.mp4",
                source_rect=NormalizedRect(x=0.25, y=0.0, width=0.5, height=1.0),
            ),
        ),
    )

    chains = build_layout_filter_chains(
        layout_spec=layout,
        canvas_width=720,
        canvas_height=1280,
        source_dimensions=((1920, 1080),),
    )

    assert chains == (
        (
            "[0:v]crop=960:1080:480:0,"
            "scale=720:1280:force_original_aspect_ratio=increase,"
            "crop=720:1280,setsar=1[b0]"
        ),
    )


def test_multi_band_layout_uses_even_band_heights_and_black_gaps() -> None:
    layout = LayoutSpec(
        layout=Layout.DIVIDIDO,
        bands=(
            Band(source="source.mp4", source_rect=NormalizedRect()),
            Band(source="source.mp4", source_rect=NormalizedRect()),
        ),
        gap=20,
    )
    chains = build_layout_filter_chains(
        layout_spec=layout,
        canvas_width=720,
        canvas_height=1280,
        source_dimensions=((1920, 1080), (1920, 1080)),
    )

    assert chains == (
        (
            "[0:v]crop=1920:1080:0:0,"
            "scale=720:620:force_original_aspect_ratio=increase,"
            "crop=720:620,setsar=1,pad=720:640:0:0:color=black[b0]"
        ),
        (
            "[1:v]crop=1920:1080:0:0,"
            "scale=720:620:force_original_aspect_ratio=increase,"
            "crop=720:620,setsar=1,pad=720:640:0:0:color=black[b1]"
        ),
    )


def test_geometry_accepts_explicit_input_labels_for_one_source_split_frame() -> None:
    layout = LayoutSpec(
        layout=Layout.DIVIDIDO,
        bands=(
            Band(source="source.mp4", source_rect=NormalizedRect()),
            Band(source="source.mp4", source_rect=NormalizedRect()),
        ),
    )

    chains = build_layout_filter_chains(
        layout_spec=layout,
        canvas_width=720,
        canvas_height=1280,
        source_dimensions=((1920, 1080), (1920, 1080)),
        input_labels=("[frame0]", "[frame1]"),
    )

    assert chains[0].startswith("[frame0]crop=")
    assert chains[1].startswith("[frame1]crop=")


@pytest.mark.parametrize("dimensions", [((1, 1080),), ((1920, 1),)])
def test_geometry_rejects_source_dimensions_below_two_pixels(
    dimensions: tuple[tuple[int, int], ...],
) -> None:
    layout = LayoutSpec(
        layout=Layout.VERTICAL,
        bands=(Band(source="source.mp4", source_rect=NormalizedRect()),),
    )

    with pytest.raises(ValueError, match="at least two"):
        _ = build_layout_filter_chains(
            layout_spec=layout,
            canvas_width=720,
            canvas_height=1280,
            source_dimensions=dimensions,
        )
