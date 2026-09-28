"""Sprint 2 (Objetivo 3): el fingerprint invalida `--resume` ante cambios de watermark.

Cambiar la posición del watermark o los bytes del PNG debe mover el digest
de entradas de `long_video`/`slideshow`; entradas idénticas deben producir
el mismo digest.
"""

from pathlib import Path

from kliptych.encoding import RenderConfig
from kliptych.orchestrator import (
    PipelineConfig,
    compute_long_video_fingerprint,
    compute_slideshow_fingerprint,
)
from tests.support import make_contract

_URL = "https://example.com/video.mp4"


def _config(
    tmp_path: Path,
    name: str,
    *,
    position: str = "center_bottom",
    pixels: bytes = b"png-a",
) -> PipelineConfig:
    watermark = tmp_path / name
    _ = watermark.write_bytes(pixels)
    return PipelineConfig(
        output_dir=tmp_path / "out",
        contract=make_contract(
            watermark_required=True,
            watermark_visible_full_video=True,
            watermark_position=position,
        ),
        render=RenderConfig(),
        watermark_path=watermark,
    )


def test_same_watermark_inputs_keep_fingerprint(tmp_path: Path) -> None:
    first = _config(tmp_path, "wm.png")
    second = _config(tmp_path, "wm.png")
    assert compute_long_video_fingerprint(_URL, config=first) == (
        compute_long_video_fingerprint(_URL, config=second)
    )


def test_watermark_position_change_invalidates_fingerprint(tmp_path: Path) -> None:
    before = _config(tmp_path, "wm-before.png", position="center_bottom")
    after = _config(tmp_path, "wm-after.png", position="top_right")
    assert compute_long_video_fingerprint(_URL, config=before) != (
        compute_long_video_fingerprint(_URL, config=after)
    )


def test_watermark_bytes_change_invalidates_fingerprint(tmp_path: Path) -> None:
    before = _config(tmp_path, "wm-a.png", pixels=b"png-a")
    after = _config(tmp_path, "wm-b.png", pixels=b"png-b")
    assert compute_long_video_fingerprint(_URL, config=before) != (
        compute_long_video_fingerprint(_URL, config=after)
    )


def test_watermark_bytes_change_invalidates_slideshow_fingerprint(tmp_path: Path) -> None:
    image = tmp_path / "slide.jpg"
    _ = image.write_bytes(b"slide")
    before = _config(tmp_path, "wm-a.png", pixels=b"png-a")
    after = _config(tmp_path, "wm-b.png", pixels=b"png-b")
    assert compute_slideshow_fingerprint([image], config=before, slide_duration_s=3.0) != (
        compute_slideshow_fingerprint([image], config=after, slide_duration_s=3.0)
    )
