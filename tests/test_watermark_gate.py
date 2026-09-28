"""Sprint 2 (Objetivo 4): validador OpenCV del watermark en el gate.

`watermark.present` exige el logo en alguna muestra; `watermark.full_video`
lo exige en todas las muestras con densidad temporal (<= 0.25 s por
muestra). Cada muestra verifica similitud invariante a la opacidad
(`cv2.matchTemplate` para ubicar + NCC enmascarada o contraste de borde
para puntuar), zona (`WatermarkPosition` con margen) y tamaño mínimo
(`min_width_ratio`). Fail-closed: sin PNG, sin video o con cualquier
fallo, el resultado es FAIL.
"""

import shutil
import subprocess
from collections.abc import Sequence
from pathlib import Path

import cv2
import numpy as np
import pytest

from kliptych.assembler import FFmpegAssembler
from kliptych.assets import AssetRegistry
from kliptych.contract import Contract, Platform, Watermark, WatermarkPosition
from kliptych.gate import (
    CheckStatus,
    Gate,
    GateStatus,
    check_watermark_full_video,
    check_watermark_present,
)
from kliptych.gate.checks import GateContext
from kliptych.gate.watermark import Sample, expected_top_left, sample_within_zone
from tests.support import FakeProbe, make_contract, make_media, make_piece

_FFMPEG = shutil.which("ffmpeg")
_FFPROBE = shutil.which("ffprobe")
_NEEDS_TOOLS = _FFMPEG is None or _FFPROBE is None

_MARGIN = 20
_WIDTH = 1080
_HEIGHT = 1920

_POSITIONS: dict[WatermarkPosition, tuple[float, float]] = {
    WatermarkPosition.TOP_LEFT: (_MARGIN, _MARGIN),
    WatermarkPosition.TOP_RIGHT: (_WIDTH - 216 - _MARGIN, _MARGIN),
    WatermarkPosition.BOTTOM_LEFT: (_MARGIN, _HEIGHT - 144 - _MARGIN),
    WatermarkPosition.BOTTOM_RIGHT: (_WIDTH - 216 - _MARGIN, _HEIGHT - 144 - _MARGIN),
    WatermarkPosition.CENTER: ((_WIDTH - 216) / 2, (_HEIGHT - 144) / 2),
    WatermarkPosition.CENTER_TOP: ((_WIDTH - 216) / 2, _MARGIN),
    WatermarkPosition.CENTER_BOTTOM: ((_WIDTH - 216) / 2, _HEIGHT - 144 - _MARGIN),
}


def _sample_at(x: int, y: int) -> Sample:
    return Sample(
        t_s=1.0,
        matched=True,
        correlation=0.99,
        x=x,
        y=y,
        expected_x=432.0,
        expected_y=1684.0,
    )


@pytest.mark.parametrize(("position", "expected"), list(_POSITIONS.items()))
def test_expected_top_left_covers_all_positions(
    position: WatermarkPosition, expected: tuple[float, float]
) -> None:
    assert expected_top_left(_WIDTH, _HEIGHT, 216, 144, position) == (
        pytest.approx(expected[0]),
        pytest.approx(expected[1]),
    )


def test_within_zone_accepts_exact_match() -> None:
    assert sample_within_zone(_sample_at(432, 1684), _WIDTH, _HEIGHT) is True


def test_within_zone_rejects_corner_when_center_bottom_required() -> None:
    assert sample_within_zone(_sample_at(844, 20), _WIDTH, _HEIGHT) is False


def test_within_zone_rejects_far_miss() -> None:
    assert sample_within_zone(_sample_at(0, 0), _WIDTH, _HEIGHT) is False


def _watermark_png(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas: np.ndarray = np.zeros((80, 120, 4), dtype=np.uint8)
    canvas[:, :, 0:3] = 255
    canvas[:, :, 3] = 255
    _ = cv2.rectangle(canvas, (0, 0), (119, 79), (0, 0, 0, 255), thickness=4)
    _ = cv2.line(canvas, (8, 8), (112, 72), (0, 0, 255, 255), thickness=6)
    _ = cv2.circle(canvas, (60, 40), 16, (255, 0, 0, 255), thickness=-1)
    canvas[60:80, 100:120, 3] = 0
    _ = cv2.imwrite(str(path), canvas)
    return path


def _white_watermark_png(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas: np.ndarray = np.zeros((80, 120, 4), dtype=np.uint8)
    canvas[:, :, 0:3] = 255
    canvas[:, :, 3] = 255
    _ = cv2.imwrite(str(path), canvas)
    return path


def _registry_with_png(tmp_path: Path, *, white: bool = False) -> AssetRegistry:
    registry = AssetRegistry(tmp_path)
    if white:
        _ = _white_watermark_png(tmp_path / "assets" / "wm.png")
    else:
        _ = _watermark_png(tmp_path / "assets" / "wm.png")
    _ = registry.register(asset_id="wm-marca", kind="image", uri="assets/wm.png", origin="brief")
    return registry


def _contract(
    *, full_video: bool, position: str = "center_bottom", opacity: float = 1.0
) -> Contract:
    rule = "watermark.full_video" if full_video else "watermark.present"
    return make_contract(
        hard=["artifact.integrity", rule],
        audio_rule="any",
        min_s=None,
        max_s=None,
        required_mentions=(),
        required_hashtags=(),
        watermark_required=True,
        watermark_visible_full_video=full_video,
        watermark_position=position,
        watermark_opacity=opacity,
    )


def _context(
    tmp_path: Path,
    video: Path,
    *,
    full_video: bool,
    position: str = "center_bottom",
    opacity: float = 1.0,
    duration_s: float = 2.0,
    white_template: bool = False,
) -> GateContext:
    contract = _contract(full_video=full_video, position=position, opacity=opacity)
    return GateContext(
        contract=contract,
        rules=contract.platforms[Platform.TIKTOK],
        piece=make_piece(video),
        artifact_sha256="a" * 64,
        media=make_media(duration_s=duration_s),
        assets=_registry_with_png(tmp_path, white=white_template),
    )


def _generate_clip(path: Path, *, duration: float = 2.0) -> Path:
    assert _FFMPEG is not None
    argv = [
        _FFMPEG,
        "-y",
        "-v",
        "error",
        "-f",
        "lavfi",
        "-i",
        f"color=c=blue:s=320x240:d={duration}",
        "-f",
        "lavfi",
        "-i",
        f"sine=frequency=440:duration={duration}",
        "-shortest",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        str(path),
    ]
    _ = subprocess.run(argv, capture_output=True, check=True, timeout=120)
    return path


def _render(
    tmp_path: Path,
    name: str,
    *,
    position: WatermarkPosition | None,
    duration: float = 2.0,
    opacity: float = 1.0,
    white_template: bool = False,
) -> Path:
    assert _FFMPEG is not None
    clip = _generate_clip(tmp_path / f"{name}-clip.mp4", duration=duration)
    destination = tmp_path / f"{name}.mp4"
    watermark = _registry_with_png(tmp_path, white=white_template).path_for("wm-marca")
    if position is None:
        _ = FFmpegAssembler(ffmpeg=_FFMPEG).assemble(clip=clip, destination=destination)
        return destination
    config = Watermark(
        required=True,
        asset_id="wm-marca",
        visible_full_video=True,
        position=position,
        opacity=opacity,
    )
    _ = FFmpegAssembler(ffmpeg=_FFMPEG).assemble(
        clip=clip, destination=destination, watermark=watermark, watermark_config=config
    )
    return destination


def _generate_textured_clip(path: Path, *, lavfi: str, duration: float = 2.0) -> Path:
    assert _FFMPEG is not None
    argv = [
        _FFMPEG,
        "-y",
        "-nostdin",
        "-v",
        "error",
        "-f",
        "lavfi",
        "-i",
        lavfi,
        "-t",
        f"{duration}",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        str(path),
    ]
    _ = subprocess.run(argv, capture_output=True, check=True, timeout=120)
    return path


def _render_plain_from_source(
    tmp_path: Path, name: str, *, lavfi: str, duration: float = 2.0
) -> Path:
    assert _FFMPEG is not None
    clip = _generate_textured_clip(tmp_path / f"{name}-clip.mp4", lavfi=lavfi, duration=duration)
    destination = tmp_path / f"{name}.mp4"
    _ = FFmpegAssembler(ffmpeg=_FFMPEG).assemble(clip=clip, destination=destination)
    return destination


def _concat(first: Path, second: Path, destination: Path) -> Path:
    return _concat_all((first, second), destination)


def _concat_all(parts: Sequence[Path], destination: Path) -> Path:
    assert _FFMPEG is not None
    listing = destination.with_suffix(".txt")
    _ = listing.write_text("".join(f"file '{part.as_posix()}'\n" for part in parts))
    argv = [
        _FFMPEG,
        "-y",
        "-v",
        "error",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(listing),
        "-c",
        "copy",
        str(destination),
    ]
    _ = subprocess.run(argv, capture_output=True, check=True, timeout=120)
    return destination


def test_missing_png_is_fail_closed(tmp_path: Path) -> None:
    video = tmp_path / "video.mp4"
    _ = video.write_bytes(b"video")
    contract = _contract(full_video=True)
    context = GateContext(
        contract=contract,
        rules=contract.platforms[Platform.TIKTOK],
        piece=make_piece(video),
        artifact_sha256="a" * 64,
        media=make_media(duration_s=2.0),
        assets=AssetRegistry(tmp_path),
    )
    assert check_watermark_full_video(context).status is CheckStatus.FAIL
    assert check_watermark_present(context).status is CheckStatus.FAIL


def test_missing_video_is_fail_closed(tmp_path: Path) -> None:
    context = _context(tmp_path, tmp_path / "ausente.mp4", full_video=True)
    assert check_watermark_full_video(context).status is CheckStatus.FAIL


def test_unreadable_video_is_fail_closed(tmp_path: Path) -> None:
    video = tmp_path / "video.mp4"
    _ = video.write_bytes(b"video")
    context = _context(tmp_path, video, full_video=True)
    assert check_watermark_full_video(context).status is CheckStatus.FAIL


@pytest.mark.integration
@pytest.mark.skipif(_NEEDS_TOOLS, reason="ffmpeg/ffprobe no disponibles")
def test_rendered_watermark_passes_present_and_full_video(tmp_path: Path) -> None:
    video = _render(tmp_path, "marked", position=WatermarkPosition.CENTER_BOTTOM)
    assert (
        check_watermark_present(_context(tmp_path, video, full_video=False)).status
        is CheckStatus.PASS
    )
    assert (
        check_watermark_full_video(_context(tmp_path, video, full_video=True)).status
        is CheckStatus.PASS
    )


@pytest.mark.integration
@pytest.mark.skipif(_NEEDS_TOOLS, reason="ffmpeg/ffprobe no disponibles")
def test_gate_passes_rendered_watermark(tmp_path: Path) -> None:
    video = _render(tmp_path, "marked", position=WatermarkPosition.CENTER_BOTTOM)
    contract = _contract(full_video=True)
    result = Gate(FakeProbe(info=make_media(duration_s=2.0))).run(
        contract=contract,
        piece=make_piece(video),
        assets=_registry_with_png(tmp_path),
    )
    assert result.status is GateStatus.PASSED


@pytest.mark.integration
@pytest.mark.skipif(_NEEDS_TOOLS, reason="ffmpeg/ffprobe no disponibles")
def test_wrong_position_fails(tmp_path: Path) -> None:
    video = _render(tmp_path, "corner", position=WatermarkPosition.TOP_RIGHT)
    outcome = check_watermark_full_video(_context(tmp_path, video, full_video=True))
    assert outcome.status is CheckStatus.FAIL
    assert (
        check_watermark_present(
            _context(tmp_path, video, full_video=False, position="center_bottom")
        ).status
        is CheckStatus.FAIL
    )


@pytest.mark.integration
@pytest.mark.skipif(_NEEDS_TOOLS, reason="ffmpeg/ffprobe no disponibles")
def test_video_without_watermark_fails(tmp_path: Path) -> None:
    video = _render(tmp_path, "plain", position=None)
    assert (
        check_watermark_full_video(_context(tmp_path, video, full_video=True)).status
        is CheckStatus.FAIL
    )
    assert (
        check_watermark_present(_context(tmp_path, video, full_video=False)).status
        is CheckStatus.FAIL
    )


@pytest.mark.integration
@pytest.mark.skipif(_NEEDS_TOOLS, reason="ffmpeg/ffprobe no disponibles")
def test_partial_watermark_fails_full_video_but_passes_present(tmp_path: Path) -> None:
    first = _render(tmp_path, "head", position=WatermarkPosition.CENTER_BOTTOM, duration=1.0)
    second = _render(tmp_path, "tail", position=None, duration=1.0)
    video = _concat(first, second, tmp_path / "partial.mp4")
    assert (
        check_watermark_full_video(_context(tmp_path, video, full_video=True)).status
        is CheckStatus.FAIL
    )
    assert (
        check_watermark_present(_context(tmp_path, video, full_video=False)).status
        is CheckStatus.PASS
    )


@pytest.mark.integration
@pytest.mark.skipif(_NEEDS_TOOLS, reason="ffmpeg/ffprobe no disponibles")
def test_narrow_gap_fails_full_video_but_passes_present(tmp_path: Path) -> None:
    head = _render(tmp_path, "gap-head", position=WatermarkPosition.CENTER_BOTTOM, duration=1.05)
    mid = _render(tmp_path, "gap-mid", position=None, duration=0.30)
    tail = _render(tmp_path, "gap-tail", position=WatermarkPosition.CENTER_BOTTOM, duration=1.65)
    video = _concat_all((head, mid, tail), tmp_path / "gap.mp4")
    assert (
        check_watermark_full_video(
            _context(tmp_path, video, full_video=True, duration_s=3.0)
        ).status
        is CheckStatus.FAIL
    )
    assert (
        check_watermark_present(_context(tmp_path, video, full_video=False, duration_s=3.0)).status
        is CheckStatus.PASS
    )


def test_unrequired_watermark_passes_without_png(tmp_path: Path) -> None:
    video = tmp_path / "video.mp4"
    _ = video.write_bytes(b"video")
    contract = make_contract(audio_rule="any", min_s=None)
    context = GateContext(
        contract=contract,
        rules=contract.platforms[next(iter(contract.platforms))],
        piece=make_piece(video),
        artifact_sha256="a" * 64,
        media=make_media(duration_s=2.0),
        assets=AssetRegistry(tmp_path),
    )
    assert check_watermark_present(context).status is CheckStatus.PASS
    assert check_watermark_full_video(context).status is CheckStatus.PASS


@pytest.mark.integration
@pytest.mark.skipif(_NEEDS_TOOLS, reason="ffmpeg/ffprobe no disponibles")
@pytest.mark.parametrize("opacity", [0.3, 0.5])
def test_semitransparent_watermark_passes_present_and_full_video(
    tmp_path: Path, opacity: float
) -> None:
    video = _render(
        tmp_path,
        f"translucent-{opacity}",
        position=WatermarkPosition.CENTER_BOTTOM,
        opacity=opacity,
    )
    assert (
        check_watermark_present(_context(tmp_path, video, full_video=False, opacity=opacity)).status
        is CheckStatus.PASS
    )
    assert (
        check_watermark_full_video(
            _context(tmp_path, video, full_video=True, opacity=opacity)
        ).status
        is CheckStatus.PASS
    )


@pytest.mark.integration
@pytest.mark.skipif(_NEEDS_TOOLS, reason="ffmpeg/ffprobe no disponibles")
@pytest.mark.parametrize("opacity", [0.3, 0.5])
def test_plain_background_fails_with_configured_opacity(tmp_path: Path, opacity: float) -> None:
    video = _render(tmp_path, f"plain-{opacity}", position=None)
    assert (
        check_watermark_present(_context(tmp_path, video, full_video=False, opacity=opacity)).status
        is CheckStatus.FAIL
    )
    assert (
        check_watermark_full_video(
            _context(tmp_path, video, full_video=True, opacity=opacity)
        ).status
        is CheckStatus.FAIL
    )


@pytest.mark.integration
@pytest.mark.skipif(_NEEDS_TOOLS, reason="ffmpeg/ffprobe no disponibles")
@pytest.mark.parametrize("lavfi", ["testsrc2=s=320x240:r=30", "rgbtestsrc=size=320x240:rate=30"])
@pytest.mark.parametrize("opacity", [0.3, 0.5])
def test_textured_background_without_watermark_fails(
    tmp_path: Path, lavfi: str, opacity: float
) -> None:
    name = "testsrc2" if lavfi.startswith("testsrc2") else "rgbtestsrc"
    video = _render_plain_from_source(tmp_path, f"{name}-plain-{opacity}", lavfi=lavfi)
    assert (
        check_watermark_present(_context(tmp_path, video, full_video=False, opacity=opacity)).status
        is CheckStatus.FAIL
    )
    assert (
        check_watermark_full_video(
            _context(tmp_path, video, full_video=True, opacity=opacity)
        ).status
        is CheckStatus.FAIL
    )


@pytest.mark.integration
@pytest.mark.skipif(_NEEDS_TOOLS, reason="ffmpeg/ffprobe no disponibles")
@pytest.mark.parametrize("opacity", [0.3, 0.5])
def test_white_template_on_white_background_fails(tmp_path: Path, opacity: float) -> None:
    video = _render_plain_from_source(
        tmp_path, f"white-plain-{opacity}", lavfi="color=c=white:s=540x960", duration=2.0
    )
    assert (
        check_watermark_present(
            _context(
                tmp_path,
                video,
                full_video=False,
                position="top_left",
                opacity=opacity,
                white_template=True,
            )
        ).status
        is CheckStatus.FAIL
    )
    assert (
        check_watermark_full_video(
            _context(
                tmp_path,
                video,
                full_video=True,
                position="top_left",
                opacity=opacity,
                white_template=True,
            )
        ).status
        is CheckStatus.FAIL
    )


@pytest.mark.integration
@pytest.mark.skipif(_NEEDS_TOOLS, reason="ffmpeg/ffprobe no disponibles")
def test_uniform_white_watermark_on_blue_passes(tmp_path: Path) -> None:
    video = _render(
        tmp_path,
        "white-on-blue",
        position=WatermarkPosition.CENTER_BOTTOM,
        opacity=0.5,
        white_template=True,
    )
    assert (
        check_watermark_present(
            _context(tmp_path, video, full_video=False, opacity=0.5, white_template=True)
        ).status
        is CheckStatus.PASS
    )
    assert (
        check_watermark_full_video(
            _context(tmp_path, video, full_video=True, opacity=0.5, white_template=True)
        ).status
        is CheckStatus.PASS
    )
