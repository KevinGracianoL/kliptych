"""Sprint 2 (Objetivo 3): render dinámico del watermark en el ensamblado.

El watermark deja de estar clavado en la esquina superior derecha: el
ensamblado lo escala con `scale2ref` y lo superpone con `overlay` en la zona
del `WatermarkConfig`, pasando el PNG como entrada explícita (`-i`, nunca
`movie=`). Debe convivir con el mute (`-af volume=0`) y funcionar con
`libx264` (CI) y `h264_nvenc` (GPU).
"""

import shutil
import subprocess
from pathlib import Path

import pytest

from kliptych.assembler import FFmpegAssembler, RenderSpec
from kliptych.contract import Watermark, WatermarkPosition
from kliptych.gate.probe import FFprobeProbe

_FFMPEG = shutil.which("ffmpeg")
_FFPROBE = shutil.which("ffprobe")

_MARGIN = 20
_BASE = (
    "scale=trunc(iw*sar/2)*2:ih,setsar=1,"
    "scale=1080:1920:force_original_aspect_ratio=decrease,"
    "pad=1080:1920:(ow-iw)/2:(oh-ih)/2:color=black,setsar=1"
)

_POSITIONS: dict[WatermarkPosition, str] = {
    WatermarkPosition.TOP_LEFT: f"{_MARGIN}:{_MARGIN}",
    WatermarkPosition.TOP_RIGHT: f"W-w-{_MARGIN}:{_MARGIN}",
    WatermarkPosition.BOTTOM_LEFT: f"{_MARGIN}:H-h-{_MARGIN}",
    WatermarkPosition.BOTTOM_RIGHT: f"W-w-{_MARGIN}:H-h-{_MARGIN}",
    WatermarkPosition.CENTER: "(W-w)/2:(H-h)/2",
    WatermarkPosition.CENTER_TOP: f"(W-w)/2:{_MARGIN}",
    WatermarkPosition.CENTER_BOTTOM: f"(W-w)/2:H-h-{_MARGIN}",
}


def _config(position: WatermarkPosition, **overrides: object) -> Watermark:
    data: dict[str, object] = {
        "required": True,
        "asset_id": "wm-marca",
        "visible_full_video": True,
        "position": position,
    }
    data.update(overrides)
    return Watermark.model_validate(data)


def _file(tmp_path: Path, name: str) -> Path:
    path = tmp_path / name
    _ = path.write_bytes(b"contenido")
    return path


def _graph(tmp_path: Path, config: Watermark) -> str:
    clip = _file(tmp_path, "clip.mp4")
    watermark = _file(tmp_path, "wm.png")
    destination = tmp_path / "piece.mp4"
    recipe = FFmpegAssembler(ffmpeg="ffmpeg").render_arguments(
        RenderSpec(clip=clip, destination=destination, watermark=watermark, watermark_config=config)
    )
    return recipe[recipe.index("-filter_complex") + 1]


@pytest.mark.parametrize(("position", "overlay"), list(_POSITIONS.items()))
def test_overlay_matches_configured_position(
    tmp_path: Path, position: WatermarkPosition, overlay: str
) -> None:
    graph = _graph(tmp_path, _config(position))
    assert f"overlay={overlay}[v]" in graph


def test_watermark_scales_to_configured_ratio(tmp_path: Path) -> None:
    graph = _graph(tmp_path, _config(WatermarkPosition.CENTER_BOTTOM, scale_ratio=0.25))
    assert "[1:v]format=rgba,scale=270:-2[wm]" in graph
    assert "scale2ref" not in graph


def test_opacity_filter_only_when_partial(tmp_path: Path) -> None:
    opaque = _graph(tmp_path, _config(WatermarkPosition.CENTER))
    assert "colorchannelmixer" not in opaque
    translucent = _graph(tmp_path, _config(WatermarkPosition.CENTER, opacity=0.5))
    assert "[wm]colorchannelmixer=aa=0.5[wmf]" in translucent
    assert "[base][wmf]overlay=" in translucent


def test_watermark_uses_explicit_inputs_never_movie(tmp_path: Path) -> None:
    clip = _file(tmp_path, "clip.mp4")
    watermark = _file(tmp_path, "wm.png")
    destination = tmp_path / "piece.mp4"
    recipe = FFmpegAssembler(ffmpeg="ffmpeg").render_arguments(
        RenderSpec(
            clip=clip,
            destination=destination,
            watermark=watermark,
            watermark_config=_config(WatermarkPosition.BOTTOM_RIGHT),
        )
    )
    inputs = [recipe[index + 1] for index, arg in enumerate(recipe) if arg == "-i"]
    assert inputs == [str(clip), str(watermark)]
    graph = recipe[recipe.index("-filter_complex") + 1]
    assert "movie=" not in graph


def test_default_config_matches_legacy_top_right(tmp_path: Path) -> None:
    clip = _file(tmp_path, "clip.mp4")
    watermark = _file(tmp_path, "wm.png")
    destination = tmp_path / "piece.mp4"
    recipe = FFmpegAssembler(ffmpeg="ffmpeg").render_arguments(
        RenderSpec(clip=clip, destination=destination, watermark=watermark)
    )
    graph = recipe[recipe.index("-filter_complex") + 1]
    assert f"overlay=W-w-{_MARGIN}:{_MARGIN}[v]" in graph


def test_watermark_coexists_with_mute(tmp_path: Path) -> None:
    clip = _file(tmp_path, "clip.mp4")
    watermark = _file(tmp_path, "wm.png")
    destination = tmp_path / "piece.mp4"
    recipe = FFmpegAssembler(ffmpeg="ffmpeg").render_arguments(
        RenderSpec(
            clip=clip,
            destination=destination,
            watermark=watermark,
            watermark_config=_config(WatermarkPosition.CENTER_BOTTOM),
            mute_audio=True,
        )
    )
    assert "-filter_complex" in recipe
    assert "-af" in recipe
    assert "volume=0" in recipe
    assert "0:a?" in recipe


def _generate_clip(path: Path) -> Path:
    assert _FFMPEG is not None
    argv = [
        _FFMPEG,
        "-y",
        "-v",
        "error",
        "-f",
        "lavfi",
        "-i",
        "color=c=blue:s=320x240:d=1.0",
        "-f",
        "lavfi",
        "-i",
        "sine=frequency=440:duration=1.0",
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


def _generate_watermark(path: Path) -> Path:
    assert _FFMPEG is not None
    argv = [
        _FFMPEG,
        "-y",
        "-v",
        "error",
        "-f",
        "lavfi",
        "-i",
        "color=c=white:s=64x64",
        "-frames:v",
        "1",
        str(path),
    ]
    _ = subprocess.run(argv, capture_output=True, check=True, timeout=120)
    return path


def _gray_frame(path: Path, *, width: int, height: int) -> bytes:
    assert _FFMPEG is not None
    frame_path = path.with_suffix(".gray")
    argv = [
        _FFMPEG,
        "-y",
        "-v",
        "error",
        "-i",
        str(path),
        "-frames:v",
        "1",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "gray",
        str(frame_path),
    ]
    _ = subprocess.run(argv, capture_output=True, check=True, timeout=120)
    frame = frame_path.read_bytes()
    assert len(frame) == width * height
    return frame


def _box_mean(frame: bytes, *, width: int, x0: int, y0: int, x1: int, y1: int) -> float:
    total = 0
    for y in range(y0, y1):
        total += sum(frame[y * width + x0 : y * width + x1])
    return total / ((x1 - x0) * (y1 - y0))


@pytest.mark.integration
@pytest.mark.skipif(
    _FFMPEG is None or _FFPROBE is None,
    reason="ffmpeg/ffprobe no disponibles",
)
def test_renders_scaled_watermark_at_center_bottom(tmp_path: Path) -> None:
    assert _FFMPEG is not None
    clip = _generate_clip(tmp_path / "clip.mp4")
    watermark = _generate_watermark(tmp_path / "wm.png")
    plain = tmp_path / "plain.mp4"
    marked = tmp_path / "marked.mp4"
    _ = FFmpegAssembler(ffmpeg=_FFMPEG).assemble(RenderSpec(clip=clip, destination=plain))
    _ = FFmpegAssembler(ffmpeg=_FFMPEG).assemble(
        RenderSpec(
            clip=clip,
            destination=marked,
            watermark=watermark,
            watermark_config=_config(WatermarkPosition.CENTER_BOTTOM),
        )
    )
    # 64x64 escalado al 20% de 1080 = 216 px, centrado abajo con margen 20.
    box = {"width": 1080, "x0": 432, "y0": 1684, "x1": 648, "y1": 1900}
    plain_frame = _gray_frame(plain, width=1080, height=1920)
    marked_frame = _gray_frame(marked, width=1080, height=1920)
    assert _box_mean(plain_frame, **box) < 50
    assert _box_mean(marked_frame, **box) > 200
    # El PNG en loop no debe truncar el video al primer frame.
    assert _FFPROBE is not None
    marked_info = FFprobeProbe(ffprobe=_FFPROBE).probe(marked)
    assert marked_info.duration_s == pytest.approx(1.0, abs=0.2)
