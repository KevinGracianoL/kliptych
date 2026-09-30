"""Sprint 3 Parte 3: composición Layout.SPLIT_SCREEN (Objetivos 1-4).

Objetivo 1: enum ``Layout`` con ``SPLIT_SCREEN``, ``SplitScreenConfig``
(top/bottom/gap/panel_ratio sobre lienzo 9:16) y composición en
``FFmpegAssembler`` con filtros scale/pad/crop/vstack, para dos videos o
video + imagen estática.

Objetivo 2: gate ``layout.geometry`` que verifica con ffprobe (vía
``MediaInfo``) el stream de video, las dimensiones esperadas y el aspecto
9:16, en cerrado (FAIL) ante cualquier desvío.

Objetivo 3: procedencia estricta T6 para ``SplitScreenDraft`` en el
resolutor (confianza válida + cita propia en el brief, sin alias sueltos,
con correspondencia numérica; MANUAL_REVIEW en cerrado).

Objetivo 4: invariante ``--resume`` (fingerprint incluye el split exacto)
e integración del pipeline given_clips.
"""

import shutil
import subprocess
from pathlib import Path

import pytest
from pydantic import ValidationError

from kliptych.assembler import AssembleError, FFmpegAssembler, RenderSpec
from kliptych.contract import Layout, SplitScreenConfig
from kliptych.gate.probe import FFprobeProbe

_FFMPEG = shutil.which("ffmpeg")
_FFPROBE = shutil.which("ffprobe")
_NEEDS_TOOLS = _FFMPEG is None or _FFPROBE is None


def _file(tmp_path: Path, name: str) -> Path:
    path = tmp_path / name
    _ = path.write_bytes(b"contenido")
    return path


def _split_spec(
    tmp_path: Path,
    *,
    gap: int = 0,
    panel_ratio: float = 0.5,
    bottom_suffix: str = ".mp4",
) -> RenderSpec:
    top = _file(tmp_path, "top.mp4")
    bottom = _file(tmp_path, f"bottom{bottom_suffix}")
    config = SplitScreenConfig(
        top_source="clip-top",
        bottom_source="clip-bottom",
        gap=gap,
        panel_ratio=panel_ratio,
    )
    return RenderSpec(
        clip=top,
        destination=tmp_path / "out.mp4",
        layout=Layout.SPLIT_SCREEN,
        split_screen=config,
        top_clip=top,
        bottom_clip=bottom,
    )


def test_layout_enum_has_split_screen() -> None:
    assert Layout.SPLIT_SCREEN == "split_screen"
    assert Layout.SPLIT_SCREEN.value == "split_screen"
    assert Layout.SINGLE == "single"


def test_split_screen_config_defaults() -> None:
    config = SplitScreenConfig(top_source="clip-top", bottom_source="clip-bottom")
    assert config.gap == 0
    assert config.panel_ratio == pytest.approx(0.5)
    assert (config.width, config.height) == (1080, 1920)
    assert config.panel_heights() == (960, 960)


def test_split_screen_config_panel_heights_with_gap() -> None:
    config = SplitScreenConfig(
        top_source="clip-top", bottom_source="clip-bottom", gap=20, panel_ratio=0.5
    )
    assert config.panel_heights() == (950, 950)


def test_split_screen_config_rejects_degenerate_ratio() -> None:
    with pytest.raises(ValidationError):
        _ = SplitScreenConfig(top_source="clip-top", bottom_source="clip-bottom", panel_ratio=0.0)
    with pytest.raises(ValidationError):
        _ = SplitScreenConfig(top_source="clip-top", bottom_source="clip-bottom", panel_ratio=1.0)


def test_split_screen_config_rejects_odd_gap() -> None:
    with pytest.raises(ValidationError):
        _ = SplitScreenConfig(top_source="clip-top", bottom_source="clip-bottom", gap=1)


def test_split_screen_config_rejects_non_vertical_canvas() -> None:
    with pytest.raises(ValidationError):
        _ = SplitScreenConfig(
            top_source="clip-top", bottom_source="clip-bottom", width=1920, height=1080
        )


def test_split_screen_config_rejects_unsafe_source() -> None:
    with pytest.raises(ValidationError):
        _ = SplitScreenConfig(top_source="../escape", bottom_source="clip-bottom")


def test_split_argv_stacks_two_videos(tmp_path: Path) -> None:
    argv = FFmpegAssembler().render_arguments(_split_spec(tmp_path))
    assert "-filter_complex" in argv
    graph = argv[argv.index("-filter_complex") + 1]
    assert "scale=1080:960" in graph
    assert "crop=1080:960" in graph
    assert "vstack=inputs=2" in graph
    assert argv.count("-i") == 2
    assert "-loop" not in argv
    assert "-shortest" not in argv


def test_split_argv_gap_inserts_pad(tmp_path: Path) -> None:
    argv = FFmpegAssembler().render_arguments(_split_spec(tmp_path, gap=20))
    graph = argv[argv.index("-filter_complex") + 1]
    assert "scale=1080:950" in graph
    assert "pad=1080:970" in graph
    assert "vstack=inputs=2" in graph


def test_split_argv_image_panel_needs_no_loop(tmp_path: Path) -> None:
    argv = FFmpegAssembler().render_arguments(_split_spec(tmp_path, bottom_suffix=".png"))
    assert argv.count("-i") == 2
    assert "-loop" not in argv
    assert "-shortest" not in argv
    graph = argv[argv.index("-filter_complex") + 1]
    assert "vstack=inputs=2" in graph


def test_split_without_config_fails(tmp_path: Path) -> None:
    clip = _file(tmp_path, "clip.mp4")
    with pytest.raises(AssembleError, match="split_screen"):
        _ = FFmpegAssembler().render_arguments(
            RenderSpec(
                clip=clip,
                destination=tmp_path / "out.mp4",
                layout=Layout.SPLIT_SCREEN,
            )
        )


def test_split_without_panels_fails(tmp_path: Path) -> None:
    clip = _file(tmp_path, "clip.mp4")
    config = SplitScreenConfig(top_source="clip-top", bottom_source="clip-bottom")
    with pytest.raises(AssembleError, match="panel"):
        _ = FFmpegAssembler().assemble(
            RenderSpec(
                clip=clip,
                destination=tmp_path / "out.mp4",
                layout=Layout.SPLIT_SCREEN,
                split_screen=config,
            )
        )


def _synth_clip(path: Path, source: str, *, duration: float = 2.0) -> Path:
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
        source,
        "-t",
        f"{duration}",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        str(path),
    ]
    _ = subprocess.run(argv, capture_output=True, check=True, timeout=180)
    return path


@pytest.mark.integration
@pytest.mark.skipif(_NEEDS_TOOLS, reason="ffmpeg/ffprobe no disponibles")
def test_split_screen_renders_vertical_1080x1920(tmp_path: Path) -> None:
    assert _FFMPEG is not None
    assert _FFPROBE is not None
    top = _synth_clip(tmp_path / "top.mp4", "testsrc=s=640x480:r=30")
    bottom = _synth_clip(tmp_path / "bottom.mp4", "smptebars=s=640x480:r=30")
    config = SplitScreenConfig(top_source="clip-top", bottom_source="clip-bottom")
    destination = tmp_path / "split.mp4"
    _ = FFmpegAssembler(ffmpeg=_FFMPEG).assemble(
        RenderSpec(
            clip=top,
            destination=destination,
            layout=Layout.SPLIT_SCREEN,
            split_screen=config,
            top_clip=top,
            bottom_clip=bottom,
        )
    )
    media = FFprobeProbe(ffprobe=_FFPROBE).probe(destination)
    assert (media.width, media.height) == (1080, 1920)
    assert media.has_video


@pytest.mark.integration
@pytest.mark.skipif(_NEEDS_TOOLS, reason="ffmpeg/ffprobe no disponibles")
def test_split_screen_renders_video_plus_image(tmp_path: Path) -> None:
    assert _FFMPEG is not None
    assert _FFPROBE is not None
    top = _synth_clip(tmp_path / "top.mp4", "testsrc=s=640x480:r=30")
    image = tmp_path / "bottom.png"
    argv = [
        _FFMPEG,
        "-y",
        "-nostdin",
        "-v",
        "error",
        "-f",
        "lavfi",
        "-i",
        "color=c=red:s=320x240:r=30",
        "-frames:v",
        "1",
        str(image),
    ]
    _ = subprocess.run(argv, capture_output=True, check=True, timeout=180)
    config = SplitScreenConfig(top_source="clip-top", bottom_source="img-bottom")
    destination = tmp_path / "split-img.mp4"
    _ = FFmpegAssembler(ffmpeg=_FFMPEG).assemble(
        RenderSpec(
            clip=top,
            destination=destination,
            layout=Layout.SPLIT_SCREEN,
            split_screen=config,
            top_clip=top,
            bottom_clip=image,
        )
    )
    media = FFprobeProbe(ffprobe=_FFPROBE).probe(destination)
    assert (media.width, media.height) == (1080, 1920)
