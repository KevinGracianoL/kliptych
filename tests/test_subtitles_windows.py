"""Fix 5: filtro subtitles con rutas Windows relativas al output_dir."""

from pathlib import Path, PureWindowsPath

from kliptych.encoding import RenderConfig
from kliptych.subtitles import SubtitleRenderer


def _video(tmp_path: Path) -> Path:
    path = tmp_path / "video.mp4"
    _ = path.write_bytes(b"video")
    return path


def _ass(tmp_path: Path) -> Path:
    path = tmp_path / "subs.ass"
    _ = path.write_text("[Script Info]\n", encoding="utf-8")
    return path


def test_windows_absolute_path_uses_relative_in_argv(tmp_path: Path) -> None:
    video = _video(tmp_path)
    ass = _ass(tmp_path)
    destination = tmp_path / "out.mp4"
    renderer = SubtitleRenderer(render=RenderConfig())
    argv = renderer.render_arguments(video=video, subtitles=ass, destination=destination)
    filter_index = argv.index("-vf") + 1
    filter_arg = argv[filter_index]
    assert "C\\:" not in filter_arg
    assert "C:/" not in filter_arg
    assert "subs.ass" in filter_arg


def test_windows_style_path_does_not_emit_drive_escape(tmp_path: Path) -> None:
    win_sub = PureWindowsPath("C:/Users/test/out/subs.ass")
    win_video = PureWindowsPath("C:/Users/test/out/video.mp4")
    win_dest = PureWindowsPath("C:/Users/test/out/final.mp4")
    renderer = SubtitleRenderer(render=RenderConfig())
    argv = renderer.render_arguments(
        video=Path(str(win_video)),
        subtitles=Path(str(win_sub)),
        destination=Path(str(win_dest)),
    )
    _ = tmp_path
    filter_index = argv.index("-vf") + 1
    filter_arg = argv[filter_index]
    assert "C\\:" not in filter_arg
