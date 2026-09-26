"""Fix 5: filtro subtitles con rutas Windows relativas al output_dir."""

from pathlib import Path, PureWindowsPath
from types import SimpleNamespace

import pytest

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


def test_relative_paths_in_different_dirs_use_absolute_argv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    vids = tmp_path / "vids"
    subs_dir = tmp_path / "subs"
    out_dir = tmp_path / "out"
    _ = vids.mkdir(parents=True)
    _ = subs_dir.mkdir(parents=True)
    _ = out_dir.mkdir(parents=True)
    _ = (vids / "video.mp4").write_bytes(b"video")
    _ = (subs_dir / "subs.ass").write_text("[Script Info]\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    video_rel = Path("vids/video.mp4")
    ass_rel = Path("subs/subs.ass")
    dest_rel = Path("out/final.mp4")
    calls: list[tuple[list[str], dict[str, object]]] = []

    def _fake_run(argv: list[str], **kwargs: object) -> object:
        calls.append((list(argv), dict(kwargs)))
        _ = Path(argv[-1]).write_bytes(b"video")
        return SimpleNamespace(args=argv, returncode=0, stdout="", stderr="")

    monkeypatch.setattr("kliptych.subtitles.subprocess.run", _fake_run)
    renderer = SubtitleRenderer(render=RenderConfig())
    result = renderer.burn(video=video_rel, subtitles=ass_rel, destination=dest_rel)
    assert result == dest_rel
    assert (tmp_path / dest_rel).is_file()
    assert len(calls) == 1
    argv, kwargs = calls[0]
    input_index = argv.index("-i") + 1
    assert Path(argv[input_index]).is_absolute()
    assert Path(argv[-1]).is_absolute()
    filter_index = argv.index("-vf") + 1
    filter_arg = argv[filter_index]
    assert "subs.ass" in filter_arg
    assert "C\\:" not in filter_arg
    assert kwargs.get("cwd") == dest_rel.parent
