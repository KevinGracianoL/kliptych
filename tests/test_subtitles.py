"""Tests unitarios de generación ASS y quemado: ffmpeg simulado, cero red.

El contenido ASS se verifica sobre texto controlado y ningún test ejecuta ffmpeg
real (eso vive en ``test_subtitles_integration.py``).
"""

import subprocess
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import cast

import pytest
from pydantic import ValidationError

from kliptych import subtitles
from kliptych.encoding import RenderConfig
from kliptych.subtitles import (
    SubtitleError,
    SubtitleLayout,
    SubtitleRenderer,
    SubtitleStyle,
)
from kliptych.transcribe import Word

_Call = tuple[list[str], dict[str, object]]
_FakeRun = Callable[..., subprocess.CompletedProcess[str]]


def _private(name: str) -> object:
    return cast("object", getattr(subtitles, name))


_format_time = cast("Callable[[float], str]", _private("_format_time"))
_escape_text = cast("Callable[[str], str]", _private("_escape_text"))
_style_line = cast("Callable[[SubtitleStyle], str]", _private("_style_line"))
_dialogue_line = cast("Callable[[Word], str]", _private("_dialogue_line"))
_filter_path = cast("Callable[[Path], str]", _private("_filter_path"))


def _word(start: float, end: float, text: str, token_id: int = 0) -> Word:
    return Word(start_s=start, end_s=end, text=text, confidence=0.9, token_id=token_id)


def _completed(*, returncode: int = 0, stderr: str = "") -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(
        args=["ffmpeg"], returncode=returncode, stdout="", stderr=stderr
    )


def _fake_run(
    calls: list[_Call],
    *,
    returncode: int = 0,
    stderr: str = "",
    write_output: bool = True,
) -> _FakeRun:
    def run(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append((argv, kwargs))
        if write_output and returncode == 0:
            _ = Path(argv[-1]).write_bytes(b"video")
        return _completed(returncode=returncode, stderr=stderr)

    return run


def _raising_run(exc: BaseException) -> _FakeRun:
    def run(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        _ = (argv, kwargs)
        raise exc

    return run


def _video(tmp_path: Path) -> Path:
    path = tmp_path / "video.mp4"
    _ = path.write_bytes(b"video")
    return path


def _ass(tmp_path: Path) -> Path:
    path = tmp_path / "subs.ass"
    _ = path.write_text("[Script Info]\n", encoding="utf-8")
    return path


def test_format_time_zero() -> None:
    assert _format_time(0.0) == "0:00:00.00"


def test_format_time_rounds_to_centiseconds() -> None:
    assert _format_time(1.234) == "0:00:01.23"
    assert _format_time(0.999) == "0:00:01.00"


def test_format_time_with_hours() -> None:
    assert _format_time(3661.5) == "1:01:01.50"


def test_format_time_negative_clamps_to_zero() -> None:
    assert _format_time(-1.0) == "0:00:00.00"


def test_escape_text_escapes_braces() -> None:
    assert _escape_text("{hola}") == "\\{hola\\}"


def test_escape_text_escapes_newlines() -> None:
    assert _escape_text("a\nb") == "a\\Nb"
    assert _escape_text("a\r\nb") == "a\\Nb"


def test_escape_text_escapes_backslash() -> None:
    assert _escape_text("a\\b") == "a\\\\b"


def test_escape_text_plain_is_unchanged() -> None:
    assert _escape_text("hola mundo") == "hola mundo"


def test_style_line_serializes_all_fields() -> None:
    style = SubtitleStyle(fontname="Verdana", fontsize=60, outline=3, shadow=1, margin_v=80)
    line = _style_line(style)
    assert line.startswith("Style: Default,Verdana,60,")
    assert "&H00FFFFFF" in line
    assert ",3,1," in line
    assert ",2,20,20,80,1" in line


def test_dialogue_line_has_karaoke_and_times() -> None:
    line = _dialogue_line(_word(0.5, 1.0, "hola"))
    assert line == "Dialogue: 0,0:00:00.50,0:00:01.00,Default,,0,0,0,,{\\k50}hola"


def test_dialogue_line_escapes_text() -> None:
    line = _dialogue_line(_word(0.0, 0.5, "{x}"))
    assert line.endswith("{\\k50}\\{x\\}")


def test_build_contains_sections_and_formats() -> None:
    content = SubtitleRenderer().build([_word(0.0, 0.5, "hola")])
    assert "[Script Info]" in content
    assert "[V4+ Styles]" in content
    assert "[Events]" in content
    assert content.startswith("[Script Info]\n")
    assert content.endswith("\n")
    assert content.count("Dialogue:") == 1


def test_build_play_res_is_vertical() -> None:
    content = SubtitleRenderer(layout=SubtitleLayout(width=720, height=1280)).build(
        [_word(0.0, 0.5, "hola")]
    )
    assert "PlayResX: 720" in content
    assert "PlayResY: 1280" in content


def test_build_one_event_per_word_with_matching_times() -> None:
    words = [_word(0.0, 0.5, "hola", 0), _word(0.6, 1.2, "mundo", 1)]
    content = SubtitleRenderer().build(words)
    dialogues = [line for line in content.splitlines() if line.startswith("Dialogue:")]
    assert len(dialogues) == 2
    assert "0:00:00.00,0:00:00.50" in dialogues[0]
    assert "0:00:00.60,0:00:01.20" in dialogues[1]
    assert "{\\k50}hola" in dialogues[0]
    assert "{\\k60}mundo" in dialogues[1]


def test_build_empty_words_raise() -> None:
    with pytest.raises(SubtitleError, match="no hay palabras"):
        _ = SubtitleRenderer().build([])


def test_write_creates_file(tmp_path: Path) -> None:
    destination = tmp_path / "nested" / "subs.ass"
    written = SubtitleRenderer().write([_word(0.0, 0.5, "hola")], destination)
    assert written == destination
    assert "Dialogue:" in destination.read_text(encoding="utf-8")


def test_write_without_words_raises(tmp_path: Path) -> None:
    with pytest.raises(SubtitleError, match="no hay palabras"):
        _ = SubtitleRenderer().write([], tmp_path / "subs.ass")


def test_filter_path_escapes_drive_colon() -> None:
    assert _filter_path(Path("C:/tmp/subs.ass")) == "C\\:/tmp/subs.ass"


def test_filter_path_posix_is_unchanged() -> None:
    assert _filter_path(Path("/data/subs.ass")) == "/data/subs.ass"


def test_render_arguments_fall_back_to_libx264(tmp_path: Path) -> None:
    argv = SubtitleRenderer().render_arguments(
        video=_video(tmp_path), subtitles=_ass(tmp_path), destination=tmp_path / "out.mp4"
    )
    assert "-c:v" in argv
    assert "libx264" in argv
    assert "h264_nvenc" not in argv


def test_render_arguments_use_nvenc_when_available(tmp_path: Path) -> None:
    renderer = SubtitleRenderer(render=RenderConfig(nvenc_available=True))
    argv = renderer.render_arguments(
        video=_video(tmp_path), subtitles=_ass(tmp_path), destination=tmp_path / "out.mp4"
    )
    assert "h264_nvenc" in argv
    assert "libx264" not in argv


def test_render_arguments_reference_subtitles_filter(tmp_path: Path) -> None:
    ass = _ass(tmp_path)
    argv = SubtitleRenderer().render_arguments(
        video=_video(tmp_path), subtitles=ass, destination=tmp_path / "out.mp4"
    )
    filter_index = argv.index("-vf") + 1
    assert argv[filter_index] == f"subtitles=filename='{_filter_path(ass)}'"


def test_burn_missing_video_raises(tmp_path: Path) -> None:
    with pytest.raises(SubtitleError, match="video no existe"):
        _ = SubtitleRenderer().burn(
            video=tmp_path / "falta.mp4",
            subtitles=_ass(tmp_path),
            destination=tmp_path / "out.mp4",
        )


def test_burn_missing_subtitles_raises(tmp_path: Path) -> None:
    with pytest.raises(SubtitleError, match="subtítulos no existe"):
        _ = SubtitleRenderer().burn(
            video=_video(tmp_path),
            subtitles=tmp_path / "falta.ass",
            destination=tmp_path / "out.mp4",
        )


def test_burn_publishes_artifact(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[_Call] = []
    monkeypatch.setattr("kliptych.subtitles.subprocess.run", _fake_run(calls))
    destination = tmp_path / "out" / "burned.mp4"
    result = SubtitleRenderer().burn(
        video=_video(tmp_path), subtitles=_ass(tmp_path), destination=destination
    )
    assert result == destination
    assert destination.read_bytes() == b"video"
    assert len(calls) == 1


def test_burn_failure_leaves_destination_intact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "kliptych.subtitles.subprocess.run",
        _fake_run([], returncode=1, stderr="boom", write_output=False),
    )
    destination = tmp_path / "out.mp4"
    _ = destination.write_bytes(b"previo")
    with pytest.raises(SubtitleError, match="falló con código 1"):
        _ = SubtitleRenderer().burn(
            video=_video(tmp_path), subtitles=_ass(tmp_path), destination=destination
        )
    assert destination.read_bytes() == b"previo"


def test_burn_missing_ffmpeg_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "kliptych.subtitles.subprocess.run", _raising_run(FileNotFoundError("ffmpeg"))
    )
    with pytest.raises(SubtitleError, match="no está disponible"):
        _ = SubtitleRenderer().burn(
            video=_video(tmp_path),
            subtitles=_ass(tmp_path),
            destination=tmp_path / "out.mp4",
        )


def test_burn_timeout_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "kliptych.subtitles.subprocess.run",
        _raising_run(subprocess.TimeoutExpired(cmd="ffmpeg", timeout=600.0)),
    )
    with pytest.raises(SubtitleError, match="timeout"):
        _ = SubtitleRenderer().burn(
            video=_video(tmp_path),
            subtitles=_ass(tmp_path),
            destination=tmp_path / "out.mp4",
        )


def test_burn_os_error_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "kliptych.subtitles.subprocess.run", _raising_run(OSError("permiso denegado"))
    )
    with pytest.raises(SubtitleError, match="no se pudo ejecutar"):
        _ = SubtitleRenderer().burn(
            video=_video(tmp_path),
            subtitles=_ass(tmp_path),
            destination=tmp_path / "out.mp4",
        )


def test_subtitle_style_defaults() -> None:
    style = SubtitleStyle()
    assert style.fontname == "Arial"
    assert style.fontsize == 48
    assert style.alignment == 2


@pytest.mark.parametrize("colour", ["FFFFFF", "&HFFF", "rojo", "&HZZZZZZZZ"])
def test_subtitle_style_rejects_bad_colour(colour: str) -> None:
    with pytest.raises(ValidationError):
        _ = SubtitleStyle(primary_colour=colour)


def test_subtitle_style_rejects_non_positive_fontsize() -> None:
    with pytest.raises(ValidationError):
        _ = SubtitleStyle(fontsize=0)


def test_subtitle_style_rejects_alignment_out_of_range() -> None:
    with pytest.raises(ValidationError):
        _ = SubtitleStyle(alignment=10)


def test_subtitle_style_rejects_extra_fields() -> None:
    with pytest.raises(ValidationError):
        _ = SubtitleStyle.model_validate({"fontsize": 48, "extra": 1})


def test_subtitle_style_is_frozen() -> None:
    style = SubtitleStyle()
    with pytest.raises(ValidationError):
        style.fontsize = 10


def test_subtitle_layout_rejects_invalid_dimensions() -> None:
    with pytest.raises(ValueError, match="lienzo"):
        _ = SubtitleLayout(width=0, height=1280)


def test_style_sequence_is_immutable_input() -> None:
    words: Sequence[Word] = (_word(0.0, 0.5, "hola"),)
    assert "hola" in SubtitleRenderer().build(words)
