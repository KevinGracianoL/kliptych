"""Integración de subtítulos ASS con ffmpeg real.

El fixture es un video sintético generado localmente con ffmpeg (nunca se
descarga material de la red). El módulo se salta entero si ffmpeg no está
instalado. El quemado usa ``libx264`` para no depender de NVENC.
"""

import shutil
import subprocess
from pathlib import Path

import pytest

from kliptych.gate.probe import FFprobeProbe
from kliptych.subtitles import SubtitleRenderer, SubtitleStyle
from kliptych.transcribe import Word

_FFMPEG = shutil.which("ffmpeg")

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(_FFMPEG is None, reason="ffmpeg no instalado"),
]

_FIXTURE_TIMEOUT_S = 120


def _generate_fixture(path: Path) -> Path:
    assert _FFMPEG is not None
    argv = [
        _FFMPEG,
        "-hide_banner",
        "-v",
        "error",
        "-y",
        "-f",
        "lavfi",
        "-i",
        "testsrc=size=320x240:rate=10:duration=2",
        "-f",
        "lavfi",
        "-i",
        "sine=frequency=440:duration=2",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-shortest",
        str(path),
    ]
    completed = subprocess.run(
        argv,
        capture_output=True,
        text=True,
        timeout=_FIXTURE_TIMEOUT_S,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    return path


def _words() -> tuple[Word, ...]:
    return (
        Word(start_s=0.0, end_s=1.0, text="hola", confidence=0.9, token_id=0),
        Word(start_s=1.0, end_s=2.0, text="mundo", confidence=0.9, token_id=1),
    )


def test_real_ass_file_is_written(tmp_path: Path) -> None:
    destination = tmp_path / "subs.ass"
    written = SubtitleRenderer().write(_words(), destination)
    content = written.read_text(encoding="utf-8")
    assert "PlayResX: 1080" in content
    assert "PlayResY: 1920" in content
    # Una linea por evento: estas dos palabras caben juntas, asi que el karaoke
    # las recorre dentro de un unico evento en vez de una por palabra.
    #
    # Esta asercion tambine FIJA LA BASE DE RANURAS. El evento dura 2.00 s y las
    # dos palabras duran 1.00 s cada una, luego 100 + 100 = 200 cs sobre un
    # evento de 200 cs. Con la base por DURACION las dos sumarian 190 cs y el
    # evento terminaria en 0:00:01.90, diez centisegundos antes de que termine
    # la ultima palabra. Quien "arregle" esto de vuelta a duraciones rompera
    # esta asercion sin saber por que estaba.
    assert content.count("Dialogue:") == 1
    assert "{\\k100}hola{\\k100}mundo" in content


def test_real_burn_produces_video_with_audio(tmp_path: Path) -> None:
    video = _generate_fixture(tmp_path / "fixture.mp4")
    subtitles = SubtitleRenderer().write(_words(), tmp_path / "subs.ass")
    destination = tmp_path / "burned.mp4"
    result = SubtitleRenderer().burn(
        video=video,
        subtitles=subtitles,
        destination=destination,
    )
    assert result == destination
    assert destination.stat().st_size > 0
    media = FFprobeProbe().probe(destination)
    assert media.has_video
    assert media.has_audio
    assert media.duration_s is not None
    assert media.duration_s > 0


def test_real_burn_respects_custom_style(tmp_path: Path) -> None:
    video = _generate_fixture(tmp_path / "fixture.mp4")
    renderer = SubtitleRenderer(style=SubtitleStyle(fontsize=36, outline=3))
    subtitles = renderer.write(_words(), tmp_path / "subs.ass")
    destination = renderer.burn(video=video, subtitles=subtitles, destination=tmp_path / "b.mp4")
    assert destination.is_file()
