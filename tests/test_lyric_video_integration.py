"""Integración de formato LYRIC_VIDEO con ffmpeg real y --resume (Sprint 3 Parte 2 Objetivo 4 y 5).

Valida con ffmpeg real:
1. Render de clip LYRIC_VIDEO con subtítulos .ass derivados de .lrc.
2. Comprobación del artefacto final con FFprobeProbe.
3. Invariant de --resume: reutilización de etapas previas.
4. Invalidación de --resume cuando el contenido del archivo .lrc cambia.
"""

import math
import shutil
import subprocess
from collections.abc import Mapping
from pathlib import Path

import pytest

from kliptych.assets import AssetRegistry
from kliptych.contract import Format, LyricConfig, Segment
from kliptych.encoding import RenderConfig
from kliptych.gate.probe import FFprobeProbe
from kliptych.orchestrator import (
    PipelineConfig,
    PipelineError,
    run_long_video,
)
from tests.support import make_contract

_FFMPEG = shutil.which("ffmpeg")

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(_FFMPEG is None, reason="ffmpeg no instalado"),
]

_FIXTURE_TIMEOUT_S = 120


def _generate_synthetic_video(path: Path, duration: int = 3) -> Path:
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
        f"testsrc=size=360x640:rate=10:duration={duration}",
        "-f",
        "lavfi",
        "-i",
        f"sine=frequency=440:duration={duration}",
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


class _DummyModel:
    @staticmethod
    def select_segments(prompt: Mapping[str, object]) -> object:
        _ = prompt
        return {"segments": [{"start": 0.0, "end": 2.0, "reason": "intro"}]}


_VIDEO_URL = "https://example.com/source.mp4"


def _install_fixture_downloader(monkeypatch: pytest.MonkeyPatch, fixture: Path) -> None:
    class _Downloader:
        def __init__(self, *, timeout_s: float, max_size_bytes: int) -> None:
            _ = (timeout_s, max_size_bytes)
            self._fixture: Path = fixture

        def download_video(
            self, *, url: str, destination: Path, format_selector: str | None = None
        ) -> Path:
            _ = (url, format_selector)
            destination.parent.mkdir(parents=True, exist_ok=True)
            _ = shutil.copyfile(self._fixture, destination)
            return destination

    monkeypatch.setattr("kliptych.orchestrator.MediaDownloader", _Downloader)


def test_lyric_video_renders_and_resumes_with_ffmpeg(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _generate_synthetic_video(tmp_path / "source.mp4", duration=2)
    _install_fixture_downloader(monkeypatch, source)
    lrc_path = tmp_path / "lyrics.lrc"
    _ = lrc_path.write_text("[00:00.00]Canto inicial\n[00:01.00]Canto medio\n", encoding="utf-8")

    out_dir = tmp_path / "out"
    contract = make_contract(
        format_=Format.LYRIC_VIDEO,
        lyric_video=LyricConfig(track_name="Tema", artist_name="Artista"),
    )
    contract = contract.model_copy(update={"segments": (Segment(start_s=0.0, end_s=2.0),)})

    render = RenderConfig(nvenc_available=False)
    config = PipelineConfig(
        output_dir=out_dir,
        contract=contract,
        render=render,
        repost_mode=True,  # omite whisper/momentos, enfoca en render de video + subtitulos lrc
        lrc_path=lrc_path,
    )

    # 1. Primer run: genera final.mp4 y subtitles.ass
    result1 = run_long_video(
        _VIDEO_URL,
        model=_DummyModel(),
        config=config,
        resume=False,
    )
    assert (out_dir / "final.mp4").is_file()
    assert (out_dir / "subtitles.ass").is_file()
    assert result1.subtitles is not None
    assert result1.subtitles.is_file()

    probe = FFprobeProbe()
    info = probe.probe(out_dir / "final.mp4")
    assert info.has_video
    assert info.has_audio
    assert info.duration_s is not None
    assert math.isclose(info.duration_s, 2.0, abs_tol=0.5)

    # 2. Segundo run con resume=True: no re-renderiza
    mtime_before = (out_dir / "final.mp4").stat().st_mtime
    result2 = run_long_video(
        _VIDEO_URL,
        model=_DummyModel(),
        config=config,
        resume=True,
    )
    mtime_after = (out_dir / "final.mp4").stat().st_mtime
    assert mtime_before == mtime_after
    assert result2.subtitles == result1.subtitles

    # 3. Tercer run con cambio en el contenido de .lrc: invalida resume
    _ = lrc_path.write_text("[00:00.00]Letra cambiada completamente\n", encoding="utf-8")
    _ = run_long_video(
        _VIDEO_URL,
        model=_DummyModel(),
        config=config,
        resume=True,
    )
    mtime_third = (out_dir / "final.mp4").stat().st_mtime
    assert mtime_third != mtime_before

    # Verificar que el nuevo .ass contiene la nueva letra
    new_ass = (out_dir / "subtitles.ass").read_text(encoding="utf-8")
    assert "Letra cambiada completamente" in new_ass


def test_lyric_video_resolves_lrc_from_asset_id_and_burns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _generate_synthetic_video(tmp_path / "source.mp4", duration=2)
    _install_fixture_downloader(monkeypatch, source)

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    song_file = workspace / "song.lrc"
    _ = song_file.write_text("[00:00.00]hello world\n", encoding="utf-8")

    registry = AssetRegistry(workspace)
    _ = registry.register(
        asset_id="song",
        kind="lyrics",
        uri="song.lrc",
        origin="test",
    )

    out_dir = tmp_path / "out"
    contract = make_contract(
        format_=Format.LYRIC_VIDEO,
        lyric_video=LyricConfig(lrc_asset_id="song"),
    )
    contract = contract.model_copy(update={"segments": (Segment(start_s=0.0, end_s=2.0),)})

    class _FailIfCalledLrclib:
        @staticmethod
        def get_synced_lyrics(
            *,
            track_name: str,
            artist_name: str | None = None,
            album_name: str | None = None,
            duration_s: float | None = None,
        ) -> str:
            _ = (track_name, artist_name, album_name, duration_s)
            msg = "lrclib provider no debe ser invocado"
            raise RuntimeError(msg)

    render = RenderConfig(nvenc_available=False)
    config = PipelineConfig(
        output_dir=out_dir,
        contract=contract,
        render=render,
        repost_mode=True,
        synced_lyrics_provider=_FailIfCalledLrclib(),
        assets=registry,
    )

    result = run_long_video(
        _VIDEO_URL,
        model=_DummyModel(),
        config=config,
    )
    assert (out_dir / "final.mp4").is_file()
    assert (out_dir / "subtitles.ass").is_file()
    assert result.subtitles is not None
    ass_text = result.subtitles.read_text(encoding="utf-8")
    assert "hello world" in ass_text


def test_lyric_video_missing_lrc_asset_raises_pipeline_error_without_fallback_to_lrclib(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _generate_synthetic_video(tmp_path / "source.mp4", duration=2)
    _install_fixture_downloader(monkeypatch, source)

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    registry = AssetRegistry(workspace)

    out_dir = tmp_path / "out"
    contract = make_contract(
        format_=Format.LYRIC_VIDEO,
        lyric_video=LyricConfig(
            lrc_asset_id="missing_song",
            track_name="Some Track",
            lrclib_enabled=True,
        ),
    )
    contract = contract.model_copy(update={"segments": (Segment(start_s=0.0, end_s=2.0),)})

    class _FailIfCalledLrclib:
        @staticmethod
        def get_synced_lyrics(
            *,
            track_name: str,
            artist_name: str | None = None,
            album_name: str | None = None,
            duration_s: float | None = None,
        ) -> str:
            _ = (track_name, artist_name, album_name, duration_s)
            msg = "lrclib provider no debe ser invocado cuando lrc_asset_id está ausente"
            raise RuntimeError(msg)

    render = RenderConfig(nvenc_available=False)
    config = PipelineConfig(
        output_dir=out_dir,
        contract=contract,
        render=render,
        repost_mode=True,
        synced_lyrics_provider=_FailIfCalledLrclib(),
        assets=registry,
    )

    with pytest.raises(PipelineError):
        _ = run_long_video(
            _VIDEO_URL,
            model=_DummyModel(),
            config=config,
        )
