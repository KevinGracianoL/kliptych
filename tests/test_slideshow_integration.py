"""Integración del modo Slideshow con ffmpeg real (cero red).

Las imágenes y la pista de audio se generan localmente con ffmpeg. Se verifica
que el pipeline ensambla las imágenes en un vídeo vertical 9:16, inyecta el
audio obligatorio, publica ``final.mp4`` y limpia todos los temporales. El
módulo se salta entero si falta ffmpeg o ffprobe.
"""

import shutil
import subprocess
from pathlib import Path

import pytest

from kliptych.contract import Contract
from kliptych.encoding import RenderConfig
from kliptych.gate.probe import FFprobeProbe
from kliptych.orchestrator import PipelineConfig, SlideshowResult, run_slideshow

_FFMPEG = shutil.which("ffmpeg")
_FFPROBE = shutil.which("ffprobe")

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        _FFMPEG is None or _FFPROBE is None,
        reason="ffmpeg/ffprobe no instalados",
    ),
]

_FIXTURE_TIMEOUT_S = 120
_SLIDE_DURATION_S = 1.0
_AUDIO_DURATION_S = 2.0
_EXPECTED_WIDTH = 1080
_EXPECTED_HEIGHT = 1920


def _contract() -> Contract:
    return Contract.model_validate(
        {
            "schema_version": "1.1",
            "campaign_id": "camp-slideshow",
            "format": "slideshow",
            "mode": "slideshow",
            "platforms": {
                "tiktok": {
                    "duration": {"min_s": None, "max_s": None},
                    "caption_rules": {"must_mention": [], "first_line": None, "forbidden": []},
                    "audio_rule": "any",
                    "required_hashtags": [],
                    "required_mentions": [],
                    "attribution": {"type": "none", "value": None},
                    "link_rules": {"link_in_bio": False},
                }
            },
            "languages": {"source": "es", "subtitles": None, "caption": "es", "voice": None},
            "official_audio": None,
            "watermark": {"required": False, "asset_id": None, "visible_full_video": False},
            "spelling_locks": [],
            "prohibitions": [],
            "rules": {"hard": [], "recommended": [], "manual_review": []},
            "assets": {"required": [], "optional": []},
            "segments": [],
            "geo_target": None,
            "min_views_for_payout": {"value": None, "enforcement": "post_publication_manual"},
            "analytics_proof_required": {"value": False, "enforcement": "post_publication_manual"},
        }
    )


def _run_fixture(argv: list[str]) -> None:
    completed = subprocess.run(
        argv,
        capture_output=True,
        text=True,
        timeout=_FIXTURE_TIMEOUT_S,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr


def _generate_image(path: Path, *, color: str, size: str) -> Path:
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
        f"color=c={color}:s={size}:d=1",
        "-frames:v",
        "1",
        str(path),
    ]
    _run_fixture(argv)
    return path


def _generate_audio(path: Path) -> Path:
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
        f"sine=frequency=880:duration={_AUDIO_DURATION_S:g}",
        "-c:a",
        "pcm_s16le",
        str(path),
    ]
    _run_fixture(argv)
    return path


def _config(tmp_path: Path, *, audio_track_path: Path) -> PipelineConfig:
    return PipelineConfig(
        output_dir=tmp_path / "out",
        contract=_contract(),
        render=RenderConfig(),
        audio_locked=True,
        audio_track_path=audio_track_path,
    )


def _leftovers(output_dir: Path) -> list[str]:
    return [path.name for path in output_dir.iterdir() if ".part-" in path.name]


def test_slideshow_end_to_end(tmp_path: Path) -> None:
    images = (
        _generate_image(tmp_path / "red.png", color="red", size="320x240"),
        _generate_image(tmp_path / "blue.png", color="blue", size="240x320"),
    )
    track = _generate_audio(tmp_path / "track.wav")
    result = run_slideshow(
        images,
        config=_config(tmp_path, audio_track_path=track),
        slide_duration_s=_SLIDE_DURATION_S,
    )
    assert isinstance(result, SlideshowResult)
    assert result.images == images
    assert result.subtitles is None
    assert result.final_video == tmp_path / "out" / "final.mp4"
    assert result.final_video.is_file()
    media = FFprobeProbe().probe(result.final_video)
    assert media.has_video
    assert media.has_audio
    assert media.width == _EXPECTED_WIDTH
    assert media.height == _EXPECTED_HEIGHT
    assert media.duration_s is not None
    assert media.duration_s == pytest.approx(_SLIDE_DURATION_S * len(images), abs=0.3)
    assert track.is_file()
    assert all(image.is_file() for image in images)
    assert _leftovers(tmp_path / "out") == []


def test_slideshow_cleans_downloaded_images(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    image = _generate_image(tmp_path / "only.png", color="green", size="320x240")
    track = _generate_audio(tmp_path / "track.wav")

    class _CopyDownloader:
        def __init__(self, *, timeout_s: float, max_size_bytes: int) -> None:
            _ = (timeout_s, max_size_bytes)
            self._image: Path = image

        def download_video(
            self, *, url: str, destination: Path, format_selector: str | None = None
        ) -> Path:
            _ = (url, format_selector)
            destination.parent.mkdir(parents=True, exist_ok=True)
            _ = shutil.copyfile(self._image, destination)
            return destination

    monkeypatch.setattr("kliptych.orchestrator.MediaDownloader", _CopyDownloader)
    result = run_slideshow(
        ("https://example.com/only.png",),
        config=_config(tmp_path, audio_track_path=track),
        slide_duration_s=_SLIDE_DURATION_S,
    )
    assert result.final_video.is_file()
    assert result.images[0].is_file()
    assert not result.slideshow_video.exists()
    assert _leftovers(tmp_path / "out") == []
