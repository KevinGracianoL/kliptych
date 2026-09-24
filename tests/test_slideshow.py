"""Tests unitarios del modo Slideshow (D3-PR3) con todos los límites simulados.

Descarga, ffmpeg y ffprobe se simulan: cero red y cero subproceso real. Se
verifica la validación de la duración por slide, el audio obligatorio y la
lista de imágenes; el formato del archivo ``concat`` (duraciones y último frame
repetido); la construcción del argv de ensamblado; la inyección obligatoria de
audio; la limpieza de temporales; la estructura de ``SlideshowResult`` y que el
pipeline long_video existente no cambia.
"""

import subprocess
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, cast

import pytest

from kliptych import orchestrator
from kliptych.contract import Contract
from kliptych.encoding import RenderConfig
from kliptych.orchestrator import (
    PipelineConfig,
    PipelineError,
    SlideshowResult,
    run_slideshow,
)

_DEFAULT_PROBE = "width=1080\nheight=1920\nduration=2.000000\n"


class _Registry(Protocol):
    """Superficie del registro de limpieza que usan los tests."""

    paths: list[Path]

    def register(self, path: Path) -> Path: ...


def _private(name: str) -> object:
    return cast("object", getattr(orchestrator, name))


_cleanup_registry = cast("Callable[[], _Registry]", _private("_CleanupRegistry"))
_concat_file_content = cast("Callable[..., str]", _private("_concat_file_content"))
_concat_file_line = cast("Callable[[Path], str]", _private("_concat_file_line"))
_assemble_slideshow = cast("Callable[..., Path]", _private("_assemble_slideshow"))


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


@dataclass
class _Harness:
    """Bitácora de llamadas, comandos y códigos de salida simulados."""

    events: list[str] = field(default_factory=list)
    commands: list[list[str]] = field(default_factory=list)
    ffmpeg_returncode: int = 0
    downloads: list[tuple[str, Path]] = field(default_factory=list)


class _Downloader:
    def __init__(self, harness: _Harness) -> None:
        self._harness: _Harness = harness

    def download_video(
        self, *, url: str, destination: Path, format_selector: str | None = None
    ) -> Path:
        _ = format_selector
        self._harness.events.append(f"download:{url}")
        self._harness.downloads.append((url, destination))
        destination.parent.mkdir(parents=True, exist_ok=True)
        _ = destination.write_bytes(b"image")
        return destination


def _ffmpeg(harness: _Harness) -> Callable[..., object]:
    def run(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        _ = kwargs
        harness.commands.append(list(argv))
        if "-f" in argv and argv[argv.index("-f") + 1] == "concat":
            harness.events.append("assemble")
        elif "-filter_complex" in argv:
            harness.events.append("inject_mix")
        elif "-map" in argv:
            harness.events.append("inject_replace")
        else:
            harness.events.append("ffmpeg")
        _ = Path(argv[-1]).write_bytes(b"render")
        return subprocess.CompletedProcess(
            args=argv,
            returncode=harness.ffmpeg_returncode,
            stdout="",
            stderr="boom",
        )

    return run


def _install(monkeypatch: pytest.MonkeyPatch, harness: _Harness) -> None:
    def factory(*, timeout_s: float, max_size_bytes: int) -> _Downloader:
        _ = (timeout_s, max_size_bytes)
        return _Downloader(harness)

    monkeypatch.setattr("kliptych.orchestrator.MediaDownloader", factory)
    monkeypatch.setattr("kliptych.orchestrator.subprocess.run", _ffmpeg(harness))


def _config(
    tmp_path: Path,
    *,
    audio_locked: bool = True,
    audio_track_path: Path | None = None,
    audio_mix_ratio: float = 1.0,
) -> PipelineConfig:
    return PipelineConfig(
        output_dir=tmp_path / "out",
        contract=_contract(),
        render=RenderConfig(),
        audio_locked=audio_locked,
        audio_track_path=audio_track_path,
        audio_mix_ratio=audio_mix_ratio,
    )


def _images(tmp_path: Path, count: int = 2) -> tuple[Path, ...]:
    paths: list[Path] = []
    for index in range(count):
        path = tmp_path / f"slide_{index}.png"
        _ = path.write_bytes(b"image")
        paths.append(path)
    return tuple(paths)


def _run(
    images: Sequence[Path | str],
    config: PipelineConfig,
    *,
    slide_duration_s: float = 3.0,
) -> SlideshowResult:
    return run_slideshow(images, config=config, slide_duration_s=slide_duration_s)


def _leftovers(output_dir: Path) -> list[str]:
    return [path.name for path in output_dir.iterdir() if ".part-" in path.name]


def test_rejects_non_positive_slide_duration(tmp_path: Path) -> None:
    config = _config(tmp_path)
    with pytest.raises(PipelineError, match="positiva"):
        _ = _run(_images(tmp_path), config, slide_duration_s=0.0)
    with pytest.raises(PipelineError, match="positiva"):
        _ = _run(_images(tmp_path), config, slide_duration_s=-1.0)


def test_requires_audio_locked(tmp_path: Path) -> None:
    config = _config(tmp_path, audio_locked=False)
    with pytest.raises(PipelineError, match="audio_locked"):
        _ = _run(_images(tmp_path), config)


def test_rejects_mix_ratio_below_one(tmp_path: Path) -> None:
    config = _config(tmp_path, audio_locked=True, audio_mix_ratio=0.5)
    with pytest.raises(PipelineError, match=r"audio_mix_ratio >= 1\.0"):
        _ = _run(_images(tmp_path), config)


def test_rejects_empty_images(tmp_path: Path) -> None:
    with pytest.raises(PipelineError, match="imagen"):
        _ = _run((), _config(tmp_path))


def test_rejects_missing_local_image(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    harness = _Harness()
    _install(monkeypatch, harness)
    missing = tmp_path / "missing.png"
    with pytest.raises(PipelineError, match="no existe"):
        _ = _run((missing,), _config(tmp_path))


def test_concat_file_content_repeats_last_frame(tmp_path: Path) -> None:
    first = tmp_path / "a.png"
    second = tmp_path / "b.png"
    content = _concat_file_content((first, second), slide_duration_s=2.5)
    lines = content.splitlines()
    assert lines == [
        f"file '{first.as_posix()}'",
        "duration 2.500",
        f"file '{second.as_posix()}'",
        "duration 2.500",
        f"file '{second.as_posix()}'",
    ]
    assert content.endswith("\n")


def test_concat_file_line_escapes_single_quote(tmp_path: Path) -> None:
    path = tmp_path / "o'brien.png"
    expected = f"file '{tmp_path.as_posix()}/o'\\''brien.png'"
    assert _concat_file_line(path) == expected


def test_assemble_slideshow_builds_concat_argv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness = _Harness()
    monkeypatch.setattr("kliptych.orchestrator.subprocess.run", _ffmpeg(harness))
    concat = tmp_path / "slideshow_input.txt"
    _ = concat.write_text("", encoding="utf-8")
    registry = _cleanup_registry()
    destination = _assemble_slideshow(
        concat,
        render=RenderConfig(),
        output_dir=tmp_path,
        registry=registry,
    )
    assert destination.is_file()
    assert destination in registry.paths
    argv = harness.commands[0]
    assert isinstance(argv, list)
    assert argv[argv.index("-f") + 1] == "concat"
    assert argv[argv.index("-safe") + 1] == "0"
    assert argv[argv.index("-i") + 1] == str(concat)
    assert argv[argv.index("-r") + 1] == "30"
    assert "force_original_aspect_ratio=decrease" in argv[argv.index("-vf") + 1]
    assert argv[-1] == str(destination)


def test_full_pipeline_downloads_assembles_and_injects_audio(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness = _Harness()
    _install(monkeypatch, harness)
    track = tmp_path / "track.wav"
    _ = track.write_bytes(b"audio")
    config = _config(tmp_path, audio_track_path=track)
    result = _run(
        ("https://example.com/a.png", "https://example.com/b.png"),
        config,
        slide_duration_s=1.5,
    )
    assert harness.events == [
        "download:https://example.com/a.png",
        "download:https://example.com/b.png",
        "assemble",
        "inject_replace",
    ]
    assert isinstance(result, SlideshowResult)
    assert len(result.images) == 2
    assert all(image.parent == tmp_path / "out" for image in result.images)
    assert result.slideshow_video.parent == tmp_path / "out"
    assert "slideshow" in result.slideshow_video.name
    assert result.final_video == tmp_path / "out" / "final.mp4"
    assert result.final_video.is_file()
    assert result.subtitles is None
    assert _leftovers(tmp_path / "out") == []
    assert len(result.cleaning) == 3
    assert all(image.is_file() for image in result.images)


def test_cleanup_captures_concat_images_and_intermediates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness = _Harness()
    _install(monkeypatch, harness)
    track = tmp_path / "track.wav"
    _ = track.write_bytes(b"audio")
    config = _config(tmp_path, audio_track_path=track)
    result = _run(("https://example.com/a.png",), config, slide_duration_s=1.0)
    cleaned = set(result.cleaning)
    assert any(name.endswith(".txt") for name in cleaned)
    assert any("slideshow" in name for name in cleaned)
    assert any("audio_injected" in name for name in cleaned)
    assert not result.slideshow_video.exists()
    assert result.images[0].is_file()
    assert result.final_video.is_file()


def test_local_images_are_not_deleted(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    harness = _Harness()
    _install(monkeypatch, harness)
    track = tmp_path / "track.wav"
    _ = track.write_bytes(b"audio")
    images = _images(tmp_path)
    result = _run(images, _config(tmp_path, audio_track_path=track))
    assert result.images == images
    assert all(image.is_file() for image in images)


def test_audio_track_required_by_pipeline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    harness = _Harness()
    _install(monkeypatch, harness)
    with pytest.raises(PipelineError, match="audio_track"):
        _ = _run(_images(tmp_path), _config(tmp_path))


def test_ffmpeg_failure_is_pipeline_error_and_cleans(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness = _Harness(ffmpeg_returncode=1)
    _install(monkeypatch, harness)
    track = tmp_path / "track.wav"
    _ = track.write_bytes(b"audio")
    with pytest.raises(PipelineError, match="ffmpeg falló"):
        _ = _run(_images(tmp_path), _config(tmp_path, audio_track_path=track))
    assert _leftovers(tmp_path / "out") == []


def test_rejects_unwritable_output_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    harness = _Harness()
    _install(monkeypatch, harness)
    blocker = tmp_path / "out"
    _ = blocker.write_bytes(b"soy un archivo")
    config = PipelineConfig(
        output_dir=blocker,
        contract=_contract(),
        render=RenderConfig(),
        audio_locked=True,
        audio_track_path=tmp_path / "track.wav",
    )
    with pytest.raises(PipelineError, match="directorio de salida"):
        _ = _run(_images(tmp_path), config)


def test_backward_compatible_long_video_untouched() -> None:
    assert callable(orchestrator.run_long_video)
    assert orchestrator.run_long_video is not run_slideshow
