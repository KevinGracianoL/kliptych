"""Fix 3: revalidación remota fail-closed ante fallo de red."""

from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, cast

import pytest

from kliptych import orchestrator
from kliptych.orchestrator import PipelineError

if TYPE_CHECKING:
    from kliptych.download import MediaDownloader


def _private(name: str) -> Callable[..., Path]:
    return cast("Callable[..., Path]", cast("object", getattr(orchestrator, name)))


class _FailingDownloader:
    def download_video(
        self, *, url: str, destination: Path, format_selector: str | None = None
    ) -> Path:
        _ = (self, url, destination, format_selector)
        msg = "red inestable"
        raise OSError(msg)


class _Registry:
    def __init__(self) -> None:
        self.paths: list[Path] = []

    def register(self, path: Path, *, is_artifact: bool = False) -> Path:
        _ = is_artifact
        self.paths.append(path)
        return path


def test_revalidate_remote_failure_raises_pipeline_error(tmp_path: Path) -> None:
    source_path = tmp_path / "source.mp4"
    _ = source_path.write_bytes(b"old bytes")
    registry = _Registry()
    revalidate = _private("_revalidate_remote_source")
    downloader = cast("MediaDownloader", cast("object", _FailingDownloader()))
    registry_cast = cast("object", registry)
    with pytest.raises(PipelineError, match="revalid"):
        _ = revalidate(
            "https://example.com/video",
            source_path,
            downloader=downloader,
            registry=registry_cast,
        )
    assert source_path.read_bytes() == b"old bytes"
