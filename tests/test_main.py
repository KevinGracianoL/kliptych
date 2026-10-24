"""Tests de la CLI."""

import json
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Protocol, cast, override
from uuid import UUID

import pytest

from kliptych import __version__
from kliptych.__main__ import main
from kliptych.contract import Band, Layout, LayoutSpec
from kliptych.environment import EnvironmentReport
from kliptych.studio.analyzer import AnalysisCancelled, AnalysisEngineError, AnalysisRequest
from kliptych.studio.models import (
    ANALYSIS_SCHEMA_VERSION,
    AnalysisDocument,
    AnalysisProvenance,
    AnalysisSource,
)
from kliptych.studio.preview import PreviewRequest, PreviewResult
from kliptych.studio.project_models import LayoutPosition
from kliptych.studio.project_store import ProjectStoreError


def _parse(text: str) -> dict[str, object]:
    return cast("dict[str, object]", json.loads(text))


def _disable_logging_setup(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_setup_logging(*, verbose: bool = False, quiet: bool = False) -> None:
        _ = verbose
        _ = quiet

    monkeypatch.setattr("kliptych.__main__.setup_logging", fake_setup_logging)


class _DownloaderProtocol(Protocol):
    def download_video(self, *, url: str, destination: Path) -> Path:
        """Download one source artifact."""
        ...


class _CancelledDownloader(_DownloaderProtocol):
    @override
    def download_video(self, *, url: str, destination: Path) -> Path:
        """Simulate cancellation at the download boundary.

        Raises:
            AnalysisCancelled: Always, to model a controlled cancellation.
        """
        _ = url
        _ = destination
        raise AnalysisCancelled


def test_console_script_is_installed() -> None:
    script = shutil.which("kliptych", path=str(Path(sys.executable).parent))
    assert script is not None
    completed = subprocess.run(
        [script, "--version"],
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    assert completed.returncode == 0
    assert completed.stdout.strip() == __version__


def test_main_without_args_prints_help(capsys: pytest.CaptureFixture[str]) -> None:
    assert main([]) == 0
    assert "usage" in capsys.readouterr().out


def test_main_version_flag_prints_version(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as excinfo:
        _ = main(["--version"])
    assert excinfo.value.code == 0
    assert __version__ in capsys.readouterr().out


def test_main_env_prints_environment(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    report = EnvironmentReport(ffmpeg_version="N-118380", nvenc_available=True)

    def fake_detect(_runner: object) -> EnvironmentReport:
        return report

    monkeypatch.setattr("kliptych.__main__.detect_environment", fake_detect)
    assert main(["env"]) == 0
    payload = _parse(capsys.readouterr().out)
    assert payload["ffmpeg_version"] == "N-118380"
    assert payload["nvenc_available"] is True


def test_main_analyze_emits_versioned_ndjson_and_completed_event(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    project_dir = tmp_path / "project"
    _disable_logging_setup(monkeypatch)

    def fake_analyze(
        request: AnalysisRequest,
        *,
        downloader: object,
        transcriber: object,
        progress: object,
    ) -> AnalysisDocument:
        _ = downloader
        _ = transcriber
        assert request.url == "https://example.com/video"
        assert request.project_dir == project_dir
        assert request.project_name == "My clip"
        assert callable(progress)
        _ = progress("downloading")
        _ = progress("transcribing")
        _ = progress("finding_candidates")
        return AnalysisDocument(
            schema_version=ANALYSIS_SCHEMA_VERSION,
            project_id="project-id",
            project_name="My clip",
            source=AnalysisSource(url=request.url, sha256="a" * 64),
            provenance=AnalysisProvenance(transcription_version="1", segmenter_version="1"),
            candidates=(),
        )

    monkeypatch.setattr("kliptych.__main__.analyze_project", fake_analyze)

    assert (
        main(
            [
                "--quiet",
                "analyze",
                "--url",
                "https://example.com/video",
                "--project-dir",
                str(project_dir),
                "--project-name",
                "My clip",
            ]
        )
        == 0
    )

    events = [
        cast("dict[str, object]", json.loads(line)) for line in capsys.readouterr().out.splitlines()
    ]
    run_ids = {event["run_id"] for event in events}
    assert len(run_ids) == 1
    assert [event["stage"] for event in events] == [
        "downloading",
        "transcribing",
        "finding_candidates",
        "completed",
    ]
    assert events[-1]["type"] == "completed"
    assert events[-1]["analysis_path"] == "analysis.json"
    assert events[-1]["candidate_count"] == 0


def test_main_analyze_emits_safe_error_event(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _disable_logging_setup(monkeypatch)

    def fake_analyze(
        _request: AnalysisRequest,
        *,
        downloader: object,
        transcriber: object,
        progress: object,
    ) -> AnalysisDocument:
        _ = downloader
        _ = transcriber
        _ = progress
        code = "transcript_empty"
        message = "Transcription produced no timestamped words."
        raise AnalysisEngineError(code, message)

    monkeypatch.setattr("kliptych.__main__.analyze_project", fake_analyze)

    assert (
        main(
            [
                "--quiet",
                "analyze",
                "--url",
                "https://example.com/video",
                "--project-dir",
                str(tmp_path / "project"),
                "--project-name",
                "My clip",
            ]
        )
        == 1
    )

    event = _parse(capsys.readouterr().out)
    assert event["type"] == "error"
    assert event["error_code"] == "transcript_empty"
    assert event["stage"] == "failed"


def test_main_analyze_returns_cancel_status_and_event(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _disable_logging_setup(monkeypatch)

    def fake_analyze(
        _request: AnalysisRequest,
        *,
        downloader: object,
        transcriber: object,
        progress: object,
    ) -> AnalysisDocument:
        _ = downloader
        _ = transcriber
        _ = progress
        raise AnalysisCancelled

    monkeypatch.setattr("kliptych.__main__.analyze_project", fake_analyze)

    assert (
        main(
            [
                "--quiet",
                "analyze",
                "--url",
                "https://example.com/video",
                "--project-dir",
                str(tmp_path / "project"),
                "--project-name",
                "My clip",
            ]
        )
        == 2
    )

    event = _parse(capsys.readouterr().out)
    assert event["type"] == "cancelled"
    assert event["stage"] == "cancelled"


def test_main_analyze_preserves_cancellation_from_download_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _disable_logging_setup(monkeypatch)

    monkeypatch.setattr("kliptych.__main__.MediaDownloader", _CancelledDownloader)

    assert (
        main(
            [
                "--quiet",
                "analyze",
                "--url",
                "https://example.com/video",
                "--project-dir",
                str(tmp_path / "project"),
                "--project-name",
                "My clip",
            ]
        )
        == 2
    )

    events = [
        cast("dict[str, object]", json.loads(line)) for line in capsys.readouterr().out.splitlines()
    ]
    assert events[-1]["type"] == "cancelled"
    assert not (tmp_path / "project").exists()


def test_main_analyze_translates_project_store_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _disable_logging_setup(monkeypatch)

    def fake_analyze(
        _request: AnalysisRequest,
        *,
        downloader: object,
        transcriber: object,
        progress: object,
    ) -> AnalysisDocument:
        _ = downloader
        _ = transcriber
        _ = progress
        code = "project_directory_invalid"
        message = "Project directory is not a completed Kliptych Studio project."
        raise ProjectStoreError(code, message)

    monkeypatch.setattr("kliptych.__main__.analyze_project", fake_analyze)

    assert (
        main(
            [
                "--quiet",
                "analyze",
                "--url",
                "https://example.com/video",
                "--project-dir",
                str(tmp_path / "project"),
                "--project-name",
                "My clip",
            ]
        )
        == 1
    )

    event = _parse(capsys.readouterr().out)
    assert event["error_code"] == "project_directory_invalid"


def test_analyze_argument_error_uses_status_one_not_cancellation() -> None:
    with pytest.raises(SystemExit) as excinfo:
        _ = main(["analyze", "--url", "https://example.com/video"])

    assert excinfo.value.code == 1


def test_other_command_with_analyze_as_argument_keeps_argparse_status_two() -> None:
    with pytest.raises(SystemExit) as excinfo:
        _ = main(["campaign", "analyze"])

    assert excinfo.value.code == 2


def test_analyze_console_invocation_error_uses_status_one() -> None:
    script = shutil.which("kliptych", path=str(Path(sys.executable).parent))
    assert script is not None

    completed = subprocess.run(
        [script, "analyze", "--url", "https://example.com/video"],
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )

    assert completed.returncode == 1
    assert not completed.stdout
    assert "--project-dir" in completed.stderr


def test_main_preview_emits_structured_completed_event(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _disable_logging_setup(monkeypatch)
    variant_id = "11111111-1111-1111-1111-111111111111"
    position_id = "22222222-2222-2222-2222-222222222222"

    def fake_preview(request: PreviewRequest, *, probe: object, runner: object) -> PreviewResult:
        _ = probe
        _ = runner
        assert request.at_ms == 15_000
        return PreviewResult(
            preview_path=tmp_path / "previews" / variant_id / f"{'a' * 64}.jpg",
            source_frame_path=tmp_path / "previews" / variant_id / f"{'a' * 64}-source.jpg",
            preview_fingerprint="a" * 64,
            frame_time_ms=15_000,
            width=720,
            height=1280,
            layout_position=LayoutPosition(
                position_id=UUID(position_id),
                start_ms=10_000,
                end_ms=20_000,
                layout_spec=LayoutSpec(layout=Layout.VERTICAL, bands=(Band(source="source.mp4"),)),
            ),
        )

    monkeypatch.setattr("kliptych.__main__.generate_preview", fake_preview)

    assert (
        main(
            [
                "--quiet",
                "preview",
                "--project-dir",
                str(tmp_path),
                "--variant-id",
                variant_id,
                "--position-id",
                position_id,
                "--at-ms",
                "15000",
            ]
        )
        == 0
    )

    event = _parse(capsys.readouterr().out)
    assert event["type"] == "completed"
    assert event["preview_path"] == f"previews/{variant_id}/{'a' * 64}.jpg"
    assert event["source_frame_path"] == f"previews/{variant_id}/{'a' * 64}-source.jpg"
    assert event["width"] == 720
    assert event["height"] == 1280


def test_main_preview_returns_error_for_invalid_uuid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _disable_logging_setup(monkeypatch)

    assert (
        main(
            [
                "--quiet",
                "preview",
                "--project-dir",
                str(tmp_path),
                "--variant-id",
                "not-a-uuid",
                "--position-id",
                "22222222-2222-2222-2222-222222222222",
                "--at-ms",
                "15000",
            ]
        )
        == 1
    )
    event = _parse(capsys.readouterr().out)
    assert event["type"] == "error"
    assert event["error_code"] == "preview_request_invalid"


def test_main_studio_inspect_emits_project_candidates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _disable_logging_setup(monkeypatch)
    analysis = AnalysisDocument(
        schema_version=ANALYSIS_SCHEMA_VERSION,
        project_id="opaque-project",
        project_name="Mi clip",
        source=AnalysisSource(url="https://example.com/video", sha256="a" * 64),
        provenance=AnalysisProvenance(transcription_version="1", segmenter_version="1"),
        candidates=(),
    )

    class _Store:
        existing_analysis: AnalysisDocument = analysis

        @staticmethod
        def read_project() -> None:
            return None

    def open_store(_path: Path) -> _Store:
        return _Store()

    monkeypatch.setattr("kliptych.__main__.open_project_store", open_store)

    assert main(["--quiet", "studio", "inspect", "--project-dir", str(tmp_path)]) == 0

    event = _parse(capsys.readouterr().out)
    assert event["type"] == "project"
    assert event["project_name"] == "Mi clip"
    assert event["editable_state"] == "missing"


def test_main_studio_initialize_uses_project_store_and_ffprobe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _disable_logging_setup(monkeypatch)
    project = cast(
        "object", SimpleNamespace(project_id="opaque-project", source_duration_ms=60_000)
    )

    class _Store:
        @staticmethod
        def initialize_project(*, probe: object) -> object:
            assert probe.__class__.__name__ == "FFprobeProbe"
            return project

    def open_store(_path: Path) -> _Store:
        return _Store()

    monkeypatch.setattr("kliptych.__main__.open_project_store", open_store)

    assert main(["--quiet", "studio", "initialize", "--project-dir", str(tmp_path)]) == 0

    event = _parse(capsys.readouterr().out)
    assert event["type"] == "initialized"
    assert event["project_id"] == "opaque-project"
    assert event["source_duration_ms"] == 60_000


def test_main_studio_create_variant_emits_variant(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _disable_logging_setup(monkeypatch)

    class _Variant:
        variant_id: str = "11111111-1111-1111-1111-111111111111"

        @staticmethod
        def model_dump(**_kwargs: object) -> dict[str, str]:
            return {"variant_id": "id"}

    variant = cast("object", _Variant())
    project = cast("object", SimpleNamespace(variants=(variant,)))

    class _Store:
        @staticmethod
        def create_variant(*, candidate_id: str, name: str) -> object:
            assert candidate_id == "candidate"
            assert name == "Variante 1"
            return project

    def open_store(_path: Path) -> _Store:
        return _Store()

    monkeypatch.setattr("kliptych.__main__.open_project_store", open_store)

    assert (
        main(
            [
                "--quiet",
                "studio",
                "create-variant",
                "--project-dir",
                str(tmp_path),
                "--candidate-id",
                "candidate",
                "--name",
                "Variante 1",
            ]
        )
        == 0
    )

    event = _parse(capsys.readouterr().out)
    assert event == {
        "schema_version": ANALYSIS_SCHEMA_VERSION,
        "type": "variant_created",
        "variant": {"variant_id": "id"},
    }


def test_main_studio_update_position_emits_variant(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _disable_logging_setup(monkeypatch)

    class _Variant:
        variant_id: str = "11111111-1111-1111-1111-111111111111"

        @staticmethod
        def model_dump(**_kwargs: object) -> dict[str, str]:
            return {"variant_id": "id"}

    variant = cast("object", _Variant())
    project = cast("object", SimpleNamespace(variants=(variant,)))

    class _Store:
        @staticmethod
        def update_position(
            *, variant_id: object, position_id: object, layout_spec: object
        ) -> object:
            assert str(variant_id)
            assert str(position_id)
            assert layout_spec is not None
            return project

    def open_store(_path: Path) -> _Store:
        return _Store()

    monkeypatch.setattr("kliptych.__main__.open_project_store", open_store)

    layout = (
        '{"layout":"vertical","bands":[{"source":"source.mp4",'
        '"source_rect":{"x":0,"y":0,"width":1,"height":1}}]}'
    )
    assert (
        main(
            [
                "--quiet",
                "studio",
                "update-position",
                "--project-dir",
                str(tmp_path),
                "--variant-id",
                "11111111-1111-1111-1111-111111111111",
                "--position-id",
                "22222222-2222-2222-2222-222222222222",
                "--layout-json",
                layout,
            ]
        )
        == 0
    )

    event = _parse(capsys.readouterr().out)
    assert event["type"] == "position_updated"


def test_main_studio_split_position_emits_variant(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _disable_logging_setup(monkeypatch)

    class _Variant:
        variant_id: str = "11111111-1111-1111-1111-111111111111"

        @staticmethod
        def model_dump(**_kwargs: object) -> dict[str, str]:
            return {"variant_id": "id"}

    variant = cast("object", _Variant())
    project = cast("object", SimpleNamespace(variants=(variant,)))

    class _Store:
        @staticmethod
        def split_position(*, variant_id: object, position_id: object, at_ms: int) -> object:
            assert variant_id is not None
            assert position_id is not None
            assert at_ms >= 0
            return project

    def open_store(_path: Path) -> _Store:
        return _Store()

    monkeypatch.setattr("kliptych.__main__.open_project_store", open_store)

    assert (
        main(
            [
                "--quiet",
                "studio",
                "split-position",
                "--project-dir",
                str(tmp_path),
                "--variant-id",
                "11111111-1111-1111-1111-111111111111",
                "--position-id",
                "22222222-2222-2222-2222-222222222222",
                "--at-ms",
                "15000",
            ]
        )
        == 0
    )
    event = _parse(capsys.readouterr().out)
    assert event["type"] == "position_split"


def test_main_studio_delete_position_emits_variant(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _disable_logging_setup(monkeypatch)

    class _Variant:
        variant_id: str = "11111111-1111-1111-1111-111111111111"

        @staticmethod
        def model_dump(**_kwargs: object) -> dict[str, str]:
            return {"variant_id": "id"}

    variant = cast("object", _Variant())
    project = cast("object", SimpleNamespace(variants=(variant,)))

    class _Store:
        @staticmethod
        def delete_position(
            *, variant_id: object, position_id: object, absorb_neighbor: object
        ) -> object:
            assert variant_id is not None
            assert position_id is not None
            _ = absorb_neighbor
            return project

    def open_store(_path: Path) -> _Store:
        return _Store()

    monkeypatch.setattr("kliptych.__main__.open_project_store", open_store)

    assert (
        main(
            [
                "--quiet",
                "studio",
                "delete-position",
                "--project-dir",
                str(tmp_path),
                "--variant-id",
                "11111111-1111-1111-1111-111111111111",
                "--position-id",
                "22222222-2222-2222-2222-222222222222",
                "--absorb",
                "previous",
            ]
        )
        == 0
    )
    event = _parse(capsys.readouterr().out)
    assert event["type"] == "position_deleted"


def test_main_studio_duplicate_variant_emits_variant(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _disable_logging_setup(monkeypatch)

    class _Variant:
        variant_id: str = "11111111-1111-1111-1111-111111111111"

        @staticmethod
        def model_dump(**_kwargs: object) -> dict[str, str]:
            return {"variant_id": "id"}

    variant = cast("object", _Variant())
    project = cast("object", SimpleNamespace(variants=(variant,)))

    class _Store:
        @staticmethod
        def duplicate_variant(*, variant_id: object, name: str) -> object:
            assert variant_id is not None
            assert name
            return project

    def open_store(_path: Path) -> _Store:
        return _Store()

    monkeypatch.setattr("kliptych.__main__.open_project_store", open_store)

    assert (
        main(
            [
                "--quiet",
                "studio",
                "duplicate-variant",
                "--project-dir",
                str(tmp_path),
                "--variant-id",
                "11111111-1111-1111-1111-111111111111",
                "--name",
                "Copia",
            ]
        )
        == 0
    )
    event = _parse(capsys.readouterr().out)
    assert event["type"] == "variant_duplicated"


def test_main_studio_import_watermark_emits_variant(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _disable_logging_setup(monkeypatch)

    class _Variant:
        variant_id: str = "11111111-1111-1111-1111-111111111111"

        @staticmethod
        def model_dump(**_kwargs: object) -> dict[str, str]:
            return {"variant_id": "id"}

    variant = cast("object", _Variant())
    project = cast("object", SimpleNamespace(variants=(variant,)))

    class _Store:
        @staticmethod
        def import_watermark(
            *, variant_id: object, source_path: object, position: object
        ) -> object:
            assert variant_id is not None
            assert source_path is not None
            _ = position
            return project

    def open_store(_path: Path) -> _Store:
        return _Store()

    monkeypatch.setattr("kliptych.__main__.open_project_store", open_store)

    png = tmp_path / "logo.png"
    _ = png.write_bytes(b"x")
    assert (
        main(
            [
                "--quiet",
                "studio",
                "import-watermark",
                "--project-dir",
                str(tmp_path),
                "--variant-id",
                "11111111-1111-1111-1111-111111111111",
                "--source",
                str(png),
                "--position",
                "bottom_right",
            ]
        )
        == 0
    )
    event = _parse(capsys.readouterr().out)
    assert event["type"] == "watermark_imported"


def test_main_studio_set_watermark_emits_variant(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _disable_logging_setup(monkeypatch)

    class _Variant:
        variant_id: str = "11111111-1111-1111-1111-111111111111"

        @staticmethod
        def model_dump(**_kwargs: object) -> dict[str, str]:
            return {"variant_id": "id"}

    variant = cast("object", _Variant())
    project = cast("object", SimpleNamespace(variants=(variant,)))

    class _Store:
        @staticmethod
        def set_watermark(
            *, variant_id: object, enabled: object = None, position: object = None
        ) -> object:
            assert variant_id is not None
            _ = (enabled, position)
            return project

    def open_store(_path: Path) -> _Store:
        return _Store()

    monkeypatch.setattr("kliptych.__main__.open_project_store", open_store)

    assert (
        main(
            [
                "--quiet",
                "studio",
                "set-watermark",
                "--project-dir",
                str(tmp_path),
                "--variant-id",
                "11111111-1111-1111-1111-111111111111",
                "--enabled",
                "true",
                "--position",
                "top_left",
            ]
        )
        == 0
    )
    event = _parse(capsys.readouterr().out)
    assert event["type"] == "watermark_updated"


def test_main_studio_set_captions_emits_variant(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _disable_logging_setup(monkeypatch)

    class _Variant:
        variant_id: str = "11111111-1111-1111-1111-111111111111"

        @staticmethod
        def model_dump(**_kwargs: object) -> dict[str, str]:
            return {"variant_id": "id"}

    variant = cast("object", _Variant())
    project = cast("object", SimpleNamespace(variants=(variant,)))

    class _Store:
        @staticmethod
        def set_captions(
            *,
            variant_id: object,
            enabled: object = None,
            highlight_color: object = None,
            capitalization: object = None,
        ) -> object:
            assert variant_id is not None
            _ = (enabled, highlight_color, capitalization)
            return project

    def open_store(_path: Path) -> _Store:
        return _Store()

    monkeypatch.setattr("kliptych.__main__.open_project_store", open_store)

    assert (
        main(
            [
                "--quiet",
                "studio",
                "set-captions",
                "--project-dir",
                str(tmp_path),
                "--variant-id",
                "11111111-1111-1111-1111-111111111111",
                "--enabled",
                "true",
                "--highlight-color",
                "&H00FF0000",
                "--capitalization",
                "uppercase",
            ]
        )
        == 0
    )
    event = _parse(capsys.readouterr().out)
    assert event["type"] == "captions_updated"


def test_main_studio_set_caption_correction_emits_variant(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _disable_logging_setup(monkeypatch)

    class _Variant:
        variant_id: str = "11111111-1111-1111-1111-111111111111"

        @staticmethod
        def model_dump(**_kwargs: object) -> dict[str, str]:
            return {"variant_id": "id"}

    variant = cast("object", _Variant())
    project = cast("object", SimpleNamespace(variants=(variant,)))

    class _Store:
        @staticmethod
        def set_caption_correction(
            *,
            variant_id: object,
            interval: object,
            corrected_text: str,
            author: str,
            created_at_ms: int,
        ) -> object:
            assert variant_id is not None
            _ = (interval, corrected_text, author, created_at_ms)
            return project

    def open_store(_path: Path) -> _Store:
        return _Store()

    monkeypatch.setattr("kliptych.__main__.open_project_store", open_store)

    assert (
        main(
            [
                "--quiet",
                "studio",
                "set-caption-correction",
                "--project-dir",
                str(tmp_path),
                "--variant-id",
                "11111111-1111-1111-1111-111111111111",
                "--start-ms",
                "1000",
                "--end-ms",
                "2000",
                "--text",
                "Texto corregido",
                "--author",
                "user",
                "--created-at-ms",
                "5000",
            ]
        )
        == 0
    )
    event = _parse(capsys.readouterr().out)
    assert event["type"] == "caption_correction_set"


def test_main_studio_clear_caption_correction_emits_variant(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _disable_logging_setup(monkeypatch)

    class _Variant:
        variant_id: str = "11111111-1111-1111-1111-111111111111"

        @staticmethod
        def model_dump(**_kwargs: object) -> dict[str, str]:
            return {"variant_id": "id"}

    variant = cast("object", _Variant())
    project = cast("object", SimpleNamespace(variants=(variant,)))

    class _Store:
        @staticmethod
        def clear_caption_correction(*, variant_id: object, correction_id: object) -> object:
            assert variant_id is not None
            assert correction_id is not None
            return project

    def open_store(_path: Path) -> _Store:
        return _Store()

    monkeypatch.setattr("kliptych.__main__.open_project_store", open_store)

    assert (
        main(
            [
                "--quiet",
                "studio",
                "clear-caption-correction",
                "--project-dir",
                str(tmp_path),
                "--variant-id",
                "11111111-1111-1111-1111-111111111111",
                "--correction-id",
                "22222222-2222-2222-2222-222222222222",
            ]
        )
        == 0
    )
    event = _parse(capsys.readouterr().out)
    assert event["type"] == "caption_correction_cleared"
