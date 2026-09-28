"""Tests del orquestador given_clips: PASS, REJECTED, contratos no resolubles y CLI."""

import json
import os
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import cast, override

import pytest

from kliptych.__main__ import main
from kliptych.assembler import AssembleError, FFmpegAssembler
from kliptych.assets import AssetRegistry
from kliptych.config import Settings
from kliptych.contract import Contract, ContractDraft, Watermark
from kliptych.environment import EnvironmentReport
from kliptych.exporter import ExportError, ExportStatus
from kliptych.gate import Gate
from kliptych.pipeline import (
    PieceAssembler,
    PipelineError,
    RunOutcome,
    RunRequest,
    RunResult,
    run_given_clips,
)
from kliptych.runtime import (
    CAPTION_PROMPT_VERSION,
    PROMPT_VERSION,
    CampaignModel,
    Caption,
    ModelError,
    PieceContext,
    RecordedModel,
    record_response,
)
from tests.support import (
    FakeProbe,
    candidate,
    make_asset_draft,
    make_draft,
    make_media,
    mock_silent_volumedetect,
)

_REPO = Path(__file__).resolve().parents[1]
_FIXTURES = _REPO / "campaigns" / "fixtures"
_FFMPEG = shutil.which("ffmpeg")
_FFPROBE = shutil.which("ffprobe")

_NEEDS_FFMPEG = pytest.mark.skipif(
    _FFMPEG is None or _FFPROBE is None,
    reason="ffmpeg/ffprobe no disponibles",
)


def _parse(text: str) -> dict[str, object]:
    return cast("dict[str, object]", json.loads(text))


def _json(path: Path) -> dict[str, object]:
    return cast("dict[str, object]", json.loads(path.read_text(encoding="utf-8")))


def _generate_sample(root: Path, *, duration: int = 9, audio: bool = True) -> Path:
    assert _FFMPEG is not None
    path = root / "assets" / "samples" / "given-clips-sample.mp4"
    path.parent.mkdir(parents=True, exist_ok=True)
    argv = [
        _FFMPEG,
        "-y",
        "-v",
        "error",
        "-f",
        "lavfi",
        "-i",
        f"color=c=blue:s=320x240:d={duration}",
    ]
    if audio:
        argv += [
            "-f",
            "lavfi",
            "-i",
            f"sine=frequency=440:duration={duration}",
            "-shortest",
        ]
    argv += ["-c:v", "libx264", "-pix_fmt", "yuv420p"]
    if audio:
        argv += ["-c:a", "aac"]
    argv += ["-movflags", "+faststart", str(path)]
    _ = subprocess.run(argv, capture_output=True, check=True, timeout=120)
    return path


def _model(fixture: str) -> RecordedModel:
    return RecordedModel.from_directory(
        _FIXTURES / fixture / "recorded",
        expected_prompt_version=PROMPT_VERSION,
        expected_caption_prompt_version=CAPTION_PROMPT_VERSION,
    )


def _request(fixture: str, root: Path, *, approve_manual_review: bool = False) -> RunRequest:
    brief = (_FIXTURES / fixture / "brief.md").read_text(encoding="utf-8")
    return RunRequest(
        brief=brief,
        destination=root / "delivery",
        environment=EnvironmentReport(ffmpeg_version="test", ffprobe_version="test"),
        model_version=RecordedModel.model_version,
        prompt_version=PROMPT_VERSION,
        caption_prompt_version=CAPTION_PROMPT_VERSION,
        approve_manual_review=approve_manual_review,
        approved_by="auditor-fixture" if approve_manual_review else None,
    )


def _run(fixture: str, root: Path, *, approve_manual_review: bool = False) -> RunResult:
    return run_given_clips(
        model=_model(fixture),
        settings=Settings.from_root(root),
        request=_request(fixture, root, approve_manual_review=approve_manual_review),
    )


class _StaticModel(CampaignModel):
    """Modelo de prueba con draft fijo y caption opcional."""

    model_version: str = "static"

    def __init__(self, draft: ContractDraft, caption: Caption | None = None) -> None:
        self._draft: ContractDraft = draft
        self._caption: Caption | None = caption

    @override
    def extract_contract(self, brief: str) -> ContractDraft:
        _ = brief
        return self._draft

    @override
    def write_caption(self, contract: Contract, piece: PieceContext) -> Caption:
        _ = contract
        _ = piece
        if self._caption is None:
            msg = "no debe redactar captions en este test"
            raise AssertionError(msg)
        return self._caption


class _StubAssembler(PieceAssembler):
    """Ensamblador de prueba: escribe bytes y registra el watermark recibido."""

    def __init__(self) -> None:
        self.watermarks: list[Path | None] = []
        self.mutes: list[bool] = []

    @override
    def assemble(
        self,
        *,
        clip: Path,
        destination: Path,
        watermark: Path | None,
        watermark_config: Watermark | None = None,
        mute_audio: bool = False,
    ) -> Path:
        _ = clip
        self.watermarks.append(watermark)
        self.mutes.append(mute_audio)
        destination.parent.mkdir(parents=True, exist_ok=True)
        _ = destination.write_bytes(b"video")
        return destination

    @override
    def render_arguments(
        self,
        *,
        clip: Path,
        destination: Path,
        watermark: Path | None,
        watermark_config: Watermark | None = None,
        mute_audio: bool = False,
    ) -> tuple[str, ...]:
        _ = (watermark, watermark_config, mute_audio)
        return ("ffmpeg", str(clip), str(destination))


def _caption() -> Caption:
    return Caption(caption="mira @marca #marca", hashtags=("#marca",))


def _platform_draft() -> dict[str, object]:
    return {
        "duration": {"min_s": candidate(8), "max_s": candidate(60)},
        "required_hashtags": candidate(["#marca"]),
        "required_mentions": candidate(["@marca"]),
    }


def _asset_draft(asset_id: str, kind: str, uri: str) -> dict[str, object]:
    return {
        "asset_id": candidate(asset_id),
        "kind": candidate(kind),
        "uri": candidate(uri),
        "origin": candidate("brief"),
    }


def _unit_request(
    root: Path,
    *,
    brief: str = "cita del brief",
    assembler: PieceAssembler | None = None,
    gate: Gate | None = None,
    registry: AssetRegistry | None = None,
    run_id: str | None = None,
    brief_path: Path | None = None,
    approve_manual_review: bool = False,
    approved_by: str | None = None,
) -> RunRequest:
    content = brief if brief.startswith("cita del brief") else f"cita del brief\n{brief}"
    return RunRequest(
        brief=content,
        destination=root / "delivery",
        environment=EnvironmentReport(),
        model_version="static",
        prompt_version=PROMPT_VERSION,
        caption_prompt_version=CAPTION_PROMPT_VERSION,
        run_id=run_id,
        assembler=assembler,
        gate=gate,
        registry=registry,
        brief_path=brief_path,
        approve_manual_review=approve_manual_review,
        approved_by=approved_by,
    )


@pytest.mark.integration
@_NEEDS_FFMPEG
def test_pipeline_exports_pass_package(tmp_path: Path) -> None:
    _ = _generate_sample(tmp_path)
    result = _run("given-clips", tmp_path, approve_manual_review=True)
    assert result.outcome is RunOutcome.EXPORTED
    assert result.delivery is not None
    assert result.delivery.status is ExportStatus.EXPORTED
    assert result.delivery.manually_approved_rules == ("audio.own_clip",)
    package = Path(cast("str", result.package_path))
    artifact = package / "demo-given-clips" / "tiktok" / "clip-01.mp4"
    assert artifact.is_file()
    metadata = _json(package / "demo-given-clips" / "tiktok" / "clip-01.metadata.json")
    assert "@marca" in cast("str", metadata["caption"])
    assert metadata["hashtags"] == ["#marca"]
    manifest = _json(Path(result.manifest_path))
    gates = cast("list[dict[str, object]]", manifest["gates"])
    assert gates[0]["status"] == "pending_review"
    assert manifest["caption_prompt_version"] == CAPTION_PROMPT_VERSION
    outputs = cast("list[dict[str, object]]", manifest["outputs"])
    assert outputs[0]["sha256"] == metadata["artifact_sha256"]
    recipe = FFmpegAssembler().render_arguments(
        clip=tmp_path / "assets" / "samples" / "given-clips-sample.mp4",
        destination=Path(result.manifest_path).parent / "artifacts" / "tiktok" / "clip-01.mp4",
        watermark=None,
    )
    assert outputs[0]["render_arguments"] == list(recipe)


def _assert_rejected_check(result: RunResult, check_id: str, *, status: str = "fail") -> None:
    assert result.delivery is not None
    rejected = result.delivery.rejected[0]
    assert check_id in rejected.reason
    manifest = _json(Path(result.manifest_path))
    gates = cast("list[dict[str, object]]", manifest["gates"])
    checks = cast("list[dict[str, object]]", gates[0]["checks"])
    matched = [check for check in checks if check["id"] == check_id]
    assert matched
    assert matched[0]["status"] == status
    package = Path(cast("str", result.package_path))
    assert not (package / "demo-given-clips" / "tiktok" / "clip-01.mp4").exists()


@pytest.mark.integration
@_NEEDS_FFMPEG
@pytest.mark.parametrize(
    ("fixture", "check_id"),
    [
        ("given-clips-rejected", "caption.required_mention"),
        ("given-clips-rejected-hashtag", "caption.required_hashtag"),
    ],
)
def test_pipeline_blocks_rejected_caption(tmp_path: Path, fixture: str, check_id: str) -> None:
    _ = _generate_sample(tmp_path)
    result = _run(fixture, tmp_path)
    assert result.outcome is RunOutcome.BLOCKED
    _assert_rejected_check(result, check_id)


@pytest.mark.integration
@_NEEDS_FFMPEG
def test_pipeline_blocks_short_artifact(tmp_path: Path) -> None:
    _ = _generate_sample(tmp_path, duration=3)
    result = _run("given-clips", tmp_path)
    assert result.outcome is RunOutcome.BLOCKED
    _assert_rejected_check(result, "duration.min")


@pytest.mark.integration
@_NEEDS_FFMPEG
def test_pipeline_blocks_artifact_without_audio(tmp_path: Path) -> None:
    _ = _generate_sample(tmp_path, audio=False)
    result = _run("given-clips", tmp_path)
    assert result.outcome is RunOutcome.BLOCKED
    _assert_rejected_check(result, "audio.present")


def test_given_clips_mutes_audio_for_internal_official_sound(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # El artefacto es falso (bytes, no MP4): se simula la medición de silencio.
    mock_silent_volumedetect(monkeypatch)
    _ = (tmp_path / "clip.mp4").write_bytes(b"clip")
    draft = make_draft(
        audio_policy=candidate("internal_official_sound"),
        assets={"required": [make_asset_draft()], "optional": []},
    )
    assembler = _StubAssembler()
    request = _unit_request(
        tmp_path,
        brief="brief con sonido oficial interno",
        assembler=assembler,
        gate=Gate(FakeProbe(info=make_media())),
        run_id="run-internal-sound",
    )
    model = _StaticModel(draft, _caption())
    blocked = run_given_clips(model=model, settings=Settings.from_root(tmp_path), request=request)
    assert assembler.mutes == [True]
    assert blocked.delivery is not None
    assert blocked.delivery.status is ExportStatus.BLOCKED
    manifest = _json(Path(blocked.manifest_path))
    gates = cast("list[dict[str, object]]", manifest["gates"])
    assert gates[0]["status"] == "pending_review"
    checks = cast("list[dict[str, object]]", gates[0]["checks"])
    policy = [check for check in checks if check["id"] == "audio.policy"]
    assert len(policy) == 1
    assert policy[0]["status"] == "manual_review"

    approved_request = _unit_request(
        tmp_path,
        brief="brief con sonido oficial interno",
        assembler=_StubAssembler(),
        gate=Gate(FakeProbe(info=make_media())),
        run_id="run-internal-sound-approved",
        approve_manual_review=True,
        approved_by="operador-01",
    )
    approved = run_given_clips(
        model=_StaticModel(draft, _caption()),
        settings=Settings.from_root(tmp_path),
        request=approved_request,
    )
    assert approved.outcome is RunOutcome.EXPORTED
    assert approved.delivery is not None
    assert approved.delivery.status is ExportStatus.EXPORTED
    assert approved.delivery.manually_approved_rules == ("audio.policy",)
    assert approved.delivery.approved_by == "operador-01"


def test_pipeline_requires_video_assets(tmp_path: Path) -> None:
    model = _StaticModel(make_draft())
    request = RunRequest(
        brief="cita del brief\nbrief sin clips",
        destination=tmp_path / "delivery",
        environment=EnvironmentReport(),
        model_version="static",
        prompt_version=PROMPT_VERSION,
        caption_prompt_version=CAPTION_PROMPT_VERSION,
    )
    with pytest.raises(PipelineError, match="clips de video"):
        _ = run_given_clips(model=model, settings=Settings.from_root(tmp_path), request=request)


def test_pipeline_reports_new_archetype(tmp_path: Path) -> None:
    model = _StaticModel(make_draft(mode={"confidence": "missing"}))
    request = RunRequest(
        brief="cita del brief\nbrief sin modo",
        destination=tmp_path / "delivery",
        environment=EnvironmentReport(),
        model_version="static",
        prompt_version=PROMPT_VERSION,
        caption_prompt_version=CAPTION_PROMPT_VERSION,
    )
    result = run_given_clips(model=model, settings=Settings.from_root(tmp_path), request=request)
    assert result.outcome is RunOutcome.NEW_ARCHETYPE
    assert result.delivery is None
    assert result.package_path is None
    assert result.issues
    manifest = _json(Path(result.manifest_path))
    assert manifest["contract_sha256"] is None
    assert manifest["gates"] == []


@pytest.mark.integration
@_NEEDS_FFMPEG
def test_cli_run_exports_package(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _ = _generate_sample(tmp_path)
    code = main(
        [
            "run",
            str(_FIXTURES / "given-clips" / "brief.md"),
            "--out",
            str(tmp_path / "delivery"),
            "--recorded",
            str(_FIXTURES / "given-clips" / "recorded"),
            "--root",
            str(tmp_path),
            "--approve-manual-review",
            "--approved-by",
            "auditor-fixture",
        ]
    )
    assert code == 0
    payload = _parse(capsys.readouterr().out)
    assert payload["outcome"] == "exported"
    assert payload["exported"] == ["clip-01"]
    assert payload["rejected"] == []


@pytest.mark.integration
@_NEEDS_FFMPEG
def test_cli_run_reports_rejected_package(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _ = _generate_sample(tmp_path)
    code = main(
        [
            "run",
            str(_FIXTURES / "given-clips-rejected" / "brief.md"),
            "--out",
            str(tmp_path / "delivery"),
            "--recorded",
            str(_FIXTURES / "given-clips-rejected" / "recorded"),
            "--root",
            str(tmp_path),
        ]
    )
    assert code == 1
    payload = _parse(capsys.readouterr().out)
    assert payload["outcome"] == "blocked"
    rejected = cast("list[dict[str, object]]", payload["rejected"])
    assert rejected[0]["piece_id"] == "clip-01"
    assert "caption.required_mention" in cast("str", rejected[0]["reason"])


def test_pipeline_reports_manual_review(tmp_path: Path) -> None:
    model = _StaticModel(make_draft(campaign_id={"confidence": "missing"}))
    request = _unit_request(tmp_path, brief="brief sin campaign_id")
    result = run_given_clips(model=model, settings=Settings.from_root(tmp_path), request=request)
    assert result.outcome is RunOutcome.MANUAL_REVIEW
    assert result.delivery is None
    assert result.package_path is None
    assert result.issues
    assert _json(Path(result.manifest_path))["contract_sha256"] is None


def test_pipeline_reports_unsupported_mode(tmp_path: Path) -> None:
    _ = (tmp_path / "clip.mp4").write_bytes(b"clip")
    draft = make_draft(
        mode=candidate("long_video"),
        assets={"required": [make_asset_draft()], "optional": []},
    )
    model = _StaticModel(draft, caption=_caption())
    request = _unit_request(
        tmp_path,
        brief="brief long_video",
        assembler=_StubAssembler(),
        gate=Gate(FakeProbe(info=make_media())),
        run_id="run-longvideo",
    )
    result = run_given_clips(model=model, settings=Settings.from_root(tmp_path), request=request)
    assert result.outcome is RunOutcome.MANUAL_REVIEW
    assert result.delivery is None
    assert result.package_path is None
    assert any(code == "missing_required" for code in (issue.code.value for issue in result.issues))
    assert not (tmp_path / "delivery").exists()
    manifest = _json(Path(result.manifest_path))
    assert manifest["contract_sha256"] is None


def test_pipeline_rejects_unsafe_asset_id(tmp_path: Path) -> None:
    _ = (tmp_path / "clip.mp4").write_bytes(b"clip")
    draft = make_draft(
        assets={"required": [make_asset_draft(asset_id="../escape")], "optional": []}
    )
    model = _StaticModel(draft)
    request = _unit_request(tmp_path, brief="brief con asset_id hostil")
    result = run_given_clips(model=model, settings=Settings.from_root(tmp_path), request=request)
    assert result.outcome is RunOutcome.MANUAL_REVIEW
    assert result.package_path is None
    assert list(tmp_path.rglob("escape*")) == []


def test_pipeline_exports_piece_per_platform(tmp_path: Path) -> None:
    _ = (tmp_path / "clip.mp4").write_bytes(b"clip")
    _ = (tmp_path / "logo.png").write_bytes(b"logo")
    draft = make_draft(
        platforms={"tiktok": _platform_draft(), "instagram_reels": _platform_draft()},
        assets={
            "required": [make_asset_draft(), _asset_draft("logo", "image", "logo.png")],
            "optional": [],
        },
    )
    model = _StaticModel(draft, _caption())
    request = _unit_request(
        tmp_path,
        brief="brief multiplataforma",
        assembler=_StubAssembler(),
        gate=Gate(FakeProbe(info=make_media())),
    )
    result = run_given_clips(model=model, settings=Settings.from_root(tmp_path), request=request)
    assert result.outcome is RunOutcome.EXPORTED
    assert result.delivery is not None
    exported = {(piece.platform.value, piece.piece_id) for piece in result.delivery.exported}
    assert exported == {("tiktok", "clip-01"), ("instagram_reels", "clip-01")}
    package = Path(cast("str", result.package_path))
    assert (package / "camp-01" / "tiktok" / "clip-01.mp4").is_file()
    assert (package / "camp-01" / "instagram_reels" / "clip-01.mp4").is_file()
    run_dir = Path(result.manifest_path).parent
    outputs = cast("list[dict[str, object]]", _json(Path(result.manifest_path))["outputs"])
    assert len(outputs) == 2
    for output in outputs:
        expected_destination = run_dir / cast("str", output["path"])
        assert output["render_arguments"] == [
            "ffmpeg",
            str(tmp_path / "clip.mp4"),
            str(expected_destination),
        ]


def test_pipeline_writes_manifest_when_export_fails(tmp_path: Path) -> None:
    _ = (tmp_path / "clip.mp4").write_bytes(b"clip")
    draft = make_draft(assets={"required": [make_asset_draft()], "optional": []})
    model = _StaticModel(draft, _caption())
    blocked = tmp_path / "delivery"
    _ = blocked.write_bytes(b"soy un archivo")
    request = _unit_request(
        tmp_path,
        brief="brief con export fallido",
        assembler=_StubAssembler(),
        gate=Gate(FakeProbe(info=make_media())),
        run_id="run-export-fail",
    )
    with pytest.raises(ExportError):
        _ = run_given_clips(model=model, settings=Settings.from_root(tmp_path), request=request)
    run_dir = tmp_path / "runs" / "run-export-fail"
    assert (run_dir / "artifacts" / "tiktok" / "clip-01.mp4").is_file()
    manifest = _json(run_dir / "run_manifest.json")
    gates = cast("list[dict[str, object]]", manifest["gates"])
    assert gates[0]["status"] == "passed"


def _watermark_draft() -> ContractDraft:
    return make_draft(
        watermark={
            "required": candidate(value=True),
            "asset_id": candidate("wm"),
            "visible_full_video": candidate(value=True),
        },
        assets={"required": [make_asset_draft()], "optional": []},
    )


def test_pipeline_passes_registered_watermark_to_assembler(tmp_path: Path) -> None:
    _ = (tmp_path / "clip.mp4").write_bytes(b"clip")
    _ = (tmp_path / "wm.png").write_bytes(b"wm")
    registry = AssetRegistry(tmp_path)
    _ = registry.register(asset_id="wm", kind="image", uri="wm.png", origin="brief")
    assembler = _StubAssembler()
    request = _unit_request(
        tmp_path,
        brief="brief con watermark",
        assembler=assembler,
        gate=Gate(FakeProbe(info=make_media())),
        registry=registry,
    )
    result = run_given_clips(
        model=_StaticModel(_watermark_draft(), _caption()),
        settings=Settings.from_root(tmp_path),
        request=request,
    )
    assert assembler.watermarks == [registry.path_for("wm")]
    assert result.outcome is RunOutcome.BLOCKED


def test_pipeline_watermark_falls_back_when_asset_missing(tmp_path: Path) -> None:
    _ = (tmp_path / "clip.mp4").write_bytes(b"clip")
    assembler = _StubAssembler()
    request = _unit_request(
        tmp_path,
        brief="brief con watermark ausente",
        assembler=assembler,
        gate=Gate(FakeProbe(info=make_media())),
    )
    _ = run_given_clips(
        model=_StaticModel(_watermark_draft(), _caption()),
        settings=Settings.from_root(tmp_path),
        request=request,
    )
    assert assembler.watermarks == [None]


def test_pipeline_blocks_duration_above_max(tmp_path: Path) -> None:
    _ = (tmp_path / "clip.mp4").write_bytes(b"clip")
    draft = make_draft(
        platforms={"tiktok": _platform_draft()},
        assets={"required": [make_asset_draft()], "optional": []},
    )
    request = _unit_request(
        tmp_path,
        brief="brief con duración excesiva",
        assembler=_StubAssembler(),
        gate=Gate(FakeProbe(info=make_media(duration_s=120.0))),
    )
    result = run_given_clips(
        model=_StaticModel(draft, _caption()),
        settings=Settings.from_root(tmp_path),
        request=request,
    )
    assert result.outcome is RunOutcome.BLOCKED
    _assert_rejected_check(result, "duration.max")


def test_pipeline_blocks_forbidden_term(tmp_path: Path) -> None:
    _ = (tmp_path / "clip.mp4").write_bytes(b"clip")
    draft = make_draft(
        prohibitions=candidate(["sorteo"]),
        assets={"required": [make_asset_draft()], "optional": []},
    )
    request = _unit_request(
        tmp_path,
        brief="brief con término prohibido",
        assembler=_StubAssembler(),
        gate=Gate(FakeProbe(info=make_media())),
    )
    caption = Caption(caption="gran sorteo @marca #marca", hashtags=("#marca",))
    result = run_given_clips(
        model=_StaticModel(draft, caption),
        settings=Settings.from_root(tmp_path),
        request=request,
    )
    assert result.outcome is RunOutcome.BLOCKED
    _assert_rejected_check(result, "caption.forbidden")


def test_pipeline_blocks_spelling_without_subtitles(tmp_path: Path) -> None:
    _ = (tmp_path / "clip.mp4").write_bytes(b"clip")
    draft = make_draft(
        spelling_locks=candidate(["MarcaX"]),
        assets={"required": [make_asset_draft()], "optional": []},
    )
    request = _unit_request(
        tmp_path,
        brief="brief con spelling lock",
        assembler=_StubAssembler(),
        gate=Gate(FakeProbe(info=make_media())),
    )
    result = run_given_clips(
        model=_StaticModel(draft, _caption()),
        settings=Settings.from_root(tmp_path),
        request=request,
    )
    assert result.outcome is RunOutcome.BLOCKED
    _assert_rejected_check(result, "subtitles.spelling_lock", status="unsupported")


def test_cli_run_missing_brief_exits_1_with_json_stderr(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = main(
        [
            "run",
            str(tmp_path / "nope.md"),
            "--out",
            str(tmp_path / "delivery"),
            "--root",
            str(tmp_path),
        ]
    )
    assert code == 1
    captured = capsys.readouterr()
    assert not captured.out
    assert "no existe" in str(_parse(captured.err)["error"])


def test_cli_run_reports_unresolved_with_issues(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    brief_text = "cita del brief\nbrief sin modo"
    brief_path = tmp_path / "brief.md"
    _ = brief_path.write_text(brief_text, encoding="utf-8")
    recorded = tmp_path / "recorded"
    _ = record_response(
        brief_text,
        make_draft(mode={"confidence": "missing"}),
        prompt_version=PROMPT_VERSION,
        directory=recorded,
    )
    code = main(
        [
            "run",
            str(brief_path),
            "--out",
            str(tmp_path / "delivery"),
            "--recorded",
            str(recorded),
            "--root",
            str(tmp_path),
        ]
    )
    assert code == 1
    payload = _parse(capsys.readouterr().out)
    assert payload["outcome"] == "new_archetype"
    assert payload["issues"] == ["missing_required"]
    assert "exported" not in payload


def _stub_backend() -> SimpleNamespace:
    return SimpleNamespace(model_version="model-test")


def _write_brief(tmp_path: Path, text: str = "brief de prueba") -> Path:
    content = text if text.startswith("cita del brief") else f"cita del brief\n{text}"
    path = tmp_path / "brief.md"
    _ = path.write_text(content, encoding="utf-8")
    return path


@pytest.mark.parametrize("strength", ["recommended", "manual_review"])
def test_pipeline_blocks_required_watermark_even_if_soft_classified(
    tmp_path: Path, strength: str
) -> None:
    _ = (tmp_path / "clip.mp4").write_bytes(b"clip")
    draft = make_draft(
        watermark={
            "required": candidate(value=True),
            "asset_id": candidate("wm"),
            "visible_full_video": candidate(value=True),
        },
        rules={
            "hard": candidate(
                ["duration.min", "caption.required_hashtag", "caption.required_mention"]
            ),
            "recommended": candidate(["watermark.full_video"] if strength == "recommended" else []),
            "manual_review": candidate(
                ["watermark.full_video"] if strength == "manual_review" else []
            ),
        },
        assets={"required": [make_asset_draft()], "optional": []},
    )
    request = _unit_request(
        tmp_path,
        brief="brief con watermark exigido",
        assembler=_StubAssembler(),
        gate=Gate(FakeProbe(info=make_media())),
    )
    result = run_given_clips(
        model=_StaticModel(draft, _caption()),
        settings=Settings.from_root(tmp_path),
        request=request,
    )
    assert result.outcome is RunOutcome.BLOCKED
    _assert_rejected_check(result, "watermark.full_video", status="fail")


def test_pipeline_rejects_assets_under_private_scope(tmp_path: Path) -> None:
    secret = tmp_path / "campaigns" / "private" / "other" / "secret.mp4"
    secret.parent.mkdir(parents=True)
    _ = secret.write_bytes(b"clip")
    draft = make_draft(
        assets={
            "required": [make_asset_draft(uri="campaigns/private/other/secret.mp4")],
            "optional": [],
        }
    )
    request = _unit_request(
        tmp_path,
        brief="brief con asset privado",
        assembler=_StubAssembler(),
        gate=Gate(FakeProbe(info=make_media())),
    )
    result = run_given_clips(
        model=_StaticModel(draft, _caption()),
        settings=Settings.from_root(tmp_path),
        request=request,
    )
    assert result.outcome is RunOutcome.MANUAL_REVIEW
    assert result.package_path is None
    assert [issue.code.value for issue in result.issues] == ["unsafe_asset_scope"]
    assert not (tmp_path / "delivery").exists()
    assert _json(Path(result.manifest_path))["contract_sha256"] is not None


def _campaign_asset(
    tmp_path: Path,
    *,
    campaign: str,
    brief: str | None = None,
) -> tuple[Path, Path | None]:
    campaign_dir = tmp_path / "campaigns" / "private" / campaign
    campaign_dir.mkdir(parents=True, exist_ok=True)
    _ = (campaign_dir / "clip.mp4").write_bytes(b"clip")
    brief_path: Path | None = None
    if brief is not None:
        brief_path = campaign_dir / "brief.md"
        _ = brief_path.write_text(brief, encoding="utf-8")
    return campaign_dir, brief_path


def test_pipeline_allows_own_campaign_private_assets(tmp_path: Path) -> None:
    campaign_dir, brief_path = _campaign_asset(tmp_path, campaign="camp", brief="brief de campaña")
    draft = make_draft(
        assets={
            "required": [make_asset_draft(uri="campaigns/private/camp/clip.mp4")],
            "optional": [],
        }
    )
    request = _unit_request(
        tmp_path,
        brief="brief de campaña",
        assembler=_StubAssembler(),
        gate=Gate(FakeProbe(info=make_media())),
        brief_path=brief_path,
    )
    result = run_given_clips(
        model=_StaticModel(draft, _caption()),
        settings=Settings.from_root(tmp_path),
        request=request,
    )
    assert result.outcome is RunOutcome.EXPORTED
    assert campaign_dir.is_dir()


def test_pipeline_rejects_other_campaign_private_assets(tmp_path: Path) -> None:
    _, brief_path = _campaign_asset(tmp_path, campaign="attacker", brief="brief atacante")
    _ = _campaign_asset(tmp_path, campaign="victim")
    draft = make_draft(
        assets={
            "required": [make_asset_draft(uri="campaigns/private/victim/clip.mp4")],
            "optional": [],
        }
    )
    request = _unit_request(
        tmp_path,
        brief="brief atacante",
        assembler=_StubAssembler(),
        gate=Gate(FakeProbe(info=make_media())),
        brief_path=brief_path,
    )
    result = run_given_clips(
        model=_StaticModel(draft, _caption()),
        settings=Settings.from_root(tmp_path),
        request=request,
    )
    assert result.outcome is RunOutcome.MANUAL_REVIEW
    assert [issue.code.value for issue in result.issues] == ["unsafe_asset_scope"]


def test_pipeline_rejects_assets_under_runs_scope(tmp_path: Path) -> None:
    previous = tmp_path / "runs" / "prev"
    previous.mkdir(parents=True)
    _ = (previous / "clip.mp4").write_bytes(b"clip")
    draft = make_draft(
        assets={"required": [make_asset_draft(uri="runs/prev/clip.mp4")], "optional": []}
    )
    request = _unit_request(
        tmp_path,
        brief="brief con asset de runs",
        assembler=_StubAssembler(),
        gate=Gate(FakeProbe(info=make_media())),
    )
    result = run_given_clips(
        model=_StaticModel(draft, _caption()),
        settings=Settings.from_root(tmp_path),
        request=request,
    )
    assert result.outcome is RunOutcome.MANUAL_REVIEW
    assert [issue.code.value for issue in result.issues] == ["unsafe_asset_scope"]


def test_pipeline_rejects_private_watermark(tmp_path: Path) -> None:
    _ = (tmp_path / "clip.mp4").write_bytes(b"clip")
    private = tmp_path / "campaigns" / "private" / "other"
    private.mkdir(parents=True)
    _ = (private / "wm.png").write_bytes(b"wm")
    registry = AssetRegistry(tmp_path)
    _ = registry.register(
        asset_id="wm", kind="image", uri="campaigns/private/other/wm.png", origin="brief"
    )
    assembler = _StubAssembler()
    request = _unit_request(
        tmp_path,
        brief="brief con watermark privado",
        assembler=assembler,
        gate=Gate(FakeProbe(info=make_media())),
        registry=registry,
    )
    result = run_given_clips(
        model=_StaticModel(_watermark_draft(), _caption()),
        settings=Settings.from_root(tmp_path),
        request=request,
    )
    assert result.outcome is RunOutcome.MANUAL_REVIEW
    assert [issue.code.value for issue in result.issues] == ["unsafe_asset_scope"]
    assert assembler.watermarks == []


@pytest.mark.skipif(os.name != "nt", reason="junctions solo en Windows")
def test_pipeline_rejects_private_scope_through_junction(tmp_path: Path) -> None:
    cmd = shutil.which("cmd")
    if cmd is None:
        pytest.skip("cmd no disponible")
    storage = tmp_path / "storage"
    storage.mkdir()
    _ = (storage / "secret.mp4").write_bytes(b"clip")
    campaigns = tmp_path / "campaigns"
    campaigns.mkdir()
    completed = subprocess.run(
        [cmd, "/c", "mklink", "/J", str(campaigns / "private"), str(storage)],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        pytest.skip("no se pudo crear el junction")
    draft = make_draft(
        assets={
            "required": [make_asset_draft(uri="campaigns/private/secret.mp4")],
            "optional": [],
        }
    )
    request = _unit_request(
        tmp_path,
        brief="brief con junction privado",
        assembler=_StubAssembler(),
        gate=Gate(FakeProbe(info=make_media())),
    )
    result = run_given_clips(
        model=_StaticModel(draft, _caption()),
        settings=Settings.from_root(tmp_path),
        request=request,
    )
    assert result.outcome is RunOutcome.MANUAL_REVIEW
    assert [issue.code.value for issue in result.issues] == ["unsafe_asset_scope"]


@pytest.mark.parametrize("strength", ["recommended", "manual_review"])
def test_pipeline_blocks_tampered_asset_even_if_soft_classified(
    tmp_path: Path, strength: str
) -> None:
    _ = (tmp_path / "clip.mp4").write_bytes(b"clip")
    registry = AssetRegistry(tmp_path)
    _ = registry.register(asset_id="clip-01", kind="video", uri="clip.mp4", origin="brief")
    _ = (tmp_path / "clip.mp4").write_bytes(b"tampered")
    draft = make_draft(
        rules={
            "hard": candidate(
                ["duration.min", "caption.required_hashtag", "caption.required_mention"]
            ),
            "recommended": candidate(["assets.required"] if strength == "recommended" else []),
            "manual_review": candidate(["assets.required"] if strength == "manual_review" else []),
        },
        assets={"required": [make_asset_draft()], "optional": []},
    )
    request = _unit_request(
        tmp_path,
        brief="brief con asset manipulado",
        assembler=_StubAssembler(),
        gate=Gate(FakeProbe(info=make_media())),
        registry=registry,
    )
    result = run_given_clips(
        model=_StaticModel(draft, _caption()),
        settings=Settings.from_root(tmp_path),
        request=request,
    )
    assert result.outcome is RunOutcome.BLOCKED
    _assert_rejected_check(result, "assets.required")


@pytest.mark.parametrize(
    ("name", "content"),
    [
        ("invalido.draft.json", "{no es json"),
        ("esquema.draft.json", '{"brief_sha256": "abc"}'),
        ("canario.draft.json", '{"CANARIO-SECRETO-9999": 1}'),
    ],
)
def test_cli_run_rejects_invalid_recordings_as_json(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    name: str,
    content: str,
) -> None:
    brief_path = _write_brief(tmp_path)
    recorded = tmp_path / "recorded"
    recorded.mkdir()
    _ = (recorded / name).write_text(content, encoding="utf-8")
    code = main(
        [
            "run",
            str(brief_path),
            "--out",
            str(tmp_path / "delivery"),
            "--recorded",
            str(recorded),
            "--root",
            str(tmp_path),
        ]
    )
    assert code == 1
    captured = capsys.readouterr()
    assert not captured.out
    assert "grabaciones inválidas" in str(_parse(captured.err)["error"])
    assert "CANARIO" not in captured.err


def test_cli_run_rejects_duplicate_recordings_as_json(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    brief_path = _write_brief(tmp_path)
    recorded = tmp_path / "recorded"
    recorded.mkdir()
    payload = {
        "prompt_sha256": "a" * 64,
        "platform": "tiktok",
        "piece_id": "CANARIO-1234",
        "prompt_version": CAPTION_PROMPT_VERSION,
        "caption": {"caption": "hola", "hashtags": ["#marca"]},
    }
    for name in ("uno.caption.json", "dos.caption.json"):
        _ = (recorded / name).write_text(json.dumps(payload), encoding="utf-8")
    code = main(
        [
            "run",
            str(brief_path),
            "--out",
            str(tmp_path / "delivery"),
            "--recorded",
            str(recorded),
            "--root",
            str(tmp_path),
        ]
    )
    assert code == 1
    captured = capsys.readouterr()
    assert not captured.out
    assert "grabaciones inválidas" in str(_parse(captured.err)["error"])
    assert "CANARIO" not in captured.err


def test_cli_run_builds_openai_backend_without_recorded(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    brief_path = _write_brief(tmp_path)
    backend = _stub_backend()
    captured: dict[str, object] = {}

    def _fake_from_env(*_args: object, **_kwargs: object) -> object:
        return backend

    def _fake_run(*, model: object, request: RunRequest, **_kwargs: object) -> None:
        captured["model"] = model
        captured["model_version"] = request.model_version
        msg = "stop"
        raise PipelineError(msg)

    monkeypatch.setattr("kliptych.__main__.OpenAIChatModel.from_env", _fake_from_env)
    monkeypatch.setattr("kliptych.__main__.run_given_clips", _fake_run)
    code = main(
        ["run", str(brief_path), "--out", str(tmp_path / "delivery"), "--root", str(tmp_path)]
    )
    assert code == 1
    assert captured["model"] is backend
    assert captured["model_version"] == "model-test"
    assert "stop" in str(_parse(capsys.readouterr().err)["error"])


def test_cli_run_without_backend_env_reports_json(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    brief_path = _write_brief(tmp_path)
    for name in ("KLIPTYCH_LLM_BASE_URL", "KLIPTYCH_LLM_API_KEY", "KLIPTYCH_LLM_MODEL"):
        monkeypatch.delenv(name, raising=False)
    code = main(
        ["run", str(brief_path), "--out", str(tmp_path / "delivery"), "--root", str(tmp_path)]
    )
    assert code == 1
    assert "faltan variables" in str(_parse(capsys.readouterr().err)["error"])


@pytest.mark.parametrize("error_type", [ModelError, PipelineError, AssembleError, ExportError])
def test_cli_run_maps_managed_errors_to_json(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    error_type: type[Exception],
) -> None:
    brief_path = _write_brief(tmp_path)

    def _fake_from_env(*_args: object, **_kwargs: object) -> object:
        return _stub_backend()

    def _boom(**_kwargs: object) -> None:
        msg = "boom"
        raise error_type(msg)

    monkeypatch.setattr("kliptych.__main__.OpenAIChatModel.from_env", _fake_from_env)
    monkeypatch.setattr("kliptych.__main__.run_given_clips", _boom)
    code = main(
        ["run", str(brief_path), "--out", str(tmp_path / "delivery"), "--root", str(tmp_path)]
    )
    assert code == 1
    captured = capsys.readouterr()
    assert not captured.out
    assert "boom" in str(_parse(captured.err)["error"])


def test_cli_run_unknown_recording_maps_model_error_to_json(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    brief_path = _write_brief(tmp_path)
    recorded = tmp_path / "recorded"
    recorded.mkdir()
    _ = (recorded / "suelto.json").write_text("{}", encoding="utf-8")
    code = main(
        [
            "run",
            str(brief_path),
            "--out",
            str(tmp_path / "delivery"),
            "--recorded",
            str(recorded),
            "--root",
            str(tmp_path),
        ]
    )
    assert code == 1
    assert "no reconocidas" in str(_parse(capsys.readouterr().err)["error"])
