"""Sprint 3 Parte 3: composición Layout.SPLIT_SCREEN (Objetivos 1-4).

Objetivo 1: enum ``Layout`` con ``SPLIT_SCREEN``, ``SplitScreenConfig``
(top/bottom/gap/panel_ratio sobre lienzo 9:16) y composición en
``FFmpegAssembler`` con filtros scale/pad/crop/vstack, para dos videos o
video + imagen estática.

Objetivo 2: gate ``layout.geometry`` que verifica con ffprobe (vía
``MediaInfo``) el stream de video, las dimensiones esperadas y el aspecto
9:16, en cerrado (FAIL) ante cualquier desvío.

Objetivo 3: procedencia estricta T6 para ``SplitScreenDraft`` en el
resolutor (confianza válida + cita propia en el brief, sin alias sueltos,
con correspondencia numérica; MANUAL_REVIEW en cerrado).

Objetivo 4: invariante ``--resume`` (fingerprint incluye el split exacto)
e integración del pipeline given_clips.
"""

import shutil
import subprocess
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import NoReturn, override

import pytest
from pydantic import ValidationError

from kliptych import assembler
from kliptych.assembler import AssembleError, FFmpegAssembler, RenderSpec
from kliptych.assets import AssetRegistry
from kliptych.config import Settings
from kliptych.contract import Contract, Layout, Platform, SplitScreenConfig, contract_digest
from kliptych.contract.draft import ContractDraft, SplitScreenDraft
from kliptych.encoding import RenderConfig
from kliptych.environment import EnvironmentReport
from kliptych.gate import CheckStatus, Gate, GateContext
from kliptych.gate.checks import check_split_screen_geometry
from kliptych.gate.probe import FFprobeProbe
from kliptych.orchestrator import (
    LongVideoModel,
    PipelineConfig,
    PipelineError,
    compute_long_video_fingerprint,
    run_long_video,
    run_slideshow,
)
from kliptych.pipeline import PieceAssembler, RunOutcome, RunRequest, run_given_clips
from kliptych.resolver import IssueCode, ResolutionStatus, resolve_contract
from kliptych.runtime import (
    CAPTION_PROMPT_VERSION,
    PROMPT_VERSION,
    CampaignModel,
    Caption,
    PieceContext,
)
from tests.support import FakeProbe, candidate, make_contract, make_draft, make_media, make_piece

_FFMPEG = shutil.which("ffmpeg")
_FFPROBE = shutil.which("ffprobe")
_NEEDS_TOOLS = _FFMPEG is None or _FFPROBE is None


def _file(tmp_path: Path, name: str) -> Path:
    path = tmp_path / name
    _ = path.write_bytes(b"contenido")
    return path


def _split_spec(
    tmp_path: Path,
    *,
    gap: int = 0,
    panel_ratio: float = 0.5,
    bottom_suffix: str = ".mp4",
    mute_audio: bool = False,
) -> RenderSpec:
    top = _file(tmp_path, "top.mp4")
    bottom = _file(tmp_path, f"bottom{bottom_suffix}")
    config = SplitScreenConfig(
        top_source="clip-top",
        bottom_source="clip-bottom",
        gap=gap,
        panel_ratio=panel_ratio,
    )
    return RenderSpec(
        clip=top,
        destination=tmp_path / "out.mp4",
        layout=Layout.SPLIT_SCREEN,
        split_screen=config,
        top_clip=top,
        bottom_clip=bottom,
        mute_audio=mute_audio,
    )


def test_layout_enum_has_split_screen() -> None:
    assert Layout.SPLIT_SCREEN == "split_screen"
    assert Layout.SPLIT_SCREEN.value == "split_screen"
    assert Layout.SINGLE == "single"


def test_split_screen_config_defaults() -> None:
    config = SplitScreenConfig(top_source="clip-top", bottom_source="clip-bottom")
    assert config.gap == 0
    assert config.panel_ratio == pytest.approx(0.5)
    assert (config.width, config.height) == (1080, 1920)
    assert config.panel_heights() == (960, 960)


def test_split_screen_config_panel_heights_with_gap() -> None:
    config = SplitScreenConfig(
        top_source="clip-top", bottom_source="clip-bottom", gap=20, panel_ratio=0.5
    )
    assert config.panel_heights() == (950, 950)


def test_split_screen_config_rejects_degenerate_ratio() -> None:
    with pytest.raises(ValidationError):
        _ = SplitScreenConfig(top_source="clip-top", bottom_source="clip-bottom", panel_ratio=0.0)
    with pytest.raises(ValidationError):
        _ = SplitScreenConfig(top_source="clip-top", bottom_source="clip-bottom", panel_ratio=1.0)


def test_split_screen_config_rejects_odd_gap() -> None:
    with pytest.raises(ValidationError):
        _ = SplitScreenConfig(top_source="clip-top", bottom_source="clip-bottom", gap=1)


def test_split_screen_config_rejects_non_vertical_canvas() -> None:
    with pytest.raises(ValidationError):
        _ = SplitScreenConfig(
            top_source="clip-top", bottom_source="clip-bottom", width=1920, height=1080
        )


def test_split_screen_config_rejects_unsafe_source() -> None:
    with pytest.raises(ValidationError):
        _ = SplitScreenConfig(top_source="../escape", bottom_source="clip-bottom")


def test_split_argv_stacks_two_videos(tmp_path: Path) -> None:
    argv = FFmpegAssembler().render_arguments(_split_spec(tmp_path))
    assert "-filter_complex" in argv
    graph = argv[argv.index("-filter_complex") + 1]
    assert "scale=1080:960" in graph
    assert "crop=1080:960" in graph
    assert "vstack=inputs=2" in graph
    assert argv.count("-i") == 2
    assert "-loop" not in argv
    assert "-shortest" not in argv


def test_split_argv_gap_inserts_pad(tmp_path: Path) -> None:
    argv = FFmpegAssembler().render_arguments(_split_spec(tmp_path, gap=20))
    graph = argv[argv.index("-filter_complex") + 1]
    assert "scale=1080:950" in graph
    assert "pad=1080:970" in graph
    assert "vstack=inputs=2" in graph


def test_split_argv_image_panel_needs_no_loop(tmp_path: Path) -> None:
    argv = FFmpegAssembler().render_arguments(_split_spec(tmp_path, bottom_suffix=".png"))
    assert argv.count("-i") == 2
    assert "-loop" not in argv
    assert "-shortest" not in argv
    graph = argv[argv.index("-filter_complex") + 1]
    assert "vstack=inputs=2" in graph


def test_split_without_config_fails(tmp_path: Path) -> None:
    clip = _file(tmp_path, "clip.mp4")
    with pytest.raises(AssembleError, match="split_screen"):
        _ = FFmpegAssembler().render_arguments(
            RenderSpec(
                clip=clip,
                destination=tmp_path / "out.mp4",
                layout=Layout.SPLIT_SCREEN,
            )
        )


def test_split_without_panels_fails(tmp_path: Path) -> None:
    clip = _file(tmp_path, "clip.mp4")
    config = SplitScreenConfig(top_source="clip-top", bottom_source="clip-bottom")
    with pytest.raises(AssembleError, match="panel"):
        _ = FFmpegAssembler().assemble(
            RenderSpec(
                clip=clip,
                destination=tmp_path / "out.mp4",
                layout=Layout.SPLIT_SCREEN,
                split_screen=config,
            )
        )


def _synth_clip(path: Path, source: str, *, duration: float = 2.0) -> Path:
    assert _FFMPEG is not None
    argv = [
        _FFMPEG,
        "-y",
        "-nostdin",
        "-v",
        "error",
        "-f",
        "lavfi",
        "-i",
        source,
        "-t",
        f"{duration}",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        str(path),
    ]
    _ = subprocess.run(argv, capture_output=True, check=True, timeout=180)
    return path


@pytest.mark.integration
@pytest.mark.skipif(_NEEDS_TOOLS, reason="ffmpeg/ffprobe no disponibles")
def test_split_screen_renders_vertical_1080x1920(tmp_path: Path) -> None:
    assert _FFMPEG is not None
    assert _FFPROBE is not None
    top = _synth_clip(tmp_path / "top.mp4", "testsrc=s=640x480:r=30")
    bottom = _synth_clip(tmp_path / "bottom.mp4", "smptebars=s=640x480:r=30")
    config = SplitScreenConfig(top_source="clip-top", bottom_source="clip-bottom")
    destination = tmp_path / "split.mp4"
    _ = FFmpegAssembler(ffmpeg=_FFMPEG).assemble(
        RenderSpec(
            clip=top,
            destination=destination,
            layout=Layout.SPLIT_SCREEN,
            split_screen=config,
            top_clip=top,
            bottom_clip=bottom,
        )
    )
    media = FFprobeProbe(ffprobe=_FFPROBE).probe(destination)
    assert (media.width, media.height) == (1080, 1920)
    assert media.has_video


@pytest.mark.integration
@pytest.mark.skipif(_NEEDS_TOOLS, reason="ffmpeg/ffprobe no disponibles")
def test_split_screen_renders_video_plus_image(tmp_path: Path) -> None:
    assert _FFMPEG is not None
    assert _FFPROBE is not None
    top = _synth_clip(tmp_path / "top.mp4", "testsrc=s=640x480:r=30")
    image = tmp_path / "bottom.png"
    argv = [
        _FFMPEG,
        "-y",
        "-nostdin",
        "-v",
        "error",
        "-f",
        "lavfi",
        "-i",
        "color=c=red:s=320x240:r=30",
        "-frames:v",
        "1",
        str(image),
    ]
    _ = subprocess.run(argv, capture_output=True, check=True, timeout=180)
    config = SplitScreenConfig(top_source="clip-top", bottom_source="img-bottom")
    destination = tmp_path / "split-img.mp4"
    _ = FFmpegAssembler(ffmpeg=_FFMPEG).assemble(
        RenderSpec(
            clip=top,
            destination=destination,
            layout=Layout.SPLIT_SCREEN,
            split_screen=config,
            top_clip=top,
            bottom_clip=image,
        )
    )
    media = FFprobeProbe(ffprobe=_FFPROBE).probe(destination)
    assert (media.width, media.height) == (1080, 1920)


def _split_payload() -> dict[str, object]:
    return {
        "top_source": "clip-top",
        "bottom_source": "clip-bottom",
        "gap": 0,
        "panel_ratio": 0.5,
        "width": 1080,
        "height": 1920,
    }


def _geometry_context(
    tmp_path: Path,
    video: Path,
    *,
    width: int | None = 1080,
    height: int | None = 1920,
    has_video: bool = True,
    with_probe: bool = True,
    with_config: bool = True,
) -> GateContext:
    contract = make_contract(
        hard=["artifact.integrity", "artifact.video_stream"],
        audio_rule="any",
        min_s=None,
        max_s=None,
        required_mentions=(),
        required_hashtags=(),
        split_screen=_split_payload() if with_config else None,
    )
    return GateContext(
        contract=contract,
        rules=contract.platforms[Platform.TIKTOK],
        piece=make_piece(video),
        artifact_sha256="a" * 64,
        media=make_media(width=width, height=height, has_video=has_video) if with_probe else None,
        assets=AssetRegistry(tmp_path),
    )


def test_geometry_passes_on_expected_dims(tmp_path: Path) -> None:
    video = _file(tmp_path, "dummy.mp4")
    outcome = check_split_screen_geometry(_geometry_context(tmp_path, video))
    assert outcome.status is CheckStatus.PASS
    assert outcome.evidence["width"] == 1080
    assert outcome.evidence["height"] == 1920


def test_geometry_fails_on_wrong_dims(tmp_path: Path) -> None:
    video = _file(tmp_path, "dummy.mp4")
    outcome = check_split_screen_geometry(
        _geometry_context(tmp_path, video, width=720, height=1280)
    )
    assert outcome.status is CheckStatus.FAIL


def test_geometry_fails_on_square_canvas(tmp_path: Path) -> None:
    video = _file(tmp_path, "dummy.mp4")
    outcome = check_split_screen_geometry(
        _geometry_context(tmp_path, video, width=1080, height=1080)
    )
    assert outcome.status is CheckStatus.FAIL


def test_geometry_fails_without_video(tmp_path: Path) -> None:
    video = _file(tmp_path, "dummy.mp4")
    outcome = check_split_screen_geometry(_geometry_context(tmp_path, video, has_video=False))
    assert outcome.status is CheckStatus.FAIL


def test_geometry_fails_without_probe(tmp_path: Path) -> None:
    video = _file(tmp_path, "dummy.mp4")
    outcome = check_split_screen_geometry(_geometry_context(tmp_path, video, with_probe=False))
    assert outcome.status is CheckStatus.FAIL


def test_geometry_fails_without_config(tmp_path: Path) -> None:
    video = _file(tmp_path, "dummy.mp4")
    outcome = check_split_screen_geometry(_geometry_context(tmp_path, video, with_config=False))
    assert outcome.status is CheckStatus.FAIL


@pytest.mark.integration
@pytest.mark.skipif(_NEEDS_TOOLS, reason="ffmpeg/ffprobe no disponibles")
def test_geometry_gate_passes_on_split_artifact(tmp_path: Path) -> None:
    assert _FFMPEG is not None
    assert _FFPROBE is not None
    top = _synth_clip(tmp_path / "top.mp4", "testsrc=s=640x480:r=30")
    bottom = _synth_clip(tmp_path / "bottom.mp4", "smptebars=s=640x480:r=30")
    config = SplitScreenConfig(top_source="clip-top", bottom_source="clip-bottom")
    destination = tmp_path / "split.mp4"
    _ = FFmpegAssembler(ffmpeg=_FFMPEG).assemble(
        RenderSpec(
            clip=top,
            destination=destination,
            layout=Layout.SPLIT_SCREEN,
            split_screen=config,
            top_clip=top,
            bottom_clip=bottom,
        )
    )
    contract = make_contract(
        hard=["artifact.integrity", "artifact.video_stream", "layout.geometry"],
        audio_rule="any",
        min_s=None,
        max_s=None,
        required_mentions=(),
        required_hashtags=(),
        split_screen=_split_payload(),
    )
    result = Gate(FFprobeProbe(ffprobe=_FFPROBE)).run(
        contract=contract,
        piece=make_piece(destination),
        assets=AssetRegistry(tmp_path),
    )
    outcome = next(check for check in result.checks if check.id == "layout.geometry")
    assert outcome.status is CheckStatus.PASS


@pytest.mark.integration
@pytest.mark.skipif(_NEEDS_TOOLS, reason="ffmpeg/ffprobe no disponibles")
def test_geometry_gate_rejects_single_clip_artifact(tmp_path: Path) -> None:
    assert _FFMPEG is not None
    assert _FFPROBE is not None
    clip = _synth_clip(tmp_path / "clip.mp4", "testsrc=s=640x480:r=30")
    destination = tmp_path / "single.mp4"
    _ = FFmpegAssembler(ffmpeg=_FFMPEG).assemble(
        RenderSpec(clip=clip, destination=destination, width=540, height=960)
    )
    contract = make_contract(
        hard=["artifact.integrity", "artifact.video_stream", "layout.geometry"],
        audio_rule="any",
        min_s=None,
        max_s=None,
        required_mentions=(),
        required_hashtags=(),
        split_screen=_split_payload(),
    )
    result = Gate(FFprobeProbe(ffprobe=_FFPROBE)).run(
        contract=contract,
        piece=make_piece(destination),
        assets=AssetRegistry(tmp_path),
    )
    outcome = next(check for check in result.checks if check.id == "layout.geometry")
    assert outcome.status is CheckStatus.FAIL


_SPLIT_BRIEF = (
    "cita del brief. Directors cut: panel superior con clip-top y panel inferior "
    "con clip-bottom, gap de 20 píxeles, reparto 50/50 del lienzo 1080x1920."
)


def _cand_at(value: object, quote: str, start: int) -> dict[str, object]:
    return {
        "value": value,
        "evidence": {
            "quote": quote,
            "start": start,
            "end": start + len(quote),
            "location": "brief.md#l1",
        },
        "confidence": "explicit",
    }


def _split_draft(
    *,
    top: object = "default",
    bottom: object = "default",
    gap: object = "default",
    ratio: object = "default",
    width: object = "default",
    height: object = "default",
) -> SplitScreenDraft:
    top_quote = "panel superior con clip-top"
    bottom_quote = "panel inferior con clip-bottom"
    gap_quote = "gap de 20 píxeles"
    ratio_quote = "reparto 50/50"
    canvas_quote = "lienzo 1080x1920"
    values: dict[str, object] = {
        "top_source": _cand_at("clip-top", top_quote, _SPLIT_BRIEF.index(top_quote))
        if top == "default"
        else top,
        "bottom_source": _cand_at("clip-bottom", bottom_quote, _SPLIT_BRIEF.index(bottom_quote))
        if bottom == "default"
        else bottom,
        "gap": _cand_at(20, gap_quote, _SPLIT_BRIEF.index(gap_quote)) if gap == "default" else gap,
        "panel_ratio": _cand_at(0.5, ratio_quote, _SPLIT_BRIEF.index(ratio_quote))
        if ratio == "default"
        else ratio,
        "width": _cand_at(1080, canvas_quote, _SPLIT_BRIEF.index(canvas_quote))
        if width == "default"
        else width,
        "height": _cand_at(1920, canvas_quote, _SPLIT_BRIEF.index(canvas_quote))
        if height == "default"
        else height,
    }
    return SplitScreenDraft.model_validate(values)


def test_resolver_resolves_split_screen_from_draft(tmp_path: Path) -> None:
    draft = make_draft(split_screen=_split_draft())
    result = resolve_contract(draft, registry=AssetRegistry(tmp_path), brief_text=_SPLIT_BRIEF)
    assert result.status == ResolutionStatus.RESOLVED
    assert result.contract is not None
    assert result.contract.split_screen is not None
    split = result.contract.split_screen
    assert split.top_source == "clip-top"
    assert split.bottom_source == "clip-bottom"
    assert split.gap == 20
    assert split.panel_ratio == pytest.approx(0.5)
    assert (split.width, split.height) == (1080, 1920)
    assert "layout.geometry" in result.contract.rules.hard


def test_resolver_rejects_split_without_provenance(tmp_path: Path) -> None:
    draft = make_draft(split_screen=_split_draft(top="clip-top"))
    result = resolve_contract(draft, registry=AssetRegistry(tmp_path), brief_text=_SPLIT_BRIEF)
    assert result.status == ResolutionStatus.MANUAL_REVIEW
    assert any(issue.field == "split_screen" for issue in result.issues)


def test_resolver_rejects_split_confidence_missing(tmp_path: Path) -> None:
    draft = make_draft(
        split_screen=_split_draft(top={"value": "clip-top", "confidence": "missing"})
    )
    result = resolve_contract(draft, registry=AssetRegistry(tmp_path), brief_text=_SPLIT_BRIEF)
    assert result.status == ResolutionStatus.MANUAL_REVIEW
    assert any(issue.field == "split_screen" for issue in result.issues)


def test_resolver_rejects_split_confidence_conflict(tmp_path: Path) -> None:
    draft = make_draft(split_screen=_split_draft(top={"confidence": "conflict"}))
    result = resolve_contract(draft, registry=AssetRegistry(tmp_path), brief_text=_SPLIT_BRIEF)
    assert result.status == ResolutionStatus.MANUAL_REVIEW
    assert any(issue.field == "split_screen" for issue in result.issues)


def test_resolver_rejects_split_without_evidence_quote(tmp_path: Path) -> None:
    draft = make_draft(
        split_screen=_split_draft(top={"value": "clip-top", "confidence": "explicit"})
    )
    result = resolve_contract(draft, registry=AssetRegistry(tmp_path), brief_text=_SPLIT_BRIEF)
    assert result.status == ResolutionStatus.MANUAL_REVIEW
    assert any(issue.field == "split_screen" for issue in result.issues)


def test_resolver_rejects_split_with_root_citation_alias(tmp_path: Path) -> None:
    draft = make_draft(
        split_screen=_split_draft(
            top={"value": "clip-top", "confidence": "explicit", "citation": "clip-top"}
        )
    )
    result = resolve_contract(draft, registry=AssetRegistry(tmp_path), brief_text=_SPLIT_BRIEF)
    assert result.status == ResolutionStatus.MANUAL_REVIEW
    assert any(issue.field == "split_screen" for issue in result.issues)


def test_resolver_rejects_split_with_root_quote_alias(tmp_path: Path) -> None:
    draft = make_draft(
        split_screen=_split_draft(
            top={"value": "clip-top", "confidence": "explicit", "quote": "clip-top"}
        )
    )
    result = resolve_contract(draft, registry=AssetRegistry(tmp_path), brief_text=_SPLIT_BRIEF)
    assert result.status == ResolutionStatus.MANUAL_REVIEW
    assert any(issue.field == "split_screen" for issue in result.issues)


def test_resolver_rejects_split_quote_not_in_brief(tmp_path: Path) -> None:
    draft = make_draft(
        split_screen=_split_draft(
            top={
                "value": "clip-top",
                "confidence": "explicit",
                "evidence": {"quote": "cita inventada fuera del brief"},
            },
        )
    )
    result = resolve_contract(draft, registry=AssetRegistry(tmp_path), brief_text=_SPLIT_BRIEF)
    assert result.status == ResolutionStatus.MANUAL_REVIEW
    assert any(issue.field == "split_screen" for issue in result.issues)


def test_resolver_rejects_split_value_outside_own_quote(tmp_path: Path) -> None:
    other_quote = "panel inferior con clip-bottom"
    draft = make_draft(
        split_screen=_split_draft(
            top=_cand_at("clip-top", other_quote, _SPLIT_BRIEF.index(other_quote)),
        )
    )
    result = resolve_contract(draft, registry=AssetRegistry(tmp_path), brief_text=_SPLIT_BRIEF)
    assert result.status == ResolutionStatus.MANUAL_REVIEW
    assert any(issue.field == "split_screen" for issue in result.issues)


def test_resolver_rejects_split_numeric_mismatch(tmp_path: Path) -> None:
    gap_quote = "gap de 20 píxeles"
    draft = make_draft(
        split_screen=_split_draft(
            gap=_cand_at(40, gap_quote, _SPLIT_BRIEF.index(gap_quote)),
        )
    )
    result = resolve_contract(draft, registry=AssetRegistry(tmp_path), brief_text=_SPLIT_BRIEF)
    assert result.status == ResolutionStatus.MANUAL_REVIEW
    assert any(issue.field == "split_screen" for issue in result.issues)


def test_resolver_rejects_split_missing_sources(tmp_path: Path) -> None:
    draft = make_draft(split_screen=_split_draft(top=None, bottom=None))
    result = resolve_contract(draft, registry=AssetRegistry(tmp_path), brief_text=_SPLIT_BRIEF)
    assert result.status == ResolutionStatus.MANUAL_REVIEW
    assert any(issue.field == "split_screen" for issue in result.issues)


_RESUME_URL = "https://example.com/stream.mp4"


def _resume_config(
    tmp_path: Path,
    *,
    gap: int = 0,
    panel_ratio: float = 0.5,
    top: str = "clip-top",
    bottom: str = "clip-bottom",
    with_split: bool = True,
) -> PipelineConfig:
    payload: dict[str, object] | None = (
        None
        if not with_split
        else {
            "top_source": top,
            "bottom_source": bottom,
            "gap": gap,
            "panel_ratio": panel_ratio,
        }
    )
    return PipelineConfig(
        output_dir=tmp_path / "out",
        contract=make_contract(split_screen=payload),
        render=RenderConfig(),
    )


def test_same_split_inputs_keep_fingerprint(tmp_path: Path) -> None:
    first = _resume_config(tmp_path, gap=20)
    second = _resume_config(tmp_path, gap=20)
    assert compute_long_video_fingerprint(_RESUME_URL, config=first) == (
        compute_long_video_fingerprint(_RESUME_URL, config=second)
    )


def test_split_gap_change_invalidates_fingerprint(tmp_path: Path) -> None:
    before = _resume_config(tmp_path, gap=0)
    after = _resume_config(tmp_path, gap=20)
    assert compute_long_video_fingerprint(_RESUME_URL, config=before) != (
        compute_long_video_fingerprint(_RESUME_URL, config=after)
    )


def test_split_ratio_change_invalidates_fingerprint(tmp_path: Path) -> None:
    before = _resume_config(tmp_path, panel_ratio=0.5)
    after = _resume_config(tmp_path, panel_ratio=0.3)
    assert compute_long_video_fingerprint(_RESUME_URL, config=before) != (
        compute_long_video_fingerprint(_RESUME_URL, config=after)
    )


def test_split_source_change_invalidates_fingerprint(tmp_path: Path) -> None:
    before = _resume_config(tmp_path, bottom="clip-bottom")
    after = _resume_config(tmp_path, bottom="clip-otro")
    assert compute_long_video_fingerprint(_RESUME_URL, config=before) != (
        compute_long_video_fingerprint(_RESUME_URL, config=after)
    )


def test_split_declaration_invalidates_fingerprint(tmp_path: Path) -> None:
    before = _resume_config(tmp_path, with_split=False)
    after = _resume_config(tmp_path, with_split=True)
    assert compute_long_video_fingerprint(_RESUME_URL, config=before) != (
        compute_long_video_fingerprint(_RESUME_URL, config=after)
    )


def test_contract_digest_stable_without_split() -> None:
    assert contract_digest(make_contract()) == contract_digest(make_contract(split_screen=None))


def test_contract_digest_changes_when_split_declared() -> None:
    assert contract_digest(make_contract()) != contract_digest(
        make_contract(split_screen=_split_payload())
    )


class _SplitStaticModel(CampaignModel):
    """Modelo de prueba con draft split fijo y caption válida."""

    model_version: str = "static-split"

    def __init__(self, draft: ContractDraft) -> None:
        self._draft: ContractDraft = draft

    @override
    def extract_contract(self, brief: str) -> ContractDraft:
        _ = brief
        return self._draft

    @override
    def write_caption(self, contract: Contract, piece: PieceContext) -> Caption:
        _ = (contract, piece)
        return Caption(caption="mira @marca #marca", hashtags=("#marca",))


class _SplitStubAssembler(PieceAssembler):
    """Ensamblador de prueba: registra los specs y escribe bytes."""

    def __init__(self) -> None:
        self.specs: list[RenderSpec] = []

    @override
    def assemble(self, spec: RenderSpec) -> Path:
        self.specs.append(spec)
        spec.destination.parent.mkdir(parents=True, exist_ok=True)
        _ = spec.destination.write_bytes(b"video")
        return spec.destination

    @override
    def render_arguments(self, spec: RenderSpec) -> tuple[str, ...]:
        return ("ffmpeg", str(spec.clip), str(spec.destination))


def _split_asset_draft(asset_id: str, uri: str) -> dict[str, object]:
    return {
        "asset_id": candidate(asset_id),
        "kind": candidate("video"),
        "uri": candidate(uri),
        "origin": candidate("brief"),
    }


def _split_workspace(root: Path) -> AssetRegistry:
    assets = root / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    _ = (assets / "top.mp4").write_bytes(b"top")
    _ = (assets / "bottom.mp4").write_bytes(b"bottom")
    registry = AssetRegistry(root)
    _ = registry.register(asset_id="clip-top", kind="video", uri="assets/top.mp4", origin="brief")
    _ = registry.register(
        asset_id="clip-bottom", kind="video", uri="assets/bottom.mp4", origin="brief"
    )
    return registry


def test_pipeline_assembles_split_pieces(tmp_path: Path) -> None:
    root = tmp_path / "ws"
    registry = _split_workspace(root)
    draft = make_draft(
        split_screen=_split_draft(),
        assets={
            "required": [
                _split_asset_draft("clip-top", "assets/top.mp4"),
                _split_asset_draft("clip-bottom", "assets/bottom.mp4"),
            ]
        },
    )
    stub = _SplitStubAssembler()
    request = RunRequest(
        brief=_SPLIT_BRIEF,
        destination=root / "delivery",
        environment=EnvironmentReport(),
        model_version="static-split",
        prompt_version=PROMPT_VERSION,
        caption_prompt_version=CAPTION_PROMPT_VERSION,
        assembler=stub,
        gate=Gate(FakeProbe(info=make_media())),
        registry=registry,
    )
    result = run_given_clips(
        model=_SplitStaticModel(draft),
        settings=Settings.from_root(root),
        request=request,
    )
    assert result.outcome is RunOutcome.EXPORTED
    assert len(stub.specs) == 1
    spec = stub.specs[0]
    assert spec.layout is Layout.SPLIT_SCREEN
    assert spec.split_screen is not None
    assert spec.split_screen.gap == 20
    assert spec.top_clip == root / "assets" / "top.mp4"
    assert spec.bottom_clip == root / "assets" / "bottom.mp4"


@pytest.mark.integration
@pytest.mark.skipif(_NEEDS_TOOLS, reason="ffmpeg/ffprobe no disponibles")
def test_pipeline_split_screen_end_to_end(tmp_path: Path) -> None:
    assert _FFPROBE is not None
    root = tmp_path / "ws"
    assets = root / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    _ = _synth_clip(assets / "top.mp4", "testsrc=s=640x480:r=30")
    _ = _synth_clip(assets / "bottom.mp4", "smptebars=s=640x480:r=30")
    registry = AssetRegistry(root)
    _ = registry.register(asset_id="clip-top", kind="video", uri="assets/top.mp4", origin="brief")
    _ = registry.register(
        asset_id="clip-bottom", kind="video", uri="assets/bottom.mp4", origin="brief"
    )
    draft = make_draft(
        split_screen=_split_draft(),
        assets={
            "required": [
                _split_asset_draft("clip-top", "assets/top.mp4"),
                _split_asset_draft("clip-bottom", "assets/bottom.mp4"),
            ]
        },
        platforms={
            "tiktok": {
                "required_hashtags": candidate(["#marca"]),
                "required_mentions": candidate(["@marca"]),
            }
        },
    )
    request = RunRequest(
        brief=_SPLIT_BRIEF,
        destination=root / "delivery",
        environment=EnvironmentReport(),
        model_version="static-split",
        prompt_version=PROMPT_VERSION,
        caption_prompt_version=CAPTION_PROMPT_VERSION,
        registry=registry,
    )
    result = run_given_clips(
        model=_SplitStaticModel(draft),
        settings=Settings.from_root(root),
        request=request,
    )
    assert result.outcome is RunOutcome.EXPORTED
    artifacts = list((root / "runs").rglob("clip-top__clip-bottom.mp4"))
    assert len(artifacts) == 1
    media = FFprobeProbe(ffprobe=_FFPROBE).probe(artifacts[0])
    assert (media.width, media.height) == (1080, 1920)


def _long_video_split_draft() -> ContractDraft:
    return make_draft(
        mode=candidate("long_video"),
        segments={"segments": [{"start_s": candidate(0.0), "end_s": candidate(8.5)}]},
        split_screen=_split_draft(),
    )


def test_resolver_rejects_split_screen_outside_given_clips(tmp_path: Path) -> None:
    draft = _long_video_split_draft()
    result = resolve_contract(draft, registry=AssetRegistry(tmp_path), brief_text=_SPLIT_BRIEF)
    assert result.status == ResolutionStatus.MANUAL_REVIEW
    assert result.contract is None
    assert any(
        issue.field == "split_screen" and issue.code is IssueCode.INVALID_CONTRACT
        for issue in result.issues
    )


class _GuardStubModel(LongVideoModel):
    @override
    def select_segments(self, prompt: Mapping[str, object]) -> object:
        _ = prompt
        return {}


def _explode_downloader(*args: object, **kwargs: object) -> NoReturn:
    _ = (args, kwargs)
    msg = "el pipeline no debe descargar con split_screen declarado"
    raise AssertionError(msg)


def test_long_video_rejects_split_screen_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("kliptych.orchestrator.MediaDownloader", _explode_downloader)
    config = PipelineConfig(
        output_dir=tmp_path / "out",
        contract=make_contract(split_screen=_split_payload()),
        render=RenderConfig(),
    )
    with pytest.raises(PipelineError, match="split_screen"):
        _ = run_long_video(_RESUME_URL, model=_GuardStubModel(), config=config)


def test_slideshow_rejects_split_screen_contract(tmp_path: Path) -> None:
    config = PipelineConfig(
        output_dir=tmp_path / "out",
        contract=make_contract(split_screen=_split_payload()),
        render=RenderConfig(),
    )
    with pytest.raises(PipelineError, match="split_screen"):
        _ = run_slideshow(images=[tmp_path / "img.png"], config=config)


def _probe_run(
    stdout: str, *, returncode: int = 0
) -> Callable[..., subprocess.CompletedProcess[str]]:
    def fake_run(argv: Sequence[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        _ = (argv, kwargs)
        return subprocess.CompletedProcess(args=[], returncode=returncode, stdout=stdout, stderr="")

    return fake_run


def _raise_probe_error(argv: Sequence[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
    _ = (argv, kwargs)
    msg = "ffprobe no disponible"
    raise OSError(msg)


def test_split_panel_has_audio_with_stream(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    clip = _file(tmp_path, "clip.mp4")
    monkeypatch.setattr("kliptych.assembler.subprocess.run", _probe_run("0\n"))
    assert assembler._split_panel_has_audio(clip, ffprobe="ffprobe", timeout_s=5.0) is True


def test_split_panel_has_audio_without_stream(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clip = _file(tmp_path, "clip.mp4")
    monkeypatch.setattr("kliptych.assembler.subprocess.run", _probe_run(""))
    assert assembler._split_panel_has_audio(clip, ffprobe="ffprobe", timeout_s=5.0) is False


def test_split_panel_has_audio_probe_failure_is_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clip = _file(tmp_path, "clip.mp4")
    monkeypatch.setattr("kliptych.assembler.subprocess.run", _probe_run("", returncode=1))
    assert assembler._split_panel_has_audio(clip, ffprobe="ffprobe", timeout_s=5.0) is True


def test_split_panel_has_audio_probe_error_is_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clip = _file(tmp_path, "clip.mp4")
    monkeypatch.setattr("kliptych.assembler.subprocess.run", _raise_probe_error)
    assert assembler._split_panel_has_audio(clip, ffprobe="ffprobe", timeout_s=5.0) is True


def test_split_panel_has_audio_missing_binary_is_fail_closed(tmp_path: Path) -> None:
    clip = _file(tmp_path, "clip.mp4")
    assert (
        assembler._split_panel_has_audio(
            clip, ffprobe="binario-inexistente-kliptych-test", timeout_s=5.0
        )
        is True
    )


def _stub_panel_audio(monkeypatch: pytest.MonkeyPatch, *, top: bool, bottom: bool) -> None:
    def fake(path: Path, *, ffprobe: str, timeout_s: float) -> bool:
        _ = (ffprobe, timeout_s)
        return top if path.name == "top.mp4" else bottom

    monkeypatch.setattr("kliptych.assembler._split_panel_has_audio", fake)


def test_split_argv_mixes_audio_when_both_panels_have_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stub_panel_audio(monkeypatch, top=True, bottom=True)
    argv = FFmpegAssembler().render_arguments(_split_spec(tmp_path))
    graph = argv[argv.index("-filter_complex") + 1]
    assert "amix=inputs=2:duration=longest" in graph
    assert "[a]" in argv
    assert "0:a?" not in argv


def test_split_argv_falls_back_to_bottom_audio(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stub_panel_audio(monkeypatch, top=False, bottom=True)
    argv = FFmpegAssembler().render_arguments(_split_spec(tmp_path))
    graph = argv[argv.index("-filter_complex") + 1]
    assert "amix" not in graph
    assert "1:a" in argv
    assert "0:a" not in argv
    assert "0:a?" not in argv


def test_split_argv_keeps_top_audio(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_panel_audio(monkeypatch, top=True, bottom=False)
    argv = FFmpegAssembler().render_arguments(_split_spec(tmp_path))
    graph = argv[argv.index("-filter_complex") + 1]
    assert "amix" not in graph
    assert "0:a" in argv
    assert "1:a" not in argv


def test_split_argv_without_audio_maps_optional(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stub_panel_audio(monkeypatch, top=False, bottom=False)
    argv = FFmpegAssembler().render_arguments(_split_spec(tmp_path))
    assert "0:a?" in argv


def test_split_argv_muted_mix_embeds_volume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stub_panel_audio(monkeypatch, top=True, bottom=True)
    argv = FFmpegAssembler().render_arguments(_split_spec(tmp_path, mute_audio=True))
    graph = argv[argv.index("-filter_complex") + 1]
    assert "volume=0" in graph
    assert "-af" not in argv


def test_split_argv_muted_single_panel_uses_af(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stub_panel_audio(monkeypatch, top=False, bottom=True)
    argv = FFmpegAssembler().render_arguments(_split_spec(tmp_path, mute_audio=True))
    assert "-af" in argv
    assert argv[argv.index("-af") + 1] == "volume=0"


def _synth_clip_with_audio(path: Path, *, duration: float = 2.0) -> Path:
    assert _FFMPEG is not None
    argv = [
        _FFMPEG,
        "-y",
        "-nostdin",
        "-v",
        "error",
        "-f",
        "lavfi",
        "-i",
        "testsrc=s=640x480:r=30",
        "-f",
        "lavfi",
        "-i",
        "sine=frequency=440:sample_rate=48000",
        "-t",
        f"{duration}",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-shortest",
        str(path),
    ]
    _ = subprocess.run(argv, capture_output=True, check=True, timeout=180)
    return path


@pytest.mark.integration
@pytest.mark.skipif(_NEEDS_TOOLS, reason="ffmpeg/ffprobe no disponibles")
def test_split_screen_preserves_bottom_audio_when_top_is_image(tmp_path: Path) -> None:
    assert _FFMPEG is not None
    assert _FFPROBE is not None
    image = tmp_path / "top.png"
    argv = [
        _FFMPEG,
        "-y",
        "-nostdin",
        "-v",
        "error",
        "-f",
        "lavfi",
        "-i",
        "color=c=red:s=320x240:r=30",
        "-frames:v",
        "1",
        str(image),
    ]
    _ = subprocess.run(argv, capture_output=True, check=True, timeout=180)
    bottom = _synth_clip_with_audio(tmp_path / "bottom.mp4")
    config = SplitScreenConfig(top_source="img-top", bottom_source="clip-bottom")
    destination = tmp_path / "split-fallback.mp4"
    _ = FFmpegAssembler(ffmpeg=_FFMPEG).assemble(
        RenderSpec(
            clip=image,
            destination=destination,
            layout=Layout.SPLIT_SCREEN,
            split_screen=config,
            top_clip=image,
            bottom_clip=bottom,
        )
    )
    media = FFprobeProbe(ffprobe=_FFPROBE).probe(destination)
    assert (media.width, media.height) == (1080, 1920)
    assert media.has_audio
