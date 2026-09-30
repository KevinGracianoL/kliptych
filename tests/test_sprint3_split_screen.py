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
from pathlib import Path

import pytest
from pydantic import ValidationError

from kliptych.assembler import AssembleError, FFmpegAssembler, RenderSpec
from kliptych.assets import AssetRegistry
from kliptych.contract import Layout, Platform, SplitScreenConfig
from kliptych.contract.draft import SplitScreenDraft
from kliptych.gate import CheckStatus, Gate, GateContext
from kliptych.gate.checks import check_split_screen_geometry
from kliptych.gate.probe import FFprobeProbe
from kliptych.resolver import ResolutionStatus, resolve_contract
from tests.support import make_contract, make_draft, make_media, make_piece

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
