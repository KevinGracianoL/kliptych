"""Sprint 2 (P1 + W1-bis): captura single-pass y watermark semitransparente.

P1: ``check_watermark_full_video`` sobre un clip real de 20 s debe evaluarse
en menos de 15 s (la extracción per-frame con un ffmpeg por muestra tarda
~129 s para 80 muestras).

W1-bis: un template multicolor semitransparente (opacidad 0.3/0.5) sobre un
fondo texturizado vertical (``testsrc2``) no alcanza el umbral NCC de 0.75
(~0.37 a opacidad 0.3) pero deja una señal consistente del contorno alfa;
el gate debe devolver ``manual_review`` (o ``pass`` con evidencia de borde
fuerte) en vez de ``fail``. Sin watermark el mismo fondo no da señal
(NCC ~0.0) y debe seguir fallando, igual que una posición incorrecta.
"""

import shutil
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np
import pytest

from kliptych.assembler import FFmpegAssembler
from kliptych.assets import AssetRegistry
from kliptych.contract import Contract, Platform, Watermark, WatermarkPosition
from kliptych.gate import (
    CheckStatus,
    GateContext,
    check_watermark_full_video,
    check_watermark_present,
)
from kliptych.gate.watermark import Sample, WatermarkError, expected_top_left
from tests.support import make_contract, make_media, make_piece

_FFMPEG = shutil.which("ffmpeg")
_FFPROBE = shutil.which("ffprobe")
_NEEDS_TOOLS = _FFMPEG is None or _FFPROBE is None

_REVIEW_OR_PASS = (CheckStatus.MANUAL_REVIEW, CheckStatus.PASS)
_SYNTH_W = 160
_SYNTH_H = 120


def _color_png(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas: np.ndarray = np.zeros((80, 120, 4), dtype=np.uint8)
    canvas[:, :, 0:3] = 255
    canvas[:, :, 3] = 255
    _ = cv2.rectangle(canvas, (0, 0), (119, 79), (0, 0, 0, 255), thickness=4)
    _ = cv2.line(canvas, (8, 8), (112, 72), (0, 0, 255, 255), thickness=6)
    _ = cv2.circle(canvas, (60, 40), 16, (255, 0, 0, 255), thickness=-1)
    canvas[60:80, 100:120, 3] = 0
    _ = cv2.imwrite(str(path), canvas)
    return path


def _registry(tmp_path: Path) -> AssetRegistry:
    registry = AssetRegistry(tmp_path)
    _ = _color_png(tmp_path / "assets" / "wm.png")
    _ = registry.register(asset_id="wm-marca", kind="image", uri="assets/wm.png", origin="brief")
    return registry


def _contract(*, full_video: bool, position: str, opacity: float) -> Contract:
    rule = "watermark.full_video" if full_video else "watermark.present"
    return make_contract(
        hard=["artifact.integrity", rule],
        audio_rule="any",
        min_s=None,
        max_s=None,
        required_mentions=(),
        required_hashtags=(),
        watermark_required=True,
        watermark_visible_full_video=full_video,
        watermark_position=position,
        watermark_opacity=opacity,
    )


def _context(
    tmp_path: Path,
    video: Path,
    *,
    full_video: bool,
    position: str = "center_bottom",
    opacity: float = 1.0,
    duration_s: float = 2.0,
) -> GateContext:
    contract = _contract(full_video=full_video, position=position, opacity=opacity)
    return GateContext(
        contract=contract,
        rules=contract.platforms[Platform.TIKTOK],
        piece=make_piece(video),
        artifact_sha256="a" * 64,
        media=make_media(duration_s=duration_s),
        assets=_registry(tmp_path),
    )


def _vertical_source(path: Path, *, duration: float) -> Path:
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
        "testsrc2=s=540x960:r=30",
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


def _assemble(
    tmp_path: Path,
    name: str,
    *,
    duration: float,
    opacity: float | None,
    position: WatermarkPosition | None,
) -> Path:
    assert _FFMPEG is not None
    clip = _vertical_source(tmp_path / f"{name}-src.mp4", duration=duration)
    destination = tmp_path / f"{name}.mp4"
    assembler = FFmpegAssembler(ffmpeg=_FFMPEG)
    if opacity is None or position is None:
        _ = assembler.assemble(clip=clip, destination=destination, width=540, height=960)
        return destination
    config = Watermark(
        required=True,
        asset_id="wm-marca",
        visible_full_video=True,
        position=position,
        opacity=opacity,
    )
    _ = assembler.assemble(
        clip=clip,
        destination=destination,
        watermark=tmp_path / "assets" / "wm.png",
        watermark_config=config,
        width=540,
        height=960,
    )
    return destination


def _textured(
    tmp_path: Path,
    name: str,
    *,
    duration: float = 2.0,
    opacity: float | None = 0.3,
    position: WatermarkPosition | None = WatermarkPosition.CENTER_BOTTOM,
) -> Path:
    _ = _color_png(tmp_path / "assets" / "wm.png")
    return _assemble(tmp_path, name, duration=duration, opacity=opacity, position=position)


def _signal_sample(*, correlation: float, edge_score: float | None) -> Sample:
    return Sample(
        t_s=1.0,
        matched=False,
        correlation=correlation,
        x=216,
        y=796,
        expected_x=216.0,
        expected_y=796.0,
        metric="ccoeff",
        edge_score=edge_score,
    )


def _dummy_video(tmp_path: Path, name: str) -> Path:
    path = tmp_path / f"{name}.mp4"
    _ = path.write_bytes(b"video")
    return path


def _install_samples(monkeypatch: pytest.MonkeyPatch, samples: list[Sample]) -> None:
    def _fake_evaluate(ready: object, timestamps: object) -> list[Sample]:
        _ = (ready, timestamps)
        return samples

    monkeypatch.setattr("kliptych.gate.watermark._evaluate_samples", _fake_evaluate)


def _install_evaluation_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def _fake_evaluate(ready: object, timestamps: object) -> list[Sample]:
        _ = (ready, timestamps)
        msg = "el frame no se pudo leer"
        raise WatermarkError(msg)

    monkeypatch.setattr("kliptych.gate.watermark._evaluate_samples", _fake_evaluate)


class _UnopenableCapture:
    """Doble de VideoCapture que nunca abre el video (fail-closed)."""

    def __init__(self, path: str) -> None:
        _ = path

    def isOpened(self) -> bool:
        return False

    def release(self) -> None:
        return None


class _UnreadableCapture:
    """Doble de VideoCapture que abre pero no entrega frames (fail-closed)."""

    def __init__(self, path: str) -> None:
        _ = path

    def isOpened(self) -> bool:
        return True

    def set(self, prop: int, value: float) -> bool:
        _ = (prop, value)
        return True

    def read(self) -> tuple[bool, None]:
        return (False, None)

    def release(self) -> None:
        return None


def _synthetic_video(path: Path, png: Path, *, overlay: bool) -> Path:
    fourcc = int.from_bytes(b"mp4v", byteorder="little")
    writer = cv2.VideoWriter(str(path), fourcc, 30, (_SYNTH_W, _SYNTH_H))
    assert writer.isOpened()
    seed = 7 if overlay else 11
    rng = np.random.default_rng(seed)
    noise: np.ndarray = rng.integers(0, 256, size=(_SYNTH_H, _SYNTH_W)).astype(np.uint8)
    background: np.ndarray = cv2.GaussianBlur(noise, (11, 11), 0)
    frame: np.ndarray = cv2.cvtColor(background, cv2.COLOR_GRAY2BGR)
    if overlay:
        _blend_template(frame, png)
    for _ in range(30):
        _ = writer.write(frame)
    _ = writer.release()
    return path


def _blend_template(frame: np.ndarray, png: Path) -> None:
    image = cv2.imread(str(png), cv2.IMREAD_UNCHANGED)
    assert image is not None
    parts = cv2.split(image)
    gray = cv2.cvtColor(cv2.merge(parts[0:3]), cv2.COLOR_BGR2GRAY)
    width = int(_SYNTH_W * 0.20 // 2 * 2)
    height = int(width * 80 / 120 // 2 * 2)
    small_gray: np.ndarray = cv2.resize(gray, (width, height), interpolation=cv2.INTER_AREA)
    small_alpha: np.ndarray = cv2.resize(parts[3], (width, height), interpolation=cv2.INTER_AREA)
    x, y = expected_top_left(_SYNTH_W, _SYNTH_H, width, height, WatermarkPosition.CENTER_BOTTOM)
    left, top = int(x), int(y)
    region = frame[top : top + height, left : left + width].astype(np.float32)
    logo = cv2.cvtColor(small_gray, cv2.COLOR_GRAY2BGR).astype(np.float32)
    visible: np.ndarray = (small_alpha > 127).astype(np.float32)
    blended = (0.3 * logo + 0.7 * region) * visible[..., None] + region * (1.0 - visible[..., None])
    frame[top : top + height, left : left + width] = blended.astype(np.uint8)


def test_synthetic_translucent_overlay_needs_review_or_pass(tmp_path: Path) -> None:
    png = _color_png(tmp_path / "assets" / "wm.png")
    video = _synthetic_video(tmp_path / "synthetic-marked.mp4", png, overlay=True)
    assert (
        check_watermark_present(
            _context(tmp_path, video, full_video=False, opacity=0.3, duration_s=1.0)
        ).status
        in _REVIEW_OR_PASS
    )
    assert (
        check_watermark_full_video(
            _context(tmp_path, video, full_video=True, opacity=0.3, duration_s=1.0)
        ).status
        in _REVIEW_OR_PASS
    )


def test_synthetic_texture_without_overlay_fails(tmp_path: Path) -> None:
    png = _color_png(tmp_path / "assets" / "wm.png")
    video = _synthetic_video(tmp_path / "synthetic-plain.mp4", png, overlay=False)
    assert (
        check_watermark_present(
            _context(tmp_path, video, full_video=False, opacity=0.3, duration_s=1.0)
        ).status
        is CheckStatus.FAIL
    )
    assert (
        check_watermark_full_video(
            _context(tmp_path, video, full_video=True, opacity=0.3, duration_s=1.0)
        ).status
        is CheckStatus.FAIL
    )


def test_verdict_translucent_consistent_signal_needs_review(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    samples = [_signal_sample(correlation=0.37, edge_score=0.75) for _ in range(3)]
    _install_samples(monkeypatch, samples)
    video = _dummy_video(tmp_path, "dummy")
    outcome = check_watermark_full_video(_context(tmp_path, video, full_video=True, opacity=0.3))
    assert outcome.status is CheckStatus.MANUAL_REVIEW


def test_verdict_strict_edge_match_passes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    samples = [_signal_sample(correlation=0.70, edge_score=0.95) for _ in range(3)]
    _install_samples(monkeypatch, samples)
    video = _dummy_video(tmp_path, "dummy")
    outcome = check_watermark_full_video(_context(tmp_path, video, full_video=True, opacity=0.5))
    assert outcome.status is CheckStatus.PASS


def test_verdict_without_signal_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    samples = [_signal_sample(correlation=0.05, edge_score=0.02) for _ in range(3)]
    _install_samples(monkeypatch, samples)
    video = _dummy_video(tmp_path, "dummy")
    outcome = check_watermark_full_video(_context(tmp_path, video, full_video=True, opacity=0.3))
    assert outcome.status is CheckStatus.FAIL


def test_verdict_present_mode_single_translucent_signal_needs_review(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    samples = [
        _signal_sample(correlation=0.02, edge_score=0.01),
        _signal_sample(correlation=0.40, edge_score=0.72),
        _signal_sample(correlation=0.03, edge_score=0.00),
    ]
    _install_samples(monkeypatch, samples)
    video = _dummy_video(tmp_path, "dummy")
    outcome = check_watermark_present(_context(tmp_path, video, full_video=False, opacity=0.3))
    assert outcome.status is CheckStatus.MANUAL_REVIEW


def test_verdict_opaque_without_signal_stays_fail(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    samples = [_signal_sample(correlation=0.40, edge_score=0.75) for _ in range(3)]
    _install_samples(monkeypatch, samples)
    video = _dummy_video(tmp_path, "dummy")
    outcome = check_watermark_full_video(_context(tmp_path, video, full_video=True, opacity=1.0))
    assert outcome.status is CheckStatus.FAIL


def test_evaluation_error_is_fail_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _install_evaluation_error(monkeypatch)
    video = _dummy_video(tmp_path, "dummy")
    outcome = check_watermark_full_video(_context(tmp_path, video, full_video=True, opacity=0.3))
    assert outcome.status is CheckStatus.FAIL


def test_unopenable_video_is_fail_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cv2, "VideoCapture", _UnopenableCapture)
    video = _dummy_video(tmp_path, "dummy")
    outcome = check_watermark_full_video(_context(tmp_path, video, full_video=True, opacity=0.3))
    assert outcome.status is CheckStatus.FAIL


def test_unreadable_frame_is_fail_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cv2, "VideoCapture", _UnreadableCapture)
    video = _dummy_video(tmp_path, "dummy")
    outcome = check_watermark_full_video(_context(tmp_path, video, full_video=True, opacity=0.3))
    assert outcome.status is CheckStatus.FAIL


@pytest.mark.integration
@pytest.mark.skipif(_NEEDS_TOOLS, reason="ffmpeg/ffprobe no disponibles")
@pytest.mark.parametrize("opacity", [0.3, 0.5])
def test_translucent_watermark_on_textured_background_needs_review(
    tmp_path: Path, opacity: float
) -> None:
    video = _textured(tmp_path, f"textured-marked-{opacity}", opacity=opacity)
    assert (
        check_watermark_present(_context(tmp_path, video, full_video=False, opacity=opacity)).status
        in _REVIEW_OR_PASS
    )
    assert (
        check_watermark_full_video(
            _context(tmp_path, video, full_video=True, opacity=opacity)
        ).status
        in _REVIEW_OR_PASS
    )


@pytest.mark.integration
@pytest.mark.skipif(_NEEDS_TOOLS, reason="ffmpeg/ffprobe no disponibles")
@pytest.mark.parametrize("opacity", [0.3, 0.5])
def test_textured_vertical_without_watermark_fails(tmp_path: Path, opacity: float) -> None:
    video = _textured(tmp_path, f"textured-plain-{opacity}", opacity=None)
    assert (
        check_watermark_present(_context(tmp_path, video, full_video=False, opacity=opacity)).status
        is CheckStatus.FAIL
    )
    assert (
        check_watermark_full_video(
            _context(tmp_path, video, full_video=True, opacity=opacity)
        ).status
        is CheckStatus.FAIL
    )


@pytest.mark.integration
@pytest.mark.skipif(_NEEDS_TOOLS, reason="ffmpeg/ffprobe no disponibles")
def test_textured_wrong_position_fails(tmp_path: Path) -> None:
    video = _textured(
        tmp_path, "textured-wrong-spot", opacity=0.5, position=WatermarkPosition.CENTER_TOP
    )
    assert (
        check_watermark_full_video(_context(tmp_path, video, full_video=True, opacity=0.5)).status
        is CheckStatus.FAIL
    )


@pytest.mark.integration
@pytest.mark.skipif(_NEEDS_TOOLS, reason="ffmpeg/ffprobe no disponibles")
def test_full_video_twenty_seconds_completes_fast(tmp_path: Path) -> None:
    video = _textured(tmp_path, "long-marked", duration=20.0, opacity=1.0)
    context = _context(tmp_path, video, full_video=True, opacity=1.0, duration_s=20.0)
    started = time.monotonic()
    outcome = check_watermark_full_video(context)
    elapsed = time.monotonic() - started
    assert outcome.status is CheckStatus.PASS
    assert elapsed < 15.0
