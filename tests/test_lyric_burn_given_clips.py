"""Regresión P42-5: quemar letras reales en la ruta given_clips (sin falsos PASS).

La ruta given_clips ensamblaba el vídeo ignorando los subtítulos y escribía el
.ass como sidecar después del render: el gate leía el .ass y daba PASS, pero el
MP4 no tenía texto quemado. El .ass de letras debe entrar al ensamblador como
dependencia obligatoria ANTES del render cuando el formato es LYRIC_VIDEO, y un
ensamblador sin soporte de quemado debe fallar en cerrado.
"""

import json
import shutil
import subprocess
from pathlib import Path
from typing import cast, override

import cv2
import numpy as np
import pytest

from kliptych.assets import AssetRegistry
from kliptych.config import Settings
from kliptych.contract import Contract, ContractDraft, Watermark
from kliptych.contract.draft import LyricConfigDraft, TimestampRangeDraft
from kliptych.environment import EnvironmentReport
from kliptych.gate import Gate
from kliptych.gate.probe import FFprobeProbe
from kliptych.pipeline import (
    PieceAssembler,
    PipelineError,
    RunRequest,
    run_given_clips,
)
from kliptych.resolver import ResolutionStatus
from kliptych.runtime import (
    CAPTION_PROMPT_VERSION,
    PROMPT_VERSION,
    CampaignModel,
    Caption,
    PieceContext,
)
from tests.support import FakeProbe, candidate, make_asset_draft, make_draft, make_media

_FFMPEG = shutil.which("ffmpeg")
_FFPROBE = shutil.which("ffprobe")

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(_FFMPEG is None or _FFPROBE is None, reason="ffmpeg/ffprobe no disponibles"),
]

_LRC_BLACK = "[00:00.50]luz primera\n[00:02.00]luz segunda\n"
_FIXTURE_TIMEOUT_S = 120


def _generate_black_clip(path: Path, duration: int = 4) -> Path:
    assert _FFMPEG is not None
    path.parent.mkdir(parents=True, exist_ok=True)
    argv = [
        _FFMPEG,
        "-hide_banner",
        "-v",
        "error",
        "-y",
        "-f",
        "lavfi",
        "-i",
        f"color=c=black:size=360x640:rate=10:duration={duration}",
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
        "-movflags",
        "+faststart",
        str(path),
    ]
    completed = subprocess.run(
        argv, capture_output=True, text=True, timeout=_FIXTURE_TIMEOUT_S, check=False
    )
    assert completed.returncode == 0, completed.stderr
    return path


class _StaticModel(CampaignModel):
    """Modelo de prueba con draft fijo y caption fijo."""

    model_version: str = "static"

    def __init__(self, draft: ContractDraft, caption: Caption) -> None:
        self._draft: ContractDraft = draft
        self._caption: Caption = caption

    @override
    def extract_contract(self, brief: str) -> ContractDraft:
        _ = brief
        return self._draft

    @override
    def write_caption(self, contract: Contract, piece: PieceContext) -> Caption:
        _ = (contract, piece)
        return self._caption


def _lyric_platform_draft() -> dict[str, object]:
    return {
        "duration": {"min_s": candidate(2), "max_s": candidate(60)},
        "required_hashtags": candidate(["#marca"]),
        "required_mentions": candidate(["@marca"]),
    }


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


def _lyric_draft(*, timestamp_ranges: list[TimestampRangeDraft] | None = None) -> ContractDraft:
    return make_draft(
        format=candidate("lyric_video"),
        mode=candidate("given_clips"),
        platforms={"tiktok": _lyric_platform_draft()},
        lyric_video=LyricConfigDraft(
            lrc_asset_id=candidate("song_lrc"),
            lrclib_enabled=candidate(value=False),
        ),
        timestamp_ranges=timestamp_ranges or [],
        assets={
            "required": [
                make_asset_draft(asset_id="clip-01", uri="clip.mp4"),
                {
                    "asset_id": candidate("song_lrc"),
                    "kind": candidate("lyrics"),
                    "uri": candidate("song.lrc"),
                    "origin": candidate("brief"),
                },
            ],
            "optional": [],
        },
    )


def _register_lyric_assets(root: Path) -> AssetRegistry:
    _ = _generate_black_clip(root / "clip.mp4")
    _ = (root / "song.lrc").write_text(_LRC_BLACK, encoding="utf-8")
    registry = AssetRegistry(root)
    _ = registry.register(asset_id="clip-01", kind="video", uri="clip.mp4", origin="test")
    _ = registry.register(asset_id="song_lrc", kind="lyrics", uri="song.lrc", origin="test")
    return registry


def _caption() -> Caption:
    return Caption(caption="mira @marca #marca", hashtags=("#marca",))


def test_p42_5_given_clips_burns_lyrics_into_mp4(tmp_path: Path) -> None:
    """El MP4 final de given_clips LYRIC_VIDEO muestra las letras quemadas.

    El gate puede bloquear la exportación por reglas ajenas al formato
    (p. ej. ``artifact.video_stream`` no aplica fuera de ``Format.VIDEO``);
    lo que este test exige es que el veredicto de letras sea un PASS real:
    ``subtitles.spelling_lock`` en PASS y píxeles brillantes en el MP4.
    """
    registry = _register_lyric_assets(tmp_path)
    request = RunRequest(
        brief="cita del brief\nlyric video con clip negro y letra de prueba",
        destination=tmp_path / "delivery",
        environment=EnvironmentReport(ffmpeg_version="test", ffprobe_version="test"),
        model_version="static",
        prompt_version=PROMPT_VERSION,
        caption_prompt_version=CAPTION_PROMPT_VERSION,
        run_id="p42-5-burn",
        gate=Gate(FFprobeProbe()),
        registry=registry,
    )
    result = run_given_clips(
        model=_StaticModel(_lyric_draft(), _caption()),
        settings=Settings.from_root(tmp_path),
        request=request,
    )
    assert result.resolution_status is ResolutionStatus.RESOLVED
    artifact = tmp_path / "runs" / "p42-5-burn" / "artifacts" / "tiktok" / "clip-01.mp4"
    assert artifact.is_file()
    sidecar = artifact.with_suffix(".ass")
    assert sidecar.is_file()

    manifest = cast(
        "dict[str, object]",
        json.loads(Path(result.manifest_path).read_text(encoding="utf-8")),
    )
    gates = cast("list[dict[str, object]]", manifest["gates"])
    checks = cast("list[dict[str, object]]", gates[0]["checks"])
    lyric_checks = [check for check in checks if check["id"] == "subtitles.spelling_lock"]
    assert lyric_checks
    assert lyric_checks[0]["status"] == "pass"

    capture = cv2.VideoCapture(str(artifact))
    try:
        assert capture.isOpened()
        _ = capture.set(cv2.CAP_PROP_POS_MSEC, 2500.0)
        ok, frame = capture.read()
    finally:
        capture.release()
    assert ok
    assert np.any(frame > 50)


class _LegacyAssembler:
    """Ensamblador anterior a P42-5: no acepta subtítulos para quemar."""

    def assemble(
        self,
        *,
        clip: Path,
        destination: Path,
        watermark: Path | None,
        watermark_config: Watermark | None = None,
        mute_audio: bool = False,
    ) -> Path:
        _ = (self, clip, watermark, watermark_config, mute_audio)
        destination.parent.mkdir(parents=True, exist_ok=True)
        _ = destination.write_bytes(b"video")
        return destination

    def render_arguments(
        self,
        *,
        clip: Path,
        destination: Path,
        watermark: Path | None,
        watermark_config: Watermark | None = None,
        mute_audio: bool = False,
    ) -> tuple[str, ...]:
        _ = (self, clip, watermark, watermark_config, mute_audio)
        return ("ffmpeg", str(destination))


def test_p42_5_assembler_without_subtitle_support_fails_closed(tmp_path: Path) -> None:
    """LYRIC_VIDEO con un ensamblador sin quemado falla en cerrado, nunca PASS."""
    registry = _register_lyric_assets(tmp_path)
    request = RunRequest(
        brief="cita del brief\nlyric video con ensamblador sin soporte de subtítulos",
        destination=tmp_path / "delivery",
        environment=EnvironmentReport(),
        model_version="static",
        prompt_version=PROMPT_VERSION,
        caption_prompt_version=CAPTION_PROMPT_VERSION,
        run_id="p42-5-legacy",
        # El doble cast simula una inyección legacy anterior a P42-5: el objeto
        # no declara `subtitles` y el pipeline debe rechazarlo en cerrado.
        assembler=cast("PieceAssembler", cast("object", _LegacyAssembler())),
        gate=Gate(FFprobeProbe()),
        registry=registry,
    )
    with pytest.raises(NotImplementedError, match="subt"):
        _ = run_given_clips(
            model=_StaticModel(_lyric_draft(), _caption()),
            settings=Settings.from_root(tmp_path),
            request=request,
        )


def test_p42_5_empty_lyric_window_fails_closed_before_render(tmp_path: Path) -> None:
    """Ventana sin letras: falla antes del render y no publica ningún MP4."""
    registry = _register_lyric_assets(tmp_path)
    brief = "cita del brief\nlyric video del segundo 50 al segundo 55 con clip negro"
    ranges = [
        TimestampRangeDraft(
            start_sec=_cand_at(50.0, "50", brief.index("50")),
            end_sec=_cand_at(55.0, "55", brief.index("55")),
        )
    ]
    request = RunRequest(
        brief=brief,
        destination=tmp_path / "delivery",
        environment=EnvironmentReport(),
        model_version="static",
        prompt_version=PROMPT_VERSION,
        caption_prompt_version=CAPTION_PROMPT_VERSION,
        run_id="p42-5-empty",
        assembler=cast("PieceAssembler", cast("object", _LegacyAssembler())),
        gate=Gate(FakeProbe(info=make_media())),
        registry=registry,
    )
    with pytest.raises(PipelineError, match="letras"):
        _ = run_given_clips(
            model=_StaticModel(_lyric_draft(timestamp_ranges=ranges), _caption()),
            settings=Settings.from_root(tmp_path),
            request=request,
        )
    assert not (tmp_path / "runs" / "p42-5-empty" / "artifacts" / "tiktok" / "clip-01.mp4").exists()
