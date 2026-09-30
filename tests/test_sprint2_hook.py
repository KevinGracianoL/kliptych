"""Sprint 2 (Objetivo 4): reglas sin mapear y hook CB22.

Las reglas declaradas sin validador mecánico van a revisión humana con la
cita del brief que las motivó (``Contract.unmapped``). El hook CB22 exige
la palabra clave en subtítulos o texto en pantalla con inicio <= 3.0 s
(normalización NFKD + casefold); ausente o tardía falla. El volumen de los
primeros 3 s es nota informativa: jamás cambia el veredicto.
"""

import shutil
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING, cast

import pytest
from pydantic import ValidationError

import kliptych.gate.hook as hook_module
from kliptych.assets import AssetRegistry
from kliptych.contract import Platform, UnmappedRule
from kliptych.gate import (
    CheckStatus,
    Gate,
    GateStatus,
    SubtitleSegment,
    check_hook_keyword,
)
from kliptych.gate.checks import GateContext
from tests.support import FakeProbe, make_contract, make_media, make_piece

if TYPE_CHECKING:
    from collections.abc import Callable

_FFMPEG = shutil.which("ffmpeg")
_FFPROBE = shutil.which("ffprobe")
_NEEDS_TOOLS = _FFMPEG is None or _FFPROBE is None

_HOOK = "hook.keyword"


def _private(name: str) -> object:
    return cast("object", getattr(hook_module, name))


_measure_first_3s_volume = cast(
    "Callable[[Path], float | None]", _private("_measure_first_3s_volume")
)


def _artifact(tmp_path: Path) -> Path:
    path = tmp_path / "piece.mp4"
    _ = path.write_bytes(b"video")
    return path


def _segments(*items: tuple[str, float, float]) -> tuple[SubtitleSegment, ...]:
    return tuple(SubtitleSegment(text=text, start_s=start, end_s=end) for text, start, end in items)


def _hook_check(
    tmp_path: Path,
    *,
    hook_keyword: str | None,
    subtitle_segments: tuple[SubtitleSegment, ...] = (),
    screen_text_segments: tuple[SubtitleSegment, ...] = (),
    subtitle_text: str | None = None,
    artifact: Path | None = None,
) -> tuple[CheckStatus, dict[str, object]]:
    contract = make_contract(
        hard=["artifact.integrity", _HOOK] if hook_keyword else ["artifact.integrity"],
        audio_rule="any",
        min_s=None,
        max_s=None,
        required_mentions=(),
        required_hashtags=(),
        hook_keyword=hook_keyword,
    )
    piece = make_piece(
        _artifact(tmp_path) if artifact is None else artifact,
        subtitle_text=subtitle_text,
        subtitle_segments=subtitle_segments,
        screen_text_segments=screen_text_segments,
    )
    outcome = check_hook_keyword(
        GateContext(
            contract=contract,
            rules=contract.platforms[Platform.TIKTOK],
            piece=piece,
            artifact_sha256="a" * 64,
            media=make_media(),
            assets=AssetRegistry(tmp_path),
        )
    )
    return outcome.status, dict(outcome.evidence)


def test_unmapped_rule_with_citation_needs_review(tmp_path: Path) -> None:
    contract = make_contract(
        hard=["artifact.integrity"],
        manual_review=["caption.tone"],
        audio_rule="any",
        min_s=None,
        max_s=None,
        required_mentions=(),
        required_hashtags=(),
        unmapped=(("caption.tone", "usa un tono épico, parce"),),
    )
    result = Gate(FakeProbe(info=make_media())).run(
        contract=contract,
        piece=make_piece(_artifact(tmp_path)),
        assets=AssetRegistry(tmp_path),
    )
    matches = [check for check in result.checks if check.id == "caption.tone"]
    assert len(matches) == 1
    assert matches[0].status is CheckStatus.MANUAL_REVIEW
    assert matches[0].evidence["citations"] == ["usa un tono épico, parce"]
    assert result.status is GateStatus.PENDING_REVIEW


def test_unmapped_rule_without_citation_needs_review(tmp_path: Path) -> None:
    contract = make_contract(
        hard=["artifact.integrity"],
        manual_review=["caption.tone"],
        audio_rule="any",
        min_s=None,
        max_s=None,
        required_mentions=(),
        required_hashtags=(),
    )
    result = Gate(FakeProbe(info=make_media())).run(
        contract=contract,
        piece=make_piece(_artifact(tmp_path)),
        assets=AssetRegistry(tmp_path),
    )
    matches = [check for check in result.checks if check.id == "caption.tone"]
    assert len(matches) == 1
    assert matches[0].status is CheckStatus.MANUAL_REVIEW
    assert matches[0].evidence.get("citations", []) == []


def test_hook_keyword_at_1_5s_passes(tmp_path: Path) -> None:
    status, evidence = _hook_check(
        tmp_path,
        hook_keyword="mira esto",
        subtitle_segments=_segments(("hola", 0.0, 1.0), ("mira esto parce", 1.5, 2.5)),
    )
    assert status is CheckStatus.PASS
    assert evidence["matched_at_s"] == [1.5]
    assert "volume_first_3s_db" in evidence


def test_hook_keyword_at_4_2s_fails(tmp_path: Path) -> None:
    status, _ = _hook_check(
        tmp_path,
        hook_keyword="mira esto",
        subtitle_segments=_segments(("hola", 0.0, 1.0), ("mira esto parce", 4.2, 5.0)),
    )
    assert status is CheckStatus.FAIL


def test_hook_keyword_absent_fails(tmp_path: Path) -> None:
    status, evidence = _hook_check(
        tmp_path,
        hook_keyword="mira esto",
        subtitle_segments=_segments(("hola", 0.0, 1.0), ("seguimos", 1.5, 2.5)),
    )
    assert status is CheckStatus.FAIL
    assert evidence["window_s"] == pytest.approx(3.0)


def test_hook_keyword_in_screen_text_passes(tmp_path: Path) -> None:
    status, _ = _hook_check(
        tmp_path,
        hook_keyword="mira esto",
        screen_text_segments=_segments(("MIRA ESTO", 0.5, 2.0)),
    )
    assert status is CheckStatus.PASS


def test_hook_keyword_normalizes_accents_and_case(tmp_path: Path) -> None:
    status, _ = _hook_check(
        tmp_path,
        hook_keyword="¡ACTÍVA la racha!",
        subtitle_segments=_segments(("grité ¡activa la racha! ya", 2.0, 2.8)),
    )
    assert status is CheckStatus.PASS


def test_hook_inactive_without_keyword_passes(tmp_path: Path) -> None:
    status, _ = _hook_check(tmp_path, hook_keyword=None)
    assert status is CheckStatus.PASS


def test_hook_flat_subtitles_without_segments_fails(tmp_path: Path) -> None:
    status, _ = _hook_check(
        tmp_path,
        hook_keyword="mira esto",
        subtitle_text="mira esto parce",
    )
    assert status is CheckStatus.FAIL


def test_hook_volume_is_info_only(tmp_path: Path) -> None:
    status, evidence = _hook_check(
        tmp_path,
        hook_keyword="mira esto",
        subtitle_segments=_segments(("mira esto", 1.0, 2.0)),
    )
    assert status is CheckStatus.PASS
    assert (
        evidence["volume_note"]
        == "informativo: el volumen de los primeros 3 s no modifica el veredicto"
    )
    status, evidence = _hook_check(
        tmp_path,
        hook_keyword="mira esto",
        subtitle_segments=_segments(("otra cosa", 1.0, 2.0)),
    )
    assert status is CheckStatus.FAIL
    assert "volume_first_3s_db" in evidence


def test_hook_failure_rejects_piece(tmp_path: Path) -> None:
    contract = make_contract(
        hard=["artifact.integrity", _HOOK],
        audio_rule="any",
        min_s=None,
        max_s=None,
        required_mentions=(),
        required_hashtags=(),
        hook_keyword="mira esto",
    )
    piece = make_piece(
        _artifact(tmp_path),
        subtitle_segments=_segments(("otra cosa", 4.2, 5.0)),
    )
    result = Gate(FakeProbe(info=make_media())).run(
        contract=contract, piece=piece, assets=AssetRegistry(tmp_path)
    )
    assert result.status is GateStatus.REJECTED


def test_h2_hook_keyword_gana_fails_with_ganador(tmp_path: Path) -> None:
    status, evidence = _hook_check(
        tmp_path,
        hook_keyword="gana",
        subtitle_segments=_segments(("el ganador", 1.0, 2.0)),
    )
    assert status is CheckStatus.FAIL
    assert "no aparece" in cast("str", evidence["reason"])


def test_h2_hook_keyword_split_across_segments_passes(tmp_path: Path) -> None:
    status, evidence = _hook_check(
        tmp_path,
        hook_keyword="gana ya",
        subtitle_segments=_segments(("gana", 1.0, 1.5), ("ya", 1.8, 2.3)),
    )
    assert status is CheckStatus.PASS
    assert evidence["matched_at_s"] == [1.0]
    assert evidence["matched_in"] == ["subtitles"]


@pytest.mark.integration
@pytest.mark.skipif(_NEEDS_TOOLS, reason="ffmpeg/ffprobe no disponibles")
def test_hook_volume_measured_on_real_clip(tmp_path: Path) -> None:
    assert _FFMPEG is not None
    clip = tmp_path / "audio.mp4"
    argv = [
        _FFMPEG,
        "-y",
        "-nostdin",
        "-v",
        "error",
        "-f",
        "lavfi",
        "-i",
        "color=c=blue:s=320x240:d=2",
        "-f",
        "lavfi",
        "-i",
        "sine=frequency=440:duration=2",
        "-shortest",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        str(clip),
    ]
    _ = subprocess.run(argv, capture_output=True, check=True, timeout=120)
    status, evidence = _hook_check(
        tmp_path,
        hook_keyword="mira esto",
        subtitle_segments=_segments(("mira esto", 1.0, 2.0)),
        artifact=clip,
    )
    assert status is CheckStatus.PASS
    assert isinstance(evidence["volume_first_3s_db"], float)


def test_unmapped_rule_in_hard_with_citation_needs_review(tmp_path: Path) -> None:
    contract = make_contract(
        hard=["artifact.integrity", "caption.tone"],
        audio_rule="any",
        min_s=None,
        max_s=None,
        required_mentions=(),
        required_hashtags=(),
        unmapped=(("caption.tone", "usa un tono épico, parce"),),
    )
    result = Gate(FakeProbe(info=make_media())).run(
        contract=contract,
        piece=make_piece(_artifact(tmp_path)),
        assets=AssetRegistry(tmp_path),
    )
    matches = [check for check in result.checks if check.id == "caption.tone"]
    assert len(matches) == 1
    assert matches[0].status is CheckStatus.MANUAL_REVIEW
    assert matches[0].evidence["citations"] == ["usa un tono épico, parce"]
    assert result.status is GateStatus.PENDING_REVIEW


def test_hook_keyword_normalizing_to_empty_fails(tmp_path: Path) -> None:
    status, evidence = _hook_check(
        tmp_path,
        hook_keyword="\u0300\u0301",
        subtitle_segments=_segments(("hola", 1.0, 2.0)),
    )
    assert status is CheckStatus.FAIL
    assert evidence["reason"] == "la palabra clave quedó vacía tras normalizar"


def test_measure_volume_missing_file_returns_none(tmp_path: Path) -> None:
    assert _measure_first_3s_volume(tmp_path / "non_existent.mp4") is None


def test_measure_volume_missing_ffmpeg_returns_none(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _no_tool(_cmd: str) -> None:
        return None

    monkeypatch.setattr(shutil, "which", _no_tool)
    video = _artifact(tmp_path)
    assert _measure_first_3s_volume(video) is None


def test_measure_volume_timeout_or_oserror_returns_none(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _explode(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        _ = (args, kwargs)
        raise subprocess.TimeoutExpired(cmd="ffmpeg", timeout=30.0)

    monkeypatch.setattr(subprocess, "run", _explode)
    video = _artifact(tmp_path)
    assert _measure_first_3s_volume(video) is None


def test_measure_volume_unparseable_float_returns_none(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _bad_output(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        _ = (args, kwargs)
        return subprocess.CompletedProcess(
            args=["ffmpeg"], returncode=0, stdout="", stderr="max_volume: not_a_number dB"
        )

    monkeypatch.setattr(subprocess, "run", _bad_output)
    video = _artifact(tmp_path)
    assert _measure_first_3s_volume(video) is None


def test_subtitle_segment_validation() -> None:
    with pytest.raises(ValueError, match="la ventana del segmento debe ser finita"):
        _ = SubtitleSegment(text="hola", start_s=0.0, end_s=float("inf"))
    with pytest.raises(ValueError, match=r"start_s \(2\.0\) debe ser menor que end_s \(1\.0\)"):
        _ = SubtitleSegment(text="hola", start_s=2.0, end_s=1.0)
    with pytest.raises(ValueError, match=r"start_s \(2\.0\) debe ser menor que end_s \(2\.0\)"):
        _ = SubtitleSegment(text="hola", start_s=2.0, end_s=2.0)


def test_contract_hook_and_unmapped_schema_validation() -> None:
    with pytest.raises(ValueError, match="hook_keyword no puede ser vacío o solo espacios"):
        _ = make_contract(
            hard=["artifact.integrity"],
            audio_rule="any",
            min_s=None,
            max_s=None,
            required_mentions=(),
            required_hashtags=(),
            hook_keyword="   ",
        )
    with pytest.raises(ValidationError, match="String should have at least 1 character"):
        _ = UnmappedRule(rule="", quote="cita")
    with pytest.raises(ValidationError, match="String should have at least 1 character"):
        _ = UnmappedRule(rule="regla", quote="")

    contract = make_contract(
        hard=["artifact.integrity", _HOOK],
        audio_rule="any",
        min_s=None,
        max_s=None,
        required_mentions=(),
        required_hashtags=(),
        hook_keyword="mira esto",
        unmapped=(("caption.tone", "épico"),),
    )
    dump = contract.model_dump(mode="python")
    assert dump["hook_keyword"] == "mira esto"
    assert dump["unmapped"] == ({"rule": "caption.tone", "quote": "épico"},)


def test_h1_unmapped_outside_rules_forces_pending_review(tmp_path: Path) -> None:
    contract = make_contract(
        hard=["artifact.integrity"],
        audio_rule="any",
        min_s=None,
        max_s=None,
        required_mentions=(),
        required_hashtags=(),
        unmapped=(("caption.tone", "tono épico"),),
    )
    result = Gate(FakeProbe(info=make_media())).run(
        contract=contract,
        piece=make_piece(_artifact(tmp_path)),
        assets=AssetRegistry(tmp_path),
    )
    assert result.status is GateStatus.PENDING_REVIEW
    matches = [check for check in result.checks if check.id == "rules.unmapped"]
    assert len(matches) == 1
    assert matches[0].status is CheckStatus.MANUAL_REVIEW
    assert matches[0].evidence["rules"] == ["caption.tone"]
    assert matches[0].evidence["citations"] == ["tono épico"]


def test_h1_contract_rejects_unmapped_colliding_with_known_validators() -> None:
    for rule_id in (
        "artifact.integrity",
        "caption.forbidden",
        "watermark.present",
        " Artifact.Integrity ",
        " artifact.integrity ",
    ):
        with pytest.raises(ValueError, match="colisiona con un validador conocido"):
            _ = make_contract(
                hard=["artifact.integrity"],
                audio_rule="any",
                min_s=None,
                max_s=None,
                required_mentions=(),
                required_hashtags=(),
                unmapped=((rule_id, "cita"),),
            )


def test_h1_engine_with_colliding_unmapped_is_never_passed(tmp_path: Path) -> None:
    base = make_contract(
        hard=["artifact.integrity"],
        audio_rule="any",
        min_s=None,
        max_s=None,
        required_mentions=(),
        required_hashtags=(),
    )
    bypassed_contract = base.model_copy(
        update={"unmapped": (UnmappedRule(rule="artifact.integrity", quote="cita"),)}
    )
    result = Gate(FakeProbe(info=make_media())).run(
        contract=bypassed_contract,
        piece=make_piece(_artifact(tmp_path)),
        assets=AssetRegistry(tmp_path),
    )
    assert result.status is not GateStatus.PASSED
