"""Tests de la política de audio por campaña (Fix 3 Sprint 1).

La campaña declara ``audio_policy`` (``internal_official_sound``,
``original_audio``, ``any_audio``). El gate evalúa el sonido oficial interno
como revisión manual (``PENDING_REVIEW``, jamás ``UNSUPPORTED`` ni ``PASS``
silencioso): exportar exige firma humana. Si además falla otra regla, el
fallo manda (``REJECTED``): fail-closed.
"""

import subprocess
from collections.abc import Sequence
from pathlib import Path

import pytest
from pydantic import ValidationError

from kliptych.assembler import FFmpegAssembler, RenderSpec
from kliptych.assets import AssetRegistry
from kliptych.contract import AudioPolicy, Contract, Platform, contract_digest
from kliptych.encoding import RenderConfig
from kliptych.exporter import ExportStatus, export_delivery
from kliptych.gate import CheckResult, CheckStatus, Gate, GateResult, GateStatus, Piece
from kliptych.orchestrator import PipelineConfig, compute_long_video_fingerprint
from kliptych.resolver import resolve_contract
from kliptych.subtitles import SubtitleRenderer
from tests.support import (
    ALL_HARD_RULES_WITHOUT_AUDIO,
    FakeProbe,
    candidate,
    make_contract,
    make_draft,
    make_media,
    mock_silent_volumedetect,
)


@pytest.fixture
def _silent_render(monkeypatch: pytest.MonkeyPatch) -> None:
    """Simula un render correctamente silenciado: cero ffmpeg real en este módulo.

    Estos tests afirman resultados del gate sobre contratos silenciados con
    artefactos falsos; la medición real con ffmpeg vive en
    ``tests/test_audio_silence.py``.
    """
    mock_silent_volumedetect(monkeypatch)


pytestmark = pytest.mark.usefixtures("_silent_render")


def _artifact(tmp_path: Path) -> Path:
    path = tmp_path / "piece.mp4"
    _ = path.write_bytes(b"video")
    return path


def _piece(artifact: Path, *, subtitle_text: str | None = None) -> Piece:
    return Piece(
        piece_id="piece-01",
        platform=Platform.TIKTOK,
        caption="mira @marca #marca",
        hashtags=(),
        subtitle_text=subtitle_text,
        artifact_path=artifact,
    )


def _policy_contract(*, spelling_locks: Sequence[str] = ()) -> Contract:
    return make_contract(
        audio_rule="any",
        audio_policy=AudioPolicy.INTERNAL_OFFICIAL_SOUND,
        manual_review=["audio.policy"],
        spelling_locks=spelling_locks,
    )


def _check(result: GateResult, rule_id: str) -> CheckResult:
    matched = [check for check in result.checks if check.id == rule_id]
    assert len(matched) == 1
    return matched[0]


def _run(contract: Contract, piece: Piece, root: Path) -> GateResult:
    return Gate(FakeProbe(info=make_media())).run(
        contract=contract,
        piece=piece,
        assets=AssetRegistry(root),
    )


def test_audio_policy_values() -> None:
    assert AudioPolicy.INTERNAL_OFFICIAL_SOUND.value == "internal_official_sound"
    assert AudioPolicy.ORIGINAL_AUDIO.value == "original_audio"
    assert AudioPolicy.ANY_AUDIO.value == "any_audio"


def test_internal_official_sound_is_manual_review(tmp_path: Path) -> None:
    result = _run(_policy_contract(), _piece(_artifact(tmp_path)), tmp_path)
    assert _check(result, "audio.policy").status is CheckStatus.MANUAL_REVIEW
    assert result.status is GateStatus.PENDING_REVIEW


def test_internal_official_sound_in_hard_is_never_unsupported(tmp_path: Path) -> None:
    contract = make_contract(
        audio_rule="any",
        audio_policy=AudioPolicy.INTERNAL_OFFICIAL_SOUND,
        hard=[*ALL_HARD_RULES_WITHOUT_AUDIO, "audio.policy"],
    )
    result = _run(contract, _piece(_artifact(tmp_path)), tmp_path)
    assert _check(result, "audio.policy").status is CheckStatus.MANUAL_REVIEW
    assert result.status is GateStatus.PENDING_REVIEW


def test_internal_official_sound_requires_human_signature(tmp_path: Path) -> None:
    contract = _policy_contract()
    piece = _piece(_artifact(tmp_path))
    gate = Gate(FakeProbe(info=make_media()))
    assets = AssetRegistry(tmp_path)

    blocked = export_delivery(
        contract=contract,
        pieces=[piece],
        gate=gate,
        assets=assets,
        destination=tmp_path / "delivery",
    )
    assert blocked.status is ExportStatus.BLOCKED
    assert "audio.policy" in blocked.rejected[0].reason

    approved = export_delivery(
        contract=contract,
        pieces=[piece],
        gate=gate,
        assets=assets,
        destination=tmp_path / "delivery",
        approve_manual_review=True,
        approved_by="operador-01",
    )
    assert approved.status is ExportStatus.EXPORTED
    assert approved.manually_approved_rules == ("audio.policy",)
    assert approved.approved_by == "operador-01"


def test_fail_closed_policy_plus_spelling_fail_is_rejected(tmp_path: Path) -> None:
    contract = _policy_contract(spelling_locks=["MarcaX"])
    piece = _piece(_artifact(tmp_path), subtitle_text="hablamos de maracax")
    result = _run(contract, piece, tmp_path)
    assert result.status is GateStatus.REJECTED
    assert _check(result, "subtitles.spelling_lock").status is CheckStatus.FAIL


@pytest.mark.parametrize("policy", [AudioPolicy.ORIGINAL_AUDIO, AudioPolicy.ANY_AUDIO, None])
def test_non_internal_policies_pass(tmp_path: Path, policy: AudioPolicy | None) -> None:
    contract = make_contract(
        audio_rule="any",
        audio_policy=policy,
        hard=[*ALL_HARD_RULES_WITHOUT_AUDIO, "audio.policy"],
    )
    result = _run(contract, _piece(_artifact(tmp_path)), tmp_path)
    assert _check(result, "audio.policy").status is CheckStatus.PASS
    assert result.status is GateStatus.PASSED


def test_schema_rejects_unclassified_policy_rule() -> None:
    data: dict[str, object] = make_contract().model_dump(mode="json")
    data["audio_policy"] = AudioPolicy.INTERNAL_OFFICIAL_SOUND.value
    with pytest.raises(ValidationError, match=r"audio\.policy"):
        _ = Contract.model_validate(data)


def test_resolver_defaults_policy_rule_to_manual_review(tmp_path: Path) -> None:
    draft = make_draft(audio_policy=candidate("internal_official_sound"))
    result = resolve_contract(draft, registry=AssetRegistry(tmp_path))
    assert result.contract is not None
    assert result.contract.audio_policy is AudioPolicy.INTERNAL_OFFICIAL_SOUND
    assert "audio.policy" in result.contract.rules.manual_review


def test_unset_policy_keeps_digest() -> None:
    assert contract_digest(make_contract()) == contract_digest(make_contract(audio_policy=None))


def test_declared_policy_changes_digest() -> None:
    assert contract_digest(
        make_contract(audio_policy=AudioPolicy.INTERNAL_OFFICIAL_SOUND)
    ) != contract_digest(make_contract())


def _render_paths(tmp_path: Path) -> tuple[Path, Path]:
    clip = tmp_path / "clip.mp4"
    _ = clip.write_bytes(b"video")
    return clip, tmp_path / "final.mp4"


def test_muted_assemble_preserves_audio_track_and_silences(tmp_path: Path) -> None:
    clip, destination = _render_paths(tmp_path)
    recipe = FFmpegAssembler().render_arguments(
        RenderSpec(clip=clip, destination=destination, mute_audio=True)
    )
    assert "-af" in recipe
    assert "volume=0" in recipe
    assert "0:a?" in recipe


def test_assemble_without_mute_keeps_audible_argv(tmp_path: Path) -> None:
    clip, destination = _render_paths(tmp_path)
    recipe = FFmpegAssembler().render_arguments(RenderSpec(clip=clip, destination=destination))
    assert "-af" not in recipe
    assert "volume=0" not in recipe


def test_muted_subtitle_burn_preserves_audio_track_and_silences(tmp_path: Path) -> None:
    clip, destination = _render_paths(tmp_path)
    subtitles = tmp_path / "subtitles.ass"
    _ = subtitles.write_text("[Events]\n", encoding="utf-8")
    recipe = SubtitleRenderer().render_arguments(
        video=clip, subtitles=subtitles, destination=destination, mute_audio=True
    )
    assert "-af" in recipe
    assert "volume=0" in recipe
    assert "0:a?" in recipe


def test_subtitle_burn_without_mute_keeps_audible_argv(tmp_path: Path) -> None:
    clip, destination = _render_paths(tmp_path)
    subtitles = tmp_path / "subtitles.ass"
    _ = subtitles.write_text("[Events]\n", encoding="utf-8")
    recipe = SubtitleRenderer().render_arguments(
        video=clip, subtitles=subtitles, destination=destination
    )
    assert "-af" not in recipe
    assert "volume=0" not in recipe


def test_muted_burn_reaches_ffmpeg_argv(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    captured: list[list[str]] = []

    def fake_run(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        _ = kwargs
        captured.append(list(argv))
        _ = Path(argv[-1]).write_bytes(b"final")
        return subprocess.CompletedProcess(args=argv, returncode=0, stdout="", stderr="")

    monkeypatch.setattr("kliptych.subtitles.subprocess.run", fake_run)
    clip, destination = _render_paths(tmp_path)
    subtitles = tmp_path / "subtitles.ass"
    _ = subtitles.write_text("[Events]\n", encoding="utf-8")
    _ = SubtitleRenderer(render=RenderConfig()).burn(
        video=clip, subtitles=subtitles, destination=destination, mute_audio=True
    )
    assert len(captured) == 1
    assert "volume=0" in captured[0]
    assert "0:a?" in captured[0]
    assert destination.is_file()


def test_audio_policy_change_changes_resume_fingerprint(tmp_path: Path) -> None:
    url = "https://example.com/video"
    audible = PipelineConfig(
        output_dir=tmp_path / "out",
        contract=make_contract(),
        render=RenderConfig(),
    )
    muted = PipelineConfig(
        output_dir=tmp_path / "out",
        contract=make_contract(audio_policy=AudioPolicy.INTERNAL_OFFICIAL_SOUND),
        render=RenderConfig(),
    )
    assert compute_long_video_fingerprint(url, config=audible) == (
        compute_long_video_fingerprint(url, config=audible)
    )
    assert compute_long_video_fingerprint(url, config=audible) != (
        compute_long_video_fingerprint(url, config=muted)
    )


def test_muted_track_passes_presence_check_and_pends_review(tmp_path: Path) -> None:
    contract = make_contract(audio_policy=AudioPolicy.INTERNAL_OFFICIAL_SOUND)
    result = _run(contract, _piece(_artifact(tmp_path)), tmp_path)
    assert _check(result, "audio.present").status is CheckStatus.PASS
    assert _check(result, "audio.policy").status is CheckStatus.MANUAL_REVIEW
    assert result.status is GateStatus.PENDING_REVIEW
