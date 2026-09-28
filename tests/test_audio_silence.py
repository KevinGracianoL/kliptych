"""Hal C3: verificación de silencio real con `volumedetect` en el gate.

El gate no confía en que se pasara `-af volume=0` a ffmpeg: mide el
`max_volume` real del MP4 final cuando
`audio_policy=internal_official_sound`. Fail-closed: sin ffmpeg, sin archivo
o sin medición, el resultado es FAIL (REJECTED).
"""

import shutil
import subprocess
from collections.abc import Sequence
from pathlib import Path

import pytest

from kliptych.assets import AssetRegistry
from kliptych.contract import AudioPolicy, Contract, Platform
from kliptych.gate import CheckStatus, Gate, GateStatus
from kliptych.gate.checks import GateContext, check_audio_silence
from tests.support import ALL_HARD_RULES, FakeProbe, make_contract, make_media, make_piece

_SILENT_STDERR = (
    "[Parsed_volumedetect_0 @ 0x7fab] n: 48000 | mean_volume: -91.0 dB | max_volume: -91.0 dB"
)
_LOUD_STDERR = (
    "[Parsed_volumedetect_0 @ 0x7fab] n: 48000 | mean_volume: -25.3 dB | max_volume: -20.0 dB"
)

_FFMPEG_MISSING = shutil.which("ffmpeg") is None


def _artifact(tmp_path: Path) -> Path:
    path = tmp_path / "final.mp4"
    _ = path.write_bytes(b"video")
    return path


def _context(tmp_path: Path, *, policy: AudioPolicy | None) -> GateContext:
    contract = make_contract(audio_rule="any", audio_policy=policy)
    artifact = _artifact(tmp_path)
    return GateContext(
        contract=contract,
        rules=contract.platforms[Platform.TIKTOK],
        piece=make_piece(artifact),
        artifact_sha256="a" * 64,
        media=make_media(),
        assets=AssetRegistry(tmp_path),
    )


def _mock_ffmpeg(
    monkeypatch: pytest.MonkeyPatch,
    *,
    stderr: str = _SILENT_STDERR,
    returncode: int = 0,
    which: str | None = "ffmpeg",
    ffprobe_stdout: str = "0\n",
    ffprobe_returncode: int = 0,
) -> list[list[str]]:
    captured: list[list[str]] = []

    def fake_run(argv: Sequence[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        _ = kwargs
        assert isinstance(argv, list)
        captured.append(list(argv))
        if "-select_streams" in argv:
            return subprocess.CompletedProcess(
                args=argv, returncode=ffprobe_returncode, stdout=ffprobe_stdout, stderr=""
            )
        return subprocess.CompletedProcess(
            args=argv, returncode=returncode, stdout="", stderr=stderr
        )

    def fake_which(_: str) -> str | None:
        return which

    monkeypatch.setattr("kliptych.gate.checks.subprocess.run", fake_run)
    monkeypatch.setattr("kliptych.gate.checks.shutil.which", fake_which)
    return captured


def test_silent_render_passes_with_argv_list_and_no_shell(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen_kwargs: list[dict[str, object]] = []
    original = _mock_ffmpeg(monkeypatch)

    def fake_run(argv: Sequence[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        seen_kwargs.append(dict(kwargs))
        assert isinstance(argv, list)
        original.append(list(argv))
        if "-select_streams" in argv:
            return subprocess.CompletedProcess(args=argv, returncode=0, stdout="0\n", stderr="")
        return subprocess.CompletedProcess(
            args=argv, returncode=0, stdout="", stderr=_SILENT_STDERR
        )

    monkeypatch.setattr("kliptych.gate.checks.subprocess.run", fake_run)
    outcome = check_audio_silence(_context(tmp_path, policy=AudioPolicy.INTERNAL_OFFICIAL_SOUND))
    assert outcome.status is CheckStatus.PASS
    assert len(original) == 2
    probe_argv, measure_argv = original
    assert "-select_streams" in probe_argv
    assert "stream=index" in probe_argv
    assert "-af" in measure_argv
    assert "volumedetect" in measure_argv
    assert "0:a:0" in measure_argv
    assert "-f" in measure_argv
    assert "null" in measure_argv
    assert all("shell" not in kwargs for kwargs in seen_kwargs)


def test_loud_render_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _ = _mock_ffmpeg(monkeypatch, stderr=_LOUD_STDERR)
    outcome = check_audio_silence(_context(tmp_path, policy=AudioPolicy.INTERNAL_OFFICIAL_SOUND))
    assert outcome.status is CheckStatus.FAIL


def test_threshold_boundary(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    context = _context(tmp_path, policy=AudioPolicy.INTERNAL_OFFICIAL_SOUND)
    _ = _mock_ffmpeg(monkeypatch, stderr="max_volume: -80.0 dB")
    assert check_audio_silence(context).status is CheckStatus.PASS
    _ = _mock_ffmpeg(monkeypatch, stderr="max_volume: -79.9 dB")
    assert check_audio_silence(context).status is CheckStatus.FAIL


def test_digital_silence_minus_inf_passes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _ = _mock_ffmpeg(monkeypatch, stderr="max_volume: -inf dB")
    outcome = check_audio_silence(_context(tmp_path, policy=AudioPolicy.INTERNAL_OFFICIAL_SOUND))
    assert outcome.status is CheckStatus.PASS


def test_non_internal_policy_passes_without_ffmpeg(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def exploding_run(argv: Sequence[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        _ = (argv, kwargs)
        msg = "no debe invocar ffmpeg sin internal_official_sound"
        raise AssertionError(msg)

    monkeypatch.setattr("kliptych.gate.checks.subprocess.run", exploding_run)
    for policy in (AudioPolicy.ORIGINAL_AUDIO, AudioPolicy.ANY_AUDIO, None):
        outcome = check_audio_silence(_context(tmp_path, policy=policy))
        assert outcome.status is CheckStatus.PASS


def test_missing_ffmpeg_is_fail_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _ = _mock_ffmpeg(monkeypatch, which=None)
    outcome = check_audio_silence(_context(tmp_path, policy=AudioPolicy.INTERNAL_OFFICIAL_SOUND))
    assert outcome.status is CheckStatus.FAIL


def test_missing_artifact_is_fail_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _ = _mock_ffmpeg(monkeypatch)
    context = _context(tmp_path, policy=AudioPolicy.INTERNAL_OFFICIAL_SOUND)
    missing = context.piece.model_copy(update={"artifact_path": tmp_path / "ausente.mp4"})
    context = GateContext(
        contract=context.contract,
        rules=context.rules,
        piece=missing,
        artifact_sha256=None,
        media=None,
        assets=context.assets,
    )
    assert check_audio_silence(context).status is CheckStatus.FAIL


def test_ffmpeg_error_is_fail_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _ = _mock_ffmpeg(monkeypatch, stderr="moov atom not found", returncode=1)
    outcome = check_audio_silence(_context(tmp_path, policy=AudioPolicy.INTERNAL_OFFICIAL_SOUND))
    assert outcome.status is CheckStatus.FAIL


def test_unparseable_output_is_fail_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _ = _mock_ffmpeg(monkeypatch, stderr="max_volume: n/a")
    outcome = check_audio_silence(_context(tmp_path, policy=AudioPolicy.INTERNAL_OFFICIAL_SOUND))
    assert outcome.status is CheckStatus.FAIL


def test_two_audio_tracks_rejected_without_measuring(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured = _mock_ffmpeg(monkeypatch, ffprobe_stdout="0\n1\n")
    outcome = check_audio_silence(_context(tmp_path, policy=AudioPolicy.INTERNAL_OFFICIAL_SOUND))
    assert outcome.status is CheckStatus.FAIL
    assert all("volumedetect" not in argv for argv in captured)
    probe_argv = next(argv for argv in captured if "-select_streams" in argv)
    assert "stream=index" in probe_argv
    assert "csv=p=0" in probe_argv


def test_zero_audio_tracks_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    captured = _mock_ffmpeg(monkeypatch, ffprobe_stdout="")
    outcome = check_audio_silence(_context(tmp_path, policy=AudioPolicy.INTERNAL_OFFICIAL_SOUND))
    assert outcome.status is CheckStatus.FAIL
    assert all("volumedetect" not in argv for argv in captured)


@pytest.mark.parametrize("ffprobe_stdout", ["0\n", "0 \n", "\n0\n\n"])
def test_single_audio_track_is_measured_with_explicit_map(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, ffprobe_stdout: str
) -> None:
    captured = _mock_ffmpeg(monkeypatch, ffprobe_stdout=ffprobe_stdout)
    outcome = check_audio_silence(_context(tmp_path, policy=AudioPolicy.INTERNAL_OFFICIAL_SOUND))
    assert outcome.status is CheckStatus.PASS
    measure_argv = next(argv for argv in captured if "volumedetect" in argv)
    assert "0:a:0" in measure_argv


def test_ffprobe_failure_is_fail_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    captured = _mock_ffmpeg(monkeypatch, ffprobe_returncode=1, ffprobe_stdout="")
    outcome = check_audio_silence(_context(tmp_path, policy=AudioPolicy.INTERNAL_OFFICIAL_SOUND))
    assert outcome.status is CheckStatus.FAIL
    assert all("volumedetect" not in argv for argv in captured)


def test_ffprobe_crash_is_fail_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def crashing_run(argv: Sequence[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        _ = kwargs
        assert isinstance(argv, list)
        if "-select_streams" in argv:
            msg = "ffprobe"
            raise FileNotFoundError(msg)
        return subprocess.CompletedProcess(
            args=argv, returncode=0, stdout="", stderr=_SILENT_STDERR
        )

    monkeypatch.setattr("kliptych.gate.checks.subprocess.run", crashing_run)
    outcome = check_audio_silence(_context(tmp_path, policy=AudioPolicy.INTERNAL_OFFICIAL_SOUND))
    assert outcome.status is CheckStatus.FAIL


def test_ffmpeg_crash_is_fail_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def crashing_run(argv: Sequence[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        _ = (argv, kwargs)
        msg = "ffmpeg"
        raise FileNotFoundError(msg)

    monkeypatch.setattr("kliptych.gate.checks.subprocess.run", crashing_run)
    outcome = check_audio_silence(_context(tmp_path, policy=AudioPolicy.INTERNAL_OFFICIAL_SOUND))
    assert outcome.status is CheckStatus.FAIL


def _silence_contract() -> Contract:
    return make_contract(
        audio_rule="any",
        audio_policy=AudioPolicy.INTERNAL_OFFICIAL_SOUND,
        hard=[*ALL_HARD_RULES, "audio.policy", "audio.silence"],
    )


def test_gate_rejects_loud_render(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _ = _mock_ffmpeg(monkeypatch, stderr=_LOUD_STDERR)
    result = Gate(FakeProbe(info=make_media())).run(
        contract=_silence_contract(),
        piece=make_piece(_artifact(tmp_path)),
        assets=AssetRegistry(tmp_path),
    )
    assert result.status is GateStatus.REJECTED
    silence = next(check for check in result.checks if check.id == "audio.silence")
    assert silence.status is CheckStatus.FAIL


def test_gate_pends_review_for_verified_silent_render(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _ = _mock_ffmpeg(monkeypatch, stderr=_SILENT_STDERR)
    result = Gate(FakeProbe(info=make_media())).run(
        contract=_silence_contract(),
        piece=make_piece(_artifact(tmp_path)),
        assets=AssetRegistry(tmp_path),
    )
    silence = next(check for check in result.checks if check.id == "audio.silence")
    assert silence.status is CheckStatus.PASS
    assert result.status is GateStatus.PENDING_REVIEW


def _run_ffmpeg(argv: list[str]) -> None:
    completed = subprocess.run(argv, capture_output=True, text=True, timeout=120.0, check=False)
    assert completed.returncode == 0, f"ffmpeg falló: {completed.stderr[-500:]}"


@pytest.mark.skipif(_FFMPEG_MISSING, reason="requiere ffmpeg en el PATH")
def test_integration_muted_mp4_measures_silent(tmp_path: Path) -> None:
    loud = tmp_path / "loud.mp4"
    _run_ffmpeg(
        [
            "ffmpeg",
            "-hide_banner",
            "-nostdin",
            "-v",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc=duration=1:size=320x240:rate=30",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:duration=1",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-shortest",
            str(loud),
        ]
    )
    muted = tmp_path / "muted.mp4"
    _run_ffmpeg(
        [
            "ffmpeg",
            "-hide_banner",
            "-nostdin",
            "-v",
            "error",
            "-y",
            "-i",
            str(loud),
            "-map",
            "0:v:0",
            "-map",
            "0:a?",
            "-c:v",
            "copy",
            "-af",
            "volume=0",
            "-c:a",
            "aac",
            "-movflags",
            "+faststart",
            str(muted),
        ]
    )
    base = _context(tmp_path, policy=AudioPolicy.INTERNAL_OFFICIAL_SOUND)

    def _for(path: Path) -> GateContext:
        return GateContext(
            contract=base.contract,
            rules=base.rules,
            piece=make_piece(path),
            artifact_sha256=None,
            media=None,
            assets=base.assets,
        )

    assert check_audio_silence(_for(muted)).status is CheckStatus.PASS
    assert check_audio_silence(_for(loud)).status is CheckStatus.FAIL


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="requiere ffmpeg en el PATH")
def test_integration_two_audio_tracks_rejected(tmp_path: Path) -> None:
    loud = tmp_path / "loud.mp4"
    _run_ffmpeg(
        [
            "ffmpeg",
            "-hide_banner",
            "-nostdin",
            "-v",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc=duration=1:size=320x240:rate=30",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:duration=1",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-shortest",
            str(loud),
        ]
    )
    muted = tmp_path / "muted.mp4"
    _run_ffmpeg(
        [
            "ffmpeg",
            "-hide_banner",
            "-nostdin",
            "-v",
            "error",
            "-y",
            "-i",
            str(loud),
            "-map",
            "0:v:0",
            "-map",
            "0:a?",
            "-c:v",
            "copy",
            "-af",
            "volume=0",
            "-c:a",
            "aac",
            "-movflags",
            "+faststart",
            str(muted),
        ]
    )
    twoaudio = tmp_path / "twoaudio.mp4"
    _run_ffmpeg(
        [
            "ffmpeg",
            "-hide_banner",
            "-nostdin",
            "-v",
            "error",
            "-y",
            "-i",
            str(muted),
            "-i",
            str(loud),
            "-map",
            "0:v",
            "-map",
            "0:a",
            "-map",
            "1:a",
            "-c",
            "copy",
            str(twoaudio),
        ]
    )
    base = _context(tmp_path, policy=AudioPolicy.INTERNAL_OFFICIAL_SOUND)

    def _for(path: Path) -> GateContext:
        return GateContext(
            contract=base.contract,
            rules=base.rules,
            piece=make_piece(path),
            artifact_sha256=None,
            media=None,
            assets=base.assets,
        )

    outcome = check_audio_silence(_for(twoaudio))
    assert outcome.status is CheckStatus.FAIL
    assert outcome.evidence.get("audio_tracks") == 2
    result = Gate(FakeProbe(info=make_media())).run(
        contract=_silence_contract(),
        piece=make_piece(twoaudio),
        assets=AssetRegistry(tmp_path),
    )
    assert result.status is GateStatus.REJECTED
    silence = next(check for check in result.checks if check.id == "audio.silence")
    assert silence.status is CheckStatus.FAIL
