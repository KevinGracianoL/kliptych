"""Tests del hallazgo 4: AudioRule.OWN_CLIP y NO_TRENDING son revisión manual.

El gate no puede verificar mecánicamente el origen/uso del audio, así que el
contrato proyecta reglas manual_review explícitas y el gate exige confirmación
humana.
"""

from pathlib import Path

import pytest

from kliptych.assets import AssetRegistry
from kliptych.contract import AudioRule
from kliptych.gate import CheckStatus, Gate, GateStatus
from kliptych.resolver import resolve_contract
from tests.support import FakeProbe, candidate, make_contract, make_draft, make_media, make_piece


def _probe_15s() -> FakeProbe:
    return FakeProbe(info=make_media(duration_s=15.0, has_video=True, has_audio=True))


def _artifact(tmp_path: Path) -> Path:
    path = tmp_path / "piece.mp4"
    _ = path.write_bytes(b"video payload")
    return path


@pytest.mark.parametrize(
    ("audio_rule", "rule_id"),
    [
        (AudioRule.OWN_CLIP, "audio.own_clip"),
        (AudioRule.NO_TRENDING, "audio.no_trending"),
    ],
)
def test_audio_rule_projects_manual_review_rule(
    audio_rule: AudioRule, rule_id: str, tmp_path: Path
) -> None:
    """OWN_CLIP y NO_TRENDING generan reglas manual_review explícitas."""
    _ = tmp_path
    contract = make_contract(audio_rule=audio_rule.value)
    assert rule_id in contract.rules.manual_review
    assert rule_id not in contract.rules.hard


@pytest.mark.parametrize(
    ("audio_rule", "rule_id"),
    [
        (AudioRule.OWN_CLIP, "audio.own_clip"),
        (AudioRule.NO_TRENDING, "audio.no_trending"),
    ],
)
def test_audio_gate_requires_human_confirmation(
    audio_rule: AudioRule, rule_id: str, tmp_path: Path
) -> None:
    """El gate pide confirmación humana del origen/uso del audio."""
    contract = make_contract(audio_rule=audio_rule.value)
    piece = make_piece(_artifact(tmp_path), caption="mira @marca #marca")
    result = Gate(probe=_probe_15s()).evaluate_piece(
        piece, contract=contract, assets=AssetRegistry(tmp_path)
    )
    assert result.status is GateStatus.PENDING_REVIEW
    audio_check = next(check for check in result.checks if check.id == rule_id)
    assert audio_check.status is CheckStatus.MANUAL_REVIEW


@pytest.mark.parametrize(
    ("audio_rule", "rule_id"),
    [
        ("own_clip", "audio.own_clip"),
        ("no_trending", "audio.no_trending"),
    ],
)
def test_audio_rule_resolves_to_manual_review_by_default(
    audio_rule: str, rule_id: str, tmp_path: Path
) -> None:
    """El resolver clasifica la restricción de audio como manual_review."""
    draft = make_draft(
        platforms={
            "tiktok": {
                "duration": {"min_s": candidate(8)},
                "required_hashtags": candidate(["#marca"]),
                "required_mentions": candidate(["@marca"]),
                "audio_rule": candidate(audio_rule),
            }
        },
        rules=None,
    )
    result = resolve_contract(draft, registry=AssetRegistry(tmp_path))
    assert result.contract is not None
    assert rule_id in result.contract.rules.manual_review
