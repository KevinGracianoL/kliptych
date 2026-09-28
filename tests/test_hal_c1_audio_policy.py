"""Hal C1: el extractor conoce `audio_policy` y el resolver es fail-closed sin ella.

El prompt de extracción debe instruir la extracción de `audio_policy` con
evidencia, y el resolver debe clasificar `audio.policy` como `manual_review`
cuando el brief exige sonido oficial (`official_required`) pero no hay
política resuelta: jamás pasa en silencio sin política/mute.
"""

from pathlib import Path
from typing import cast

import pytest

from kliptych.assets import AssetRegistry
from kliptych.contract import ContractDraft
from kliptych.resolver import IssueCode, ResolutionStatus, resolve_contract
from kliptych.runtime import openai_compatible
from tests.support import candidate, conflict_candidate, make_draft


def _prompt_text(name: str) -> str:
    return cast("str", getattr(openai_compatible, name))


_EXTRACT_SYSTEM_PROMPT = _prompt_text("_EXTRACT_SYSTEM_PROMPT")


def test_extract_prompt_instructs_audio_policy_with_evidence() -> None:
    assert "audio_policy" in _EXTRACT_SYSTEM_PROMPT
    for value in ("internal_official_sound", "original_audio", "any_audio"):
        assert value in _EXTRACT_SYSTEM_PROMPT
    assert "evidence" in _EXTRACT_SYSTEM_PROMPT


def _official_required_draft(**overrides: object) -> ContractDraft:
    base: dict[str, object] = {
        "platforms": {
            "tiktok": {
                "duration": {"min_s": candidate(8)},
                "required_hashtags": candidate(["#marca"]),
                "required_mentions": candidate(["@marca"]),
                "audio_rule": candidate("official_required"),
            }
        },
        "official_audio": {"tiktok_url": candidate("https://example.com/audio.mp3")},
        "rules": None,
    }
    base.update(overrides)
    return make_draft(**base)


def test_official_required_without_policy_forces_policy_manual_review(tmp_path: Path) -> None:
    result = resolve_contract(_official_required_draft(), registry=AssetRegistry(tmp_path))
    assert result.status is ResolutionStatus.RESOLVED
    assert result.contract is not None
    assert "audio.policy" in result.contract.rules.manual_review
    defaulted = [issue.field for issue in result.issues if issue.code is IssueCode.RULE_DEFAULTED]
    assert "rules.audio.policy" in defaulted


def test_official_required_with_policy_does_not_force_review(tmp_path: Path) -> None:
    draft = _official_required_draft(audio_policy=candidate("original_audio"))
    result = resolve_contract(draft, registry=AssetRegistry(tmp_path))
    assert result.status is ResolutionStatus.RESOLVED
    assert result.contract is not None
    assert "audio.policy" not in result.contract.rules.manual_review


def test_conflicted_audio_policy_goes_to_manual_review(tmp_path: Path) -> None:
    draft = _official_required_draft(audio_policy=conflict_candidate())
    result = resolve_contract(draft, registry=AssetRegistry(tmp_path))
    assert result.status is ResolutionStatus.MANUAL_REVIEW
    conflicts = [issue.field for issue in result.issues if issue.code is IssueCode.CONFLICT]
    assert "audio_policy" in conflicts


def _any_audio_draft(**overrides: object) -> ContractDraft:
    base: dict[str, object] = {
        "platforms": {
            "tiktok": {
                "duration": {"min_s": candidate(8)},
                "required_hashtags": candidate(["#marca"]),
                "required_mentions": candidate(["@marca"]),
                "audio_rule": candidate("any"),
            }
        },
        "rules": None,
    }
    base.update(overrides)
    return make_draft(**base)


def _brief_with(mention: str) -> str:
    return f"cita del brief. La campaña usará {mention} en la publicación."


def test_brief_mentioning_official_sound_forces_policy_manual_review(tmp_path: Path) -> None:
    result = resolve_contract(
        _any_audio_draft(),
        registry=AssetRegistry(tmp_path),
        brief_text=_brief_with("official sound"),
    )
    assert result.status is ResolutionStatus.RESOLVED
    assert result.contract is not None
    assert "audio.policy" in result.contract.rules.manual_review
    defaulted = [issue.field for issue in result.issues if issue.code is IssueCode.RULE_DEFAULTED]
    assert "rules.audio.policy" in defaulted


@pytest.mark.parametrize(
    "mention",
    [
        "official sound",
        "Official Sound",
        "official audio",
        "sonido oficial",
        "SONIDO OFICIAL",
        "audio oficial",
        "Audio Oficial",
    ],
)
def test_brief_mention_variants_force_policy_manual_review(tmp_path: Path, mention: str) -> None:
    result = resolve_contract(
        _any_audio_draft(),
        registry=AssetRegistry(tmp_path),
        brief_text=_brief_with(mention),
    )
    assert result.status is ResolutionStatus.RESOLVED
    assert result.contract is not None
    assert "audio.policy" in result.contract.rules.manual_review


def test_brief_without_official_mention_does_not_force_review(tmp_path: Path) -> None:
    result = resolve_contract(
        _any_audio_draft(),
        registry=AssetRegistry(tmp_path),
        brief_text="cita del brief. La campaña usa audio libre sin restricciones.",
    )
    assert result.status is ResolutionStatus.RESOLVED
    assert result.contract is not None
    assert "audio.policy" not in result.contract.rules.manual_review


def test_official_mention_with_resolved_policy_does_not_force_review(tmp_path: Path) -> None:
    draft = _any_audio_draft(audio_policy=candidate("original_audio"))
    result = resolve_contract(
        draft,
        registry=AssetRegistry(tmp_path),
        brief_text=_brief_with("official sound"),
    )
    assert result.status is ResolutionStatus.RESOLVED
    assert result.contract is not None
    assert "audio.policy" not in result.contract.rules.manual_review
