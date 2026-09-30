"""Tests para contrato, schema y resolución de Format.LYRIC_VIDEO (Sprint 3 Parte 2 Objetivo 1).

Valida Format.LYRIC_VIDEO, LyricConfig, LyricConfigDraft, la instrucción en el
system prompt y la resolución con estricta procedencia T6 en resolver.py.
"""

from pathlib import Path
from typing import cast

import pytest
from pydantic import ValidationError

from kliptych.assets import AssetRegistry
from kliptych.contract import (
    Format,
    LyricConfig,
    contract_digest,
)
from kliptych.contract.draft import LyricConfigDraft
from kliptych.resolver import IssueCode, ResolutionStatus, resolve_contract
from kliptych.runtime import openai_compatible
from tests.support import candidate, make_contract, make_draft


def _prompt_text(name: str) -> str:
    return cast("str", getattr(openai_compatible, name))


def test_format_lyric_video_enum() -> None:
    assert Format.LYRIC_VIDEO == "lyric_video"
    assert Format.LYRIC_VIDEO.value == "lyric_video"


def test_lyric_config_valid_and_defaults() -> None:
    config = LyricConfig()
    assert config.lrc_asset_id is None
    assert config.track_name is None
    assert config.artist_name is None
    assert config.lrclib_enabled is True

    custom = LyricConfig(
        lrc_asset_id="lyrics_track_01",
        track_name="Canción Test",
        artist_name="Artista Test",
        lrclib_enabled=False,
    )
    assert custom.lrc_asset_id == "lyrics_track_01"
    assert custom.track_name == "Canción Test"
    assert custom.artist_name == "Artista Test"
    assert custom.lrclib_enabled is False


def test_lyric_config_rejects_unsafe_asset_id() -> None:
    with pytest.raises(ValidationError):
        _ = LyricConfig(lrc_asset_id="../escape_path")


def test_contract_includes_lyric_video() -> None:
    contract = make_contract(
        format_=Format.LYRIC_VIDEO,
        lyric_video={"track_name": "Mi Tema", "lrclib_enabled": True},
    )
    assert contract.format is Format.LYRIC_VIDEO
    assert contract.lyric_video is not None
    assert contract.lyric_video.track_name == "Mi Tema"


def test_contract_digest_empty_lyric_video_preserves_base() -> None:
    base = make_contract()
    with_none = make_contract(lyric_video=None)
    assert contract_digest(base) == contract_digest(with_none)


def test_contract_digest_changes_when_lyric_video_declared() -> None:
    base = make_contract()
    with_lyric = make_contract(
        lyric_video={"track_name": "Tema", "artist_name": "Artista", "lrclib_enabled": True}
    )
    assert contract_digest(base) != contract_digest(with_lyric)


def test_extract_prompt_instructs_lyric_video() -> None:
    prompt = _prompt_text("_EXTRACT_SYSTEM_PROMPT")
    assert "lyric_video" in prompt
    assert "lrc_asset_id" in prompt
    assert "track_name" in prompt
    assert "artist_name" in prompt


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


def test_resolver_resolves_lyric_video_from_draft(tmp_path: Path) -> None:
    brief = "cita del brief. generar un lyric video para la cancion Bohemian Rhapsody de Queen"
    track_quote = "Bohemian Rhapsody"
    artist_quote = "Queen"
    draft = make_draft(
        format=candidate("lyric_video"),
        lyric_video=LyricConfigDraft(
            track_name=_cand_at(track_quote, track_quote, brief.index(track_quote)),
            artist_name=_cand_at(artist_quote, artist_quote, brief.index(artist_quote)),
        ),
    )
    result = resolve_contract(draft, registry=AssetRegistry(tmp_path), brief_text=brief)
    assert result.status == ResolutionStatus.RESOLVED
    assert result.contract is not None
    assert result.contract.format is Format.LYRIC_VIDEO
    assert result.contract.lyric_video is not None
    assert result.contract.lyric_video.track_name == "Bohemian Rhapsody"
    assert result.contract.lyric_video.artist_name == "Queen"


def test_resolver_rejects_candidate_without_confidence(tmp_path: Path) -> None:
    brief = "cita del brief. lyric video de Song"
    draft = make_draft(
        format=candidate("lyric_video"),
        lyric_video=LyricConfigDraft(
            track_name={"value": "Song", "evidence": {"quote": "Song"}},
        ),
    )
    result = resolve_contract(draft, registry=AssetRegistry(tmp_path), brief_text=brief)
    assert result.status == ResolutionStatus.MANUAL_REVIEW
    assert any(issue.field.startswith("lyric_video") for issue in result.issues)


def test_resolver_rejects_candidate_confidence_missing(tmp_path: Path) -> None:
    brief = "cita del brief. lyric video de Song"
    draft = make_draft(
        format=candidate("lyric_video"),
        lyric_video=LyricConfigDraft(
            track_name={"value": "Song", "confidence": "missing"},
        ),
    )
    result = resolve_contract(draft, registry=AssetRegistry(tmp_path), brief_text=brief)
    assert result.status == ResolutionStatus.MANUAL_REVIEW
    assert any(issue.field.startswith("lyric_video") for issue in result.issues)


def test_resolver_rejects_candidate_confidence_conflict(tmp_path: Path) -> None:
    brief = "cita del brief. lyric video de Song"
    draft = make_draft(
        format=candidate("lyric_video"),
        lyric_video=LyricConfigDraft(
            track_name={"confidence": "conflict"},
        ),
    )
    result = resolve_contract(draft, registry=AssetRegistry(tmp_path), brief_text=brief)
    assert result.status == ResolutionStatus.MANUAL_REVIEW
    assert any(issue.field.startswith("lyric_video") for issue in result.issues)


def test_resolver_rejects_candidate_without_evidence_quote(tmp_path: Path) -> None:
    brief = "cita del brief. lyric video de Song"
    draft = make_draft(
        format=candidate("lyric_video"),
        lyric_video=LyricConfigDraft(
            track_name={"value": "Song", "confidence": "explicit"},
        ),
    )
    result = resolve_contract(draft, registry=AssetRegistry(tmp_path), brief_text=brief)
    assert result.status == ResolutionStatus.MANUAL_REVIEW
    assert any(issue.field.startswith("lyric_video") for issue in result.issues)


def test_resolver_rejects_candidate_with_root_citation_probe(tmp_path: Path) -> None:
    brief = "cita del brief. lyric video de Song"
    draft = make_draft(
        format=candidate("lyric_video"),
        lyric_video=LyricConfigDraft(
            track_name={"value": "Song", "confidence": "explicit", "citation": "Song"},
        ),
    )
    result = resolve_contract(draft, registry=AssetRegistry(tmp_path), brief_text=brief)
    assert result.status == ResolutionStatus.MANUAL_REVIEW
    assert any(issue.field.startswith("lyric_video") for issue in result.issues)


def test_resolver_rejects_candidate_with_root_quote_probe(tmp_path: Path) -> None:
    brief = "cita del brief. lyric video de Song"
    draft = make_draft(
        format=candidate("lyric_video"),
        lyric_video=LyricConfigDraft(
            track_name={"value": "Song", "confidence": "explicit", "quote": "Song"},
        ),
    )
    result = resolve_contract(draft, registry=AssetRegistry(tmp_path), brief_text=brief)
    assert result.status == ResolutionStatus.MANUAL_REVIEW
    assert any(issue.field.startswith("lyric_video") for issue in result.issues)


def test_resolver_rejects_candidate_quote_not_in_brief(tmp_path: Path) -> None:
    brief = "cita del brief. solo texto ordinario."
    draft = make_draft(
        format=candidate("lyric_video"),
        lyric_video=LyricConfigDraft(
            track_name={
                "value": "Inventada",
                "confidence": "explicit",
                "evidence": {"quote": "cancion inventada no en brief"},
            },
        ),
    )
    result = resolve_contract(draft, registry=AssetRegistry(tmp_path), brief_text=brief)
    assert result.status == ResolutionStatus.MANUAL_REVIEW
    assert any(issue.field.startswith("lyric_video") for issue in result.issues)


@pytest.mark.parametrize(
    "mention",
    [
        "crear un lyric video oficial",
        "necesitamos un video con letra para el lanzamiento",
        "usar la letra official del tema",
        "usar la letra oficial del tema",
        "adjuntamos el archivo .lrc sincronizado",
    ],
)
def test_resolver_fallback_activates_only_for_explicit_mentions(
    tmp_path: Path, mention: str
) -> None:
    brief = f"cita del brief. instrucciones: {mention}."
    draft = make_draft(format=None)
    result = resolve_contract(draft, registry=AssetRegistry(tmp_path), brief_text=brief)
    assert result.status == ResolutionStatus.RESOLVED
    assert result.contract is not None
    assert result.contract.format is Format.LYRIC_VIDEO


def test_resolver_fallback_does_not_activate_without_mention(tmp_path: Path) -> None:
    brief = "cita del brief. instrucciones generales para un video regular."
    draft = make_draft(format=None)
    result = resolve_contract(draft, registry=AssetRegistry(tmp_path), brief_text=brief)
    assert result.status == ResolutionStatus.NEW_ARCHETYPE
    assert any(
        issue.field == "format" and issue.code == IssueCode.MISSING_REQUIRED
        for issue in result.issues
    )


def test_resolver_auto_includes_lrc_asset_in_contract_required_assets(tmp_path: Path) -> None:
    lrc_file = tmp_path / "song.lrc"
    _ = lrc_file.write_text("[00:00.00]hello\n", encoding="utf-8")
    registry = AssetRegistry(tmp_path)
    ref = registry.register(
        asset_id="song",
        kind="lyrics",
        uri="song.lrc",
        origin="test",
    )

    brief = "cita del brief. usar letras song en lyric video."
    draft = make_draft(
        format=candidate("lyric_video"),
        lyric_video=LyricConfigDraft(
            lrc_asset_id=_cand_at("song", "song", brief.index("song")),
        ),
    )
    result = resolve_contract(draft, registry=registry, brief_text=brief)
    assert result.status == ResolutionStatus.RESOLVED
    assert result.contract is not None
    assert any(asset.asset_id == ref.asset_id for asset in result.contract.assets.required)


def test_resolver_rejects_tampered_lrc_asset(tmp_path: Path) -> None:
    lrc_file = tmp_path / "song.lrc"
    _ = lrc_file.write_text("[00:00.00]hello\n", encoding="utf-8")
    registry = AssetRegistry(tmp_path)
    _ = registry.register(
        asset_id="song",
        kind="lyrics",
        uri="song.lrc",
        origin="test",
    )
    # Tamper file
    _ = lrc_file.write_text("[00:00.00]tampered\n", encoding="utf-8")

    brief = "cita del brief. usar letras song en lyric video."
    draft = make_draft(
        format=candidate("lyric_video"),
        lyric_video=LyricConfigDraft(
            lrc_asset_id=_cand_at("song", "song", brief.index("song")),
        ),
    )
    result = resolve_contract(draft, registry=registry, brief_text=brief)
    assert result.status == ResolutionStatus.MANUAL_REVIEW
    assert any(issue.code == IssueCode.UNRESOLVED_ASSET for issue in result.issues)
