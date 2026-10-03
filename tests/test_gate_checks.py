"""Tests de los validadores deterministas del gate."""

import json
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path

import pytest

from kliptych.assets import AssetRegistry
from kliptych.contract import KNOWN_VALIDATOR_RULES, Contract, Platform
from kliptych.contract.enums import AudioRule
from kliptych.gate import (
    CheckResult,
    CheckStatus,
    Gate,
    GateContext,
    GateResult,
    GateStatus,
    MediaInfo,
    Piece,
)
from kliptych.gate.checks import DEFAULT_VALIDATORS, check_audio_present
from tests.support import (
    ALL_HARD_RULES,
    ALL_HARD_RULES_WITHOUT_AUDIO,
    FakeProbe,
    make_asset_ref,
    make_contract,
    make_media,
    make_piece,
)


def _hand_declared_audio_contract(audio_rule: str) -> Contract:
    """Construye un contrato que declara ``audio.present`` a mano.

    ``make_contract`` ya no hace esa combinacion cuando ``audio_rule`` es
    ``any``, porque el resolver no la produce nunca. Este helper existe para
    fijar que hace el check si alguien lo invoca igualmente, que es justo el
    caso que tiene que ser ``UNSUPPORTED`` en vez de un ``PASS`` sin comprobar.
    El motor ya no llega aqui por su cuenta: omite ``audio.present`` para una
    plataforma con ``any`` (ver ``rule_applies``).

    Args:
        audio_rule: Valor de ``audio_rule`` de la plataforma.

    Returns:
        Un contrato valido con ``audio.present`` en reglas duras.
    """
    base = make_contract(audio_rule=audio_rule)
    payload = base.model_dump(mode="json")
    payload["rules"]["hard"] = [*payload["rules"]["hard"], "audio.present"]
    return Contract.model_validate(payload)


def _artifact(tmp_path: Path, content: bytes = b"video") -> Path:
    path = tmp_path / "piece.mp4"
    _ = path.write_bytes(content)
    return path


def test_make_contract_rejects_hand_declared_audio_rule_with_any() -> None:
    """Un ``hard`` explicito incoherente falla en vez de repararse en silencio.

    Este es el contrato del fixture: si el llamador pide ``audio.present`` y a la
    vez ``audio_rule="any"``, esa combinacion no la produce el resolver, asi que
    el fixture dice que no en vez de quietly quitar la regla y devolver un
    contrato distinto del pedido. Un fixture que repara su entrada hace que
    cada test construido encima afirme algo que nadie pidio.
    """
    with pytest.raises(ValueError, match=r"audio_rule='any'.*audio\.present"):
        _ = make_contract(audio_rule="any", hard=[*ALL_HARD_RULES])


def test_make_contract_accepts_explicit_hard_without_audio_rule() -> None:
    """El inverso exacto: el mismo ``hard`` sin ``audio.present`` si se acepta.

    Comprueba que el raise discrimina la incoherencia y no simply la presencia
    de ``audio_rule="any"``, que es una combinacion legitima y frecuente.
    """
    contract = make_contract(audio_rule="any", hard=[*ALL_HARD_RULES_WITHOUT_AUDIO])

    assert "audio.present" not in contract.rules.hard


def test_make_contract_default_hard_with_any_is_not_an_error() -> None:
    """``hard=None`` con ``audio_rule="any"`` es legitimo: es el default.

    El conjunto por defecto si depende de ``audio_rule``, porque el resolver
    tambien depende de el. Eso es una definicion de default, no una reparacion:
    lo que nunca se toca es un ``hard`` explicito.
    """
    contract = make_contract(audio_rule="any")

    assert "audio.present" not in contract.rules.hard
    assert make_contract().rules.hard != contract.rules.hard


def test_hand_declared_audio_contract_is_the_documented_way_in() -> None:
    """La salida que el error senala existe y construye el contrato.

    El mensaje del ``ValueError`` apunta a este helper, asi que si el helper
    dejara de funcionar el mensaje estaria mandando a nadie a un sitio que no
    existe.
    """
    contract = _hand_declared_audio_contract("any")

    assert "audio.present" in contract.rules.hard
    assert contract.platforms[Platform.TIKTOK].audio_rule is AudioRule.ANY


def _result(
    tmp_path: Path,
    contract: Contract,
    piece: Piece,
    *,
    media: MediaInfo | None = None,
    registry: AssetRegistry | None = None,
) -> GateResult:
    gate = Gate(FakeProbe(info=make_media() if media is None else media))
    return gate.run(
        contract=contract,
        piece=piece,
        assets=AssetRegistry(tmp_path) if registry is None else registry,
    )


def _check(result: GateResult, rule_id: str) -> CheckResult:
    matches = [check for check in result.checks if check.id == rule_id]
    assert len(matches) == 1
    return matches[0]


def test_mention_from_must_mention_is_enforced(tmp_path: Path) -> None:
    contract = make_contract(required_mentions=(), must_mention=["@extra"])
    result = _result(tmp_path, contract, make_piece(_artifact(tmp_path)))
    check = _check(result, "caption.required_mention")
    assert check.status is CheckStatus.FAIL
    assert check.evidence["missing"] == ["@extra"]


def test_all_mentions_present_pass(tmp_path: Path) -> None:
    contract = make_contract(required_mentions=["@marca", "@extra"])
    piece = make_piece(_artifact(tmp_path), caption="gracias @marca y @extra #marca")
    assert _check(_result(tmp_path, contract, piece), "caption.required_mention").status is (
        CheckStatus.PASS
    )


def test_required_hashtag_can_live_in_hashtags_field(tmp_path: Path) -> None:
    contract = make_contract(required_hashtags=["#nueva"])
    piece = make_piece(_artifact(tmp_path), caption="texto sin etiquetas", hashtags=["#Nueva"])
    assert _check(_result(tmp_path, contract, piece), "caption.required_hashtag").status is (
        CheckStatus.PASS
    )


def test_missing_hashtag_fails_with_evidence(tmp_path: Path) -> None:
    contract = make_contract(required_hashtags=["#nueva"])
    piece = make_piece(_artifact(tmp_path), caption="solo @marca #marca")
    check = _check(_result(tmp_path, contract, piece), "caption.required_hashtag")
    assert check.status is CheckStatus.FAIL
    assert check.evidence["missing"] == ["#nueva"]


def test_forbidden_term_is_case_insensitive(tmp_path: Path) -> None:
    contract = make_contract(hard=["caption.forbidden"], forbidden=["sorteo"])
    piece = make_piece(_artifact(tmp_path), caption="gran SORTEO @marca #marca")
    check = _check(_result(tmp_path, contract, piece), "caption.forbidden")
    assert check.status is CheckStatus.FAIL
    assert check.evidence["found"] == ["sorteo"]


def test_forbidden_term_absent_passes(tmp_path: Path) -> None:
    contract = make_contract(hard=["caption.forbidden"], forbidden=["sorteo"])
    piece = make_piece(_artifact(tmp_path))
    assert _check(_result(tmp_path, contract, piece), "caption.forbidden").status is (
        CheckStatus.PASS
    )


def test_forbidden_term_in_hashtags_fails(tmp_path: Path) -> None:
    contract = make_contract(hard=["caption.forbidden"], prohibitions=["estafa"])
    piece = make_piece(
        _artifact(tmp_path),
        caption="mira @marca #marca",
        hashtags=("#marca", "#estafa"),
    )
    check = _check(_result(tmp_path, contract, piece), "caption.forbidden")
    assert check.status is CheckStatus.FAIL
    assert check.evidence["found"] == ["estafa"]


def test_first_line_must_open_the_caption(tmp_path: Path) -> None:
    contract = make_contract(hard=["caption.first_line"], first_line="Escribe al 555-1234")
    good = make_piece(_artifact(tmp_path), caption="Escribe al 555-1234 y gana @marca #marca")
    assert _check(_result(tmp_path, contract, good), "caption.first_line").status is (
        CheckStatus.PASS
    )
    bad = make_piece(_artifact(tmp_path), caption="Gana ya\nEscribe al 555-1234 @marca #marca")
    check = _check(_result(tmp_path, contract, bad), "caption.first_line")
    assert check.status is CheckStatus.FAIL
    assert check.evidence["first_line"] == "Gana ya"


def test_first_line_without_requirement_passes(tmp_path: Path) -> None:
    contract = make_contract(hard=["caption.first_line"], first_line=None)
    result = _result(tmp_path, contract, make_piece(_artifact(tmp_path)))
    assert _check(result, "caption.first_line").status is CheckStatus.PASS


def test_spelling_lock_checked_in_subtitles(tmp_path: Path) -> None:
    contract = make_contract(hard=["subtitles.spelling_lock"], spelling_locks=["MarcaX"])
    bad = make_piece(_artifact(tmp_path), subtitle_text="bienvenidos a maracax")
    check = _check(_result(tmp_path, contract, bad), "subtitles.spelling_lock")
    assert check.status is CheckStatus.FAIL
    assert check.evidence["missing"] == ["MarcaX"]

    good = make_piece(_artifact(tmp_path), subtitle_text="bienvenidos a MarcaX")
    assert _check(_result(tmp_path, contract, good), "subtitles.spelling_lock").status is (
        CheckStatus.PASS
    )


def test_spelling_lock_without_subtitles_is_unsupported(tmp_path: Path) -> None:
    contract = make_contract(hard=["subtitles.spelling_lock"], spelling_locks=["MarcaX"])
    piece = make_piece(_artifact(tmp_path), subtitle_text=None)
    check = _check(_result(tmp_path, contract, piece), "subtitles.spelling_lock")
    assert check.status is CheckStatus.UNSUPPORTED
    assert "subtítulos" in str(check.evidence["reason"])


def test_spelling_lock_without_declared_locks_passes(tmp_path: Path) -> None:
    contract = make_contract(hard=["subtitles.spelling_lock"], spelling_locks=[])
    piece = make_piece(_artifact(tmp_path), subtitle_text=None)
    check = _check(_result(tmp_path, contract, piece), "subtitles.spelling_lock")
    assert check.status is CheckStatus.PASS


def test_duration_bounds_are_enforced(tmp_path: Path) -> None:
    contract = make_contract(hard=["duration.min", "duration.max"], min_s=8, max_s=20)
    short = _result(
        tmp_path, contract, make_piece(_artifact(tmp_path)), media=make_media(duration_s=5)
    )
    assert _check(short, "duration.min").status is CheckStatus.FAIL
    assert _check(short, "duration.max").status is CheckStatus.PASS

    long = _result(
        tmp_path, contract, make_piece(_artifact(tmp_path)), media=make_media(duration_s=25)
    )
    assert _check(long, "duration.min").status is CheckStatus.PASS
    assert _check(long, "duration.max").status is CheckStatus.FAIL

    inside = _result(
        tmp_path, contract, make_piece(_artifact(tmp_path)), media=make_media(duration_s=12)
    )
    assert _check(inside, "duration.min").status is CheckStatus.PASS
    assert _check(inside, "duration.max").status is CheckStatus.PASS


def test_duration_without_bounds_passes(tmp_path: Path) -> None:
    contract = make_contract(hard=["duration.min", "duration.max"], min_s=None, max_s=None)
    result = _result(tmp_path, contract, make_piece(_artifact(tmp_path)))
    assert _check(result, "duration.min").status is CheckStatus.PASS
    assert _check(result, "duration.max").status is CheckStatus.PASS


def test_unknown_duration_is_unsupported(tmp_path: Path) -> None:
    contract = make_contract(hard=["duration.min", "duration.max"], min_s=8, max_s=20)
    media = make_media(duration_s=None)
    result = _result(tmp_path, contract, make_piece(_artifact(tmp_path)), media=media)
    assert _check(result, "duration.min").status is CheckStatus.UNSUPPORTED
    assert _check(result, "duration.max").status is CheckStatus.UNSUPPORTED
    assert result.status is GateStatus.UNSUPPORTED


def test_missing_audio_track_fails(tmp_path: Path) -> None:
    contract = make_contract(hard=["audio.present"])
    media = make_media(has_audio=False)
    result = _result(tmp_path, contract, make_piece(_artifact(tmp_path)), media=media)
    assert _check(result, "audio.present").status is CheckStatus.FAIL


def test_video_stream_required_for_video_format(tmp_path: Path) -> None:
    contract = make_contract(hard=["artifact.video_stream"])
    media = make_media(has_video=False)
    result = _result(tmp_path, contract, make_piece(_artifact(tmp_path)), media=media)
    assert _check(result, "artifact.video_stream").status is CheckStatus.FAIL


def test_video_stream_present_passes(tmp_path: Path) -> None:
    contract = make_contract(hard=["artifact.video_stream"])
    result = _result(tmp_path, contract, make_piece(_artifact(tmp_path)))
    check = _check(result, "artifact.video_stream")
    assert check.status is CheckStatus.PASS
    assert check.evidence == {"width": 1080, "height": 1920}


def test_required_asset_must_be_registered(tmp_path: Path) -> None:
    contract = make_contract(hard=["assets.required"], required_assets=[make_asset_ref("clip-01")])
    result = _result(tmp_path, contract, make_piece(_artifact(tmp_path)))
    check = _check(result, "assets.required")
    assert check.status is CheckStatus.FAIL
    assert check.evidence["missing"] == ["clip-01"]


def test_registered_asset_passes_and_tampering_fails(tmp_path: Path) -> None:
    clip = tmp_path / "clip.mp4"
    _ = clip.write_bytes(b"video")
    registry = AssetRegistry(tmp_path)
    ref = registry.register(asset_id="clip-01", kind="video", uri="clip.mp4", origin="brief")
    contract = make_contract(hard=["assets.required"], required_assets=[ref])
    piece = make_piece(_artifact(tmp_path))
    assert _check(
        _result(tmp_path, contract, piece, registry=registry), "assets.required"
    ).status is (CheckStatus.PASS)

    _ = clip.write_bytes(b"manipulado")
    check = _check(_result(tmp_path, contract, piece, registry=registry), "assets.required")
    assert check.status is CheckStatus.FAIL
    assert check.evidence["tampered"] == ["clip-01"]


def test_contract_asset_hash_mismatch_fails(tmp_path: Path) -> None:
    clip = tmp_path / "clip.mp4"
    _ = clip.write_bytes(b"video")
    registry = AssetRegistry(tmp_path)
    _ = registry.register(asset_id="clip-01", kind="video", uri="clip.mp4", origin="brief")
    wrong = make_asset_ref("clip-01", sha256="a" * 64, size_bytes=999)
    contract = make_contract(hard=["assets.required"], required_assets=[wrong])
    result = _result(tmp_path, contract, make_piece(_artifact(tmp_path)), registry=registry)
    check = _check(result, "assets.required")
    assert check.status is CheckStatus.FAIL
    assert check.evidence["mismatched"] == ["clip-01"]


def test_unsafe_registry_uri_fails_without_exception(tmp_path: Path) -> None:
    registry_file = tmp_path / "registry.json"
    payload = {
        "schema_version": "1.1",
        "assets": [
            {
                "asset_id": "malo",
                "kind": "file",
                "uri": "../fuera.txt",
                "sha256": "a" * 64,
                "size_bytes": 1,
                "mime": "text/plain",
                "origin": "brief",
                "license": None,
                "resolved_at": datetime(2026, 9, 22, tzinfo=UTC).isoformat(),
            }
        ],
    }
    _ = registry_file.write_text(json.dumps(payload), encoding="utf-8")
    registry = AssetRegistry.load(registry_file, tmp_path / "workspace")
    contract = make_contract(hard=["assets.required"], required_assets=[make_asset_ref("malo")])
    result = _result(tmp_path, contract, make_piece(_artifact(tmp_path)), registry=registry)
    check = _check(result, "assets.required")
    assert check.status is CheckStatus.FAIL
    assert check.evidence["unsafe"]
    assert result.status is GateStatus.REJECTED


def test_campaign_prohibitions_are_enforced(tmp_path: Path) -> None:
    contract = make_contract(hard=["caption.forbidden"], prohibitions=["sorteo"])
    piece = make_piece(_artifact(tmp_path), caption="gran SORTEO @marca #marca")
    check = _check(_result(tmp_path, contract, piece), "caption.forbidden")
    assert check.status is CheckStatus.FAIL
    assert check.evidence["found"] == ["sorteo"]


def test_mention_prefix_is_not_a_match(tmp_path: Path) -> None:
    contract = make_contract()
    piece = make_piece(_artifact(tmp_path), caption="gracias @marcado #marca")
    check = _check(_result(tmp_path, contract, piece), "caption.required_mention")
    assert check.status is CheckStatus.FAIL
    assert check.evidence["missing"] == ["@marca"]


def test_mention_inside_email_is_not_a_match(tmp_path: Path) -> None:
    contract = make_contract()
    piece = make_piece(_artifact(tmp_path), caption="correo@marca.com y #marca")
    check = _check(_result(tmp_path, contract, piece), "caption.required_mention")
    assert check.status is CheckStatus.FAIL
    assert check.evidence["missing"] == ["@marca"]


def test_mention_match_is_case_insensitive(tmp_path: Path) -> None:
    contract = make_contract(required_mentions=["@Marca"])
    piece = make_piece(_artifact(tmp_path), caption="hola @marca #marca")
    assert _check(_result(tmp_path, contract, piece), "caption.required_mention").status is (
        CheckStatus.PASS
    )


def test_hashtag_prefix_is_not_a_match(tmp_path: Path) -> None:
    contract = make_contract(required_hashtags=["#marca"])
    piece = make_piece(_artifact(tmp_path), caption="vamos #marcado @marca")
    check = _check(_result(tmp_path, contract, piece), "caption.required_hashtag")
    assert check.status is CheckStatus.FAIL
    assert check.evidence["missing"] == ["#marca"]


def test_no_required_assets_passes(tmp_path: Path) -> None:
    contract = make_contract(hard=["assets.required"])
    result = _result(tmp_path, contract, make_piece(_artifact(tmp_path)))
    assert _check(result, "assets.required").status is CheckStatus.PASS


def test_partial_watermark_without_png_is_fail_closed(tmp_path: Path) -> None:
    contract = make_contract(watermark_required=True, watermark_visible_full_video=False)
    result = _result(tmp_path, contract, make_piece(_artifact(tmp_path)))
    assert _check(result, "watermark.present").status is CheckStatus.FAIL
    assert result.status is GateStatus.REJECTED


def test_full_video_watermark_without_png_is_fail_closed(tmp_path: Path) -> None:
    contract = make_contract(watermark_required=True, watermark_visible_full_video=True)
    result = _result(tmp_path, contract, make_piece(_artifact(tmp_path)))
    assert _check(result, "watermark.full_video").status is CheckStatus.FAIL
    assert result.status is GateStatus.REJECTED


def test_known_validator_rules_matches_default_validators() -> None:
    assert frozenset(DEFAULT_VALIDATORS.keys()) == KNOWN_VALIDATOR_RULES


# --- D2: reglas de audio sin validador mecanico no pueden pasar ---------------


def test_audio_rule_no_trending_is_not_a_pass(tmp_path: Path) -> None:
    """``no_trending`` no se verifica: el gate no puede emitir PASS.

    Este check solo sabe mirar si existe un stream de audio. Eso no dice que
    el audio no sea trending, que es lo que la regla exige. Brief §5.1: sin
    validador aplicable, nunca PASS.
    """
    contract = make_contract(hard=["audio.present"], audio_rule="no_trending")
    media = make_media(has_audio=True)
    result = _result(tmp_path, contract, make_piece(_artifact(tmp_path)), media=media)
    check = _check(result, "audio.present")
    assert check.status is not CheckStatus.PASS, check


def test_audio_rule_official_required_is_not_a_pass(tmp_path: Path) -> None:
    """``official_required`` tampoco: un stream existente no es el oficial.

    Brief §5.2 dice literal para el audio oficial: si el fingerprint contra el
    asset oficial no está implementado, el resultado es MANUAL_REVIEW.
    """
    contract = make_contract(
        hard=["audio.present"],
        audio_rule="official_required",
        official_audio_url="https://www.tiktok.com/music/official",
    )
    media = make_media(has_audio=True)
    result = _result(tmp_path, contract, make_piece(_artifact(tmp_path)), media=media)
    check = _check(result, "audio.present")
    assert check.status is not CheckStatus.PASS, check


def test_audio_rule_any_is_unsupported_when_checked_directly(tmp_path: Path) -> None:
    """``any`` no tiene nada que verificar: el resultado es UNSUPPORTED.

    Se invoca el validador directamente, saltandose el motor a proposito. Por la
    via normal no se llega: el motor omite ``audio.present`` cuando la
    plataforma declara ``any`` (ver ``rule_applies``). Este test fija que debe
    hacer el check si alguien lo invoca igualmente, para que no degrade a un
    PASS con evidencia que parece una atestaci�n de audio verificado.
    """
    contract = _hand_declared_audio_contract("any")
    piece = make_piece(_artifact(tmp_path))
    media = make_media(has_audio=True)
    context = GateContext(
        contract=contract,
        rules=contract.platforms[piece.platform],
        piece=piece,
        artifact_sha256=sha256(b"video").hexdigest(),
        media=media,
        assets=AssetRegistry(tmp_path),
    )

    outcome = check_audio_present(context)

    assert outcome.status is CheckStatus.UNSUPPORTED
    assert outcome.evidence["audio_rule"] == "any"
    assert outcome.evidence["verifiable"] is False
    assert "has_audio" not in outcome.evidence


def test_audio_rule_any_is_not_a_fail_when_there_is_no_audio(tmp_path: Path) -> None:
    """``any`` con un artefacto sin pista tampoco puede ser FAIL.

    Rama hermana de la anterior, y la que mas importa: el motor omite
    ``audio.present`` para una plataforma con ``any``, asi que esta rama no se
    alcanza por la via normal. Es la ultima linea de defensa si el predicado
    ``rule_applies`` se rompe alguna vez, y tiene que impedir las dos
    respuestas incorrectas. Un FAIL rechazaria la pieza por exigir audio que
    nadie pidio; un PASS con ``{"has_audio": false}`` atestiguaria una
    comprobacion que no ocurrio.
    """
    contract = _hand_declared_audio_contract("any")
    piece = make_piece(_artifact(tmp_path))
    context = GateContext(
        contract=contract,
        rules=contract.platforms[piece.platform],
        piece=piece,
        artifact_sha256=sha256(b"video").hexdigest(),
        media=make_media(has_audio=False),
        assets=AssetRegistry(tmp_path),
    )

    outcome = check_audio_present(context)

    assert outcome.status is not CheckStatus.PASS
    assert outcome.status is not CheckStatus.FAIL
    assert outcome.status is CheckStatus.UNSUPPORTED
    assert "has_audio" not in outcome.evidence


def test_unverifiable_audio_rule_says_so_in_its_evidence(tmp_path: Path) -> None:
    """La evidencia dice LITERALMENTE que no se verificó y qué faltaría.

    Un ``{"has_audio": true}`` en un PASS de una regla que nadie comprobó es
    exactamente la mentira que este cambio quita.
    """
    contract = make_contract(
        hard=["audio.present"],
        audio_rule="official_required",
        official_audio_url="https://www.tiktok.com/music/official",
    )
    media = make_media(has_audio=True)
    result = _result(tmp_path, contract, make_piece(_artifact(tmp_path)), media=media)
    evidence = _check(result, "audio.present").evidence
    assert "has_audio" not in evidence, evidence
    assert evidence["verifiable"] is False
    assert evidence["audio_rule"] == "official_required"
    assert evidence["missing_validator"], evidence


def test_verifiable_audio_rule_never_claims_to_be_unverified(tmp_path: Path) -> None:
    """``own_clip`` sí se verifica: su evidencia debe decirlo."""
    contract = make_contract(hard=["audio.present"], audio_rule="own_clip")
    media = make_media(has_audio=True)
    result = _result(tmp_path, contract, make_piece(_artifact(tmp_path)), media=media)
    check = _check(result, "audio.present")
    assert check.status is CheckStatus.PASS
    assert check.evidence == {"has_audio": True}
