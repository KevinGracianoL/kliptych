"""Tests del motor del gate: estados fail-closed, hashes y orden de checks."""

from hashlib import sha256
from pathlib import Path

import pytest

from kliptych.assets import AssetRegistry
from kliptych.contract import Platform, contract_digest
from kliptych.contract.enums import AudioRule
from kliptych.contract.schema import PlatformRules
from kliptych.gate import (
    DEFAULT_VALIDATORS,
    CheckOutcome,
    CheckResult,
    CheckStatus,
    Gate,
    GateContext,
    GateError,
    GateResult,
    GateStatus,
    ProbeError,
)
from tests.support import (
    ALL_HARD_RULES,
    FakeProbe,
    make_contract,
    make_media,
    make_piece,
    with_platform,
)


def _artifact(tmp_path: Path) -> Path:
    path = tmp_path / "piece.mp4"
    _ = path.write_bytes(b"video")
    return path


def _gate(probe: FakeProbe | None = None) -> Gate:
    return Gate(FakeProbe(info=make_media()) if probe is None else probe)


def _check(result: GateResult, rule_id: str) -> CheckResult:
    matches = [check for check in result.checks if check.id == rule_id]
    assert len(matches) == 1
    return matches[0]


def _always_manual_review(_context: GateContext) -> CheckOutcome:
    return CheckOutcome(status=CheckStatus.MANUAL_REVIEW, evidence={"reason": "revisión visual"})


def test_all_checks_pass_yields_passed(tmp_path: Path) -> None:
    artifact = _artifact(tmp_path)
    contract = make_contract(audio_rule="any")
    result = _gate().run(
        contract=contract,
        piece=make_piece(artifact),
        assets=AssetRegistry(tmp_path),
    )
    assert result.status is GateStatus.PASSED
    assert result.passed is True
    # Se contrasta contra las reglas que el contrato declara de verdad, no
    # contra ALL_HARD_RULES: con audio_rule="any" el resolver no declara
    # audio.present, asi que el fixture tampoco, y afirmar el conjunto completo
    # seria exigir una regla que un contrato real no lleva.
    assert [check.id for check in result.checks] == list(contract.rules.hard)
    assert all(check.status is CheckStatus.PASS for check in result.checks)
    assert result.artifact_sha256 == sha256(b"video").hexdigest()
    expected_contract_hash = contract_digest(make_contract(audio_rule="any"))
    assert result.contract_sha256 == expected_contract_hash


def test_hard_rule_without_validator_never_passes(tmp_path: Path) -> None:
    artifact = _artifact(tmp_path)
    contract = make_contract(hard=[*ALL_HARD_RULES, "attribution.required"])
    result = _gate().run(
        contract=contract,
        piece=make_piece(artifact),
        assets=AssetRegistry(tmp_path),
    )
    assert result.status is GateStatus.UNSUPPORTED
    assert result.passed is False
    check = _check(result, "attribution.required")
    assert check.status is CheckStatus.UNSUPPORTED
    assert "reason" in check.evidence


def test_hard_failure_rejects_and_wins_over_unsupported(tmp_path: Path) -> None:
    artifact = _artifact(tmp_path)
    contract = make_contract(hard=[*ALL_HARD_RULES, "watermark.full_video"])
    result = _gate().run(
        contract=contract,
        piece=make_piece(artifact, caption="sin mención #marca"),
        assets=AssetRegistry(tmp_path),
    )
    assert result.status is GateStatus.REJECTED
    assert _check(result, "caption.required_mention").status is CheckStatus.FAIL


def test_manual_review_class_is_non_blocking(tmp_path: Path) -> None:
    artifact = _artifact(tmp_path)
    contract = make_contract(manual_review=["audio.official_selection"])
    result = _gate().run(
        contract=contract,
        piece=make_piece(artifact),
        assets=AssetRegistry(tmp_path),
    )
    assert result.status is GateStatus.PENDING_REVIEW
    assert _check(result, "audio.official_selection").status is CheckStatus.MANUAL_REVIEW
    assert result.passed is False


def test_recommended_failure_rejects_fail_closed(tmp_path: Path) -> None:
    artifact = _artifact(tmp_path)
    contract = make_contract(recommended=["duration.max"], min_s=5, max_s=6)
    result = _gate().run(
        contract=contract,
        piece=make_piece(artifact),
        assets=AssetRegistry(tmp_path),
    )
    assert result.status is GateStatus.REJECTED
    assert result.passed is False
    assert _check(result, "duration.max").status is CheckStatus.FAIL


def test_hard_manual_review_status_blocks_as_manual_review(tmp_path: Path) -> None:
    artifact = _artifact(tmp_path)
    contract = make_contract(hard=[*ALL_HARD_RULES, "watermark.full_video"])
    validators = {**DEFAULT_VALIDATORS, "watermark.full_video": _always_manual_review}
    gate = Gate(FakeProbe(info=make_media()), validators=validators)
    result = gate.run(
        contract=contract,
        piece=make_piece(artifact),
        assets=AssetRegistry(tmp_path),
    )
    assert result.status is GateStatus.PENDING_REVIEW
    assert _check(result, "watermark.full_video").status is CheckStatus.MANUAL_REVIEW
    assert result.passed is False


def test_probe_failure_blocks_media_checks(tmp_path: Path) -> None:
    artifact = _artifact(tmp_path)
    probe = FakeProbe(error=ProbeError("ffprobe no disponible"))
    contract = make_contract(hard=[*ALL_HARD_RULES, "artifact.video_stream"])
    result = _gate(probe).run(
        contract=contract,
        piece=make_piece(artifact),
        assets=AssetRegistry(tmp_path),
    )
    assert result.status is GateStatus.UNSUPPORTED
    assert _check(result, "duration.min").status is CheckStatus.UNSUPPORTED
    assert _check(result, "audio.present").status is CheckStatus.UNSUPPORTED
    assert _check(result, "artifact.video_stream").status is CheckStatus.UNSUPPORTED
    assert _check(result, "artifact.integrity").status is CheckStatus.PASS
    assert _check(result, "caption.required_mention").status is CheckStatus.PASS


def _always_pass(_context: GateContext) -> CheckOutcome:
    return CheckOutcome(status=CheckStatus.PASS, evidence={"reason": "cumple"})


def test_manual_review_strength_needs_human_even_when_the_check_passes(tmp_path: Path) -> None:
    """Una regla `manual_review` que el validador aprueba igual exige a una persona.

    Este es el unico caso en el que la FUERZA es lo unico que distingue el
    resultado: el check sale PASS, asi que sin mirar la fuerza la pieza se daria
    por buena sin que nadie la haya mirado. Un operador tiene que poder abrirla y
    decidir, aunque la maquina diga que cumple.

    Sin este test, la rama `strength is RuleStrength.MANUAL_REVIEW` de
    `_derive_status` se puede borrar entera y la suite sigue verde: es lo que
    pasaria con una mutacion que la dejara solo mirando `check.status`.
    """
    contract = make_contract(
        hard=["artifact.integrity"],
        manual_review=["review.me"],
        audio_rule="any",
        min_s=None,
        max_s=None,
    )
    validators = {**DEFAULT_VALIDATORS, "review.me": _always_pass}
    result = Gate(FakeProbe(info=make_media()), validators=validators).run(
        contract=contract,
        piece=make_piece(_artifact(tmp_path)),
        assets=AssetRegistry(tmp_path),
    )

    assert _check(result, "review.me").status is CheckStatus.PASS
    assert result.status is GateStatus.PENDING_REVIEW
    assert result.passed is False


def test_missing_artifact_fails_integrity(tmp_path: Path) -> None:
    result = _gate().run(
        contract=make_contract(),
        piece=make_piece(tmp_path / "no-existe.mp4"),
        assets=AssetRegistry(tmp_path),
    )
    assert result.status is GateStatus.REJECTED
    assert _check(result, "artifact.integrity").status is CheckStatus.FAIL
    assert result.artifact_sha256 is None


def _always_unsupported(_context: GateContext) -> CheckOutcome:
    return CheckOutcome(status=CheckStatus.UNSUPPORTED, evidence={"reason": "no verificable"})


@pytest.mark.parametrize("strength", ["hard", "recommended", "manual_review"])
def test_unsupported_blocks_at_every_strength(tmp_path: Path, strength: str) -> None:
    """UNSUPPORTED bloquea con CUALQUIER fuerza, no solo con `hard`.

    Este es el fail-open que se cerro. `_derive_status` miraba solo las reglas
    `hard`, asi que un UNSUPPORTED en `recommended` o `manual_review` caia al
    final y devolvia PASSED: un artefacto que nadie pudo verificar salia
    aprobado. El aviso estaba escrito en un comentario del propio motor, dos
    funciones mas arriba, y ningun test lo alcanzaba.

    El brief, §5.1, acota su clausula "jamás PASS" a las reglas DURAS, asi que
    esto no es cumplir el brief: es un endurecimiento deliberado mas alla de el.
    El motivo esta en `_is_piece_exportable`: `PENDING_REVIEW` es una salida, y
    una firma es una atestacion de una persona sobre una geometria que no se
    midio. No hay nada que un operador pueda revisar a mano.
    """
    if strength == "hard":
        contract = make_contract(
            hard=["artifact.integrity", "probe.unsupported"],
            audio_rule="any",
            min_s=None,
            max_s=None,
        )
    elif strength == "recommended":
        contract = make_contract(
            hard=["artifact.integrity"],
            recommended=["probe.unsupported"],
            audio_rule="any",
            min_s=None,
            max_s=None,
        )
    else:
        contract = make_contract(
            hard=["artifact.integrity"],
            manual_review=["probe.unsupported"],
            audio_rule="any",
            min_s=None,
            max_s=None,
        )
    validators = {**DEFAULT_VALIDATORS, "probe.unsupported": _always_unsupported}
    result = Gate(FakeProbe(info=make_media()), validators=validators).run(
        contract=contract,
        piece=make_piece(_artifact(tmp_path)),
        assets=AssetRegistry(tmp_path),
    )

    assert _check(result, "probe.unsupported").status is CheckStatus.UNSUPPORTED
    assert result.status is GateStatus.UNSUPPORTED, strength
    assert result.passed is False, strength


def test_unsupported_outranks_manual_review_strength(tmp_path: Path) -> None:
    """UNSUPPORTED se escala ANTES que la rama de revision manual.

    El orden importa: si una regla es `manual_review` y su check sale
    UNSUPPORTED, el resultado correcto es UNSUPPORTED. Degradarlo a
    PENDING_REVIEW lo haria exportable con `approve_manual_review`, y no hay
    veredicto que un operador pueda revisar a mano cuando no se pudo medir.
    """
    contract = make_contract(
        hard=["artifact.integrity"],
        manual_review=["probe.unsupported"],
        audio_rule="any",
        min_s=None,
        max_s=None,
    )
    validators = {**DEFAULT_VALIDATORS, "probe.unsupported": _always_unsupported}
    result = Gate(FakeProbe(info=make_media()), validators=validators).run(
        contract=contract,
        piece=make_piece(_artifact(tmp_path)),
        assets=AssetRegistry(tmp_path),
    )

    assert result.status is GateStatus.UNSUPPORTED


def test_fail_still_outranks_unsupported(tmp_path: Path) -> None:
    """Un FAIL medido sigue gainandole a un UNSUPPORTED de otra regla.

    El orden de escalado no se ha reordenado: FAIL -> REJECTED es lo primero, y
    un artefacto que se pudo medir y esta mal se rechaza aunque otra regla no se
    haya podido verificar.
    """
    contract = make_contract(
        hard=["artifact.integrity", "probe.unsupported"],
        recommended=["duration.max"],
        audio_rule="any",
        min_s=None,
        max_s=6,
    )
    validators = {**DEFAULT_VALIDATORS, "probe.unsupported": _always_unsupported}
    result = Gate(FakeProbe(info=make_media()), validators=validators).run(
        contract=contract,
        piece=make_piece(_artifact(tmp_path)),
        assets=AssetRegistry(tmp_path),
    )

    assert _check(result, "duration.max").status is CheckStatus.FAIL
    assert _check(result, "probe.unsupported").status is CheckStatus.UNSUPPORTED
    assert result.status is GateStatus.REJECTED


def test_piece_platform_must_exist_in_contract(tmp_path: Path) -> None:
    artifact = _artifact(tmp_path)
    with pytest.raises(GateError, match="plataforma"):
        _ = _gate().run(
            contract=make_contract(),
            piece=make_piece(artifact, platform=Platform.X),
            assets=AssetRegistry(tmp_path),
        )


def test_gate_result_round_trips_through_json(tmp_path: Path) -> None:
    artifact = _artifact(tmp_path)
    result = _gate().run(
        contract=make_contract(),
        piece=make_piece(artifact),
        assets=AssetRegistry(tmp_path),
    )
    restored = GateResult.model_validate_json(result.model_dump_json())
    assert restored == result


def test_audio_present_runs_for_every_platform_that_requires_audio(tmp_path: Path) -> None:
    """Con dos plataformas que exigen audio, el check se emite para las dos.

    ``own_clip`` se verifica de verdad con ``has_audio``; ``no_trending`` no se
    puede verificar y para en MANUAL_REVIEW. El scoping solo omite cuando la
    plataforma no declara requisitos.
    """
    artifact = _artifact(tmp_path)
    contract = with_platform(
        make_contract(audio_rule="no_trending", manual_review=("audio.own_clip",)),
        Platform.INSTAGRAM_REELS,
        PlatformRules(audio_rule=AudioRule.OWN_CLIP),
    )
    gate = Gate(FakeProbe(info=make_media(has_audio=True)))

    tiktok = gate.run(
        contract=contract,
        piece=make_piece(artifact),
        assets=AssetRegistry(tmp_path),
    )
    reels = gate.run(
        contract=contract,
        piece=make_piece(artifact, platform=Platform.INSTAGRAM_REELS),
        assets=AssetRegistry(tmp_path),
    )

    assert _check(tiktok, "audio.present").status is CheckStatus.MANUAL_REVIEW
    assert _check(reels, "audio.present").status is CheckStatus.PASS
    assert _check(reels, "audio.present").evidence == {"has_audio": True}


def test_empty_requirement_set_keeps_emitting_its_evidence(tmp_path: Path) -> None:
    """Un conjunto de requisitos VACIO no es una regla no aplicable.

    ``caption.required_mention`` se declara global porque tiktok exige @marca,
    pero instagram no exige ninguna. El check debe seguir ejecutandose en
    instagram y dejar constancia de que se evaluo. Si el motor empieza a omitir
    reglas con requisitos vacios, esta evidencia desaparece y con ella la
    prueba de que la regla se considero.
    """
    artifact = _artifact(tmp_path)
    contract = with_platform(
        make_contract(hard=["caption.required_mention"]),
        Platform.INSTAGRAM_REELS,
        PlatformRules(),
    )
    result = _gate().run(
        contract=contract,
        piece=make_piece(artifact, caption="sin menciones", platform=Platform.INSTAGRAM_REELS),
        assets=AssetRegistry(tmp_path),
    )

    check = _check(result, "caption.required_mention")
    assert check.status is CheckStatus.PASS
    assert check.evidence == {"required": [], "missing": []}
