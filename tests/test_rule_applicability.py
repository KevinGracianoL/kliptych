"""Aplicabilidad por plataforma de las reglas declaradas globalmente.

``contract.rules`` es global: una sola declaracion cubre todas las plataformas.
Lo que describen es por plataforma. Estos tests fijan que una regla solo afecta
a las plataformas que la declararon, en el motor, que es donde se decide eso, y
en los recordatorios de la entrega, que ya no son planos (#57).

El riesgo que estos tests cubren no es que el motor pase de mas, sino que pase
de menos: una plataforma que SI declara una regla tiene que seguir exigiendo su
firma. Por eso hay tests de no-regresion explicitos.
"""

from collections.abc import Sequence
from pathlib import Path

import pytest

from kliptych.assets import AssetRegistry
from kliptych.contract import (
    Attribution,
    AttributionType,
    Contract,
    LinkRules,
    Platform,
    contract_digest,
)
from kliptych.contract.enums import AudioRule
from kliptych.contract.schema import PlatformRules, canonical_rule_id
from kliptych.exporter import DeliveryReport, ExportStatus, export_delivery
from kliptych.gate import (
    CheckOutcome,
    CheckResult,
    CheckStatus,
    Gate,
    GateResult,
    GateStatus,
    Piece,
)
from kliptych.gate.engine import rule_applies, rule_applies_anywhere
from tests.support import (
    FakeProbe,
    make_contract,
    make_media,
    make_piece,
    with_platform,
)

_AUDIO_HARD = ["artifact.integrity", "duration.min", "audio.present"]


def _mixed_contract() -> Contract:
    """Tiktok exige audio (no_trending), instagram_reels no exige nada (any).

    Returns:
        El contrato de dos plataformas con la regla de audio contradictoria.
    """
    return with_platform(
        make_contract(audio_rule="no_trending", hard=_AUDIO_HARD),
        Platform.INSTAGRAM_REELS,
        PlatformRules(),
    )


def _piece(artifact: Path, *, piece_id: str, platform: Platform) -> Piece:
    """Pieza con identificador propio, que ``make_piece`` no expone.

    Args:
        artifact: Ruta del artefacto ya escrito.
        piece_id: Identificador unico de la pieza.
        platform: Plataforma de destino.

    Returns:
        La pieza lista para el gate.
    """
    return make_piece(artifact, platform=platform).model_copy(update={"piece_id": piece_id})


def _artifact(tmp_path: Path) -> Path:
    path = tmp_path / "piece.mp4"
    _ = path.write_bytes(b"video-bytes-real")
    return path


def _gate() -> Gate:
    """Gate con un probe fijo y audio presente.

    Returns:
        El gate usado por todos los tests del modulo.
    """
    return Gate(FakeProbe(info=make_media(has_audio=True, has_video=True)))


def _export(
    tmp_path: Path,
    *,
    contract: Contract,
    pieces: list[Piece],
    approve: bool,
    tag: str,
) -> DeliveryReport:
    """Exporta piezas con o sin firma de aprobacion manual.

    Args:
        tmp_path: Directorio del test.
        contract: Contrato a evaluar.
        pieces: Piezas a publicar.
        approve: Si se concede la firma de revision manual.
        tag: Nombre del paquete de destino.

    Returns:
        El informe de entrega.
    """
    return export_delivery(
        contract=contract,
        pieces=pieces,
        gate=_gate(),
        assets=AssetRegistry(tmp_path),
        destination=tmp_path / tag,
        approve_manual_review=approve,
        approved_by="auditor-fixture" if approve else None,
    )


def _check(result: GateResult, rule_id: str) -> CheckResult:
    """Devuelve el unico check con ese id, fallando si no hay exactamente uno.

    Args:
        result: Resultado del gate.
        rule_id: Identificador del check buscado.

    Returns:
        El check encontrado.
    """
    matches = [check for check in result.checks if check.id == rule_id]
    assert len(matches) == 1, [check.id for check in result.checks]
    return matches[0]


def test_any_platform_exports_unattended_while_declaring_platform_still_blocks(
    tmp_path: Path,
) -> None:
    """T1: la plataforma que no declara la regla no paga su firma.

    Es el defecto de #59. ``audio.no_trending`` se declara globalmente porque
    tiktok lo pide, pero instagram_reels declara ``audio_rule=any`` y no lo
    pidió: exigirle una firma es pedir una firma por un requisito que nadie
    declaró.
    """
    reels = _piece(_artifact(tmp_path), piece_id="ig", platform=Platform.INSTAGRAM_REELS)

    result = _export(
        tmp_path, contract=_mixed_contract(), pieces=[reels], approve=False, tag="t1-ig"
    )

    assert result.status is ExportStatus.EXPORTED
    assert [piece.piece_id for piece in result.exported] == ["ig"]
    assert result.rejected == ()


def test_declaring_platform_is_untouched_and_still_requires_signature(
    tmp_path: Path,
) -> None:
    """T1, segunda mitad: tiktok sigue exigiendo firma. ESTE ES EL NO-REGRESION.

    Si este test falla, este PR ha abierto un agujero: una plataforma que declara
    la regla estaria exportando sin revision de nadie.
    """
    tiktok = _piece(_artifact(tmp_path), piece_id="tk", platform=Platform.TIKTOK)

    result = _export(
        tmp_path, contract=_mixed_contract(), pieces=[tiktok], approve=False, tag="t1-tk"
    )

    assert result.status is ExportStatus.BLOCKED
    assert [piece.piece_id for piece in result.rejected] == ["tk"]
    assert result.rejected[0].gate_status is GateStatus.PENDING_REVIEW


def test_declaring_platform_exports_with_signature_recorded(tmp_path: Path) -> None:
    """T2: con firma, tiktok sale, y la firma queda registrada en la pieza.

    La atribucion importa: la pieza que no necesitaba firma no debe aparecer
    como firmada, o el informe acreditaria una revision que nadie hizo.
    """
    tiktok = _piece(_artifact(tmp_path), piece_id="tk", platform=Platform.TIKTOK)
    reels = _piece(_artifact(tmp_path), piece_id="ig", platform=Platform.INSTAGRAM_REELS)

    result = _export(
        tmp_path, contract=_mixed_contract(), pieces=[tiktok, reels], approve=True, tag="t2"
    )

    assert result.status is ExportStatus.EXPORTED
    exported = {piece.piece_id: piece for piece in result.exported}
    assert exported["tk"].approved_by == "auditor-fixture"
    assert "audio.no_trending" in exported["tk"].manually_approved_rules
    assert exported["ig"].approved_by is None
    assert exported["ig"].manually_approved_rules == ()
    gate_report = GateResult.model_validate_json(
        (tmp_path / "t2" / exported["tk"].gate_path).read_text("utf-8")
    )
    assert gate_report.contract_sha256 == contract_digest(_mixed_contract())


def test_single_platform_requiring_audio_behaves_exactly_as_before(
    tmp_path: Path,
) -> None:
    """T3: NO-REGRESION. Una sola plataforma con no_trending, sin mas.

    El caso mas simple, y el que mas fácil se rompe al tocar la derivacion de
    fuerzas. Sin firma: bloqueada. Con firma: exportada.
    """
    contract = make_contract(audio_rule="no_trending", hard=_AUDIO_HARD)
    piece = make_piece(_artifact(tmp_path))

    blocked = _export(tmp_path, contract=contract, pieces=[piece], approve=False, tag="t3-no")
    assert blocked.status is ExportStatus.BLOCKED
    assert blocked.rejected[0].gate_status is GateStatus.PENDING_REVIEW

    approved = _export(tmp_path, contract=contract, pieces=[piece], approve=True, tag="t3-yes")
    assert approved.status is ExportStatus.EXPORTED


def test_reminder_is_still_emitted_for_a_rule_no_platform_claims(tmp_path: Path) -> None:
    """CASO C: una regla declarada que NADIE reclama sigue emitiendo su aviso.

    Acotar los recordatorios a las plataformas de la entrega (#57) solo puede
    callar el CASO B, donde la regla es de otra plataforma. El CASO C es el
    contrario: la regla esta declarada en el contrato y ninguna plataforma la
    pide, o sea que el contrato es incoherente. Ahi el aviso es la UNICA senal
    de que hay que mirarlo, y el filtro no puede suprimirlo.

    Por eso `_recuerda_regla` no es "aplica a alguna plataforma de la entrega" a
    secas: cuando no aplica en ninguna, distingue si el contrato entero la pide
    en otra (callar) de si no la pide nadie (avisar).

    Este test fija esa mitad: si la regla esta declarada, el aviso sale.
    """
    contract = make_contract(
        audio_rule="any",
        hard=["artifact.integrity", "duration.min"],
        manual_review=["audio.no_trending"],
    )
    piece = make_piece(_artifact(tmp_path))

    result = _export(tmp_path, contract=contract, pieces=[piece], approve=True, tag="t4")

    manual = [r.detail for r in result.reminders if r.kind == "manual_review"]
    assert any("audio.no_trending" in detail for detail in manual)


def test_reminder_is_still_emitted_for_the_platform_that_declares_the_rule(
    tmp_path: Path,
) -> None:
    """T5: la otra mitad de #57. El recordatorio no se ha perdido.

    Filtrar por plataforma no puede degenerar en callarlos todos.
    """
    tiktok = _piece(_artifact(tmp_path), piece_id="tk", platform=Platform.TIKTOK)

    result = _export(tmp_path, contract=_mixed_contract(), pieces=[tiktok], approve=True, tag="t5")

    manual = [r.detail for r in result.reminders if r.kind == "manual_review"]
    assert any("audio.no_trending" in detail for detail in manual)


def test_recommended_rule_with_unsupported_check_blocks_the_gate(tmp_path: Path) -> None:
    """T6: una regla recommended sin validador bloquea, y ya no se exporta.

    Este test fijaba lo contrario: declaraba ``link.in_bio`` como recommended y
    comprobaba que su UNSUPPORTED se traducía en un gate ``passed`` y una pieza
    EXPORTADA. Eso congelaba un fail-open, no una intencion. El razonamiento que
    lo justificaba ("esa condicion solo mira las reglas hard") era una
    descripcion de un defecto, no una decision.

    Conviene decir como se mantuvo tanto tiempo: NO era un test que nadie miraba.
    Era un test que AFIRMABA el defecto, con `assert result.status is PASSED` y
    `assert exported.status is EXPORTED`. Green y equivocado a la vez.

    El brief, §5.1, dice: "Si no hay validador para una regla DURA, el resultado
    es UNSUPPORTED o MANUAL_REVIEW, jamas PASS". Ojo a la palabra DURA: la
    clausula esta acotada a las reglas duras, y este caso es `recommended`. Asi
    que este test no es el brief aplicado, es un endurecimiento deliberado mas
    alla de §5.1, y el motivo no es de norma sino de salida: `PENDING_REVIEW` es una
    salida, y una firma es una atestacion de una persona. Una persona no puede
    atestiguar una geometria que no se midio, de modo que "no verificable por
    nadie" no puede ser un estado firmable.

    Lo que este test sigue fijando, y sigue siendo cierto: el check se despacha
    y se registra. No se omite en silencio.
    """
    contract = make_contract(
        audio_rule="any",
        hard=["artifact.integrity", "duration.min"],
        recommended=["link.in_bio"],
    )
    piece = make_piece(_artifact(tmp_path))

    result = _gate().run(contract=contract, piece=piece, assets=AssetRegistry(tmp_path))
    assert _check(result, "link.in_bio").status is CheckStatus.UNSUPPORTED
    assert result.status is GateStatus.UNSUPPORTED

    exported = _export(tmp_path, contract=contract, pieces=[piece], approve=False, tag="t6")
    assert exported.status is not ExportStatus.EXPORTED

    # Y con la firma de aprobacion manual, que es donde el fail-open se
    # escondia: `_is_piece_exportable` solo rescata PENDING_REVIEW, y aqui el
    # estado es UNSUPPORTED. El estado del gate no basta para probarlo; hace
    # falta pasar por el exportador.
    con_firma = _export(tmp_path, contract=contract, pieces=[piece], approve=True, tag="t6b")
    assert con_firma.status is not ExportStatus.EXPORTED


def test_contract_without_manual_review_rules_exports_clean(tmp_path: Path) -> None:
    """T7: sin reglas manual_review no hay nada que revisar ni que recordar.

    Este test NO es cobertura de aplicabilidad: con cero reglas declaradas el
    filtro no tiene nada que filtrar. Lo que fija es que un contrato sin
    revisiones pendientes sale limpio y sin atribuir aprobaciones.
    """
    contract = make_contract(audio_rule="any", hard=["artifact.integrity", "duration.min"])
    piece = make_piece(_artifact(tmp_path))

    result = _export(tmp_path, contract=contract, pieces=[piece], approve=False, tag="t7")

    assert result.status is ExportStatus.EXPORTED
    assert [r for r in result.reminders if r.kind == "manual_review"] == []
    assert result.approved_by is None


def _two_platforms(declaring_rule: str) -> Contract:
    """Contrato donde tiktok declara ``declaring_rule`` y reels declara ``any``.

    Args:
        declaring_rule: ``audio_rule`` de la plataforma que si exige audio.

    Returns:
        El contrato de dos plataformas.
    """
    return with_platform(
        make_contract(
            audio_rule=declaring_rule,
            official_audio_url=(
                "https://example.test/official" if declaring_rule == "official_required" else None
            ),
            hard=_AUDIO_HARD,
        ),
        Platform.INSTAGRAM_REELS,
        PlatformRules(),
    )


@pytest.mark.parametrize(
    ("audio_rule", "rule_id"),
    [
        ("no_trending", "audio.no_trending"),
        ("own_clip", "audio.own_clip"),
        ("official_required", "audio.official_track"),
    ],
)
def test_each_seeded_predicate_scopes_its_own_rule(
    tmp_path: Path,
    audio_rule: str,
    rule_id: str,
) -> None:
    """Un test por predicado sembrado, para que borrar cualquiera se note.

    Cada predicado de la tabla tiene aqui su propio caso. Sin esto, borrar
    ``audio.own_clip`` o ``audio.official_track`` de la tabla no rompia nada y
    la tabla podia afirmar cobertura que no tenia.
    """
    contract = _two_platforms(audio_rule)
    reels = _piece(_artifact(tmp_path), piece_id="ig", platform=Platform.INSTAGRAM_REELS)
    tiktok = _piece(_artifact(tmp_path), piece_id="tk", platform=Platform.TIKTOK)

    gate = _gate()
    reels_result = gate.run(contract=contract, piece=reels, assets=AssetRegistry(tmp_path))
    tiktok_result = gate.run(contract=contract, piece=tiktok, assets=AssetRegistry(tmp_path))

    # La plataforma que no lo pidio no lo carga: ni el check, ni el estado.
    assert rule_id not in [check.id for check in reels_result.checks]
    assert reels_result.status is not GateStatus.PENDING_REVIEW
    # La que si lo pidio lo carga y por eso sigue pidiendo firma.
    assert rule_id in [check.id for check in tiktok_result.checks]
    assert tiktok_result.status is GateStatus.PENDING_REVIEW


@pytest.mark.parametrize("alias", ["audio.official_selection", "audio.rule"])
def test_alias_rule_ids_scope_the_same_as_the_canonical_one(
    tmp_path: Path,
    alias: str,
) -> None:
    """Un alias no puede esquivar el scoping.

    El contrato acepta ``audio.official_selection`` y ``audio.rule`` como
    clasificacion valida de ``audio.official_track``. Si el predicado solo
    reconoce la grafia canonica, un alias deja pasar el defecto: la regla se despacha
    para plataformas que no la piden.
    """
    contract = _two_platforms("official_required")
    reels = _piece(_artifact(tmp_path), piece_id="ig", platform=Platform.INSTAGRAM_REELS)

    with_alias = contract.model_copy(
        update={
            "rules": contract.rules.model_copy(
                update={
                    "manual_review": [
                        *(r for r in contract.rules.manual_review if r != "audio.official_track"),
                        alias,
                    ]
                }
            )
        }
    )

    result = _gate().run(contract=with_alias, piece=reels, assets=AssetRegistry(tmp_path))

    assert alias not in [check.id for check in result.checks]
    assert result.status is not GateStatus.PENDING_REVIEW


@pytest.mark.parametrize("alias", ["audio.official_selection", "audio.rule"])
def test_validator_registered_under_canonical_id_is_reached_by_every_spelling(
    tmp_path: Path,
    alias: str,
) -> None:
    """El lookup de validadores canoniza, no solo el predicado.

    Si solo canonicalizara la tabla de aplicabilidad, el primer validador que se
    registrara bajo ``audio.official_track`` seria esquivado por cualquier
    alias, que pasaria a caer en la rama de "sin validador" y a mostrar un
    motivo equivocado. Se registra bajo el id canonico y se despacha por alias.
    """
    contract = make_contract(audio_rule="any", hard=["artifact.integrity", "duration.min", alias])
    gate = Gate(FakeProbe(info=make_media(has_audio=True, has_video=True)))
    gate.register_validator(
        "audio.official_track",
        lambda _context: CheckOutcome(
            status=CheckStatus.MANUAL_REVIEW, evidence={"canonico": True}
        ),
    )

    result = gate.run(
        contract=contract,
        piece=make_piece(_artifact(tmp_path)),
        assets=AssetRegistry(tmp_path),
    )

    check = _check(result, alias)
    assert check.evidence.get("canonico") is True


def test_rule_declared_but_claimed_by_no_platform_is_recorded_and_blocks(
    tmp_path: Path,
) -> None:
    """CASO C: una regla declarada que nadie pide se REGISTRA, no se silencia.

    Este es el fail-open que casi entra: con solo el scoping por plataforma, una
    regla ``hard`` que ninguna plataforma activa desaparecia del ``gate.json`` y
    la pieza salia ``passed`` con el ``contract_sha256`` apuntando a un
    contrato que la declaraba dura.
    """
    contract = make_contract(
        audio_rule="any",
        hard=["artifact.integrity", "duration.min", "audio.no_trending"],
    )
    piece = _piece(_artifact(tmp_path), piece_id="tk", platform=Platform.TIKTOK)

    result = _gate().run(contract=contract, piece=piece, assets=AssetRegistry(tmp_path))

    orphan = _check(result, "audio.no_trending")
    assert orphan.status is CheckStatus.UNSUPPORTED
    assert result.status is GateStatus.UNSUPPORTED
    assert "ninguna plataforma" in str(orphan.evidence["unclaimed"])

    for approve in (False, True):
        report = _export(
            tmp_path, contract=contract, pieces=[piece], approve=approve, tag=f"t8-{approve}"
        )
        assert report.status is ExportStatus.BLOCKED


def test_manual_review_rule_claimed_by_no_platform_is_recorded_and_blocks(
    tmp_path: Path,
) -> None:
    """CASO C tambien con fuerza ``manual_review``, que es el caso de #59.

    Comprueba que el estado registrado es el mismo que producia la ausencia de
    scoping, para una regla ``manual_review`` huerfana.
    """
    contract = make_contract(
        audio_rule="any",
        hard=["artifact.integrity", "duration.min"],
        manual_review=["audio.no_trending"],
    )
    piece = _piece(_artifact(tmp_path), piece_id="tk", platform=Platform.TIKTOK)

    result = _gate().run(contract=contract, piece=piece, assets=AssetRegistry(tmp_path))

    assert _check(result, "audio.no_trending").status is CheckStatus.MANUAL_REVIEW
    assert result.status is GateStatus.PENDING_REVIEW


# --- #63: atribucion y link.in_bio tambien son requisitos por plataforma ------


def _contract_with_platform_restrictions(
    *,
    tiktok_rules: PlatformRules,
    reels_rules: PlatformRules,
    hard: Sequence[str],
    manual_review: Sequence[str] = (),
) -> Contract:
    """Contrato de dos plataformas con reglas de atribucion/link distintas.

    Args:
        tiktok_rules: Reglas de la plataforma que SI declara las restricciones.
        reels_rules: Reglas de la plataforma que NO las declara.
        hard: Reglas duras declaradas globalmente.
        manual_review: Reglas de revision manual declaradas globalmente.

    Returns:
        El contrato de dos plataformas, ya revalidado.
    """
    base = make_contract(audio_rule="any", hard=hard, manual_review=manual_review)
    return with_platform(
        with_platform(base, Platform.TIKTOK, tiktok_rules), Platform.INSTAGRAM_REELS, reels_rules
    )


_LINK_RULES_ON = PlatformRules(link_rules=LinkRules(link_in_bio=True))
_LINK_RULES_OFF = PlatformRules()
_ATTRIBUTION_ON = PlatformRules(attribution=Attribution(type=AttributionType.TAG, value="@marca"))
_ATTRIBUTION_OFF = PlatformRules()


def test_link_in_bio_does_not_block_a_platform_that_never_declared_it(tmp_path: Path) -> None:
    """#63: `link.in_bio` es de tiktok; reels no lo pidió y no paga su firma.

    Sin predicado de aplicabilidad, la regla se despacha en TODAS las plataformas
    del contrato y reels queda bloqueada por un requisito que no declaró.
    """
    contract = _contract_with_platform_restrictions(
        tiktok_rules=_LINK_RULES_ON,
        reels_rules=_LINK_RULES_OFF,
        hard=["artifact.integrity", "duration.min", "link.in_bio"],
    )
    reels = _piece(_artifact(tmp_path), piece_id="ig", platform=Platform.INSTAGRAM_REELS)

    result = _gate().run(contract=contract, piece=reels, assets=AssetRegistry(tmp_path))

    assert "link.in_bio" not in [check.id for check in result.checks]
    assert result.status is GateStatus.PASSED


def test_link_in_bio_still_blocks_the_platform_that_declared_it(tmp_path: Path) -> None:
    """NO-REGRESION: tiktok si pidió el link y sigue exigiendo revision.

    Si este test falla, #63 ha abierto un agujero: la plataforma que declara la
    regla estaría publicando sin que nadie mire.
    """
    contract = _contract_with_platform_restrictions(
        tiktok_rules=_LINK_RULES_ON,
        reels_rules=_LINK_RULES_OFF,
        hard=["artifact.integrity", "duration.min", "link.in_bio"],
    )
    tiktok = _piece(_artifact(tmp_path), piece_id="tk", platform=Platform.TIKTOK)

    result = _gate().run(contract=contract, piece=tiktok, assets=AssetRegistry(tmp_path))

    assert _check(result, "link.in_bio").status is CheckStatus.UNSUPPORTED
    assert result.status is GateStatus.UNSUPPORTED


def test_attribution_required_does_not_block_a_platform_with_attribution_none(
    tmp_path: Path,
) -> None:
    """#63, atribucion: `attribution.type=none` significa que no hay requisito."""
    contract = _contract_with_platform_restrictions(
        tiktok_rules=_ATTRIBUTION_ON,
        reels_rules=_ATTRIBUTION_OFF,
        hard=["artifact.integrity", "duration.min", "attribution.required"],
    )
    reels = _piece(_artifact(tmp_path), piece_id="ig", platform=Platform.INSTAGRAM_REELS)

    result = _gate().run(contract=contract, piece=reels, assets=AssetRegistry(tmp_path))

    assert "attribution.required" not in [check.id for check in result.checks]
    assert result.status is GateStatus.PASSED


def test_attribution_required_still_blocks_the_platform_that_declared_it(
    tmp_path: Path,
) -> None:
    """NO-REGRESION de atribucion."""
    contract = _contract_with_platform_restrictions(
        tiktok_rules=_ATTRIBUTION_ON,
        reels_rules=_ATTRIBUTION_OFF,
        hard=["artifact.integrity", "duration.min", "attribution.required"],
    )
    tiktok = _piece(_artifact(tmp_path), piece_id="tk", platform=Platform.TIKTOK)

    result = _gate().run(contract=contract, piece=tiktok, assets=AssetRegistry(tmp_path))

    assert _check(result, "attribution.required").status is CheckStatus.UNSUPPORTED
    assert result.status is GateStatus.UNSUPPORTED


@pytest.mark.parametrize(
    ("alias", "canonico", "exigida", "no_exigida"),
    [
        ("attribution.present", "attribution.required", _ATTRIBUTION_ON, _ATTRIBUTION_OFF),
        ("link_rules.link_in_bio", "link.in_bio", _LINK_RULES_ON, _LINK_RULES_OFF),
    ],
)
def test_los_alias_de_atribucion_y_link_respetan_el_predicado(
    alias: str,
    canonico: str,
    exigida: PlatformRules,
    no_exigida: PlatformRules,
) -> None:
    """Un alias debe heredar la aplicabilidad de su id canonico, no la suprimir.

    Si el motor buscara el predicado por el alias literal, no lo encontraria y la
    plataforma sin requisitos volveria a quedar bloqueada: el alias seria una
    puerta trasera del defecto de #63.
    """
    assert canonical_rule_id(alias) == canonico
    # El alias y su canonico tienen que coincidir en las DOS plataformas, no solo
    # en la que exige: si solo coincidieran en una, el fallo aparece en runtime.
    assert rule_applies(alias, exigida) is True
    assert rule_applies(alias, no_exigida) is False
    assert rule_applies(alias, exigida) is rule_applies(canonico, exigida)
    assert rule_applies(alias, no_exigida) is rule_applies(canonico, no_exigida)


# --- #57: los recordatorios tambien son por plataforma ------------------------


def _audio_contract() -> Contract:
    """Tiktok exige `no_trending`; reels declara `any` y no exige nada.

    Returns:
        El contrato de dos plataformas con una regla de audio que solo una pide.
    """
    return _contract_with_platform_restrictions(
        tiktok_rules=PlatformRules(audio_rule=AudioRule.NO_TRENDING),
        reels_rules=PlatformRules(audio_rule=AudioRule.ANY),
        hard=["artifact.integrity", "duration.min"],
        manual_review=["audio.no_trending"],
    )


def test_recordatorio_de_audio_ya_no_aparece_en_una_entrega_solo_de_reels(
    tmp_path: Path,
) -> None:
    """#57: una entrega solo de reels no recuerda una regla que solo pide tiktok.

    El recordatorio global obligaba al operador a repasar `audio.no_trending`
    en entregas donde ninguna pieza podia cumplirlo. Enseñar a ignorar el
    informe es peor que no mostrarlo.
    """
    reels = _piece(_artifact(tmp_path), piece_id="ig", platform=Platform.INSTAGRAM_REELS)

    report = _export(tmp_path, contract=_audio_contract(), pieces=[reels], approve=True, tag="ig")

    assert [r.detail for r in report.reminders if r.kind == "manual_review"] == []


def test_recordatorio_de_audio_si_aparece_cuando_la_entrega_tiene_una_pieza_de_tiktok(
    tmp_path: Path,
) -> None:
    """NO-REGRESION de #57: con tiktok en la entrega, el recordatorio vuelve."""
    tiktok = _piece(_artifact(tmp_path), piece_id="tk", platform=Platform.TIKTOK)

    report = _export(tmp_path, contract=_audio_contract(), pieces=[tiktok], approve=True, tag="tk")

    assert any(
        r.kind == "manual_review" and "audio.no_trending" in r.detail for r in report.reminders
    )


def test_recordatorio_sigue_apareciendo_para_una_regla_sin_predicado_de_plataforma(
    tmp_path: Path,
) -> None:
    """Una regla que aplica en todas las plataformas no puede desaparecer nunca.

    Si el alcance no la puede acotar, el recordatorio se mantiene: el filtro
    solo puede quitar recordatorios, nunca inventar una ausencia.
    """
    contract = _contract_with_platform_restrictions(
        tiktok_rules=PlatformRules(),
        reels_rules=PlatformRules(),
        hard=["artifact.integrity", "duration.min"],
        manual_review=["caption.tone"],
    )
    reels = _piece(_artifact(tmp_path), piece_id="ig", platform=Platform.INSTAGRAM_REELS)

    report = _export(tmp_path, contract=contract, pieces=[reels], approve=True, tag="tone")

    assert any(r.kind == "manual_review" and "caption.tone" in r.detail for r in report.reminders)


@pytest.mark.parametrize(
    ("exigida_tiktok", "regla"),
    [(_LINK_RULES_ON, "link.in_bio"), (_ATTRIBUTION_ON, "attribution.required")],
)
def test_recordatorio_de_atribucion_y_link_tambien_se_acota_a_la_entrega(
    tmp_path: Path,
    exigida_tiktok: PlatformRules,
    regla: str,
) -> None:
    """Los recordatorios de las reglas nuevas de #63 obeyecen el mismo filtro.

    Sin este caso, `_recuerda_regla` podria estar acotando solo el audio y las
    reglas de atribucion y link seguirian apareciendo en todas las entregas.
    """
    contract = _contract_with_platform_restrictions(
        tiktok_rules=exigida_tiktok,
        reels_rules=PlatformRules(),
        hard=["artifact.integrity", "duration.min"],
        manual_review=[regla],
    )
    reels = _piece(_artifact(tmp_path), piece_id="ig", platform=Platform.INSTAGRAM_REELS)

    report = _export(tmp_path, contract=contract, pieces=[reels], approve=True, tag=f"ig-{regla}")

    assert [r.detail for r in report.reminders if r.kind == "manual_review"] == []

    tiktok = _piece(_artifact(tmp_path), piece_id="tk", platform=Platform.TIKTOK)
    con_tiktok = _export(
        tmp_path, contract=contract, pieces=[tiktok], approve=True, tag=f"tk-{regla}"
    )

    assert any(r.kind == "manual_review" and regla in r.detail for r in con_tiktok.reminders)


# ---------------------------------------------------- matriz enumerada (#65)

_REGLAS_MATRIZ: tuple[str, ...] = (
    "audio.present",
    "audio.own_clip",
    "audio.no_trending",
    "audio.official_track",
    "audio.official_selection",
    "audio.rule",
    "attribution.required",
    "attribution.present",
    "link.in_bio",
    "link_rules.link_in_bio",
)

_PLATAFORMAS_MATRIZ: tuple[Platform, ...] = (
    Platform.TIKTOK,
    Platform.INSTAGRAM_REELS,
    Platform.YOUTUBE_SHORTS,
    Platform.X,
)


def _reglas_activas_para(regla: str) -> PlatformRules:
    """Construye las PlatformRules que activan el requisito para esta regla.

    Args:
        regla: Identificador de la regla o alias a activar.

    Returns:
        Las PlatformRules con la configuración que activa la regla.

    Raises:
        ValueError: Si la regla no pertenece al catálogo de la matriz.
    """
    canonico = canonical_rule_id(regla)
    if canonico in {"audio.present", "audio.own_clip"}:
        return PlatformRules(audio_rule=AudioRule.OWN_CLIP)
    if canonico == "audio.no_trending":
        return PlatformRules(audio_rule=AudioRule.NO_TRENDING)
    if canonico == "audio.official_track":
        return PlatformRules(audio_rule=AudioRule.OFFICIAL_REQUIRED)
    if canonico == "attribution.required":
        return PlatformRules(attribution=Attribution(type=AttributionType.TAG, value="@marca"))
    if canonico == "link.in_bio":
        return PlatformRules(link_rules=LinkRules(link_in_bio=True))
    msg = f"regla desconocida en la matriz: {regla}"
    raise ValueError(msg)


def _reglas_neutras() -> PlatformRules:
    """PlatformRules con todos los requisitos de la matriz inactivos.

    Returns:
        PlatformRules con configuración neutral para todas las reglas.
    """
    return PlatformRules(
        audio_rule=AudioRule.ANY,
        attribution=Attribution(type=AttributionType.NONE),
        link_rules=LinkRules(link_in_bio=False),
    )


def _requisito_activo_en(regla: str, rules: PlatformRules) -> bool:
    """Verifica directamente el campo de PlatformRules sin invocar rule_applies.

    Se inspecciona el estado puro del modelo para asegurar que el fixture
    construyó exactamente la condición requerida por el caso de prueba antes
    de evaluar la predicada.

    Args:
        regla: Identificador de la regla o alias.
        rules: Reglas de plataforma a inspeccionar.

    Returns:
        True si la regla está activada según los campos del modelo.

    Raises:
        ValueError: Si la regla no pertenece al catálogo de la matriz.
    """
    canonico = canonical_rule_id(regla)
    if canonico == "audio.present":
        return rules.audio_rule is not AudioRule.ANY
    if canonico == "audio.own_clip":
        return rules.audio_rule is AudioRule.OWN_CLIP
    if canonico == "audio.no_trending":
        return rules.audio_rule is AudioRule.NO_TRENDING
    if canonico == "audio.official_track":
        return rules.audio_rule is AudioRule.OFFICIAL_REQUIRED
    if canonico == "attribution.required":
        return rules.attribution.type is not AttributionType.NONE
    if canonico == "link.in_bio":
        return bool(rules.link_rules.link_in_bio)
    msg = f"regla desconocida en la matriz: {regla}"
    raise ValueError(msg)


def _construir_contrato_matriz(
    configuraciones: dict[Platform, PlatformRules],
) -> Contract:
    """Construye un contrato con las cuatro plataformas y todas las reglas clasificadas.

    Args:
        configuraciones: Mapeo de plataforma a sus PlatformRules.

    Returns:
        El contrato multi-plataforma construido y revalidado.
    """
    c = make_contract(
        audio_rule="any",
        official_audio_url="https://example.test/sound",
        hard=["artifact.integrity", "duration.min"],
        manual_review=list(_REGLAS_MATRIZ),
    )
    for p, r in configuraciones.items():
        c = with_platform(c, p, r)
    return c


@pytest.mark.parametrize("caso", ["A", "B", "C"])
@pytest.mark.parametrize("target", _PLATAFORMAS_MATRIZ)
@pytest.mark.parametrize("regla", _REGLAS_MATRIZ)
def test_matriz_enumerada_de_aplicabilidad_de_reglas_por_plataforma(
    caso: str,
    target: Platform,
    regla: str,
) -> None:
    """#65: enumera todas las reglas, plataformas y casos A/B/C.

    - Caso A: la plataforma target declara el requisito.
    - Caso B: otra plataforma distinta declara el requisito; target no lo declara.
    - Caso C: ninguna plataforma declara el requisito.

    Invariante: antes de comprobar el veredicto, el test comprueba explícitamente
    qué construyó el fixture para no pasar en verde por una construcción vacía.
    """
    otra = next(p for p in _PLATAFORMAS_MATRIZ if p != target)
    if caso == "A":
        cfg = {
            p: (_reglas_activas_para(regla) if p == target else _reglas_neutras())
            for p in _PLATAFORMAS_MATRIZ
        }
    elif caso == "B":
        cfg = {
            p: (_reglas_activas_para(regla) if p == otra else _reglas_neutras())
            for p in _PLATAFORMAS_MATRIZ
        }
    else:  # Caso C
        cfg = {p: _reglas_neutras() for p in _PLATAFORMAS_MATRIZ}

    contrato = _construir_contrato_matriz(cfg)

    # 1. Invariante: aserción estricta de la precondición construida en el fixture.
    activas = [p for p in _PLATAFORMAS_MATRIZ if _requisito_activo_en(regla, contrato.platforms[p])]
    if caso == "A":
        assert activas == [target], f"Precondición Caso A falló: activas={activas}"
    elif caso == "B":
        assert target not in activas, f"Precondición Caso B falló: target {target} activa"
        assert activas == [otra], f"Precondición Caso B falló: activas={activas}"
    else:  # Caso C
        assert activas == [], f"Precondición Caso C falló: activas={activas}"

    # 2. Invariante: veredicto de aplicabilidad en target y globalmente.
    if caso == "A":
        assert rule_applies(regla, contrato.platforms[target]) is True
        assert rule_applies_anywhere(regla, contrato) is True
    elif caso == "B":
        assert rule_applies(regla, contrato.platforms[target]) is False
        assert rule_applies_anywhere(regla, contrato) is True
    else:  # Caso C
        assert rule_applies(regla, contrato.platforms[target]) is False
        assert rule_applies_anywhere(regla, contrato) is False
