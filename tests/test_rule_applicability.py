"""Aplicabilidad por plataforma de las reglas declaradas globalmente.

``contract.rules`` es global: una sola declaracion cubre todas las plataformas.
Lo que describen es por plataforma. Estos tests fijan que una regla solo
afecta a las plataformas que la declararon, en los DOS consumidores que dependen
de la plataforma: el motor, al derivar el estado, y el exportador, al listar
recordatorios.

El riesgo que estos tests cubren no es que el motor pase de mas, sino que pase
de menos: una plataforma que SI declara una regla tiene que seguir exigiendo su
firma. Por eso hay tests de no-regresion explicitos.
"""

from pathlib import Path

from kliptych.assets import AssetRegistry
from kliptych.contract import Contract, Platform, contract_digest
from kliptych.contract.schema import PlatformRules
from kliptych.exporter import DeliveryReport, ExportStatus, export_delivery
from kliptych.gate import CheckResult, CheckStatus, Gate, GateResult, GateStatus, Piece
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


def test_reminder_is_not_emitted_when_no_platform_declares_the_rule(
    tmp_path: Path,
) -> None:
    """T4: cierra la parte vigente de #57.

    ``audio.no_trending`` esta declarada globalmente a mano, pero NINGUNA
    plataforma del contrato la pidio: las dos declaran ``audio_rule=any``. El
    recordatorio se suprime porque no le aplica a ninguna plataforma.

    Lo que NO se arregla aqui, y conviene saber: cuando una plataforma si lo
    pide y otra no, el recordatorio sigue apareciendo en el informe. ``
    DeliveryReport.reminders`` es una lista plana de todo el informe, no una por
    pieza, asi que no hay forma de mostrarlo para tiktok y callarlo para
    instagram dentro del mismo informe. Eso exigiria recordatorios por pieza,
    que es un cambio de schema y no cabe aqui.
    """
    contract = make_contract(
        audio_rule="any",
        hard=["artifact.integrity", "duration.min"],
        manual_review=["audio.no_trending"],
    )
    piece = make_piece(_artifact(tmp_path))

    result = _export(tmp_path, contract=contract, pieces=[piece], approve=False, tag="t4")

    assert result.status is ExportStatus.EXPORTED
    assert [r for r in result.reminders if r.kind == "manual_review"] == []


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


def test_recommended_rule_with_unsupported_check_is_unaffected(tmp_path: Path) -> None:
    """T6: una regla recommended sin validador se comporta igual que en main.

    ``link.in_bio`` no tiene validador registrado. Declarada como recommended, su
    UNSUPPORTED no deriva en UNSUPPORTED de gate, porque esa condicion solo mira
    las reglas ``hard``. Este PR no debe alterar esa combinacion.
    """
    contract = make_contract(
        audio_rule="any",
        hard=["artifact.integrity", "duration.min"],
        recommended=["link.in_bio"],
    )
    piece = make_piece(_artifact(tmp_path))

    result = _gate().run(contract=contract, piece=piece, assets=AssetRegistry(tmp_path))
    assert _check(result, "link.in_bio").status is CheckStatus.UNSUPPORTED
    assert result.status is GateStatus.PASSED

    exported = _export(tmp_path, contract=contract, pieces=[piece], approve=False, tag="t6")
    assert exported.status is ExportStatus.EXPORTED


def test_contract_without_manual_review_rules_exports_clean(tmp_path: Path) -> None:
    """T7: sin reglas manual_review no hay nada que revisar ni que recordar."""
    contract = make_contract(audio_rule="any", hard=["artifact.integrity", "duration.min"])
    piece = make_piece(_artifact(tmp_path))

    result = _export(tmp_path, contract=contract, pieces=[piece], approve=False, tag="t7")

    assert result.status is ExportStatus.EXPORTED
    assert [r for r in result.reminders if r.kind == "manual_review"] == []
    assert result.approved_by is None
