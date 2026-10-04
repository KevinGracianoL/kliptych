"""El gate valida el ASPECTO DECLARADO, no una constante de modulo.

PR 2 de la escalera de #76. PR 1 anadio la declaracion; aqui se verifica.

La pregunta que este archivo responde es una sola: cuando el contrato dice
16:9, ¿el gate mira el 16:9 del contrato, o recalcula el aspecto desde la
geometria y espera que salga 9:16? Son dos cosas distintas. Una honra el
contrato; la otra adivina y encima lo hardcodea.

La invariante que esta PR existe para hacer cumplir:

    Una regla declarada que no se puede verificar mecanicamente es UNSUPPORTED,
    nunca PASS. Y un aspecto declarado que el gate no puede medir contra el
    artefacto tambien es UNSUPPORTED, no PASS.

No hay red, ni ffmpeg, ni ffprobe: el `MediaInfo` del contexto es sintetico.
La inspeccion real ya la cubre `test_gate_probe.py`.
"""

from collections.abc import Sequence
from pathlib import Path

import pytest

from kliptych.assets import AssetRegistry
from kliptych.contract.enums import Aspect
from kliptych.gate.checks import (
    DEFAULT_VALIDATORS,
    CheckOutcome,
    GateContext,
    check_declared_outputs,
    check_split_screen_geometry,
)
from kliptych.gate.models import CheckStatus, MediaInfo
from tests.support import make_contract, make_piece

_VERTICAL: dict[str, object] = {"aspect": "9:16", "index": 0}
_HORIZONTAL: dict[str, object] = {"aspect": "16:9", "index": 1}
_DUAL: list[dict[str, object]] = [_VERTICAL, _HORIZONTAL]


def _context(
    tmp_path: Path,
    *,
    outputs: Sequence[dict[str, object]] | None = None,
    split_aspect: str | None = "9:16",
    width: int | None = 1080,
    height: int | None = 1920,
    has_video: bool = True,
    with_probe: bool = True,
) -> GateContext:
    """Arma un contexto de gate con la declaracion y el artefacto que se eligen.

    El contrato declara `artifact.aspect` ademas de las reglas de split, porque
    la pertenencia a las salidas declaradas corre en su propio validador y no
    atada al lienzo.

    Args:
        tmp_path: Directorio de trabajo del asset registry.
        outputs: Las salidas declaradas, o `None` para un contrato 1.1.
        split_aspect: Aspecto del lienzo declarado, o `None` para no declarar
            `split_screen`.
        width: Ancho reportado por ffprobe.
        height: Alto reportado por ffprobe.
        has_video: Si el artefacto tiene pista de video.
        with_probe: Si el probe devolvio media o fallo.

    Returns:
        El contexto, con el contrato ya validado.
    """
    split: dict[str, object] | None = None
    if split_aspect is not None:
        lienzo: dict[str, object] = {
            "aspect": split_aspect,
            "top_source": "clip-top",
            "bottom_source": "clip-bottom",
        }
        if split_aspect == "16:9":
            lienzo["width"] = 1920
            lienzo["height"] = 1080
        split = lienzo

    reglas = ["artifact.integrity", "artifact.video_stream", "artifact.aspect"]
    if split is not None:
        reglas.append("layout.geometry")
    contract = make_contract(
        hard=reglas,
        audio_rule="any",
        min_s=None,
        max_s=None,
        required_mentions=(),
        required_hashtags=(),
        schema_version="1.2" if outputs is not None else "1.1",
        outputs=outputs,
        split_screen=split,
    )
    media = None
    if with_probe:
        media = MediaInfo(
            width=width,
            height=height,
            has_video=has_video,
            duration_s=15.0,
            has_audio=True,
            format_name="mov,mp4",
        )
    return GateContext(
        contract=contract,
        rules=contract.platforms[next(iter(contract.platforms))],
        piece=make_piece(tmp_path / "a.mp4"),
        artifact_sha256="a" * 64,
        media=media,
        assets=AssetRegistry(tmp_path),
    )


def test_a1_vertical_declarado_con_artefacto_vertical_pasa(tmp_path: Path) -> None:
    """El caso de siempre: 9:16 declarado, 9:16 medido."""
    outcome = check_split_screen_geometry(_context(tmp_path))

    assert outcome.status is CheckStatus.PASS
    assert outcome.evidence["aspect_ratio"] == "9:16"
    assert outcome.evidence["declared_aspect"] == "9:16"


def test_a2_declarado_vertical_con_artefacto_horizontal_falla(tmp_path: Path) -> None:
    """9:16 declarado, 16:9 medido: FAIL, y la evidencia nombra los DOS.

    Los dos numeros importan. Con solo el medido no se sabe que se exigio algo;
    con solo el declarado no se ve que hay que arreglar. Y el nombre del aspecto
    declarado tiene que estar, porque "exige 9:16" a secas no dice de donde sale.
    """
    outcome = check_declared_outputs(
        _context(
            tmp_path,
            outputs=[_VERTICAL],
            split_aspect=None,
            width=1920,
            height=1080,
        )
    )

    assert outcome.status is CheckStatus.FAIL
    assert outcome.evidence["declared_aspects"] == ["9:16"]
    assert outcome.evidence["measured_aspect"] == "16:9"


def test_a3_horizontal_declarado_con_artefacto_horizontal_pasa(tmp_path: Path) -> None:
    """16:9 declarado, 16:9 medido: PASS. Esta es la razon de ser de la PR.

    Con la constante de 9:16, este contrato no podria pasar jams: el gate
    compararia 1920x1080 contra 9 y 16 y fallaria con un mensaje que blames el
    contrato por algo que el contrato dijo bien.
    """
    outcome = check_split_screen_geometry(
        _context(
            tmp_path,
            outputs=[_HORIZONTAL],
            split_aspect="16:9",
            width=1920,
            height=1080,
        )
    )

    assert outcome.status is CheckStatus.PASS
    assert outcome.evidence["aspect_ratio"] == "16:9"


def test_a4_el_artefacto_no_puede_ir_mas_grande_que_la_geometria(tmp_path: Path) -> None:
    """Un artefacto con las dimensiones correctas pero otro aspecto no pasa.

    Sin lienzo, `check_split_screen_geometry` falla: la regla exige declaracion.
    Este caso necesita lienzo declarado, asi que va por `a3` con las dimensiones
    cruzadas, que es lo que el gate tiene que ver.
    """
    outcome = check_split_screen_geometry(_context(tmp_path, width=1920, height=1080))

    assert outcome.status is CheckStatus.FAIL
    assert "dimensiones" in str(outcome.evidence["reason"])


# ---------------------------------------------------------------- membership


def test_b1_una_de_dos_salidas_satisfecha_pasa(tmp_path: Path) -> None:
    """El artefacto tiene que satisfacer AL MENOS una salida declarada."""
    outcome = check_declared_outputs(
        _context(tmp_path, outputs=_DUAL, split_aspect=None, width=1080, height=1920)
    )

    assert outcome.status is CheckStatus.PASS
    assert outcome.evidence["satisfied_by"] == "9:16"
    assert outcome.evidence["declared_aspects"] == ["9:16", "16:9"]


def test_b2_ninguna_salida_satisfecha_falla(tmp_path: Path) -> None:
    """Satisfacer una no basta: hay que satisfacer ALGUNA.

    Este es el fallo que la pertenencia existe para cazar. Sin ella, un contrato
    que declara (9:16, 16:9) aceptaria cualquier artefacto, porque siempre habria
    un formato declarado que "podria" ser. Un 1x1 no es ninguno de los dos.
    """
    outcome = check_declared_outputs(
        _context(tmp_path, outputs=_DUAL, split_aspect=None, width=1000, height=1000)
    )

    assert outcome.status is CheckStatus.FAIL
    assert outcome.evidence["measured_aspect"] == "1000:1000"
    assert outcome.evidence["declared_aspects"] == ["9:16", "16:9"]


def test_b2b_la_pertenencia_corre_sin_split_screen(tmp_path: Path) -> None:
    """La comprobacion no depende de que el contrato traiga lienzo.

    Un contrato multi-formato no puede declarar lienzo (limitacion deliberada de
    PR 1), asi que si la pertenencia viviera dentro del validador del lienzo no
    existiria justo donde hace falta.
    """
    sin_lienzo = _context(tmp_path, outputs=_DUAL, split_aspect=None)

    assert check_declared_outputs(sin_lienzo).status is CheckStatus.PASS


def test_b2c_la_pertenencia_es_una_regla_registrada() -> None:
    """`artifact.aspect` tiene que estar en el catalogo de validadores.

    Sin registro, un contrato que la declare degrada a UNSUPPORTED por "no hay
    validador mecanico", que es un diagnostico falso: el validador existe.
    """
    assert "artifact.aspect" in DEFAULT_VALIDATORS
    assert DEFAULT_VALIDATORS["artifact.aspect"] is check_declared_outputs


def test_b3_un_1_1_se_mide_contra_su_default_y_lo_dice(tmp_path: Path) -> None:
    """Un 1.1 se mide, y la evidencia avisa de que el requisito vino de un default.

    El 1.1 no declaro salidas, asi que no hay nada que le "pertenezca"; pero su
    conjunto efectivo es una sola entrada vertical, y para un conjunto de un
    elemento "¿es miembro?" y "¿es 9:16?" son la misma pregunta. Se mide igual que
    un 1.2, y `declared_by_version` queda como procedencia para que el operador
    distinga "lo dijo el contrato" de "venia del default de la version".
    """
    outcome = check_declared_outputs(
        _context(tmp_path, outputs=None, split_aspect=None, width=1080, height=1920)
    )

    assert outcome.status is CheckStatus.PASS
    assert outcome.evidence["declared_by_version"] == "1.1"
    assert outcome.evidence["satisfied_by"] == "9:16"


def test_b3b_el_1_1_no_declara_aspectos_pero_se_mide(tmp_path: Path) -> None:
    """La pertenencia y la medicion son preguntas distintas, y aqui caen juntas.

    Antes hubo una exencion por version que hacia pasar un 16:9 de una corrida 1.1.
    El detalle de fondo es que para un 1.1 las dos preguntas se reducen a la misma,
    asi que la exencion no protegia nada. Este test deja fijada la medicion;
    `f2` deja fijada la pertenencia con varias salidas, donde si significa algo.
    """
    con_una = check_declared_outputs(
        _context(tmp_path, outputs=None, split_aspect=None, width=1080, height=1920)
    )
    con_dos = check_declared_outputs(
        _context(tmp_path, outputs=_DUAL, split_aspect=None, width=1080, height=1920)
    )

    assert con_una.evidence["declared_aspects"] == ["9:16"]
    assert con_dos.evidence["declared_aspects"] == ["9:16", "16:9"]


# ------------------------------------------------------- lo no medible


def test_c1_sin_probe_es_unsupported_nunca_pass(tmp_path: Path) -> None:
    """Un aspecto que no se puede medir es UNSUPPORTED.

    Aqui hay que distinguir dos cosas que el codigo de hoy mezcla: un artefacto
    MALO y un artefacto que no se pudo MIRAR. Lo no medible nunca se lee como
    conforme.
    """
    outcome = check_split_screen_geometry(_context(tmp_path, with_probe=False))

    assert outcome.status is CheckStatus.UNSUPPORTED


def test_c2_sin_dimensiones_es_unsupported(tmp_path: Path) -> None:
    """Un artefacto sin dimensiones medidas no tiene aspecto."""
    outcome = check_split_screen_geometry(_context(tmp_path, width=None, height=None))

    assert outcome.status is CheckStatus.UNSUPPORTED


def test_c3_sin_pista_de_video_es_fail(tmp_path: Path) -> None:
    """Sin video es FAIL, porque el probe RESPONDIO: es una medicion.

    Es la otra mitad de la distincion, y por eso importa que este test y
    `c1` digan cosas distintas. Un probe que no devuelve nada (c1) es no
    medible; un probe que responde "no hay pista de video" es una medicion con
    veredicto. Devolver UNSUPPORTED en los dos casos daria el mismo diagnostico
    para un archivo corrupto y para una herramienta caida.
    """
    outcome = check_split_screen_geometry(_context(tmp_path, has_video=False))

    assert outcome.status is CheckStatus.FAIL

    pertenencia = check_declared_outputs(
        _context(tmp_path, outputs=_DUAL, split_aspect=None, has_video=False)
    )

    assert pertenencia.status is CheckStatus.FAIL


def test_c3b_alto_cero_no_es_una_division_por_cero(tmp_path: Path) -> None:
    """`height == 0` es UNSUPPORTED, y el gate no revienta.

    `MediaInfo` no declara cota, asi que un cero desde ffprobe llegaria aqui. La
    aritmetica de `_measured_aspect` divide, y un `ZeroDivisionError` no
    envuelto convertia un veredicto fail-closed en un crash. Que el camino
    aritmetico tambien sea fail-closed es parte del contrato del gate.
    """
    outcome = check_split_screen_geometry(_context(tmp_path, height=0))

    assert outcome.status is CheckStatus.UNSUPPORTED
    assert "dimensiones" in str(outcome.evidence["reason"])

    pertenencia = check_declared_outputs(
        _context(tmp_path, outputs=_DUAL, split_aspect=None, height=0)
    )

    assert pertenencia.status is CheckStatus.UNSUPPORTED


def test_c4_sin_lienzo_declarado_falla(tmp_path: Path) -> None:
    """La regla 'layout.geometry' sin lienzo declarado es un FAIL del contrato.

    Aqui si es FAIL y no UNSUPPORTED: la falta esta en el contrato, que es
    legible, y no en la inspeccion, que no ocurrio.
    """
    outcome = check_split_screen_geometry(_context(tmp_path, split_aspect=None))

    assert outcome.status is CheckStatus.FAIL
    assert "layout.geometry" in str(outcome.evidence["reason"])


def test_c5_ninguna_evidence_afirma_cumplimiento_sin_medir(tmp_path: Path) -> None:
    """Guardarrail: ningun veredicto lee como conforme algo que no se midio.

    Recorre las tres formas de no poder medir y los dos validadores. La clave es
    que en ningun caso la evidencia pueda leerse como una atestación: si
    `satisfied_by` aparece, hubo medición.

    `has_video=False` queda fuera de este bucle a proposito: el probe SI
    respondio, asi que es una medicion con veredicto FAIL, no una ausencia de
    medicion. Ese caso tiene su propio test (`c3`), y meterlo aqui volveria a
    fusionar las dos cosas que este PR separa.
    """
    for con_probe, ancho, alto in (
        (False, 1080, 1920),
        (True, None, None),
        (True, 1080, 0),
    ):
        # Sin lienzo, porque un contrato multi-formato no puede declararlo.
        for validar in (check_split_screen_geometry, check_declared_outputs):
            outcome = validar(
                _context(
                    tmp_path,
                    outputs=_DUAL,
                    split_aspect=None,
                    with_probe=con_probe,
                    width=ancho,
                    height=alto,
                )
            )
            etiqueta = (validar.__name__, con_probe, ancho, alto)
            if validar is check_split_screen_geometry:
                # Sin `split_screen` la regla no aplica: eso es FAIL del
                # contrato, no una falta de medicion. Se cubre en `c4`.
                assert outcome.status is CheckStatus.FAIL, etiqueta
            else:
                assert outcome.status is CheckStatus.UNSUPPORTED, etiqueta
                assert "satisfied_by" not in outcome.evidence, etiqueta


# ---------------------------------------------------- la constante, no el contrato


def test_d1_el_enum_es_la_unica_fuente_de_la_razon(tmp_path: Path) -> None:
    """El gate no reescribe la aritmetica del aspecto.

    La razon se lee de `Aspect.ratio` porque el enum es la unica fuente que
    conoce el 9:16 y el 16:9. Si alguien vuelve a escribir 9/16 dentro del gate,
    este test no lo pilla por si solo; lo pilla `test_d2` y la mutacion que
    revierte el gate a la constante.
    """
    vertical = check_split_screen_geometry(_context(tmp_path))
    horizontal = check_split_screen_geometry(
        _context(
            tmp_path,
            outputs=[_HORIZONTAL],
            split_aspect="16:9",
            width=1920,
            height=1080,
        )
    )

    assert vertical.status is CheckStatus.PASS
    assert horizontal.status is CheckStatus.PASS
    assert vertical.evidence["aspect_ratio"] == Aspect.VERTICAL.value
    assert horizontal.evidence["aspect_ratio"] == Aspect.HORIZONTAL.value


def test_d2_el_mensaje_nombra_el_aspecto_declarado(tmp_path: Path) -> None:
    """El mensaje de FAIL tiene que decir QUE se exigia, no un 9:16 fijo.

    Revertir el gate a la constante produce "el lienzo exige aspecto 9:16" sobre
    un contrato que declaro 16:9. Eso no es un mensaje feo: es un diagnostico
    falso, porque afirma que el contrato va contra el 9:16 cuando va contra otra
    cosa. Este test falla con la constante puesta.
    """
    outcome = check_split_screen_geometry(
        _context(
            tmp_path,
            outputs=[_HORIZONTAL],
            split_aspect="16:9",
            width=1920,
            height=1080,
        )
    )

    assert outcome.evidence["declared_aspect"] == "16:9"


def test_d3_el_gate_no_elige_por_el_contrato(tmp_path: Path) -> None:
    """El contrato dice cuales se emiten: el gate no elige el mayoritario."""
    context = _context(tmp_path, outputs=_DUAL, split_aspect=None, width=1080, height=1920)
    outcome = check_declared_outputs(context)

    assert outcome.evidence["declared_aspects"] == ["9:16", "16:9"]
    assert outcome.evidence["measured_aspect"] == "9:16"


@pytest.mark.parametrize(
    ("declarado", "medido", "pasa"),
    [
        (Aspect.VERTICAL, (1080, 1920), True),
        (Aspect.VERTICAL, (1920, 1080), False),
        (Aspect.HORIZONTAL, (1920, 1080), True),
        (Aspect.HORIZONTAL, (1080, 1920), False),
    ],
)
def test_e1_matriz_declarado_contra_medido(
    tmp_path: Path,
    declarado: Aspect,
    medido: tuple[int, int],
    *,
    pasa: bool,
) -> None:
    """La matriz de la pertenencia, para no razonar de memoria.

    El lienzo se declara del mismo aspecto que la salida, asi que aqui lo que
    se mide es si el artefacto MEDIDO pertenece a las salidas DECLARADAS, que es
    la regla nueva.
    """
    ancho, alto = medido
    salida: dict[str, object] = {"aspect": declarado.value, "index": 1}
    outcome = check_declared_outputs(
        _context(
            tmp_path,
            outputs=[salida],
            split_aspect=None,
            width=ancho,
            height=alto,
        )
    )

    if pasa:
        assert outcome.status is CheckStatus.PASS
        assert outcome.evidence["satisfied_by"] == declarado.value
    else:
        assert outcome.status is CheckStatus.FAIL
        assert outcome.evidence["measured_aspect"] != declarado.value


# ------------------------------------------------- la clase, cerrada


def _sondeo_una_regla(tmp_path: Path, regla: str, media: MediaInfo | None) -> CheckOutcome:
    """Provoca una regla con un artefacto concrete y devuelve su veredicto.

    Args:
        tmp_path: Directorio de trabajo del asset registry.
        regla: El rule_id que se quiere provocar.
        media: Los metadatos del artefacto, o `None` si el probe fallo.

    Returns:
        El veredicto del validador registrado para esa regla.
    """
    contrato = make_contract(
        hard=[regla],
        audio_rule="own_clip",
        min_s=1,
        max_s=600,
        required_mentions=(),
        required_hashtags=(),
        split_screen=(
            {"top_source": "clip-top", "bottom_source": "clip-bottom"}
            if regla == "layout.geometry"
            else None
        ),
    )
    contexto = GateContext(
        contract=contrato,
        rules=contrato.platforms[next(iter(contrato.platforms))],
        piece=make_piece(tmp_path / "a.mp4"),
        artifact_sha256="a" * 64,
        media=media,
        assets=AssetRegistry(tmp_path),
    )
    return DEFAULT_VALIDATORS[regla](contexto)


# Las reglas cuyo veredicto depende de haber MIRADO el artefacto. Un validador de
# esta clase no puede devolver FAIL cuando el probe noRespondio, porque entonces
# le esta diciendo al operador que su contenido esta mal cuando lo que fallo fue
# la herramienta.
#
# Esta lista NO es una puerta estructural y no pretende serlo. Se declaro como
# una, y no lo era: las uniones `| {"artifact.aspect"}` eran no-ops porque la
# entrada ya estaba, y el test que "verificaba" que la lista cubriese el catalogo
# era una identidad que se cumple siempre. Un validador nuevo que conflara los dos
# casos pasaba las dos.
#
# Lo que si funciona, y es lo unico que se afirma aqui, es que la lista no se
# desincroniza con el catalogo por la via de inventar nombres: `test_f3` lo
# comprueba, y esa comprobacion puede fallar de verdad.
#
# Lo que NO se automatiza, y conviene decirlo para que nadie lo busque: que un
# validador NUEVO que dependa del artefacto se apunte aqui. Se puede escribir esa
# comprobacion, y es tentadora, pero necesita saber si cada validador depende del
# artefacto, y esa informacion no existe en ningun sitio: no hay marcador, ni tipo,
# ni anotacion. Una version de esa puerta tendria que hardcodear la misma lista
# que ya hay, mas un diccionario de "estas si dependen", y las dos cosas se
# desincronizan igual. Un test que depende de una tabla que el test mismo mantiene
# no es una puerta: es una assertion con pasos extra.
#
# La decision de si un validador nuevo es de esta clase o de la otra sigue siendo
# del que lo escribe, y por eso no se disfraza de automatismo. Lo que si hay es el
# inventario (`f1`), que dice la verdad sobre las reglas de hoy.
_REGLAS_DEL_ARTIFACTO: frozenset[str] = frozenset(
    {
        "artifact.aspect",
        "artifact.integrity",
        "artifact.video_stream",
        "assets.required",
        "audio.present",
        "audio.silence",
        "duration.max",
        "duration.min",
        "layout.geometry",
        "watermark.full_video",
        "watermark.present",
    }
)


def test_f1_inventario_de_validadores_que_conflan(tmp_path: Path) -> None:
    """Ningun validador del gate puede decir FAIL sin haber mirado el artefacto.

    Un validador que devuelve FAIL con `media is None` le esta diciendo al
    operador que su contenido esta mal cuando en realidad no pudo mirarlo, y lo
    manda a depurar lo que no esta roto: eso es un problema de herramienta, no de
    contenido.

    Lo que fija es el estado actual de las once reglas que dependen del artefacto.
    Si una de ellas pasa a conflacionar, este test lo dice. Si anaden un validador
    nuevo y no lo anaden a la lista, esto no lo vera: para eso esta `f3`.
    """
    conflatan = [
        regla
        for regla in sorted(_REGLAS_DEL_ARTIFACTO)
        if _sondeo_una_regla(tmp_path, regla, None).status is CheckStatus.FAIL
    ]

    assert conflatan == [], f"devuelven FAIL sin haber medido: {conflatan}"


def test_f3_la_lista_y_el_catalogo_no_se_desincronizan() -> None:
    """`_REGLAS_DEL_ARTIFACTO` tiene que seguir existiendo en el catalogo real.

    Esta es la asercion que si puede fallar, a diferencia de la que sustituye. Si
    `_REGLAS_DEL_ARTIFACTO` gana un nombre que no esta en `DEFAULT_VALIDATORS`, el
    inventario de `f1` estaria sondeando un validador inexistente y `f1` pasaria
    sin comprobar nada. Falla en esa direccion.
    """
    declaradas = _REGLAS_DEL_ARTIFACTO
    todas = set(DEFAULT_VALIDATORS)

    assert declaradas <= todas, f"en la lista y sin validador: {sorted(declaradas - todas)}"


def test_f1c_el_artefacto_sin_video_sigue_siendo_una_medicion(tmp_path: Path) -> None:
    """El caso hermano: ffprobe respondio, asi que FAIL es lo correcto.

    Confundir "no hay pista de video" (una medicion) con "no pude medir" (una
    herramienta caida) seria el error opuesto al que esta PR arregla. Este test
    fija que los dos casos siguen distinguidos.
    """
    sin_video = MediaInfo(
        width=None,
        height=None,
        has_video=False,
        duration_s=None,
        has_audio=False,
        format_name="",
    )

    sin_probe = _sondeo_una_regla(tmp_path, "artifact.video_stream", None)
    con_video_ausente = _sondeo_una_regla(tmp_path, "artifact.video_stream", sin_video)

    assert sin_probe.status is CheckStatus.UNSUPPORTED
    assert con_video_ausente.status is CheckStatus.FAIL


def test_f2_un_1_1_sin_lienzo_tambien_se_mide(tmp_path: Path) -> None:
    """El agujero: un 1.1 sin split NO se saltaba, y un 16:9 pasaba como PASS.

    Este test es el motivo por el que el agujero no vuelve. Antes
    `check_declared_outputs` hacia corto-circuito en un 1.1 y devolvia PASS sin
    medir nada, de modo que un artefacto 16:9 de una corrida 1.1 quedaba
    certificado como conforme. El render de un 1.1 produce 9:16, asi que ese
    artefacto era una corrida mal formada y el gate no lo detectaba.

    No hay exencion por version: un 1.1 tiene conjunto efectivo de una sola
    entrada vertical, y para un conjunto de un elemento "pertenece a sus
    salidas" y "es 9:16" son la misma pregunta. La exencion no protegia nada.
    """
    con_horizontal = check_declared_outputs(
        _context(tmp_path, outputs=None, split_aspect=None, width=1920, height=1080)
    )

    assert con_horizontal.status is CheckStatus.FAIL
    assert con_horizontal.evidence["measured_aspect"] == "16:9"
    assert con_horizontal.evidence["declared_aspects"] == ["9:16"]


def test_f2b_un_1_1_con_9_16_pasa(tmp_path: Path) -> None:
    """La mitad positiva: medir al 1.1 no significa rechazarlo.

    Un 1.1 que produce el 9:16 que siempre produjo tiene que seguir pasando. Si
    esto fallara, la "correccion" habria roto las corridas normales en vez de
    cerrar el agujero.
    """
    outcome = check_declared_outputs(
        _context(tmp_path, outputs=None, split_aspect=None, width=1080, height=1920)
    )

    assert outcome.status is CheckStatus.PASS
    assert outcome.evidence["satisfied_by"] == "9:16"


def test_f2c_la_evidencia_dice_de_donde_venia_el_aspecto(tmp_path: Path) -> None:
    """`declared_by_version` es procedencia, no exencion.

    El operador tiene que poder distinguir "el contrato dijo 9:16" de "9:16 vino
    de un default de version". Por eso el campo esta en la evidencia del 1.1, y
    por eso el 1.1 tambien tiene `declared_aspects`: se mide igual, pero se dice
    de donde salio el requisito.
    """
    outcome = check_declared_outputs(
        _context(tmp_path, outputs=None, split_aspect=None, width=1080, height=1920)
    )

    assert outcome.evidence["declared_by_version"] == "1.1"
    assert outcome.evidence["contract_schema_version"] == "1.1"

    declarado = check_declared_outputs(
        _context(tmp_path, outputs=[_VERTICAL], split_aspect=None, width=1080, height=1920)
    )

    assert "declared_by_version" not in declarado.evidence
    assert declarado.evidence["contract_schema_version"] == "1.2"


def test_f2d_un_1_2_horizontal_con_artefacto_vertical_falla(tmp_path: Path) -> None:
    """La comprobacion tambien aplica cuando el contrato declara de verdad.

    Es el caso que la exencion tapaba por el lado del 1.2, y fija que la
    pertenencia no es un TRIVIAL que siempre pasa.
    """
    outcome = check_declared_outputs(
        _context(
            tmp_path,
            outputs=[_HORIZONTAL],
            split_aspect=None,
            width=1080,
            height=1920,
        )
    )

    assert outcome.status is CheckStatus.FAIL
    assert outcome.evidence["declared_aspects"] == ["16:9"]
    assert outcome.evidence["measured_aspect"] == "9:16"
    assert "16:9" in str(outcome.evidence["reason"])
    assert "9:16" in str(outcome.evidence["reason"])


def test_f2e_multi_salida_registra_cual_satisficio(tmp_path: Path) -> None:
    """Con varias salidas, la evidencia tiene que decir cual cumplio.

    "Pasa" a secas no basta cuando el contrato permitia mas de un formato: hay
    que poder leer que el artefacto es el 9:16 y no el 16:9.
    """
    outcome = check_declared_outputs(
        _context(
            tmp_path,
            outputs=_DUAL,
            split_aspect=None,
            width=1080,
            height=1920,
        )
    )

    assert outcome.status is CheckStatus.PASS
    assert outcome.evidence["satisfied_by"] == "9:16"
    assert outcome.evidence["declared_aspects"] == ["9:16", "16:9"]


def test_f2f_el_1_1_no_se_escapa_por_la_otra_puerta(tmp_path: Path) -> None:
    """Un 1.1 con split tampoco se escapa: los dos validadores miden.

    Antes solo el de split-screen media un 1.1 (usando `split.aspect`) y el de
    salidas no. Esa asimetria no era un principio, era donde habia puesto el
    corto-circuito. Ahora los dos miden, asi que el 1.1 no tiene una puerta
    trasera por la que colarse.
    """
    contexto = _context(
        tmp_path,
        outputs=None,
        split_aspect="9:16",
        width=1920,
        height=1080,
    )

    assert check_split_screen_geometry(contexto).status is CheckStatus.FAIL
    assert check_declared_outputs(contexto).status is CheckStatus.FAIL


def test_f2g_la_complementa_se_sostiene_en_el_1_1(tmp_path: Path) -> None:
    """FAIL y UNSUPPORTED siguen distinguidos en el 1.1, en las dos direcciones.

    Sin probe, UNSUPPORTED aunque sea un 1.1. Medido y mal, FAIL aunque sea un
    1.1. Si estas dos se fusionaran, el 1.1 seria el unico contrato donde "no
    pude verificar" y "esta mal" dicen lo mismo.
    """
    sin_probe = check_declared_outputs(
        _context(tmp_path, outputs=None, split_aspect=None, with_probe=False)
    )
    medido_y_mal = check_declared_outputs(
        _context(tmp_path, outputs=None, split_aspect=None, width=1920, height=1080)
    )
    medido_y_bien = check_declared_outputs(
        _context(tmp_path, outputs=None, split_aspect=None, width=1080, height=1920)
    )

    assert sin_probe.status is CheckStatus.UNSUPPORTED
    assert medido_y_mal.status is CheckStatus.FAIL
    assert medido_y_bien.status is CheckStatus.PASS
