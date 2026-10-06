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
from kliptych.encoding import RenderConfig
from kliptych.gate import Gate
from kliptych.gate.checks import (
    DEFAULT_VALIDATORS,
    CheckOutcome,
    GateContext,
    check_declared_outputs,
    check_split_screen_geometry,
    matches_declared_aspect,
)
from kliptych.gate.models import CheckStatus, MediaInfo
from kliptych.reframe import crop_size
from tests.support import FakeProbe, make_contract, make_media, make_piece

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
    render: RenderConfig | None = None,
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
        render: Config de render EFECTIVA de la corrida; por omision la de
            `RenderConfig()`, que es lo correcto cuando nadie la sobreescribio.

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
        render=RenderConfig() if render is None else render,
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


# ----------------------------- la geometria que el RENDERER produce de verdad


@pytest.mark.parametrize(
    ("ancho", "alto"),
    [
        (608, 1080),
        (404, 720),
        (1068, 1900),
    ],
)
def test_g1_regresion_artefacto_del_reframe_pasa(tmp_path: Path, ancho: int, alto: int) -> None:
    """REGRESION: la salida del reframe tiene que pasar el gate.

    Este es el test que faltaba y por el que la suite estaba verde mientras el
    producto roto. Todas las pruebas de aspecto usaban `FakeProbe(1080, 1920)`,
    unas dimensiones que el renderer NO PUEDE emitir: el reframe cuantiza a
    entero par, `width = _even(round(height * 9/16))`, y para un alto de 1080 eso
    da 608x1080, no 1080x1920. Comparar a 1e-9 no era mas riguroso, era exigir
    un 9:16 exacto que esta aritmetica no produce.

    Medido con `crop_size` del propio repo:
        1920x1080 -> 608x1080    (0,08 % de 9:16)
        1280x720  -> 404x720     (0,25 % de 9:16)
        1080x1900 -> 1068x1900   (0,07 % de 9:16)

    Si alguien vuelve a endurecer la comparacion, este test falla y dice por que.
    """
    outcome = check_declared_outputs(
        _context(
            tmp_path,
            outputs=[_VERTICAL],
            split_aspect=None,
            width=ancho,
            height=alto,
        )
    )

    assert outcome.status is CheckStatus.PASS, f"{ancho}x{alto}"
    assert outcome.evidence["measured_ratio"] == pytest.approx(ancho / alto, abs=1e-6)


def test_g2_el_artefacto_del_reframe_lo_emite_el_reframe() -> None:
    """Las dimensiones de `g1` no estan inventadas: salen de `crop_size`.

    Si el reframe cambiara su cuantizacion, este test se moveria con el y `g1`
    dejaria de proteger la geometria real. Por eso se leen del codigo y no se
    escriben a mano en el parametrize.
    """
    for origen in ((1920, 1080), (1280, 720)):
        ancho, alto = crop_size(*origen)
        assert matches_declared_aspect(
            ancho, alto, Aspect.VERTICAL, tolerancia=RenderConfig().aspect_tolerance
        ), f"{origen} -> {ancho}x{alto}"


@pytest.mark.parametrize(
    ("ancho", "alto"),
    [
        (1920, 1080),
        (1280, 720),
    ],
)
def test_g3_horizontal_genuino_falla(tmp_path: Path, ancho: int, alto: int) -> None:
    """El otro lado: un 16:9 de verdad NO pasa un contrato vertical.

    Sin esto, "aceptar lo que el renderer emite" seria indistinguible de "aceptar
    cualquier cosa". La tolerancia del render es 5 % relativo, y un 16:9 esta a
    un 78 % de 9:16: no hay ambiguedad posible entre los dos casos.
    """
    outcome = check_declared_outputs(
        _context(
            tmp_path,
            outputs=[_VERTICAL],
            split_aspect=None,
            width=ancho,
            height=alto,
        )
    )

    assert outcome.status is CheckStatus.FAIL
    assert f"{ancho}x{alto}" in str(outcome.evidence["reason"])


def test_g4_un_vertical_reframed_no_sirve_para_un_contrato_horizontal(
    tmp_path: Path,
) -> None:
    """La tolerancia es respecto al ASPECTO DECLARADO, no una regla global.

    608x1080 pasa un contrato vertical. El mismo artefacto, con un contrato que
    declara 16:9, tiene que fallar: si no, la tolerancia se habria convertido en
    "vertical dentro de +-5 %, me da igual lo que dijeras".
    """
    outcome = check_declared_outputs(
        _context(
            tmp_path,
            outputs=[_HORIZONTAL],
            split_aspect=None,
            width=608,
            height=1080,
        )
    )

    assert outcome.status is CheckStatus.FAIL


def test_g5_el_lienzo_tambien_honra_la_predicada_del_render(tmp_path: Path) -> None:
    """El validador del lienzo usa la misma predicada, no la suya.

    El lienzo declara su geometria exacta (`width`/`height`), asi que aqui la
    comparacion de dimensiones ya coraje. La de aspecto se mantiene, y comparte
    predicada con el render: si las dos se separaran, el mismo artefacto podria
    ser conforme para un validador y no para el otro.
    """
    outcome = check_split_screen_geometry(
        _context(tmp_path, outputs=[_VERTICAL], split_aspect="9:16", width=608, height=1080)
    )

    # Las dimensiones no coinciden con el lienzo declarado (1080x1920), asi que
    # el guard de dimensiones falla antes. Lo que se comprueba aqui es que la
    # comparacion de aspecto reconoce 608x1080 como 9:16, que es lo que estaba
    # en litigio.
    assert matches_declared_aspect(
        608, 1080, Aspect.VERTICAL, tolerancia=RenderConfig().aspect_tolerance
    )
    assert outcome.status is CheckStatus.FAIL
    assert "dimensiones" in str(outcome.evidence["reason"])


def _con_tolerancia(tmp_path: Path, tolerancia: float) -> GateContext:
    """Un contexto cuya corrida renderizo con la tolerancia dada.

    Args:
        tmp_path: Directorio de trabajo.
        tolerancia: La tolerancia RELATIVA que uso ESA corrida.

    Returns:
        El contexto, con `render` ajustado.
    """
    return _context(
        tmp_path,
        outputs=[_VERTICAL],
        split_aspect=None,
        width=608,
        height=1080,
        render=RenderConfig(aspect_tolerance=tolerancia),
    )


def test_g6_el_gate_juzga_con_la_tolerancia_de_la_corrida(tmp_path: Path) -> None:
    """El gate lee la tolerancia del RENDER DE LA CORRIDA, no un default.

    Este test REEMPLAZA a uno anterior que no servia para nada: sombreaba la
    clase `RenderConfig` con una tolerancia de 1e-6 y comprobaba que 608x1080
    fallaba, pero un 1e-6 HARDCODEADO tambien cumple las dos aserciones. No
    distinguia leer de copiar, que es justo lo que decia comprobar.

    Aqui los dos casos son incompatibles con cualquier constante fija:
      - una corrida a 0,0001 tiene que RECHAZAR 608x1080: se desvía un 0,082 %
        y la banda de 0,0001 es de 0,006 %,
      - una corrida a 0,05 tiene que ACEPTARlo: la banda es de 2,8 %.
    Ningun numero hardcodeado puede satisfacer los dos, asi que si pasan, el gate
    esta leyendo la declaracion de la corrida.
    """
    estricto = _con_tolerancia(tmp_path, 0.0001)
    holgado = _con_tolerancia(tmp_path, 0.05)

    assert check_declared_outputs(estricto).status is CheckStatus.FAIL
    assert check_declared_outputs(holgado).status is CheckStatus.PASS


def test_g7_la_tolerancia_del_render_alcanza_lo_que_reframe_emite() -> None:
    """Recorre los altos que el pipeline puede producir y no encuentra ninguno rechazado.

    Una version anterior de este test barria SOLO `crop_size(1920, alto)` con
        `alto` en 320..3840 y concluyo que el gate no rechaza nada. No se podia ver:
        un unico ancho fijo con esas alturas no toca la region que falla, y un
        barrido prueba SU rango, no el espacio. Este barre las DOS dimensiones.

        El rango sale de la ARITMETICA del productor, no de un rango comodo. `_even`
        redondea al par, y en un recorte pequeno el error de cuantizacion (hasta 2 px
        de ancho) supera el 5 % relativo. El limite medido de la banda es un lado
        menor de 24 px: por encima no hay ni un rechazo en toda la rejilla.

        OJO con lo que este test NO dice: que el gate RECHAZA los recortes por debajo
        de ese minimo es correcto, no un defecto. El defecto es del PRODUCTOR, que
        deberia negarse a construir un recorte que su propio gate va a rechazar. Va
        al issue #84, con su recorrido end-to-end; aqui se acota la frontera.
    """
    tolerancia = RenderConfig().aspect_tolerance

    por_encima_del_limite: list[str] = []
    por_debajo: list[str] = []
    for ancho_origen in range(2, 400, 2):
        for alto_origen in range(2, 400, 2):
            ancho, alto = crop_size(ancho_origen, alto_origen)
            if matches_declared_aspect(ancho, alto, Aspect.VERTICAL, tolerancia=tolerancia):
                continue
            caso = f"origen {ancho_origen}x{alto_origen} -> {ancho}x{alto}"
            if min(ancho, alto) >= 24:
                por_encima_del_limite.append(caso)
            else:
                por_debajo.append(caso)

    assert por_encima_del_limite == [], (
        f"el gate rechaza geometria de origenes utilizables que el reframe emite: "
        f"{por_encima_del_limite[:8]}"
    )
    # El lado contrario: la region por debajo del limite TIENE que rechazar, o
    # este test solo estaria comprobando que el gate es permisivo.
    assert por_debajo, "esperaba recortes rechazados por debajo del lado menor de 24"


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


# Las reglas cuyo veredicto depende de haber MIRADO el artefacto, y entre las que
# este archivo SI puede observar con el sondeo de `f1` (que activa el contrato minimo).
# No incluyen las que necesitan que la campana exija watermark o silencio: esas
# se miden en #83, con el PNG registrado en el registry, y aqui solo se rozarian.
# `assets.required` queda fuera por otra razon: `check_required_assets` no lee
# NUNCA `context.media` (sus FAIL vienen del registry de assets), asi que no
# puede conflacionar nada con la medicion.
#
# Esta lista NO es una puerta estructural y no pretende serlo. Se declaro como
# una, y no lo era: las uniones `| {"artifact.aspect"}` eran no-ops porque la
# entrada ya estaba, y el test que "verificaba" que la lista cubriese el catalogo
# era una identidad que se cumple siempre. Un validador nuevo que conflara los dos
# casos pasaba las dos. Se borro en vez de reemplazarlo por una version mas
# lista de la misma nada: decidir si un validador nuevo depende del artefacto
# necesita informacion que no existe en el codigo (no hay marcador, ni tipo, ni
# anotacion), asi que cualquier puerta de ese tipo mantiene su propia tabla y se
# desincroniza igual.
_REGLAS_DEL_ARTIFACTO: frozenset[str] = frozenset(
    {
        "artifact.aspect",
        "artifact.integrity",
        "artifact.video_stream",
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
    """Ninguna de las reglas que este test puede ACTIVAR conflana los dos fallos.

    Son dos salidas distintas y no deben fusionarse: `FAIL` es "el artefacto
    esta mal" y `UNSUPPORTED` es "no pude verificarlo". Un validador que devuelve
    FAIL con `media is None` manda al operador a depurar su contenido cuando lo
    que fallo fue la herramienta.

    ALCANCE REAL, y hay que leerlo antes que el titulo: el sondeo activa un
    contrato MINIMO, asi que solo observa las reglas que se disparan con el. Las
    que necesitan que la campana exija watermark o silencio se miden en #83, con
    el PNG registrado, y aqui no se tocan. Para esas, este test no dice nada.

    `assets.required` queda fuera por otra razon: `check_required_assets` no lee
    NUNCA `context.media` (sus FAIL vienen del registry de assets), asi que no
    puede conflacionar nada con la medicion.
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
    assert "1080x1920" in str(outcome.evidence["reason"])


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


def test_h1_el_motor_entrega_la_tolerancia_de_la_corrida(tmp_path: Path) -> None:
    """El `RenderConfig` del motor llega al veredicto, no solo al helper.

    `test_g6` arma el `GateContext` a mano, asi que deja abierta la mitad
    importante: que la tolerancia sobreviva el camino de produccion. Aqui se
    evalua con `Gate.run`, que es como lo llama el pipeline, y la unica variable
    es la tolerancia del constructor. Si el motor se olvidara de `self._render` y
    fabricase un `RenderConfig()` nuevo, aqui pasaria lo mismo y el veredicto
    seria PASS donde tiene que ser FAIL.
    """
    artefacto = tmp_path / "pieza.mp4"
    _ = artefacto.write_bytes(b"video")
    contrato = make_contract(
        hard=["artifact.aspect"],
        audio_rule="any",
        min_s=None,
        max_s=None,
        required_mentions=(),
        required_hashtags=(),
        schema_version="1.2",
        outputs=[_VERTICAL],
    )
    # 608x1080 es lo que el reframe emite de verdad para un alto de 1080.
    probe = FakeProbe(info=make_media(width=608, height=1080))
    registro = AssetRegistry(tmp_path)

    estricto = Gate(probe, render=RenderConfig(aspect_tolerance=0.0001)).run(
        contract=contrato,
        piece=make_piece(artefacto),
        assets=registro,
    )
    holgado = Gate(probe, render=RenderConfig(aspect_tolerance=0.05)).run(
        contract=contrato,
        piece=make_piece(artefacto),
        assets=registro,
    )

    estricto_asp = next(c for c in estricto.checks if c.id == "artifact.aspect")
    holgado_asp = next(c for c in holgado.checks if c.id == "artifact.aspect")
    assert estricto_asp.status is CheckStatus.FAIL
    assert holgado_asp.status is CheckStatus.PASS


# NOTA sobre el catalogo de reglas: la invariante "toda regla con validador esta
# en KNOWN_VALIDATOR_RULES y viceversa" YA la cubria
# `tests/test_gate_checks.py::test_known_validator_rules_matches_default_validators`
# antes de que existiera este archivo. Se intento duplicar aqui y se elimino: un
# segundo test que dice lo mismo no cierra una clase, solo hace que el conteo de
# tests parezca mejor de lo que es.
