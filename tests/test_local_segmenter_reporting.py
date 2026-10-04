"""Lo que el segmentador DICE que dejo fuera, no solo lo que entrega.

`SpeechSegmentResult` existe porque una tupla no puede llevar dos hechos, y con
el techo puesto hacen falta dos: 12 de 19. Informar solo de lo entregado
convierte una operacion PARCIAL en un informe que parece COMPLETO.

Estas pruebas son afirmaciones sobre RECUENTOS. La igualdad del tipo compara
todos sus campos, asi que un recuento equivocado no puede pasar inadvertido
contra otro resultado con el mismo recuento erroneo.
"""

from collections.abc import Sequence
from typing import TYPE_CHECKING, cast

import pytest

from kliptych import orchestrator as orch
from kliptych import segmenter
from kliptych.segmenter import SpeechSegmentResult
from kliptych.transcribe import Word

if TYPE_CHECKING:
    from collections.abc import Callable


def _rationale(resultado: SpeechSegmentResult) -> str:
    """Invoca la justificacion privada del orquestador, con su nombre resuelto.

    Args:
        resultado: El resultado que el orquestador registraria.

    Returns:
        La cadena que va al manifest.
    """
    nombre = "_rationale_segmentador"
    rationale = cast("Callable[[SpeechSegmentResult], str]", getattr(orch, nombre))
    return rationale(resultado)


def _resultado_de(palabras: Sequence[Word], **opciones: float | None) -> SpeechSegmentResult:
    """Llama al segmentador y devuelve su resultado tipado.

    Args:
        palabras: Las palabras de entrada.
        **opciones: `min_s` y `max_s` del contrato.

    Returns:
        El `SpeechSegmentResult` que devuelve el segmentador.
    """
    return segmenter.segment_by_speech_rate(palabras, **opciones)


def _span_de_palabras(n_palabras: int, *, paso: float = 0.7, inicio: float = 0.0) -> list[Word]:
    """Un tramo de `n_palabras` palabras, sin huecos.

    Args:
        n_palabras: Cuantas palabras.
        paso: Separacion entre inicios.
        inicio: Comienzo del tramo.

    Returns:
        Las palabras del tramo.
    """
    return [
        Word(
            start_s=inicio + i * paso,
            end_s=inicio + i * paso + paso * 0.8,
            text="w",
            confidence=0.9,
            token_id=i,
        )
        for i in range(n_palabras)
    ]


def _varios_tramos(n: int, *, palabras_por_tramo: int = 30, paso: float = 0.7) -> list[Word]:
    """N tramos que pasan los dos suelos, separados por un hueco grande.

    Args:
        n: Cuantos tramos.
        palabras_por_tramo: Palabras de cada tramo.
        paso: Separacion entre palabras.

    Returns:
        Todas las palabras.
    """
    palabras: list[Word] = []
    t = 0.0
    token = 0
    for _ in range(n):
        for _ in range(palabras_por_tramo):
            palabras.append(
                Word(start_s=t, end_s=t + paso * 0.8, text="w", confidence=0.9, token_id=token)
            )
            token += 1
            t += paso
        t += segmenter.MAX_GAP_S + 1.0
    return palabras


# --------------------------------------------------------------------------
# T1 - el techo muerde: qualifying es mayor que lo entregado
# --------------------------------------------------------------------------


def test_t1_qualifying_supera_a_lo_entregado_cuando_el_techo_muerde() -> None:
    """Con 19 tramos que califican se entregan 12: el techo recorta de verdad.

    Se afirma el numero construido, no el contado: si el segmentador perdiera un
    tramo por el camino, `qualifying` ya no seria 19 y la asercion lo diria.
    """
    construidos = segmenter.MAX_MOMENTS + 7
    resultado = _resultado_de(_varios_tramos(construidos))

    assert resultado.qualifying == construidos, resultado.qualifying
    assert len(resultado.moments) == segmenter.MAX_MOMENTS
    assert resultado.qualifying > len(resultado.moments)


# --------------------------------------------------------------------------
# T2 - el techo no muerde: qualifying es exactamente lo entregado
# --------------------------------------------------------------------------


def test_t2_qualifying_iguala_a_lo_entregado_cuando_el_techo_no_muerde() -> None:
    """Con 5 tramos que califican se entregan 5 y no se rellena hasta 12."""
    construidos = segmenter.MAX_MOMENTS - 7
    resultado = _resultado_de(_varios_tramos(construidos))

    assert resultado.qualifying == construidos
    assert resultado.qualifying == len(resultado.moments)


# --------------------------------------------------------------------------
# T3 - los dos suelos, por separado y juntos
# --------------------------------------------------------------------------


def test_t3_descartados_por_suelo_cuenta_ambos_fallos() -> None:
    """3 que pasan, 3 que mueren por palabras y 3 por duracion: 6 descartados.

    Se construyen los dos fallos por separado para que el recuento no pueda
    acertar por suma afortunada: si `discarded_by_floor` fuera simply el total
    de tramos crudos menos los buenos, daria 6 igual, asi que ademas se afirma
    que un fallo por palabras y un fallo por duracion cuentan lo mismo.
    """
    palabras: list[Word] = []
    token = 0
    t = 0.0

    def anadir(tramo: Sequence[Word]) -> None:
        nonlocal token, t
        for w in tramo:
            palabras.append(
                Word(
                    start_s=w.start_s + t,
                    end_s=w.end_s + t,
                    text=w.text,
                    confidence=0.9,
                    token_id=token,
                )
            )
            token += 1
        t += tramo[-1].end_s - tramo[0].start_s + segmenter.MAX_GAP_S + 1.0

    for _ in range(3):
        anadir(_span_de_palabras(30))  # pasa los dos suelos
    for _ in range(3):
        anadir(_span_de_palabras(2, paso=1.5))  # falla MIN_WORDS, dura 2,3 s
    for _ in range(3):
        anadir(_span_de_palabras(8, paso=0.2))  # pasa palabras, dura 1,4 s

    resultado = _resultado_de(palabras)

    assert resultado.discarded_by_floor == 6, resultado.discarded_by_floor
    assert resultado.qualifying == 3, resultado.qualifying


def test_t3b_un_fallo_por_palabras_cuenta_igual_que_uno_por_duracion() -> None:
    """Los dos suelos se cuentan por igual, y ninguno se disimula.

    Un caso con un solo fallo por palabras y otro con un solo fallo por duracion
    tienen que reportar UN descartado cada uno. Asi se ve que el recuento no
    distingue solo un tipo de fallo.

    Los tramos que deben fallar llevan un `inicio` explicito y POSTERIOR al de
    los buenos. Un tramo en `inicio=0.0` se solapa con lo que haya antes, se
    fusiona con el, y el resultado es un tramo que PASA: el fixture no construye
    el caso que dice construir. Por eso el `inicio` va aqui y no por defecto.
    """
    buenos = _varios_tramos(2)
    fin_de_los_buenos = buenos[-1].end_s + segmenter.MAX_GAP_S + 1.0

    solo_palabras = buenos + _span_de_palabras(2, paso=1.5, inicio=fin_de_los_buenos)
    solo_palabras += _span_de_palabras(30, inicio=fin_de_los_buenos + 20.0)
    r1 = _resultado_de(solo_palabras)
    assert r1.discarded_by_floor == 1, r1.discarded_by_floor

    solo_duracion = buenos + _span_de_palabras(8, paso=0.2, inicio=fin_de_los_buenos)
    solo_duracion += _span_de_palabras(30, inicio=fin_de_los_buenos + 20.0)
    r2 = _resultado_de(solo_duracion)
    assert r2.discarded_by_floor == 1, r2.discarded_by_floor


def test_t3c_los_dos_suelos_suman_cuando_ocurren_a_la_vez() -> None:
    """Un fallo por palabras y otro por duracion, en la misma entrada, cuentan 2.

    Es el caso que faltaba: T3b prueba que cada fallo cuenta por separado, y esto
    prueba que los dos cuentan JUNTOS. Los tramos que fallan van despues de los
    buenos por el mismo motivo que en T3b: en `inicio=0.0` se fusionarian con el
    primer tramo bueno y pasarian los dos.
    """
    buenos = _varios_tramos(3)
    fin = buenos[-1].end_s + segmenter.MAX_GAP_S + 1.0

    palabras = buenos + _span_de_palabras(2, paso=1.5, inicio=fin)
    palabras += _span_de_palabras(8, paso=0.2, inicio=fin + 20.0)
    palabras += _span_de_palabras(30, inicio=fin + 40.0)

    resultado = _resultado_de(palabras)

    assert resultado.discarded_by_floor == 2, resultado.discarded_by_floor
    assert resultado.qualifying == 4, resultado.qualifying


# --------------------------------------------------------------------------
# T4 - la particion por tope
# --------------------------------------------------------------------------


def test_t4_la_particion_por_tope_informa_de_su_conteo() -> None:
    """Una fila de 150 palabras se parte, y `split` lo cuenta como el implementation cuenta.

    OJO, esta es una prueba de la REALIDAD, no de una especificacion: el
    docstring de `split` se contradice a si mismo. Su primera linea dice "cuantas
    PARTES produjo la particion por tope", que se leeria 2; la segunda dice
    "trozos resultantes POR ENCIMA de los originales", que es 1. La
    implementacion cuenta `len(partidos) - len(crudos)`, o sea 1.

    Aqui se fija lo que el codigo HACE, y se deja constancia de que la frase
    "partes" del docstring no significa "partes". Si alguien cree que deberia
    ser 2, el sitio para discutirlo es el docstring, no esta prueba.
    """
    palabras = _span_de_palabras(150)
    assert palabras[-1].end_s - palabras[0].start_s > segmenter.MAX_CLIP_S

    resultado = _resultado_de(palabras)

    assert len(resultado.moments) == 2, len(resultado.moments)
    # La implementacion cuenta los trozos POR ENCIMA de los tramos originales.
    assert resultado.split == 1, resultado.split


# --------------------------------------------------------------------------
# T5 - lo que el operador lee de verdad
# --------------------------------------------------------------------------


def test_t5_la_justificacion_dice_cuantos_se_descartaron() -> None:
    """El TEXTO que se registra lleva las cuentas, no solo el objeto.

    Afirmar el objeto de resultado no dice nada de lo que lee el operador: lo que
    va al manifest es la cadena. Si esa cadena omite el recuento, un operador ve
    "12 tramo(s)" y no tiene como saber que faltaron 7.
    """
    resultado = SpeechSegmentResult(moments=(), qualifying=19, discarded_by_floor=6, split=2)
    texto = _rationale(resultado)

    assert f"calificados: {resultado.qualifying}" in texto, texto
    assert "techo" in texto, texto
    assert "suelos" in texto, texto
    assert "particiones" in texto, texto


def test_t5b_la_justificacion_real_refleja_el_recuento_real() -> None:
    """La cadena que se registra lleva las cuentas DEL resultado que se entrego."""
    resultado = segmenter.segment_by_speech_rate(_varios_tramos(19))
    texto = _rationale(resultado)

    assert f"entregados: {len(resultado.moments)}" in texto, texto
    assert f"calificados: {resultado.qualifying}" in texto, texto
    assert f"{resultado.discarded_by_floor}" in texto, texto


def test_t5c_la_igualdad_del_tipo_no_acepta_una_tupla_de_momentos() -> None:
    """Un resultado NO es igual a la tupla de sus momentos.

    Si lo fuera, una asercion podria pasar con `qualifying`,
    `discarded_by_floor` y `split` todos equivocados, que es justo lo que este
    tipo existe para impedir.
    """
    resultado = segmenter.segment_by_speech_rate(_varios_tramos(19))
    assert resultado != tuple(resultado.moments)
    assert resultado == segmenter.segment_by_speech_rate(_varios_tramos(19))

    assert resultado != pytest.approx(0)
