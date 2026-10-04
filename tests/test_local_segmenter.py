"""El segmentador local determinista: de palabras a momentos.

Es una funcion PURA: `tuple[Word, ...] -> tuple[Moment, ...]`. Sin reloj, sin
ffprobe, sin ficheros, sin red. Todo lo que necesita ya viene en las palabras.

Las cinco reglas hacen trabajos DISTINTOS, y por eso ninguna sobra:

- `MAX_GAP_S` SEGMENTA: es la unica que crea fronteras. Define donde acaba el
  habla activa. Corte duro.
- `MIN_SPEECH_RATE_WPS` ORDENA: es una senal de ranking. Nunca descarta un
  tramo; los coloca.
- `MIN_WORDS`/`MIN_CLIP_S` DESCARTAN: la unica regla que elimina.

Si el ranking descartara, tendria la misma autoridad que el descarte y la
semantica AND de la regla de descarte se ensuciaria.
"""

from collections.abc import Sequence
from typing import cast

import pytest

from kliptych import moments, segmenter
from kliptych.moments import MomentSource
from kliptych.transcribe import Word


def _palabras(textos: Sequence[str], *, paso: float = 0.5, inicio: float = 0.0) -> list[Word]:
    """Palabras consecutivas de `paso` cada una, sin huecos.

    Args:
        textos: Los textos, en orden.
        paso: Duracion de cada palabra.
        inicio: Comienzo de la primera.

    Returns:
        Las palabras con su marca de tiempo.
    """
    return [
        Word(
            start_s=inicio + i * paso,
            end_s=inicio + i * paso + paso * 0.8,
            text=t,
            confidence=0.9,
            token_id=i,
        )
        for i, t in enumerate(textos)
    ]


def test_dos_tramos_separados_por_silencio_dan_dos_momentos() -> None:
    palabras = _palabras(["a", "b", "c", "d"])
    palabras += _palabras(["e", "f", "g", "h"], paso=0.5, inicio=3.0)
    momentos = segmenter.segment_by_speech_rate(palabras)
    assert len(momentos) == 2, momentos
    assert momentos[0].start_s == pytest.approx(0.0)
    assert momentos[1].start_s == pytest.approx(3.0)


def test_un_silencio_claramente_por_debajo_del_maximo_no_corta() -> None:
    """Con un epsilon deliberado por debajo del maximo, no corta.

    Aqui el objetivo NO es el borde exacto sino la INTENCION de la regla: un
    silencio de medio milisegundo por debajo del maximo esta claramente dentro.
    Este test no depende de ningun accidente de representacion.
    """
    palabras = _borde(1.0, 1.0 + (segmenter.MAX_GAP_S - 1e-9))
    momentos = segmenter.segment_by_speech_rate(palabras)
    assert len(momentos) == 1, momentos


def test_un_silencio_claramente_por_encima_del_maximo_si_corta() -> None:
    """Con un epsilon deliberado por encima del maximo, corta."""
    palabras = _borde(1.0, 1.0 + (segmenter.MAX_GAP_S + 1e-9))
    momentos = segmenter.segment_by_speech_rate(palabras)
    assert len(momentos) == 2, momentos


def test_el_borde_exactamente_igual_no_corta() -> None:
    """Un hueco que ES exactamente el maximo no corta: el limite es estricto.

    Este es el UNICO test que puede matar la mutacion `>` -> `>=`, y no por
    casualidad sino por aritmetica: con un epsilon de 1e-9, las dos
    comparaciones coinciden, asi que un test con epsilon no distinguiria nada.
    Solo un hueco cuyo valor float sea exactamente el doble 0.6 las separa:

        0.6 - 1e-9 = 0.599999999   > 0.6 falso, >= 0.6 falso   (iguales)
        0.6 + 1e-9 = 0.600000001   > 0.6 cierto, >= 0.6 cierto (iguales)
        1.0 - 0.4   = 0.6          > 0.6 falso, >= 0.6 CIERTO  (distinguen)

    Se construye el hueco restando 1.0 menos 0.4, que en IEEE754 da exactamente
    el doble 0.6. No es arbitrario: es el unico camino, y por eso se escribe el
    porque en el codigo en vez de confiar en la suerte.
    """
    assert pytest.approx(0.6, abs=0.0) == (1.0 - 0.4)
    palabras = _borde(0.4, 1.0)
    momentos = segmenter.segment_by_speech_rate(palabras)
    assert pytest.approx(0.6) == segmenter.MAX_GAP_S
    assert len(momentos) == 1, momentos


def test_un_tramo_de_tres_palabras_y_1_5s_se_descarta() -> None:
    """Regla 4: hacen falta LAS DOS condiciones para descartar."""
    palabras = [
        Word(start_s=0.0, end_s=0.5, text="a", confidence=0.9, token_id=0),
        Word(start_s=0.5, end_s=1.0, text="b", confidence=0.9, token_id=1),
        Word(start_s=1.0, end_s=1.5, text="c", confidence=0.9, token_id=2),
    ]
    assert segmenter.segment_by_speech_rate(palabras) == ()


def test_un_tramo_de_tres_palabras_y_5s_se_conserva() -> None:
    """Con min_s=None, 5.0 >= MIN_CLIP_S as que la regla 4 no descarta."""
    palabras = _palabras(["a", "b", "c"], paso=1.67, inicio=0.0)
    momentos = segmenter.segment_by_speech_rate(palabras)
    assert len(momentos) == 1, momentos


def test_un_tramo_de_seis_palabras_y_1s_se_conserva() -> None:
    """Y al reves: muchas palabras y poco tiempo tambien se conserva."""
    palabras = _palabras(["a", "b", "c", "d", "e", "f"], paso=0.167, inicio=0.0)
    momentos = segmenter.segment_by_speech_rate(palabras)
    assert len(momentos) == 1, momentos


def test_el_contrato_manda_en_el_piso() -> None:
    """Con min_s=8.0 el mismo tramo de 5.0 s ahora SI se descarta.

    Esta es la prueba de que el contrato manda de verdad: la misma entrada con y
    sin piso da distinto resultado, y el piso viene del contrato, no del modulo.
    """
    palabras = _palabras(["a", "b", "c"], paso=1.67, inicio=0.0)
    assert len(segmenter.segment_by_speech_rate(palabras)) == 1
    assert segmenter.segment_by_speech_rate(palabras, min_s=8.0) == ()


def test_un_tramo_de_90s_separte_en_el_limite() -> None:
    """Un tramo continuo de 90 s no tiene silencios: el tope tiene que PARTIR."""
    palabras = _palabras([f"w{i}" for i in range(180)], paso=0.5, inicio=0.0)
    # La salida va ordenada por score, y dos mitades de un tramo continuo casi
    # igual de densas pueden salir en cualquier orden. Aqui se ordena por inicio
    # porque lo que se afirma es donde CAE el corte, no quien gana el ranking.
    momentos = tuple(sorted(segmenter.segment_by_speech_rate(palabras), key=lambda m: m.start_s))
    assert len(momentos) == 2, momentos
    izquierda, derecha = momentos[0], momentos[1]
    assert izquierda.end_s == pytest.approx(60.0, abs=1.0)
    assert derecha.start_s == pytest.approx(60.0, abs=1.0)


def test_la_partida_prefiere_el_ultimo_signo_de_frase() -> None:
    """El corte busca un cierre de frase aunque no sea el mas cercano al tope.

    El signo va en el indice 110 (~55.4 s), dentro de la ventana [48, 60]. El
    limite de palabra mas proximo al topping es el 119 (~59.9 s), que NO tiene
    signo. Si la salida corta en 55.4 y no en 59.9, la preferencia esta viva; si
    corta en 59.9, el signo se esta ignorando y esto es decoracion.
    """
    palabras = _palabras([f"w{i}" for i in range(180)], paso=0.5, inicio=0.0)
    for indice in (108, 109, 110):
        palabras[indice] = Word(
            start_s=palabras[indice].start_s,
            end_s=palabras[indice].end_s,
            text=palabras[indice].text + ".",
            confidence=0.9,
            token_id=indice,
        )
    momentos = tuple(sorted(segmenter.segment_by_speech_rate(palabras), key=lambda m: m.start_s))
    assert len(momentos) == 2, momentos
    assert momentos[0].end_s == pytest.approx(palabras[110].end_s, abs=0.01)
    assert momentos[0].end_s == pytest.approx(55.4, abs=0.05)


def test_la_busqueda_del_corte_esta_acotada_al_20_por_ciento_final() -> None:
    """La ventana de busqueda es [cap - 0.2*cap, cap], no el ultimo 20% del tramo.

    Un tramo de 90 s con un cierre de frase en el segundo 75 NO debe atraer el
    corte: 75 > 60, asi que la particion se pasaria del tope y seria inútil.
    """
    palabras = _palabras([f"w{i}" for i in range(180)], paso=0.5, inicio=0.0)
    palabras[150] = Word(
        start_s=palabras[150].start_s,
        end_s=palabras[150].end_s,
        text="w150.",
        confidence=0.9,
        token_id=150,
    )
    momentos = tuple(sorted(segmenter.segment_by_speech_rate(palabras), key=lambda m: m.start_s))
    assert momentos[0].end_s == pytest.approx(60.0, abs=1.0)


def test_sin_signo_de_frase_usa_el_limite_de_palabra_mas_cercano() -> None:
    """Sin puntuacion, el corte cae en el limite de palabra mas proximo al tope."""
    palabras = _palabras([f"w{i}" for i in range(180)], paso=0.5, inicio=0.0)
    momentos = segmenter.segment_by_speech_rate(palabras)
    fin_izquierda = momentos[0].end_s
    limites = [w.end_s for w in palabras]
    assert min(abs(fin_izquierda - limite) for limite in limites) < 0.3


def test_max_s_del_contrato_manda_en_la_partida() -> None:
    """Con max_s=20.0 un tramo de 40 s parte en 20, no en 60."""
    palabras = _palabras([f"w{i}" for i in range(80)], paso=0.5, inicio=0.0)
    momentos = tuple(
        sorted(segmenter.segment_by_speech_rate(palabras, max_s=20.0), key=lambda m: m.start_s)
    )
    assert len(momentos) == 2, moments if (moments := momentos) else ()
    assert momentos[0].end_s == pytest.approx(20.0, abs=1.0)


def test_entrada_vacia_da_tupla_vacia() -> None:
    assert segmenter.segment_by_speech_rate([]) == ()


def test_una_sola_palabra_se_descarta() -> None:
    assert segmenter.segment_by_speech_rate(_palabras(["solo"])) == ()


def test_palabras_desordenadas_se_ordenan_antes_de_procesar() -> None:
    """Comportamiento fijado: se ordena una copia; la entrada no se toca."""
    palabras = _palabras(["a", "b", "c", "d"])
    revueltas = [palabras[3], palabras[1], palabras[0], palabras[2]]
    momentos = segmenter.segment_by_speech_rate(revueltas)
    assert len(momentos) == 1, momentos
    assert momentos[0].start_s == pytest.approx(0.0)
    assert [w.token_id for w in revueltas] == [3, 1, 0, 2], "la entrada no debe mutarse"


def test_la_salida_es_determinista_byte_a_byte() -> None:
    palabras = _palabras(["a", "b", "c", "d"]) + _palabras(["e", "f", "g", "h"], inicio=3.0)
    primero = segmenter.segment_by_speech_rate(palabras)
    for _ in range(5):
        assert segmenter.segment_by_speech_rate(palabras) == primero


def test_el_score_es_relativo_al_tramo_mas_rapido() -> None:
    """Score = wps_del_tramo / wps_del_tramo_mas_rapido_de_esta_entrada."""
    rapido = _palabras(["a", "b", "c", "d", "e", "f"], paso=0.2, inicio=0.0)
    lento = _palabras(["g", "h", "i", "j", "k", "l"], paso=0.8, inicio=10.0)
    momentos = segmenter.segment_by_speech_rate(rapido + lento)
    assert momentos[0].source is MomentSource.SPEECH_RATE
    assert momentos[0].score == pytest.approx(1.0)
    assert momentos[-1].score < momentos[0].score
    assert all(0.0 <= m.score <= 1.0 for m in momentos)


def _borde(fin_primero: float, inicio_segundo: float) -> list[Word]:
    """Dos tramos de cuatro palabras, unidos por el hueco `inicio_segundo - fin_primero`.

    Se construyen a mano porque el hueco tiene que caer en un valor concreto de
    coma flotante, y un helper que acumula floats no lo garantiza.

    Args:
        fin_primero: El `end_s` de la ultima palabra del primer tramo.
        inicio_segundo: El `start_s` de la primera palabra del segundo tramo.

    Returns:
        Las ocho palabras.
    """
    palabras: list[Word] = [
        Word(
            start_s=indice * 0.1,
            end_s=indice * 0.1 + 0.1,
            text=f"a{indice}",
            confidence=0.9,
            token_id=indice,
        )
        for indice in range(4)
    ]
    palabras[-1] = Word(start_s=0.3, end_s=fin_primero, text="a3", confidence=0.9, token_id=3)
    palabras += [
        Word(
            start_s=inicio_segundo + indice * 0.1,
            end_s=inicio_segundo + indice * 0.1 + 0.1,
            text=f"e{indice}",
            confidence=0.9,
            token_id=4 + indice,
        )
        for indice in range(4)
    ]
    return palabras


_PESOS = "_DEFAULT_WEIGHTS"


def test_la_fuente_nueva_no_entra_en_los_pesos_por_defecto() -> None:
    """`_DEFAULT_WEIGHTS` es exhaustivo y NO se le anade un peso inventado.

    Anadir un peso seria una conjetura sobre el ranking de la fusion, que es
    otro camino. Se documenta que esta fuente no participa en la fusion.
    """
    pesos = cast("dict[str, float]", getattr(moments, _PESOS))
    assert set(pesos) == {
        MomentSource.SCENE,
        MomentSource.AUDIO_ENERGY,
        MomentSource.CHAT_DENSITY,
    }
    assert MomentSource.SPEECH_RATE not in pesos


def test_la_fuente_nueva_es_exhaustiva() -> None:
    """Prueba estructural: si el enum crece, esta prueba obliga a decidir."""
    assert MomentSource.SPEECH_RATE.value == "speech_rate"
    assert len(MomentSource) == 5


def test_la_salida_va_en_orden_cronologico() -> None:
    """El orden es el tiempo, no el score.

    Se fuerza el caso que rompia el orden anterior: dos tramos de la MISMA
    densidad, de forma que ordenar por score no tiene motivo visible y las
    mitades de un partido salen invertidas sin explicacion.
    """
    palabras = _palabras(["a", "b", "c", "d"], paso=0.5)
    palabras += _palabras(["e", "f", "g", "h"], paso=0.5, inicio=5.0)
    momentos = segmenter.segment_by_speech_rate(palabras)
    assert len(momentos) == 2, momentos
    # El segundo tramo va despues en el tiempo, aunque su score sea igual o peor.
    assert momentos[0].start_s < momentos[1].start_s
    assert [m.start_s for m in momentos] == sorted(m.start_s for m in momentos)


def test_el_orden_cronologico_aguanta_tras_un_parto_con_scores_distintos() -> None:
    """Las dos mitades de un partido salen en orden temporal, no por score.

    Se comprueba que el score realmente difiere entre las mitades; si no
    difiriera, la prueba no distinguiria las dos ordenaciones.
    """
    palabras = _palabras([f"w{i}" for i in range(180)], paso=0.5, inicio=0.0)
    momentos = segmenter.segment_by_speech_rate(palabras)
    assert len(momentos) == 2, momentos
    assert momentos[0].start_s < momentos[1].start_s
    assert momentos[0].end_s <= momentos[1].start_s + 1e-9
    assert momentos[0].score != momentos[1].score, (
        "los scores deben diferir para que la prueba signifique algo"
    )


def test_un_tramo_rapido_temprano_no_va_antes_que_uno_lento_tardio() -> None:
    """Ranking por score invertiria el tiempo; el tiempo no se invierte."""
    temprano = _palabras(["a", "b", "c", "d", "e", "f"], paso=0.2, inicio=0.0)
    tardio = _palabras(["g", "h", "i", "j"], paso=1.2, inicio=10.0)
    momentos = segmenter.segment_by_speech_rate(temprano + tardio)
    assert [m.start_s for m in momentos] == [0.0, 10.0]
    assert momentos[0].score == pytest.approx(1.0)


def test_ordenar_por_score_invertiria_el_tiempo() -> None:
    """Discrimina de verdad: el tramo rapido es el ULTIMO, no el primero.

    Un caso con densidades iguales NO distingue las dos ordenaciones, porque con
    score igual el desempate por `start_s` las hace coincidir. Por eso este caso
    pone el tramo rapido al final: ordenar por score lo moveria al principio, que
    es justo lo que este test prohibe.
    """
    lento = _palabras(["a", "b", "c", "d", "e", "f"], paso=1.2, inicio=0.0)
    rapido = _palabras(["g", "h", "i", "j", "k", "l"], paso=0.2, inicio=10.0)
    momentos = segmenter.segment_by_speech_rate(lento + rapido)
    assert len(momentos) == 2, momentos
    assert momentos[0].start_s == pytest.approx(0.0)
    assert momentos[1].start_s == pytest.approx(10.0)
    assert momentos[0].score < momentos[1].score, "el lento debe puntuar menos"
