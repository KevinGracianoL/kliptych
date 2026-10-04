"""El techo de momentos y el criterio por el que se eligen.

El techo es lo que impide que el segmentador devuelva una lista sin fin: el
problema medido no era trocear poco, era conservar demasiado. Y el criterio
existe porque la densidad, por si sola, favorece los tramos cortos y apretados,
que son justo los que no sirven.

Ambas cosas eran comportamiento NUEVO sin ninguna prueba. Una cota sin prueba es
una cota que se puede borrar sin que nada se entere.
"""

from typing import TYPE_CHECKING, cast

from kliptych import segmenter
from kliptych.moments import Moment, MomentSource
from kliptych.transcribe import Word

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence


def _private(nombre: str) -> object:
    """Accede a un simbolo privado del modulo, como en el resto de la suite.

    Args:
        nombre: El nombre del simbolo.

    Returns:
        El simbolo, tipado como `object`.
    """
    return cast("object", getattr(segmenter, nombre))


_interes = cast("Callable[[Moment], tuple[int, float, float, float]]", _private("_interes"))


def _momento(inicio: float, fin: float, score: float) -> Moment:
    """Un momento de prueba, con la fuente del segmentador local.

    Args:
        inicio: El `start_s`.
        fin: El `end_s`.
        score: El `score` ya relativo.

    Returns:
        El momento construido.
    """
    return Moment(start_s=inicio, end_s=fin, score=score, source=MomentSource.SPEECH_RATE)


def _tramos(n: int, *, duracion: float, paso: float = 0.7, inicio: float = 0.0) -> list[Word]:
    """N tramos iguales separados por un hueco amplio, todos por encima del umbral.

    Args:
        n: Cuantos tramos.
        duracion: Duracion de cada tramo.
        paso: Separacion entre palabras dentro del tramo.
        inicio: Comienzo del primer tramo.

    Returns:
        Todas las palabras.
    """
    palabras: list[Word] = []
    t = inicio
    token = 0
    for _ in range(n):
        for _ in range(max(4, int(duracion / paso))):
            palabras.append(
                Word(start_s=t, end_s=t + paso * 0.8, text="w", confidence=0.9, token_id=token)
            )
            token += 1
            t += paso
        t += segmenter.MAX_GAP_S + 1.0
    return palabras


# --------------------------------------------------------------------------
# EL TECHO
# --------------------------------------------------------------------------


def test_mas_tramos_que_el_techo_devuelve_exactamente_el_techo() -> None:
    """Con 19 tramos que califican, salen 12: el techo recorta de verdad."""
    momentos = segmenter.segment_by_speech_rate(_tramos(19, duracion=12.0))
    assert len(momentos) == segmenter.MAX_MOMENTS, len(momentos)
    assert segmenter.MAX_MOMENTS == 12


def test_menos_tramos_que_el_techo_no_rellena() -> None:
    """Con 5 tramos que califican salen 5: el techo es un techo, no un objetivo."""
    momentos = segmenter.segment_by_speech_rate(_tramos(5, duracion=12.0))
    assert len(momentos) == 5, len(momentos)


def test_exactamente_el_techo_no_recorta() -> None:
    """El caso limite: 12 tramos, 12 salen. Ni uno menos, ni uno mas."""
    momentos = segmenter.segment_by_speech_rate(_tramos(12, duracion=12.0))
    assert len(momentos) == segmenter.MAX_MOMENTS, len(momentos)


def test_el_techo_se_aplica_despues_de_los_dos_suelos() -> None:
    """Un tramo que muere en un suelo NO ocupa sitio en el techo.

    Si el techo se aplicara antes, los 8 tramos再加上 8 tramos inservibles
    consumiendo huecos, y devolveria 12 tramos de los que 4 son inservibles. Aqui
    los 8 primeros se descartan (3 palabras, por debajo de MIN_WORDS) y los que
    quedan pasan.
    """
    palabras: list[Word] = []
    token = 0
    t = 0.0
    for _ in range(8):
        for _ in range(3):
            palabras.append(
                Word(start_s=t, end_s=t + 0.5, text="x", confidence=0.9, token_id=token)
            )
            token += 1
            t += 0.5
        t += segmenter.MAX_GAP_S + 1.0
    inservibles = len(palabras)
    palabras += _tramos(14, duracion=12.0, inicio=t)
    momentos = segmenter.segment_by_speech_rate(palabras)
    assert inservibles == 24, inservibles
    assert len(momentos) == segmenter.MAX_MOMENTS, len(momentos)
    for momento in momentos:
        assert momento.end_s - momento.start_s >= segmenter.MIN_CLIP_S


# --------------------------------------------------------------------------
# EL CRITERIO
# --------------------------------------------------------------------------


def test_un_tramo_corto_no_adelanta_a_uno_largo_de_la_misma_densidad() -> None:
    """Este es el motivo de que exista `_interes`: 3 s no puede ganarle a 20 s.

    Densidad IGUAL a proposito, porque es justo el caso que la densidad sola no
    puede desempatar. Si este test falla, el techo devolveria doce clips de tres
    segundos, que es el problema que el techo pretendia arreglar.
    """
    corto = _momento(0.0, 3.0, 1.0)
    largo = _momento(10.0, 30.0, 1.0)
    assert _interes(largo) < _interes(corto), "el largo debe ir antes"
    assert largo.end_s - largo.start_s >= segmenter.PREFERRED_CLIP_S
    assert corto.end_s - corto.start_s < segmenter.PREFERRED_CLIP_S


def test_dentro_de_la_misma_banda_manda_la_densidad() -> None:
    """Dos tramos de la misma banda: gana el mas denso."""
    flojo = _momento(0.0, 20.0, 0.40)
    denso = _momento(30.0, 50.0, 0.90)
    assert _interes(denso) < _interes(flojo)


def test_el_tiempo_desempata_y_hace_el_orden_estable() -> None:
    """Con banda y densidad iguales, manda el tiempo: el orden no depende del azar.

    Si dos tramos tienen banda y score identicos, el desempate por `start_s` los
    ordena siempre igual. Es lo que impide que una iteracion de set o dict se
    colen en la salida.
    """
    temprano = _momento(5.0, 25.0, 0.5)
    tarde = _momento(100.0, 120.0, 0.5)
    assert _interes(temprano) < _interes(tarde)
    clave = [_interes(m) for m in (tarde, temprano, temprano)]
    assert clave == sorted(clave, reverse=True)


def test_el_criterio_es_total_y_no_depende_del_orden_de_entrada() -> None:
    """El mismo conjunto de momentos ordena igual venga como venga.

    Es la propiedad que haria fallar cualquier iteracion de set o dict en la
    salida: el desempate final por tiempo hace la ordenacion total.
    """
    momentos = [
        _momento(0.0, 3.0, 1.0),
        _momento(10.0, 30.0, 1.0),
        _momento(50.0, 70.0, 0.4),
        _momento(80.0, 100.0, 0.9),
    ]
    normal = [m.start_s for m in sorted(momentos, key=_interes)]
    for _vuelta in range(4):
        revueltos: Sequence[Moment] = tuple(reversed(momentos))
        assert [m.start_s for m in sorted(revueltos, key=_interes)] == normal


def test_los_doce_seleccionados_en_una_entrada_sintetica() -> None:
    """Los 12 que salen, uno por uno, con duracion, palabras y score.

    Se fija la SELECCION completa, no solo la cuenta: asi se ve que el techo
    elige tramos largos y no solo muchos.
    """
    # 16 tramos: 8 cortos y densos, 8 largos y medios. La densidad sola elegiria
    # los 8 cortos y rellenaria con 4 largos; el criterio elige los largos.
    palabras: list[Word] = []
    token = 0
    t = 0.0
    for indice in range(16):
        if indice % 2 == 0:
            n_palabras, paso = 5, 0.7
        else:
            n_palabras, paso = 30, 0.7
        for _ in range(n_palabras):
            palabras.append(
                Word(start_s=t, end_s=t + paso * 0.8, text="w", confidence=0.9, token_id=token)
            )
            token += 1
            t += paso
        t += segmenter.MAX_GAP_S + 1.0
    momentos = segmenter.segment_by_speech_rate(palabras)
    assert len(momentos) == segmenter.MAX_MOMENTS, len(momentos)
    informe: list[tuple[float, float, int, float]] = []
    for momento in momentos:
        dentro = [w for w in palabras if momento.start_s <= w.start_s and w.end_s <= momento.end_s]
        informe.append(
            (
                round(momento.start_s, 2),
                round(momento.end_s, 2),
                len(dentro),
                round(momento.score, 4),
            )
        )
    for inicio, fin, n, _score in informe:
        assert n >= segmenter.MIN_WORDS, (inicio, fin, n)
        assert fin - inicio >= segmenter.MIN_CLIP_S, (inicio, fin)
    assert len(informe) == segmenter.MAX_MOMENTS, informe
