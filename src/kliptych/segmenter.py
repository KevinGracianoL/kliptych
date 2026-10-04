"""Segmentador local determinista: de palabras a momentos candidatos.

Es una funcion PURA. No abre ficheros, no llama a ffprobe, no lee el reloj y no
habla con la red. Todo lo que necesita son las palabras y, si el contrato dice
algo, los limites de duracion. La misma entrada produce exactamente la misma
salida, byte a byte.

Las reglas hacen trabajos distintos y por eso ninguna sobra:

- ``MAX_GAP_S`` SEGMENTA. Es la unica regla que crea fronteras: un silencio
  mayor que el maximo define donde ACABA el habla activa. Corte duro.
- ``MIN_SPEECH_RATE_WPS`` ORDENA. Es una senal de ranking: coloca los tramos,
  nunca los elimina.
- ``MIN_WORDS`` y el piso de duracion DESCARTAN. Es la unica regla que elimina,
  y descarta cuando falla CUALQUIERA de los dos.

Los dos pisos son DUROS, no mitades de una conjuncion. La conjuncion fue un
error, y lo从句 los datos reales de un VOD: "5 y 10." son 3 palabras en 2,0 s y
"diferencia es Diff." son 3 palabras en 7,2 s. Con AND ambos sobrevivian,
porque la duracion pasaba el suelo de palabras y las palabras no llegaban al
suelo de duracion. Ninguno de los dos es un clip: son alguien escribiendo en una
diapositiva mientras whisper transcribe.

Si el ranking descartara, tendria la misma autoridad que el descarte y la regla
de descarte dejaria de significar nada.

El piso de duracion existe, ademas, como previsor: si un brief fija ``min_s``
manana, un tramo mas corto que ese piso haria fallar la seleccion entera de una
vez, no solo ese tramo. Hoy el modo zero-contract no fija piso, asi que el suelo
propio es el unico que aplica; el motivo se mantiene aunque el riesgo aun sea
latente.

El ``score`` es RELATIVO a la entrada: las palabras por segundo del tramo
divididas por las del tramo mas rapido del mismo video. No hay constante que
ajustar y queda acotado en 0..1 por construccion. Como depende del resto de la
entrada, dos tramos con la misma densidad pueden puntuar distinto en videos
distintos; el gate nunca lo lee y aqui solo importa el orden.

Deuda conocida para A4: si los momentos necesitan titulos o una puntuacion
estable entre ejecuciones, habra que anadir una magnitud ABSOLUTA (densidad
simple, o densidad normalizada por una constante con nombre). Ese cambio no se
anticipa aqui porque anadiria una constante que hoy nadie justify.
"""

from collections.abc import Sequence
from itertools import pairwise
from typing import Final

from kliptych.moments import Moment, MomentSource
from kliptych.transcribe import Word

# Frecuencia de habla que ordena los tramos. NO descarta: ordena.
MIN_SPEECH_RATE_WPS: Final[float] = 2.0
# Silencio que cierra un tramo. ESTRICTAMENTE mayor: justo en el borde no corta.
#
# VALOR MEDIDO, no elegido por gusto. Viene del histograma de huecos de un
# excerpt real de 10 min (una clase, 915 palabras de faster-whisper) y del barrido
# de umbrales sobre el, todo con la regla de descarte ya corregida:
#
#     MAX_GAP_S   clips   mediana   <6s
#          0.6      56      4.82    33
#          1.0      46      6.24    22
#          1.5      33     10.36     9
#          2.0      24     18.64     2   <- la rodilla
#          3.0      18     20.24     1   (y el maximo.topa el tope de 60 s)
#
# 2,0 es donde la cola de clips por debajo de 6 s se hunde (22 -> 9 -> 2) sin que
# la mediana se pase de lo publicable. 3,0 no compra nada: un clip mas de cola y
# empuja el maximo contra el tope, o sea que parte tramos por la mitad.
#
# LO QUE EL HISTOGRAMA DICE Y QUE NO HAY QUE VOLVER A MEDIR: entre 0,2 y 3,0 s el
# histograma es una meseta sin separacion. Reparto de los 914 huecos:
#     <0,2 s  766 huecos    5,6 s
#     0,2-0,5  38          11,5 s
#     0,5-1,0  43          32,7 s
#     1,0-1,5  26          31,5 s
#     1,5-2,0  18          30,4 s
#     2,0-3,0   7          16,4 s
#     >3,0 s   16          95,2 s   <- 42,6 % de todo el silencio
#
# Es decir: la unica frontera con separacion real esta por encima de 3 s, y ahi
# solo quedan 18 tramos. Cualquier constante puesta entre 0,6 y 3,0 esta partiendo
# una meseta, y por eso 1,2 o 1,7 no son "ajustar": son otro punto arbitrario de la
# misma meseta. Ver la issue sobre por que el silencio, por si solo, no puede dar
# la forma de salida de un competidor.
#
# Ojo con la aritmetica: la comparacion es contra una resta de flotantes, asi que
# un hueco de "2,0 s" es aproximado en el orden de 1e-16. La salida sigue siendo
# DETERMINISTA: es la misma entrada la misma vez, siempre. Lo unico difuso es el
# SIGNIFICADO del numero en una medida de cero, y anadir redondeo para taparlo
# seria maquinaria para esconder un problema que no llega a ser un problema.
MAX_GAP_S: Final[float] = 2.0
# Tope de momentos emitidos. Es un TECHO, no un filtro: si hay mas tramos que
# este, se ordenan por interes y se quedan los mejores.
MAX_MOMENTS: Final[int] = 12
# Descartan JUNTAS: menos palabras que esto Y mas corto que el piso.
MIN_WORDS: Final[int] = 4
# Piso propio del segmentador, usado cuando el contrato no dice nada.
MIN_CLIP_S: Final[float] = 2.0
# Duracion preferida: ordena por encima, nunca descarta.
PREFERRED_CLIP_S: Final[float] = 8.0
# Tope propio del segmentador, usado cuando el contrato no dice nada.
MAX_CLIP_S: Final[float] = 60.0
# Fraccion final del tope donde se busca un corte que no parta una frase.
SPLIT_SEARCH_TAIL: Final[float] = 0.2

_SENTENCE_ENDINGS: Final[frozenset[str]] = frozenset(".?!,;:")


def segment_by_speech_rate(
    words: Sequence[Word],
    *,
    min_s: float | None = None,
    max_s: float | None = None,
) -> tuple[Moment, ...]:
    """Trocea las palabras en momentos candidatos, de mayor a menor interes.

    Honra los limites de duracion del contrato cuando los tiene: el piso es el
    mayor entre el suyo y ``min_s``, y el tope es el menor entre el suyo y
    ``max_s``. Las constantes del modulo son solo los valores por defecto que se
    usan cuando el contrato calla.

    Args:
        words: Las palabras con marca de tiempo, en cualquier orden.
        min_s: Piso de duracion del contrato, si lo fija.
        max_s: Tope de duracion del contrato, si lo fija.

    Returns:
        Los momentos, ordenados. Vacio si no hay nada que.emitir.
    """
    if not words:
        return ()
    ordenadas = tuple(sorted(words, key=lambda w: (w.start_s, w.end_s, w.token_id)))
    piso = _piso_efectivo(min_s)
    tope = _tope_efectivo(max_s)
    tramos = _descartar(tramos=_partir(tramos=_tramos(ordenadas), tope=tope), piso=piso)
    if not tramos:
        return ()
    puntuados = [(tramo, _densidad(tramo)) for tramo in tramos]
    maximo = max(densidad for _, densidad in puntuados)
    momentos = [
        Moment(
            start_s=tramo[0].start_s,
            end_s=tramo[-1].end_s,
            score=densidad / maximo if maximo > 0.0 else 0.0,
            source=MomentSource.SPEECH_RATE,
        )
        for tramo, densidad in puntuados
    ]
    # El TECHO no se aplica por tiempo sino por INTERES, y el interes es
    # LEXICOGRAFICO: primero la banda de duracion, despues la densidad dentro de
    # la banda. Es una regla, no una formula, asi que no anade constantes.
    #
    # Por que no basta con ordenar por score: la densidad favorece los tramos
    # cortos y apretados, porque un burst de 3 s es mas denso que una reflexion de
    # 40 s. Un tope sobre la densidad sola devolveria doce clips de tres segundos,
    # que es justo lo que se quiere dejar de producir. La banda de duracion va
    # primero, de modo que un tramo de 3 s NO puede adelantar a uno de 20 s con la
    # misma densidad.
    momentos = sorted(momentos, key=_interes)
    momentos = momentos[:MAX_MOMENTS]
    # ORDEN CRONOLOGICO POR CONTRATO. El score es un ATRIBUTO, no la clave de
    # orden: ordenar por score es decision del consumidor, no del productor. Con
    # orden por score las dos mitades de un tramo partido salian en orden
    # arbitrario, porque sus densidades son casi iguales y nadie ve por que una
    # va antes que la otra. Para quien lee la lista, el tiempo es la unica
    # ordenacion que no necesita explicacion.
    return tuple(sorted(momentos, key=lambda m: (m.start_s, m.end_s)))


def _interes(momento: Moment) -> tuple[int, float, float, float]:
    """Ordena momentos para quedarse con los mejores, sin anadir constantes.

    Es LEXICOGRAFICO y en este orden: primero la banda de duracion (un tramo que
    llega a ``PREFERRED_CLIP_S`` por delante de uno que no), despues la densidad
    dentro de la banda, y al final el tiempo como desempate estable.

    Args:
        momento: El momento a ordenar.

    Returns:
        La clave, con ``0`` en la banda preferida.
    """
    duracion = momento.end_s - momento.start_s
    banda = 0 if duracion >= PREFERRED_CLIP_S else 1
    return (banda, -momento.score, momento.start_s, momento.end_s)


def _piso_efectivo(min_s: float | None) -> float:
    """El piso de duracion.

    Args:
        min_s: El piso del contrato, si lo fija.

    Returns:
        El mayor entre el piso del contrato y el del modulo.
    """
    return max(MIN_CLIP_S, min_s or 0.0)


def _tope_efectivo(max_s: float | None) -> float:
    """El tope de duracion.

    Args:
        max_s: El tope del contrato, si lo fija.

    Returns:
        El menor entre el tope del contrato y el del modulo.
    """
    return min(MAX_CLIP_S, max_s) if max_s else MAX_CLIP_S


def _tramos(words: tuple[Word, ...]) -> list[list[Word]]:
    """Parte en tramos por silencio.

    Es la unica regla que crea fronteras: un silencio mayor que el maximo define
    donde acaba el habla activa.

    Args:
        words: Las palabras ya ordenadas.

    Returns:
        Los tramos, cada uno con al menos una palabra.
    """
    tramos: list[list[Word]] = [[words[0]]]
    for anterior, palabra in pairwise(words):
        if palabra.start_s - anterior.end_s > MAX_GAP_S:
            tramos.append([palabra])
        else:
            tramos[-1].append(palabra)
    return tramos


def _partir(*, tramos: list[list[Word]], tope: float) -> list[list[Word]]:
    """Parte los tramos que superen el tope, sin perder ni repetir palabras.

    Args:
        tramos: Los tramos venidos de la segmentacion por silencio.
        tope: La duracion maxima de un tramo.

    Returns:
        Los tramos, partidos donde hacia falta.
    """
    resultado: list[list[Word]] = []
    for tramo in tramos:
        resultado.extend(_partir_tramo(tramo, tope))
    return resultado


def _partir_tramo(tramo: list[Word], tope: float) -> list[list[Word]]:
    """Corta un tramo largo en el punto mas natural dentro del tope.

    Args:
        tramo: El tramo a partir.
        tope: La duracion maxima.

    Returns:
        Los trozos resultantes, ninguno mas largo que el tope.
    """
    if (tramo[-1].end_s - tramo[0].start_s) <= tope:
        return [tramo]
    corte = _punto_de_corte(tramo, tope)
    izquierda = tramo[: corte + 1]
    derecha = tramo[corte + 1 :]
    if not izquierda or not derecha:
        return [tramo]
    return [*_partir_tramo(izquierda, tope), *_partir_tramo(derecha, tope)]


def _punto_de_corte(tramo: list[Word], tope: float) -> int:
    """Elige el indice de la ultima palabra de la parte izquierda.

    Primero busca un cierre de frase dentro de la cola final del tope; si no hay
    ninguno, toma el limite de palabra mas proximo al tope. La ventana es
    ``[tope - 0.2*tope, tope]`` y NO el ultimo 20% del tramo: un cierre de frase
    posterior al tope atraeria el corte mas alla del tope, que es inutil.

    Args:
        tramo: Las palabras del tramo a partir.
        tope: La duracion maxima.

    Returns:
        El indice de la ultima palabra de la parte izquierda.
    """
    desde = tope * (1.0 - SPLIT_SEARCH_TAIL)
    for indice in range(len(tramo) - 1, -1, -1):
        fin = tramo[indice].end_s
        if fin > tope:
            continue
        if fin >= desde and _cierra_frase(tramo[indice]):
            return indice
    candidatos = [i for i, w in enumerate(tramo[:-1]) if w.end_s <= tope]
    if candidatos:
        return max(candidatos, key=lambda i: tramo[i].end_s)
    return 0


def _descartar(*, tramos: list[list[Word]], piso: float) -> list[list[Word]]:
    """La unica regla que elimina: si falla CUALQUIERA de los dos suelos.

    Args:
        tramos: Los tramos ya partidos.
        piso: El piso de duracion efectivo.

    Returns:
        Los tramos que sobreviven.
    """
    return [
        tramo
        for tramo in tramos
        if len(tramo) >= MIN_WORDS and (tramo[-1].end_s - tramo[0].start_s) >= piso
    ]


def _densidad(tramo: list[Word]) -> float:
    """Palabras por segundo del tramo.

    Args:
        tramo: Las palabras del tramo.

    Returns:
        La densidad, o cero si el tramo no dura nada.
    """
    duracion = tramo[-1].end_s - tramo[0].start_s
    return len(tramo) / duracion if duracion > 0.0 else 0.0


def _cierra_frase(palabra: Word) -> bool:
    """Si la palabra termina en puntuacion de cierre.

    Args:
        palabra: La palabra a inspeccionar.

    Returns:
        True si su texto acaba en puntuacion.
    """
    return palabra.text.rstrip()[-1:] in _SENTENCE_ENDINGS
