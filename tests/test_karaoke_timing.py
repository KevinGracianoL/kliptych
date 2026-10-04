"""Tests de la aritmetica de centisegundos del karaoke.

La identidad que sostiene todo esto es que los centisegundos emitidos por una
linea suman EXACTAMENTE la duracion del evento, en centisegundos enteros. Las
ranuras (``next_start - start``, y la ultima hasta ``end``) suman el span en
float de forma telescopica, pero esa identidad no sobrevive a la cuantizacion:
si se redondea cada palabra por su cuenta, la suma deja de cuadrar y el error se
acumula siempre en el mismo sentido.
"""

from typing import TYPE_CHECKING, cast

import pytest

from kliptych import subtitles
from kliptych.transcribe import Word

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence


def _private(name: str) -> object:
    """Accede a un simbolo privado del modulo, como en el resto de la suite.

    Args:
        name: Nombre del simbolo.

    Returns:
        El simbolo privado, tipado como `object` para que el `cast` de cada
        prueba sea explicito.
    """
    return cast("object", getattr(subtitles, name))


_allocate = cast("Callable[[Sequence[Word]], list[int]]", _private("_allocate_centiseconds"))
_line_event = cast("Callable[[Sequence[Word]], str]", _private("_line_event"))


def _slots(slot_s: float, count: int, *, first_start: float = 0.0) -> list[Word]:
    """Linea de ``count`` palabras con ranura constante, sin huecos.

    Args:
        slot_s: Ranura de cada palabra en segundos.
        count: Cuantas palabras.
        first_start: Inicio de la linea.

    Returns:
        Las palabras, cada una empezando donde acaba la anterior.
    """
    words: list[Word] = []
    for index in range(count):
        start = first_start + index * slot_s
        words.append(
            Word(
                start_s=start,
                end_s=start + slot_s,
                text=f"w{index}",
                confidence=0.9,
                token_id=index,
            )
        )
    return words


def test_floor_is_one_centisecond_because_zero_renders_nothing() -> None:
    """El suelo es 1cs porque lo que importa es lo que se DIBUJA, no el numero.

    El motivo NO es que 1 sea un numero redondo. Es el render medido con libass:
    una palabra a la que se le asignan 0cs NO SE DIBUJA en absoluto, sin error ni
    aviso, mientras que a 1cs se dibuja entera. Un suelo de 0 seria un cementerio de
    palabras que desaparecen del subtitulo; 1cs es el menor valor que todavia
    muestra la palabra.
    """
    assert subtitles.MIN_WORD_CENTISECONDS == 1

    # Dos palabras cuya ranura real es menor que un centisegundo: sin suelo,
    # int() las dejaria en 0 y desaparecerian.
    words = [
        Word(start_s=0.000, end_s=0.0004, text="ay", confidence=0.9, token_id=0),
        Word(start_s=0.0004, end_s=0.0500, text="que", confidence=0.9, token_id=1),
    ]
    allocated = _allocate(words)
    assert min(allocated) >= subtitles.MIN_WORD_CENTISECONDS, allocated
    assert allocated == [1, 4], allocated
    assert "{\\k1}ay" in _line_event(words)


def test_degenerate_line_names_both_counts() -> None:
    """El error de linea degenerada dice cuantas palabras y cuantos centisegundos.

    Sin los dos numeros el diagnostico es inutil: hay que poder distinguir "demasiadas
    palabras para el tiempo" de "tiempo insuficiente".
    """
    words = [
        Word(start_s=0.000, end_s=0.002, text="a", confidence=0.9, token_id=0),
        Word(start_s=0.002, end_s=0.004, text="b", confidence=0.9, token_id=1),
        Word(start_s=0.004, end_s=0.006, text="c", confidence=0.9, token_id=2),
    ]
    total_cs = round((words[-1].end_s - words[0].start_s) * 100)
    with pytest.raises(subtitles.SubtitleError) as excinfo:
        _ = _allocate(words)
    message = str(excinfo.value)
    assert str(len(words)) in message, message
    assert str(total_cs) in message, message
    assert "degenerada" in message, message


def test_eight_words_at_point_six_sum_to_the_event_duration() -> None:
    """Ocho palabras de ranura 0.156s: 15.6 cs cada una.

    El total cuantizado de la linea son 125 cs. Redondeando cada palabra por su
    cuenta sale 16 x 8 = 128 cs, tres centisegundos (30 ms) de desviacion, y
    esa desviacion crece con el numero de palabras. El reparto por resto mayor
    reparte el total entre las ocho y la suma vuelve a cuadrar.
    """
    words = _slots(0.156, 8)
    event_cs = round((words[-1].end_s - words[0].start_s) * 100)

    allocated = _allocate(words)

    assert event_cs == 125, event_cs
    assert sum(allocated) == event_cs, (allocated, sum(allocated), event_cs)
    # floor 15 x8 = 120, sobran 5 -> los cinco mayores restos van al primero.
    assert allocated == [16, 16, 16, 16, 16, 15, 15, 15], allocated


def test_event_end_agrees_with_the_sum_of_its_k_values() -> None:
    r"""Las marcas del evento y los {\\k} son la MISMA rejilla.

    Si el final del evento se calculara con los floats de las palabras en vez de
    con la suma de los centisegundos emitidos, las dos representaciones de la
    linea temporal se separarian.
    """
    words = _slots(0.156, 8)
    event = _line_event(words)

    prefix = "Dialogue: 0,"
    body = event[len(prefix) :]
    start_token, end_token, _rest = body.split(",", 2)
    allocated = _allocate(words)

    def to_cs(token: str) -> int:
        hours, minutes, rest = token.split(":")
        secs, centis = rest.split(".")
        return ((int(hours) * 60 + int(minutes)) * 60 + int(secs)) * 100 + int(centis)

    assert to_cs(end_token) - to_cs(start_token) == sum(allocated)
