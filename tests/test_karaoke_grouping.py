"""Agrupacion de palabras en lineas, una por evento ``Dialogue``.

Las palabras son atomicas y una linea nunca queda vacia. Aqui se fijan los tres
limites que deciden donde se corta: longitud, duracion y puntuacion.
"""

from collections.abc import Callable, Sequence
from typing import cast

import pytest

from kliptych import subtitles
from kliptych.transcribe import Word


def _private(name: str) -> object:
    return cast("object", getattr(subtitles, name))


_group = cast("Callable[[Sequence[Word]], list[list[Word]]]", _private("_group_words"))


def _textos(words: Sequence[Word]) -> list[list[str]]:
    """Agrupa y devuelve solo los textos, que es lo que se afirma.

    Args:
        words: Las palabras de la pieza.

    Returns:
        Una lista de lineas, cada una con sus textos.
    """
    return [[w.text for w in linea] for linea in _group(words)]


def _words(texts: Sequence[str], *, step: float = 0.5, gap: float = 0.0) -> list[Word]:
    """Fabrica palabras seguidas, de `step` cada una, con `gap` de silencio.

    Args:
        texts: Los textos, en orden.
        step: Duracion de cada palabra.
        gap: Silencio entre palabra y palabra.

    Returns:
        Las palabras con su marca de tiempo.
    """
    words: list[Word] = []
    t = 0.0
    for index, text in enumerate(texts):
        words.append(Word(start_s=t, end_s=t + step, text=text, confidence=0.9, token_id=index))
        t += step + gap
    return words


def test_empty_input_gives_no_lines() -> None:
    assert _textos([]) == []


def test_single_word_is_a_valid_line() -> None:
    assert _textos(_words(["hola"])) == [["hola"]]


def test_line_breaks_when_adding_a_word_would_exceed_max_chars() -> None:
    assert subtitles.MAX_CHARS_PER_LINE == 18
    # Tres palabras de 5 caracteres: "abcde abcde abcde" son 17 y caben.
    assert _textos(_words(["abcde", "abcde", "abcde"])) == [["abcde", "abcde", "abcde"]]
    # Cuatro son 23 y no caben: la cuarta abre linea.
    assert _textos(_words(["abcde", "abcde", "abcde", "abcde"])) == [
        ["abcde", "abcde", "abcde"],
        ["abcde"],
    ]


def test_line_breaks_when_the_span_exceeds_max_duration() -> None:
    assert pytest.approx(7.0) == subtitles.MAX_DURATION_S
    # Palabras de un caracter pero separadas 1 s: caben en caracteres y no pueden
    # compartir linea porque la linea duraria mas de MAX_DURATION_S.
    words = _words(["a", "b", "c", "d", "e", "f", "g", "h", "i"], step=0.01, gap=0.99)
    lines = _textos(words)
    assert len(lines) > 1, lines
    por_texto = {w.text: w for w in words}
    for linea in lines:
        span = por_texto[linea[-1]].end_s - por_texto[linea[0]].start_s
        assert span <= subtitles.MAX_DURATION_S + 1e-9, span


def test_punctuation_prefers_to_break_a_sentence() -> None:
    # "dos." cierra frase: la linea se parte aunque "cuatro" cabria en 18.
    assert _textos(_words(["uno", "dos.", "cuatro", "cinco"])) == [
        ["uno", "dos."],
        ["cuatro", "cinco"],
    ]


def test_punctuation_does_not_orphan_a_single_word_line() -> None:
    # "solo." es la primera palabra de la linea: partir aqui dejaria un
    # huerfano, asi que se sigue anadiendo.
    assert _textos(_words(["solo.", "dos", "tres"])) == [["solo.", "dos", "tres"]]


@pytest.mark.parametrize("ending", [".", "?", "!", ",", ";", ":"])
def test_every_declared_sentence_ending_breaks(ending: str) -> None:
    assert subtitles.SENTENCE_ENDINGS == ".?!,;:"
    assert _textos(_words(["uno", f"dos{ending}", "tres"])) == [
        ["uno", f"dos{ending}"],
        ["tres"],
    ]


def test_a_word_longer_than_the_limit_gets_its_own_line() -> None:
    larga = "x" * (subtitles.MAX_CHARS_PER_LINE + 5)
    assert _textos(_words([larga, "corta"])) == [[larga], ["corta"]]


def test_no_line_is_ever_empty_and_every_word_is_kept() -> None:
    texts = ["uno", "dos.", "tres", "cuatro.", "cinco", "seis"]
    lines = _textos(_words(texts))
    assert all(linea for linea in lines), lines
    assert [t for linea in lines for t in linea] == texts


def test_grouping_keeps_the_words_in_order_without_repeats() -> None:
    texts = ["uno", "dos.", "tres", "cuatro", "cinco."]
    lines = _textos(_words(texts, step=0.3, gap=0.05))
    flat = [t for linea in lines for t in linea]
    assert flat == texts, flat
    assert len(flat) == len(set(flat))
