"""Tests de normalización y coincidencia de texto para validadores del gate."""

from kliptych.gate.text import contains_phrase, normalize_text


def test_normalize_text_preserves_enie_and_removes_accents() -> None:
    assert normalize_text("CAMPEÓN") == "campeon"
    assert normalize_text("Féliz Año Nuevo") == "feliz año nuevo"
    assert normalize_text("PEÑA") == "peña"
    assert normalize_text("COÑO") == "coño"


def test_normalize_text_collapses_whitespace() -> None:
    assert normalize_text("  hola   \n\t  mundo  ") == "hola mundo"


def test_contains_phrase_handles_empty_inputs() -> None:
    assert not contains_phrase("", "algo")
    assert not contains_phrase("algo", "")
    assert not contains_phrase("", "")
    assert not contains_phrase("algo", "   ")
    assert not contains_phrase("   ", "algo")


def test_contains_phrase_word_boundaries_and_separators() -> None:
    assert contains_phrase("sorteo_gratis", "sorteo")
    assert contains_phrase("gran_sorteo_gratis", "sorteo")
    assert contains_phrase("sorteo_gratis", "sorteo_gratis")
    assert contains_phrase("sorteo gratis", "sorteo_gratis")
    assert contains_phrase("sorteo_gratis", "sorteo gratis")
    assert not contains_phrase("sorteos", "sorteo")
    assert not contains_phrase("el ganador", "gana")
    assert contains_phrase("gana ya", "gana")
    assert contains_phrase("gana ya", "gana ya")
    assert contains_phrase("gana_ya", "gana ya")


def test_contains_phrase_preserves_enie_semantics() -> None:
    assert not contains_phrase("feliz año", "ano")
    assert not contains_phrase("feliz ano", "año")
    assert contains_phrase("feliz año", "año")
    assert not contains_phrase("peña", "pena")
    assert not contains_phrase("pena", "peña")
    assert contains_phrase("peña", "peña")


def test_h7_bis_decomposed_enie_normalization() -> None:
    assert normalize_text("an\u0303o") == "año"
    assert contains_phrase("feliz an\u0303o", "año")
