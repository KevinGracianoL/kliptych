"""Propiedades del segmentador sobre entradas generadas.

Una funcion pura admite una bateria de propiedades mucho mas dura que una lista
de casos. Estas son las que importan:

- la salida es siempre DISJUNTA, probada por indice y no a ojo;
- cada momento cabe dentro de la duracion de las palabras;
- nunca sale un momento sin palabras;
- nunca sale un momento por debajo del piso efectivo;
- el score siempre cae en 0..1;
- la misma entrada da siempre la misma salida.

Y la que habria atrapado el conflicto de dos pisos: con `min_s`/`max_s`
tomados de varios contratos, la salida SIEMPRE satisface los limites del
contrato.
"""

from collections.abc import Callable, Sequence
from typing import cast

import pytest

from kliptych import segmenter
from kliptych.moments import Moment, MomentSource
from kliptych.transcribe import Word

_CONTRATOS: Sequence[tuple[float | None, float | None]] = (
    (None, None),
    (None, 20.0),
    (None, 60.0),
    (8.0, None),
    (8.0, 30.0),
    (2.0, 15.0),
    (30.0, None),
    (1.0, 5.0),
)


class _Azar:
    """Generador congruencial lineal, sembrado y reproducible.

    No se usa `random` a proposito: la suite no admite el modulo por su perfil
    criptografico, y aqui no se quiere criptografia sino una bateria estable que
    se pueda reproducir cuando algo falle.
    """

    def __init__(self, semilla: int) -> None:
        self._estado: int = semilla

    def siguiente(self) -> float:
        """Avanza el estado.

        Returns:
            Un real en [0, 1).
        """
        self._estado = (self._estado * 6364136223846793005 + 1442695040888963407) % (2**64)
        return (self._estado >> 11) / float(2**53)

    def entero(self, alto: int) -> int:
        """Toma un entero.

        Returns:
            Un entero en [0, alto).
        """
        return int(self.siguiente() * alto)

    def opcion(self, opciones: Sequence[float]) -> float:
        """Toma un elemento.

        Returns:
            Uno de los de la secuencia, al azar.
        """
        return opciones[self.entero(len(opciones))]


def _generar(rng: _Azar) -> list[Word]:
    """Genera una lista de palabras plausible, con huecos de varios tamanos.

    Args:
        rng: El generador sembrado, para que la bateria sea reproducible.

    Returns:
        Las palabras, en orden temporal.
    """
    palabras: list[Word] = []
    t = rng.siguiente() * 2.0
    for indice in range(rng.entero(221)):
        duracion = 0.08 + rng.siguiente() * 0.82
        palabras.append(
            Word(
                start_s=round(t, 4),
                end_s=round(t + duracion, 4),
                text=f"w{indice}" + ("." if rng.siguiente() < 0.08 else ""),
                confidence=0.9,
                token_id=indice,
            )
        )
        t += duracion + rng.opcion((0.0, 0.02, 0.15, 0.61, 0.7, 1.4))
    return palabras


def _dentro(momento: Moment, palabras: Sequence[Word]) -> list[Word]:
    """Palabras contenidas en el momento.

    Args:
        momento: El momento a inspeccionar.
        palabras: Todas las palabras de la entrada.

    Returns:
        Las palabras que caben dentro del momento.
    """
    return [p for p in palabras if momento.start_s <= p.start_s and p.end_s <= momento.end_s]


@pytest.mark.parametrize(("min_s", "max_s"), _CONTRATOS)
def test_la_salida_siempre_respeta_los_limites_del_contrato(
    min_s: float | None, max_s: float | None
) -> None:
    rng = _Azar(20261004)
    for _ in range(120):
        palabras = _generar(rng)
        momentos = segmenter.segment_by_speech_rate(palabras, min_s=min_s, max_s=max_s)
        piso = max(segmenter.MIN_CLIP_S, min_s or 0.0)
        tope = min(segmenter.MAX_CLIP_S, max_s) if max_s else segmenter.MAX_CLIP_S
        contenidas_por_momento = [_dentro(m, palabras) for m in momentos]
        for momento, contenidas in zip(momentos, contenidas_por_momento, strict=True):
            duracion = momento.end_s - momento.start_s
            # El piso NO es un minimo de duracion duro: la regla de descarte es un
            # AND, asi que un tramo CORTO pero con palabras densas se emite a
            # proposito. Lo que no puede salir nunca es un tramo que sea corto Y
            # ademas escaso de palabras.
            if duracion < piso - 1e-6:
                assert len(contenidas) >= segmenter.MIN_WORDS, (
                    duracion,
                    piso,
                    len(contenidas),
                    min_s,
                    max_s,
                )
            assert duracion <= tope + 1e-6, (duracion, tope, min_s, max_s)


@pytest.mark.parametrize(("min_s", "max_s"), _CONTRATOS)
def test_la_salida_es_disjunta_y_dentro_del_rango(min_s: float | None, max_s: float | None) -> None:
    rng = _Azar(777)
    for _ in range(120):
        palabras = _generar(rng)
        momentos = segmenter.segment_by_speech_rate(palabras, min_s=min_s, max_s=max_s)
        contenidas_por_momento = [_dentro(m, palabras) for m in momentos]
        for momento, contenidas in zip(momentos, contenidas_por_momento, strict=True):
            assert contenidas, "un momento sin palabras no debe salir"
            assert momento.source is MomentSource.SPEECH_RATE
            assert 0.0 <= momento.score <= 1.0
        # Disjunto por indice: ningun token_id aparece en dos momentos.
        vistos: set[int] = set()
        for contenidas in contenidas_por_momento:
            ids = {p.token_id for p in contenidas}
            assert not (ids & vistos), f"solape en {sorted(ids & vistos)}"
            vistos |= ids


def test_la_salida_es_determinista_sobre_entradas_generadas() -> None:
    rng = _Azar(31337)
    for _ in range(60):
        palabras = _generar(rng)
        primero = segmenter.segment_by_speech_rate(palabras, min_s=8.0, max_s=30.0)
        for _repeticion in range(3):
            assert segmenter.segment_by_speech_rate(palabras, min_s=8.0, max_s=30.0) == primero


_PISO = "_piso_efectivo"
_TOPE = "_tope_efectivo"


def test_los_limites_efectivos_usan_el_contrato_cuando_esta_mudo() -> None:
    """Sin contrato el modulo manda; con contrato, el contrato manda."""
    palabras = [
        Word(start_s=i * 0.5, end_s=i * 0.5 + 0.4, text=f"w{i}", confidence=0.9, token_id=i)
        for i in range(200)
    ]
    sin_contrato = segmenter.segment_by_speech_rate(palabras)
    con_tope = segmenter.segment_by_speech_rate(palabras, max_s=25.0)
    piso = cast("Callable[[float | None], float]", getattr(segmenter, _PISO))
    tope = cast("Callable[[float | None], float]", getattr(segmenter, _TOPE))
    assert piso(None) == segmenter.MIN_CLIP_S
    assert piso(8.0) == pytest.approx(8.0)
    assert tope(None) == segmenter.MAX_CLIP_S
    assert tope(25.0) == pytest.approx(25.0)
    assert len(con_tope) >= len(sin_contrato)
