"""Propiedades del segmentador sobre entradas generadas.

Una funcion pura admite una bateria de propiedades mucho mas dura que una lista
de casos. Estas son las SIETE clausulas invariantes, y este modulo las comprueba
una a una sobre cada momento emitido, en cada corrida generada y para cada
combinacion de limites de contrato:

- **C1** el momento contiene al menos ``segmenter.MIN_WORDS`` palabras;
- **C2** su duracion es ``>= max(segmenter.MIN_CLIP_S, min_s or 0.0)``;
- **C3** su duracion es ``<= min(segmenter.MAX_CLIP_S, max_s)`` si hay ``max_s``,
  y ``<= segmenter.MAX_CLIP_S`` si no;
- **C4** los momentos son disjuntos: el anterior acaba antes de que empiece el
  siguiente;
- **C5** los momentos son estrictamente cronologicos: ``start_s`` crece;
- **C6** cada frontera cae en una frontera de palabra: ``start_s`` es el
  ``start_s`` de alguna palabra y ``end_s`` es el ``end_s`` de alguna palabra;
- **C7** no salen mas de ``segmenter.MAX_MOMENTS`` momentos.

Las siete se enuncian por separado y se comprueban por separado. Fundirlas en
una sola asercion con varias condiciones hace que un fallo diga "algo no cuadra"
en lugar de decir QUE clausula se rompio, y una clausula sin comprobar no se
distingue de una que se cumple.

La que mas surprise es C2, y conviene dejar escrito por que: la regla de
descarte de ``segmenter`` es un AND, ``len(tramo) >= MIN_WORDS AND duracion >=
piso``, asi que la duracion minima es DURA. Un tramo corto con palabras densas
NO sale. Antes este modulo decia lo contrario en un comentario y lo cubria con
una asercion condicional que nunca se ejecutaba, porque el caso que describia no
se puede dar; la comprobacion real es directa y no necesita guarda.
"""

from collections.abc import Callable, Sequence
from itertools import pairwise
from pathlib import Path
from typing import cast

import pytest

from kliptych import segmenter
from kliptych.moments import Moment, MomentSource
from kliptych.transcribe import Word

# (min_s, max_s) de varios contratos, incluidos los mudos, los que solo fijan un
# lado y los que fijan los dos por debajo y por encima de las constantes del
# modulo. Los tres casos limite importan: con (1.0, 5.0) manda el contrato en
# ambos extremos, con (30.0, None) manda el tope del modulo, y con (None, None)
# mandan las dos constantes.
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

_TOLERANCIA_S = 1e-6

# Una corrida de la bateria: las palabras generadas y el rango con el que se
# evaluan. Es un alias y no un `object` suelto para que el tipado siga siendo
# util: el segmenter recibe `float | None` en los dos lados.
_Corrida = tuple[list[Word], tuple[float | None, float | None]]


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

    Los huecos importan mas que las palabras: son los que crean tramos
    distintos y los que hacen que un tramo caiga por debajo del piso, que es de
    donde salen los casos que hacen trabajar a las clausulas.

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


def _piso(min_s: float | None) -> float:
    """El piso efectivo que la clausula C2 tiene que cumplir.

    Returns:
        El mayor entre el piso del modulo y el del contrato.
    """
    return max(segmenter.MIN_CLIP_S, min_s or 0.0)


def _tope(max_s: float | None) -> float:
    """El tope efectivo que la clausula C3 tiene que cumplir.

    Returns:
        El menor entre el tope del modulo y el del contrato.
    """
    return min(segmenter.MAX_CLIP_S, max_s) if max_s else segmenter.MAX_CLIP_S


def _corridas(semilla: int, repeticiones: int) -> list[_Corrida]:
    """Genera las corridas de la bateria.

    Args:
        semilla: La semilla del generador, para reproducir un fallo.
        repeticiones: Cuantas entradas generar.

    Returns:
        Las palabras de cada corrida y el rango de contrato con el que se
        evaluaron, en el orden en que se recorren.
    """
    rng = _Azar(semilla)
    corridas: list[_Corrida] = []
    for _ in range(repeticiones):
        palabras = _generar(rng)
        for min_s, max_s in _CONTRATOS:
            corridas.append((palabras, (min_s, max_s)))
    return corridas


# La clausula se comprueba por una funcion nombrada, para que el fallo diga cual
# se rompio y no "una asercion de este test".
def _c1_palabras(momento: Moment, palabras: Sequence[Word]) -> None:
    """C1: ningun momento sale por debajo del minimo de palabras."""
    contenidas = _dentro(momento, palabras)
    assert len(contenidas) >= segmenter.MIN_WORDS, (
        f"C1: {len(contenidas)} palabras < MIN_WORDS {segmenter.MIN_WORDS}"
    )


def _c2_piso(momento: Moment, min_s: float | None) -> None:
    """C2: ningun momento sale por debajo del piso efectivo de duracion."""
    duracion = momento.end_s - momento.start_s
    piso = _piso(min_s)
    assert duracion >= piso - _TOLERANCIA_S, (
        f"C2: duracion {duracion:.4f} < piso {piso} (min_s={min_s})"
    )


def _c3_tope(momento: Moment, max_s: float | None) -> None:
    """C3: ningun momento sale por encima del tope efectivo de duracion."""
    duracion = momento.end_s - momento.start_s
    tope = _tope(max_s)
    assert duracion <= tope + _TOLERANCIA_S, (
        f"C3: duracion {duracion:.4f} > tope {tope} (max_s={max_s})"
    )


def _c4_disjuntos(momentos: Sequence[Moment], palabras: Sequence[Word]) -> None:
    """C4: los momentos no se solapan ni en tiempo ni en palabras."""
    for anterior, siguiente in pairwise(momentos):
        assert anterior.end_s <= siguiente.start_s + _TOLERANCIA_S, (
            f"C4: [{anterior.start_s}, {anterior.end_s}] solapa "
            f"[{siguiente.start_s}, {siguiente.end_s}]"
        )
    # Por indice ademas: ningun token_id aparece en dos momentos. El solape
    # temporal no lo detectaria si dos momentos compartieran palabras pero
    # tuvieran ventanas disjuntas, que es justo lo que C6 forbids.
    vistos: set[int] = set()
    for momento in momentos:
        ids = {p.token_id for p in _dentro(momento, palabras)}
        assert not (ids & vistos), f"C4: solape de palabras {sorted(ids & vistos)}"
        vistos |= ids


def _c5_cronologicos(momentos: Sequence[Moment]) -> None:
    """C5: `start_s` crece estrictamente de un momento al siguiente."""
    for anterior, siguiente in pairwise(momentos):
        assert siguiente.start_s > anterior.start_s, (
            f"C5: start_s no crece: {anterior.start_s} -> {siguiente.start_s}"
        )


def _c6_fronteras_en_palabra(momento: Moment, palabras: Sequence[Word]) -> None:
    """C6: las dos fronteras del momento caen en fronteras de palabra."""
    inicios = {p.start_s for p in palabras}
    finales = {p.end_s for p in palabras}
    assert momento.start_s in inicios, f"C6: inicio {momento.start_s} no es inicio de palabra"
    assert momento.end_s in finales, f"C6: final {momento.end_s} no es final de palabra"


def _c7_techo(momentos: Sequence[Moment]) -> None:
    """C7: no se emiten mas momentos de los que el techo permite."""
    assert len(momentos) <= segmenter.MAX_MOMENTS, (
        f"C7: {len(momentos)} momentos > MAX_MOMENTS {segmenter.MAX_MOMENTS}"
    )


def test_c1_cada_momento_tiene_suficientes_palabras() -> None:
    """C1 sobre entradas generadas, en todas las bandas de contrato."""
    for palabras, (c_min, c_max) in _corridas(20261004, 60):
        resultado = segmenter.segment_by_speech_rate(palabras, min_s=c_min, max_s=c_max)
        for momento in resultado.moments:
            _ = _c1_palabras(momento, palabras)


def test_c2_ningun_momento_baja_del_piso_efectivo() -> None:
    """C2: el piso de duracion es DURO, no una preferencia.

    La regla de descarte del segmenter es un AND, de modo que un tramo corto se
    descarta aunque tenga palabras de sobra. Este test lo fija para que nadie
    lea el AND al reves y empiece a emitir clips por debajo del piso.
    """
    for palabras, (c_min, c_max) in _corridas(20261004, 60):
        resultado = segmenter.segment_by_speech_rate(palabras, min_s=c_min, max_s=c_max)
        for momento in resultado.moments:
            _ = _c2_piso(momento, c_min)


def test_c3_ningun_momento_supera_el_tope_efectivo() -> None:
    """C3: el tope manda el del modulo cuando el contrato calla."""
    for palabras, (c_min, c_max) in _corridas(20261004, 60):
        resultado = segmenter.segment_by_speech_rate(palabras, min_s=c_min, max_s=c_max)
        for momento in resultado.moments:
            _ = _c3_tope(momento, c_max)


def test_c4_los_momentos_son_disjuntos() -> None:
    """C4: disjuntos en el tiempo y en las palabras."""
    for palabras, (c_min, c_max) in _corridas(777, 60):
        resultado = segmenter.segment_by_speech_rate(palabras, min_s=c_min, max_s=c_max)
        _ = _c4_disjuntos(resultado.moments, palabras)


def test_c5_los_momentos_van_en_orden_cronologico() -> None:
    """C5: el score ordena para ELEGIR, no para EMITIR.

    El segmenter ordena por interes (`_interes`) para quedarse con los mejores y
    despues vuelve a cronologiar la salida. Si esa segunda ordenacion desapareciera,
    los momentos seguirian siendo validos uno a uno y la lista seria ilegible;
    por eso la clausula va sobre la lista, no sobre cada momento.
    """
    for palabras, (c_min, c_max) in _corridas(4242, 60):
        resultado = segmenter.segment_by_speech_rate(palabras, min_s=c_min, max_s=c_max)
        _ = _c5_cronologicos(resultado.moments)


def test_c6_las_fronteras_caen_en_fronteras_de_palabra() -> None:
    """C6: ningun corte cae en mitad de una palabra."""
    for palabras, (c_min, c_max) in _corridas(31337, 60):
        resultado = segmenter.segment_by_speech_rate(palabras, min_s=c_min, max_s=c_max)
        for momento in resultado.moments:
            _ = _c6_fronteras_en_palabra(momento, palabras)


def test_c7_no_se_supera_el_techo_de_momentos() -> None:
    """C7: el techo es un contrato con el consumidor, no una preferencia."""
    for palabras, (c_min, c_max) in _corridas(99, 60):
        resultado = segmenter.segment_by_speech_rate(palabras, min_s=c_min, max_s=c_max)
        _ = _c7_techo(resultado.moments)


def test_la_salida_siempre_lleva_metadatos_y_score_en_rango() -> None:
    """Lo que no es clausula pero tampoco puede fallar: identidad y score.

    Cada momento se construye desde `tramos`, asi que su `source` no es una
    decision, y su score se normaliza por la densidad maxima de la propia corrida,
    asi que cae en 0..1.
    """
    for palabras, (c_min, c_max) in _corridas(555, 40):
        resultado = segmenter.segment_by_speech_rate(palabras, min_s=c_min, max_s=c_max)
        for momento in resultado.moments:
            assert momento.source is MomentSource.SPEECH_RATE
            assert 0.0 <= momento.score <= 1.0


def test_la_salida_es_determinista_sobre_entradas_generadas() -> None:
    for palabras, _rango in _corridas(31337, 30):
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


def test_la_bateria_de_propiedades_cubre_todas_las_clausulas() -> None:
    """Siete clausulas, siete tests. Este test es el que lo impide.

    Una clausula nueva escrita en el docstring y no comprobada es DOCUMENTACION,
    no una propiedad: pasaria el tiempo sin existir.

    QUE GARANTIZA Y QUE NO, para no vender mas de lo que comprueba: garantiza que
    cada clausula de la tabla tiene su test nombrado, que ese test existe en el
    modulo, y que el cuerpo del test invoca de verdad a la funcion que comprueba
    la clausula. NO garantiza que la asercion del test sea real: un test que
    llama a la comprobadora pero descarta el resultado (`_ = comprobar(...)`)
    pasaria aqui. Eso lo atrapa el revisor y las mutaciones, no este gate.
    """
    # Clausula -> (funcion que la comprueba, nombre del test que la invoca).
    clausulas = {
        "C1": (_c1_palabras, "test_c1_cada_momento_tiene_suficientes_palabras"),
        "C2": (_c2_piso, "test_c2_ningun_momento_baja_del_piso_efectivo"),
        "C3": (_c3_tope, "test_c3_ningun_momento_supera_el_tope_efectivo"),
        "C4": (_c4_disjuntos, "test_c4_los_momentos_son_disjuntos"),
        "C5": (_c5_cronologicos, "test_c5_los_momentos_van_en_orden_cronologico"),
        "C6": (_c6_fronteras_en_palabra, "test_c6_las_fronteras_caen_en_fronteras_de_palabra"),
        "C7": (_c7_techo, "test_c7_no_se_supera_el_techo_de_momentos"),
    }
    assert len(clausulas) == 7, f"se esperaban 7 clausulas, hay {len(clausulas)}"
    fuente = Path(__file__).read_text(encoding="utf-8")
    for clave, (comprobador, nombre_test) in clausulas.items():
        assert nombre_test in globals(), f"{clave} no tiene su test ({nombre_test})"
        cuerpo = _cuerpo_de(nombre_test, fuente)
        assert comprobador.__name__ in cuerpo, (
            f"{clave}: el test {nombre_test} no llama a {comprobador.__name__}"
        )


def _cuerpo_de(nombre_test: str, fuente: str) -> str:
    """El bloque de un test, desde su `def` hasta el siguiente `def` de nivel 0.

    Args:
        nombre_test: Nombre de la funcion de test.
        fuente: El texto completo del modulo.

    Returns:
        El cuerpo del test, o la cadena vacia si no aparece en la fuente.
    """
    inicio = fuente.find(f"def {nombre_test}(")
    if inicio < 0:
        return ""
    siguiente = fuente.find("\ndef ", inicio + 1)
    return fuente[inicio : siguiente if siguiente > 0 else len(fuente)]
