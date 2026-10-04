"""La justificacion que ve el operador tiene que LEERSE bien con cualquier numero.

Es texto generado, y un texto generado con recuentos lleva concordancia de
genero y de numero. Esa concordancia no se puede escribir a mano sin fallar:
"1 particiones" es exactamente lo que salia.

Por eso la forma es ETIQUETA: VALOR. No tiene genero ni numero en ningun idioma
y no se rompe con 0, con 1 ni con 2. Estos tests fijan las tres cadenas
EXACTAS: si alguien devuelve la forma de frase, que es la que se ha probado que
falla, mueren estos tres.
"""

from typing import TYPE_CHECKING, cast

from kliptych import orchestrator as orch
from kliptych.moments import Moment, MomentSource
from kliptych.segmenter import SpeechSegmentResult

if TYPE_CHECKING:
    from collections.abc import Callable

_RAZON = "_rationale" + "_segmentador"


def _rationale(resultado: SpeechSegmentResult) -> str:
    """Invoca la justificacion privada del orquestador.

    Args:
        resultado: El resultado que el orquestador registraria.

    Returns:
        La cadena que va al manifest.
    """
    nombre = _RAZON
    return cast("Callable[[SpeechSegmentResult], str]", getattr(orch, nombre))(resultado)


def _resultado(entregados: int, calificados: int, suelos: int, split: int) -> SpeechSegmentResult:
    """Un resultado con los recuentos que interesan y momento ninguno.

    La forma de la cadena depende SOLO de los recuentos, que es justo lo que
    estos tests necesitan comprobar.

    Args:
        entregados: Cuantos momentos se entregan.
        calificados: Cuantos pasaron los suelos.
        suelos: Cuantos murieron en un suelo.
        split: Cuantos clips se recortaron.

    Returns:
        El resultado con esos recuentos.
    """
    momentos = tuple(
        Moment(start_s=float(i), end_s=float(i) + 1.0, score=1.0, source=MomentSource.SPEECH_RATE)
        for i in range(entregados)
    )
    return SpeechSegmentResult(
        moments=momentos,
        qualifying=calificados,
        discarded_by_floor=suelos,
        split=split,
    )


def test_la_justificacion_se_lee_bien_con_un_solo_calificado() -> None:
    """Un solo calificado: el caso donde una frase diria "1 tramos"."""
    texto = _rationale(_resultado(entregados=1, calificados=1, suelos=0, split=0))

    assert texto == (
        "segmentador local determinista; tramos por palabras por segundo y "
        "silencios; entregados: 1; calificados: 1; descartados por el techo: 0; "
        "descartados por los suelos: 0; particiones por duracion: 0"
    )


def test_la_justificacion_se_lee_bien_con_dos_calificados() -> None:
    """Dos calificados y un recorte: donde una frase diria "1 particiones"."""
    texto = _rationale(_resultado(entregados=2, calificados=2, suelos=0, split=1))

    assert texto == (
        "segmentador local determinista; tramos por palabras por segundo y "
        "silencios; entregados: 2; calificados: 2; descartados por el techo: 0; "
        "descartados por los suelos: 0; particiones por duracion: 1"
    )


def test_la_justificacion_se_lee_bien_con_recuentos_a_cero() -> None:
    """Todo a cero es donde una frase se rompe al plural, y aqui no."""
    texto = _rationale(_resultado(entregados=0, calificados=0, suelos=0, split=0))

    assert texto == (
        "segmentador local determinista; tramos por palabras por segundo y "
        "silencios; entregados: 0; calificados: 0; descartados por el techo: 0; "
        "descartados por los suelos: 0; particiones por duracion: 0"
    )


def test_la_justificacion_no_tiene_una_frase_con_el_numero_dentro() -> None:
    """Ningun recuento va pegado a un sustantivo, que es donde nace el error.

    Este es el test que protege la FORMA y no un caso concreto: si alguien
    reescribe la cadena como "1 particiones", este falla aunque los tres casos
    concretos de arriba se ajusten.
    """
    texto = _rationale(_resultado(entregados=1, calificados=1, suelos=0, split=1))

    assert "particiones por duracion: 1" in texto
    # La forma rota, la que se ha probado que falla: el numero pegado al
    # sustantivo. No debe aparecer, que es exactamente lo que sale al releerlo.
    assert "duracion: 1 particiones" not in texto
    assert "duracion: 1 particion," not in texto
