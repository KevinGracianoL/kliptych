"""El segmentador local determinista: de palabras a momentos.

Es una funcion PURA. No abre ficheros, no llama a ffprobe, no lee el reloj y no
habla con la red. La misma entrada produce exactamente la misma salida.

Las reglas hacen trabajos distintos y por eso ninguna sobra:

- ``MAX_GAP_S`` SEGMENTA. Es la unica que crea fronteras: un silencio mayor que
  el maximo define donde ACABA el habla activa. Vive en ``_tramos``, que no
  aplica ningun suelo, y por eso las pruebas de su borde atacan ahi.
- ``MIN_SPEECH_RATE_WPS`` ORDENA. Coloca tramos, nunca los elimina.
- ``MIN_WORDS`` y el piso de duracion DESCARTAN. Es la unica regla que elimina,
  y descarta cuando falla CUALQUIERA de los dos.

Los dos suelos son DUROS, no mitades de una conjuncion. La conjuncion fue un
error, y lo muestran los datos reales de un VOD: "5 y 10." son 3 palabras en
2,0 s, y "diferencia es Diff." son 3 palabras en 7,2 s. Con AND ambos
sobrevivian, porque la duracion pasaba un suelo y las palabras no llegaban al
otro. Ninguno de los dos es un clip: son alguien escribiendo en una diapositiva
mientras whisper transcribe. No hay ninguna prueba que afirme la conjuncion:
se elimino, y solo el VOD real pudo detectarla.

El ``score`` es RELATIVO a la entrada, y es un ATRIBUTO, no la clave de orden. La
salida va en orden CRONOLOGICO: ordenar por score es decision del consumidor, y
con score las dos mitades de un tramo partido salian en orden arbitrario porque
sus densidades son casi iguales.

RESTRICCION COMPARTIDA, que ha costado dos pruebas seguidas: como la salida es
cronologica, ``momentos[0]`` es el tramo MAS TEMPRANO. Si ese tramo es ademas el
mas lento, su score es el MAS BAJO. O sea que en estas pruebas el score CRECE con
el tiempo, y desempaquetar por score en vez de por tiempo invierte la lista y hace
fallar la asercion por el motivo equivocado.
"""

from collections.abc import Sequence
from typing import TYPE_CHECKING, cast

import pytest

from kliptych import segmenter
from kliptych.moments import MomentSource
from kliptych.transcribe import Word

if TYPE_CHECKING:
    from collections.abc import Callable


def _private(nombre: str) -> object:
    """Accede a un simbolo privado del modulo, como en el resto de la suite.

    Args:
        nombre: El nombre del simbolo.

    Returns:
        El simbolo, tipado como `object` para que el `cast` sea explicito.
    """
    return cast("object", getattr(segmenter, nombre))


_tramos = cast("Callable[[tuple[Word, ...]], list[list[Word]]]", _private("_tramos"))
_piso = cast("Callable[[float | None], float]", _private("_piso_efectivo"))
_tope = cast("Callable[[float | None], float]", _private("_tope_efectivo"))


def _palabras(textos: Sequence[str], *, paso: float, inicio: float = 0.0) -> list[Word]:
    """Palabras con separacion uniforme.

    Args:
        textos: Los textos, en orden.
        paso: Separacion entre inicios de palabra consecutive.
        inicio: Comienzo de la primera palabra.

    Returns:
        Las palabras con su marca de tiempo.
    """
    return [
        Word(
            start_s=inicio + indice * paso,
            end_s=inicio + indice * paso + paso * 0.8,
            text=textos[indice],
            confidence=0.9,
            token_id=indice,
        )
        for indice in range(len(textos))
    ]


def _palabras_ritmadas(
    textos: Sequence[str], *, pasos: Sequence[float], inicio: float = 0.0
) -> list[Word]:
    """Palabras con separacion NO uniforme, palabra a palabra.

    Existe porque una fila uniforme nunca puede producir dos densidades
    distintas: al partir un tramo por la mitad, las dos mitades tienen la misma
    densidad por construccion y sus scores coinciden. Una prueba que necesita
    scores distintos tiene que usar separaciones distintas.

    Args:
        textos: Los textos, en orden.
        pasos: Una separacion por palabra.
        inicio: Comienzo de la primera palabra.

    Returns:
        Las palabras con su marca de tiempo.
    """
    palabras: list[Word] = []
    t = inicio
    for indice, texto in enumerate(textos):
        palabras.append(
            Word(
                start_s=t,
                end_s=t + pasos[indice] * 0.8,
                text=texto,
                confidence=0.9,
                token_id=indice,
            )
        )
        t += pasos[indice]
    return palabras


def _borde(fin_primero: float, inicio_segundo: float) -> list[Word]:
    """Dos palabras que cabotean un hueco elegido, rodeadas de apoyo.

    Se construye a mano porque el hueco tiene que caer en un valor concreto de
    coma flotante, y un helper que acumula separaciones no lo garantiza.

    Args:
        fin_primero: El `end_s` de la ultima palabra del primer tramo.
        inicio_segundo: El `start_s` de la primera palabra del segundo tramo.

    Returns:
        Seis palabras: cuatro antes del hueco y dos despues.
    """
    palabras = _palabras(["a0", "a1", "a2"], paso=0.1)
    palabras.append(Word(start_s=0.3, end_s=fin_primero, text="a3", confidence=0.9, token_id=3))
    palabras.append(
        Word(
            start_s=inicio_segundo,
            end_s=inicio_segundo + 0.1,
            text="b0",
            confidence=0.9,
            token_id=4,
        )
    )
    palabras.append(
        Word(
            start_s=inicio_segundo + 0.1,
            end_s=inicio_segundo + 0.2,
            text="b1",
            confidence=0.9,
            token_id=5,
        )
    )
    return palabras


def _tramos_de(*, fin_primero: float, inicio_segundo: float) -> list[list[str]]:
    """Trocea con `_tramos` y devuelve los textos de cada tramo.

    Args:
        fin_primero: El `end_s` de la ultima palabra del primer tramo.
        inicio_segundo: El `start_s` de la primera palabra del segundo tramo.

    Returns:
        Los tramos, como listas de textos.
    """
    return [[w.text for w in t] for t in _tramos(tuple(_borde(fin_primero, inicio_segundo)))]


# --------------------------------------------------------------------------
# A - EL BORDE DE MAX_GAP_S
#
# Las tres atacan a `_tramos`, que es donde vive la constante y que no aplica
# ningun suelo. A1 y A2 NO pueden matar la mutacion `>` -> `>=`, y conviene decir
# por que: con un epsilon de 1e-9 las dos comparaciones dan el mismo resultado
# (0.6 - 1e-9 no es > 0.6 ni >= 0.6; 0.6 + 1e-9 es las dos cosas). Solo un hueco
# cuyo doble sea EXACTAMENTE 0.6 las separa, y ese es A3.
# --------------------------------------------------------------------------


def test_a1_un_hueco_claramente_menor_que_el_maximo_no_corta() -> None:
    """Un epsilon por debajo del maximo no corta: expresa la intencion, no el azar."""
    tramos = _tramos_de(fin_primero=1.0, inicio_segundo=1.0 + (segmenter.MAX_GAP_S - 1e-9))
    assert len(tramos) == 1, tramos


def test_a2_un_hueco_claramente_mayor_que_el_maximo_si_corta() -> None:
    """Un epsilon por encima del maximo corta: la contraparte de A1."""
    tramos = _tramos_de(fin_primero=1.0, inicio_segundo=1.0 + (segmenter.MAX_GAP_S + 1e-9))
    assert len(tramos) == 2, tramos


def test_a3_el_borde_exactamente_igual_no_corta() -> None:
    """Un hueco cuyo doble es exactamente MAX_GAP_S no corta: el limite es estricto.

    Este es el UNICO test que puede matar `>` -> `>=`, y solo si el hueco que
    construye es EXACTAMENTE el valor vigente de ``MAX_GAP_S``.

    OJO con el historial, porque aqui hay un test que PASA y no prueba nada: la
    version anterior de este test razonaba sobre un umbral de 0,6 s y construia
    el borde restando ``1.0 - 0.4``. Cuando ``MAX_GAP_S`` se ajusto a 2,0 s, el
    test se quedo midiendo un hueco de 0,6 s contra un umbral de 2,0 s: cierto
    con `>` y cierto con `>=`, o sea muerto. La mutacion `>` -> `>=` se colaba
    con la bateria en verde (#75).

    Asi que el borde se construye contra la CONSTANTE, no contra un literal:
    ``1.0`` y ``3.0`` son exactos en IEEE754 y su diferencia es exactamente 2,0.
    Si ``MAX_GAP_S`` vuelve a moverse, hay que volver a comprobar que el nuevo
    valor es un diádico exacto; con un decimal como ``0.1`` la resta
    ``(1.0 + 0.1) - 1.0`` da ``0.10000000000000009``, la asercion de abajo falla
    a proposito y avisa de que el borde ya no se puede construir asi.
    """
    borde = (1.0 + segmenter.MAX_GAP_S) - 1.0
    assert pytest.approx(segmenter.MAX_GAP_S, abs=0.0) == borde, (
        "el borde tiene que caer en el doble exacto de MAX_GAP_S; si MAX_GAP_S deja "
        "de ser un diádico exacto, hay que construir el borde de otra forma"
    )
    tramos = _tramos_de(fin_primero=1.0, inicio_segundo=1.0 + segmenter.MAX_GAP_S)
    assert len(tramos) == 1, tramos


# --------------------------------------------------------------------------
# B - LOS DOS SUELOS SON DUROS
#
# B1 y B2 son la pareja que fija que se descarta si falla CUALQUIERA de los dos.
# Fallen juntas o no fallan nunca: si una pasara sola, la regla habria vuelto a
# ser una conjuncion.
# --------------------------------------------------------------------------


def test_b1_un_tramo_por_debajo_de_min_words_se_descarta_aunque_dure_lo_suficiente() -> None:
    """Tres palabras sobre 2,8 s: pasa el suelo de duracion y se descarta igual."""
    palabras = _palabras(["a", "b", "c"], paso=1.0)
    duracion = palabras[-1].end_s - palabras[0].start_s
    assert len(palabras) < segmenter.MIN_WORDS
    assert duracion >= segmenter.MIN_CLIP_S, duracion
    assert segmenter.segment_by_speech_rate(palabras).moments == ()


def test_b2_un_tramo_mas_corto_de_min_clip_s_se_descarta_aunque_tenga_palabras() -> None:
    """Seis palabras sobre 1,8 s: pasa el suelo de palabras y se descarta igual."""
    palabras = _palabras(["a", "b", "c", "d", "e", "f"], paso=0.3)
    duracion = palabras[-1].end_s - palabras[0].start_s
    assert len(palabras) >= segmenter.MIN_WORDS
    assert duracion < segmenter.MIN_CLIP_S, duracion
    assert segmenter.segment_by_speech_rate(palabras).moments == ()


def test_b3_un_tramo_que_pasa_los_dos_suelos_se_conserva() -> None:
    """Cinco palabras sobre 2,8 s: pasa los dos suelos y sale."""
    palabras = _palabras(["a", "b", "c", "d", "e"], paso=0.7)
    momentos = segmenter.segment_by_speech_rate(palabras)
    assert len(momentos) == 1, momentos
    assert momentos[0].start_s == pytest.approx(0.0)


def test_b4_el_min_s_del_contrato_sube_el_suelo_por_encima_del_propio() -> None:
    """Con min_s=8.0 un tramo de 5,3 s se descarta, y sin el se conserva."""
    palabras = _palabras(["a", "b", "c", "d", "e", "f", "g"], paso=0.8)
    duracion = palabras[-1].end_s - palabras[0].start_s
    assert 2.0 <= duracion < 8.0, duracion
    assert len(segmenter.segment_by_speech_rate(palabras)) == 1
    assert segmenter.segment_by_speech_rate(palabras, min_s=8.0).moments == ()
    assert _piso(None) == segmenter.MIN_CLIP_S
    assert _piso(8.0) == pytest.approx(8.0)


def test_b5_el_max_s_del_contrato_baja_el_tope_por_debajo_del_propio() -> None:
    """Con max_s=20.0 un tramo continuo de 40 s parte en 20, no en 60."""
    palabras = _palabras([f"w{i}" for i in range(80)], paso=0.5)
    momentos = tuple(
        sorted(segmenter.segment_by_speech_rate(palabras, max_s=20.0), key=lambda m: m.start_s)
    )
    assert len(momentos) == 2, momentos
    assert momentos[0].end_s == pytest.approx(20.0, abs=1.0)
    assert _tope(None) == segmenter.MAX_CLIP_S
    assert _tope(20.0) == pytest.approx(20.0)


# --------------------------------------------------------------------------
# C - AGRUPACION Y PARTIDA
# --------------------------------------------------------------------------


def test_c1_dos_tramos_separados_por_silencio_dan_dos_momentos() -> None:
    """Un silencio grande parte el habla activa en dos tramos."""
    palabras = _palabras(["a", "b", "c", "d"], paso=0.7)
    palabras += _palabras(["e", "f", "g", "h"], paso=0.7, inicio=5.0)
    momentos = segmenter.segment_by_speech_rate(palabras)
    assert len(momentos) == 2, momentos
    assert momentos[0].start_s == pytest.approx(0.0)
    assert momentos[1].start_s == pytest.approx(5.0)


def test_c2_un_tramo_de_90s_separte_en_el_tope_sin_solaparse_ni_dejar_hueco() -> None:
    """Un tramo continuo de 90 s no tiene silencios: el tope tiene que PARTIR."""
    palabras = _palabras([f"w{i}" for i in range(180)], paso=0.5)
    momentos = tuple(sorted(segmenter.segment_by_speech_rate(palabras), key=lambda m: m.start_s))
    assert len(momentos) == 2, momentos
    izquierda, derecha = momentos
    assert izquierda.end_s == pytest.approx(60.0, abs=1.0)
    assert derecha.start_s == pytest.approx(60.0, abs=1.0)


def test_c3_la_partida_prefiere_el_ultimo_signo_de_frase_de_la_ventana() -> None:
    """El corte busca un cierre de frase aunque no sea el mas cercano al tope.

    El signo va en el indice 110, que acaba sobre 55,4 s, dentro de la ventana
    [48, 60]. El limite de palabra mas proximo al tope es el 119, sobre 59,9 s, y
    NO tiene signo. Si la salida corta en 55,4 y no en 59,9, la preferencia esta
    viva; si corta en 59,9, el signo se esta ignorando y esto seria decoracion.
    """
    palabras = _palabras([f"w{i}" for i in range(180)], paso=0.5)
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
    assert momentos[0].end_s == pytest.approx(55.4, abs=0.05)
    assert momentos[0].end_s != pytest.approx(59.9, abs=0.05)


def test_c4_sin_signo_de_frase_la_partida_cae_en_el_limite_de_palabra_mas_cercano() -> None:
    """Sin puntuacion en la ventana, el corte usa el limite mas proximo al tope.

    Sin esta prueba "el limite de palabra mas proximo" seria decoracion: nadie
    sabria si el codigo lo calcula o si le pasa lo primero que encuentra.
    """
    palabras = _palabras([f"w{i}" for i in range(180)], paso=0.5)
    momentos = tuple(sorted(segmenter.segment_by_speech_rate(palabras), key=lambda m: m.start_s))
    assert len(momentos) == 2, momentos
    limites = [w.end_s for w in palabras]
    assert min(abs(momentos[0].end_s - limite) for limite in limites) < 0.3


def test_c5_las_dos_mitades_no_comparten_ninguna_palabra() -> None:
    """La particion es una PORTAZON: la mitad izquierda termina antes de empezar la derecha.

    La desigualdad es ESTRICTA a proposito. Con palabras contiguas el fin de una
    coincide con el inicio de la siguiente, asi que `>=` seria verdad incluso si
    las mitades se solaparan. Exigir `<` sobre los limites de palabra, y ademas
    que el corte coincida con un limite real, es lo que hace la afirmacion
    util.
    """
    palabras = _palabras([f"w{i}" for i in range(180)], paso=0.5)
    momentos = tuple(sorted(segmenter.segment_by_speech_rate(palabras), key=lambda m: m.start_s))
    izquierda, derecha = momentos
    assert izquierda.end_s < derecha.start_s
    assert izquierda.end_s in [w.end_s for w in palabras]
    assert derecha.start_s in [w.start_s for w in palabras]


# --------------------------------------------------------------------------
# D - ORDEN
# --------------------------------------------------------------------------


def test_d1_la_salida_va_en_orden_cronologico_aunque_los_scores_empaten() -> None:
    """Test DEMOTADO a proposito: afirma el orden, pero no puede fallar por el orden.

    Cuando dos tramos tienen el mismo score, ``start_s`` es el desempate, y las DOS
    ordenaciones producen la misma lista. Aqui no hay nada que distinguir, y por
    eso la prueba se conserva pero no carga con el peso: D2, D3 y D4 si lo
    cargan, con scores que de verdad difieren.

    La igualdad EXACTA de score estaria mal aqui. Las dos densidades se calculan
    por separado sobre spans cuyas duraciones son 2,6599999999999997 y
    2,6599999999999999: coinciden hasta el ultimo bit y no son el mismo double.
    Comparar con `==` seria exigir algo que el codigo correcto no puede dar. Es
    la quinta vez que un borde de coma flotante muerde en este proyecto.
    """
    palabras = _palabras(["a", "b", "c", "d"], paso=0.7)
    palabras += _palabras(["e", "f", "g", "h"], paso=0.7, inicio=5.0)
    momentos = segmenter.segment_by_speech_rate(palabras)
    assert len(momentos) == 2, momentos
    assert momentos[0].score == pytest.approx(momentos[1].score, abs=1e-9)
    assert momentos[0].start_s < momentos[1].start_s


def test_d2_el_orden_es_cronologico_con_densidades_distintas_sin_partir() -> None:
    """Sujeto UNICO: el orden. Fixture POR DEBAJO del tope, asi que no hay partida.

    60 palabras a 0,7 (42 s) y 60 a 0,25 (15 s) dan 57 s: un solo tramo, sin
    silencedos, y dos densidades de verdad distintas. Al no pasar por
    ``MAX_CLIP_S`` este test no puede fallar por la partida, y una falla de aqui
    es una falla de ORDEN y solo de orden.

    Antes este sujeto venía acompañado de una particion, y un fallo del tope
    parecía un fallo de orden. Separarlos es lo que hace que eso no vuelva.
    """
    pasos = [0.7] * 60 + [0.25] * 60
    palabras = _palabras_ritmadas([f"w{i}" for i in range(120)], pasos=pasos)
    momentos = segmenter.segment_by_speech_rate(palabras)
    assert len(momentos) == 1, f"esta fixture no debe partir, queda bajo el tope: {momentos}"
    assert momentos[0].end_s - momentos[0].start_s < segmenter.MAX_CLIP_S


def test_d4_la_partida_produce_mitades_con_scores_distintos() -> None:
    """Sujeto UNICO: una particion produce mitades con densidades distintas.

    Fixture de 72 s, que supera ``MAX_CLIP_S`` y por tanto se parte. La salida
    medida es ``[(0.0, 59.95, 0.5482), (60.0, 71.95, 1.0)]``. Se afirma PRIMERO
    que los scores difieren y DESPUES el orden, para que la prueba no pueda pasar
    en vazio, y se afirma la contiguedad con desigualdad ESTRICTA.
    """
    pasos = [0.7] * 60 + [0.25] * 120
    palabras = _palabras_ritmadas([f"w{i}" for i in range(180)], pasos=pasos)
    momentos = tuple(sorted(segmenter.segment_by_speech_rate(palabras), key=lambda m: m.start_s))
    assert len(momentos) == 2, momentos
    assert momentos[0].score != momentos[1].score, "si empataran, la prueba no diria nada"
    assert momentos[0].start_s < momentos[1].start_s
    assert momentos[0].end_s < momentos[1].start_s


def test_d3_ordenar_por_score_moveria_el_tramo_rapido_al_principio() -> None:
    """El tramo rapido va al FINAL, que es justo lo que el orden por score impediria.

    Es el caso que de verdad distingue las dos ordenaciones: con score, el tramo
    rapido pasaria delante; con tiempo, se queda donde esta.

    El tramo rapido usa 0,3 y no 0,2 a proposito. Con 0,2 mide 1,56 s y el suelo
    duro de duracion lo descarta, y el test se pasaria comprobando un unico
    momento sin comprobar nada del orden. Con 0,3 mide 2,34 s, sobrevive a los
    dos suelos, y los scores quedan en 0,25 y 1,0: ordenar por score moveria el
    tramo rapido y tardio al principio.
    """
    lento = _palabras_ritmadas(
        ["a", "b", "c", "d", "e", "f", "g", "h"], pasos=[1.2] * 8, inicio=0.0
    )
    rapido = _palabras_ritmadas(
        ["i", "j", "k", "l", "m", "n", "o", "p"], pasos=[0.3] * 8, inicio=20.0
    )
    momentos = segmenter.segment_by_speech_rate(lento + rapido)
    assert len(momentos) == 2, momentos
    assert momentos[0].start_s == pytest.approx(0.0)
    assert momentos[1].start_s == pytest.approx(20.0)
    assert momentos[0].score < momentos[1].score, "el lento debe puntuar menos"


# --------------------------------------------------------------------------
# E - SCORE
# --------------------------------------------------------------------------


def test_e1_el_score_es_la_densidad_relativa_al_tramo_mas_rapido() -> None:
    """Score = palabras_por_segundo_del_tramo / las_del_tramo_mas_rapido.

    Aritmetica del ejemplo, con dos tramos:

        tramo A: 12 palabras sobre 6,0 s -> 2,00 p/s
        tramo B: 30 palabras sobre 10,0 s -> 3,00 p/s   <- el mas rapido
        score(A) = 2,00 / 3,00 = 0,667
        score(B) = 3,00 / 3,00 = 1,000

    No hay constante que ajustar y queda acotado en 0..1 por construccion. Como
    depende del resto de la entrada, dos tramos con la misma densidad puntuan
    igual DENTRO de una entrada, pero el mismo tramo puede puntuar distinto en
    entradas distintas. El gate nunca lo lee y aqui solo importa el orden.
    """
    palabras = _palabras_ritmadas([f"a{i}" for i in range(12)], pasos=[0.5] * 12, inicio=0.0)
    palabras += _palabras_ritmadas([f"b{i}" for i in range(30)], pasos=[1 / 3] * 30, inicio=20.0)
    momentos = segmenter.segment_by_speech_rate(palabras)
    assert len(momentos) == 2, momentos
    # Se desempaqueta POR TIEMPO, no por score: la salida es cronologica, asi que
    # momentos[0] es el primer grupo, que es el mas LENTO. Desempaquetar por score
    # daria el orden invertido y haria fallar la prueba sin motivo.
    temprano, denso = momentos[0], momentos[1]
    assert temprano.score == pytest.approx(2.0 / 3.0, abs=0.02)
    assert denso.score == pytest.approx(1.0)
    assert all(0.0 <= m.score <= 1.0 for m in momentos)


def test_e2_un_solo_tramo_punta_exactamente_uno() -> None:
    """Con un unico tramo, el mas rapido es el mismo, y su score es 1,0."""
    palabras = _palabras(["a", "b", "c", "d", "e"], paso=0.7)
    momentos = segmenter.segment_by_speech_rate(palabras)
    assert len(momentos) == 1, momentos
    assert momentos[0].score == pytest.approx(1.0)


def test_e3_el_score_no_es_el_orden_de_salida() -> None:
    """Sujeto: el score es un atributo y la salida va por tiempo.

    La DIRECCION importa y es la que se equivoca al escribir estas pruebas. La
    salida es cronologica, de modo que ``momentos[0]`` es el tramo MAS LENTO y
    por tanto lleva el score MAS BAJO: aqui el score CRECE con el tiempo. Si se
    ordenara por score, esta misma lista saldria invertida.
    """
    palabras = _palabras(["a", "b", "c", "d", "e", "f"], paso=0.8, inicio=0.0)
    palabras += _palabras(["g", "h", "i", "j", "k", "l"], paso=0.7, inicio=20.0)
    momentos = segmenter.segment_by_speech_rate(palabras)
    assert len(momentos) == 2, momentos
    assert momentos[0].score < momentos[1].score, "el primero es el mas lento"
    assert [m.start_s for m in momentos] == [0.0, pytest.approx(20.0)]


# --------------------------------------------------------------------------
# F - RESERVAS Y ENTRADAS DEGRADADAS
# --------------------------------------------------------------------------


def test_f1_entrada_vacia_da_tupla_vacia() -> None:
    assert segmenter.segment_by_speech_rate([]).moments == ()


def test_f2_una_sola_palabra_se_descarta() -> None:
    """Una palabra no llega al suelo de palabras, por muy larga que sea."""
    palabras = [Word(start_s=0.0, end_s=9.0, text="sola", confidence=0.9, token_id=0)]
    assert segmenter.segment_by_speech_rate(palabras).moments == ()


def test_f3_las_palabras_desordenadas_se_ordenan_sin_mutar_la_entrada() -> None:
    """Comportamiento fijado: se ordena una COPIA y la entrada no se toca.

    Ordenar la entrada seria un efecto secundario invisible para quien llama, y
    aqui la lista es la del segmento: si se mutara, dos llamadas distintas sobre
    el mismo objeto darian resultados distintos.
    """
    palabras = _palabras(["a", "b", "c", "d", "e"], paso=0.7)
    revueltas = [palabras[4], palabras[2], palabras[0], palabras[3], palabras[1]]
    antes = [(w.start_s, w.end_s, w.text) for w in revueltas]
    momentos = segmenter.segment_by_speech_rate(revueltas)
    despues = [(w.start_s, w.end_s, w.text) for w in revueltas]
    assert len(momentos) == 1, momentos
    assert momentos[0].start_s == pytest.approx(0.0)
    assert antes == despues, "la entrada no debe mutarse"


_PESOS = "_DEFAULT_WEIGHTS"


def test_la_fuente_nueva_no_participa_en_la_fusion() -> None:
    """`SPEECH_RATE` no tiene peso en `_DEFAULT_WEIGHTS`, y eso es deliberado.

    `_fuse` usa `weights.get(source, 0.0)`: si esta fuente pasara por ahi, su
    score valdria cero y el ranking seria arbitrario SIN NINGUN ERROR. Anadir un
    peso sin medir seria inventar el valor de una señal que todavia no participa
    en la fusion.
    """
    pesos = cast(
        "dict[str, float]",
        __import__("kliptych.moments", fromlist=["x"])._DEFAULT_WEIGHTS,
    )
    assert MomentSource.SPEECH_RATE not in pesos
    assert set(pesos) == {
        MomentSource.SCENE,
        MomentSource.AUDIO_ENERGY,
        MomentSource.CHAT_DENSITY,
    }
    assert MomentSource.SPEECH_RATE.value == "speech_rate"
