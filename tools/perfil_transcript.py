"""Perfil de material: audita transcripts y tabula mecanismos de corte.

Instrumento de medicion para #72. NO es codigo de producto: no toca el
segmentador, ni el gate, ni el contrato, ni la CLI. Su unico producto es
`docs/perfil_transcript.md`, que escribe entero.

## Comando exacto

    uv run python tools/perfil_transcript.py

Lee los transcripts que ya estan en disco, produce el informe en
`docs/perfil_transcript.md` y sale distinto de cero si un transcript no se
puede leer. No escribe en ninguna parte mas.

## Que mide y por que en este orden

La pregunta que ordena todo es si la transcripcion es confiable, porque un
histograma de huecos construido sobre timestamps que retroceden, o sobre
alucinaciones, no describe el audio: describe el modelo. Por eso la auditoria va
PRIMERO y el resto se marca como condicional a su veredicto.

Las alucinaciones de Whisper no se buscan solo por frase. El campo
`confidence` las separa mejor: una frase inventada sobre silencio tiene
confianza baja y el resto del transcript no. Por eso el informe reporta las dos
cosas y las cruza.

## Lo que NO puede medir, y no estima

El delta de energia necesita audio. `astats` lee muestras, y aqui no hay
muestras: solo los word timestamps que Whisper produjo. Cualquier numero de
energia aqui seria inventado, asi que no hay ninguno.
"""

from __future__ import annotations

import json
import statistics as st
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Final, cast

if TYPE_CHECKING:
    from collections.abc import Sequence

RAIZ: Final[Path] = Path(__file__).resolve().parents[1]
INFORME: Final[Path] = RAIZ / "docs" / "perfil_transcript.md"

# Rutas RELATIVAS y sin nombre de canal. El informe se commitea; un nombre de
# canal lo identificaria. Solo el nombre del directorio de trabajo.
TRANSCRIPTS: Final[tuple[tuple[str, str], ...]] = (
    ("kick_test", "runs/kick_test/package/delivery-work/transcript.json"),
    ("youtube_test", "runs/youtube_test/package/delivery-work/transcript.json"),
)

# Frases canonicas de alucinacion de Whisper, buscadas como SECUENCIA de
# palabras. "amigos" y "merci" son palabras reales del vocabulario: contarlas
# sueltas seria contar las veces que alguien dijo "amigos", no las veces que
# Whisper lo invento.
FRASES_CANONICAS: Final[tuple[tuple[str, ...], ...]] = (
    ("gracias", "por", "ver"),
    ("gracias", "por", "ver", "el", "video"),
    ("subtitulos", "subtitulos", "realizados", "por"),
    ("suscribete",),
    ("traduccion",),
    ("sous", "titres"),
    ("merci",),
    ("thank", "you", "for", "watching"),
    ("subtitles", "by"),
    ("amigos", "de", "youtube"),
)

PUNTUACION_CIERRE: Final[frozenset[str]] = frozenset({".", "?", "!"})
PUNTUACION_TODAS: Final[frozenset[str]] = frozenset({".", "?", "!", ",", ";", ":"})

CUBOS_HUECO: Final[tuple[tuple[str, float, float | None], ...]] = (
    ("<0.2", 0.0, 0.2),
    ("0.2-0.5", 0.2, 0.5),
    ("0.5-1.0", 0.5, 1.0),
    ("1.0-1.5", 1.0, 1.5),
    ("1.5-2.0", 1.5, 2.0),
    ("2.0-3.0", 2.0, 3.0),
    (">3.0", 3.0, None),
)

MAX_GAP_CANDIDATOS: Final[tuple[float, ...]] = (0.6, 1.0, 1.5, 2.0, 3.0)
PUNCT_BREAK_CANDIDATOS: Final[tuple[float | None, ...]] = (
    10.0,
    15.0,
    20.0,
    30.0,
    45.0,
    None,
)

# Duracion mínima de un tramo para contar como clip. Es el criterio del
# operador, no una preferencia de este script.
MIN_CLIP_S: Final[float] = 8.0

# Umbral de hueco que separa "habla transcripta" de "tiempo sin voz", tal como
# lo define #72 al abrir este trabajo. Es el corte del reporte, no un ajuste.
HUECO_SIN_VOZ_S: Final[float] = 1.0

# Umbral de confianza bajo el cual una palabra se considera sospechosa. Es el
# mismo corte que usa faster-whisper para marcar palabras poco creibles, y se
# aplica aqui por eso y no por costumbre.
CONF_SOSPECHA: Final[float] = 0.3

# Cuantas ocurrencias de frase canonica convierten un transcript en "parcialmente
# confiable". Por debajo de esto es ruido de vocabulario; por encima es un bucle
# de alucinacion, porque la frase es completa y se repite.
ALUCINACION_REPETIDA: Final[int] = 5


@dataclass(frozen=True)
class Palabra:
    """Una palabra del transcript, con lo que el segmentador necesita de ella."""

    indice: int
    inicio: float
    fin: float
    texto: str
    confianza: float

    @property
    def duracion(self) -> float:
        """Cuanto dura la palabra, segun los timestamps de Whisper."""
        return self.fin - self.inicio


@dataclass(frozen=True)
class Transcript:
    """Un transcript cargado, con su procedencia y sus palabras en orden."""

    nombre: str
    ruta_relativa: str
    idioma: str
    duracion_s: float
    palabras: tuple[Palabra, ...]

    @property
    def huecos(self) -> tuple[float, ...]:
        """Silencio medido entre el fin de una palabra y el inicio de la otra."""
        return tuple(
            self.palabras[i + 1].inicio - self.palabras[i].fin
            for i in range(len(self.palabras) - 1)
        )


@dataclass
class Auditoria:
    """Lo que se aprendio de un transcript antes de tabular nada."""

    nombre: str
    n_palabras: int
    duracion_s: float
    duracion_palabra_mediana: float
    duracion_palabra_sd: float
    duracion_palabra_cero: int
    duracion_palabra_larga: int
    huecos_negativos: int
    retrocesos: int
    ejemplos_retroceso: tuple[tuple[int, float, float], ...]
    outliers_cortas: int
    outliers_largas: int
    alucinaciones: dict[str, int]
    ejemplo_alucinacion: tuple[str, float, float] | None
    conf_baja: int
    conf_baja_duracion_s: float
    conf_baja_ejemplos: tuple[tuple[str, float, float], ...]
    puntuacion_cuentas: dict[str, int]
    frac_puntuacion: float
    n_cierres: int
    gaps_entre_cierres: tuple[float, ...] = field(default=())


@dataclass(frozen=True)
class FilaCruzada:
    """Una combinacion de los dos mecanismos, medida."""

    max_gap: float
    ventana: float | None
    n_tramos: int
    n_utiles: int
    segundos_utiles: float


def _como_mapa(valor: object) -> dict[str, object]:
    """Acota un valor JSON a un mapa de campo a valor.

    `json.loads` devuelve `Any` y el repo lo resuelve con `cast` (ver
    `git_proposals._json_object`). Aqui se mantiene el criterio y se anade el
    chequeo de forma explicito, para que un transcript con otra estructura
    falle con nombre en vez de con un error de subscript.

    Args:
        valor: El valor tal como salio de `json.loads`.

    Returns:
        El mapa.

    Raises:
        TypeError: Si el valor no es un objeto JSON.
    """
    if not isinstance(valor, dict):
        msg = f"se esperaba un objeto JSON y vino {type(valor).__name__}"
        raise TypeError(msg)
    origen = cast("dict[object, object]", valor)
    campos: dict[str, object] = {}
    for clave, dato in origen.items():
        campos[str(clave)] = dato
    return campos


def _como_lista(valor: object) -> list[object]:
    """Acota un valor JSON a una lista.

    Args:
        valor: El valor tal como salio de `json.loads`.

    Returns:
        La lista.

    Raises:
        TypeError: Si el valor no es una lista JSON.
    """
    if not isinstance(valor, list):
        msg = f"se esperaba una lista JSON y vino {type(valor).__name__}"
        raise TypeError(msg)
    return list(cast("list[object]", valor))


def _como_float(valor: object, campo: str) -> float:
    """Acota un valor JSON a float, con el nombre del campo en el error.

    Args:
        valor: El valor tal como salio de `json.loads`.
        campo: El nombre del campo, para que el error sea localizable.

    Returns:
        El valor como float.

    Raises:
        TypeError: Si el valor no es de un tipo numerico convertible.
        ValueError: Si el valor es una cadena que no se puede convertir.
    """
    if isinstance(valor, bool) or not isinstance(valor, int | float | str):
        msg = f"'{campo}' no es numerico: {type(valor).__name__}"
        raise TypeError(msg)
    try:
        return float(valor)
    except ValueError as error:
        msg = f"'{campo}' no es numerico: {valor!r}"
        raise ValueError(msg) from error


def cargar(nombre: str, ruta_relativa: str) -> Transcript:
    """Lee un `transcript.json` de faster-whisper.

    Args:
        nombre: Etiqueta corta del transcript.
        ruta_relativa: Ruta al JSON, relativa a la raiz del repositorio.

    Returns:
        El transcript con sus palabras en orden de archivo.

    Raises:
        ValueError: Si falta `words`, o si una palabra no trae sus campos.
    """
    ruta = RAIZ / ruta_relativa
    crudo = _como_mapa(cast("object", json.loads(ruta.read_text(encoding="utf-8"))))
    if "words" not in crudo:
        msg = f"{ruta_relativa}: el transcript no trae 'words'"
        raise ValueError(msg)
    palabras: list[Palabra] = []
    for indice, entrada in enumerate(_como_lista(crudo["words"])):
        fila = _como_mapa(entrada)
        for campo in ("start_s", "end_s", "text", "confidence"):
            if campo not in fila:
                msg = f"{ruta_relativa}: palabra {indice} sin '{campo}'"
                raise ValueError(msg)
        palabras.append(
            Palabra(
                indice=indice,
                inicio=_como_float(fila["start_s"], "start_s"),
                fin=_como_float(fila["end_s"], "end_s"),
                texto=str(fila["text"]),
                confianza=_como_float(fila["confidence"], "confidence"),
            )
        )
    return Transcript(
        nombre=nombre,
        ruta_relativa=ruta_relativa,
        idioma=str(crudo.get("language", "?")),
        duracion_s=_como_float(crudo.get("duration_s", 0.0), "duration_s"),
        palabras=tuple(palabras),
    )


def _normalizada(palabra: Palabra) -> str:
    """El texto en minusculas y sin puntuacion de borde, para comparar.

    Args:
        palabra: La palabra a normalizar.

    Returns:
        El texto normalizado.
    """
    return palabra.texto.lower().strip(" .!?,")


def contar_alucinaciones(transcript: Transcript) -> dict[str, int]:
    """Cuenta las frases canonicas como SECUENCIA contigua de palabras.

    Args:
        transcript: El transcript a auditar.

    Returns:
        Frase -> ocurrencias, solo las que aparecen al menos una vez.
    """
    palabras = [_normalizada(p) for p in transcript.palabras]
    conteo: dict[str, int] = {}
    for frase in FRASES_CANONICAS:
        n = len(frase)
        ocurrencias = sum(
            1 for i in range(len(palabras) - n + 1) if tuple(palabras[i : i + n]) == frase
        )
        if ocurrencias:
            conteo[" ".join(frase)] = ocurrencias
    return conteo


def primer_ejemplo_alucinacion(transcript: Transcript) -> tuple[str, float, float] | None:
    """El primer bloque de frase canonica, con su tiempo y su confianza.

    Args:
        transcript: El transcript a auditar.

    Returns:
        `(texto, inicio_s, confianza_de_la_primera_palabra)`, o `None`.
    """
    palabras = [_normalizada(p) for p in transcript.palabras]
    for frase in FRASES_CANONICAS:
        n = len(frase)
        for i in range(len(palabras) - n + 1):
            if tuple(palabras[i : i + n]) == frase:
                texto = " ".join(frase)
                return texto, transcript.palabras[i].inicio, transcript.palabras[i].confianza
    return None


def auditar(transcript: Transcript) -> Auditoria:
    """Mide integridad del transcript. Va antes que cualquier histograma.

    Args:
        transcript: El transcript a auditar.

    Returns:
        La auditoria completa, con los contadores y los ejemplos que la
        sostienen.
    """
    palabras = transcript.palabras
    duraciones = [p.duracion for p in palabras]
    mediana = st.median(duraciones)
    desviacion = st.pstdev(duraciones)

    retrocesos = [
        (p.indice, p.inicio, palabras[p.indice - 1].inicio)
        for p in palabras[1:]
        if p.inicio < palabras[p.indice - 1].inicio
    ]

    conf_baja = [p for p in palabras if p.confianza < CONF_SOSPECHA]

    cuentas: dict[str, int] = {}
    con_puntuacion = 0
    for p in palabras:
        if p.texto and p.texto[-1] in PUNTUACION_TODAS:
            con_puntuacion += 1
            signo = p.texto[-1]
            cuentas[signo] = cuentas.get(signo, 0) + 1

    cierres = [p for p in palabras if p.texto and p.texto[-1] in PUNTUACION_CIERRE]
    gaps_cierres = tuple(cierres[k + 1].inicio - cierres[k].fin for k in range(len(cierres) - 1))

    return Auditoria(
        nombre=transcript.nombre,
        n_palabras=len(palabras),
        duracion_s=transcript.duracion_s,
        duracion_palabra_mediana=mediana,
        duracion_palabra_sd=desviacion,
        duracion_palabra_cero=sum(1 for v in duraciones if v <= 0.0),
        duracion_palabra_larga=sum(1 for v in duraciones if v >= 1.0),
        huecos_negativos=sum(1 for g in transcript.huecos if g < 0.0),
        retrocesos=len(retrocesos),
        ejemplos_retroceso=tuple(retrocesos[:5]),
        outliers_cortas=sum(1 for v in duraciones if v < mediana - 3 * desviacion),
        outliers_largas=sum(1 for v in duraciones if v > mediana + 3 * desviacion),
        alucinaciones=contar_alucinaciones(transcript),
        ejemplo_alucinacion=primer_ejemplo_alucinacion(transcript),
        conf_baja=len(conf_baja),
        conf_baja_duracion_s=sum(p.duracion for p in conf_baja),
        conf_baja_ejemplos=tuple((p.texto, p.inicio, p.confianza) for p in conf_baja[:8]),
        puntuacion_cuentas=cuentas,
        frac_puntuacion=con_puntuacion / len(palabras) if palabras else 0.0,
        n_cierres=len(cierres),
        gaps_entre_cierres=gaps_cierres,
    )


def veredicto(auditoria: Auditoria) -> str:
    """Veredicto de UNA linea sobre si la transcripcion sirve para este material.

    No es una suma ponderada de banderas: cada condicion decide algo distinto.
    Los retrocesos son disqualificantes porque invalidan el orden; las
    alucinaciones repetidas lo son porque inflan palabras y duracion a la vez;
    la confianza baja suelta no lo es, porque puede ser habla real difícil.

    Args:
        auditoria: La auditoria del transcript.

    Returns:
        `confiable`, `parcialmente confiable` o `no sirve`.
    """
    if auditoria.retrocesos > 0:
        return "no sirve"
    if sum(auditoria.alucinaciones.values()) >= ALUCINACION_REPETIDA:
        return "parcialmente confiable"
    return "confiable"


def histograma_huecos(huecos: Sequence[float]) -> list[tuple[str, int, float]]:
    """Reparte los huecos en cubos fijos, con cantidad y segundos.

    Args:
        huecos: Los silencios entre palabras.

    Returns:
        `(cubo, n, segundos)` por cubo, en el orden de `CUBOS_HUECO`.
    """
    filas: list[tuple[str, int, float]] = []
    for etiqueta, bajo, alto in CUBOS_HUECO:
        valores = [g for g in huecos if g >= bajo and (alto is None or g < alto)]
        filas.append((etiqueta, len(valores), sum(valores)))
    return filas


def tramos_por_silencio(
    inicio: float, fin: float, palabras: Sequence[Palabra], max_gap_s: float
) -> list[tuple[float, float]]:
    """Corta la linea por silencios mayores que `max_gap_s`.

    Reproduce la regla del segmentador: una palabra cuyo hueco con la anterior
    supera el umbral cierra el tramo y abre otro.

    Args:
        inicio: Segundo inicial del span.
        fin: Segundo final del span.
        palabras: Las palabras del span.
        max_gap_s: El umbral `MAX_GAP_S` que se evalua.

    Returns:
        Tramos `(inicio, fin)`. El ultimo llega hasta `fin`, para no perder cola.
    """
    tramos: list[tuple[float, float]] = []
    corte = inicio
    for i, palabra in enumerate(palabras):
        if i > 0 and palabra.inicio - palabras[i - 1].fin > max_gap_s:
            tramos.append((corte, palabras[i - 1].fin))
            corte = palabra.inicio
    if palabras:
        tramos.append((corte, fin))
    return tramos


def tramos_por_puntuacion(
    inicio: float, fin: float, palabras: Sequence[Palabra], ventana_s: float | None
) -> list[tuple[float, float]]:
    """Corta la linea por puntuacion, dentro de una ventana de duracion.

    Es el mecanismo (B) que #72 propone. La ventana es una cota de duracion
    DESDE el inicio del tramo: cuando se cumple, se espera al siguiente cierre
    de frase y ahi se corta. `ventana_s=None` corta en cada cierre.

    Args:
        inicio: Segundo inicial del span.
        fin: Segundo final del span.
        palabras: Las palabras del span.
        ventana_s: La ventana, o `None` para cortar en cada cierre.

    Returns:
        Tramos `(inicio, fin)`.
    """
    tramos: list[tuple[float, float]] = []
    corte = inicio
    for palabra in palabras:
        if not palabra.texto or palabra.texto[-1] not in PUNTUACION_CIERRE:
            continue
        if ventana_s is not None and palabra.fin - corte < ventana_s:
            continue
        tramos.append((corte, palabra.fin))
        corte = palabra.fin
    if corte < fin:
        tramos.append((corte, fin))
    return tramos


def resumen_tramos(tramos: Sequence[tuple[float, float]]) -> list[str]:
    """Describe un conjunto de tramos con los numeros que importan.

    Args:
        tramos: Los tramos medidos.

    Returns:
        `[n, min, mediana, max, capturada]` en segundos, o `["vacio"]`.
    """
    if not tramos:
        return ["vacio"]
    duraciones = sorted(f - i for i, f in tramos)
    return [
        str(len(tramos)),
        f"{duraciones[0]:.1f}",
        f"{st.median(duraciones):.1f}",
        f"{duraciones[-1]:.1f}",
        f"{sum(duraciones):.1f}",
    ]


def tramos_combinados(
    inicio: float,
    fin: float,
    palabras: Sequence[Palabra],
    max_gap_s: float,
    ventana_s: float | None,
) -> list[tuple[float, float]]:
    """Corta donde haya silencio grande O cierre de frase tras la ventana.

    Es la combinacion que #72 propone como (B): el silencio sigue cortando y la
    puntuacion agrega cortes donde el silencio no llega. Con `ventana_s=None` la
    puntuacion no aporta nada y el resultado DEBE reducir exactamente a
    `tramos_por_silencio`. Esa reduccion es un invariante, y se comprueba.

    El invariante fallo dos veces antes de fijarse, y por motivos distintos:

    1. El corte arrancaba el tramo siguiente en `palabra.fin`, con lo que la
       palabra que lo abria perdia su propia duracion: 317 cortes y 208 s de
       habla real descartados.
    2. El corte por silencio cerraba el tramo en `palabra.inicio`, con lo que el
       hueco de cierre entraba dentro del tramo. `tramos_por_silencio` lo
       excluye, asi que las dos rutas median objetos distintos.

    Ahora las dos ramas usan el criterio de `tramos_por_silencio`: un tramo
    contiene HABLA, no la pausa que la sigue. Es la convencion que hace
    comparables las dos columnas de la tabla, y es la que evita el mismo defecto
    por tercera vez.

    Args:
        inicio: Segundo inicial del span.
        fin: Segundo final del span.
        palabras: Las palabras del span.
        max_gap_s: Umbral de silencio.
        ventana_s: Ventana minima desde el inicio del tramo, o `None`.

    Returns:
        Tramos `(inicio, fin)`.
    """
    tramos: list[tuple[float, float]] = []
    corte = inicio
    for i, palabra in enumerate(palabras):
        if i > 0 and palabra.inicio - palabras[i - 1].fin > max_gap_s:
            if corte < palabras[i - 1].fin:
                tramos.append((corte, palabras[i - 1].fin))
            corte = palabra.inicio
            continue
        if ventana_s is None or not palabra.texto:
            continue
        if palabra.texto[-1] not in PUNTUACION_CIERRE:
            continue
        if palabra.fin - corte < ventana_s:
            continue
        if corte < palabra.fin:
            tramos.append((corte, palabra.fin))
        corte = palabra.fin
    if corte < fin:
        tramos.append((corte, fin))
    return tramos


def tabla_cruzada(inicio: float, fin: float, palabras: Sequence[Palabra]) -> list[FilaCruzada]:
    """La tabla que decide: cada par de mecanismos, medido.

    Criterio de victoria: tiempo capturado en tramos UTILIZABLES, no cantidad de
    tramos. Un mecanismo que produce mas clips de 3 s no gana; produce menos
    tiempo y se descarta.

    Args:
        inicio: Segundo inicial del span.
        fin: Segundo final del span.
        palabras: Las palabras del span.

    Returns:
        Una `FilaCruzada` por combinacion de los dos mecanismos.
    """
    filas: list[FilaCruzada] = []
    for max_gap in MAX_GAP_CANDIDATOS:
        for ventana in PUNCT_BREAK_CANDIDATOS:
            tramos = tramos_combinados(inicio, fin, palabras, max_gap, ventana)
            utiles = [t for t in tramos if (t[1] - t[0]) >= MIN_CLIP_S]
            filas.append(
                FilaCruzada(
                    max_gap=max_gap,
                    ventana=ventana,
                    n_tramos=len(tramos),
                    n_utiles=len(utiles),
                    segundos_utiles=sum(f - i for i, f in utiles),
                )
            )
    return filas


def _tabla_md(encabezados: Sequence[str], filas: Sequence[Sequence[object]]) -> list[str]:
    """Arma una tabla Markdown.

    Args:
        encabezados: Los titulos de columna.
        filas: Los valores, ya formateados.

    Returns:
        Las lineas de la tabla.
    """
    lineas = ["| " + " | ".join(encabezados) + " |"]
    lineas.append("|" + "|".join("---" for _ in encabezados) + "|")
    lineas.extend("| " + " | ".join(str(v) for v in fila) + " |" for fila in filas)
    return lineas


def _p(*partes: str) -> str:
    """Une fragmentos de un parrafo en una sola cadena.

    Existe porque la concatenacion implicita esta prohibida en `mode=all` de
    basedpyright, y escribir un parrafo largo con `+` explicito lo hace ilegible.
    Une con ESPACIO entre partes, asi que cada parte es una frase.

    Args:
        partes: Las frases del parrafo.

    Returns:
        El parrafo unido.
    """
    return " ".join(partes)


def _b(*lineas: str) -> str:
    """Une renglones de una lista con salto de linea.

    `_p` une con espacio y por eso sirve para prosa. Para una lista de vinetas
    hace falta lo contrario: cada vineta es su propia linea en Markdown, y
    unirlas con espacio produce un bloque que se lee como una frase sola.

    Args:
        lineas: Las lineas de la lista.

    Returns:
        La lista, con una linea por vineta.
    """
    return "\n".join(lineas)


def _bloque(salida: list[str], titulo: str, cuerpo: str) -> None:
    """Agrega un parrafo de una sola pieza logica al informe.

    Existe para que una cadena larga no se escriba por concatenacion
    implicita, que basedpyright rechaza en `mode=all`.

    Args:
        salida: El informe en construccion.
        titulo: La linea de titulo.
        cuerpo: El parrafo.
    """
    salida.extend((titulo, cuerpo, ""))


def _seccion_transcript(t: Transcript, a: Auditoria) -> list[str]:
    """La parte del informe que corresponde a UN transcript.

    Args:
        t: El transcript.
        a: Su auditoria.

    Returns:
        Las lineas de la seccion.
    """
    salida: list[str] = []
    _bloque(salida, f"### `{t.nombre}`", f"Ruta: `{t.ruta_relativa}`. Idioma: `{t.idioma}`.")
    salida.extend(_lineas_sin_voz(t, a))
    salida.extend(_lineas_alucinaciones(a))
    salida.extend(_lineas_duraciones(a))
    salida.extend(_lineas_monotonia(a))
    salida.extend(_lineas_puntuacion(a))
    return salida


def _cuenta_y_suma(huecos: Sequence[float], umbral: float) -> tuple[int, float]:
    """Cuantos huecos superan un umbral, y cuanto tiempo suman.

    Args:
        huecos: Los silencios entre palabras.
        umbral: El corte, en segundos.

    Returns:
        `(cantidad, segundos)`.
    """
    valores = [g for g in huecos if g > umbral]
    return len(valores), sum(valores)


def _lineas_sin_voz(t: Transcript, a: Auditoria) -> list[str]:
    """(a) La duracion sin voz transcripta, por umbral de hueco.

    Args:
        t: El transcript.
        a: Su auditoria.

    Returns:
        Las lineas del bloque.
    """
    salida: list[str] = []
    huecos = t.huecos
    n_1, s_1 = _cuenta_y_suma(huecos, HUECO_SIN_VOZ_S)
    n_2, s_2 = _cuenta_y_suma(huecos, 2.0)
    n_3, s_3 = _cuenta_y_suma(huecos, 3.0)
    _bloque(
        salida,
        "**(a) Porcion sin voz transcripta**",
        _b(
            _p(
                f"- Huecos entre palabras: {len(huecos)}. Huecos > {HUECO_SIN_VOZ_S} s:",
                f"**{n_1}** = **{s_1:.0f} s** = **{s_1 / a.duracion_s * 100:.1f} %** del span.",
            ),
            f"- Huecos > 2,0 s: {n_2} = {s_2:.0f} s.",
            f"- Huecos > 3,0 s: {n_3} = {s_3:.0f} s.",
            f"- Huecos negativos (palabras solapadas): {a.huecos_negativos}.",
        ),
    )
    return salida


def _lineas_alucinaciones(a: Auditoria) -> list[str]:
    """(b) Alucinaciones, por frase y por confianza.

    Args:
        a: La auditoria del transcript.

    Returns:
        Las lineas del bloque.
    """
    salida: list[str] = []
    cuerpo = (
        "- Ninguna frase canonica."
        if not a.alucinaciones
        else "\n".join(
            _tabla_md(
                ["frase canonica", "ocurrencias"],
                [[frase, n] for frase, n in sorted(a.alucinaciones.items())],
            )
        )
    )
    if a.ejemplo_alucinacion is not None:
        frase, momento, conf = a.ejemplo_alucinacion
        primer = _p(
            f"Primer bloque: `{frase}` en t={momento:.1f} s,",
            f"confianza de la primera palabra {conf:.3f}.",
        )
        cuerpo += "\n\n" + primer
    conf_baja = _p(
        f"- Palabras con confianza < {CONF_SOSPECHA}: **{a.conf_baja}**",
        f"({a.conf_baja / a.n_palabras * 100:.2f} %),",
        f"**{a.conf_baja_duracion_s:.0f} s** de duracion atribuida.",
    )
    cuerpo += "\n\n" + conf_baja
    if a.conf_baja_ejemplos:
        cuerpo += "\n\n" + "\n".join(
            _tabla_md(
                ["texto", "t (s)", "confianza"],
                [
                    [texto, f"{momento:.1f}", f"{conf:.3f}"]
                    for texto, momento, conf in a.conf_baja_ejemplos[:5]
                ],
            )
        )
    _bloque(salida, "**(b) Alucinaciones de Whisper**", cuerpo)
    return salida


def _lineas_duraciones(a: Auditoria) -> list[str]:
    """(c) Palabras con duracion imposible.

    Args:
        a: La auditoria del transcript.

    Returns:
        Las lineas del bloque.
    """
    salida: list[str] = []
    _bloque(
        salida,
        "**(c) Palabras con duracion imposible**",
        _b(
            _p(
                f"- Mediana de duracion de palabra: **{a.duracion_palabra_mediana:.4f} s**,",
                f"desviacion tipica **{a.duracion_palabra_sd:.4f} s**.",
            ),
            _p(
                f"- A mas de 3 desviaciones: cortas **{a.outliers_cortas}**,",
                f"largas **{a.outliers_largas}**.",
            ),
            _p(
                f"- Duracion <= 0: **{a.duracion_palabra_cero}**",
                f"({a.duracion_palabra_cero / a.n_palabras * 100:.1f} %).",
                f"Duracion >= 1,0 s: **{a.duracion_palabra_larga}**.",
            ),
        ),
    )
    return salida


def _lineas_monotonia(a: Auditoria) -> list[str]:
    """(d) Retrocesos de `start_s`.

    Args:
        a: La auditoria del transcript.

    Returns:
        Las lineas del bloque.
    """
    salida: list[str] = []
    cuerpo = f"- Retrocesos de `start_s`: **{a.retrocesos}**."
    if a.ejemplos_retroceso:
        cuerpo += "\n\n" + "\n".join(
            _tabla_md(
                ["indice", "start_s", "start_s anterior"],
                [
                    [i, f"{actual:.2f}", f"{previo:.2f}"]
                    for i, actual, previo in a.ejemplos_retroceso
                ],
            )
        )
    _bloque(salida, "**(d) Monotonia**", cuerpo)
    return salida


def _lineas_puntuacion(a: Auditoria) -> list[str]:
    """(e) Puntuacion y distribucion entre cierres de frase.

    Args:
        a: La auditoria del transcript.

    Returns:
        Las lineas del bloque.
    """
    salida: list[str] = []
    cuerpo = _b(
        f"- Palabras cerradas con puntuacion: **{a.frac_puntuacion * 100:.2f} %** de las palabras.",
    )
    if a.puntuacion_cuentas:
        cuerpo += "\n\n" + "\n".join(
            _tabla_md(
                ["signo", "palabras"],
                [[s, n] for s, n in sorted(a.puntuacion_cuentas.items())],
            )
        )
    cuerpo += "\n\n" + _b(
        _p(
            f"- Cierres de frase (`.` `?` `!`): **{a.n_cierres}** =",
            f"{a.n_cierres / (a.duracion_s / 60):.1f} por minuto.",
        )
    )
    if a.gaps_entre_cierres:
        g = sorted(a.gaps_entre_cierres)
        cuerpo += "\n" + _b(
            _p(
                f"- Distancia entre cierres consecutivos: mediana **{st.median(g):.2f} s**,",
                f"media {st.mean(g):.2f} s, min {g[0]:.2f} s, max {g[-1]:.2f} s.",
            )
        )
    else:
        cuerpo += "\n- No hay cierres suficientes para comparar."
    _bloque(salida, "**(e) Puntuacion**", cuerpo)
    return salida


def _lineas_histograma(t: Transcript, a: Auditoria) -> list[str]:
    """STEP 2: el histograma de huecos de UN transcript.

    Args:
        t: El transcript.
        a: Su auditoria.

    Returns:
        Las lineas del bloque.
    """
    salida: list[str] = []
    filas = [
        [etiqueta, n, f"{segundos:.1f}"] for etiqueta, n, segundos in histograma_huecos(t.huecos)
    ]
    _bloque(
        salida,
        f"### `{t.nombre}` — {len(t.huecos)} huecos",
        "\n".join(_tabla_md(["cubo (s)", "n", "segundos"], filas))
        + "\n\n"
        + f"Veredicto de integridad: **{veredicto(a)}**.",
    )
    return salida


def _lineas_tablas(t: Transcript, a: Auditoria) -> list[str]:
    """STEP 3: los mecanismos aislados y la cruzada, para UN transcript.

    Args:
        t: El transcript.
        a: Su auditoria.

    Returns:
        Las lineas de los bloques.
    """
    salida: list[str] = []
    palabras = [p for p in t.palabras if 0.0 <= p.inicio <= a.duracion_s]
    _bloque(salida, f"### `{t.nombre}`", "")

    cabeceras = ["mecanismo", "tramos", "min", "mediana", "max", "capturada s"]
    silencio: list[list[object]] = [
        [
            f"MAX_GAP_S={max_gap}",
            *resumen_tramos(tramos_por_silencio(0.0, a.duracion_s, palabras, max_gap)),
        ]
        for max_gap in MAX_GAP_CANDIDATOS
    ]
    puntuacion: list[list[object]] = []
    for ventana in PUNCT_BREAK_CANDIDATOS:
        etiqueta = "None" if ventana is None else f"{ventana:.0f}"
        puntuacion.append(
            [
                f"PUNCT_BREAK_S={etiqueta}",
                *resumen_tramos(tramos_por_puntuacion(0.0, a.duracion_s, palabras, ventana)),
            ]
        )
    _bloque(
        salida,
        "**Por mecanismo, aislado**",
        "\n".join(_tabla_md(cabeceras, silencio))
        + "\n\n"
        + "\n".join(_tabla_md(cabeceras, puntuacion)),
    )

    cruzada = [
        [
            f"{f.max_gap}",
            "None" if f.ventana is None else f"{f.ventana:.0f}",
            f.n_tramos,
            f.n_utiles,
            f"{f.segundos_utiles:.0f}",
        ]
        for f in tabla_cruzada(0.0, a.duracion_s, palabras)
    ]
    _bloque(
        salida,
        "**Cruzada**",
        "\n".join(
            _tabla_md(
                ["MAX_GAP_S", "PUNCT_BREAK_S", "tramos", "utiles", "segundos utiles"],
                cruzada,
            )
        ),
    )
    return salida


def _lineas_bloqueado() -> list[str]:
    """STEP 4: lo que no se puede medir sin audio.

    Returns:
        Las lineas del bloque.
    """
    salida: list[str] = []
    _bloque(
        salida,
        "## STEP 4 — Lo que NO se midio",
        _p(
            "**Delta de energia: BLOQUEADO.** Requiere audio. `astats` lee muestras y",
            "este material es solo word timestamps de Whisper. No hay cifra de RMS,",
            "de pico ni de variacion aqui, y no se estima.",
        )
        + "\n\n"
        + _p(
            "Tambien sin medir, por falta de audio: huecos que contienen voz",
            "superpuesta, numero de hablantes, y si un silencio va seguido de grito",
            "o risa. Los histogramas de arriba tratan TODO hueco grande como",
            "silencio, y en material de stream eso es una aproximacion que puede",
            "estar equivocada.",
        ),
    )
    return salida


def construir_informe(transcripts: Sequence[Transcript], auditorias: Sequence[Auditoria]) -> str:
    """Compone el informe completo.

    Args:
        transcripts: Los transcripts cargados.
        auditorias: Sus auditorias, en el mismo orden.

    Returns:
        El Markdown completo.
    """
    salida: list[str] = []
    salida.extend(
        (
            "# Perfil de material (transcripts)",
            "",
            "GENERADO por `tools/perfil_transcript.py`. Nada escrito a mano.",
            "",
        )
    )
    _bloque(
        salida,
        "",
        _p(
            "Se lee antes que cualquier conclusion de #72: un histograma de huecos",
            "construido sobre timestamps que no son de fiar describe el modelo, no",
            "el audio.",
        ),
    )

    _bloque(salida, "## STEP 1 — Auditoria de las transcripciones", "")
    filas: list[list[object]] = [
        [
            t.nombre,
            a.n_palabras,
            f"{a.duracion_s:.0f}",
            f"{a.duracion_s / 60:.0f} min",
            veredicto(a),
        ]
        for t, a in zip(transcripts, auditorias, strict=True)
    ]
    salida += _tabla_md(["transcript", "palabras", "span s", "span", "veredicto"], filas)
    salida.append("")
    for t, a in zip(transcripts, auditorias, strict=True):
        salida.extend(_seccion_transcript(t, a))

    _bloque(salida, "## STEP 2 — Histograma de huecos sobre estos transcripts", "")
    for t, a in zip(transcripts, auditorias, strict=True):
        salida.extend(_lineas_histograma(t, a))

    _bloque(
        salida,
        "## STEP 3 — La tabla que decide",
        _p(
            f"Un tramo cuenta como utilizable a partir de **{MIN_CLIP_S:.0f} s**.",
            "Gana el mecanismo que captura MAS TIEMPO en tramos utilizables,",
            "no el que hace mas tramos.",
        ),
    )
    for t, a in zip(transcripts, auditorias, strict=True):
        salida.extend(_lineas_tablas(t, a))

    salida.extend(_lineas_bloqueado())
    return "\n".join(salida) + "\n"


def main() -> int:
    """Carga, audita y escribe el informe.

    Returns:
        0 si todo se leyo y escribio; 1 si falta o no se puede leer un
        transcript.
    """
    transcripts: list[Transcript] = []
    for nombre, ruta in TRANSCRIPTS:
        try:
            transcripts.append(cargar(nombre, ruta))
        except (OSError, TypeError, ValueError, KeyError) as error:
            print(f"ERROR: {error}")
            return 1
    auditorias: list[Auditoria] = [auditar(t) for t in transcripts]
    informe = construir_informe(transcripts, auditorias)
    INFORME.parent.mkdir(parents=True, exist_ok=True)
    _ = INFORME.write_text(informe, encoding="utf-8")
    print(f"escrito: {INFORME.relative_to(RAIZ)}")
    for t, a in zip(transcripts, auditorias, strict=True):
        print(f"  {t.nombre}: {a.n_palabras} palabras, veredicto {veredicto(a)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
