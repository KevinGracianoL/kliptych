r"""Generación de subtítulos karaoke en ASS y quemado con ffmpeg.

El módulo toma las palabras con marca de tiempo de la transcripción (C2) y
produce un archivo Advanced SubStation Alpha (``.ass``) con una línea de
diálogo por palabra y etiquetas ``\k`` de karaoke. El estilo por defecto es
legible sobre video vertical 9:16 (fuente clara, contorno negro y posición
inferior centrada).

El quemado usa el filtro ``subtitles`` de ffmpeg, invocado con lista de
argumentos (nunca shell) y codificado con NVENC si está disponible (brief §8).
Todo fallo externo se traduce a ``SubtitleError`` conservando la causa.
"""

import contextlib
import functools
import os
import subprocess
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

from pydantic import BaseModel, ConfigDict, Field

from kliptych.encoding import (
    RenderConfig,
    audio_and_container_arguments,
    muted_audio_arguments,
    run_ffmpeg_with_fallback,
    video_encoder_arguments,
)
from kliptych.transcribe import Word

_STYLE_NAME = "Default"
_SECONDARY_COLOUR = "&H000000FF"
_DEFAULT_WIDTH = 1080
_DEFAULT_HEIGHT = 1920
_MARGIN_H = 20
_ENCODING = 1

# Grouping of words into one Dialogue event each. A karaoke line is read as a
# unit: the highlight sweeps across its words, so words are grouped and emitted
# as consecutive {\k} tags inside a single event, not one event per word.
# Vertical 9:16 is narrow, and the SubStation readable-line convention is the
# upper bound for how long a line should stay up.
MAX_CHARS_PER_LINE = 18
MAX_DURATION_S = 7.0
# Word endings that make a break after them preferable to breaking at the limit.
SENTENCE_ENDINGS = ".?!,;:"
# A sentence break is only taken once the line already holds this many words,
# otherwise every sentence-ending word would leave an orphan line.
MIN_WORDS_BEFORE_SENTENCE_BREAK = 2

# Smallest {\k} that libass actually draws. Measured, not assumed: a word
# allocated 0cs renders NOTHING (no error, no warning), so a sub-centisecond
# word would silently vanish from the output. {\k1}, {\k2} and {\k5} all draw the
# word in full and differ only in how long the highlight lasts.
#
# The effective floor is min(MIN_WORD_CENTISECONDS, total_cs // n_words), so the
# repair can never break the rule that the {\k} values sum to the event
# duration; it moves centiseconds, it never adds them. Real speech runs
# 200-500ms per word, so this floor almost never binds and 1cs is enough. A2.2
# makes it configurable.
MIN_WORD_CENTISECONDS = 1

_STYLE_FORMAT = (
    "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
    "OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, "
    "ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, "
    "MarginL, MarginR, MarginV, Encoding"
)
_EVENT_FORMAT = "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text"


class SubtitleError(Exception):
    """Los subtítulos no se pudieron generar o quemar."""


class SubtitleStyle(BaseModel):
    """Estilo ASS de los subtítulos, en formato ``&HAABBGGRR`` para colores."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", frozen=True)

    fontname: str = Field(default="Arial", min_length=1)
    fontsize: int = Field(default=48, gt=0)
    primary_colour: str = Field(default="&H00FFFFFF", pattern=r"^&H[0-9A-Fa-f]{8}$")
    outline_colour: str = Field(default="&H00000000", pattern=r"^&H[0-9A-Fa-f]{8}$")
    back_colour: str = Field(default="&H00000000", pattern=r"^&H[0-9A-Fa-f]{8}$")
    outline: int = Field(default=2, ge=0)
    shadow: int = Field(default=0, ge=0)
    margin_v: int = Field(default=40, ge=0)
    alignment: int = Field(default=2, ge=1, le=9)


@dataclass(frozen=True, slots=True)
class SubtitleLayout:
    """Resolución lógica del lienzo ASS, pensada para video vertical 9:16."""

    width: int = _DEFAULT_WIDTH
    height: int = _DEFAULT_HEIGHT

    def __post_init__(self) -> None:
        """Valida las dimensiones del lienzo.

        Raises:
            ValueError: Si alguna dimensión no es positiva.
        """
        if self.width <= 0 or self.height <= 0:
            msg = f"dimensiones de lienzo inválidas: {self.width}x{self.height}"
            raise ValueError(msg)


class SubtitleRenderer:
    """Genera el ``.ass`` y lo quema en el video con ffmpeg."""

    def __init__(
        self,
        *,
        style: SubtitleStyle | None = None,
        layout: SubtitleLayout | None = None,
        render: RenderConfig | None = None,
    ) -> None:
        """Configura el estilo, el lienzo y el render.

        Args:
            style: Estilo ASS; por defecto el legible para 9:16.
            layout: Resolución lógica del ``.ass``; por defecto 1080x1920.
            render: Binario, timeout y NVENC del quemado; por defecto los estándar.
        """
        self._style: SubtitleStyle = SubtitleStyle() if style is None else style
        self._layout: SubtitleLayout = SubtitleLayout() if layout is None else layout
        self._render: RenderConfig = RenderConfig() if render is None else render

    def build(self, words: Sequence[Word]) -> str:
        """Genera el contenido ASS a partir de las palabras transcritas.

        Args:
            words: Palabras con marca de tiempo, en orden temporal.

        Returns:
            El contenido completo del archivo ``.ass``.

        Raises:
            SubtitleError: Si no hay palabras para subtitular.
        """
        if not words:
            msg = "no hay palabras para subtitular"
            raise SubtitleError(msg)
        lines = [
            "[Script Info]",
            "Title: Kliptych",
            "ScriptType: v4.00+",
            "WrapStyle: 0",
            "ScaledBorderAndShadow: yes",
            f"PlayResX: {self._layout.width}",
            f"PlayResY: {self._layout.height}",
            "",
            "[V4+ Styles]",
            _STYLE_FORMAT,
            _style_line(self._style),
            "",
            "[Events]",
            _EVENT_FORMAT,
        ]
        lines.extend(_line_event(line) for line in _group_words(words))
        return "\n".join(lines) + "\n"

    def write(self, words: Sequence[Word], destination: Path) -> Path:
        """Escribe el ``.ass`` de forma atómica.

        Args:
            words: Palabras con marca de tiempo.
            destination: Ruta del archivo ``.ass``; se crean los directorios
                padre que falten.

        Returns:
            La ruta del archivo escrito.

        Raises:
            SubtitleError: Si no hay palabras o el destino no se puede escribir.
        """
        content = self.build(words)
        try:
            destination.parent.mkdir(parents=True, exist_ok=True)
        except OSError as error:
            msg = f"no se pudo preparar el directorio del destino {destination}: {error}"
            raise SubtitleError(msg) from error
        temporary = _temporary_path(destination)
        try:
            _ = temporary.write_text(content, encoding="utf-8")
        except OSError as error:
            _remove_quietly(temporary)
            msg = f"no se pudo escribir el archivo de subtítulos {destination}: {error}"
            raise SubtitleError(msg) from error
        try:
            _ = temporary.replace(destination)
        except OSError as error:
            _remove_quietly(temporary)
            msg = f"no se pudo publicar el archivo de subtítulos en {destination}: {error}"
            raise SubtitleError(msg) from error
        return destination

    def burn(
        self, *, video: Path, subtitles: Path, destination: Path, mute_audio: bool = False
    ) -> Path:
        """Quema los subtítulos en el video.

        El render ocurre en un temporal hermano y se publica con un reemplazo
        atómico solo si ffmpeg termina con éxito; ante cualquier fallo el
        artefacto previo en ``destination`` queda intacto.

        Args:
            video: Ruta del video fuente.
            subtitles: Ruta del archivo ``.ass`` a quemar.
            destination: Ruta del artefacto final; se crean los directorios
                padre que falten.
            mute_audio: Si es True, silencia la pista sin eliminarla
                (``audio_policy=internal_official_sound``).

        Returns:
            La ruta del artefacto con subtítulos quemados.

        Raises:
            SubtitleError: Si las entradas no existen, el destino no se puede
                preparar, ffmpeg falla o expira.
        """
        _require_file(video, what="video")
        _require_file(subtitles, what="subtítulos")
        try:
            destination.parent.mkdir(parents=True, exist_ok=True)
        except OSError as error:
            msg = f"no se pudo preparar el directorio del destino {destination}: {error}"
            raise SubtitleError(msg) from error
        temporary = _temporary_path(destination)
        argv = self._build_argv(
            video=video, subtitles=subtitles, destination=temporary, mute_audio=mute_audio
        )
        _run_ffmpeg(argv, temporary=temporary, render=self._render, cwd=destination.parent)
        try:
            _ = temporary.replace(destination)
        except OSError as error:
            _remove_quietly(temporary)
            msg = f"no se pudo publicar el artefacto en {destination}: {error}"
            raise SubtitleError(msg) from error
        return destination

    def render_arguments(
        self,
        *,
        video: Path,
        subtitles: Path,
        destination: Path,
        mute_audio: bool = False,
    ) -> tuple[str, ...]:
        """Devuelve el argv de ffmpeg que se usaría para este quemado.

        Es la receta de render que se registra en el manifiesto; el quemado real
        escribe primero en un temporal y publica al final.

        Args:
            video: Ruta del video fuente.
            subtitles: Ruta del archivo ``.ass``.
            destination: Ruta final del artefacto.
            mute_audio: Si es True, la receta incluye el silenciado de audio.

        Returns:
            El argv completo de ffmpeg, como tupla inmutable.
        """
        return tuple(
            self._build_argv(
                video=video, subtitles=subtitles, destination=destination, mute_audio=mute_audio
            )
        )

    def _build_argv(
        self, *, video: Path, subtitles: Path, destination: Path, mute_audio: bool
    ) -> list[str]:
        video_resolved = video.resolve()
        destination_resolved = destination.resolve()
        filter_graph = (
            f"subtitles=filename='{_relative_filter_path(subtitles, destination_resolved.parent)}'"
        )
        argv = [
            self._render.ffmpeg,
            "-hide_banner",
            "-nostdin",
            "-v",
            "error",
            "-y",
            "-i",
            str(video_resolved),
            "-vf",
            filter_graph,
            "-map",
            "0:v:0",
            "-map",
            "0:a?",
        ]
        if mute_audio:
            argv += list(muted_audio_arguments())
        argv += list(video_encoder_arguments(nvenc_available=self._render.nvenc_available))
        argv += list(audio_and_container_arguments())
        argv.append(str(destination_resolved))
        return argv


def _format_time(seconds: float) -> str:
    total_centiseconds = max(0, round(seconds * 100))
    hours, remainder = divmod(total_centiseconds, 360_000)
    minutes, remainder = divmod(remainder, 6_000)
    secs, centiseconds = divmod(remainder, 100)
    return f"{hours}:{minutes:02d}:{secs:02d}.{centiseconds:02d}"


def _escape_text(text: str) -> str:
    escaped = text.replace("\\", "\\\\")
    escaped = escaped.replace("{", "\\{").replace("}", "\\}")
    return escaped.replace("\r\n", "\\N").replace("\n", "\\N").replace("\r", "\\N")


def _style_line(style: SubtitleStyle) -> str:
    return (
        f"Style: {_STYLE_NAME},{style.fontname},{style.fontsize},"
        f"{style.primary_colour},{_SECONDARY_COLOUR},{style.outline_colour},{style.back_colour},"
        f"0,0,0,0,100,100,0,0,1,{style.outline},{style.shadow},"
        f"{style.alignment},{_MARGIN_H},{_MARGIN_H},{style.margin_v},{_ENCODING}"
    )


def _ends_sentence(word: Word) -> bool:
    """Dice si la palabra cierra una frase segun ``SENTENCE_ENDINGS``.

    Args:
        word: La palabra a inspeccionar.

    Returns:
        True si su texto termina en puntuacion de cierre.
    """
    return word.text.rstrip().endswith(tuple(SENTENCE_ENDINGS))


def _group_words(words: Sequence[Word]) -> list[list[Word]]:
    """Agrupa palabras en lineas, una por evento ``Dialogue``.

    Las palabras son atomicas: nunca se parte una palabra. Una linea se cierra
    cuando anadir la siguiente superaria ``MAX_CHARS_PER_LINE`` **o**
    ``MAX_DURATION_S``. Si dentro del limite hay una palabra que termina en
    ``SENTENCE_ENDINGS``, se rompe ahi; si no, se rompe en el limite. No hay
    duracion minima: una linea corta es valida y rellenarla mentiria sobre el
    tiempo.

    Args:
        words: Palabras con marca de tiempo, en orden temporal.

    Returns:
        Las lineas, cada una con al menos una palabra.
    """
    lines: list[list[Word]] = []
    current: list[Word] = []
    current_chars = 0
    for word in words:
        length = len(word.text)
        line_start = current[0].start_s if current else word.start_s
        # El span va del inicio de la PRIMERA palabra de la linea al fin de la
        # que se anade. Sumar el span anterior seria contarlo dos veces y
        # partiria las lineas antes de tiempo.
        span = word.end_s - line_start
        projected = current_chars + (1 + length if current else length)
        if current and (projected > MAX_CHARS_PER_LINE or span > MAX_DURATION_S):
            lines.append(current)
            current = [word]
            current_chars = length
            continue
        current.append(word)
        current_chars = projected
        # La puntuacion de cierre es un punto de corte PREFERIDO, pero no se
        # parte una linea en su primera palabra: eso deja un huerfano por
        # Oracion.
        if len(current) >= MIN_WORDS_BEFORE_SENTENCE_BREAK and _ends_sentence(word):
            lines.append(current)
            current = []
            current_chars = 0
    if current:
        lines.append(current)
    return lines


def _allocate_centiseconds(words: Sequence[Word]) -> list[int]:
    r"""Reparte los centisegundos de una linea entre sus palabras.

    Los ``{\k}`` van en centisegundos enteros y las marcas de faster-whisper son
    float en segundos. Redondear cada palabra por su cuenta hace que la suma no
    cuadre con la duracion real, y el error se acumula siempre en el mismo
    sentido: el resalte se queda atras de la voz y no la alcanza nunca.

    Reparto por resto mayor: se convierte el TOTAL de la linea una sola vez a
    centisegundos enteros, cada palabra recibe ``floor(d*100)``, y los
    centisegundos que sobran van uno a una a las palabras con mayor resto
    fraccionario, desempatando por indice. Asi la suma es exacta por
    construccion.

    Luego se reparan los ceros MOVIENDO un centisegundo desde las asignaciones
    mayores, nunca sumando, porque un ``{\k0}`` no se dibuja. La suma sigue
    cuadrando con la duracion.

    Args:
        words: Palabras de una sola linea.

    Returns:
        Los centisegundos de cada palabra, en orden.

    Raises:
        SubtitleError: Si la linea tiene mas palabras que centisegundos, que no
            puede ocurrir con habla real pero si con una transcripcion
            sintetica o corrupta.
    """
    if not words:
        msg = "no hay palabras que repartir"
        raise SubtitleError(msg)
    # La base del reparto son las RANURAS: desde el inicio de cada palabra
    # hasta el inicio de la siguiente, y la ultima hasta el final de la linea.
    # Asi los centisegundos teselan la linea entera, silencios incluidos. Con la
    # duracion propia de cada palabra los silencios se perderian y el evento
    # terminaria antes de que termine la ultima palabra.
    #
    # ALTERNATIVA EVALUADA Y DESCARTADA: cuantizar cada limite a centisegundos
    # enteros primero y repartir por diferencias enteras. Borra 22 lineas de este
    # bloque y elimina por completo el round(..., 9) de abajo, porque no habria
    # ni resto ni orden. Se descarto por FRECUENCIA, no por correccion:
    #     ranura de 0 cs con aritmetica float : practicamente imposible
    #     ranura de 0 cs cuantizando limites  : 15109 / 20000 lineas = 75%
    # con palabras de 3 a 15 ms. La reparacion de ceros ya existe y lo absorbe,
    # pero entonces deja de ser la excepcion y pasa a ser el camino normal del
    # habla rapida, y todo fallo futuro seria un fallo en el camino comun.
    # Misma correccion, peor forma.
    line_end = words[-1].end_s
    slots: list[float] = [
        ((words[index + 1].start_s if index + 1 < len(words) else line_end) - word.start_s) * 100.0
        for index, word in enumerate(words)
    ]
    total = sum(slots)
    total_cs = round(total)
    floor_cs = min(MIN_WORD_CENTISECONDS, total_cs // len(words))
    if floor_cs < 1:
        msg = (
            f"linea degenerada: {len(words)} palabras en {total_cs} centisegundos; "
            "no hay reparto posible que las dibuje sin superar la duracion"
        )
        raise SubtitleError(msg)

    allocated = [max(floor_cs, int(value)) for value in slots]
    difference = total_cs - sum(allocated)
    if difference > 0:
        # El resto se compara redondeado: dos ranuras iguales pueden diferir en
        # el ultimo bit por el ruido de coma flotante al restar marcas, y sin
        # redondear el desempate por indice seria inestable.
        order = sorted(range(len(words)), key=lambda i: (-round(slots[i] - int(slots[i]), 9), i))
        for step in range(difference):
            allocated[order[step % len(order)]] += 1
    elif difference < 0:
        order = sorted(range(len(words)), key=lambda i: (-allocated[i], i))
        for step in range(-difference):
            allocated[order[step % len(order)]] -= 1
            if allocated[order[step % len(order)]] < floor_cs:
                msg = (
                    f"linea degenerada: {len(words)} palabras en {total_cs} centisegundos; "
                    "el suelo por palabra no cabe"
                )
                raise SubtitleError(msg)
    if sum(allocated) != total_cs:
        # INALCANZABLE POR CONSTRUCCION, y aun asi se deja el guard.
        #
        # Structuralmente: los tres caminos terminan en suma == total_cs. Si la
        # diferencia es positiva, el bucle mueve exactamente esa diferencia, una
        # unidad por paso, sobre una permutacion de los indices. Si es negativa,
        # el bucle descuenta exactamente su opuesto y, en cuanto una asignacion
        # bajaria del suelo, lanza OTRO error antes de llegar aqui. Si es cero,
        # la suma ya es total_cs. No queda ninguna rama que termine con otra
        # suma.
        #
        # Medido ademas: 69.723 entradas (rejilla exhaustiva de 1 a 3 palabras
        # por 20 duraciones, mas 60.000 lineas aleatorias con silencios
        # arbitrarios) y 0 veces alcanzado. Patron H2 de PR #54.
        #
        # No lleva `# pragma: no cover`: no estamos tapando un hueco de
        # cobertura, estamos documentando una linea inalcanzable por
        # construccion, que es una afirmacion distinta y mas fuerte.
        msg = f"reparto inconsistente: {sum(allocated)} != {total_cs}"
        raise SubtitleError(msg)
    return allocated


def _line_event(words: Sequence[Word]) -> str:
    r"""Construye un evento ``Dialogue`` con un ``{\k}`` por palabra.

    Las marcas del evento y los valores ``{\\k}`` salen de la MISMA rejilla de
    centisegundos enteros: son una linea temporal en dos representaciones, no dos
    lineales temporales. El final del evento es el inicio mas la suma de los
    ``{\\k}``, de modo que no pueden separarse.

    La identidad telescopica es EXACTA EN EL ARCHIVO EMITIDO: la suma de los
    centisegundos enteros es siempre igual a ``End - Start``, que es lo que
    consume libass. No es exacta contra un extremo redondeado por separado,
    porque en coma flotante ``sum(ranuras)`` y ``end - start`` difieren en el
    ultimo bit y, cuando el total cae a un pelo de un ``.5``, los dos ``round()``
    caen en lados opuestos: 96 de 69.723 lineas, un 0,14%. Es la resolucion del
    propio formato, porque las marcas ASS son de centisegundo.

    Args:
        words: Palabras de una sola linea.

    Returns:
        La linea de dialogo ASS.
    """
    allocated = _allocate_centiseconds(words)
    start_cs = round(words[0].start_s * 100)
    end_cs = start_cs + sum(allocated)
    text = "".join(
        f"{{\\k{value}}}{_escape_text(word.text)}"
        for word, value in zip(words, allocated, strict=True)
    )
    start = _format_time(start_cs / 100)
    end = _format_time(end_cs / 100)
    return f"Dialogue: 0,{start},{end},{_STYLE_NAME},,0,0,0,,{text}"


def _filter_path(path: Path) -> str:
    # En Windows el ':' de la unidad rompe el parser de filtros; se escapa.
    # Solo se usa como fallback cuando no se puede relativizar: el filtro
    # `subtitles` en Windows solo acepta rutas relativas al cwd.
    return path.as_posix().replace(":", "\\:")


def _relative_filter_path(subtitles: Path, base: Path) -> str:
    r"""Devuelve la ruta del .ass relativa al cwd de ffmpeg.

    ffmpeg ejecuta con ``cwd`` en el directorio de salida, así que una ruta
    relativa funciona en Windows donde una absoluta rompe el parser del
    filtro ``subtitles``.

    Args:
        subtitles: Ruta del archivo ``.ass``.
        base: Directorio base (padre del destino) usado como cwd.

    Returns:
        La ruta relativa en formato POSIX, sin escape de ':' (las rutas
        relativas no llevan letra de unidad; solo el fallback absoluto
        de :func:`_filter_path` escapa el ':' de la unidad Windows).
    """
    try:
        rel = os.path.relpath(subtitles, start=base)
    except (ValueError, OSError):
        return _filter_path(subtitles)
    return Path(rel).as_posix()


def _require_file(path: Path, *, what: str) -> None:
    if not path.is_file():
        msg = f"el {what} no existe: {path}"
        raise SubtitleError(msg)


def _run_ffmpeg(
    argv: Sequence[str], *, temporary: Path, render: RenderConfig, cwd: Path | None = None
) -> None:
    runner = subprocess.run if cwd is None else functools.partial(subprocess.run, cwd=cwd)
    run_ffmpeg_with_fallback(
        argv,
        render=render,
        temporary=temporary,
        error_cls=SubtitleError,
        runner=runner,
    )


def _temporary_path(destination: Path) -> Path:
    return destination.with_name(f".{destination.stem}.part-{uuid.uuid4().hex}{destination.suffix}")


def _remove_quietly(path: Path) -> None:
    with contextlib.suppress(OSError):
        path.unlink(missing_ok=True)
