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
        lines.extend(_dialogue_line(word) for word in words)
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

    def burn(self, *, video: Path, subtitles: Path, destination: Path) -> Path:
        """Quema los subtítulos en el video.

        El render ocurre en un temporal hermano y se publica con un reemplazo
        atómico solo si ffmpeg termina con éxito; ante cualquier fallo el
        artefacto previo en ``destination`` queda intacto.

        Args:
            video: Ruta del video fuente.
            subtitles: Ruta del archivo ``.ass`` a quemar.
            destination: Ruta del artefacto final; se crean los directorios
                padre que falten.

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
        argv = self._build_argv(video=video, subtitles=subtitles, destination=temporary)
        _run_ffmpeg(argv, temporary=temporary, render=self._render)
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
    ) -> tuple[str, ...]:
        """Devuelve el argv de ffmpeg que se usaría para este quemado.

        Es la receta de render que se registra en el manifiesto; el quemado real
        escribe primero en un temporal y publica al final.

        Args:
            video: Ruta del video fuente.
            subtitles: Ruta del archivo ``.ass``.
            destination: Ruta final del artefacto.

        Returns:
            El argv completo de ffmpeg, como tupla inmutable.
        """
        return tuple(self._build_argv(video=video, subtitles=subtitles, destination=destination))

    def _build_argv(self, *, video: Path, subtitles: Path, destination: Path) -> list[str]:
        filter_graph = f"subtitles=filename='{_filter_path(subtitles)}'"
        argv = [
            self._render.ffmpeg,
            "-hide_banner",
            "-nostdin",
            "-v",
            "error",
            "-y",
            "-i",
            str(video),
            "-vf",
            filter_graph,
            "-map",
            "0:v:0",
            "-map",
            "0:a?",
        ]
        argv += list(video_encoder_arguments(nvenc_available=self._render.nvenc_available))
        argv += list(audio_and_container_arguments())
        argv.append(str(destination))
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


def _dialogue_line(word: Word) -> str:
    start = _format_time(word.start_s)
    end = _format_time(word.end_s)
    karaoke = max(0, round((word.end_s - word.start_s) * 100))
    text = f"{{\\k{karaoke}}}{_escape_text(word.text)}"
    return f"Dialogue: 0,{start},{end},{_STYLE_NAME},,0,0,0,,{text}"


def _filter_path(path: Path) -> str:
    # En Windows el ':' de la unidad rompe el parser de filtros; se escapa.
    return path.as_posix().replace(":", "\\:")


def _require_file(path: Path, *, what: str) -> None:
    if not path.is_file():
        msg = f"el {what} no existe: {path}"
        raise SubtitleError(msg)


def _run_ffmpeg(argv: Sequence[str], *, temporary: Path, render: RenderConfig) -> None:
    run_ffmpeg_with_fallback(
        argv,
        render=render,
        temporary=temporary,
        error_cls=SubtitleError,
        runner=subprocess.run,
    )


def _temporary_path(destination: Path) -> Path:
    return destination.with_name(f".{destination.stem}.part-{uuid.uuid4().hex}{destination.suffix}")


def _remove_quietly(path: Path) -> None:
    with contextlib.suppress(OSError):
        path.unlink(missing_ok=True)
