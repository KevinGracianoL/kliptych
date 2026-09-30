"""Módulo de letras sincronizadas (.lrc) y cliente lrclib.

Implementa el parseo determinista de archivos y contenido .lrc con validación
estricta de marcas de tiempo ASCII, soporte de offset, descarte de metadatos,
clasificación cronológica y el cliente inyectable para lrclib.
"""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar, Protocol, Self, cast

from kliptych.subtitles import SubtitleLayout, SubtitleStyle

if TYPE_CHECKING:
    from collections.abc import Sequence

    from kliptych.assets import AssetRegistry
    from kliptych.contract import Contract
    from kliptych.gate.models import SubtitleSegment

__all__ = [
    "LrcEmptyWindowError",
    "LrcParseError",
    "LrclibClient",
    "LyricLine",
    "LyricsError",
    "LyricsNotFoundError",
    "SyncedLyricsProvider",
    "cut_lyric_window",
    "lyric_lines_to_ass",
    "lyric_lines_to_subtitle_segments",
    "lyric_lines_to_subtitle_text",
    "parse_lrc",
    "resolve_synced_lyrics",
]

_OFFSET_TAG_RE = re.compile(r"^\[offset:\s*([+-]?[0-9]+)\s*\]$", re.IGNORECASE)
_TIMESTAMP_TAG_RE = re.compile(r"\[([0-9:.]+)\]")

_PARTS_SPLIT = 2
_MAX_SECONDS = 60
_HTTP_NOT_FOUND = 404
_MAX_FRAC_DIGITS = 3
_FRAC_BASE_10 = 10.0
_FRAC_BASE_100 = 100.0
_FRAC_BASE_1000 = 1000.0
_MS_PER_SECOND = 1000.0
_SECONDS_PER_MINUTE = 60.0
_ALLOWED_SCHEMES = frozenset({"http", "https"})
_DEFAULT_LINE_DURATION_S = 4.0
_CENTISECONDS_PER_HOUR = 360_000
_CENTISECONDS_PER_MINUTE = 6_000
_CENTISECONDS_PER_SECOND = 100


class LrcParseError(ValueError):
    """Fallo en el formato o contenido del archivo .lrc."""


class LyricsError(Exception):
    """Fallo general al obtener letras sincronizadas."""


class LyricsNotFoundError(LyricsError):
    """No se encontraron letras sincronizadas para la pista."""


class LrcEmptyWindowError(LyricsError):
    """No hay líneas de letra dentro de la ventana de recorte solicitada."""


@dataclass(frozen=True, slots=True)
class LyricLine:
    """Línea de letra con su marca de inicio sincronizada y cota final opcional."""

    start_sec: float
    text: str
    end_sec: float | None = None


class _HttpResponseProtocol(Protocol):
    """Protocolo mínimo de respuesta HTTP con context manager."""

    def read(self) -> bytes: ...
    def __enter__(self) -> Self: ...
    def __exit__(self, *args: object) -> None: ...


class _OpenerProtocol(Protocol):
    """Protocolo de apertura de peticiones HTTP."""

    def open(
        self,
        fullurl: urllib.request.Request | str,
        data: bytes | None = ...,
        timeout: float = ...,
    ) -> _HttpResponseProtocol: ...


class SyncedLyricsProvider(Protocol):
    """Protocolo inyectable para proveedores de letras sincronizadas."""

    def get_synced_lyrics(
        self,
        *,
        track_name: str,
        artist_name: str | None = None,
        album_name: str | None = None,
        duration_s: float | None = None,
    ) -> str:
        """Obtiene la letra sincronizada en formato .lrc."""
        ...


def _parse_fraction_seconds(frac_str: str, token: str) -> float:
    """Parsea la fracción de segundo decimal (1 a 3 dígitos).

    Args:
        frac_str: Dígitos de la fracción.
        token: Marca de tiempo original para reportar errores.

    Returns:
        Valor de la fracción en segundos.

    Raises:
        LrcParseError: Si la fracción contiene caracteres no numéricos o demasiados dígitos.
    """
    if not (frac_str.isascii() and frac_str.isdigit()):
        msg = f"fracción de segundo no numérica en la marca de tiempo: {token!r}"
        raise LrcParseError(msg)
    frac_digits = len(frac_str)
    if frac_digits == 1:
        return int(frac_str) / _FRAC_BASE_10
    if frac_digits == _PARTS_SPLIT:
        return int(frac_str) / _FRAC_BASE_100
    if frac_digits == _MAX_FRAC_DIGITS:
        return int(frac_str) / _FRAC_BASE_1000
    msg = f"fracción de segundo con demasiados dígitos: {token!r}"
    raise LrcParseError(msg)


def _parse_seconds_part(ss_str: str, token: str) -> float:
    """Parsea y valida la parte de segundos y su fracción opcional.

    Args:
        ss_str: Cadena con segundos y fracción opcional.
        token: Marca de tiempo original para reportar errores.

    Returns:
        Valor en segundos flotantes dentro de [0, 60).

    Raises:
        LrcParseError: Si los segundos no son válidos o superan 59.
    """
    ss_parts = ss_str.split(".")
    if len(ss_parts) == 1:
        ss_int_str = ss_parts[0]
        frac_val = 0.0
    elif len(ss_parts) == _PARTS_SPLIT:
        ss_int_str, frac_str = ss_parts
        frac_val = _parse_fraction_seconds(frac_str, token)
    else:
        msg = f"segundos mal formateados en la marca de tiempo: {token!r}"
        raise LrcParseError(msg)

    if not (ss_int_str.isascii() and ss_int_str.isdigit()):
        msg = f"segundos no numéricos en la marca de tiempo: {token!r}"
        raise LrcParseError(msg)

    ss_val = int(ss_int_str)
    if not 0 <= ss_val < _MAX_SECONDS:
        msg = f"segundos fuera de rango [0, 59]: {ss_val} en {token!r}"
        raise LrcParseError(msg)

    return ss_val + frac_val


def _parse_timestamp_seconds(token: str) -> float:
    """Parsea y valida estrictamente una marca de tiempo ASCII mm:ss[.xx[x]].

    Args:
        token: Marca de tiempo sin corchetes.

    Returns:
        Tiempo en segundos.

    Raises:
        LrcParseError: Si la marca no es ASCII, es negativa o está mal formateada.
    """
    if not token.isascii():
        msg = f"la marca de tiempo contiene caracteres no ASCII: {token!r}"
        raise LrcParseError(msg)
    if "-" in token:
        msg = f"la marca de tiempo no admite valores negativos: {token!r}"
        raise LrcParseError(msg)
    parts = token.split(":")
    if len(parts) != _PARTS_SPLIT:
        msg = f"formato de tiempo inválido (se esperaba mm:ss): {token!r}"
        raise LrcParseError(msg)

    mm_str, ss_str = parts
    if not (mm_str.isascii() and mm_str.isdigit()):
        msg = f"minutos no numéricos en la marca de tiempo: {token!r}"
        raise LrcParseError(msg)

    ss_seconds = _parse_seconds_part(ss_str, token)
    return int(mm_str) * _SECONDS_PER_MINUTE + ss_seconds


def parse_lrc(content: str) -> tuple[LyricLine, ...]:
    """Parsea deterministamente el contenido de un archivo .lrc.

    Args:
        content: Texto del archivo .lrc.

    Returns:
        Tupla ordenada cronológicamente de LyricLine.

    Raises:
        LrcParseError: Si el archivo no contiene líneas válidas sincronizadas
            con texto, contiene caracteres no ASCII en marcas o tiempos invertidos/negativos.
    """
    lines: list[LyricLine] = []
    current_offset_sec = 0.0

    for raw_line in content.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        offset_match = _OFFSET_TAG_RE.match(line)
        if offset_match:
            offset_ms = int(offset_match.group(1))
            current_offset_sec = offset_ms / _MS_PER_SECOND
            continue

        matches = list(_TIMESTAMP_TAG_RE.finditer(line))
        if not matches:
            continue

        last_match = matches[-1]
        text = line[last_match.end() :].strip()
        if not text:
            continue

        for m in matches:
            raw_ts = m.group(1)
            parsed_sec = _parse_timestamp_seconds(raw_ts)
            adjusted_sec = max(0.0, parsed_sec + current_offset_sec)
            lines.append(LyricLine(start_sec=adjusted_sec, text=text))

    if not lines:
        msg = "el archivo .lrc no contiene líneas sincronizadas válidas con texto"
        raise LrcParseError(msg)

    lines.sort(key=lambda line: line.start_sec)
    return tuple(lines)


def cut_lyric_window(
    lines: Sequence[LyricLine],
    *,
    start_sec: float,
    end_sec: float,
    default_line_duration_s: float = _DEFAULT_LINE_DURATION_S,
) -> tuple[LyricLine, ...]:
    """Filtra y desplaza las líneas de letra al rango temporal de la ventana.

    Desplaza las marcas temporales restando start_sec (0.0s = inicio del clip final).

    Args:
        lines: Líneas de letra sincronizadas ordenadas cronológicamente.
        start_sec: Inicio de la ventana en segundos.
        end_sec: Fin de la ventana en segundos.
        default_line_duration_s: Duración por defecto para la última línea.

    Returns:
        Tupla de líneas filtradas y desplazadas dentro de la ventana.

    Raises:
        LrcEmptyWindowError: Si la ventana queda sin líneas de letra.
        LrcParseError: Si las cotas son inválidas.
    """
    if start_sec >= end_sec:
        msg = f"start_sec ({start_sec}) debe ser menor que end_sec ({end_sec})"
        raise LrcParseError(msg)

    cut_lines: list[LyricLine] = []
    total_lines = len(lines)
    for i, line in enumerate(lines):
        line_end = (
            line.end_sec
            if line.end_sec is not None
            else (
                lines[i + 1].start_sec
                if i + 1 < total_lines
                else line.start_sec + default_line_duration_s
            )
        )
        if line_end <= start_sec or line.start_sec >= end_sec:
            continue
        shifted_start = max(0.0, line.start_sec - start_sec)
        shifted_end = min(end_sec - start_sec, line_end - start_sec)
        if shifted_end <= shifted_start:
            continue
        cut_lines.append(
            LyricLine(
                start_sec=shifted_start,
                text=line.text,
                end_sec=shifted_end,
            )
        )

    if not cut_lines:
        msg = f"no hay líneas de letra en el rango temporal [{start_sec}, {end_sec}]"
        raise LrcEmptyWindowError(msg)

    return tuple(cut_lines)


def _format_ass_time(seconds: float) -> str:
    total_centiseconds = max(0, round(seconds * _CENTISECONDS_PER_SECOND))
    hours, remainder = divmod(total_centiseconds, _CENTISECONDS_PER_HOUR)
    minutes, remainder = divmod(remainder, _CENTISECONDS_PER_MINUTE)
    secs, centiseconds = divmod(remainder, _CENTISECONDS_PER_SECOND)
    return f"{hours}:{minutes:02d}:{secs:02d}.{centiseconds:02d}"


def _escape_ass_text(text: str) -> str:
    escaped = text.replace("\\", "\\\\")
    escaped = escaped.replace("{", "\\{").replace("}", "\\}")
    return escaped.replace("\r\n", "\\N").replace("\n", "\\N").replace("\r", "\\N")


def lyric_lines_to_ass(
    lines: Sequence[LyricLine],
    *,
    duration_s: float | None = None,
    layout: SubtitleLayout | None = None,
    style: SubtitleStyle | None = None,
) -> str:
    """Genera el contenido ASS para las líneas de letra especificadas.

    Args:
        lines: Líneas de letra dentro de la ventana del clip.
        duration_s: Duración máxima del clip en segundos.
        layout: Resolución lógica del lienzo.
        style: Estilo ASS de los subtítulos.

    Returns:
        Texto en formato ASS con eventos Dialogue.
    """
    resolved_layout = layout or SubtitleLayout()
    resolved_style = style or SubtitleStyle()
    header = [
        "[Script Info]",
        "Title: Kliptych Lyrics",
        "ScriptType: v4.00+",
        "WrapStyle: 0",
        "ScaledBorderAndShadow: yes",
        f"PlayResX: {resolved_layout.width}",
        f"PlayResY: {resolved_layout.height}",
        "",
        "[V4+ Styles]",
        (
            "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, "
            "BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, "
            "BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding"
        ),
        (
            f"Style: Default,{resolved_style.fontname},{resolved_style.fontsize},"
            f"{resolved_style.primary_colour},&H000000FF,"
            f"{resolved_style.outline_colour},{resolved_style.back_colour},0,0,0,0,100,100,0,0,1,"
            f"{resolved_style.outline},{resolved_style.shadow},"
            f"{resolved_style.alignment},20,20,{resolved_style.margin_v},1"
        ),
        "",
        "[Events]",
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
    ]
    for line in lines:
        start_str = _format_ass_time(line.start_sec)
        end_val = (
            line.end_sec if line.end_sec is not None else line.start_sec + _DEFAULT_LINE_DURATION_S
        )
        if duration_s is not None:
            end_val = min(duration_s, end_val)
        end_str = _format_ass_time(end_val)
        escaped = _escape_ass_text(line.text)
        header.append(f"Dialogue: 0,{start_str},{end_str},Default,,0,0,0,,{escaped}")
    return "\n".join(header) + "\n"


def lyric_lines_to_subtitle_segments(
    lines: Sequence[LyricLine],
    *,
    default_duration_s: float = _DEFAULT_LINE_DURATION_S,
) -> tuple[SubtitleSegment, ...]:
    """Convierte líneas de letra en SubtitleSegment relativos al clip.

    Args:
        lines: Líneas de letra dentro del clip.
        default_duration_s: Duración por defecto si falta end_sec.

    Returns:
        Tupla de SubtitleSegment.
    """
    from kliptych.gate.models import SubtitleSegment

    segments: list[SubtitleSegment] = []
    for line in lines:
        end_s = line.end_sec if line.end_sec is not None else line.start_sec + default_duration_s
        segments.append(
            SubtitleSegment(
                text=line.text,
                start_s=line.start_sec,
                end_s=end_s,
            )
        )
    return tuple(segments)


def lyric_lines_to_subtitle_text(lines: Sequence[LyricLine]) -> str:
    """Extrae el texto completo continuo de las líneas de letra.

    Args:
        lines: Líneas de letra del clip.

    Returns:
        Cadena con todas las líneas separadas por espacios.
    """
    return " ".join(line.text.strip() for line in lines if line.text.strip())


class LrclibClient:
    """Cliente para el endpoint público de lrclib.net."""

    DEFAULT_USER_AGENT: ClassVar[str] = "Kliptych/1.0"
    _base_url: str
    _timeout_s: float
    _opener: _OpenerProtocol

    def __init__(
        self,
        *,
        base_url: str = "https://lrclib.net",
        timeout_s: float = 10.0,
        opener: _OpenerProtocol | None = None,
    ) -> None:
        """Inicializa el cliente de lrclib.

        Args:
            base_url: URL base de la API de lrclib.
            timeout_s: Límite de tiempo en segundos para la petición.
            opener: Instancia opcional de opener para inyectar en pruebas.

        Raises:
            LyricsError: Si el esquema de base_url no es http o https.
        """
        scheme = urllib.parse.urlsplit(base_url).scheme
        if scheme not in _ALLOWED_SCHEMES:
            msg = f"esquema no permitido '{scheme}': solo http y https"
            raise LyricsError(msg)
        self._base_url = base_url.rstrip("/")
        self._timeout_s = timeout_s
        self._opener = opener or urllib.request.build_opener()

    def _fetch_bytes(self, req: urllib.request.Request, track_name: str) -> bytes:
        try:
            with self._opener.open(req, timeout=self._timeout_s) as response:
                return response.read()
        except urllib.error.HTTPError as error:
            if error.code == _HTTP_NOT_FOUND:
                msg = f"pista '{track_name}' no encontrada en lrclib (404)"
                raise LyricsNotFoundError(msg) from error
            msg = f"error HTTP al consultar lrclib ({error.code}): {error.reason}"
            raise LyricsError(msg) from error
        except (TimeoutError, urllib.error.URLError) as error:
            msg = f"petición a lrclib expiró (timed out): {error}"
            raise LyricsError(msg) from error

    def _fetch_payload(self, req: urllib.request.Request, track_name: str) -> dict[str, object]:
        raw_bytes = self._fetch_bytes(req, track_name)
        try:
            data = cast("object", json.loads(raw_bytes.decode("utf-8")))
        except (json.JSONDecodeError, OSError) as error:
            msg = f"respuesta inválida de lrclib: {error}"
            raise LyricsError(msg) from error

        if not isinstance(data, dict):
            msg = "formato JSON inesperado en respuesta de lrclib"
            raise LyricsError(msg)
        return cast("dict[str, object]", data)

    def get_synced_lyrics(
        self,
        *,
        track_name: str,
        artist_name: str | None = None,
        album_name: str | None = None,
        duration_s: float | None = None,
    ) -> str:
        """Consulta lrclib para obtener letras sincronizadas.

        Args:
            track_name: Nombre de la pista musical.
            artist_name: Nombre opcional del artista.
            album_name: Nombre opcional del álbum.
            duration_s: Duración opcional de la pista en segundos.

        Returns:
            Contenido en texto .lrc con las letras sincronizadas.

        Raises:
            LyricsNotFoundError: Si la pista no existe o solo tiene plainLyrics.
            LyricsError: Si la petición falla o expira.
        """
        query: dict[str, str] = {"track_name": track_name}
        if artist_name:
            query["artist_name"] = artist_name
        if album_name:
            query["album_name"] = album_name
        if duration_s is not None:
            query["duration"] = str(round(duration_s))

        url = f"{self._base_url}/api/get?{urllib.parse.urlencode(query)}"
        req = urllib.request.Request(
            url,
            headers={"User-Agent": self.DEFAULT_USER_AGENT},
        )
        payload = self._fetch_payload(req, track_name)
        synced = payload.get("syncedLyrics")
        if not isinstance(synced, str) or not synced.strip():
            msg = f"la respuesta de lrclib para '{track_name}' no contiene syncedLyrics válidas"
            raise LyricsNotFoundError(msg)

        return synced


def resolve_synced_lyrics(
    *,
    contract: Contract,
    registry: AssetRegistry,
    provider: SyncedLyricsProvider | None = None,
) -> tuple[LyricLine, ...]:
    """Resuelve las líneas sincronizadas de un contrato con fail-closed.

    Prioridad:
    1. Archivo .lrc local en AssetRegistry si contract.lyric_video.lrc_asset_id está declarado.
    2. Proveedor lrclib inyectado si lrclib_enabled es True.

    Args:
        contract: Contrato de la campaña.
        registry: Registro de assets del workspace.
        provider: Proveedor SyncedLyricsProvider opcional para lrclib.

    Returns:
        Tupla ordenada de LyricLine.

    Raises:
        LyricsError: Si no hay fuente o la resolución falla.
    """
    lyric_config = contract.lyric_video
    if lyric_config is not None and lyric_config.lrc_asset_id:
        from kliptych.assets import AssetError

        lrc_id = lyric_config.lrc_asset_id
        try:
            intact = registry.verify(lrc_id)
        except (AssetError, OSError) as error:
            msg = f"error al verificar el asset .lrc '{lrc_id}': {error}"
            raise LyricsError(msg) from error
        if not intact:
            msg = f"el asset .lrc '{lrc_id}' no existe o fue modificado (hash inválido)"
            raise LyricsError(msg)
        try:
            path = registry.path_for(lrc_id)
            content = path.read_text(encoding="utf-8")
        except (AssetError, OSError) as error:
            msg = f"no se pudo leer el archivo .lrc local: {error}"
            raise LyricsError(msg) from error
        return parse_lrc(content)

    if lyric_config is not None and lyric_config.lrclib_enabled:
        if provider is None:
            msg = "lrclib_enabled=True pero no se configuró ningún SyncedLyricsProvider"
            raise LyricsError(msg)
        if not lyric_config.track_name:
            msg = "se requiere track_name en lyric_video para consultar letras en lrclib"
            raise LyricsError(msg)
        content = provider.get_synced_lyrics(
            track_name=lyric_config.track_name,
            artist_name=lyric_config.artist_name,
        )
        return parse_lrc(content)

    msg = "no hay fuente de letras configurada en el contrato (ni lrc_asset_id ni lrclib)"
    raise LyricsError(msg)
