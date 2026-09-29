"""Ensamblado de piezas ``given_clips`` con ffmpeg (argumentos, nunca shell).

El modo ``given_clips`` recibe clips ya cortados: el ensamblado los normaliza
a un lienzo vertical (píxeles cuadrados, SAR del clip respetado) y conserva el
audio propio del clip. El watermark opcional se superpone desde un asset local
durante todo el video, en la zona y tamaño del ``WatermarkConfig``. El
artefacto que sale de aquí es el que inspecciona el gate: nunca se valida
sobre los parámetros de entrada.

La publicación es atómica: ffmpeg escribe en un temporal hermano y solo un
render exitoso reemplaza el destino; un fallo deja intacto el artefacto previo.
"""

import contextlib
import math
import subprocess
import uuid
from pathlib import Path
from typing import assert_never

from kliptych.contract import Watermark, WatermarkPosition
from kliptych.encoding import muted_audio_arguments

_DEFAULT_WIDTH = 1080
_DEFAULT_HEIGHT = 1920
_DEFAULT_TIMEOUT_S = 300.0
_WATERMARK_MARGIN = 20
_STDERR_TAIL = 400


class AssembleError(Exception):
    """La pieza no se pudo ensamblar."""


class FFmpegAssembler:
    """Ensambla clips entregados en piezas verticales usando ffmpeg."""

    def __init__(
        self,
        *,
        ffmpeg: str = "ffmpeg",
        timeout_s: float = _DEFAULT_TIMEOUT_S,
    ) -> None:
        """Configura el binario y el timeout del ensamblado.

        Args:
            ffmpeg: Nombre o ruta del binario ffmpeg.
            timeout_s: Timeout máximo del ensamblado, en segundos.
        """
        self._ffmpeg: str = ffmpeg
        self._timeout_s: float = timeout_s

    def assemble(
        self,
        *,
        clip: Path,
        destination: Path,
        watermark: Path | None = None,
        watermark_config: Watermark | None = None,
        width: int = _DEFAULT_WIDTH,
        height: int = _DEFAULT_HEIGHT,
        mute_audio: bool = False,
    ) -> Path:
        """Ensambla un clip entregado en una pieza vertical para el gate.

        El render ocurre en un temporal hermano y se publica con un reemplazo
        atómico solo si ffmpeg termina con éxito; ante cualquier fallo el
        artefacto previo en ``destination`` queda intacto.

        Args:
            clip: Clip entregado por la campaña (ya cortado).
            destination: Ruta del artefacto final; se reemplaza al publicar y
                se crean los directorios padre que falten.
            watermark: Imagen opcional para superponer durante todo el video,
                en la zona y tamaño de ``watermark_config``.
            watermark_config: Posición, tamaño y opacidad del watermark; sin
                valor se usa el defecto del contrato (arriba a la derecha).
            width: Ancho del lienzo vertical.
            height: Alto del lienzo vertical.
            mute_audio: Si es True, silencia la pista sin eliminarla
                (``audio_policy=internal_official_sound``).

        Returns:
            La ruta del artefacto ensamblado.

        Raises:
            AssembleError: Si las entradas no existen, las dimensiones son
                inválidas, el destino no se puede preparar, ffmpeg falla o
                expira.
        """
        _require_file(clip, what="clip")
        if watermark is not None:
            _require_file(watermark, what="watermark")
        if width <= 0 or height <= 0:
            msg = f"dimensiones de lienzo inválidas: {width}x{height}"
            raise AssembleError(msg)
        try:
            destination.parent.mkdir(parents=True, exist_ok=True)
        except OSError as error:
            msg = f"no se pudo preparar el directorio del destino {destination}: {error}"
            raise AssembleError(msg) from error
        temporary = _temporary_path(destination)
        argv = self._build_argv(
            clip=clip,
            destination=temporary,
            watermark=watermark,
            watermark_config=watermark_config,
            width=width,
            height=height,
            mute_audio=mute_audio,
        )
        try:
            completed = subprocess.run(
                argv,
                capture_output=True,
                text=True,
                timeout=self._timeout_s,
                check=False,
            )
        except FileNotFoundError as error:
            _remove_quietly(temporary)
            msg = f"ffmpeg no está disponible: {self._ffmpeg}"
            raise AssembleError(msg) from error
        except subprocess.TimeoutExpired as error:
            _remove_quietly(temporary)
            msg = f"ffmpeg excedió el timeout de {self._timeout_s} s"
            raise AssembleError(msg) from error
        except OSError as error:
            _remove_quietly(temporary)
            msg = f"no se pudo ejecutar ffmpeg ({self._ffmpeg}): {error}"
            raise AssembleError(msg) from error
        if completed.returncode != 0:
            _remove_quietly(temporary)
            msg = f"ffmpeg falló con código {completed.returncode}: {_tail(completed.stderr)}"
            raise AssembleError(msg)
        try:
            _ = temporary.replace(destination)
        except OSError as error:
            _remove_quietly(temporary)
            msg = f"no se pudo publicar el artefacto en {destination}: {error}"
            raise AssembleError(msg) from error
        return destination

    def cut_exact(
        self,
        *,
        source: Path,
        destination: Path,
        start_s: float,
        duration_s: float,
    ) -> Path:
        """Corta un clip con precisión de frame usando libx264/aac y reseteando PTS a 0.0s.

        Args:
            source: Video descargado (con margen).
            destination: Ruta del artefacto cortado con precisión.
            start_s: Segundo de inicio relativo a la fuente descargada.
            duration_s: Duración exacta en segundos (end - start).

        Returns:
            La ruta del artefacto cortado con precisión.

        Raises:
            AssembleError: Si la fuente no existe, el intervalo es inválido o ffmpeg falla.
        """
        _require_file(source, what="source")
        if start_s < 0 or duration_s <= 0:
            msg = f"intervalo de corte inválido: start={start_s}, duration={duration_s}"
            raise AssembleError(msg)
        try:
            destination.parent.mkdir(parents=True, exist_ok=True)
        except OSError as error:
            msg = f"no se pudo preparar el directorio del destino {destination}: {error}"
            raise AssembleError(msg) from error
        temporary = _temporary_path(destination)
        argv = [
            self._ffmpeg,
            "-hide_banner",
            "-nostdin",
            "-v",
            "error",
            "-y",
            "-ss",
            f"{start_s:.3f}",
            "-t",
            f"{duration_s:.3f}",
            "-i",
            str(source),
            "-vf",
            "setpts=PTS-STARTPTS",
            "-map",
            "0:v:0",
            "-map",
            "0:a?",
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "20",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-movflags",
            "+faststart",
            str(temporary),
        ]
        try:
            completed = subprocess.run(
                argv,
                capture_output=True,
                text=True,
                timeout=self._timeout_s,
                check=False,
            )
        except FileNotFoundError as error:
            _remove_quietly(temporary)
            msg = f"ffmpeg no está disponible: {self._ffmpeg}"
            raise AssembleError(msg) from error
        except subprocess.TimeoutExpired as error:
            _remove_quietly(temporary)
            msg = f"ffmpeg excedió el timeout de {self._timeout_s} s"
            raise AssembleError(msg) from error
        except OSError as error:
            _remove_quietly(temporary)
            msg = f"no se pudo ejecutar ffmpeg ({self._ffmpeg}): {error}"
            raise AssembleError(msg) from error
        if completed.returncode != 0:
            _remove_quietly(temporary)
            msg = f"ffmpeg falló con código {completed.returncode}: {_tail(completed.stderr)}"
            raise AssembleError(msg)
        try:
            _ = temporary.replace(destination)
        except OSError as error:
            _remove_quietly(temporary)
            msg = f"no se pudo publicar el artefacto en {destination}: {error}"
            raise AssembleError(msg) from error
        return destination

    def render_arguments(
        self,
        *,
        clip: Path,
        destination: Path,
        watermark: Path | None = None,
        watermark_config: Watermark | None = None,
        width: int = _DEFAULT_WIDTH,
        height: int = _DEFAULT_HEIGHT,
        mute_audio: bool = False,
    ) -> tuple[str, ...]:
        """Devuelve el argv de ffmpeg que se usaría para este ensamblado.

        Es la receta de render que se registra en el manifiesto; el ensamblado
        real escribe primero en un temporal y publica al final.

        Args:
            clip: Clip entregado por la campaña.
            destination: Ruta final del artefacto.
            watermark: Imagen opcional a superponer.
            watermark_config: Posición, tamaño y opacidad del watermark.
            width: Ancho del lienzo vertical.
            height: Alto del lienzo vertical.
            mute_audio: Si es True, la receta incluye el silenciado de audio.

        Returns:
            El argv completo de ffmpeg, como tupla inmutable.
        """
        return tuple(
            self._build_argv(
                clip=clip,
                destination=destination,
                watermark=watermark,
                watermark_config=watermark_config,
                width=width,
                height=height,
                mute_audio=mute_audio,
            )
        )

    def _build_argv(
        self,
        *,
        clip: Path,
        destination: Path,
        watermark: Path | None,
        watermark_config: Watermark | None,
        width: int,
        height: int,
        mute_audio: bool,
    ) -> list[str]:
        square = "scale=trunc(iw*sar/2)*2:ih,setsar=1"
        scale = f"scale={width}:{height}:force_original_aspect_ratio=decrease"
        pad = f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:color=black"
        base = f"{square},{scale},{pad},setsar=1"
        argv = [
            self._ffmpeg,
            "-hide_banner",
            "-nostdin",
            "-v",
            "error",
            "-y",
            "-i",
            str(clip),
        ]
        if watermark is None:
            argv += ["-vf", base, "-map", "0:v:0", "-map", "0:a?"]
        else:
            config = (
                watermark_config
                if watermark_config is not None
                else Watermark(required=True, visible_full_video=True)
            )
            argv += [
                "-i",
                str(watermark),
                "-filter_complex",
                _watermark_filter(base, config, canvas_width=width),
                "-map",
                "[v]",
                "-map",
                "0:a?",
            ]
        if mute_audio:
            argv += list(muted_audio_arguments())
        argv += [
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "20",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-movflags",
            "+faststart",
            str(destination),
        ]
        return argv


def _watermark_filter(base: str, config: Watermark, *, canvas_width: int) -> str:
    """Construye el ``filter_complex`` que escala y superpone el watermark.

    El PNG viaja como segunda entrada explícita (``-i``), nunca como
    ``movie=``: así las rutas Windows con ``:`` no rompen el parser del
    grafo. El PNG se escala con ``scale`` de una sola entrada al
    ``scale_ratio`` del ancho del lienzo (altura con ``-2`` para preservar
    su aspecto en píxeles pares) y se superpone con ``overlay`` en la zona
    de ``config.position`` con el margen de seguridad. Sin filtros de doble
    entrada el framesync no puede truncar el video (``overlay`` repite el
    frame único del PNG hasta el fin del lienzo de forma determinista).

    Args:
        base: Cadena de filtros que normaliza el clip al lienzo vertical.
        config: Posición, tamaño y opacidad del watermark.
        canvas_width: Ancho del lienzo vertical, en píxeles.

    Returns:
        El grafo completo, con el video final en la etiqueta ``[v]``.
    """
    x, y = _overlay_xy(config.position)
    target_w = int(math.trunc(canvas_width * config.scale_ratio / 2) * 2)
    scale = f"[1:v]format=rgba,scale={target_w}:-2[wm]"
    if config.opacity >= 1.0:
        return f"[0:v]{base}[base];{scale};[base][wm]overlay={x}:{y}[v]"
    return (
        f"[0:v]{base}[base];{scale};"
        f"[wm]colorchannelmixer=aa={config.opacity}[wmf];"
        f"[base][wmf]overlay={x}:{y}[v]"
    )


def _overlay_xy(position: WatermarkPosition) -> tuple[str, str]:
    """Devuelve las expresiones ``(x, y)`` del ``overlay`` para una posición.

    Args:
        position: Zona del lienzo exigida por el contrato.

    Returns:
        Las expresiones de ffmpeg para la esquina superior izquierda del
        watermark, con el margen de seguridad desde los bordes.
    """
    margin = _WATERMARK_MARGIN
    if position is WatermarkPosition.TOP_LEFT:
        x, y = f"{margin}", f"{margin}"
    elif position is WatermarkPosition.TOP_RIGHT:
        x, y = f"W-w-{margin}", f"{margin}"
    elif position is WatermarkPosition.BOTTOM_LEFT:
        x, y = f"{margin}", f"H-h-{margin}"
    elif position is WatermarkPosition.BOTTOM_RIGHT:
        x, y = f"W-w-{margin}", f"H-h-{margin}"
    elif position is WatermarkPosition.CENTER:
        x, y = "(W-w)/2", "(H-h)/2"
    elif position is WatermarkPosition.CENTER_TOP:
        x, y = "(W-w)/2", f"{margin}"
    elif position is WatermarkPosition.CENTER_BOTTOM:
        x, y = "(W-w)/2", f"H-h-{margin}"
    else:
        assert_never(position)
    return (x, y)


def _temporary_path(destination: Path) -> Path:
    return destination.with_name(f".{destination.stem}.part-{uuid.uuid4().hex}{destination.suffix}")


def _remove_quietly(path: Path) -> None:
    # Limpieza best-effort del temporal; el destino final nunca se toca.
    with contextlib.suppress(OSError):
        path.unlink(missing_ok=True)


def _require_file(path: Path, *, what: str) -> None:
    if not path.is_file():
        msg = f"el {what} no existe: {path}"
        raise AssembleError(msg)


def _tail(text: str) -> str:
    stripped = text.strip()
    return stripped[-_STDERR_TAIL:]
