"""Ensamblado de piezas ``given_clips`` con ffmpeg (argumentos, nunca shell).

El modo ``given_clips`` recibe clips ya cortados: el ensamblado los normaliza
a un lienzo vertical (píxeles cuadrados, SAR del clip respetado) y conserva el
audio propio del clip. El watermark opcional se superpone desde un asset local
durante todo el video. El artefacto que sale de aquí es el que inspecciona el
gate: nunca se valida sobre los parámetros de entrada.

La publicación es atómica: ffmpeg escribe en un temporal hermano y solo un
render exitoso reemplaza el destino; un fallo deja intacto el artefacto previo.
"""

import contextlib
import subprocess
import uuid
from pathlib import Path

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
        width: int = _DEFAULT_WIDTH,
        height: int = _DEFAULT_HEIGHT,
    ) -> Path:
        """Ensambla un clip entregado en una pieza vertical para el gate.

        El render ocurre en un temporal hermano y se publica con un reemplazo
        atómico solo si ffmpeg termina con éxito; ante cualquier fallo el
        artefacto previo en ``destination`` queda intacto.

        Args:
            clip: Clip entregado por la campaña (ya cortado).
            destination: Ruta del artefacto final; se reemplaza al publicar y
                se crean los directorios padre que falten.
            watermark: Imagen opcional para superponer en la esquina superior
                derecha durante todo el video.
            width: Ancho del lienzo vertical.
            height: Alto del lienzo vertical.

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
            width=width,
            height=height,
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

    def _build_argv(
        self,
        *,
        clip: Path,
        destination: Path,
        watermark: Path | None,
        width: int,
        height: int,
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
            overlay = f"[base][1:v]overlay=W-w-{_WATERMARK_MARGIN}:{_WATERMARK_MARGIN}[v]"
            argv += [
                "-i",
                str(watermark),
                "-filter_complex",
                f"[0:v]{base}[base];{overlay}",
                "-map",
                "[v]",
                "-map",
                "0:a?",
            ]
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
