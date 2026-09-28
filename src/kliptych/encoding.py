"""Selección del codificador de video para los pasos de render.

La política del brief §8 es usar la 1650Ti (``h264_nvenc``) cuando está
disponible y caer a ``libx264`` en CPU cuando no. La decisión se concentra aquí
para que reframe y subtítulos compartan una única fuente de verdad y no
divergen en silencio.
"""

import contextlib
import logging
import subprocess
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger("kliptych.encoding")
_STDERR_TAIL = 500

_DEFAULT_TIMEOUT_S = 600.0


@dataclass(frozen=True, slots=True)
class RenderConfig:
    """Binario, timeout y disponibilidad de NVENC para un render ffmpeg."""

    ffmpeg: str = "ffmpeg"
    timeout_s: float = _DEFAULT_TIMEOUT_S
    nvenc_available: bool = False

    def __post_init__(self) -> None:
        """Valida el timeout del render.

        Raises:
            ValueError: Si el timeout no es positivo.
        """
        if self.timeout_s <= 0:
            msg = f"timeout inválido: {self.timeout_s}"
            raise ValueError(msg)


def video_encoder_arguments(*, nvenc_available: bool) -> tuple[str, ...]:
    """Devuelve los argumentos del codificador de video.

    Args:
        nvenc_available: Si NVENC está disponible en la máquina.

    Returns:
        Los argumentos de ffmpeg para el codificador elegido.
    """
    if nvenc_available:
        return ("-c:v", "h264_nvenc", "-preset", "p5", "-cq", "23", "-pix_fmt", "yuv420p")
    return ("-c:v", "libx264", "-preset", "medium", "-crf", "20", "-pix_fmt", "yuv420p")


def audio_and_container_arguments() -> tuple[str, ...]:
    """Devuelve los argumentos de audio y contenedor compartidos por los renders.

    Returns:
        Los argumentos de ffmpeg para audio AAC y ``faststart``.
    """
    return ("-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart")


def muted_audio_arguments() -> tuple[str, ...]:
    """Devuelve el filtro que silencia la pista sin eliminarla.

    El filtro ``volume=0`` deja la pista de audio presente en el contenedor
    (``ffprobe`` y las plataformas la siguen viendo) pero a silencio digital.
    Es seguro con ``-map 0:a?`` ante entradas sin audio: ffmpeg lo ignora sin
    error. Vale para ``libx264`` y ``h264_nvenc`` (solo toca el audio).

    Returns:
        Los argumentos de ffmpeg para silenciar el audio.
    """
    return ("-af", "volume=0")


def audio_injection_arguments(
    *,
    mix_ratio: float,
    video_duration_s: float | None = None,
) -> tuple[str, ...]:
    """Devuelve los argumentos que reemplazan o mezclan la pista de audio.

    Con ``mix_ratio`` igual o superior a ``1.0`` la pista externa reemplaza a la
    original (``-map 1:a:0``). Con un valor intermedio ambas pistas se mezclan
    con ``amix`` y un volumen proporcional: ``1 - mix_ratio`` para la original
    y ``mix_ratio`` para la externa. Si se proporciona ``video_duration_s``, se
    limita la duración con ``-t`` para evitar desfases con pistas más largas.

    Args:
        mix_ratio: Peso de la pista externa en la mezcla, en ``[0.0, 1.0]``.
        video_duration_s: Duración máxima en segundos para acotar el contenedor
            al largo exacto del video.

    Returns:
        Los argumentos de ffmpeg posteriores a los dos ``-i`` de entrada.

    Raises:
        ValueError: Si ``video_duration_s`` no es positivo.
    """
    if video_duration_s is not None and video_duration_s <= 0:
        msg = f"video_duration_s debe ser positivo: {video_duration_s}"
        raise ValueError(msg)
    duration_args = ("-t", f"{video_duration_s:.3f}") if video_duration_s is not None else ()
    if mix_ratio >= 1.0:
        return (
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-c:v",
            "copy",
            *audio_and_container_arguments(),
            *duration_args,
        )
    filter_graph = (
        f"[0:a]aformat=sample_fmts=fltp:channel_layouts=stereo,"
        f"volume={_volume(1.0 - mix_ratio)}[original];"
        f"[1:a]aformat=sample_fmts=fltp:channel_layouts=stereo,"
        f"volume={_volume(mix_ratio)}[external];"
        "[original][external]amix=inputs=2:duration=first:dropout_transition=2[aout]"
    )
    return (
        "-filter_complex",
        filter_graph,
        "-map",
        "0:v:0",
        "-map",
        "[aout]",
        "-c:v",
        "copy",
        *audio_and_container_arguments(),
        *duration_args,
    )


def fallback_encoder_arguments(argv: Sequence[str]) -> list[str]:
    """Reemplaza los argumentos de h264_nvenc por los equivalentes de libx264.

    Args:
        argv: Lista o secuencia de argumentos de ffmpeg.

    Returns:
        Nueva lista de argumentos con el codificador libx264 y sus parámetros.
    """
    new_argv = list(argv)
    if "h264_nvenc" not in new_argv:
        return new_argv
    nvenc_args = list(video_encoder_arguments(nvenc_available=True))
    cpu_args = list(video_encoder_arguments(nvenc_available=False))
    for i in range(len(new_argv) - len(nvenc_args) + 1):
        if new_argv[i : i + len(nvenc_args)] == nvenc_args:
            new_argv[i : i + len(nvenc_args)] = cpu_args
            return new_argv
    idx = 0
    result: list[str] = []
    while idx < len(new_argv):
        arg = new_argv[idx]
        if arg == "h264_nvenc":
            result.append("libx264")
            idx += 1
        elif arg == "-cq" and idx + 1 < len(new_argv):
            result.extend(["-crf", "20"])
            idx += 2
        elif (
            arg == "-preset"
            and idx + 1 < len(new_argv)
            and new_argv[idx + 1] in {"p1", "p2", "p3", "p4", "p5", "p6", "p7"}
        ):
            result.extend(["-preset", "medium"])
            idx += 2
        else:
            result.append(arg)
            idx += 1
    return result


def _remove_quietly(path: Path) -> None:
    with contextlib.suppress(OSError):
        path.unlink(missing_ok=True)


def _tail(text: str) -> str:
    return text.strip()[-_STDERR_TAIL:]


def _execute_single(
    cmd: Sequence[str],
    *,
    render: RenderConfig,
    temporary: Path | None,
    error_cls: type[Exception],
    runner: Callable[..., subprocess.CompletedProcess[str]] | None = None,
) -> subprocess.CompletedProcess[str]:
    proc_runner = runner if runner is not None else subprocess.run
    try:
        return proc_runner(
            list(cmd),
            capture_output=True,
            text=True,
            timeout=render.timeout_s,
            check=False,
        )
    except FileNotFoundError as error:
        if temporary is not None:
            _remove_quietly(temporary)
        msg = f"ffmpeg no está disponible: {render.ffmpeg}"
        raise error_cls(msg) from error
    except subprocess.TimeoutExpired as error:
        if temporary is not None:
            _remove_quietly(temporary)
        msg = f"ffmpeg excedió el timeout de {render.timeout_s} s"
        raise error_cls(msg) from error
    except OSError as error:
        if temporary is not None:
            _remove_quietly(temporary)
        msg = f"no se pudo ejecutar ffmpeg ({render.ffmpeg}): {error}"
        raise error_cls(msg) from error


def _execute_fallback(
    cmd: Sequence[str],
    *,
    render: RenderConfig,
    temporary: Path | None,
    error_cls: type[Exception],
    runner: Callable[..., subprocess.CompletedProcess[str]] | None = None,
) -> None:
    logger.warning("NVENC falló, reintentando con libx264...")
    fallback_argv = fallback_encoder_arguments(cmd)
    try:
        completed = _execute_single(
            fallback_argv, render=render, temporary=temporary, error_cls=error_cls, runner=runner
        )
    except subprocess.CalledProcessError as error:
        if temporary is not None:
            _remove_quietly(temporary)
        msg = f"ffmpeg falló: {error}"
        raise error_cls(msg) from error
    if completed.returncode != 0:
        if temporary is not None:
            _remove_quietly(temporary)
        msg = f"ffmpeg falló con código {completed.returncode}: {_tail(completed.stderr)}"
        raise error_cls(msg)


def run_ffmpeg_with_fallback(
    argv: Sequence[str],
    *,
    render: RenderConfig,
    temporary: Path | None = None,
    error_cls: type[Exception] = RuntimeError,
    runner: Callable[..., subprocess.CompletedProcess[str]] | None = None,
) -> None:
    """Ejecuta ffmpeg con reintento automático libx264 si h264_nvenc falla.

    Args:
        argv: Secuencia de argumentos de línea de comandos de ffmpeg.
        render: Configuración de render (binario ffmpeg, timeout, nvenc).
        temporary: Archivo temporal opcional que limpiar en caso de error fatal.
        error_cls: Clase de excepción a lanzar en caso de fallo.
        runner: Invocador de procesos ejecutables (por defecto subprocess.run).
    """
    cmd_list = list(argv)
    try:
        completed = _execute_single(
            cmd_list, render=render, temporary=temporary, error_cls=error_cls, runner=runner
        )
    except subprocess.CalledProcessError as error:
        if "h264_nvenc" in cmd_list:
            _execute_fallback(
                cmd_list, render=render, temporary=temporary, error_cls=error_cls, runner=runner
            )
            return
        if temporary is not None:
            _remove_quietly(temporary)
        msg = f"ffmpeg falló: {error}"
        raise error_cls(msg) from error

    if completed.returncode == 0:
        return
    if "h264_nvenc" in cmd_list:
        _execute_fallback(
            cmd_list, render=render, temporary=temporary, error_cls=error_cls, runner=runner
        )
        return
    if temporary is not None:
        _remove_quietly(temporary)
    msg = f"ffmpeg falló con código {completed.returncode}: {_tail(completed.stderr)}"
    raise error_cls(msg)


def _volume(value: float) -> str:
    return f"{value:.3f}"
