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
from typing import Final, NoReturn

from kliptych.degradations import registrar_caida

logger = logging.getLogger("kliptych.encoding")
_STDERR_TAIL = 500

# Aspecto por defecto del render 9:16, con la tolerancia con la que se acepta
# que una fuente ya este en vertical y no haya que reencuadrar.
_DEFAULT_TARGET_ASPECT: Final[float] = 9.0 / 16.0
_DEFAULT_ASPECT_TOLERANCE: Final[float] = 0.05
_DEFAULT_TIMEOUT_S = 600.0


@dataclass(frozen=True, slots=True)
class RenderConfig:
    """Binario, timeout y disponibilidad de NVENC para un render ffmpeg."""

    ffmpeg: str = "ffmpeg"
    timeout_s: float = _DEFAULT_TIMEOUT_S
    nvenc_available: bool = False
    # Aspecto objetivo del render, y tolerancia admisible al compararlo con el
    # de la fuente. Antes vivian en constantes de modulo del orquestador, lo que
    # hacia imposible que dos relaciones de aspecto convivieran en una misma
    # invocacion. Aqui son estado de la invocacion, y como RenderConfig ya entra
    # en el digest del checkpoint, dos aspectos dan dos fingerprints distintos
    # sin tocar la logica del fingerprint.
    target_aspect: float = _DEFAULT_TARGET_ASPECT
    aspect_tolerance: float = _DEFAULT_ASPECT_TOLERANCE

    def __post_init__(self) -> None:
        """Valida el timeout del render y el aspecto objetivo.

        Raises:
            ValueError: Si el timeout no es positivo, o el aspecto no es
                un valor de aspecto utilizable.
        """
        if self.timeout_s <= 0:
            msg = f"timeout inválido: {self.timeout_s}"
            raise ValueError(msg)
        if self.target_aspect <= 0:
            msg = f"aspecto objetivo inválido: {self.target_aspect}"
            raise ValueError(msg)
        if self.aspect_tolerance < 0:
            msg = f"tolerancia de aspecto inválida: {self.aspect_tolerance}"
            raise ValueError(msg)


def aspect_matches(
    width: int,
    height: int,
    *,
    target_aspect: float = _DEFAULT_TARGET_ASPECT,
    tolerance: float = _DEFAULT_ASPECT_TOLERANCE,
) -> bool:
    """Si unas dimensiones estan en el aspecto objetivo dentro de la tolerancia.

    Esta es LA pregunta de geometria del repo, y por eso vive aqui y no en el
    render ni en el gate: `RenderConfig` ya declara el aspecto objetivo y la
    tolerancia con la que se acepta, y este modulo es una hoja (solo stdlib), asi
    que tanto el orquestador como el gate pueden llamarla sin ciclos.

    Que sea UNA funcion y no dos copias es el punto. El reframe cuantiza a un
    entero par (`width = _even(round(height * 9/16))`), de modo que un alto de
    1080 da 608x1080: 0,08 % por debajo de 9:16. La cuantizacion NO GARANTIZA
    un 9:16 exacto, pero tampoco lo impide: `crop_size(1080, 1920)` da
    `(1080, 1920)`, exacto, y tambien 720x1280 y 1152x2048. Un gate que
    comparase a 1e-9 no estaria siendo mas riguroso en general, estaria
    exigiendo un valor que el render produce a veces y no otras.

    Comparacion RELATIVA, no absoluta: `|aspect - objetivo| <= objetivo *
    tolerancia`. Importa, porque el criterio escala con el objetivo en lugar de
    depender del tamano del lienzo.

    Args:
        width: Ancho medido.
        height: Alto medido. Se asume positivo; quien llame ya lo valido.
        target_aspect: El aspecto objetivo, por defecto el del render.
        tolerance: Tolerancia RELATIVA admitida.

    Returns:
        ``True`` si las dimensiones estan dentro de la tolerancia del objetivo.
    """
    aspect = width / height
    return abs(aspect - target_aspect) <= target_aspect * tolerance


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


def _encoder_en(argv: Sequence[str]) -> str:
    """Lee el codec de video que pide un argv de ffmpeg.

    Args:
        argv: Lista o secuencia de argumentos de ffmpeg.

    Returns:
        El valor de `-c:v`, o `"desconocido"` si el argv no lo declara.
    """
    for indice, argumento in enumerate(argv):
        if argumento == "-c:v" and indice + 1 < len(argv):
            return str(argv[indice + 1])
    return "desconocido"


def _remove_quietly(path: Path) -> None:
    with contextlib.suppress(OSError):
        path.unlink(missing_ok=True)


def _tail(text: str) -> str:
    return text.strip()[-_STDERR_TAIL:]


@dataclass(frozen=True, slots=True)
class Invocation:
    """Contrato de la llamada a ffmpeg, decidido por el llamador.

    Agrupa lo que NO es el comando: el temporal que hay que limpiar si falla, la
    excepción con la que se reporta, la función que lanza el proceso, su
    directorio de trabajo y su techo de tiempo. Va junto porque estas tres
    funciones lo comparten entero y pasarlo suelto hacía inviable añadir una
    cuarta opción sin romper cada llamada.

    Attributes:
        render: Configuración de render (binario ffmpeg, timeout, nvenc).
        temporary: Archivo a limpiar si ffmpeg falla; ``None`` si no hay.
        error_cls: Excepción que se lanza al fallar.
        runner: Lanzador de procesos; ``None`` para usar ``subprocess.run``.
        cwd: Directorio de trabajo; ``None`` para heredar el actual.
        timeout_s: Techo de tiempo propio; ``None`` para usar el de ``render``.
    """

    render: RenderConfig
    temporary: Path | None = None
    error_cls: type[Exception] = RuntimeError
    runner: Callable[..., subprocess.CompletedProcess[str]] | None = None
    cwd: Path | None = None
    timeout_s: float | None = None

    @property
    def timeout(self) -> float:
        """Techo de tiempo: el del llamador si lo fijó, el de ``render`` si no."""
        return self.timeout_s if self.timeout_s is not None else self.render.timeout_s

    @property
    def launcher(self) -> Callable[..., subprocess.CompletedProcess[str]]:
        """Lanzador de procesos: el inyectado, o ``subprocess.run`` por defecto."""
        return self.runner if self.runner is not None else subprocess.run

    @property
    def kwargs(self) -> dict[str, object]:
        """Argumentos de la llamada; ``cwd`` solo si hay subtítulos que ubique."""
        invocacion: dict[str, object] = {
            "capture_output": True,
            "text": True,
            "timeout": self.timeout,
            "check": False,
        }
        # Sin subtitulos el contrato de invocacion no lleva `cwd`; pasarlo en
        # `None` cambiaria la firma que los dobles de los tests comprueban.
        if self.cwd is not None:
            invocacion["cwd"] = self.cwd
        return invocacion

    def fail(self, message: str, *, from_error: BaseException | None = None) -> NoReturn:
        """Limpia el temporal y lanza la excepción del llamador.

        Args:
            message: Mensaje del fallo.
            from_error: Causa original, si la hay, para no perder la cadena.
        """
        if self.temporary is not None:
            _remove_quietly(self.temporary)
        if from_error is not None:
            raise self.error_cls(message) from from_error
        raise self.error_cls(message)


def _execute_single(cmd: Sequence[str], inv: Invocation) -> subprocess.CompletedProcess[str]:
    try:
        return inv.launcher(list(cmd), **inv.kwargs)
    except FileNotFoundError as error:
        msg = f"ffmpeg no está disponible: {inv.render.ffmpeg}"
        inv.fail(msg, from_error=error)
    except subprocess.TimeoutExpired as error:
        msg = f"ffmpeg excedió el timeout de {inv.timeout} s"
        inv.fail(msg, from_error=error)
    except OSError as error:
        msg = f"no se pudo ejecutar ffmpeg ({inv.render.ffmpeg}): {error}"
        inv.fail(msg, from_error=error)


def _execute_fallback(cmd: Sequence[str], inv: Invocation) -> None:
    """Reintenta con libx264 y REGISTRA la caida antes de producir el archivo.

    El registro va antes del reintento. Si no se pudiera registrar, la corrida
    revienta aqui y no se produce ningun archivo: un artefacto que dice que pidio
    NVENC mientras es libx264 es peor que una corrida fallida.

    Args:
        cmd: Argumentos con los que se intento el render.
        inv: Contrato de la invocacion (render, temporal, excepcion, runner).
    """
    fallback_argv = fallback_encoder_arguments(cmd)
    pedido = _encoder_en(cmd)
    usado = _encoder_en(fallback_argv)
    registrar_caida(pedido, usado)
    logger.warning("NVENC falló, reintentando con %s...", usado)
    try:
        completed = _execute_single(fallback_argv, inv)
    except subprocess.CalledProcessError as error:
        msg = f"ffmpeg falló: {error}"
        inv.fail(msg, from_error=error)
    if completed.returncode != 0:
        msg = f"ffmpeg falló con código {completed.returncode}: {_tail(completed.stderr)}"
        inv.fail(msg)


def run_ffmpeg_with_fallback(argv: Sequence[str], inv: Invocation) -> None:
    """Ejecuta ffmpeg con reintento automático libx264 si h264_nvenc falla.

    Args:
        argv: Secuencia de argumentos de línea de comandos de ffmpeg.
        inv: Contrato de la invocacion (render, temporal, excepcion, runner,
            ``cwd`` y timeout con los que se lanza y se reintenta).
    """
    cmd_list = list(argv)
    try:
        completed = _execute_single(cmd_list, inv)
    except subprocess.CalledProcessError as error:
        if "h264_nvenc" in cmd_list:
            _execute_fallback(cmd_list, inv)
            return
        msg = f"ffmpeg falló: {error}"
        inv.fail(msg, from_error=error)

    if completed.returncode == 0:
        return
    if "h264_nvenc" in cmd_list:
        _execute_fallback(cmd_list, inv)
        return
    msg = f"ffmpeg falló con código {completed.returncode}: {_tail(completed.stderr)}"
    inv.fail(msg)


def _volume(value: float) -> str:
    return f"{value:.3f}"
