"""Selección del codificador de video para los pasos de render.

La política del brief §8 es usar la 1650Ti (``h264_nvenc``) cuando está
disponible y caer a ``libx264`` en CPU cuando no. La decisión se concentra aquí
para que reframe y subtítulos compartan una única fuente de verdad y no
divergen en silencio.
"""

from dataclasses import dataclass

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


def audio_injection_arguments(*, mix_ratio: float) -> tuple[str, ...]:
    """Devuelve los argumentos que reemplazan o mezclan la pista de audio.

    Con ``mix_ratio`` igual o superior a ``1.0`` la pista externa reemplaza a la
    original (``-map 1:a:0``) y ``-shortest`` recorta al vídeo. Con un valor
    intermedio ambas pistas se mezclan con ``amix`` y un volumen proporcional:
    ``1 - mix_ratio`` para la original y ``mix_ratio`` para la externa.

    Args:
        mix_ratio: Peso de la pista externa en la mezcla, en ``[0.0, 1.0]``.

    Returns:
        Los argumentos de ffmpeg posteriores a los dos ``-i`` de entrada.
    """
    if mix_ratio >= 1.0:
        return (
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-c:v",
            "copy",
            *audio_and_container_arguments(),
            "-shortest",
        )
    filter_graph = (
        f"[0:a]volume={_volume(1.0 - mix_ratio)}[original];"
        f"[1:a]volume={_volume(mix_ratio)}[external];"
        "[original][external]amix=inputs=2:duration=longest:dropout_transition=2[aout]"
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
    )


def _volume(value: float) -> str:
    return f"{value:.3f}"
