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
