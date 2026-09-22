"""Detección del entorno local: ffmpeg/ffprobe, NVENC y GPU."""

import subprocess
from collections.abc import Sequence
from typing import ClassVar, Protocol

from pydantic import BaseModel, ConfigDict

DEFAULT_TIMEOUT_S = 15.0
_GPU_FIELD_COUNT = 3


class _EnvBase(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", frozen=True)


class CommandResult(_EnvBase):
    """Resultado de un comando externo, sin excepciones."""

    ok: bool
    stdout: str = ""
    stderr: str = ""


class CommandRunner(Protocol):
    """Interfaz para ejecutar comandos externos de detección."""

    def run(self, argv: Sequence[str], *, timeout_s: float = DEFAULT_TIMEOUT_S) -> CommandResult:
        """Ejecuta un comando y devuelve su resultado sin lanzar.

        Args:
            argv: Lista de argumentos, sin shell.
            timeout_s: Timeout máximo en segundos.

        Returns:
            El resultado del comando; ``ok=False`` si falló o no existe.
        """
        ...


class SubprocessRunner:
    """Runner real basado en subprocess, con lista de argumentos y timeout."""

    @staticmethod
    def run(argv: Sequence[str], *, timeout_s: float = DEFAULT_TIMEOUT_S) -> CommandResult:
        """Ejecuta el comando capturando salida y errores.

        Args:
            argv: Lista de argumentos, sin shell.
            timeout_s: Timeout máximo en segundos.

        Returns:
            El resultado del comando; ``ok=False`` ante error, timeout o
            binario ausente.
        """
        try:
            completed = subprocess.run(
                list(argv),
                capture_output=True,
                text=True,
                timeout=timeout_s,
                check=False,
            )
        except (OSError, subprocess.SubprocessError) as error:
            return CommandResult(ok=False, stderr=str(error))
        return CommandResult(
            ok=completed.returncode == 0,
            stdout=completed.stdout,
            stderr=completed.stderr,
        )


class GpuInfo(_EnvBase):
    """GPU NVIDIA detectada en la máquina."""

    name: str
    vram_mib: int
    driver_version: str


class EnvironmentReport(_EnvBase):
    """Estado del entorno local relevante para el pipeline."""

    ffmpeg_version: str | None = None
    ffprobe_version: str | None = None
    nvenc_available: bool = False
    gpu: GpuInfo | None = None
    degradations: tuple[str, ...] = ()


def detect_environment(runner: CommandRunner) -> EnvironmentReport:
    """Detecta ffmpeg, ffprobe, NVENC y GPU con el runner dado.

    Args:
        runner: Ejecutor de comandos (real o falso en tests).

    Returns:
        El reporte del entorno, con degradaciones explicadas si falta algo.
    """
    degradations: list[str] = []
    ffmpeg_version = _tool_version(runner, "ffmpeg")
    if ffmpeg_version is None:
        degradations.append("ffmpeg no disponible: sin render ni medición de audio")
    ffprobe_version = _tool_version(runner, "ffprobe")
    if ffprobe_version is None:
        degradations.append("ffprobe no disponible: el gate no puede inspeccionar artefactos")
    nvenc = _detect_nvenc(runner, ffmpeg_version)
    if ffmpeg_version is not None and not nvenc:
        degradations.append("NVENC no disponible: el render caerá a CPU")
    gpu = _detect_gpu(runner)
    if gpu is None:
        degradations.append("sin GPU NVIDIA detectable: sin aceleración por hardware")
    return EnvironmentReport(
        ffmpeg_version=ffmpeg_version,
        ffprobe_version=ffprobe_version,
        nvenc_available=nvenc,
        gpu=gpu,
        degradations=tuple(degradations),
    )


def _tool_version(runner: CommandRunner, tool: str) -> str | None:
    result = runner.run([tool, "-version"])
    if not result.ok or not result.stdout:
        return None
    first_line = result.stdout.splitlines()[0]
    remainder = first_line.removeprefix(f"{tool} version ")
    return remainder.split(" ", 1)[0]


def _detect_nvenc(runner: CommandRunner, ffmpeg_version: str | None) -> bool:
    if ffmpeg_version is None:
        return False
    result = runner.run(["ffmpeg", "-hide_banner", "-encoders"])
    return result.ok and "h264_nvenc" in result.stdout


def _detect_gpu(runner: CommandRunner) -> GpuInfo | None:
    result = runner.run(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,driver_version",
            "--format=csv,noheader,nounits",
        ]
    )
    if not result.ok or not result.stdout:
        return None
    parts = [part.strip() for part in result.stdout.splitlines()[0].split(",")]
    if len(parts) != _GPU_FIELD_COUNT:
        return None
    name, vram, driver = parts
    if not name or not vram.isdigit():
        return None
    return GpuInfo(name=name, vram_mib=int(vram), driver_version=driver)
