"""Inspección de artefactos con ffprobe: lista de argumentos, sin shell."""

import subprocess
from pathlib import Path
from typing import ClassVar, Protocol

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from kliptych.gate.models import MediaInfo

_DEFAULT_TIMEOUT_S = 30.0


class ProbeError(Exception):
    """El artefacto no se pudo inspeccionar con ffprobe."""


class MediaProbe(Protocol):
    """Interfaz de inspección de medios que consume el gate."""

    def probe(self, path: Path) -> MediaInfo:
        """Inspecciona un archivo y devuelve sus metadatos.

        Args:
            path: Ruta del artefacto.

        Returns:
            Los metadatos del artefacto.

        Raises:
            ProbeError: Si el archivo no existe o no se puede inspeccionar.
        """
        ...


class _FFprobeStream(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="ignore")

    codec_type: str | None = None
    width: int | None = None
    height: int | None = None


class _FFprobeFormat(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="ignore")

    format_name: str | None = None
    duration: str | None = None


class _FFprobeDocument(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="ignore")

    streams: list[_FFprobeStream] = Field(default_factory=list)
    format: _FFprobeFormat | None = None


class FFprobeProbe:
    """Probe real basado en ffprobe, invocado como lista de argumentos."""

    def __init__(self, *, ffprobe: str = "ffprobe", timeout_s: float = _DEFAULT_TIMEOUT_S) -> None:
        """Configura el binario y el timeout de inspección.

        Args:
            ffprobe: Nombre o ruta del binario ffprobe.
            timeout_s: Timeout máximo de la inspección, en segundos.
        """
        self._ffprobe: str = ffprobe
        self._timeout_s: float = timeout_s

    def probe(self, path: Path) -> MediaInfo:
        """Inspecciona el archivo con ffprobe.

        Args:
            path: Ruta del artefacto a inspeccionar.

        Returns:
            Los metadatos del artefacto.

        Raises:
            ProbeError: Si el archivo no existe, ffprobe falla o expira.
        """
        if not path.is_file():
            msg = f"el archivo no existe: {path}"
            raise ProbeError(msg)
        argv = [
            self._ffprobe,
            "-v",
            "error",
            "-print_format",
            "json",
            "-show_format",
            "-show_streams",
            str(path),
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
            msg = f"ffprobe no está disponible: {self._ffprobe}"
            raise ProbeError(msg) from error
        except subprocess.TimeoutExpired as error:
            msg = f"ffprobe excedió el timeout de {self._timeout_s} s"
            raise ProbeError(msg) from error
        if completed.returncode != 0:
            msg = f"ffprobe falló con código {completed.returncode}: {completed.stderr.strip()}"
            raise ProbeError(msg)
        return _parse_probe_output(completed.stdout, path)


def _parse_probe_output(stdout: str, path: Path) -> MediaInfo:
    try:
        document = _FFprobeDocument.model_validate_json(stdout)
    except ValidationError as error:
        msg = f"salida de ffprobe inválida para {path}"
        raise ProbeError(msg) from error
    video = next(
        (stream for stream in document.streams if stream.codec_type == "video"),
        None,
    )
    duration_raw = document.format.duration if document.format is not None else None
    format_name = document.format.format_name if document.format is not None else None
    return MediaInfo(
        format_name=format_name if format_name is not None else "desconocido",
        duration_s=_parse_duration(duration_raw),
        has_video=video is not None,
        has_audio=any(stream.codec_type == "audio" for stream in document.streams),
        width=video.width if video is not None else None,
        height=video.height if video is not None else None,
    )


def _parse_duration(raw: str | None) -> float:
    if raw is None:
        return 0.0
    try:
        return float(raw)
    except ValueError:
        return 0.0
