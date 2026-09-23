"""Descarga acotada de media y chat con yt-dlp, streamlink y chat-downloader.

Toda descarga se invoca como lista de argumentos (nunca shell) y queda acotada
por un timeout explícito y un tamaño máximo. yt-dlp recibe ``--max-filesize``
de forma nativa; streamlink y chat-downloader se verifican después de
descargar, antes de dar el artefacto por bueno. El alcance termina en el
artefacto descargado: no hay transcripción ni ensamblado aquí.
"""

import contextlib
import subprocess
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

from kliptych.environment import CommandResult

_DEFAULT_TIMEOUT_S = 3600.0
_DEFAULT_MAX_SIZE_BYTES = 2 * 1024**3
_PROBE_TIMEOUT_S = 15.0
_STDERR_TAIL = 400

_YTDLP = "yt-dlp"
_STREAMLINK = "streamlink"
_CHAT_DOWNLOADER = "chat_downloader"


class DownloadError(Exception):
    """La descarga no se pudo completar dentro de los límites."""


class DownloadRunner(Protocol):
    """Interfaz para ejecutar comandos de descarga."""

    def run(self, argv: Sequence[str], *, timeout_s: float) -> CommandResult:
        """Ejecuta un comando de descarga y devuelve su resultado.

        Args:
            argv: Lista de argumentos, sin shell.
            timeout_s: Timeout máximo en segundos.

        Returns:
            El resultado del comando; ``ok=False`` si falló o no existe.

        Raises:
            subprocess.TimeoutExpired: Si el comando excede ``timeout_s``.
        """
        ...


class SubprocessDownloadRunner:
    """Runner real de descargas: lista de argumentos y timeout explícito.

    A diferencia del runner de detección, deja propagar
    ``subprocess.TimeoutExpired`` para que la descarga lo traduzca a
    ``DownloadError`` conservando la causa.
    """

    @staticmethod
    def run(argv: Sequence[str], *, timeout_s: float) -> CommandResult:
        """Ejecuta el comando capturando salida y errores.

        Args:
            argv: Lista de argumentos, sin shell.
            timeout_s: Timeout máximo en segundos.

        Returns:
            El resultado del comando; ``ok=False`` si el binario no existe.

        Raises:
            subprocess.TimeoutExpired: Si el comando excede ``timeout_s``.
        """
        try:
            completed = subprocess.run(
                list(argv),
                capture_output=True,
                text=True,
                errors="replace",
                timeout=timeout_s,
                check=False,
            )
        except OSError as error:
            return CommandResult(ok=False, stderr=str(error))
        return CommandResult(
            ok=completed.returncode == 0,
            stdout=completed.stdout or "",
            stderr=completed.stderr or "",
        )


class MediaDownloader:
    """Descarga acotada de media y chat mediante herramientas externas."""

    def __init__(
        self,
        *,
        runner: DownloadRunner | None = None,
        timeout_s: float = _DEFAULT_TIMEOUT_S,
        max_size_bytes: int = _DEFAULT_MAX_SIZE_BYTES,
    ) -> None:
        """Configura el runner y los límites de descarga.

        Args:
            runner: Ejecutor de comandos; por defecto usa subprocess.
            timeout_s: Timeout máximo por descarga, en segundos.
            max_size_bytes: Tamaño máximo del artefacto descargado, en bytes.

        Raises:
            ValueError: Si el timeout o el tamaño máximo no son positivos.
        """
        if timeout_s <= 0:
            msg = f"timeout inválido: {timeout_s}"
            raise ValueError(msg)
        if max_size_bytes <= 0:
            msg = f"tamaño máximo inválido: {max_size_bytes}"
            raise ValueError(msg)
        self._runner: DownloadRunner = runner if runner is not None else SubprocessDownloadRunner()
        self._timeout_s: float = timeout_s
        self._max_size_bytes: int = max_size_bytes

    def has(self, tool: str) -> bool:
        """Indica si la herramienta existe y responde a ``--version``.

        Args:
            tool: Nombre o ruta del binario a probar.

        Returns:
            ``True`` si el binario responde con éxito; ``False`` en otro caso.
        """
        try:
            result = self._runner.run([tool, "--version"], timeout_s=_PROBE_TIMEOUT_S)
        except (OSError, subprocess.SubprocessError):
            return False
        return result.ok

    def download_video(
        self,
        *,
        url: str,
        destination: Path,
        format_selector: str | None = None,
    ) -> Path:
        """Descarga un video con yt-dlp.

        Args:
            url: URL del video.
            destination: Ruta del artefacto descargado.
            format_selector: Selector de formato de yt-dlp, si se requiere.

        Returns:
            La ruta del artefacto descargado.

        Raises:
            DownloadError: Si yt-dlp no está disponible, la descarga falla,
                excede el timeout o supera el tamaño máximo.
        """
        argv = self.build_ytdlp_argv(
            url=url,
            destination=destination,
            max_size_bytes=self._max_size_bytes,
            format_selector=format_selector,
        )
        return self._run_download(argv, destination=destination, tool=_YTDLP)

    def download_stream(
        self,
        *,
        url: str,
        destination: Path,
        stream: str = "best",
    ) -> Path:
        """Descarga un stream con streamlink.

        Args:
            url: URL del stream.
            destination: Ruta del artefacto descargado.
            stream: Nombre del stream a elegir (por defecto ``best``).

        Returns:
            La ruta del artefacto descargado.

        Raises:
            DownloadError: Si streamlink no está disponible, la descarga
                falla, excede el timeout o supera el tamaño máximo.
        """
        argv = self.build_streamlink_argv(url=url, destination=destination, stream=stream)
        return self._run_download(argv, destination=destination, tool=_STREAMLINK)

    def download_chat(self, *, url: str, destination: Path) -> Path:
        """Descarga el chat-replay de un stream con chat-downloader.

        Args:
            url: URL del video o stream.
            destination: Ruta del artefacto JSON descargado.

        Returns:
            La ruta del artefacto descargado.

        Raises:
            DownloadError: Si chat-downloader no está disponible, la descarga
                falla, excede el timeout o supera el tamaño máximo.
        """
        argv = self.build_chat_argv(url=url, destination=destination)
        return self._run_download(argv, destination=destination, tool=_CHAT_DOWNLOADER)

    @staticmethod
    def build_ytdlp_argv(
        *,
        url: str,
        destination: Path,
        max_size_bytes: int,
        format_selector: str | None = None,
    ) -> list[str]:
        """Construye el argv de yt-dlp, con límite de tamaño nativo.

        Args:
            url: URL del video.
            destination: Ruta del artefacto descargado.
            max_size_bytes: Tamaño máximo aceptado, en bytes.
            format_selector: Selector de formato opcional.

        Returns:
            El argv completo, como lista de argumentos.
        """
        argv = [
            _YTDLP,
            "--no-playlist",
            "--no-progress",
            "--max-filesize",
            str(max_size_bytes),
            "--output",
            str(destination),
        ]
        if format_selector is not None:
            argv += ["--format", format_selector]
        argv.append(url)
        return argv

    @staticmethod
    def build_streamlink_argv(
        *,
        url: str,
        destination: Path,
        stream: str = "best",
    ) -> list[str]:
        """Construye el argv de streamlink con salida a archivo.

        Args:
            url: URL del stream.
            destination: Ruta del artefacto descargado.
            stream: Nombre del stream a elegir.

        Returns:
            El argv completo, como lista de argumentos.
        """
        return [
            _STREAMLINK,
            "--no-config",
            "--output",
            str(destination),
            "--force",
            "--progress",
            "no",
            "--default-stream",
            stream,
            url,
        ]

    @staticmethod
    def build_chat_argv(*, url: str, destination: Path) -> list[str]:
        """Construye el argv de chat-downloader con salida JSON a archivo.

        Args:
            url: URL del video o stream.
            destination: Ruta del artefacto JSON descargado.

        Returns:
            El argv completo, como lista de argumentos.
        """
        return [
            _CHAT_DOWNLOADER,
            "--quiet",
            "--overwrite",
            "--output",
            str(destination),
            url,
        ]

    def _run_download(self, argv: list[str], *, destination: Path, tool: str) -> Path:
        if not self.has(tool):
            msg = f"{tool} no está disponible"
            raise DownloadError(msg)
        try:
            destination.parent.mkdir(parents=True, exist_ok=True)
        except OSError as error:
            msg = f"no se pudo preparar el directorio del destino {destination}: {error}"
            raise DownloadError(msg) from error
        try:
            result = self._runner.run(argv, timeout_s=self._timeout_s)
        except subprocess.TimeoutExpired as error:
            _remove_quietly(destination)
            msg = f"{tool} excedió el timeout de {self._timeout_s} s"
            raise DownloadError(msg) from error
        except OSError as error:
            _remove_quietly(destination)
            msg = f"no se pudo ejecutar {tool}: {error}"
            raise DownloadError(msg) from error
        if not result.ok:
            _remove_quietly(destination)
            msg = f"{tool} falló: {_tail(result.stderr)}"
            raise DownloadError(msg)
        if not destination.is_file():
            msg = f"{tool} no produjo el artefacto esperado: {destination}"
            raise DownloadError(msg)
        size = destination.stat().st_size
        if size > self._max_size_bytes:
            _remove_quietly(destination)
            msg = (
                f"la descarga de {tool} excede el tamaño máximo de "
                f"{self._max_size_bytes} bytes: {size} bytes"
            )
            raise DownloadError(msg)
        return destination


def _remove_quietly(path: Path) -> None:
    # Limpieza best-effort del artefacto descargado ante cualquier fallo.
    with contextlib.suppress(OSError):
        path.unlink(missing_ok=True)


def _tail(text: str) -> str:
    return text.strip()[-_STDERR_TAIL:]
