"""Descarga local de videos largos y chats de stream con herramientas externas.

El modo ``long_video`` necesita traer el video y su chat desde plataformas
(YouTube, Twitch, Kick, X) sin APIs en la nube: ``yt-dlp`` para VOD, ``streamlink``
para directos y ``chat-downloader`` para el chat-replay. Toda invocación usa
``subprocess`` con lista de argumentos (nunca ``shell=True``), timeout explícito
por ejecución y una cota dura del fichero descargado para evitar cuelgues y disco
infinito.

La ejecución se inyecta mediante ``DownloadRunner``. Se define un protocolo
propio en vez de reutilizar ``CommandRunner`` de ``environment.py`` porque aquel
contrato promete "no lanzar": absorbe ``subprocess.TimeoutExpired`` y lo devuelve
como ``CommandResult(ok=False)``. La descarga sí necesita distinguir el timeout
del resto de fallos para traducirlo a ``DownloadError`` con causa encadenada, de
modo que el protocolo de descarga deja propagar ``subprocess.TimeoutExpired``.

Cota de tamaño: ``yt-dlp`` la aplica de forma nativa con ``--max-filesize``.
``streamlink`` y ``chat-downloader`` no exponen una cota nativa de bytes, así que
``_run_download`` verifica el tamaño del fichero producido y falla con un
``DownloadError`` claro si excede ``max_size_bytes``; nunca se delega la cota en
un ``--limit`` del operador. Los builders de argumentos son públicos (como
``FFmpegAssembler.render_arguments``) para poder afirmar en tests que la receta
jamás contiene shell ni omite los límites.
"""

import subprocess
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

from kliptych.environment import CommandResult

DEFAULT_TIMEOUT_S = 600.0
DEFAULT_MAX_SIZE_BYTES = 2 * 1024 * 1024 * 1024
_TOOL_PROBE_TIMEOUT_S = 15.0
_STDERR_TAIL = 400

_YTDLP = "yt-dlp"
_STREAMLINK = "streamlink"
_CHAT_DOWNLOADER = "chat_downloader"


class DownloadError(Exception):
    """La descarga no se pudo completar."""


class DownloadRunner(Protocol):
    """Ejecuta una herramienta de descarga y devuelve su resultado.

    A diferencia de ``environment.CommandRunner``, este contrato permite que
    ``run`` propague ``subprocess.TimeoutExpired``: ``MediaDownloader`` lo
    traduce a ``DownloadError`` conservando la causa.
    """

    def run(self, argv: Sequence[str], *, timeout_s: float) -> CommandResult:
        """Ejecuta ``argv`` sin shell.

        Args:
            argv: Lista de argumentos, nunca una cadena con shell.
            timeout_s: Timeout máximo de la invocación, en segundos.

        Returns:
            El resultado del comando; ``ok=False`` si el binario no existe o el
            comando falla.

        Raises:
            subprocess.TimeoutExpired: Si el comando excede ``timeout_s``.
        """
        ...


class SubprocessDownloadRunner:
    """Runner real basado en ``subprocess``, con lista de argumentos y timeout.

    Propaga ``subprocess.TimeoutExpired`` (la descarga debe distinguirlo) y
    traduce los ``OSError`` de ejecución a ``CommandResult(ok=False)``.
    """

    @staticmethod
    def run(argv: Sequence[str], *, timeout_s: float) -> CommandResult:
        """Ejecuta la herramienta capturando salida y errores.

        Args:
            argv: Lista de argumentos, nunca una cadena con shell.
            timeout_s: Timeout máximo de la invocación, en segundos.

        Returns:
            El resultado del comando; ``ok=False`` si el binario no existe o el
            comando falla.

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
    """Descarga videos largos y chats de stream con herramientas locales."""

    def __init__(
        self,
        *,
        runner: DownloadRunner | None = None,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        max_size_bytes: int = DEFAULT_MAX_SIZE_BYTES,
    ) -> None:
        """Configura el runner, el timeout y la cota de tamaño.

        Args:
            runner: Ejecutor inyectable; por defecto usa ``subprocess``.
            timeout_s: Timeout máximo por invocación, en segundos.
            max_size_bytes: Cota máxima del fichero descargado, en bytes.

        Raises:
            ValueError: Si ``timeout_s`` o ``max_size_bytes`` no son positivos.
        """
        if timeout_s <= 0:
            msg = f"timeout_s debe ser positivo, no {timeout_s}"
            raise ValueError(msg)
        if max_size_bytes <= 0:
            msg = f"max_size_bytes debe ser positivo, no {max_size_bytes}"
            raise ValueError(msg)
        self._runner: DownloadRunner = runner if runner is not None else SubprocessDownloadRunner()
        self._timeout_s: float = timeout_s
        self._max_size_bytes: int = max_size_bytes

    def has(self, tool: str) -> bool:
        """Indica si la herramienta está disponible en el PATH.

        Args:
            tool: Nombre del binario (``yt-dlp``, ``streamlink``,
                ``chat_downloader``).

        Returns:
            ``True`` si responde a ``--version`` con éxito; si no, ``False``.
        """
        try:
            result = self._runner.run([tool, "--version"], timeout_s=_TOOL_PROBE_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            return False
        return result.ok

    def download_video(
        self,
        *,
        url: str,
        destination: Path,
        format_selector: str | None = None,
    ) -> Path:
        """Descarga un video con ``yt-dlp``.

        Args:
            url: URL del video (YouTube, Twitch VOD, Kick, X...).
            destination: Ruta final del fichero descargado.
            format_selector: Selector de formato de yt-dlp (``--format``),
                opcional.

        Returns:
            La ruta del fichero descargado.

        Raises:
            DownloadError: Si ``yt-dlp`` no está disponible, expira el timeout,
                se excede la cota de tamaño o el comando falla.
        """
        if not self.has(_YTDLP):
            msg = f"{_YTDLP} no está disponible para descargar {url}"
            raise DownloadError(msg)
        argv = self.build_ytdlp_argv(
            url=url,
            destination=destination,
            format_selector=format_selector,
        )
        return self._run_download(argv, destination)

    def download_stream(self, *, url: str, destination: Path) -> Path:
        """Descarga un directo con ``streamlink``.

        ``streamlink`` no expone una cota nativa de bytes; el límite de tamaño
        lo aplica ``_run_download`` midiendo el fichero producido.

        Args:
            url: URL del directo (Twitch, Kick...).
            destination: Ruta final del fichero descargado.

        Returns:
            La ruta del fichero descargado.

        Raises:
            DownloadError: Si ``streamlink`` no está disponible, expira el
                timeout, se excede la cota de tamaño o el comando falla.
        """
        if not self.has(_STREAMLINK):
            msg = f"{_STREAMLINK} no está disponible para descargar {url}"
            raise DownloadError(msg)
        argv = self.build_streamlink_argv(url=url, destination=destination)
        return self._run_download(argv, destination)

    def download_chat(self, *, url: str, destination: Path) -> Path:
        """Descarga el chat del stream con ``chat-downloader``.

        Args:
            url: URL del stream o video cuyo chat se descarga.
            destination: Ruta final del fichero de chat (JSON).

        Returns:
            La ruta del fichero de chat descargado.

        Raises:
            DownloadError: Si ``chat_downloader`` no está disponible, expira el
                timeout, se excede la cota de tamaño o el comando falla.
        """
        if not self.has(_CHAT_DOWNLOADER):
            msg = f"{_CHAT_DOWNLOADER} no está disponible para descargar el chat de {url}"
            raise DownloadError(msg)
        argv = self.build_chat_argv(url=url, destination=destination)
        return self._run_download(argv, destination)

    def build_ytdlp_argv(
        self,
        *,
        url: str,
        destination: Path,
        format_selector: str | None = None,
    ) -> list[str]:
        """Devuelve el argv de ``yt-dlp`` para esta descarga.

        La cota de tamaño es nativa (``--max-filesize``), así que yt-dlp aborta
        el download en cuanto el fichero la excede.

        Args:
            url: URL del video.
            destination: Ruta final del fichero.
            format_selector: Selector de formato opcional.

        Returns:
            El argv completo, como lista de argumentos (nunca shell).
        """
        argv = [
            _YTDLP,
            "--no-playlist",
            "--no-progress",
            "--no-part",
            "--max-filesize",
            str(self._max_size_bytes),
            "--output",
            str(destination),
        ]
        if format_selector is not None:
            argv += ["--format", format_selector]
        argv.append(url)
        return argv

    @staticmethod
    def build_streamlink_argv(*, url: str, destination: Path) -> list[str]:
        """Devuelve el argv de ``streamlink`` para esta descarga de directo.

        Decisión documentada: streamlink no ofrece una cota nativa de bytes, así
        que aquí solo se acotan los segmentos y el timeout global; la cota de
        tamaño la verifica ``_run_download`` sobre el fichero resultante.

        Args:
            url: URL del directo.
            destination: Ruta final del fichero.

        Returns:
            El argv completo, como lista de argumentos (nunca shell).
        """
        return [
            _STREAMLINK,
            "--output",
            str(destination),
            "--force",
            "--stream-segment-timeout",
            "10.0",
            url,
        ]

    @staticmethod
    def build_chat_argv(*, url: str, destination: Path) -> list[str]:
        """Devuelve el argv de ``chat-downloader`` para esta descarga de chat.

        Args:
            url: URL del stream o video.
            destination: Ruta final del fichero JSON.

        Returns:
            El argv completo, como lista de argumentos (nunca shell).
        """
        return [_CHAT_DOWNLOADER, "--output", str(destination), url]

    def _run_download(self, argv: Sequence[str], destination: Path) -> Path:
        tool = Path(argv[0]).name
        try:
            destination.parent.mkdir(parents=True, exist_ok=True)
        except OSError as error:
            msg = f"no se pudo preparar el directorio del destino {destination}: {error}"
            raise DownloadError(msg) from error
        try:
            result = self._runner.run(argv, timeout_s=self._timeout_s)
        except subprocess.TimeoutExpired as error:
            msg = f"{tool} excedió el timeout de {self._timeout_s} s"
            raise DownloadError(msg) from error
        except OSError as error:
            msg = f"no se pudo ejecutar {tool}: {error}"
            raise DownloadError(msg) from error
        if not result.ok:
            msg = f"{tool} falló con código distinto de cero: {_tail(result.stderr)}"
            raise DownloadError(msg)
        return _require_download(
            destination,
            max_size_bytes=self._max_size_bytes,
            tool=tool,
        )


def _require_download(destination: Path, *, max_size_bytes: int, tool: str) -> Path:
    if not destination.is_file():
        msg = f"{tool} no generó el fichero {destination}"
        raise DownloadError(msg)
    try:
        size = destination.stat().st_size
    except OSError as error:
        msg = f"no se pudo medir el fichero descargado {destination}: {error}"
        raise DownloadError(msg) from error
    if size > max_size_bytes:
        msg = f"{tool} excedió la cota de {max_size_bytes} bytes ({size} bytes) en {destination}"
        raise DownloadError(msg)
    return destination


def _tail(text: str) -> str:
    stripped = text.strip()
    return stripped[-_STDERR_TAIL:]
