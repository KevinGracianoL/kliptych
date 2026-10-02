"""Descarga acotada de media y chat con yt-dlp, streamlink y chat-downloader.

Toda descarga se invoca como lista de argumentos (nunca shell) y queda acotada
por un timeout explícito y un tamaño máximo. yt-dlp recibe ``--max-filesize``
de forma nativa; streamlink y chat-downloader se verifican después de
descargar, antes de dar el artefacto por bueno. El alcance termina en el
artefacto descargado: no hay transcripción ni ensamblado aquí.

La URL se valida antes de descargar (esquema http/https y sin destinos locales
o privados). Ante cualquier fallo se eliminan solo los artefactos creados por
esta llamada, incluidos los sidecars de yt-dlp; un archivo preexistente en
``destination`` nunca se toca.
"""

import contextlib
import glob
import http.client
import ipaddress
import os
import re
import signal
import subprocess
import sys
import urllib.request
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol, cast
from urllib.parse import urlsplit

from kliptych.environment import CommandResult

_DEFAULT_TIMEOUT_S = 3600.0
_DEFAULT_MAX_SIZE_BYTES = 2 * 1024**3
_PROBE_TIMEOUT_S = 15.0
_STDERR_TAIL = 400
_TERMINATE_GRACE_S = 5.0

_YTDLP = "yt-dlp"
_STREAMLINK = "streamlink"
_CHAT_DOWNLOADER = "chat_downloader"

_ALLOWED_SCHEMES = frozenset({"http", "https"})
_BLOCKED_NETWORKS = (
    ipaddress.ip_network("127.0.0.0/8"),
    ipaddress.ip_network("169.254.0.0/16"),
    ipaddress.ip_network("10.0.0.0/8"),
    ipaddress.ip_network("172.16.0.0/12"),
    ipaddress.ip_network("192.168.0.0/16"),
    ipaddress.ip_network("::1/128"),
    ipaddress.ip_network("fc00::/7"),
)

_YTDLP_SIDECAR_SUFFIXES = (".part", ".ytdl")
_YTDLP_FRAGMENT_PATTERNS = (".part-Frag*", ".f[0-9]*")
_YTDLP_APPENDED_CONTAINER_SUFFIXES = (".webm", ".mkv", ".mp4")
_YTDLP_FORMAT_ENV_VAR = "KLIPTYCH_YTDLP_FORMAT"
_DEFAULT_YTDLP_FORMAT = "bestvideo[height<=1080]+bestaudio/best[height<=1080]/best"

_KICK_VOD_PATH_MARKER = "/videos/"
_KICK_M3U8_SUFFIX = ".m3u8"
_KICK_PAGE_MAX_BYTES = 5 * 1024 * 1024
_KICK_BROWSER_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/126.0.0.0 Safari/537.36"
)
_KICK_STREAM_PATTERN = re.compile(r"https://stream\.kick\.com/[^\"'\s]+\.m3u8")


class _ReadablePage(Protocol):
    """Página HTTP mínima para extraer la playlist de Kick."""

    def read(self, size: int = -1, /) -> bytes:
        """Lee hasta ``size`` bytes de la página."""
        ...


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

    Aísla cada descarga en su propio grupo de procesos para poder matar todo el
    árbol ante un timeout: en Windows usa ``CREATE_NEW_PROCESS_GROUP`` y
    ``taskkill /T /F``; en POSIX usa ``start_new_session`` y ``os.killpg``. Deja
    propagar ``subprocess.TimeoutExpired`` para que la descarga lo traduzca a
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
            subprocess.TimeoutExpired: Si el comando excede ``timeout_s``; el
                árbol completo de procesos se termina antes de propagar.
        """
        try:
            process = _spawn(list(argv))
        except OSError as error:
            return CommandResult(ok=False, stderr=str(error))
        try:
            stdout, stderr = process.communicate(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            _terminate_process_tree(process)
            raise
        return CommandResult(
            ok=process.returncode == 0,
            stdout=stdout or "",
            stderr=stderr or "",
        )


def _spawn(command: list[str]) -> subprocess.Popen[str]:
    # Cada descarga lidera su propio grupo de procesos para poder matar el
    # árbol completo: flags nativos por plataforma, nunca shell.
    if sys.platform == "win32":
        return subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            errors="replace",
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
        )
    return subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        errors="replace",
        start_new_session=True,
    )


def _terminate_process_tree(process: subprocess.Popen[str]) -> None:
    # Matar solo al hijo directo deja vivos a los nietos (p. ej. el yt-dlp.exe
    # empaquetado con PyInstaller); se mata todo el árbol del proceso.
    if sys.platform == "win32":
        with contextlib.suppress(OSError):
            _ = subprocess.run(
                ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                capture_output=True,
                check=False,
            )
        _reap(process)
    else:
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.killpg(process.pid, signal.SIGTERM)
        if not _wait_briefly(process):
            with contextlib.suppress(ProcessLookupError, PermissionError):
                os.killpg(process.pid, signal.SIGKILL)
            _reap(process)


def _wait_briefly(process: subprocess.Popen[str]) -> bool:
    try:
        _ = process.wait(timeout=_TERMINATE_GRACE_S)
    except subprocess.TimeoutExpired:
        return False
    return True


def _reap(process: subprocess.Popen[str]) -> None:
    with contextlib.suppress(subprocess.TimeoutExpired):
        _ = process.wait(timeout=_TERMINATE_GRACE_S)


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
        section: tuple[float, float] | None = None,
    ) -> Path:
        """Descarga un video con yt-dlp.

        Ante cualquier fallo se eliminan los artefactos creados por esta llamada
        (incluidos los sidecars de yt-dlp como ``.part``); un archivo
        preexistente en ``destination`` queda intacto.

        Args:
            url: URL del video; debe ser http/https y no apuntar a una dirección
                local o privada.
            destination: Ruta del artefacto descargado.
            format_selector: Selector de formato de yt-dlp, si se requiere.
            section: Rango [start, end] en segundos para descarga quirúrgica.

        Returns:
            La ruta del artefacto descargado.

        Raises:
            DownloadError: Si la URL no es segura, yt-dlp no está disponible, la
                descarga falla, excede el timeout o supera el tamaño máximo.
        """
        _validate_url(url)
        resolved_url = resolve_kick_vod_stream_url(url)
        if format_selector is not None:
            effective_format = format_selector
        else:
            effective_format = os.environ.get(_YTDLP_FORMAT_ENV_VAR) or _DEFAULT_YTDLP_FORMAT
        argv = self.build_ytdlp_argv(
            url=resolved_url,
            destination=destination,
            max_size_bytes=self._max_size_bytes,
            format_selector=effective_format,
            section=section,
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

        Ante cualquier fallo se eliminan los artefactos creados por esta llamada
        (incluidos los sidecars de yt-dlp como ``.part``); un archivo
        preexistente en ``destination`` queda intacto.

        Args:
            url: URL del stream; debe ser http/https y no apuntar a una
                dirección local o privada.
            destination: Ruta del artefacto descargado.
            stream: Nombre del stream a elegir (por defecto ``best``).

        Returns:
            La ruta del artefacto descargado.

        Raises:
            DownloadError: Si la URL no es segura, streamlink no está
                disponible, la descarga falla, excede el timeout o supera el
                tamaño máximo.
        """
        _validate_url(url)
        argv = self.build_streamlink_argv(url=url, destination=destination, stream=stream)
        return self._run_download(argv, destination=destination, tool=_STREAMLINK)

    def download_chat(self, *, url: str, destination: Path) -> Path:
        """Descarga el chat-replay de un stream con chat-downloader.

        Ante cualquier fallo se eliminan los artefactos creados por esta llamada
        (incluidos los sidecars de yt-dlp como ``.part``); un archivo
        preexistente en ``destination`` queda intacto.

        Args:
            url: URL del video o stream; debe ser http/https y no apuntar a una
                dirección local o privada.
            destination: Ruta del artefacto JSON descargado.

        Returns:
            La ruta del artefacto descargado.

        Raises:
            DownloadError: Si la URL no es segura, chat-downloader no está
                disponible, la descarga falla, excede el timeout o supera el
                tamaño máximo.
        """
        _validate_url(url)
        argv = self.build_chat_argv(url=url, destination=destination)
        return self._run_download(argv, destination=destination, tool=_CHAT_DOWNLOADER)

    @staticmethod
    def build_ytdlp_argv(
        *,
        url: str,
        destination: Path,
        max_size_bytes: int,
        format_selector: str | None = None,
        section: tuple[float, float] | None = None,
    ) -> list[str]:
        """Construye el argv de yt-dlp, con límite de tamaño nativo.

        Args:
            url: URL del video.
            destination: Ruta del artefacto descargado.
            max_size_bytes: Tamaño máximo aceptado, en bytes.
            format_selector: Selector de formato opcional.
            section: Rango [start, end] en segundos para descarga quirúrgica con margen.

        Returns:
            El argv completo, como lista de argumentos; la URL va tras ``--``.
        """
        argv = [
            _YTDLP,
            "--ignore-config",
            "--no-playlist",
            "--no-progress",
            "--max-filesize",
            str(max_size_bytes),
            "--output",
            str(destination),
        ]
        if format_selector is not None:
            argv += ["--format", format_selector]
        if section is not None:
            start_sec, end_sec = section
            margin_start = max(0.0, start_sec - 10.0)
            margin_end = end_sec + 10.0
            start_str = (
                str(int(margin_start))
                if margin_start == int(margin_start)
                else f"{margin_start:.3f}"
            )
            end_str = str(int(margin_end)) if margin_end == int(margin_end) else f"{margin_end:.3f}"
            argv += [
                "--download-sections",
                f"*{start_str}-{end_str}",
                "--force-keyframes-at-cuts",
                "--merge-output-format",
                "mp4",
            ]
        if destination.suffix.lower() == ".mp4" and "--merge-output-format" not in argv:
            argv += ["--merge-output-format", "mp4"]
        argv += ["--", url]
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
            El argv completo, como lista de argumentos; la URL va tras ``--``.
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
            "--",
            url,
        ]

    @staticmethod
    def build_chat_argv(*, url: str, destination: Path) -> list[str]:
        """Construye el argv de chat-downloader con salida JSON a archivo.

        Args:
            url: URL del video o stream.
            destination: Ruta del artefacto JSON descargado.

        Returns:
            El argv completo, como lista de argumentos; la URL va tras ``--``.
        """
        return [
            _CHAT_DOWNLOADER,
            "--quiet",
            "--overwrite",
            "--output",
            str(destination),
            "--",
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
        preexisting = frozenset(path for path in _artifact_paths(destination) if path.exists())
        try:
            result = self._runner.run(argv, timeout_s=self._timeout_s)
        except subprocess.TimeoutExpired as error:
            _clean_download_artifacts(destination, preexisting=preexisting)
            msg = f"{tool} excedió el timeout de {self._timeout_s} s"
            raise DownloadError(msg) from error
        except OSError as error:
            _clean_download_artifacts(destination, preexisting=preexisting)
            msg = f"no se pudo ejecutar {tool}: {error}"
            raise DownloadError(msg) from error
        if not result.ok:
            _clean_download_artifacts(destination, preexisting=preexisting)
            _handle_download_failure(result, tool=tool, url=argv[-1] if argv else "")
        is_file, size = _stat_destination(destination, preexisting=preexisting)
        if not is_file:
            _recover_appended_container(destination, preexisting=preexisting)
            is_file, size = _stat_destination(destination, preexisting=preexisting)
        if not is_file:
            _clean_download_artifacts(destination, preexisting=preexisting)
            msg = f"{tool} no produjo el artefacto esperado: {destination}"
            raise DownloadError(msg)
        if size > self._max_size_bytes:
            _clean_download_artifacts(destination, preexisting=preexisting)
            msg = (
                f"la descarga de {tool} excede el tamaño máximo de "
                f"{self._max_size_bytes} bytes: {size} bytes"
            )
            raise DownloadError(msg)
        return destination


def _handle_download_failure(result: CommandResult, *, tool: str, url: str) -> None:
    if is_kick_url(url):
        stderr_lower = result.stderr.lower()
        if any(
            k in stderr_lower
            for k in ("cookie", "auth", "login", "403", "cloudflare", "bot", "sign in")
        ):
            msg = (
                f"{tool} falló en Kick (se requiere autenticación/cookies): {_tail(result.stderr)}"
            )
            raise DownloadError(msg)
        if any(
            k in stderr_lower
            for k in ("network", "connection", "timeout", "timed out", "unreachable")
        ):
            msg = f"{tool} falló en Kick por error de red: {_tail(result.stderr)}"
            raise DownloadError(msg)
        msg = f"{tool} falló en Kick: {_tail(result.stderr)}"
        raise DownloadError(msg)
    msg = f"{tool} falló: {_tail(result.stderr)}"
    raise DownloadError(msg)


def is_kick_url(url: str) -> bool:
    """Indica si una URL pertenece a Kick.

    Args:
        url: URL a verificar.

    Returns:
        True si el host es kick.com o un subdominio.
    """
    try:
        host = urlsplit(url).hostname
    except ValueError:
        return False
    if host:
        host = host.rstrip(".")
    return bool(host and (host == "kick.com" or host.endswith(".kick.com")))


def resolve_kick_vod_stream_url(url: str, timeout_s: float = 15.0) -> str:
    """Resuelve la URL directa ``.m3u8`` de un VOD de Kick.

    Kick retiró el endpoint ``/api/v1/video/{id}`` que usa el extractor de
    yt-dlp, pero la página del VOD incrusta la playlist maestra HLS
    (``https://stream.kick.com/.../master.m3u8``) en su HTML.

    Args:
        url: URL candidata; solo se intenta resolver si es de Kick, contiene
            ``/videos/`` y no termina ya en ``.m3u8``.
        timeout_s: Timeout en segundos para la petición de la página.

    Returns:
        La URL ``.m3u8`` encontrada en el HTML, o la URL original si no es un
        VOD de Kick, ya es directa, no se encontró playlist o falló la red.
    """
    if not (
        is_kick_url(url) and _KICK_VOD_PATH_MARKER in url and not url.endswith(_KICK_M3U8_SUFFIX)
    ):
        return url
    try:
        parts = urlsplit(url)
    except ValueError:
        return url
    if parts.scheme not in _ALLOWED_SCHEMES:
        return url
    try:
        request = urllib.request.Request(url, headers={"User-Agent": _KICK_BROWSER_USER_AGENT})
        with cast(
            "contextlib.AbstractContextManager[_ReadablePage]",
            urllib.request.urlopen(request, timeout=timeout_s),
        ) as page:
            raw = page.read(_KICK_PAGE_MAX_BYTES)
            html = raw.decode("utf-8", errors="replace")
        match = _KICK_STREAM_PATTERN.search(html)
    except (OSError, ValueError, http.client.HTTPException):
        return url
    if match is None:
        return url
    return match.group(0)


def _validate_url(url: str) -> None:
    # La URL entra a herramientas externas como último argumento; se rechaza
    # cualquier esquema que no sea http/https y cualquier destino local o
    # privado (loopback, link-local, rangos RFC 1918 y ULA IPv6).
    try:
        parts = urlsplit(url)
    except ValueError as error:
        msg = f"URL inválida: {url!r}"
        raise DownloadError(msg) from error
    if parts.scheme not in _ALLOWED_SCHEMES:
        msg = f"esquema de URL no permitido: {parts.scheme or url!r}"
        raise DownloadError(msg)
    host = parts.hostname
    if host is None:
        msg = f"URL sin host: {url!r}"
        raise DownloadError(msg)
    if is_kick_url(url):
        clean_path = parts.path.strip("/")
        if not clean_path:
            msg = f"URL de Kick incompleta sin canal o video: {url!r}"
            raise DownloadError(msg)
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        return
    if any(address in network for network in _BLOCKED_NETWORKS):
        msg = f"URL hacia una dirección local o privada no permitida: {host}"
        raise DownloadError(msg)
    if address.is_loopback or address.is_private:
        msg = f"URL hacia una dirección local o privada no permitida: {host}"
        raise DownloadError(msg)


def _recover_appended_container(destination: Path, *, preexisting: frozenset[Path]) -> None:
    # yt-dlp anexa el contenedor real cuando el merge no respetó el sufijo
    # pedido (p. ej. ``destino.mp4.webm`` con VP9/Opus sin
    # ``--merge-output-format mp4``). Si el destino falta pero existe el
    # artefacto anexado, se consolida renombrándolo al destino esperado.
    for appended_suffix in _YTDLP_APPENDED_CONTAINER_SUFFIXES:
        candidate = Path(f"{destination}{appended_suffix}")
        try:
            if not candidate.is_file():
                continue
        except OSError:
            continue
        try:
            _ = candidate.rename(destination)
        except OSError as error:
            _clean_download_artifacts(destination, preexisting=preexisting)
            msg = f"no se pudo consolidar el artefacto {candidate} en {destination}: {error}"
            raise DownloadError(msg) from error
        return


def _stat_destination(destination: Path, *, preexisting: frozenset[Path]) -> tuple[bool, int]:
    # Verifica el artefacto esperado sin lanzar si falta: devuelve
    # ``(es_archivo, tamaño)`` y solo falla si la verificación de E/S falla.
    try:
        is_file = destination.is_file()
        size = destination.stat().st_size if is_file else 0
    except OSError as error:
        _clean_download_artifacts(destination, preexisting=preexisting)
        msg = f"no se pudo verificar el artefacto {destination}: {error}"
        raise DownloadError(msg) from error
    return is_file, size


def _artifact_paths(destination: Path) -> set[Path]:
    # Rutas que yt-dlp puede dejar tras un fallo: el destino, sus sidecars
    # conocidos, los fragmentos previos al merge y el contenedor anexado
    # (p. ej. ``destino.mp4.webm`` cuando el merge no respetó el sufijo).
    # Nunca un glob amplio sobre ``destination.name`` que pudiera borrar
    # archivos ajenos.
    parent = destination.parent
    escaped = glob.escape(destination.name)
    paths = {destination}
    paths.update(parent / f"{destination.name}{suffix}" for suffix in _YTDLP_SIDECAR_SUFFIXES)
    paths.update(
        parent / f"{destination.name}{suffix}" for suffix in _YTDLP_APPENDED_CONTAINER_SUFFIXES
    )
    for pattern in _YTDLP_FRAGMENT_PATTERNS:
        with contextlib.suppress(OSError):
            paths.update(parent.glob(f"{escaped}{pattern}"))
    return paths


def _clean_download_artifacts(destination: Path, *, preexisting: frozenset[Path]) -> None:
    # Limpieza best-effort: solo se eliminan los artefactos que no existían
    # antes de esta descarga, preservando cualquier archivo previo.
    for path in _artifact_paths(destination):
        if path not in preexisting:
            _remove_quietly(path)


def _remove_quietly(path: Path) -> None:
    with contextlib.suppress(OSError):
        path.unlink(missing_ok=True)


def _tail(text: str) -> str:
    return text.strip()[-_STDERR_TAIL:]
