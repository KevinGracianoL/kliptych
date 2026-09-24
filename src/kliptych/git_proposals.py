"""Motor de propuestas Git/GitHub para decisiones de campaña (fase E).

Materializa una clasificación de campaña como una rama con su configuración y
un Pull Request. Nunca fusiona: no existe método de merge en ningún proveedor
ni en el motor, y el flujo termina al devolver la URL del PR.

Toda operación externa se invoca como lista de argumentos (nunca shell) o vía
la API REST con timeout explícito. Los proveedores son inyectables para poder
probarlos sin red ni Git real.
"""

import base64
import http.client
import json
import urllib.error
import urllib.request
from collections.abc import Mapping
from email.message import Message
from http import HTTPStatus
from pathlib import Path, PurePosixPath
from typing import IO, ClassVar, Protocol, cast, override
from urllib.parse import quote, urlencode, urlsplit

from pydantic import BaseModel, ConfigDict, Field

from kliptych.campaign_config import generate_campaign_config, variations_md_entry
from kliptych.campaign_types import Campaign
from kliptych.contract import Contract
from kliptych.environment import CommandResult, CommandRunner, SubprocessRunner
from kliptych.intelligence import Archetype
from kliptych.naming import is_safe_segment
from kliptych.runtime.transport import HttpError, HttpResponse

_DEFAULT_TIMEOUT_S = 60.0
_STDERR_TAIL = 400
_MAX_RESPONSE_BYTES = 5 * 1024 * 1024
_VARIATIONS_PATH = "campaigns/variations.md"
_PENDING_ROOT = "campaigns/pending"
_VARIATIONS_PLACEHOLDER = "_(vacío por ahora)_"
_VARIATIONS_HEADER = (
    "# Variaciones de campaña (KNOWN_WITH_VARIATION)\n\n"
    "Registro de valores nuevos dentro de campos existentes del contrato. Cada\n"
    "entrada referencia la campaña, el campo y el valor observado.\n"
)


class GitError(Exception):
    """La operación de Git/GitHub no se pudo completar."""


class PullRequest(BaseModel):
    """Propuesta de PR creada por el sistema."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", frozen=True)

    url: str = Field(min_length=1)
    branch: str = Field(min_length=1)
    title: str = Field(min_length=1)
    body: str = Field(min_length=1)
    campaign_id: str = Field(min_length=1)
    archetype: Archetype


class ProposalError(Exception):
    """La propuesta no se pudo crear o enviar."""


class GitProvider(Protocol):
    """Interfaz del proveedor de Git/GitHub."""

    def create_branch(self, *, base: str, name: str) -> str:
        """Crea una rama a partir de ``base`` y la deja activa.

        Args:
            base: Rama base desde la que partir.
            name: Nombre de la rama a crear.

        Returns:
            El nombre de la rama creada.

        Raises:
            GitError: Si la rama no se pudo crear.
        """
        ...

    def read_file(self, *, branch: str, path: str) -> str | None:
        """Lee un archivo de una rama.

        Args:
            branch: Rama de la que leer.
            path: Ruta relativa del archivo dentro del repositorio.

        Returns:
            El contenido UTF-8 del archivo, o ``None`` si no existe.

        Raises:
            GitError: Si la lectura falla por una razón distinta a la ausencia.
        """
        ...

    def write_file(self, *, branch: str, path: str, content: str, message: str) -> str:
        """Escribe y commitea un archivo en una rama.

        Reemplaza el contenido del archivo si ya existe.

        Args:
            branch: Rama destino del commit.
            path: Ruta relativa del archivo dentro del repositorio.
            content: Contenido UTF-8 completo del archivo.
            message: Mensaje del commit.

        Returns:
            La ruta del archivo escrito.

        Raises:
            GitError: Si el archivo no se pudo escribir o commitear.
        """
        ...

    def open_pull_request(
        self,
        *,
        branch: str,
        base: str,
        title: str,
        body: str,
        campaign_id: str,
        archetype: Archetype,
    ) -> PullRequest:
        """Abre un Pull Request de ``branch`` hacia ``base``.

        Args:
            branch: Rama de la propuesta.
            base: Rama base del Pull Request.
            title: Título del Pull Request.
            body: Cuerpo del Pull Request.
            campaign_id: Campaña que origina la propuesta.
            archetype: Arquetipo clasificado de la campaña.

        Returns:
            La propuesta con la URL del Pull Request abierto.

        Raises:
            GitError: Si el Pull Request no se pudo abrir.
        """
        ...


class ProposalEngine:
    """Motor que materializa decisiones de campaña como PRs."""

    def __init__(
        self,
        *,
        provider: GitProvider,
        base_branch: str = "main",
        variations_path: str = _VARIATIONS_PATH,
        pending_root: str = _PENDING_ROOT,
    ) -> None:
        """Configura el motor y su proveedor.

        Args:
            provider: Proveedor de Git/GitHub, real o inyectado en tests.
            base_branch: Rama base de las propuestas.
            variations_path: Ruta del log de variaciones.
            pending_root: Directorio de configuraciones pendientes.
        """
        self._provider: GitProvider = provider
        self._base_branch: str = base_branch
        self._variations_path: str = variations_path
        self._pending_root: str = pending_root

    def propose(self, campaign: Campaign, contract: Contract) -> PullRequest:
        """Materializa la decisión de campaña como una propuesta de PR.

        Crea la rama, escribe la configuración y abre el Pull Request. Nunca
        fusiona: el flujo termina en la URL devuelta.

        Args:
            campaign: Campaña clasificada (``NEW_ARCHETYPE`` o
                ``KNOWN_WITH_VARIATION``).
            contract: Contrato validado asociado a la campaña.

        Returns:
            La propuesta con la URL del Pull Request abierto.

        Raises:
            ProposalError: Si la campaña no requiere propuesta, no tiene
                arquetipo, su id no es seguro o el proveedor falla.
        """
        archetype = _proposable_archetype(campaign)
        branch = _branch_name(campaign)
        try:
            return self._materialize(campaign, contract, archetype, branch)
        except GitError as error:
            msg = f"la propuesta de {campaign.campaign_id} no se pudo materializar"
            raise ProposalError(msg) from error

    def _materialize(
        self,
        campaign: Campaign,
        contract: Contract,
        archetype: Archetype,
        branch: str,
    ) -> PullRequest:
        _ = self._provider.create_branch(base=self._base_branch, name=branch)
        path, content, message = self._artifact(campaign, contract, archetype, branch)
        _ = self._provider.write_file(
            branch=branch,
            path=path,
            content=content,
            message=message,
        )
        return self._provider.open_pull_request(
            branch=branch,
            base=self._base_branch,
            title=_proposal_title(campaign, archetype),
            body=_proposal_body(campaign, contract, archetype),
            campaign_id=campaign.campaign_id,
            archetype=archetype,
        )

    def _artifact(
        self,
        campaign: Campaign,
        contract: Contract,
        archetype: Archetype,
        branch: str,
    ) -> tuple[str, str, str]:
        if archetype is Archetype.NEW_ARCHETYPE:
            path = f"{self._pending_root}/{campaign.campaign_id}/config.json"
            content = generate_campaign_config(campaign, contract)
            message = f"feat(campaigns): propone arquetipo para {campaign.campaign_id}"
            return path, content, message
        existing = self._provider.read_file(branch=branch, path=self._variations_path)
        entry = variations_md_entry(campaign)
        message = f"docs(campaigns): registra variación de {campaign.campaign_id}"
        return self._variations_path, _append_entry(existing, entry), message


class GitHubCliProvider:
    """Proveedor real mediante GitHub CLI (gh)."""

    def __init__(
        self,
        *,
        workdir: Path,
        repo: str,
        runner: CommandRunner | None = None,
        remote: str = "origin",
        timeout_s: float = _DEFAULT_TIMEOUT_S,
    ) -> None:
        """Configura el proveedor sobre un clon local.

        Args:
            workdir: Directorio del clon local donde operar.
            repo: Slug ``owner/name`` que ``gh`` recibe con ``--repo`` (el
                runner no fija el directorio de trabajo, así que el repositorio
                se declara siempre de forma explícita).
            runner: Ejecutor de comandos; por defecto usa subprocess.
            remote: Nombre del remoto al que publicar la rama.
            timeout_s: Timeout máximo por comando, en segundos.
        """
        self._workdir: Path = workdir
        self._repo: str = repo
        self._runner: CommandRunner = SubprocessRunner() if runner is None else runner
        self._remote: str = remote
        self._timeout_s: float = timeout_s

    def create_branch(self, *, base: str, name: str) -> str:
        """Crea y activa la rama con ``git switch --create``.

        Args:
            base: Rama base desde la que partir.
            name: Nombre de la rama a crear.

        Returns:
            El nombre de la rama creada.

        Raises:
            GitError: Si ``git`` falla.
        """
        _ = self._git(["switch", "--create", name, base], action=f"crear la rama {name}")
        return name

    def read_file(self, *, branch: str, path: str) -> str | None:
        """Lee un archivo de la rama con ``git show``.

        Args:
            branch: Rama de la que leer.
            path: Ruta relativa del archivo.

        Returns:
            El contenido del archivo, o ``None`` si no existe en la rama.

        Raises:
            GitError: Si la ruta no es un segmento seguro.
        """
        _validate_repo_path(path)
        result = self._runner.run(
            ["git", "-C", str(self._workdir), "show", f"{branch}:{path}"],
            timeout_s=self._timeout_s,
        )
        return result.stdout if result.ok else None

    def write_file(self, *, branch: str, path: str, content: str, message: str) -> str:
        """Escribe el archivo en disco y lo commitea en la rama.

        Args:
            branch: Rama destino del commit.
            path: Ruta relativa del archivo.
            content: Contenido UTF-8 completo del archivo.
            message: Mensaje del commit.

        Returns:
            La ruta del archivo escrito.

        Raises:
            GitError: Si la ruta es insegura o ``git`` falla.
        """
        _validate_repo_path(path)
        _ = self._git(["switch", branch], action=f"cambiar a la rama {branch}")
        target = self._workdir / path
        target.parent.mkdir(parents=True, exist_ok=True)
        _ = target.write_text(content, encoding="utf-8")
        _ = self._git(["add", "--", path], action=f"preparar {path}")
        _ = self._git(["commit", "-m", message], action=f"commitear {path}")
        return path

    def open_pull_request(
        self,
        *,
        branch: str,
        base: str,
        title: str,
        body: str,
        campaign_id: str,
        archetype: Archetype,
    ) -> PullRequest:
        """Publica la rama y abre el PR con ``gh pr create``.

        Args:
            branch: Rama de la propuesta.
            base: Rama base del Pull Request.
            title: Título del Pull Request.
            body: Cuerpo del Pull Request.
            campaign_id: Campaña que origina la propuesta.
            archetype: Arquetipo clasificado de la campaña.

        Returns:
            La propuesta con la URL impresa por ``gh``.

        Raises:
            GitError: Si el push o ``gh`` fallan, o si no hay URL.
        """
        _ = self._git(
            ["push", "--set-upstream", self._remote, branch],
            action=f"publicar la rama {branch}",
        )
        argv = [
            "gh",
            "pr",
            "create",
            "--repo",
            self._repo,
            "--base",
            base,
            "--head",
            branch,
            "--title",
            title,
            "--body",
            body,
        ]
        result = self._run(argv, action="abrir el Pull Request")
        url = result.stdout.strip()
        if not url:
            msg = "gh no devolvió la URL del Pull Request"
            raise GitError(msg)
        return PullRequest(
            url=url,
            branch=branch,
            title=title,
            body=body,
            campaign_id=campaign_id,
            archetype=archetype,
        )

    def _git(self, args: list[str], *, action: str) -> CommandResult:
        return self._run(["git", "-C", str(self._workdir), *args], action=action)

    def _run(self, argv: list[str], *, action: str) -> CommandResult:
        result = self._runner.run(argv, timeout_s=self._timeout_s)
        if not result.ok:
            msg = f"no se pudo {action}: {_tail(result.stderr)}"
            raise GitError(msg)
        return result


class GitHubHttp(Protocol):
    """Interfaz HTTP que consume la API REST de GitHub."""

    def request(
        self,
        method: str,
        url: str,
        *,
        headers: Mapping[str, str],
        payload: Mapping[str, object] | None,
        timeout_s: float,
    ) -> HttpResponse:
        """Envía una petición HTTP y devuelve la respuesta.

        Args:
            method: Verbo HTTP (GET, POST, PUT...).
            url: URL absoluta http/https.
            headers: Cabeceras adicionales de la petición.
            payload: Cuerpo JSON serializable, o ``None`` para peticiones sin
                cuerpo.
            timeout_s: Timeout de conexión y lectura, en segundos.

        Returns:
            La respuesta con su estado y cuerpo.

        Raises:
            HttpError: Si la conexión falla o la respuesta excede el límite.
        """
        ...


class _NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Rechaza redirecciones: un 3xx no debe reenviar la credencial."""

    @override
    def redirect_request(
        self,
        req: urllib.request.Request,
        fp: IO[bytes],
        code: int,
        msg: str,
        headers: Message,
        newurl: str,
    ) -> urllib.request.Request | None:
        return None


_OPENER = urllib.request.build_opener(_NoRedirectHandler)


class UrllibGitHubHttp:
    """Cliente HTTP real para la API de GitHub (stdlib, sin shell)."""

    def __init__(self, *, max_response_bytes: int = _MAX_RESPONSE_BYTES) -> None:
        """Configura el tope de lectura de respuestas.

        Args:
            max_response_bytes: Máximo de bytes aceptados por respuesta.
        """
        self._max_response_bytes: int = max_response_bytes

    def request(
        self,
        method: str,
        url: str,
        *,
        headers: Mapping[str, str],
        payload: Mapping[str, object] | None,
        timeout_s: float,
    ) -> HttpResponse:
        """Envía una petición a la API de GitHub y acota la respuesta.

        Args:
            method: Verbo HTTP.
            url: URL absoluta http/https de la API.
            headers: Cabeceras adicionales de la petición.
            payload: Cuerpo JSON serializable, o ``None``.
            timeout_s: Timeout de conexión y lectura, en segundos.

        Returns:
            La respuesta con su estado y cuerpo acotado.

        Raises:
            HttpError: Si el esquema no es http/https, la conexión falla o la
                respuesta excede el límite de tamaño.
        """
        scheme = urlsplit(url).scheme
        if scheme not in {"http", "https"}:
            msg = f"esquema de URL no soportado: {scheme!r}"
            raise HttpError(msg)
        data = None if payload is None else json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json", **headers},
            method=method,
        )
        try:
            with _open_connection(request, timeout_s) as response:
                status = response.status
                body = _read_limited(response, self._max_response_bytes)
        except urllib.error.HTTPError as error:
            status = error.code
            body = _read_limited(error, self._max_response_bytes)
        except (OSError, http.client.HTTPException) as error:
            msg = f"falló la conexión con la API de GitHub: {error}"
            raise HttpError(msg) from error
        return HttpResponse(status=status, body=body)


def _open_connection(
    request: urllib.request.Request,
    timeout_s: float,
) -> http.client.HTTPResponse:
    response = cast("object", _OPENER.open(request, timeout=timeout_s))
    if isinstance(response, http.client.HTTPResponse):
        return response
    msg = "respuesta HTTP inesperada de la API de GitHub"
    raise HttpError(msg)


def _read_limited(
    response: http.client.HTTPResponse | urllib.error.HTTPError,
    max_bytes: int,
) -> bytes:
    body = response.read(max_bytes + 1)
    if len(body) > max_bytes:
        msg = f"la respuesta excede el límite de {max_bytes} bytes"
        raise HttpError(msg)
    return body


class GitHubApiProvider:
    """Proveedor real mediante API REST de GitHub."""

    def __init__(
        self,
        *,
        repo: str,
        token: str,
        http: GitHubHttp | None = None,
        base_url: str = "https://api.github.com",
        timeout_s: float = _DEFAULT_TIMEOUT_S,
    ) -> None:
        """Configura el proveedor y su credencial.

        Args:
            repo: Slug ``owner/name`` del repositorio.
            token: Token de acceso; nunca se registra en logs.
            http: Cliente HTTP a inyectar; por defecto el real.
            base_url: URL base de la API de GitHub.
            timeout_s: Timeout máximo por petición, en segundos.
        """
        self._repo: str = repo
        self._token: str = token
        self._http: GitHubHttp = UrllibGitHubHttp() if http is None else http
        self._base_url: str = base_url
        self._timeout_s: float = timeout_s

    def create_branch(self, *, base: str, name: str) -> str:
        """Crea la rama resolviendo primero el sha de la base.

        Args:
            base: Rama base desde la que partir.
            name: Nombre de la rama a crear.

        Returns:
            El nombre de la rama creada.

        Raises:
            GitError: Si la API falla o no devuelve el sha base.
        """
        sha = self._get_ref_sha(base)
        response = self._request(
            "POST",
            f"/repos/{self._repo}/git/refs",
            payload={"ref": f"refs/heads/{name}", "sha": sha},
        )
        _ensure_ok(response, "crear la rama")
        return name

    def read_file(self, *, branch: str, path: str) -> str | None:
        """Lee un archivo de la rama vía ``contents``.

        Args:
            branch: Rama de la que leer.
            path: Ruta relativa del archivo.

        Returns:
            El contenido UTF-8 del archivo, o ``None`` si no existe.

        Raises:
            GitError: Si la ruta es insegura, la API falla o el contenido no es
                base64 UTF-8 válido.
        """
        _validate_repo_path(path)
        response = self._request(
            "GET",
            f"/repos/{self._repo}/contents/{_quote_path(path)}",
            query={"ref": branch},
        )
        if response.status == HTTPStatus.NOT_FOUND:
            return None
        _ensure_ok(response, "leer el archivo")
        data = _json_object(response)
        content = data.get("content")
        if not isinstance(content, str):
            msg = f"la API de GitHub no devolvió contenido para {path}"
            raise GitError(msg)
        try:
            return base64.b64decode(content).decode("utf-8")
        except (ValueError, UnicodeDecodeError) as error:
            msg = f"el contenido de {path} no es base64 UTF-8 válido"
            raise GitError(msg) from error

    def write_file(self, *, branch: str, path: str, content: str, message: str) -> str:
        """Escribe el archivo vía ``contents``, actualizando si existe.

        Args:
            branch: Rama destino del commit.
            path: Ruta relativa del archivo.
            content: Contenido UTF-8 completo del archivo.
            message: Mensaje del commit.

        Returns:
            La ruta del archivo escrito.

        Raises:
            GitError: Si la ruta es insegura o la API falla.
        """
        _validate_repo_path(path)
        sha = self._existing_sha(branch, path)
        payload: dict[str, object] = {
            "message": message,
            "content": base64.b64encode(content.encode("utf-8")).decode("ascii"),
            "branch": branch,
        }
        if sha is not None:
            payload["sha"] = sha
        response = self._request(
            "PUT",
            f"/repos/{self._repo}/contents/{_quote_path(path)}",
            payload=payload,
        )
        _ensure_ok(response, "escribir el archivo")
        return path

    def open_pull_request(
        self,
        *,
        branch: str,
        base: str,
        title: str,
        body: str,
        campaign_id: str,
        archetype: Archetype,
    ) -> PullRequest:
        """Abre el Pull Request vía ``pulls``.

        Args:
            branch: Rama de la propuesta.
            base: Rama base del Pull Request.
            title: Título del Pull Request.
            body: Cuerpo del Pull Request.
            campaign_id: Campaña que origina la propuesta.
            archetype: Arquetipo clasificado de la campaña.

        Returns:
            La propuesta con la URL del Pull Request.

        Raises:
            GitError: Si la API falla o no devuelve la URL.
        """
        response = self._request(
            "POST",
            f"/repos/{self._repo}/pulls",
            payload={"title": title, "head": branch, "base": base, "body": body},
        )
        _ensure_ok(response, "abrir el Pull Request")
        data = _json_object(response)
        url = data.get("html_url")
        if not isinstance(url, str) or not url:
            msg = "la API de GitHub no devolvió la URL del Pull Request"
            raise GitError(msg)
        return PullRequest(
            url=url,
            branch=branch,
            title=title,
            body=body,
            campaign_id=campaign_id,
            archetype=archetype,
        )

    def _get_ref_sha(self, base: str) -> str:
        response = self._request(
            "GET",
            f"/repos/{self._repo}/git/ref/heads/{quote(base, safe='/')}",
        )
        _ensure_ok(response, "resolver la rama base")
        data = _json_object(response)
        obj = data.get("object")
        if not isinstance(obj, dict):
            msg = "la API de GitHub no devolvió el objeto de la referencia base"
            raise GitError(msg)
        sha = cast("dict[str, object]", obj).get("sha")
        if not isinstance(sha, str) or not sha:
            msg = "la API de GitHub no devolvió el sha de la referencia base"
            raise GitError(msg)
        return sha

    def _existing_sha(self, branch: str, path: str) -> str | None:
        response = self._request(
            "GET",
            f"/repos/{self._repo}/contents/{_quote_path(path)}",
            query={"ref": branch},
        )
        if response.status == HTTPStatus.NOT_FOUND:
            return None
        _ensure_ok(response, "leer el archivo existente")
        data = _json_object(response)
        sha = data.get("sha")
        return sha if isinstance(sha, str) and sha else None

    def _request(
        self,
        method: str,
        path: str,
        *,
        query: Mapping[str, str] | None = None,
        payload: Mapping[str, object] | None = None,
    ) -> HttpResponse:
        url = f"{self._base_url.rstrip('/')}{path}"
        if query is not None:
            url = f"{url}?{urlencode(query)}"
        try:
            return self._http.request(
                method,
                url,
                headers=self._headers(),
                payload=payload,
                timeout_s=self._timeout_s,
            )
        except HttpError as error:
            msg = f"falló la petición a la API de GitHub ({method} {path}): {error}"
            raise GitError(msg) from error

    def _headers(self) -> dict[str, str]:
        return {
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {self._token}",
            "X-GitHub-Api-Version": "2022-11-28",
        }


def _ensure_ok(response: HttpResponse, action: str) -> None:
    if HTTPStatus.OK <= response.status < HTTPStatus.MULTIPLE_CHOICES:
        return
    msg = f"no se pudo {action}: HTTP {response.status}"
    raise GitError(msg)


def _proposable_archetype(campaign: Campaign) -> Archetype:
    archetype = campaign.archetype
    if archetype is None and campaign.classification is not None:
        archetype = campaign.classification.archetype
    if archetype is None:
        msg = f"la campaña {campaign.campaign_id} no tiene arquetipo clasificado"
        raise ProposalError(msg)
    if archetype is Archetype.KNOWN:
        msg = f"la campaña {campaign.campaign_id} es KNOWN: no requiere propuesta"
        raise ProposalError(msg)
    return archetype


def _branch_name(campaign: Campaign) -> str:
    if not is_safe_segment(campaign.campaign_id):
        msg = f"campaign_id no es un segmento de rama seguro: {campaign.campaign_id!r}"
        raise ProposalError(msg)
    return f"proposal/{campaign.campaign_id}"


def _proposal_title(campaign: Campaign, archetype: Archetype) -> str:
    if archetype is Archetype.NEW_ARCHETYPE:
        return f"feat(campaigns): nuevo arquetipo para {campaign.campaign_id}"
    return f"docs(campaigns): variación registrada para {campaign.campaign_id}"


def _proposal_body(campaign: Campaign, contract: Contract, archetype: Archetype) -> str:
    lines = [
        f"Propuesta automática de Kliptych para la campaña `{campaign.campaign_id}`.",
        "",
        f"- Arquetipo: `{archetype.value}`",
        f"- Contrato: `{contract.campaign_id}`",
    ]
    classification = campaign.classification
    if classification is not None:
        lines.append(f"- Justificación: {classification.rationale}")
        if classification.variations:
            lines.append("- Variaciones detectadas:")
            lines.extend(f"  - {variation}" for variation in classification.variations)
    lines += [
        "",
        "Esta propuesta no se fusiona automáticamente: requiere aprobación del owner.",
    ]
    return "\n".join(lines)


def _append_entry(existing: str | None, entry: str) -> str:
    if existing is None:
        return f"{_VARIATIONS_HEADER}\n{entry}"
    cleaned = existing.replace(_VARIATIONS_PLACEHOLDER, "").rstrip()
    return f"{cleaned}\n\n{entry}"


def _validate_repo_path(path: str) -> None:
    candidate = PurePosixPath(path)
    if not path or candidate.is_absolute() or ".." in candidate.parts:
        msg = f"ruta de repositorio no permitida: {path!r}"
        raise GitError(msg)
    for part in candidate.parts:
        if not is_safe_segment(part):
            msg = f"segmento de ruta no permitido: {part!r}"
            raise GitError(msg)


def _quote_path(path: str) -> str:
    return quote(path, safe="/")


def _json_object(response: HttpResponse) -> dict[str, object]:
    try:
        data = cast("object", json.loads(response.body))
    except json.JSONDecodeError as error:
        msg = "la API de GitHub devolvió un JSON inválido"
        raise GitError(msg) from error
    if not isinstance(data, dict):
        msg = "la API de GitHub devolvió un JSON que no es un objeto"
        raise GitError(msg)
    return cast("dict[str, object]", data)


def _tail(text: str) -> str:
    return text.strip()[-_STDERR_TAIL:]
