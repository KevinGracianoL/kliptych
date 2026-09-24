"""Tests del motor de propuestas Git/GitHub (fase E).

El motor se prueba con un proveedor falso: cero red, cero Git real. Los
proveedores reales se prueban con runners y transportes HTTP falsos inyectados.
"""

import base64
import json
import threading
from collections.abc import Iterator, Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import ClassVar, Protocol, cast, override

import pytest
from pydantic import ValidationError

from kliptych.campaign_types import Campaign, CampaignStatus
from kliptych.environment import DEFAULT_TIMEOUT_S, CommandResult
from kliptych.git_proposals import (
    GitError,
    GitHubApiProvider,
    GitHubCliProvider,
    GitProvider,
    ProposalEngine,
    ProposalError,
    PullRequest,
    UrllibGitHubHttp,
)
from kliptych.intelligence import Archetype, ArchetypeClassification
from kliptych.runtime import HttpError, HttpResponse
from tests.support import make_contract

_URL = "https://github.com/owner/repo/pull/7"


def make_campaign(
    *,
    campaign_id: str = "camp-01",
    archetype: Archetype = Archetype.NEW_ARCHETYPE,
    variations: tuple[str, ...] = (),
    with_classification: bool = True,
) -> Campaign:
    classification = (
        ArchetypeClassification(
            archetype=archetype,
            rationale="justificación",
            variations=variations,
        )
        if with_classification
        else None
    )
    return Campaign(
        campaign_id=campaign_id,
        brief="brief crudo",
        status=CampaignStatus.PENDING,
        archetype=archetype,
        classification=classification,
        contract=make_contract(),
    )


@dataclass
class FakeProvider:
    existing_files: dict[str, str] = field(default_factory=dict)
    fail_with: GitError | None = None
    branches: list[tuple[str, str]] = field(default_factory=list)
    written: list[dict[str, str]] = field(default_factory=list)
    opened: list[dict[str, object]] = field(default_factory=list)

    def _maybe_fail(self) -> None:
        if self.fail_with is not None:
            raise self.fail_with

    def create_branch(self, *, base: str, name: str) -> str:
        self._maybe_fail()
        self.branches.append((base, name))
        return name

    def read_file(self, *, branch: str, path: str) -> str | None:
        self._maybe_fail()
        _ = branch
        return self.existing_files.get(path)

    def write_file(self, *, branch: str, path: str, content: str, message: str) -> str:
        self._maybe_fail()
        self.written.append(
            {"branch": branch, "path": path, "content": content, "message": message}
        )
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
        self._maybe_fail()
        self.opened.append(
            {
                "branch": branch,
                "base": base,
                "title": title,
                "body": body,
                "campaign_id": campaign_id,
                "archetype": archetype,
            }
        )
        return PullRequest(
            url=_URL,
            branch=branch,
            title=title,
            body=body,
            campaign_id=campaign_id,
            archetype=archetype,
        )


def make_engine(provider: FakeProvider) -> ProposalEngine:
    return ProposalEngine(provider=provider, base_branch="main")


def test_pull_request_is_frozen() -> None:
    pr = PullRequest(
        url=_URL,
        branch="proposal/camp-01",
        title="titulo",
        body="cuerpo",
        campaign_id="camp-01",
        archetype=Archetype.NEW_ARCHETYPE,
    )
    with pytest.raises(ValidationError):
        pr.url = "https://github.com/owner/repo/pull/8"


def test_pull_request_forbids_extra_fields() -> None:
    with pytest.raises(ValidationError, match="invented"):
        _ = PullRequest.model_validate(
            {
                "url": _URL,
                "branch": "proposal/camp-01",
                "title": "titulo",
                "body": "cuerpo",
                "campaign_id": "camp-01",
                "archetype": "NEW_ARCHETYPE",
                "invented": True,
            }
        )


@pytest.mark.parametrize(
    "missing",
    ["url", "branch", "title", "body", "campaign_id"],
)
def test_pull_request_requires_non_empty_text(missing: str) -> None:
    payload: dict[str, object] = {
        "url": _URL,
        "branch": "proposal/camp-01",
        "title": "titulo",
        "body": "cuerpo",
        "campaign_id": "camp-01",
        "archetype": "NEW_ARCHETYPE",
    }
    payload[missing] = ""
    with pytest.raises(ValidationError, match=missing):
        _ = PullRequest.model_validate(payload)


def test_fake_provider_satisfies_protocol() -> None:
    provider: GitProvider = FakeProvider()
    assert provider.create_branch(base="main", name="proposal/camp-01") == "proposal/camp-01"


def test_propose_new_archetype_creates_branch_config_and_pr() -> None:
    provider = FakeProvider()
    campaign = make_campaign()
    pr = make_engine(provider).propose(campaign, make_contract())

    assert provider.branches == [("main", "proposal/camp-01")]
    assert len(provider.written) == 1
    written = provider.written[0]
    assert written["path"] == "campaigns/pending/camp-01/config.json"
    payload = cast("dict[str, object]", json.loads(written["content"]))
    assert payload["archetype"] == "NEW_ARCHETYPE"
    assert provider.opened[0]["campaign_id"] == "camp-01"
    assert provider.opened[0]["archetype"] is Archetype.NEW_ARCHETYPE
    assert pr.url == _URL
    assert pr.branch == "proposal/camp-01"
    assert pr.campaign_id == "camp-01"


def test_propose_known_with_variation_appends_to_log() -> None:
    existing = "# Variaciones\n\n_(vacío por ahora)_\n"
    provider = FakeProvider(existing_files={"campaigns/variations.md": existing})
    campaign = make_campaign(
        archetype=Archetype.KNOWN_WITH_VARIATION,
        variations=("duration.max=45",),
    )
    pr = make_engine(provider).propose(campaign, make_contract())

    assert len(provider.written) == 1
    written = provider.written[0]
    assert written["path"] == "campaigns/variations.md"
    assert "## camp-01" in written["content"]
    assert "- variación: duration.max=45" in written["content"]
    assert "_(vacío por ahora)_" not in written["content"]
    assert pr.archetype is Archetype.KNOWN_WITH_VARIATION


def test_propose_known_with_variation_creates_log_when_absent() -> None:
    provider = FakeProvider()
    campaign = make_campaign(
        archetype=Archetype.KNOWN_WITH_VARIATION,
        variations=("duration.max=45",),
    )
    _ = make_engine(provider).propose(campaign, make_contract())
    content = provider.written[0]["content"]
    assert content.startswith("# Variaciones de campaña")
    assert "## camp-01" in content


def test_propose_known_raises() -> None:
    provider = FakeProvider()
    campaign = make_campaign(archetype=Archetype.KNOWN)
    with pytest.raises(ProposalError, match="KNOWN"):
        _ = make_engine(provider).propose(campaign, make_contract())
    assert provider.branches == []


def test_propose_requires_archetype() -> None:
    campaign = Campaign(campaign_id="camp-01", brief="brief crudo")
    with pytest.raises(ProposalError, match="arquetipo"):
        _ = make_engine(FakeProvider()).propose(campaign, make_contract())


def test_propose_rejects_unsafe_campaign_id() -> None:
    campaign = make_campaign(campaign_id="..")
    with pytest.raises(ProposalError, match="seguro"):
        _ = make_engine(FakeProvider()).propose(campaign, make_contract())


def test_propose_wraps_git_error_with_cause() -> None:
    provider = FakeProvider(fail_with=GitError("falló git"))
    with pytest.raises(ProposalError, match="materializar") as excinfo:
        _ = make_engine(provider).propose(make_campaign(), make_contract())
    assert isinstance(excinfo.value.__cause__, GitError)


@pytest.mark.parametrize(
    "target",
    [ProposalEngine, GitHubCliProvider, GitHubApiProvider, PullRequest],
)
def test_no_merge_surface_exists(target: type[object]) -> None:
    forbidden = [name for name in dir(target) if "merge" in name.lower()]
    assert forbidden == []


def test_no_provider_merge_method_exists() -> None:
    for provider in (GitHubCliProvider, GitHubApiProvider):
        assert not hasattr(provider, "merge_pull_request")
        assert not hasattr(provider, "auto_merge")
        assert not hasattr(provider, "approve")


@dataclass
class FakeRunner:
    results: list[CommandResult] = field(default_factory=list)
    calls: list[list[str]] = field(default_factory=list)

    def run(
        self,
        argv: Sequence[str],
        *,
        timeout_s: float = DEFAULT_TIMEOUT_S,
    ) -> CommandResult:
        _ = timeout_s
        self.calls.append(list(argv))
        assert self.results, "el runner falso no tiene resultados"
        return self.results.pop(0)


def _ok(stdout: str = "") -> CommandResult:
    return CommandResult(ok=True, stdout=stdout)


def test_cli_create_branch_runs_git_switch(tmp_path: Path) -> None:
    runner = FakeRunner(results=[_ok()])
    provider = GitHubCliProvider(workdir=tmp_path, repo="owner/repo", runner=runner)
    assert provider.create_branch(base="main", name="proposal/camp-01") == "proposal/camp-01"
    assert runner.calls == [
        ["git", "-C", str(tmp_path), "switch", "--create", "proposal/camp-01", "main"]
    ]


def test_cli_create_branch_failure_raises_git_error(tmp_path: Path) -> None:
    runner = FakeRunner(results=[CommandResult(ok=False, stderr="boom")])
    provider = GitHubCliProvider(workdir=tmp_path, repo="owner/repo", runner=runner)
    with pytest.raises(GitError, match="boom"):
        _ = provider.create_branch(base="main", name="proposal/camp-01")


def test_cli_read_file_returns_content(tmp_path: Path) -> None:
    runner = FakeRunner(results=[_ok("contenido")])
    provider = GitHubCliProvider(workdir=tmp_path, repo="owner/repo", runner=runner)
    assert provider.read_file(branch="proposal/camp-01", path="campaigns/variations.md") == (
        "contenido"
    )
    assert runner.calls[0][:3] == ["git", "-C", str(tmp_path)]
    assert runner.calls[0][3:5] == ["show", "proposal/camp-01:campaigns/variations.md"]


def test_cli_read_file_missing_returns_none(tmp_path: Path) -> None:
    runner = FakeRunner(results=[CommandResult(ok=False, stderr="missing")])
    provider = GitHubCliProvider(workdir=tmp_path, repo="owner/repo", runner=runner)
    assert provider.read_file(branch="proposal/camp-01", path="campaigns/variations.md") is None


def test_cli_read_file_rejects_traversal(tmp_path: Path) -> None:
    provider = GitHubCliProvider(workdir=tmp_path, repo="owner/repo", runner=FakeRunner())
    with pytest.raises(GitError, match="ruta"):
        _ = provider.read_file(branch="proposal/camp-01", path="../secreto.md")


def test_cli_write_file_commits_and_returns_path(tmp_path: Path) -> None:
    runner = FakeRunner(results=[_ok(), _ok(), _ok()])
    provider = GitHubCliProvider(workdir=tmp_path, repo="owner/repo", runner=runner)
    path = "campaigns/pending/camp-01/config.json"
    result = provider.write_file(
        branch="proposal/camp-01",
        path=path,
        content='{"a": 1}',
        message="feat: propone config",
    )
    assert result == path
    assert (tmp_path / path).read_text(encoding="utf-8") == '{"a": 1}'
    assert runner.calls[1] == ["git", "-C", str(tmp_path), "add", "--", path]
    assert runner.calls[2] == ["git", "-C", str(tmp_path), "commit", "-m", "feat: propone config"]


def test_cli_open_pull_request_pushes_and_returns_pr(tmp_path: Path) -> None:
    runner = FakeRunner(results=[_ok(), _ok(_URL)])
    provider = GitHubCliProvider(workdir=tmp_path, repo="owner/repo", runner=runner)
    pr = provider.open_pull_request(
        branch="proposal/camp-01",
        base="main",
        title="titulo",
        body="cuerpo",
        campaign_id="camp-01",
        archetype=Archetype.NEW_ARCHETYPE,
    )
    assert pr.url == _URL
    assert runner.calls[0] == [
        "git",
        "-C",
        str(tmp_path),
        "push",
        "--set-upstream",
        "origin",
        "proposal/camp-01",
    ]
    assert runner.calls[1] == [
        "gh",
        "pr",
        "create",
        "--repo",
        "owner/repo",
        "--base",
        "main",
        "--head",
        "proposal/camp-01",
        "--title",
        "titulo",
        "--body",
        "cuerpo",
    ]


def test_cli_open_pull_request_without_url_raises(tmp_path: Path) -> None:
    runner = FakeRunner(results=[_ok(), _ok("")])
    provider = GitHubCliProvider(workdir=tmp_path, repo="owner/repo", runner=runner)
    with pytest.raises(GitError, match="URL"):
        _ = provider.open_pull_request(
            branch="proposal/camp-01",
            base="main",
            title="titulo",
            body="cuerpo",
            campaign_id="camp-01",
            archetype=Archetype.NEW_ARCHETYPE,
        )


@dataclass
class FakeHttp:
    responses: list[HttpResponse] = field(default_factory=list)
    calls: list[dict[str, object]] = field(default_factory=list)

    def request(
        self,
        method: str,
        url: str,
        *,
        headers: Mapping[str, str],
        payload: Mapping[str, object] | None,
        timeout_s: float,
    ) -> HttpResponse:
        self.calls.append(
            {
                "method": method,
                "url": url,
                "headers": dict(headers),
                "payload": None if payload is None else dict(payload),
                "timeout_s": timeout_s,
            }
        )
        assert self.responses, "el http falso no tiene respuestas"
        return self.responses.pop(0)


def _json_response(status: int, payload: object) -> HttpResponse:
    return HttpResponse(status=status, body=json.dumps(payload).encode("utf-8"))


def _api(http: FakeHttp) -> GitHubApiProvider:
    return GitHubApiProvider(
        repo="owner/repo", token="secret", http=http, base_url="https://api.test"
    )


def test_api_create_branch_resolves_base_sha() -> None:
    http = FakeHttp(
        responses=[
            _json_response(HTTPStatus.OK, {"object": {"sha": "abc123"}}),
            _json_response(HTTPStatus.CREATED, {}),
        ]
    )
    assert _api(http).create_branch(base="main", name="proposal/camp-01") == "proposal/camp-01"
    assert http.calls[0]["method"] == "GET"
    assert cast("str", http.calls[0]["url"]).endswith("/repos/owner/repo/git/ref/heads/main")
    assert http.calls[1]["method"] == "POST"
    assert cast("str", http.calls[1]["url"]).endswith("/repos/owner/repo/git/refs")
    assert http.calls[1]["payload"] == {
        "ref": "refs/heads/proposal/camp-01",
        "sha": "abc123",
    }


def test_api_headers_carry_bearer_token() -> None:
    http = FakeHttp(
        responses=[
            _json_response(HTTPStatus.OK, {"object": {"sha": "abc"}}),
            _json_response(HTTPStatus.CREATED, {}),
        ]
    )
    _ = _api(http).create_branch(base="main", name="proposal/camp-01")
    headers = cast("dict[str, str]", http.calls[0]["headers"])
    assert headers["Authorization"] == "Bearer secret"
    assert headers["Accept"] == "application/vnd.github+json"


def test_api_create_branch_missing_base_raises() -> None:
    http = FakeHttp(responses=[_json_response(HTTPStatus.NOT_FOUND, {"message": "Not Found"})])
    with pytest.raises(GitError, match="404"):
        _ = _api(http).create_branch(base="main", name="proposal/camp-01")


def test_api_read_file_decodes_content() -> None:
    encoded = base64.b64encode(b"hola").decode()
    http = FakeHttp(responses=[_json_response(HTTPStatus.OK, {"type": "file", "content": encoded})])
    content = _api(http).read_file(branch="proposal/camp-01", path="campaigns/variations.md")
    assert content == "hola"
    assert "ref=proposal%2Fcamp-01" in cast("str", http.calls[0]["url"])


def test_api_read_file_missing_returns_none() -> None:
    http = FakeHttp(responses=[_json_response(HTTPStatus.NOT_FOUND, {"message": "Not Found"})])
    assert _api(http).read_file(branch="proposal/camp-01", path="campaigns/variations.md") is None


def test_api_write_file_creates_without_sha() -> None:
    http = FakeHttp(
        responses=[
            _json_response(HTTPStatus.NOT_FOUND, {"message": "Not Found"}),
            _json_response(HTTPStatus.CREATED, {}),
        ]
    )
    path = "campaigns/pending/camp-01/config.json"
    assert (
        _api(http).write_file(branch="proposal/camp-01", path=path, content="{}", message="feat: x")
        == path
    )
    payload = cast("dict[str, object]", http.calls[1]["payload"])
    assert "sha" not in payload
    assert payload["branch"] == "proposal/camp-01"
    assert base64.b64decode(cast("str", payload["content"])) == b"{}"


def test_api_write_file_updates_with_sha() -> None:
    encoded = base64.b64encode(b"viejo").decode()
    http = FakeHttp(
        responses=[
            _json_response(HTTPStatus.OK, {"type": "file", "sha": "deadbeef", "content": encoded}),
            _json_response(HTTPStatus.OK, {}),
        ]
    )
    _ = _api(http).write_file(
        branch="proposal/camp-01",
        path="campaigns/variations.md",
        content="nuevo",
        message="docs: x",
    )
    payload = cast("dict[str, object]", http.calls[1]["payload"])
    assert payload["sha"] == "deadbeef"


def test_api_open_pull_request_returns_pr() -> None:
    http = FakeHttp(responses=[_json_response(HTTPStatus.CREATED, {"html_url": _URL})])
    pr = _api(http).open_pull_request(
        branch="proposal/camp-01",
        base="main",
        title="titulo",
        body="cuerpo",
        campaign_id="camp-01",
        archetype=Archetype.NEW_ARCHETYPE,
    )
    assert pr.url == _URL
    payload = cast("dict[str, object]", http.calls[0]["payload"])
    assert payload["head"] == "proposal/camp-01"
    assert payload["base"] == "main"


def test_api_open_pull_request_error_maps_to_git_error() -> None:
    http = FakeHttp(responses=[_json_response(HTTPStatus.UNPROCESSABLE_ENTITY, {"message": "bad"})])
    with pytest.raises(GitError, match="422"):
        _ = _api(http).open_pull_request(
            branch="proposal/camp-01",
            base="main",
            title="titulo",
            body="cuerpo",
            campaign_id="camp-01",
            archetype=Archetype.NEW_ARCHETYPE,
        )


def test_api_wraps_http_error() -> None:
    @dataclass
    class ExplodingHttp:
        error: HttpError = field(default_factory=lambda: HttpError("sin conexión"))

        def request(
            self,
            method: str,
            url: str,
            *,
            headers: Mapping[str, str],
            payload: Mapping[str, object] | None,
            timeout_s: float,
        ) -> HttpResponse:
            _ = (method, url, headers, payload, timeout_s)
            raise self.error

    provider = GitHubApiProvider(
        repo="owner/repo", token="secret", http=ExplodingHttp(), base_url="https://api.test"
    )
    with pytest.raises(GitError, match="conexión"):
        _ = provider.create_branch(base="main", name="proposal/camp-01")


def test_api_rejects_unsafe_path() -> None:
    http = FakeHttp()
    with pytest.raises(GitError, match="ruta"):
        _ = _api(http).read_file(branch="proposal/camp-01", path="../secreto")


def test_api_rejects_unsafe_segment() -> None:
    http = FakeHttp()
    with pytest.raises(GitError, match="segmento"):
        _ = _api(http).read_file(branch="proposal/camp-01", path="campaigns/bad:name")


def test_api_read_file_rejects_non_file_content() -> None:
    http = FakeHttp(responses=[_json_response(HTTPStatus.OK, {"type": "dir"})])
    with pytest.raises(GitError, match="contenido"):
        _ = _api(http).read_file(branch="proposal/camp-01", path="campaigns")


def test_api_read_file_rejects_invalid_base64() -> None:
    http = FakeHttp(responses=[_json_response(HTTPStatus.OK, {"type": "file", "content": "a"})])
    with pytest.raises(GitError, match="base64"):
        _ = _api(http).read_file(branch="proposal/camp-01", path="campaigns/variations.md")


def test_api_open_pull_request_without_url_raises() -> None:
    http = FakeHttp(responses=[_json_response(HTTPStatus.CREATED, {})])
    with pytest.raises(GitError, match="URL"):
        _ = _api(http).open_pull_request(
            branch="proposal/camp-01",
            base="main",
            title="titulo",
            body="cuerpo",
            campaign_id="camp-01",
            archetype=Archetype.NEW_ARCHETYPE,
        )


def test_api_create_branch_without_object_raises() -> None:
    http = FakeHttp(responses=[_json_response(HTTPStatus.OK, {"object": "nope"})])
    with pytest.raises(GitError, match="objeto"):
        _ = _api(http).create_branch(base="main", name="proposal/camp-01")


def test_api_create_branch_without_sha_raises() -> None:
    http = FakeHttp(responses=[_json_response(HTTPStatus.OK, {"object": {}})])
    with pytest.raises(GitError, match="sha"):
        _ = _api(http).create_branch(base="main", name="proposal/camp-01")


def test_api_invalid_json_raises() -> None:
    http = FakeHttp(responses=[HttpResponse(status=HTTPStatus.OK, body=b"no-json")])
    with pytest.raises(GitError, match="JSON"):
        _ = _api(http).create_branch(base="main", name="proposal/camp-01")


def test_api_non_object_json_raises() -> None:
    http = FakeHttp(responses=[_json_response(HTTPStatus.OK, [])])
    with pytest.raises(GitError, match="objeto"):
        _ = _api(http).create_branch(base="main", name="proposal/camp-01")


def test_propose_without_classification_uses_archetype() -> None:
    provider = FakeProvider()
    campaign = make_campaign(with_classification=False)
    pr = make_engine(provider).propose(campaign, make_contract())
    assert pr.archetype is Archetype.NEW_ARCHETYPE
    assert provider.written[0]["path"] == "campaigns/pending/camp-01/config.json"


def test_propose_uses_classification_when_archetype_missing() -> None:
    provider = FakeProvider()
    campaign = make_campaign().model_copy(update={"archetype": None})
    pr = make_engine(provider).propose(campaign, make_contract())
    assert pr.archetype is Archetype.NEW_ARCHETYPE


class StartGitHubServer(Protocol):
    def __call__(
        self,
        body: bytes,
        *,
        status: int = 200,
        location: str | None = None,
        path: str = "/repos/owner/repo",
    ) -> str: ...


class _GitHubHandler(BaseHTTPRequestHandler):
    body: ClassVar[bytes] = b"{}"
    status: ClassVar[int] = 200
    location: ClassVar[str | None] = None
    requests: ClassVar[list[tuple[str, str]]] = []

    def do_GET(self) -> None:
        self._respond()

    def do_POST(self) -> None:
        self._respond()

    def do_PUT(self) -> None:
        self._respond()

    def _respond(self) -> None:
        _GitHubHandler.requests.append((self.command, self.path))
        self.send_response(_GitHubHandler.status)
        self.send_header("Content-Type", "application/json")
        if _GitHubHandler.location is not None:
            self.send_header("Location", _GitHubHandler.location)
        self.end_headers()
        with suppress(OSError):
            _ = self.wfile.write(_GitHubHandler.body)

    @override
    def log_message(self, format: str, *args: object) -> None:
        return


@pytest.fixture
def github_server() -> Iterator[StartGitHubServer]:
    servers: list[ThreadingHTTPServer] = []
    _GitHubHandler.requests.clear()

    def start(
        body: bytes,
        *,
        status: int = 200,
        location: str | None = None,
        path: str = "/repos/owner/repo",
    ) -> str:
        _GitHubHandler.body = body
        _GitHubHandler.status = status
        _GitHubHandler.location = location
        server = ThreadingHTTPServer(("127.0.0.1", 0), _GitHubHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        servers.append(server)
        host, port = server.server_address[:2]
        return f"http://{host}:{port}{path}"

    yield start

    for server in servers:
        server.shutdown()
        server.server_close()


def test_urllib_github_http_returns_response(github_server: StartGitHubServer) -> None:
    url = github_server(b'{"ok": true}')
    response = UrllibGitHubHttp().request("GET", url, headers={}, payload=None, timeout_s=5.0)
    assert response.status == 200
    assert response.body == b'{"ok": true}'
    assert _GitHubHandler.requests[-1] == ("GET", "/repos/owner/repo")


def test_urllib_github_http_maps_error_status(github_server: StartGitHubServer) -> None:
    url = github_server(b'{"message": "nope"}', status=HTTPStatus.NOT_FOUND)
    response = UrllibGitHubHttp().request("GET", url, headers={}, payload=None, timeout_s=5.0)
    assert response.status == 404
    assert response.body == b'{"message": "nope"}'


def test_urllib_github_http_enforces_size_limit(github_server: StartGitHubServer) -> None:
    url = github_server(b"x" * 4096)
    with pytest.raises(HttpError, match="límite"):
        _ = UrllibGitHubHttp(max_response_bytes=128).request(
            "GET",
            url,
            headers={},
            payload=None,
            timeout_s=5.0,
        )


def test_urllib_github_http_rejects_non_http_scheme() -> None:
    with pytest.raises(HttpError, match="esquema"):
        _ = UrllibGitHubHttp().request(
            "GET",
            "ftp://example.com/x",
            headers={},
            payload=None,
            timeout_s=1.0,
        )


def test_urllib_github_http_wraps_connection_errors() -> None:
    with pytest.raises(HttpError, match="conexión"):
        _ = UrllibGitHubHttp().request(
            "GET",
            "http://127.0.0.1:1/x",
            headers={},
            payload=None,
            timeout_s=1.0,
        )


def test_urllib_github_http_does_not_follow_redirects(
    github_server: StartGitHubServer,
) -> None:
    target = github_server(b"steal", path="/steal")
    source = github_server(b"", status=HTTPStatus.FOUND, location=target)
    response = UrllibGitHubHttp().request(
        "GET",
        source,
        headers={"Authorization": "Bearer x"},
        payload=None,
        timeout_s=5.0,
    )
    assert response.status == 302
    assert all(path != "/steal" for _, path in _GitHubHandler.requests)
