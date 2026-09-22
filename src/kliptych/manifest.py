"""Manifiesto reproducible de cada corrida (``runs/<run_id>/run_manifest.json``)."""

from pathlib import Path
from typing import ClassVar, Literal

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field

from kliptych.contract import AssetRef
from kliptych.environment import EnvironmentReport
from kliptych.gate import GateResult

_SHA256 = r"^[0-9a-f]{64}$"


class _ManifestBase(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", frozen=True)


class OutputHash(_ManifestBase):
    """Hash y tamaño de un artefacto de salida de la corrida."""

    path: str = Field(min_length=1)
    sha256: str = Field(pattern=_SHA256)
    size_bytes: int = Field(ge=0)


class RunManifest(_ManifestBase):
    """Registro reproducible de una corrida completa."""

    manifest_version: Literal["1.0"] = "1.0"
    run_id: str = Field(min_length=1)
    started_at: AwareDatetime
    finished_at: AwareDatetime | None = None
    brief_sha256: str | None = Field(default=None, pattern=_SHA256)
    contract_schema_version: str | None = None
    contract_sha256: str | None = Field(default=None, pattern=_SHA256)
    model_version: str | None = None
    prompt_version: str | None = None
    whisper_version: str | None = None
    ffmpeg_version: str | None = None
    render_arguments: tuple[str, ...] = ()
    environment: EnvironmentReport
    assets: tuple[AssetRef, ...] = ()
    outputs: tuple[OutputHash, ...] = ()
    gate: GateResult | None = None
    degradations: tuple[str, ...] = ()


def write_manifest(manifest: RunManifest, run_dir: Path) -> Path:
    """Escribe ``run_manifest.json`` dentro del directorio de la corrida.

    Args:
        manifest: Manifiesto a persistir.
        run_dir: Directorio de la corrida; se crea si no existe.

    Returns:
        La ruta del archivo escrito.
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "run_manifest.json"
    _ = path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
    return path


def read_manifest(path: Path) -> RunManifest:
    """Lee y valida un manifiesto desde disco.

    Args:
        path: Ruta del archivo ``run_manifest.json``.

    Returns:
        El manifiesto validado.
    """
    return RunManifest.model_validate_json(path.read_text(encoding="utf-8"))
