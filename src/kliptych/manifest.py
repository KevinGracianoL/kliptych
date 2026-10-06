"""Manifiesto reproducible de cada corrida (``runs/<run_id>/run_manifest.json``)."""

from pathlib import Path
from typing import Annotated, ClassVar, Literal

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field

from kliptych.contract import AssetRef
from kliptych.environment import EnvironmentReport
from kliptych.gate import GateResult

_SHA256 = r"^[0-9a-f]{64}$"


class _ManifestBase(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", frozen=True)


class OutputHash(_ManifestBase):
    """Hash, tamaño y receta de render de un artefacto de salida."""

    path: str = Field(min_length=1)
    sha256: str = Field(pattern=_SHA256)
    size_bytes: int = Field(ge=0)
    render_arguments: tuple[str, ...] = ()


class RunManifest(_ManifestBase):
    """Registro reproducible de una corrida completa."""

    manifest_version: Literal["1.1"] = "1.1"
    run_id: str = Field(min_length=1)
    started_at: AwareDatetime
    finished_at: AwareDatetime | None = None
    brief_sha256: str | None = Field(default=None, pattern=_SHA256)
    # El manifiesto graba QUE version del contrato produjo la corrida, asi que
    # tiene que aceptar todas las que el contrato sepa leer. Sin "1.2" aqui,
    # escribir el manifiesto de cualquier corrida que use el contrato actual
    # revienta con un error de Literal que no explica nada. Este campo NO es el
    # `manifest_version` de arriba: ese versiona ESTE documento y no cambia.
    contract_schema_version: Literal["1.1", "1.2"] | None = None
    contract_sha256: str | None = Field(default=None, pattern=_SHA256)
    model_version: str | None = None
    prompt_version: str | None = None
    caption_prompt_version: str | None = None
    whisper_version: str | None = None
    ffmpeg_version: str | None = None
    # La geometria con la que se JUZGO el artefacto. Sin estos dos campos el
    # veredicto de `artifact.aspect` no es reconstruible desde el registro: dos
    # corridas con el mismo `contract_sha256` y el mismo artefacto pueden dar
    # veredictos distintos y producir manifiestos indistinguibles. El manifiesto
    # se documenta como "registro reproducible de una corrida completa", y un
    # veredicto que no se puede reproducir desde el registro no es un registro.
    # No va en `contract_digest` porque no es un campo del contrato: es estado de
    # la invocacion, como el resto de `RenderConfig`.
    target_aspect: float | None = Field(default=None, gt=0)
    aspect_tolerance: float | None = Field(default=None, ge=0)
    environment: EnvironmentReport
    assets: tuple[AssetRef, ...] = ()
    outputs: tuple[OutputHash, ...] = ()
    gates: tuple[GateResult, ...] = ()
    degradations: tuple[str, ...] = ()
    # Milisegundos por etapa, con la etapa como clave. El brief 10.1 pide
    # "argumentos de render, hardware detectado, hashes de los outputs" y NO pide
    # tiempo: sin esto el registro no puede responder si una etapa es el cuello
    # de botella, y una mitigacion que cueste "+20 min por stream" no se puede
    # evaluar sin saber cuanto costaba antes. Las etapas reanudadas desde cache no
    # aparecen: no se ejecutaron y no hay duracion que registrar.
    stage_durations_ms: dict[str, Annotated[int, Field(ge=0)]] = Field(default_factory=dict)
    manually_approved_rules: tuple[str, ...] = ()
    approved_by: str | None = None
    approved_at_utc: str | None = None


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
