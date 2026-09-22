"""Registro de assets con sha256, MIME determinista y procedencia."""

from collections.abc import Mapping
from datetime import UTC, datetime
from mimetypes import guess_type
from pathlib import Path
from types import MappingProxyType
from typing import ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field

from kliptych.contract import AssetRef
from kliptych.hashing import sha256_file

_FALLBACK_MIME = "application/octet-stream"

# MIME determinista para los formatos del dominio: `mimetypes` depende del
# registro del sistema (p. ej. .ass resuelve a audio/aac en Windows).
_MEDIA_MIME: dict[str, str] = {
    ".ass": "text/x-ssa",
    ".jpeg": "image/jpeg",
    ".jpg": "image/jpeg",
    ".json": "application/json",
    ".m4a": "audio/mp4",
    ".mkv": "video/x-matroska",
    ".mov": "video/quicktime",
    ".mp3": "audio/mpeg",
    ".mp4": "video/mp4",
    ".png": "image/png",
    ".srt": "application/x-subrip",
    ".txt": "text/plain",
    ".wav": "audio/wav",
    ".webm": "video/webm",
    ".webp": "image/webp",
}


class AssetError(Exception):
    """Base de los errores del registro de assets."""


class AssetNotFoundError(AssetError):
    """El asset no existe en disco o no está registrado."""


class AssetAlreadyRegisteredError(AssetError):
    """Ya existe un asset registrado con ese asset_id."""


class UnsafeAssetPathError(AssetError):
    """La ruta del asset escapa de la raíz del workspace."""


class RegistryDocument(BaseModel):
    """Documento JSON del registro de assets, versionado."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["1.0"] = "1.0"
    assets: list[AssetRef] = Field(default_factory=list)


def _detect_mime(path: Path) -> str:
    explicit = _MEDIA_MIME.get(path.suffix.lower())
    if explicit is not None:
        return explicit
    guessed, _ = guess_type(path.name)
    return guessed if guessed is not None else _FALLBACK_MIME


class AssetRegistry:
    """Assets resueltos de un workspace, con hash verificado y ruta segura."""

    def __init__(self, root: Path, assets: Mapping[str, AssetRef] | None = None) -> None:
        """Crea un registro con la raíz del workspace.

        Args:
            root: Raíz del workspace; toda URI de asset debe quedar dentro.
            assets: Assets iniciales indexados por asset_id.
        """
        self._root: Path = root.resolve()
        self._assets: dict[str, AssetRef] = dict(assets or {})

    @property
    def assets(self) -> Mapping[str, AssetRef]:
        """Vista de solo lectura de los assets registrados.

        Returns:
            Mapeo asset_id -> AssetRef.
        """
        return MappingProxyType(self._assets)

    def register(
        self,
        *,
        asset_id: str,
        kind: str,
        uri: str,
        origin: str,
        license: str | None = None,
    ) -> AssetRef:
        """Registra un asset resolviendo ruta, hash, tamaño y MIME.

        La ruta debe ser relativa a la raíz del workspace; una ruta absoluta,
        con drive o UNC se rechaza sin tocar el filesystem (``Path.resolve``
        sobre UNC dispara DNS/SMB en Windows).

        Args:
            asset_id: Identificador único dentro del workspace.
            kind: Tipo de asset (video, image, audio, file, ...).
            uri: Ruta relativa a la raíz del workspace.
            origin: Procedencia declarada del asset.
            license: Licencia declarada, si existe.

        Returns:
            La referencia resuelta del asset.

        Raises:
            AssetAlreadyRegisteredError: Si ``asset_id`` ya está registrado.
            AssetNotFoundError: Si el archivo no existe en disco.
        """
        if asset_id in self._assets:
            msg = f"asset ya registrado: {asset_id}"
            raise AssetAlreadyRegisteredError(msg)
        path = self._resolve(uri)
        if not path.is_file():
            msg = f"asset no encontrado en disco: {uri}"
            raise AssetNotFoundError(msg)
        ref = AssetRef(
            asset_id=asset_id,
            kind=kind,
            uri=path.relative_to(self._root).as_posix(),
            sha256=sha256_file(path),
            size_bytes=path.stat().st_size,
            mime=_detect_mime(path),
            origin=origin,
            license=license,
            resolved_at=datetime.now(UTC),
        )
        self._assets[asset_id] = ref
        return ref

    def get(self, asset_id: str) -> AssetRef:
        """Devuelve la referencia registrada de un asset.

        Args:
            asset_id: Identificador del asset.

        Returns:
            La referencia registrada.

        Raises:
            AssetNotFoundError: Si el asset no está registrado.
        """
        try:
            return self._assets[asset_id]
        except KeyError as error:
            msg = f"asset no registrado: {asset_id}"
            raise AssetNotFoundError(msg) from error

    def path_for(self, asset_id: str) -> Path:
        """Resuelve la ruta absoluta y segura de un asset registrado.

        Falla con ``AssetNotFoundError`` si el asset no está registrado y con
        ``UnsafeAssetPathError`` si su URI registrada escapa de la raíz.

        Args:
            asset_id: Identificador del asset.

        Returns:
            Ruta absoluta dentro de la raíz del workspace.
        """
        return self._resolve(self.get(asset_id).uri)

    def verify(self, asset_id: str) -> bool:
        """Verifica que el archivo siga existiendo con tamaño y hash intactos.

        Args:
            asset_id: Identificador del asset.

        Returns:
            ``True`` si el archivo coincide con lo registrado.
        """
        ref = self.get(asset_id)
        path = self.path_for(asset_id)
        if not path.is_file():
            return False
        return path.stat().st_size == ref.size_bytes and sha256_file(path) == ref.sha256

    def canonical_uri(self, uri: str) -> str:
        """Devuelve la forma canónica relativa de una uri de asset.

        Una uri no relativa o que escape de la raíz produce
        ``UnsafeAssetPathError``.

        Args:
            uri: Ruta relativa a la raíz del workspace.

        Returns:
            La ruta relativa en formato POSIX, resuelta y normalizada igual
            que la uri almacenada por :meth:`register`.
        """
        path = self._resolve(uri)
        return path.relative_to(self._root).as_posix()

    def save(self, path: Path) -> None:
        """Escribe el registro como JSON versionado.

        Args:
            path: Ruta del archivo de registro a escribir.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        document = RegistryDocument(assets=list(self._assets.values()))
        _ = path.write_text(document.model_dump_json(indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path, root: Path) -> "AssetRegistry":
        """Carga un registro desde JSON validando su estructura.

        Args:
            path: Ruta del archivo de registro.
            root: Raíz del workspace para resolver las URIs.

        Returns:
            El registro cargado.

        Raises:
            AssetAlreadyRegisteredError: Si el archivo repite un asset_id.
        """
        document = RegistryDocument.model_validate_json(path.read_text(encoding="utf-8"))
        assets = {ref.asset_id: ref for ref in document.assets}
        if len(assets) != len(document.assets):
            msg = "asset_id duplicado en el archivo de registro"
            raise AssetAlreadyRegisteredError(msg)
        return cls(root, assets)

    def _resolve(self, uri: str) -> Path:
        if "\x00" in uri:
            msg = f"la uri del asset contiene un byte nulo: {uri!r}"
            raise UnsafeAssetPathError(msg)
        candidate = Path(uri)
        if candidate.is_absolute() or candidate.drive or uri.startswith(("\\\\", "//")):
            msg = f"la uri del asset debe ser relativa a la raíz del workspace: {uri}"
            raise UnsafeAssetPathError(msg)
        resolved = (self._root / candidate).resolve()
        if not resolved.is_relative_to(self._root):
            msg = f"ruta de asset fuera de la raíz del workspace: {uri}"
            raise UnsafeAssetPathError(msg)
        return resolved
