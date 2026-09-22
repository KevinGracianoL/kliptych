"""Resolución de rutas y configuración del workspace de Kliptych."""

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

ENV_ROOT = "KLIPTYCH_ROOT"


@dataclass(frozen=True, slots=True)
class Settings:
    """Rutas de un workspace de Kliptych, siempre resueltas a absolutas.

    Attributes:
        root: Raíz del workspace; todas las rutas derivadas cuelgan de aquí.
    """

    root: Path

    @classmethod
    def from_root(cls, root: Path) -> "Settings":
        """Construye settings a partir de un root explícito.

        Args:
            root: Ruta raíz del workspace.

        Returns:
            Settings con la raíz resuelta a absoluta.
        """
        return cls(root=root.resolve())

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> "Settings":
        """Construye settings desde ``KLIPTYCH_ROOT``.

        Args:
            environ: Mapeo de entorno a consultar; por defecto ``os.environ``.

        Returns:
            Settings con la raíz del entorno, o el directorio actual si la
            variable no está definida.
        """
        source = os.environ if environ is None else environ
        return cls.from_root(Path(source.get(ENV_ROOT, ".")))

    @property
    def runs_dir(self) -> Path:
        """Directorio de corridas locales; nunca se commitea.

        Returns:
            Ruta a ``<root>/runs``.
        """
        return self.root / "runs"

    @property
    def campaigns_dir(self) -> Path:
        """Directorio raíz de campañas.

        Returns:
            Ruta a ``<root>/campaigns``.
        """
        return self.root / "campaigns"

    @property
    def private_campaigns_dir(self) -> Path:
        """Material real de campaña; nunca se commitea.

        Returns:
            Ruta a ``<root>/campaigns/private``.
        """
        return self.campaigns_dir / "private"

    @property
    def assets_dir(self) -> Path:
        """Directorio raíz de assets.

        Returns:
            Ruta a ``<root>/assets``.
        """
        return self.root / "assets"
