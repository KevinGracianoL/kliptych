"""Recolección de basura y limpieza de directorios temporales obsoletos."""

import shutil
import time
from pathlib import Path

_TEMP_PREFIXES = ("tmp", ".tmp", ".part-")
_PROTECTED_DIR_NAMES = frozenset({"fixtures", "pending"})
_TEMP_EXACT_NAMES = frozenset({"tmp", "temp", ".temp"})


def is_temporary_directory(path: Path) -> bool:
    """Indica si una carpeta coincide con los patrones de temporales del pipeline.

    Args:
        path: Ruta a verificar.

    Returns:
        True si el nombre coincide con un patrón temporal, False en caso contrario.
    """
    name = path.name.lower()
    return (
        name.startswith(_TEMP_PREFIXES)
        or "-tmp-" in name
        or name.endswith(".tmp")
        or name in _TEMP_EXACT_NAMES
    )


def _is_expired(path: Path, cutoff_time: float) -> bool:
    try:
        return path.stat().st_mtime <= cutoff_time
    except OSError:
        return False


def _collect_subcandidates(parent: Path, cutoff_time: float) -> list[Path]:
    if parent.name in _PROTECTED_DIR_NAMES:
        return []
    return [
        subitem
        for subitem in parent.iterdir()
        if (
            subitem.is_dir()
            and is_temporary_directory(subitem)
            and _is_expired(subitem, cutoff_time)
        )
    ]


def _find_temporary_dirs(root: Path, cutoff_time: float) -> list[Path]:
    candidates: list[Path] = []
    for item in root.iterdir():
        if not item.is_dir():
            continue
        if is_temporary_directory(item):
            if _is_expired(item, cutoff_time):
                candidates.append(item)
            continue
        candidates.extend(_collect_subcandidates(item, cutoff_time))
    return candidates


def clean_temporary_directories(
    root: Path,
    *,
    days: float = 7.0,
    dry_run: bool = False,
) -> tuple[Path, ...]:
    """Escanea un directorio y elimina carpetas temporales con antigüedad mayor a `days` días.

    Args:
        root: Directorio raíz donde buscar (ej. ``campaigns/`` o ``runs/``).
        days: Antigüedad mínima en días para considerar un directorio obsoleto.
            Debe ser >= 0.
        dry_run: Si es True, no elimina los directorios, solo los lista.

    Returns:
        Tupla con las rutas de los directorios eliminados (o identificados si dry_run).

    Raises:
        ValueError: Si `days` es negativo.
    """
    if days < 0:
        msg = f"days debe ser mayor o igual a 0: {days}"
        raise ValueError(msg)
    if not root.is_dir():
        return ()

    cutoff_time = time.time() - (days * 86400.0)
    candidates = _find_temporary_dirs(root, cutoff_time)
    if not dry_run:
        for directory in candidates:
            shutil.rmtree(directory, ignore_errors=True)
    return tuple(candidates)
