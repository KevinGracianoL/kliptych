"""Tests del registro de assets."""

import json
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path

import pytest

from kliptych.assets import (
    AssetAlreadyRegisteredError,
    AssetNotFoundError,
    AssetRegistry,
    UnsafeAssetPathError,
)


def _write(root: Path, relative: str, content: bytes = b"datos") -> Path:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    _ = path.write_bytes(content)
    return path


def test_register_computes_hash_size_and_mime(tmp_path: Path) -> None:
    file_path = _write(tmp_path, "assets/samples/clip.mp4", b"video")
    registry = AssetRegistry(tmp_path)
    ref = registry.register(
        asset_id="clip-01",
        kind="video",
        uri="assets/samples/clip.mp4",
        origin="brief",
    )
    assert ref.asset_id == "clip-01"
    assert ref.sha256 == sha256(b"video").hexdigest()
    assert ref.size_bytes == 5
    assert ref.mime == "video/mp4"
    assert ref.license is None
    assert ref.resolved_at.tzinfo is not None
    assert ref.uri == "assets/samples/clip.mp4"
    assert registry.path_for("clip-01") == file_path.resolve()


def test_register_keeps_optional_license(tmp_path: Path) -> None:
    _ = _write(tmp_path, "clip.mp4")
    registry = AssetRegistry(tmp_path)
    ref = registry.register(
        asset_id="clip",
        kind="video",
        uri="clip.mp4",
        origin="banco-ugc",
        license="CC0",
    )
    assert ref.license == "CC0"


def test_register_missing_file_fails(tmp_path: Path) -> None:
    registry = AssetRegistry(tmp_path)
    with pytest.raises(AssetNotFoundError, match=r"nope\.mp4"):
        _ = registry.register(asset_id="a", kind="video", uri="nope.mp4", origin="brief")


def test_mime_falls_back_to_stdlib_guess(tmp_path: Path) -> None:
    _ = _write(tmp_path, "pagina.html")
    registry = AssetRegistry(tmp_path)
    ref = registry.register(asset_id="doc", kind="file", uri="pagina.html", origin="brief")
    assert ref.mime == "text/html"


def test_mime_falls_back_to_octet_stream(tmp_path: Path) -> None:
    _ = _write(tmp_path, "blob.zzz")
    registry = AssetRegistry(tmp_path)
    ref = registry.register(asset_id="blob", kind="file", uri="blob.zzz", origin="brief")
    assert ref.mime == "application/octet-stream"


def test_register_rejects_relative_traversal(tmp_path: Path) -> None:
    outside = _write(tmp_path, "outside.txt")
    root = tmp_path / "workspace"
    root.mkdir()
    registry = AssetRegistry(root)
    with pytest.raises(UnsafeAssetPathError, match="fuera"):
        _ = registry.register(asset_id="a", kind="file", uri="../outside.txt", origin="brief")
    assert outside.exists()


def test_register_rejects_absolute_path_inside_root(tmp_path: Path) -> None:
    inside = _write(tmp_path, "clip.mp4")
    registry = AssetRegistry(tmp_path)
    with pytest.raises(UnsafeAssetPathError, match="relativa"):
        _ = registry.register(asset_id="a", kind="video", uri=str(inside), origin="brief")


def test_register_rejects_unc_without_resolving(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry = AssetRegistry(tmp_path)

    def forbidden_resolve(_self_path: Path) -> Path:
        msg = "resolve() no debe ejecutarse sobre rutas no relativas"
        raise AssertionError(msg)

    monkeypatch.setattr(Path, "resolve", forbidden_resolve)
    with pytest.raises(UnsafeAssetPathError, match="relativa"):
        _ = registry.register(
            asset_id="a",
            kind="file",
            uri=r"\\attacker.example\share\x.mp4",
            origin="brief",
        )


def test_canonical_uri_normalizes_relative_paths(tmp_path: Path) -> None:
    _ = _write(tmp_path, "clip.mp4")
    registry = AssetRegistry(tmp_path)
    assert registry.canonical_uri("clip.mp4") == "clip.mp4"
    assert registry.canonical_uri("./sub/../clip.mp4") == "clip.mp4"


def test_canonical_uri_rejects_traversal(tmp_path: Path) -> None:
    root = tmp_path / "workspace"
    root.mkdir()
    registry = AssetRegistry(root)
    with pytest.raises(UnsafeAssetPathError, match="fuera"):
        _ = registry.canonical_uri("../fuera.txt")


def test_register_rejects_nul_byte_in_uri(tmp_path: Path) -> None:
    registry = AssetRegistry(tmp_path)
    with pytest.raises(UnsafeAssetPathError, match="nulo"):
        _ = registry.register(asset_id="a", kind="file", uri="clip\x00.mp4", origin="brief")


def test_duplicate_asset_id_is_rejected(tmp_path: Path) -> None:
    _ = _write(tmp_path, "clip.mp4")
    registry = AssetRegistry(tmp_path)
    _ = registry.register(asset_id="clip", kind="video", uri="clip.mp4", origin="brief")
    with pytest.raises(AssetAlreadyRegisteredError, match="clip"):
        _ = registry.register(asset_id="clip", kind="video", uri="clip.mp4", origin="brief")


def test_assets_view_exposes_registered_assets(tmp_path: Path) -> None:
    _ = _write(tmp_path, "clip.mp4")
    registry = AssetRegistry(tmp_path)
    ref = registry.register(asset_id="clip", kind="video", uri="clip.mp4", origin="brief")
    assert registry.assets == {"clip": ref}


def test_verify_detects_content_changes(tmp_path: Path) -> None:
    path = _write(tmp_path, "clip.mp4", b"original")
    registry = AssetRegistry(tmp_path)
    _ = registry.register(asset_id="clip", kind="video", uri="clip.mp4", origin="brief")
    assert registry.verify("clip") is True
    _ = path.write_bytes(b"manipulado")
    assert registry.verify("clip") is False
    path.unlink()
    assert registry.verify("clip") is False


def test_get_unknown_asset_fails(tmp_path: Path) -> None:
    registry = AssetRegistry(tmp_path)
    with pytest.raises(AssetNotFoundError, match="desconocido"):
        _ = registry.get("desconocido")


def test_save_and_load_round_trip(tmp_path: Path) -> None:
    _ = _write(tmp_path, "clip.mp4", b"video")
    registry = AssetRegistry(tmp_path)
    ref = registry.register(
        asset_id="clip",
        kind="video",
        uri="clip.mp4",
        origin="brief",
        license="CC0",
    )
    registry_file = tmp_path / "assets" / "registry.json"
    registry.save(registry_file)

    loaded = AssetRegistry.load(registry_file, tmp_path)
    assert loaded.get("clip") == ref
    assert loaded.path_for("clip") == (tmp_path / "clip.mp4").resolve()


def test_load_rejects_registry_with_traversal_uri(tmp_path: Path) -> None:
    registry_file = tmp_path / "registry.json"
    payload = {
        "schema_version": "1.1",
        "assets": [
            {
                "asset_id": "malo",
                "kind": "file",
                "uri": "../fuera.txt",
                "sha256": "a" * 64,
                "size_bytes": 1,
                "mime": "text/plain",
                "origin": "brief",
                "license": None,
                "resolved_at": datetime(2026, 9, 22, tzinfo=UTC).isoformat(),
            }
        ],
    }
    _ = registry_file.write_text(json.dumps(payload), encoding="utf-8")
    loaded = AssetRegistry.load(registry_file, tmp_path / "workspace")
    with pytest.raises(UnsafeAssetPathError, match="fuera"):
        _ = loaded.path_for("malo")


def test_load_rejects_duplicate_asset_ids(tmp_path: Path) -> None:
    registry_file = tmp_path / "registry.json"
    asset = {
        "asset_id": "dup",
        "kind": "file",
        "uri": "a.txt",
        "sha256": "a" * 64,
        "size_bytes": 1,
        "mime": "text/plain",
        "origin": "brief",
        "license": None,
        "resolved_at": datetime(2026, 9, 22, tzinfo=UTC).isoformat(),
    }
    _ = registry_file.write_text(
        json.dumps({"schema_version": "1.1", "assets": [asset, asset]}),
        encoding="utf-8",
    )
    with pytest.raises(AssetAlreadyRegisteredError, match="duplicado"):
        _ = AssetRegistry.load(registry_file, tmp_path)
