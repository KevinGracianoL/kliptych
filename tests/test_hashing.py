"""Tests de hashing determinista."""

from kliptych.hashing import sha256_canonical_json, sha256_text


def test_canonical_json_hash_is_order_insensitive() -> None:
    first: dict[str, object] = {"b": 1, "a": {"y": 2, "x": [1, 2]}}
    second: dict[str, object] = {"a": {"x": [1, 2], "y": 2}, "b": 1}
    assert sha256_canonical_json(first) == sha256_canonical_json(second)


def test_canonical_json_hash_changes_with_content() -> None:
    first: dict[str, object] = {"a": 1}
    second: dict[str, object] = {"a": 2}
    assert sha256_canonical_json(first) != sha256_canonical_json(second)


def test_text_hash_matches_utf8_bytes() -> None:
    assert sha256_text("hola") == sha256_text("hola")
    assert sha256_text("hola") != sha256_text("holá")
