"""Tests para parser de .lrc y cliente lrclib (Sprint 3 Parte 2 Objetivo 2).

Valida el parseo determinista de marcas de tiempo estándar, marcas múltiples por
línea, etiquetas de offset, descarte de metadatos, rechazo fail-closed de archivos
sin líneas sincronizadas y el cliente SyncedLyricsProvider/lrclib.
"""

import json
import math
import urllib.error
from email.message import Message
from pathlib import Path
from typing import Self, final

import pytest

from kliptych.assets import AssetRegistry
from kliptych.contract import LyricConfig
from kliptych.lyrics import (
    LrclibClient,
    LrcParseError,
    LyricsError,
    LyricsNotFoundError,
    parse_lrc,
    resolve_synced_lyrics,
)
from tests.support import make_contract


@final
class FakeResponse:
    _data: bytes

    def __init__(self, data: bytes) -> None:
        self._data = data

    def read(self) -> bytes:
        return self._data

    def __enter__(self) -> Self:
        """Entra en el contexto.

        Returns:
            La propia instancia.
        """
        return self

    def __exit__(self, *args: object) -> None:
        """Sale del contexto."""
        _ = args


@final
class FakeOpener:
    _response_bytes: bytes | None
    _error: Exception | None
    opened_urls: list[str]

    def __init__(
        self,
        response_bytes: bytes | None = None,
        error: Exception | None = None,
    ) -> None:
        self._response_bytes = response_bytes
        self._error = error
        self.opened_urls = []

    def open(
        self,
        fullurl: object,
        data: bytes | None = None,
        timeout: float = 10.0,
    ) -> FakeResponse:
        _ = (data, timeout)
        url_str = getattr(fullurl, "full_url", str(fullurl))
        self.opened_urls.append(url_str)
        if self._error is not None:
            raise self._error
        return FakeResponse(self._response_bytes or b"")


@final
class FakeProvider:
    lyrics: str | None
    error: Exception | None
    calls: list[tuple[str, str | None]]

    def __init__(self, lyrics: str | None = None, error: Exception | None = None) -> None:
        self.lyrics = lyrics
        self.error = error
        self.calls = []

    def get_synced_lyrics(
        self,
        *,
        track_name: str,
        artist_name: str | None = None,
        album_name: str | None = None,
        duration_s: float | None = None,
    ) -> str:
        _ = (album_name, duration_s)
        self.calls.append((track_name, artist_name))
        if self.error is not None:
            raise self.error
        if self.lyrics is not None:
            return self.lyrics
        msg = "no hay letras para la pista"
        raise LyricsNotFoundError(msg)


def test_parse_lrc_standard_timestamps() -> None:
    content = (
        "[00:10.50]Línea centisegundos\n[00:20.500]Línea milisegundos\n[01:30]Línea sin fracción\n"
    )
    lines = parse_lrc(content)
    assert len(lines) == 3
    assert math.isclose(lines[0].start_sec, 10.50)
    assert lines[0].text == "Línea centisegundos"
    assert math.isclose(lines[1].start_sec, 20.50)
    assert lines[1].text == "Línea milisegundos"
    assert math.isclose(lines[2].start_sec, 90.0)
    assert lines[2].text == "Línea sin fracción"


def test_parse_lrc_multiple_timestamps_per_line() -> None:
    content = "[00:10.00][00:30.00]Estribillo repetido\n"
    lines = parse_lrc(content)
    assert len(lines) == 2
    assert math.isclose(lines[0].start_sec, 10.0)
    assert lines[0].text == "Estribillo repetido"
    assert math.isclose(lines[1].start_sec, 30.0)
    assert lines[1].text == "Estribillo repetido"


def test_parse_lrc_offset_tag_application_and_clamping() -> None:
    content = (
        "[offset:+500]\n"
        "[00:10.00]Línea con adelanto\n"
        "[offset:-2000]\n"
        "[00:01.00]Línea con retraso excesivo\n"
    )
    lines = parse_lrc(content)
    assert len(lines) == 2
    assert math.isclose(lines[0].start_sec, 0.0)  # max(0.0, 1.0 - 2.0)
    assert lines[0].text == "Línea con retraso excesivo"
    assert math.isclose(lines[1].start_sec, 10.50)
    assert lines[1].text == "Línea con adelanto"


def test_parse_lrc_ignores_metadata_tags() -> None:
    content = (
        "[ar:Queen]\n"
        "[ti:Bohemian Rhapsody]\n"
        "[al:A Night at the Opera]\n"
        "[by:Creator]\n"
        "[length:05:55]\n"
        "[00:15.00]Is this the real life?\n"
    )
    lines = parse_lrc(content)
    assert len(lines) == 1
    assert math.isclose(lines[0].start_sec, 15.0)
    assert lines[0].text == "Is this the real life?"


def test_parse_lrc_sorts_chronologically() -> None:
    content = "[00:45.00]Tercera\n[00:15.00]Primera\n[00:30.00]Segunda\n"
    lines = parse_lrc(content)
    assert [line.text for line in lines] == ["Primera", "Segunda", "Tercera"]
    assert [line.start_sec for line in lines] == [15.0, 30.0, 45.0]


@pytest.mark.parametrize(
    "invalid_content",
    [
        "",  # vacío
        "   \n\t  \n",  # solo espacios
        "[ar:Artista]\n[ti:Titulo]\n",  # solo metadatos
        "[00:10.00]\n[00:20.00]   \n",  # timestamps sin texto
        "[01:60.00]Segundos mayores a 59\n",  # ss >= 60
        "[01:99.00]Segundos invalidos\n",
        "[-01:20.00]Minuto negativo\n",  # negativo
        "[01:-20.00]Segundo negativo\n",
        "[01:١٢.00]Digitos no ASCII\n",  # no ASCII
    ],
)
def test_parse_lrc_fail_closed_rejects_invalid(invalid_content: str) -> None:
    with pytest.raises(LrcParseError):
        _ = parse_lrc(invalid_content)


def test_lrclib_client_fetches_synced_lyrics() -> None:
    fake_opener = FakeOpener(
        response_bytes=b'{"id": 1, "trackName": "Song", "syncedLyrics": "[00:10.00]Hello world"}'
    )
    client = LrclibClient(opener=fake_opener, timeout_s=5.0)
    lyrics = client.get_synced_lyrics(track_name="Song", artist_name="Artist")
    assert lyrics == "[00:10.00]Hello world"


def test_lrclib_client_rejects_plain_lyrics_only() -> None:
    data = {"id": 1, "trackName": "Song", "plainLyrics": "Hello", "syncedLyrics": None}
    fake_opener = FakeOpener(response_bytes=json.dumps(data).encode("utf-8"))
    client = LrclibClient(opener=fake_opener)
    with pytest.raises(LyricsNotFoundError, match="syncedLyrics"):
        _ = client.get_synced_lyrics(track_name="Song")


def test_lrclib_client_http_error_translates_to_lyrics_error() -> None:
    fake_opener = FakeOpener(
        error=urllib.error.HTTPError(
            url="https://lrclib.net", code=404, msg="Not Found", hdrs=Message(), fp=None
        )
    )
    client = LrclibClient(opener=fake_opener)
    with pytest.raises(LyricsNotFoundError):
        _ = client.get_synced_lyrics(track_name="Nonexistent")


def test_lrclib_client_timeout_translates_to_lyrics_error() -> None:
    fake_opener = FakeOpener(error=TimeoutError("Connection timed out"))
    client = LrclibClient(opener=fake_opener)
    with pytest.raises(LyricsError, match="timed out"):
        _ = client.get_synced_lyrics(track_name="Song")


def test_resolve_synced_lyrics_local_asset_priority(tmp_path: Path) -> None:
    lrc_file = tmp_path / "song.lrc"
    _ = lrc_file.write_text("[00:05.00]Letra local\n", encoding="utf-8")

    registry = AssetRegistry(tmp_path)
    ref = registry.register(
        asset_id="song_lrc",
        kind="lyrics",
        uri="song.lrc",
        origin="test",
    )

    contract = make_contract(
        format_="lyric_video",
        lyric_video=LyricConfig(lrc_asset_id="song_lrc", track_name="Song"),
        required_assets=[ref],
    )

    fake_provider = FakeProvider()
    lines = resolve_synced_lyrics(contract=contract, registry=registry, provider=fake_provider)
    assert len(lines) == 1
    assert lines[0].text == "Letra local"
    assert not fake_provider.calls


def test_resolve_synced_lyrics_provider_fallback(tmp_path: Path) -> None:
    registry = AssetRegistry(tmp_path)
    contract = make_contract(
        format_="lyric_video",
        lyric_video=LyricConfig(track_name="Song", artist_name="Artist", lrclib_enabled=True),
    )

    fake_provider = FakeProvider(lyrics="[00:07.50]Letra remota\n")
    lines = resolve_synced_lyrics(contract=contract, registry=registry, provider=fake_provider)
    assert len(lines) == 1
    assert lines[0].text == "Letra remota"
    assert fake_provider.calls == [("Song", "Artist")]


def test_resolve_synced_lyrics_fail_closed_when_no_source(tmp_path: Path) -> None:
    registry = AssetRegistry(tmp_path)
    contract = make_contract(
        format_="lyric_video",
        lyric_video=LyricConfig(lrclib_enabled=False),
    )
    with pytest.raises(LyricsError, match="no hay fuente"):
        _ = resolve_synced_lyrics(contract=contract, registry=registry, provider=None)
