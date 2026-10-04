"""STEP 1b: el conteo de eventos ASS es un contrato EXCLUSIVO del modo letra.

El karaoke ahora escribe un `.ass` con VARIOS eventos `Dialogue`, uno por linea
del subtitulo, mientras que el modo letra escribe un evento por linea del `.lrc`.
Si el gate alcanzara `_verify_ass_dialogue_events` con una pieza de karaoke,
compararia el numero de eventos de karaoke contra el numero de lineas de una
letra que no existe, y fallaria por la razon equivocada.

Estos tests son el SEGURO de esa frontera: comprueban que la pieza de karaoke NO
llega al verificador, y que la de letra si sigue evaluandose. No se toca
`src/kliptych/gate/`: solo se espia la llamada.
"""

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import cast

import pytest

from kliptych.assets import AssetRegistry
from kliptych.contract import Format
from kliptych.contract.schema import Contract, PlatformRules
from kliptych.gate import checks
from kliptych.gate.models import GateContext, Piece
from kliptych.lyrics import LyricLine
from kliptych.subtitles import SubtitleRenderer
from kliptych.transcribe import Word
from tests.support import make_asset_ref, make_contract, make_piece


def _private(nombre: str) -> object:
    """Accede a un simbolo privado del modulo, como en el resto de la suite.

    Args:
        nombre: Nombre del simbolo.

    Returns:
        El objeto privado, tipado como `object`.
    """
    return cast("object", getattr(checks, nombre))


def _ground_truth(contexto: GateContext) -> object:
    """Invoca el check privado por el que pasa la frontera.

    Args:
        contexto: El contexto del gate.

    Returns:
        Lo que devuelva el check: `None` si no aplica.
    """
    return cast("Callable[[GateContext], object]", _private("_check_lyric_ground_truth"))(contexto)


def _verificador_espia(monkeypatch: pytest.MonkeyPatch) -> list[object]:
    """Sustituye el verificador por uno que registra las piezas que recibe.

    Args:
        monkeypatch: El.patch de pytest.

    Returns:
        La lista donde se acumulan las piezas recibidas.
    """
    registradas: list[object] = []
    original = cast(
        "Callable[[Piece, Sequence[LyricLine]], str | None]",
        _private("_verify_ass_dialogue_events"),
    )

    def registrador(piece: Piece, expected_lines: Sequence[LyricLine]) -> str | None:
        registradas.append(piece)
        return original(piece, expected_lines)

    monkeypatch.setattr(checks, "_verify_ass_dialogue_events", registrador)
    return registradas


def _karaoke_ass(tmp_path: Path) -> Path:
    """Un `.ass` de karaoke REAL, con tres palabras en un solo evento.

    Args:
        tmp_path: Directorio de trabajo.

    Returns:
        La ruta del `.ass` escrito.
    """
    destino = tmp_path / "karaoke.ass"
    words = [
        Word(start_s=0.0, end_s=0.5, text="hola", confidence=0.9, token_id=0),
        Word(start_s=0.5, end_s=1.0, text="mundo", confidence=0.9, token_id=1),
        Word(start_s=1.0, end_s=1.5, text="cruel", confidence=0.9, token_id=2),
    ]
    _ = SubtitleRenderer().write(words, destino)
    return destino


def _contexto(contrato: object, pieza: object, registro: AssetRegistry) -> GateContext:
    """Arma el contexto que el gate recibe, con reglas vacias a proposito.

    El tripwire no inspecciona reglas: `_check_lyric_ground_truth` solo lee el
    contrato y la pieza. Meter reglas reales seria meter otro sujet under test.

    Args:
        contrato: El contrato de la pieza.
        pieza: La pieza.
        registro: El registro de assets.

    Returns:
        El contexto listo para el verificador.
    """
    return GateContext(
        contract=cast("Contract", contrato),
        rules=PlatformRules(),
        piece=cast("Piece", pieza),
        artifact_sha256=None,
        media=None,
        assets=registro,
    )


def test_el_karaoke_escribe_menos_eventos_que_palabras(tmp_path: Path) -> None:
    """El premise del riesgo: N palabras ya NO son N eventos."""
    destino = _karaoke_ass(tmp_path)
    eventos = [
        linea
        for linea in destino.read_text(encoding="utf-8").splitlines()
        if linea.startswith("Dialogue:")
    ]
    assert len(eventos) == 1, eventos


def test_la_pieza_de_karaoke_no_llega_al_verificador_de_eventos(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Karaoke con `.ass` real y contrato NO letra: no debe entrar al conteo.

    Es la frontera que hace seguro este cambio. Si una pieza de karaoke llegara
    al verificador, el conteo de eventos se aplicaria sobre un archivo cuya
    forma de eventos es distinta, y cada linea agrupada seria un falso positivo.
    """
    registradas = _verificador_espia(monkeypatch)
    contrato = make_contract(format_="video")
    assert contrato.format is not Format.LYRIC_VIDEO, contrato.format
    pieza = make_piece(
        tmp_path / "clip.mp4",
        subtitle_text="hola mundo cruel",
        ass_path=_karaoke_ass(tmp_path),
    )

    resultado = _ground_truth(_contexto(contrato, pieza, AssetRegistry(root=tmp_path)))

    assert resultado is None, resultado
    assert registradas == [], "el karaoke no debe llegar al conteo de eventos"


def test_la_pieza_de_letra_no_se_salta_la_guarda(tmp_path: Path) -> None:
    """CONTRASTE: con contrato de letra la misma funcion NO se salta la guarda.

    Sin este contrapeso, el test de karaoke pasaria por la razon equivocada: porque
    el verificador hubiera desaparecido, y no porque la frontera este donde debe.
    Se afirma lo que si es comprobable sin montar todo el camino de letra: que
    `_check_lyric_ground_truth` sigue evaluando en vez de devolver None.
    """
    _ = (tmp_path / "letra.lrc").write_text("[00:00.00]hola\n[00:02.00]mundo\n", encoding="utf-8")
    contrato = make_contract(
        format_="lyric_video",
        lyric_video={"lrc_asset_id": "lrc-01"},
        required_assets=(make_asset_ref("lrc-01", sha256="b" * 64),),
    )
    assert contrato.format is Format.LYRIC_VIDEO, contrato.format
    assert contrato.lyric_video is not None
    assert contrato.lyric_video.lrc_asset_id == "lrc-01", contrato.lyric_video
    registro = AssetRegistry(
        root=tmp_path,
        assets={"lrc-01": make_asset_ref("lrc-01", sha256="b" * 64)},
    )
    pieza = make_piece(
        tmp_path / "clip.mp4",
        subtitle_text="hola mundo",
        ass_path=tmp_path / "letra.ass",
    )

    resultado = _ground_truth(_contexto(contrato, pieza, registro))

    assert resultado is not None, "una pieza de letra no debe saltarse la guarda"
