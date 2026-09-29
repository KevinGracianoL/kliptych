"""Sprint 2 (Objetivo 2): prohibiciones con separación de autoría.

La comparación normaliza (NFKD sin diacríticos + casefold) y exige frontera
de palabra sobre la frase escapada (``re.escape``): "¡ACTÍVA LA RACHA!"
casa con "¡activa la racha!", "win $10" no casa con "win $100" ni "sorteo"
con "sorteos". La autoría decide el veredicto: lo publicado por la cuenta
(caption, hashtags) falla; lo dicho por el streamer (``subtitle_text`` de
Whisper) exige revisión humana.
"""

from pathlib import Path

from kliptych.assets import AssetRegistry
from kliptych.gate import CheckStatus, Gate, GateStatus, SubtitleSegment
from tests.support import FakeProbe, make_contract, make_media, make_piece

_RULE = "caption.forbidden"


def _artifact(tmp_path: Path) -> Path:
    path = tmp_path / "piece.mp4"
    _ = path.write_bytes(b"video")
    return path


def _check(
    tmp_path: Path,
    *,
    forbidden: tuple[str, ...] = (),
    prohibitions: tuple[str, ...] = (),
    caption: str = "mira @marca #marca",
    hashtags: tuple[str, ...] = (),
    subtitle_text: str | None = None,
    screen_text_segments: tuple[SubtitleSegment, ...] = (),
) -> tuple[CheckStatus, dict[str, object]]:
    contract = make_contract(
        hard=[_RULE], forbidden=forbidden, prohibitions=prohibitions, min_s=None
    )
    piece = make_piece(
        _artifact(tmp_path),
        caption=caption,
        hashtags=hashtags,
        subtitle_text=subtitle_text,
        screen_text_segments=screen_text_segments,
    )
    gate = Gate(FakeProbe(info=make_media()))
    result = gate.run(
        contract=contract,
        piece=piece,
        assets=AssetRegistry(tmp_path),
    )
    matches = [check for check in result.checks if check.id == _RULE]
    assert len(matches) == 1
    return matches[0].status, dict(matches[0].evidence)


def test_forbidden_normalizes_accents_and_case(tmp_path: Path) -> None:
    status, evidence = _check(
        tmp_path,
        prohibitions=("¡ACTÍVA LA RACHA!",),
        caption="mira esto ¡activa la racha! @marca #marca",
    )
    assert status is CheckStatus.FAIL
    assert evidence["found"] == ["¡ACTÍVA LA RACHA!"]


def test_forbidden_escapes_regex_chars_and_bounds(tmp_path: Path) -> None:
    status, _ = _check(
        tmp_path,
        prohibitions=("win $10",),
        caption="solo win $100 @marca #marca",
    )
    assert status is CheckStatus.PASS
    status, evidence = _check(
        tmp_path,
        prohibitions=("win $10",),
        caption="solo win $10! @marca #marca",
    )
    assert status is CheckStatus.FAIL
    assert evidence["found"] == ["win $10"]


def test_forbidden_prefix_is_not_a_match(tmp_path: Path) -> None:
    status, _ = _check(
        tmp_path,
        forbidden=("sorteo",),
        caption="grandes sorteos @marca #marca",
    )
    assert status is CheckStatus.PASS


def test_forbidden_in_hashtags_fails(tmp_path: Path) -> None:
    status, evidence = _check(
        tmp_path,
        prohibitions=("¡ACTÍVA LA RACHA!",),
        caption="mira @marca #marca",
        hashtags=("#marca", "#ActivaLaRacha"),
        subtitle_text=None,
    )
    assert status is CheckStatus.PASS
    assert evidence["found"] == []
    status, evidence = _check(
        tmp_path,
        prohibitions=("estafa",),
        caption="mira @marca #marca",
        hashtags=("#marca", "#ESTAFA"),
    )
    assert status is CheckStatus.FAIL
    assert evidence["found"] == ["estafa"]


def test_forbidden_only_in_subtitles_needs_review(tmp_path: Path) -> None:
    status, evidence = _check(
        tmp_path,
        prohibitions=("¡ACTÍVA LA RACHA!",),
        caption="mira @marca #marca",
        subtitle_text="y entonces grité ¡activa la racha! en el directo",
    )
    assert status is CheckStatus.MANUAL_REVIEW
    assert evidence["found"] == ["¡ACTÍVA LA RACHA!"]
    assert evidence["authorship"] == "spoken"


def test_forbidden_in_caption_and_subtitles_fails(tmp_path: Path) -> None:
    status, evidence = _check(
        tmp_path,
        forbidden=("sorteo",),
        caption="gran SORTEO @marca #marca",
        subtitle_text="hablamos del sorteo de ayer",
    )
    assert status is CheckStatus.FAIL
    assert evidence["found"] == ["sorteo"]
    assert evidence["authorship"] == "published"


def test_forbidden_absent_passes(tmp_path: Path) -> None:
    status, evidence = _check(
        tmp_path,
        forbidden=("sorteo",),
        prohibitions=("estafa",),
        subtitle_text="charlamos del directo de ayer",
    )
    assert status is CheckStatus.PASS
    assert evidence["found"] == []


def test_spoken_only_marks_gate_pending_review(tmp_path: Path) -> None:
    contract = make_contract(hard=[_RULE], prohibitions=("sorteo",), min_s=None)
    piece = make_piece(
        _artifact(tmp_path),
        caption="mira @marca #marca",
        subtitle_text="charlamos del SORTEO de ayer",
    )
    gate = Gate(FakeProbe(info=make_media()))

    result = gate.run(contract=contract, piece=piece, assets=AssetRegistry(tmp_path))
    assert result.status is GateStatus.PENDING_REVIEW


def test_h7_forbidden_preserves_enie_avoids_false_positives(tmp_path: Path) -> None:
    status, evidence = _check(
        tmp_path,
        prohibitions=("ano",),
        caption="feliz año nuevo @marca #marca",
    )
    assert status is CheckStatus.PASS
    assert evidence["found"] == []

    status, evidence = _check(
        tmp_path,
        prohibitions=("pena",),
        caption="gran fiesta en la peña @marca #marca",
    )
    assert status is CheckStatus.PASS
    assert evidence["found"] == []

    status, evidence = _check(
        tmp_path,
        prohibitions=("cono",),
        caption="cuidado con el coño @marca #marca",
    )
    assert status is CheckStatus.PASS
    assert evidence["found"] == []


def test_h6_forbidden_collapses_whitespace_and_newlines(tmp_path: Path) -> None:
    status, evidence = _check(
        tmp_path,
        prohibitions=("activa la racha",),
        caption="mira esto activa   la    racha @marca #marca",
    )
    assert status is CheckStatus.FAIL
    assert evidence["found"] == ["activa la racha"]

    status, evidence = _check(
        tmp_path,
        prohibitions=("activa la racha",),
        caption="mira esto activa la\nracha @marca #marca",
    )
    assert status is CheckStatus.FAIL
    assert evidence["found"] == ["activa la racha"]


def test_h6_forbidden_treats_underscore_as_boundary(tmp_path: Path) -> None:
    status, evidence = _check(
        tmp_path,
        prohibitions=("sorteo",),
        caption="entra al sorteo_gratis ya @marca #marca",
    )
    assert status is CheckStatus.FAIL
    assert evidence["found"] == ["sorteo"]


def test_h8_forbidden_in_screen_text_segments_fails_as_published(tmp_path: Path) -> None:
    status, evidence = _check(
        tmp_path,
        prohibitions=("sorteo",),
        caption="mira este video @marca #marca",
        screen_text_segments=(SubtitleSegment(text="gran sorteo hoy", start_s=0.5, end_s=2.0),),
    )
    assert status is CheckStatus.FAIL
    assert evidence["found"] == ["sorteo"]
    assert evidence["authorship"] == "published"
