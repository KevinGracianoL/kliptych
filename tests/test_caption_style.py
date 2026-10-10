"""Pruebas de CaptionStyleConfig: estilos karaoke, colores y capitalización."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from kliptych.assets import AssetRegistry
from kliptych.contract import (
    CaptionCapitalization,
    CaptionStyleConfig,
    SubtitleLanguage,
    contract_digest,
    make_zero_contract,
    vanilla_contract,
)
from kliptych.resolver import resolve_contract
from kliptych.subtitles import SubtitleRenderer
from kliptych.transcribe import Word
from tests.support import make_contract, make_draft


def _word(start: float, end: float, text: str, token_id: int = 0) -> Word:
    return Word(
        start_s=start,
        end_s=end,
        text=text,
        confidence=1.0,
        token_id=token_id,
    )


def test_caption_capitalization_and_language_enums() -> None:
    """Verifica los valores definidos para los nuevos enums del contrato."""
    assert CaptionCapitalization.ORIGINAL == "original"
    assert CaptionCapitalization.UPPERCASE == "uppercase"
    assert CaptionCapitalization.LOWERCASE == "lowercase"

    assert SubtitleLanguage.ORIGINAL == "original"
    assert SubtitleLanguage.ENGLISH == "en"
    assert SubtitleLanguage.SPANISH == "es"
    assert SubtitleLanguage.PORTUGUESE == "pt"
    assert SubtitleLanguage.FRENCH == "fr"
    assert SubtitleLanguage.GERMAN == "de"
    assert SubtitleLanguage.ITALIAN == "it"
    assert SubtitleLanguage.JAPANESE == "ja"


def test_caption_style_config_defaults() -> None:
    """Verifica los valores por defecto de CaptionStyleConfig."""
    config = CaptionStyleConfig()
    assert config.capitalization is CaptionCapitalization.ORIGINAL
    assert config.subtitle_language is SubtitleLanguage.ORIGINAL
    assert config.primary_color == "&H00FFFFFF"
    assert config.highlight_color == "&H000000FF"
    assert config.outline_color == "&H00000000"
    assert config.back_color == "&H00000000"
    assert config.fontname == "Arial"
    assert config.fontsize == 48
    assert config.outline == 2
    assert config.shadow == 0
    assert config.margin_v == 40
    assert config.alignment == 2


@pytest.mark.parametrize(
    ("input_color", "expected_ass"),
    [
        ("#FF0000", "&H000000FF"),
        ("#00FF00", "&H0000FF00"),
        ("#0000FF", "&H00FF0000"),
        ("#FFFFFF", "&H00FFFFFF"),
        ("#000000", "&H00000000"),
        ("#ff0000", "&H000000FF"),
        ("#FF0000FF", "&H000000FF"),
        ("#FF000000", "&HFF0000FF"),
        ("&H00FFFFFF", "&H00FFFFFF"),
        ("&H000000FF", "&H000000FF"),
        ("&h00ffffff", "&H00FFFFFF"),
    ],
)
def test_color_normalizer_accepts_and_converts(input_color: str, expected_ass: str) -> None:
    """El validador de color acepta CSS hex y SubStation &HAABBGGRR y normaliza a ASS BGR."""
    cfg = CaptionStyleConfig(primary_color=input_color)
    assert cfg.primary_color == expected_ass


@pytest.mark.parametrize(
    "bad_color",
    [
        "rojo",
        "#FFF",
        "&H123",
        "FFFFFF",
        "#GGGGGG",
        "&HZZZZZZZZ",
        "#12345",
        "#1234567",
        "",
    ],
)
def test_color_normalizer_rejects_invalid_formats(bad_color: str) -> None:
    """El validador rechaza colores que no cumplan el formato CSS hex o &HAABBGGRR."""
    with pytest.raises(ValidationError):
        _ = CaptionStyleConfig(primary_color=bad_color)


def test_ass_style_header_reflects_primary_and_highlight_colors() -> None:
    """El encabezado Style de ASS refleja los colores primario y secundario (highlight)."""
    cfg = CaptionStyleConfig(
        primary_color="#00FF00",  # Verde en CSS -> &H0000FF00 en ASS
        highlight_color="#FF0000",  # Rojo en CSS -> &H000000FF en ASS
        fontname="Impact",
        fontsize=52,
    )
    renderer = SubtitleRenderer.from_caption_style(cfg)
    words = [_word(0.0, 1.0, "Prueba")]
    ass_text = renderer.build(words)

    # El Style line en ASS tiene: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour...
    assert "Style: Default,Impact,52,&H0000FF00,&H000000FF," in ass_text


def test_ass_dialogue_line_capitalization() -> None:
    """Dialogue aplica ORIGINAL (Aa), UPPERCASE (AA) y LOWERCASE (aa) sin alterar timing."""
    words = [
        _word(0.0, 0.5, "Hola"),
        _word(0.5, 1.0, "Mundo"),
    ]

    r_orig = SubtitleRenderer.from_caption_style(
        CaptionStyleConfig(capitalization=CaptionCapitalization.ORIGINAL)
    )
    r_upper = SubtitleRenderer.from_caption_style(
        CaptionStyleConfig(capitalization=CaptionCapitalization.UPPERCASE)
    )
    r_lower = SubtitleRenderer.from_caption_style(
        CaptionStyleConfig(capitalization=CaptionCapitalization.LOWERCASE)
    )

    ass_orig = r_orig.build(words)
    ass_upper = r_upper.build(words)
    ass_lower = r_lower.build(words)

    assert "{\\k50}Hola {\\k50}Mundo" in ass_orig or "{\\k50}Hola{\\k50}Mundo" in ass_orig
    assert "HOLA" in ass_upper
    assert "MUNDO" in ass_upper
    assert "hola" in ass_lower
    assert "mundo" in ass_lower

    # Timing en el evento de diálogo debe ser idéntico
    line_orig = next(line for line in ass_orig.splitlines() if line.startswith("Dialogue:"))
    line_upper = next(line for line in ass_upper.splitlines() if line.startswith("Dialogue:"))
    line_lower = next(line for line in ass_lower.splitlines() if line.startswith("Dialogue:"))

    # Extraer marcas de tiempo Dialogue: 0,Start,End,...
    prefix_orig = line_orig.split(",Default,,0,0,0,,")[0]
    prefix_upper = line_upper.split(",Default,,0,0,0,,")[0]
    prefix_lower = line_lower.split(",Default,,0,0,0,,")[0]
    assert prefix_orig == prefix_upper == prefix_lower


def test_default_caption_style_does_not_change_contract_digest() -> None:
    """CaptionStyleConfig por defecto no mueve el digest del contrato v1.1."""
    contract_def = make_contract()
    assert hasattr(contract_def, "caption_style")
    assert contract_def.caption_style == CaptionStyleConfig()

    # El digest de make_contract() con CaptionStyleConfig() por defecto es el canónico
    assert contract_digest(contract_def) == (
        "b307aa91c957f7ae38849c2e21f3e67bc4891ad948cc7c7e5e7814e94b09e006"
    )


def test_custom_caption_style_changes_contract_digest() -> None:
    """Un caption_style personalizado NO se poda y altera el contract_digest."""
    contract_def = make_contract()
    contract_custom = make_contract(
        caption_style=CaptionStyleConfig(capitalization=CaptionCapitalization.UPPERCASE)
    )

    digest_def = contract_digest(contract_def)
    digest_custom = contract_digest(contract_custom)
    assert digest_def != digest_custom


def test_make_zero_contract_passes_custom_style() -> None:
    """make_zero_contract y vanilla_contract aceptan caption_style opcional."""
    custom = CaptionStyleConfig(highlight_color="#00FF00")
    c1 = make_zero_contract(campaign_id="c1", caption_style=custom)
    c2 = vanilla_contract(campaign_id="c2", caption_style=custom)
    c_def = make_zero_contract(campaign_id="c_def")

    assert c1.caption_style.highlight_color == "&H0000FF00"
    assert c2.caption_style.highlight_color == "&H0000FF00"
    assert c_def.caption_style == CaptionStyleConfig()


def test_resolver_resolves_caption_style_draft(tmp_path: Path) -> None:
    """El resolver convierte CaptionStyleDraft en CaptionStyleConfig."""
    platforms_payload: dict[str, object] = {"tiktok": {}}
    caption_payload: dict[str, object] = {
        "capitalization": "uppercase",
        "highlight_color": "#FF0000",
        "fontsize": 60,
    }
    draft = make_draft(platforms=platforms_payload, caption_style=caption_payload)
    registry = AssetRegistry(tmp_path)
    result = resolve_contract(draft, registry=registry, brief_text="cita del brief para resolver")
    assert result.contract is not None
    assert result.contract.caption_style.capitalization is CaptionCapitalization.UPPERCASE
    assert result.contract.caption_style.highlight_color == "&H000000FF"
    assert result.contract.caption_style.fontsize == 60
