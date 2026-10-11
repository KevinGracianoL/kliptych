"""Pruebas de LayoutSpec y ensamblado multifranga (Dividido, Trío, Spotlight)."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from kliptych.assembler import FFmpegAssembler, RenderSpec
from kliptych.assets import AssetRegistry
from kliptych.contract import (
    Band,
    Layout,
    LayoutSpec,
    NormalizedRect,
    contract_digest,
)
from kliptych.resolver import resolve_contract
from tests.support import make_contract, make_draft


def test_layout_enum_members() -> None:
    """El enum Layout contiene los layouts multifranga verificados."""
    assert Layout.SINGLE == "single"
    assert Layout.SPLIT_SCREEN == "split_screen"
    assert Layout.VERTICAL == "vertical"
    assert Layout.HORIZONTAL == "horizontal"
    assert Layout.DIVIDIDO == "dividido"
    assert Layout.TRIO == "trio"
    assert Layout.SPOTLIGHT == "spotlight"


def test_normalized_rect_defaults_and_custom() -> None:
    """NormalizedRect tiene defaults en [0, 1] y acepta coordenadas válidas."""
    rect = NormalizedRect()
    assert rect.x == pytest.approx(0.0)
    assert rect.y == pytest.approx(0.0)
    assert rect.width == pytest.approx(1.0)
    assert rect.height == pytest.approx(1.0)

    custom = NormalizedRect(x=0.1, y=0.2, width=0.5, height=0.6)
    assert custom.x == pytest.approx(0.1)
    assert custom.y == pytest.approx(0.2)
    assert custom.width == pytest.approx(0.5)
    assert custom.height == pytest.approx(0.6)


@pytest.mark.parametrize(
    ("x", "y", "w", "h"),
    [
        (0.6, 0.0, 0.5, 1.0),  # x + width = 1.1 > 1.0
        (0.0, 0.6, 1.0, 0.5),  # y + height = 1.1 > 1.0
        (-0.1, 0.0, 0.5, 0.5),  # x < 0
        (0.0, -0.1, 0.5, 0.5),  # y < 0
        (0.0, 0.0, 0.0, 0.5),  # width == 0
        (0.0, 0.0, 0.5, 0.0),  # height == 0
        (0.0, 0.0, 1.1, 0.5),  # width > 1.0
        (0.0, 0.0, 0.5, 1.1),  # height > 1.0
    ],
)
def test_normalized_rect_rejects_out_of_bounds(x: float, y: float, w: float, h: float) -> None:
    """NormalizedRect rechaza coordenadas que se salgan de [0.0, 1.0]."""
    with pytest.raises(ValidationError):
        _ = NormalizedRect(x=x, y=y, width=w, height=h)


def test_band_defaults_and_custom() -> None:
    """Band define source_rect por defecto y source opcional."""
    band = Band()
    assert band.source_rect == NormalizedRect()
    assert band.source is None

    custom = Band(
        source_rect=NormalizedRect(x=0.0, y=0.0, width=1.0, height=0.5),
        source="clip-asset",
    )
    assert custom.source == "clip-asset"
    assert custom.source_rect.height == pytest.approx(0.5)


def test_layout_spec_band_count_validation() -> None:
    """LayoutSpec valida que el conteo de franjas corresponda al layout."""
    # DIVIDIDO y SPOTLIGHT exigen exactamente 2 franjas
    spec_div = LayoutSpec(layout=Layout.DIVIDIDO, bands=(Band(), Band()))
    assert len(spec_div.bands) == 2

    spec_spot = LayoutSpec(layout=Layout.SPOTLIGHT, bands=(Band(), Band()))
    assert len(spec_spot.bands) == 2

    with pytest.raises(ValidationError):
        _ = LayoutSpec(layout=Layout.DIVIDIDO, bands=(Band(),))
    with pytest.raises(ValidationError):
        _ = LayoutSpec(layout=Layout.DIVIDIDO, bands=(Band(), Band(), Band()))

    with pytest.raises(ValidationError):
        _ = LayoutSpec(layout=Layout.SPOTLIGHT, bands=(Band(),))

    # TRIO exige exactamente 3 franjas
    spec_trio = LayoutSpec(layout=Layout.TRIO, bands=(Band(), Band(), Band()))
    assert len(spec_trio.bands) == 3

    with pytest.raises(ValidationError):
        _ = LayoutSpec(layout=Layout.TRIO, bands=(Band(), Band()))
    with pytest.raises(ValidationError):
        _ = LayoutSpec(layout=Layout.TRIO, bands=(Band(), Band(), Band(), Band()))

    # VERTICAL, HORIZONTAL, SINGLE admiten 0 o 1 franja
    _ = LayoutSpec(layout=Layout.SINGLE, bands=())
    _ = LayoutSpec(layout=Layout.SINGLE, bands=(Band(),))
    _ = LayoutSpec(layout=Layout.VERTICAL, bands=(Band(),))
    _ = LayoutSpec(layout=Layout.HORIZONTAL, bands=(Band(),))

    with pytest.raises(ValidationError):
        _ = LayoutSpec(layout=Layout.SINGLE, bands=(Band(), Band()))


def test_layout_spec_band_heights_1920() -> None:
    """band_heights genera alturas pares que suman exactamente la altura del lienzo."""
    div = LayoutSpec(layout=Layout.DIVIDIDO, bands=(Band(), Band()))
    h_div = div.band_heights(1920)
    assert h_div == (960, 960)
    assert sum(h_div) == 1920
    assert all(h % 2 == 0 for h in h_div)

    trio = LayoutSpec(layout=Layout.TRIO, bands=(Band(), Band(), Band()))
    h_trio = trio.band_heights(1920)
    assert h_trio == (640, 640, 640)
    assert sum(h_trio) == 1920
    assert all(h % 2 == 0 for h in h_trio)

    spot = LayoutSpec(layout=Layout.SPOTLIGHT, bands=(Band(), Band()))
    h_spot = spot.band_heights(1920)
    assert h_spot == (1080, 840)
    assert sum(h_spot) == 1920
    assert all(h % 2 == 0 for h in h_spot)

    vert = LayoutSpec(layout=Layout.VERTICAL, bands=(Band(),))
    assert vert.band_heights(1920) == (1920,)


def test_contract_prune_unset_layout_spec_preserves_digest() -> None:
    """_prune_unset_options poda layout_spec=None preservando el golden digest."""
    contract = make_contract()
    assert hasattr(contract, "layout_spec")
    assert contract.layout_spec is None
    assert contract_digest(contract) == (
        "b307aa91c957f7ae38849c2e21f3e67bc4891ad948cc7c7e5e7814e94b09e006"
    )


def test_contract_with_layout_spec_changes_digest() -> None:
    """Declarar layout_spec personalizado altera el contract_digest."""
    base = contract_digest(make_contract())
    spec = LayoutSpec(layout=Layout.DIVIDIDO, bands=(Band(), Band()))
    with_spec = contract_digest(make_contract(layout_spec=spec))
    assert base != with_spec


def test_build_layout_spec_argv_dividido_and_trio(tmp_path: Path) -> None:
    """_build_layout_spec_argv genera vstack y setsar=1 para Dividido y Trío."""
    clip = tmp_path / "clip.mp4"
    _ = clip.write_bytes(b"dummy clip")
    dest = tmp_path / "out.mp4"

    assembler = FFmpegAssembler()

    # DIVIDIDO: 2 franjas
    spec_div = RenderSpec(
        clip=clip,
        destination=dest,
        layout=Layout.DIVIDIDO,
        layout_spec=LayoutSpec(layout=Layout.DIVIDIDO, bands=(Band(), Band())),
    )
    argv_div = assembler.render_arguments(spec_div)
    assert "-filter_complex" in argv_div
    graph_div = argv_div[argv_div.index("-filter_complex") + 1]
    assert "vstack=inputs=2" in graph_div
    assert "scale=1080:960" in graph_div
    assert "setsar=1" in graph_div
    assert "[v_stacked]" in graph_div or "[v]" in graph_div

    # TRIO: 3 franjas
    spec_trio = RenderSpec(
        clip=clip,
        destination=dest,
        layout=Layout.TRIO,
        layout_spec=LayoutSpec(layout=Layout.TRIO, bands=(Band(), Band(), Band())),
    )
    argv_trio = assembler.render_arguments(spec_trio)
    assert "-filter_complex" in argv_trio
    graph_trio = argv_trio[argv_trio.index("-filter_complex") + 1]
    assert "vstack=inputs=3" in graph_trio
    assert "scale=1080:640" in graph_trio
    assert "setsar=1" in graph_trio


def test_build_layout_spec_argv_gap_pads_bands(tmp_path: Path) -> None:
    """El gap declarado en LayoutSpec se traduce en relleno negro por banda."""
    clip = tmp_path / "clip.mp4"
    _ = clip.write_bytes(b"dummy clip")
    dest = tmp_path / "out.mp4"

    assembler = FFmpegAssembler()
    spec = RenderSpec(
        clip=clip,
        destination=dest,
        layout=Layout.DIVIDIDO,
        layout_spec=LayoutSpec(layout=Layout.DIVIDIDO, bands=(Band(), Band()), gap=20),
    )
    argv = assembler.render_arguments(spec)
    graph = argv[argv.index("-filter_complex") + 1]

    # Las dos franjas rellenan hasta 960 px: el alto no cambia y sigue sumando 1920.
    assert "pad=1080:960:0:0:color=black" in graph
    assert graph.count("pad=1080:960:0:0:color=black") == 2
    # El contenido se escala al alto útil (960 menos el gap), no al alto de franja.
    assert "scale=1080:940:force_original_aspect_ratio=increase" in graph


def test_resolver_accepts_raw_dict_layout_spec(tmp_path: Path) -> None:
    """El resolutor acepta layout_spec como diccionario crudo, igual que caption_style."""
    draft = make_draft(
        layout_spec={
            "layout": "trio",
            "bands": [
                {"source_rect": {"width": 1.0, "height": 0.34}},
                {"source_rect": {"width": 1.0, "height": 0.33}},
                {"source_rect": {"width": 1.0, "height": 0.33}},
            ],
            "gap": 10,
        },
    )
    registry = AssetRegistry(tmp_path)
    result = resolve_contract(draft, registry=registry, brief_text="cita del brief para resolver")
    assert result.contract is not None
    assert result.contract.layout_spec is not None
    assert result.contract.layout_spec.layout is Layout.TRIO
    assert result.contract.layout_spec.gap == 10
    assert len(result.contract.layout_spec.bands) == 3


def test_resolver_resolves_layout_spec_draft(tmp_path: Path) -> None:
    """El resolver resuelve layout_spec desde un ContractDraft."""
    platforms_payload: dict[str, object] = {"tiktok": {}}
    layout_payload: dict[str, object] = {
        "layout": "dividido",
        "bands": [
            {"source_rect": {"x": 0.0, "y": 0.0, "width": 1.0, "height": 0.5}},
            {"source_rect": {"x": 0.0, "y": 0.5, "width": 1.0, "height": 0.5}},
        ],
    }
    draft = make_draft(platforms=platforms_payload, layout_spec=layout_payload)
    registry = AssetRegistry(tmp_path)
    result = resolve_contract(draft, registry=registry, brief_text="cita del brief para resolver")
    assert result.contract is not None
    assert result.contract.layout_spec is not None
    assert result.contract.layout_spec.layout is Layout.DIVIDIDO
    assert len(result.contract.layout_spec.bands) == 2
    assert result.contract.layout_spec.bands[0].source_rect.height == pytest.approx(0.5)
