"""El formato entra en el `piece_id`, y el paquete no colisiona.

PR 3 de la escalera de #76. #82 hizo que el gate verificara el aspecto DECLARADO
del artefacto; aqui se produce mas de un formato y el paquete tiene que llevar
los dos sin pisarse.

El problema que estos tests cubren es concreto: el exportador arma
`<campaign_id>/<platform>/<piece_id>.mp4`. Sin el formato dentro del `piece_id`,
dos formatos del mismo span seriam el mismo archivo. Y como el exportador falla
cerrado ante ids duplicados, la consecuencia no es sobrescritura silenciosa: es
la entrega ENTERA rechazada, con los doce archivos correctos de ayer y ninguno de
hoy.
"""

from pathlib import Path

import pytest

from kliptych.assets import AssetRegistry
from kliptych.campaign_manager import CampaignManagerError, piece_id_for
from kliptych.contract import Contract
from kliptych.contract.enums import Platform
from kliptych.exporter import DeliveryReport, ExportError, export_delivery
from kliptych.gate import Gate
from kliptych.gate.models import Piece
from tests.support import FakeProbe, make_contract, make_media

# El campaign_id mas largo que se probo: 41 caracteres, todos seguros.
_CAMPANA_LARGA = "campana_de_verano_del_mayor_2026_lookbook"


def _id_de(
    campana: str,
    *,
    multiple_platforms: bool,
    index: int | None,
    output_format: str | None = None,
) -> str:
    return piece_id_for(
        campana,
        Platform.TIKTOK,
        multiple_platforms=multiple_platforms,
        index=index,
        output_format=output_format,
    )


# ------------------------------------------------------- los cuatro casos


@pytest.mark.parametrize(
    ("caso", "multiple_platforms", "index"),
    [
        ("una plataforma, una salida", False, None),
        ("una plataforma, dos salidas", False, 1),
        ("dos plataformas, una salida", True, None),
        ("dos plataformas, dos salidas", True, 1),
    ],
)
def test_los_cuatro_casos_caben_en_64(
    *,
    caso: str,
    multiple_platforms: bool,
    index: int | None,
) -> None:
    """Los cuatro casos caben con un campaign_id realista de 41 caracteres.

    El peor caso medido son 56 con el sufijo `-16x9`, o sea 8 de holgura. Un
    campaign_id mas largo no se considera realista a proposito: por eso el
    Overflow de mas abajo tiene que fallar y no truncar.
    """
    piece_id = _id_de(
        _CAMPANA_LARGA,
        multiple_platforms=multiple_platforms,
        index=index,
        output_format="16x9",
    )

    assert len(piece_id) <= 64, f"{caso}: {len(piece_id)} caracteres: {piece_id}"
    assert piece_id.endswith("-16x9")


def test_sin_formatos_multiples_el_id_no_lleva_sufijo() -> None:
    """Con una sola salida el id no crece: el sufijo solo existe para distinguir."""
    sin = _id_de("camp-01", multiple_platforms=False, index=None)
    con = _id_de("camp-01", multiple_platforms=False, index=None, output_format="16x9")

    assert sin == "camp-01"
    assert con == "camp-01-16x9"


# ------------------------------------------------------------------ overflow


def test_un_id_que_no_cabe_falla_y_no_trunca() -> None:
    """Cuando no cabe, falla con el nombre completo. Nunca trunca.

    Truncar es el modo de fallo caro: dos ids que difieren en los ultimos
    caracteres se vuelven el mismo nombre, el paquete se publica con menos
    archivos de los producidos, y el error aparece en el receptor, no aqui.
    """
    largo = "c" * 63

    with pytest.raises(CampaignManagerError) as excinfo:
        _ = _id_de(largo, multiple_platforms=True, index=3, output_format="16x9")

    mensaje = str(excinfo.value)
    assert "excede el maximo" in mensaje
    # El nombre completo va en el mensaje: sin el, el operador no sabe que recortar.
    assert f"{largo}-tiktok-03-16x9" in mensaje


def test_el_limite_es_inclusive() -> None:
    """64 caben, 65 no. El borde exacto, no un margen de seguridad."""
    base = "c" * 59  # 59 + "-16x9" (5) = 64

    assert len(_id_de(base, multiple_platforms=False, index=None, output_format="16x9")) == 64

    with pytest.raises(CampaignManagerError):
        _ = _id_de("c" * 60, multiple_platforms=False, index=None, output_format="16x9")


# ----------------------------------------------------- colision en el paquete


def _pieza(piece_id: str, artefacto: Path, caption: str = "mira @marca #marca") -> Piece:
    return Piece(
        piece_id=piece_id,
        platform=Platform.TIKTOK,
        caption=caption,
        hashtags=("#marca",),
        artifact_path=artefacto,
    )


def _gate() -> Gate:
    return Gate(FakeProbe(info=make_media(width=1920, height=1080)))


def _contrato() -> Contract:
    return make_contract(
        hard=["artifact.integrity"],
        audio_rule="any",
        min_s=None,
        max_s=None,
        required_mentions=(),
        required_hashtags=(),
        schema_version="1.2",
        outputs=[{"aspect": "16:9", "index": 0}],
        prohibitions=["sorteo"],
    )


def _exportar(tmp_path: Path, pieces: list[Piece]) -> DeliveryReport:
    """Exporta las piezas a `<tmp>/delivery` y devuelve el reporte.

    No comprueba el estado: algunos de estos tests esperan `PARTIAL` o
    `BLOCKED`, y comprobarlo aqui los haria mentir sobre donde vive la
    decision.

    Returns:
        El reporte de la entrega, con su estado, sus piezas exportadas y sus
        rechazos.
    """
    return export_delivery(
        contract=_contrato(),
        pieces=pieces,
        gate=_gate(),
        assets=AssetRegistry(tmp_path),
        destination=tmp_path / "delivery",
    )


def test_dos_formatos_del_mismo_span_caen_en_dos_paths(tmp_path: Path) -> None:
    """La prueba de que el sufijo funciona: dos piezas, dos archivos.

    No basta con que los ids sean distintos: tiene que haber dos archivos en
    disco, con el nombre que el receptor va a ver.
    """
    comun = tmp_path / "clip.mp4"
    _ = comun.write_bytes(b"video")
    vertical = _pieza("camp-01-9x16", comun)
    horizontal = _pieza("camp-01-16x9", comun)

    _ = _exportar(tmp_path, [vertical, horizontal])

    publicados = sorted(p.name for p in (tmp_path / "delivery" / "camp-test" / "tiktok").iterdir())
    assert publicados == [
        "camp-01-16x9.gate.json",
        "camp-01-16x9.metadata.json",
        "camp-01-16x9.mp4",
        "camp-01-9x16.gate.json",
        "camp-01-9x16.metadata.json",
        "camp-01-9x16.mp4",
    ]


def test_sin_el_sufijo_el_exportador_rechaza_toda_la_entrega(tmp_path: Path) -> None:
    """El contraejemplo: sin formato en el id, la entrega entera no se publica.

    Falla cerrado, que es lo correcto. La leccion para quien quite el sufijo es
    que no se pierde un archivo: se pierde el paquete.
    """
    comun = tmp_path / "clip.mp4"
    _ = comun.write_bytes(b"video")
    piezas = [
        _pieza("camp-01", comun),
        _pieza("camp-01", comun),
    ]

    with pytest.raises(ExportError, match="duplicado"):
        _ = _exportar(tmp_path, piezas)

    assert not (tmp_path / "delivery" / "camp-01").exists()


def test_un_rechazo_no_publica_nada(tmp_path: Path) -> None:
    """24 artefactos y un rechazo dan cero paquetes. La atomicidad no se toca.

    Se comprueba con una pieza buena y una mala, que es el caso que la
    atomicidad existe para cubrir: el destino previo queda intacto.
    """
    destino = tmp_path / "delivery"
    _ = destino.mkdir()
    _ = (destino / "previo.txt").write_text("paquete anterior", encoding="utf-8")

    comun = tmp_path / "clip.mp4"
    _ = comun.write_bytes(b"video")
    # La unica diferencia entre las dos es el caption: una lleva un termino
    # prohibido y la otra no. Asi el rechazo es POR PIEZA, que es lo que la
    # atomicidad tiene que cubrir.
    buena = _pieza("camp-01-16x9", comun)
    mala = _pieza("camp-01-9x16", comun, caption="mira @marca #marca sorteo")

    _ = export_delivery(
        contract=_contrato(),
        pieces=[buena, mala],
        gate=_gate(),
        assets=AssetRegistry(tmp_path),
        destination=destino,
    )

    assert not destino.exists() or sorted(p.name for p in destino.iterdir()) == ["previo.txt"]
    assert destino.is_dir()
    # Nada de la corrida nueva se publico: el paquete anterior sigue entero.
    assert sorted(p.name for p in destino.iterdir()) == ["previo.txt"]
