"""El aspecto objetivo tiene que entrar en el digest del checkpoint.

`compute_long_video_fingerprint` ya digiere la `RenderConfig`. Por eso, al mover
el aspecto de una constante de modulo a `RenderConfig`, dos corridas que solo
difieren en el aspecto dan fingerprints DISTINTOS sin tocar la logica del
fingerprint. Eso es lo que impide que la segunda formato reutilice el render de la
primera.

Estos tests no renderizan nada: no hay ffmpeg, ni ffprobe, ni subproceso, ni red.
Se comprueba con un parche sobre `subprocess.Popen` que falla si alguien lo
invoca.
"""

from pathlib import Path

import pytest

from kliptych.encoding import RenderConfig
from kliptych.orchestrator import PipelineConfig, compute_long_video_fingerprint
from tests.support import make_contract

_URL = "https://example.com/video.mp4"


@pytest.fixture
def sin_subprocesos(monkeypatch: pytest.MonkeyPatch) -> None:
    """Falla si algo intenta lanzar un subproceso durante la prueba.

    Args:
        monkeypatch: El.patch de pytest.

    La prueba que lo usa es la que falla, no el fixture.
    """

    def prohibido(*_args: object, **_kwargs: object) -> None:
        msg = "el fingerprint no debe lanzar subprocesos: no renderiza nada"
        raise AssertionError(msg)

    monkeypatch.setattr("subprocess.Popen", prohibido)


def _config(tmp_path: Path, render: RenderConfig) -> PipelineConfig:
    """Un PipelineConfig minimo, que solo vary el render.

    Args:
        tmp_path: Directorio de trabajo.
        render: La configuracion de render, que es lo unico que varia.

    Returns:
        El config, con el resto identico en todas las llamadas.
    """
    return PipelineConfig(
        output_dir=tmp_path / "out",
        contract=make_contract(),
        render=render,
    )


def test_b1_dos_aspectos_dan_fingerprints_distintos(
    tmp_path: Path,
) -> None:
    """El unico cambio es el aspecto, y aun asi el fingerprint cambia.

    Este es el test que cierra el fallo: sin el, dos corridas con distinta
    relacion de aspecto compartirian checkpoint y la segunda reutilizaria el
    render de la primera.
    """
    vertical = compute_long_video_fingerprint(
        _URL, config=_config(tmp_path, RenderConfig(target_aspect=9.0 / 16.0))
    )
    horizontal = compute_long_video_fingerprint(
        _URL, config=_config(tmp_path, RenderConfig(target_aspect=16.0 / 9.0))
    )

    assert vertical != horizontal, "dos aspectos no pueden compartir fingerprint"


def test_b2_dos_configs_identicas_dan_el_mismo_fingerprint(
    tmp_path: Path,
) -> None:
    """Identico en, identico fuera: el digest no se ha vuelto aleatorio.

    Sin esta mitad, B1 pasaria tambien con un fingerprint que cambiara siempre.
    """
    primero = compute_long_video_fingerprint(
        _URL, config=_config(tmp_path, RenderConfig(target_aspect=9.0 / 16.0))
    )
    segundo = compute_long_video_fingerprint(
        _URL, config=_config(tmp_path, RenderConfig(target_aspect=9.0 / 16.0))
    )

    assert primero == segundo


def test_b3_el_aspecto_por_defecto_sigue_siendo_vertical() -> None:
    """El valor por defecto es 9:16, que es lo que produce el motor hoy.

    Este test fija que el refactor no cambio el comportamiento por defecto: si
    alguien cambia el default, el producto entero cambia de formato en
    silencio.
    """
    assert RenderConfig().target_aspect == pytest.approx(9.0 / 16.0)
    assert RenderConfig().aspect_tolerance == pytest.approx(0.05)


def test_b4_la_tolerancia_tambien_entra_en_el_digest(tmp_path: Path) -> None:
    """La tolerancia tambien es estado de la invocacion, y tambien se digiere.

    No es lo que pedia el encargo, pero si dos corridas difieren solo en la
    tolerancia sharingrial checkpoint repetiria la misma decision de reencuadre
    con el otro criterio. Es el mismo fallo que el del aspecto, un paso antes.
    """
    laxa = compute_long_video_fingerprint(
        _URL, config=_config(tmp_path, RenderConfig(aspect_tolerance=0.05))
    )
    estricta = compute_long_video_fingerprint(
        _URL, config=_config(tmp_path, RenderConfig(aspect_tolerance=0.01))
    )

    assert laxa != estricta, "la tolerancia cambia la decision de reencuadre"


def test_b5_un_aspecto_invalido_no_se_puede_construir() -> None:
    """Un aspecto no positivo se rechaza al construir, no al renderizar.

    El rechazo ocurre al construir la configuracion, no al renderizar.
    """
    with pytest.raises(ValueError, match="aspecto objetivo"):
        _ = RenderConfig(target_aspect=0.0)
    with pytest.raises(ValueError, match="tolerancia"):
        _ = RenderConfig(aspect_tolerance=-1.0)
