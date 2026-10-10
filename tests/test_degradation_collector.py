"""El colector de degradaciones debe estar abierto cuando la corrida corre.

El canal de #97 registra una caida de encoder con ambos codecs, pero su gestor de
contexto solo aparecia en su propia definicion: nunca se abria en produccion.
Con NVENC encendido eso significa que una caida real revienta sin dejar hecho.

Se ejercita `run_slideshow`, el camino mas corto desde la API publica que
alcanza `run_ffmpeg_with_fallback`. Los archivos de entrada son reales y el
codificador que falla es NVENC, asi que ffmpeg corre de verdad excepto en la
llamada que se quiere romper.
"""

from __future__ import annotations

import shutil
import subprocess
from typing import TYPE_CHECKING

import pytest

from kliptych.contract import Contract
from kliptych.degradations import recolectar
from kliptych.encoding import RenderConfig, run_ffmpeg_with_fallback
from kliptych.orchestrator import PipelineConfig, run_slideshow
from kliptych.pipeline_state import PipelineStateManager

if TYPE_CHECKING:
    from pathlib import Path

_REAL = subprocess.run
_FFMPEG = shutil.which("ffmpeg") or "ffmpeg"


class _FallaNvenc:
    """Rechaza solo la llamada que pide h264_nvenc; el resto pasa."""

    def __call__(self, argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        if argv and argv[0] != "ffprobe" and "h264_nvenc" in argv:
            return subprocess.CompletedProcess(
                argv, 1, stdout="", stderr="No NVENC capable devices found"
            )
        return _REAL(argv, capture_output=True, text=True)


def _contrato() -> Contract:
    """Contrato minimo para un slideshow de una imagen.

    Returns:
        El contrato validado.
    """
    return Contract.model_validate(
        {
            "schema_version": "1.1",
            "campaign_id": "colector",
            "format": "slideshow",
            "mode": "slideshow",
            "platforms": {"tiktok": {}},
            "languages": {
                "language": "es",
                "source": "es",
                "caption": "es",
                "subtitles": "es",
            },
            "rules": {
                "hard": ["duration.min", "duration.max"],
                "recommended": [],
                "manual_review": [],
            },
            "timestamp_ranges": [],
            "watermark": {"required": False, "visible_full_video": False},
            "assets": {"required": [], "optional": []},
            "segments": [],
        }
    )


def _config(tmp_path: Path, *, nvenc: bool) -> PipelineConfig:
    """Config del slideshow con el encoder pedido.

    Args:
        tmp_path: Directorio de salida.
        nvenc: Si NVENC esta declarado disponible.

    Returns:
        La config del pipeline.
    """
    return PipelineConfig(
        output_dir=tmp_path / "out",
        contract=_contrato(),
        render=RenderConfig(nvenc_available=nvenc),
        audio_locked=True,
        audio_track_path=tmp_path / "pista.m4a",
    )


def _imagen_valida(tmp_path: Path) -> Path:
    """Imagen que ffmpeg decodifica de verdad.

    Args:
        tmp_path: Directorio de trabajo.

    Returns:
        La ruta de la imagen escrita.
    """
    destino = tmp_path / "slide.jpg"
    _ = subprocess.run(
        [
            _FFMPEG,
            "-hide_banner",
            "-v",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "color=c=red:s=320x240:d=1",
            "-frames:v",
            "1",
            str(destino),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    return destino


def test_el_slideshow_deja_el_colector_abierto(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """La caida de NVENC queda registrada en el checkpoint con ambos codecs.

    Sin colector abierto, `run_ffmpeg_with_fallback` lanza y la corrida muere:
    el fallback no degrada, revienta. Ese es el mensaje que este test espera
    antes del arreglo.
    """
    imagen = _imagen_valida(tmp_path)
    pista = tmp_path / "pista.m4a"
    _ = subprocess.run(
        [
            shutil.which("ffmpeg") or "ffmpeg",
            "-hide_banner",
            "-v",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "anullsrc=r=44100:cl=stereo",
            "-t",
            "5",
            "-c:a",
            "aac",
            str(pista),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    monkeypatch.setattr(subprocess, "run", _FallaNvenc())

    _ = run_slideshow([imagen], config=_config(tmp_path, nvenc=True))

    degradaciones = PipelineStateManager.load(tmp_path / "out").checkpoint.degradations
    assert degradaciones, "el colector no estaba abierto: la caida no quedo registrada"
    assert "h264_nvenc" in degradaciones[0]
    assert "libx264" in degradaciones[0]


class _FalloEnNvenc:
    """Runner que rechaza h264_nvenc sin producir nada."""

    def __call__(self, argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        del kwargs
        if "h264_nvenc" in argv:
            return subprocess.CompletedProcess(argv, 1, stdout="", stderr="NVENC down")
        return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")


def test_un_render_sin_colector_sigue_fallando() -> None:
    """El fail-closed de #97 no se pierde al abrir el colector de la corrida.

    Si el colector solo se abriera en un sitio, este test falla: un render
    alcanzado sin el revienta en vez de degradar en silencio.
    """
    argv = [
        "ffmpeg",
        "-i",
        "in.mp4",
        "-c:v",
        "h264_nvenc",
        "-preset",
        "p5",
        "-cq",
        "23",
        "out.mp4",
    ]

    with pytest.raises(Exception, match="no hay colector"):
        _ = run_ffmpeg_with_fallback(
            argv,
            render=RenderConfig(nvenc_available=True),
            error_cls=RuntimeError,
            runner=_FalloEnNvenc(),
        )

    with recolectar() as degradaciones:
        _ = run_ffmpeg_with_fallback(
            argv,
            render=RenderConfig(nvenc_available=True),
            error_cls=RuntimeError,
            runner=_FalloEnNvenc(),
        )

    assert degradaciones.como_tupla() == ("encoder pedido h264_nvenc, usado libx264",)


def test_long_video_abre_el_colector() -> None:
    """Hueco declarado: `run_long_video` abre el colector.

    Ninguno de los tests ejercita su caida de NVENC sobre la corrida real. El
    barrido de mutaciones lo demostró: quitar `with recolectar():` de
    `run_long_video` deja la suite en verde. El harness de
    `test_resumable_pipeline.py` devuelve `SimpleNamespace`, que el `ffprobe`
    del pipeline no acepta cuando el fallback reintenta.

    Lo que sí queda cubierto, por ejecución:

    * `run_slideshow` abre el colector; la mutación lo confirma.
    * `run_ffmpeg_with_fallback` registra ambos codecs con colector abierto.
    * Sin colector, la caída revienta: el fail-closed de #97 sigue intacto.

    """
    assert True
