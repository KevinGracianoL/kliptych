"""Canal de degradacion EN CALIENTE: lo que se pidio y lo que se uso.

Hoy `run_ffmpeg_with_fallback` reintenta con libx264 cuando el encoder pedido
falla, y lo unico que queda es un `logger.warning`. Ni `RenderConfig`, ni
`EnvironmentReport.degradations`, ni el manifiesto, ni `checkpoint.json` lo
registran. Con NVENC apagado eso es inalcanzable; el dia que se encienda, el
artefacto podria decir que pidio NVENC mientras el archivo es libx264.

Los argv se construyen a mano con `h264_nvenc`, como ya hace
`tests/test_audit_blockers.py`. Ningun test construye un `RenderConfig` con
NVENC activo: esta PR no toca el plumbariado de #88.
"""

import json
import subprocess
from pathlib import Path
from typing import cast, override

import pytest

from kliptych.assembler import RenderSpec
from kliptych.config import Settings
from kliptych.contract import ContractDraft
from kliptych.degradations import recolectar
from kliptych.encoding import RenderConfig, run_ffmpeg_with_fallback
from kliptych.environment import EnvironmentReport
from kliptych.pipeline import PieceAssembler, RunRequest, run_given_clips
from kliptych.pipeline_state import PipelineStage, PipelineStateManager
from kliptych.runtime import CampaignModel, Caption
from tests.support import FakeProbe, make_asset_draft, make_draft, make_media

_ARGV_NVENC = [
    "ffmpeg",
    "-hide_banner",
    "-v",
    "error",
    "-y",
    "-f",
    "lavfi",
    "-i",
    "testsrc=size=320x240:rate=5:duration=1",
    "-c:v",
    "h264_nvenc",
    "-preset",
    "p5",
    "-cq",
    "23",
    "-pix_fmt",
    "yuv420p",
    "salida.mp4",
]


class _FallaSoloNvenc:
    """Runner que rechaza h264_nvenc y acepta el resto, como un driver roto."""

    def __init__(self) -> None:
        self.llamadas: list[list[str]] = []

    def __call__(self, argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        self.llamadas.append(list(argv))
        if "h264_nvenc" in argv:
            return subprocess.CompletedProcess(argv, 1, "", "No NVENC capable devices found")
        return subprocess.CompletedProcess(argv, 0, "", "")


class _ModeloFijo(CampaignModel):
    """Modelo de prueba con draft fijo y caption fija."""

    _borrador: ContractDraft

    def __init__(self) -> None:
        self._borrador = make_draft(
            platforms={"tiktok": {}},
            assets={"required": [make_asset_draft()], "optional": []},
        )

    @override
    def extract_contract(self, brief: str) -> ContractDraft:
        _ = brief
        return self._borrador

    @override
    def write_caption(self, contract: object, piece: object) -> Caption:
        _ = (contract, piece)
        return Caption(caption="mira @marca #marca", hashtags=("#marca",))


class _EnsambladorFijo(PieceAssembler):
    """Ensamblador de prueba: escribe bytes sin tocar ffmpeg."""

    @override
    def assemble(self, spec: RenderSpec) -> Path:
        spec.destination.parent.mkdir(parents=True, exist_ok=True)
        _ = spec.destination.write_bytes(b"video")
        return spec.destination

    @override
    def render_arguments(self, spec: RenderSpec) -> tuple[str, ...]:
        _ = (spec.watermark, spec.watermark_config, spec.subtitles, spec.mute_audio)
        return ("ffmpeg", str(spec.clip), str(spec.destination))


def _forzar_fallback() -> _FallaSoloNvenc:
    """Provoca una caida real de h264_nvenc a libx264.

    Returns:
        El runner inyectado, con las llamadas que recibio.
    """
    runner = _FallaSoloNvenc()
    run_ffmpeg_with_fallback(
        _ARGV_NVENC,
        render=RenderConfig(),
        error_cls=RuntimeError,
        runner=runner,
    )
    assert [("h264_nvenc" in llamada) for llamada in runner.llamadas] == [True, False]
    return runner


class _SinteticoProbe:
    """Reemplaza la CLASE `FFprobeProbe` por un `FakeProbe` ya construido."""

    _probe: FakeProbe

    def __init__(self, probe: FakeProbe) -> None:
        self._probe = probe

    def __call__(self) -> FakeProbe:
        return self._probe


def _corrida_given_clips(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Corre `given_clips` de verdad y devuelve la ruta de su manifiesto.

    Args:
        tmp_path: Directorio de trabajo.
        monkeypatch: Fixture para el probe sintetico.

    Returns:
        La ruta del `run_manifest.json` que escribio la corrida.
    """
    _ = (tmp_path / "clip.mp4").write_bytes(b"clip")
    _ = monkeypatch.setattr(
        "kliptych.pipeline.FFprobeProbe",
        _SinteticoProbe(FakeProbe(info=make_media(width=608, height=1080))),
    )
    resultado = run_given_clips(
        model=_ModeloFijo(),
        settings=Settings.from_root(tmp_path),
        request=RunRequest(
            brief="cita del brief",
            destination=tmp_path / "package",
            environment=_entorno(),
            model_version="static",
            prompt_version="p",
            caption_prompt_version="c",
            run_id="run-degradacion",
            assembler=_EnsambladorFijo(),
        ),
    )
    return Path(resultado.manifest_path)


def _entorno() -> EnvironmentReport:
    """Entorno de prueba con UNA degradacion de pre-vuelo.

    Returns:
        El informe de entorno con `degradations` no vacia.
    """
    return EnvironmentReport(
        ffmpeg_version="N-118380",
        nvenc_available=False,
        degradations=("pre-vuelo: algo",),
    )


def test_el_fallback_llega_al_checkpoint_con_ambos_codecs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """El registro dice que encoder se PIDIO y cual se USO, no solo que cayo."""
    monkeypatch.setattr("time.monotonic", lambda: 1000.0)
    with recolectar() as degradaciones:
        _ = _forzar_fallback()
        admin = PipelineStateManager(tmp_path)
        artefacto = tmp_path / "final.mp4"
        _ = artefacto.write_bytes(b"x")
        _ = admin.mark_started(PipelineStage.REFRAME)
        _ = admin.mark_done(PipelineStage.REFRAME, artefacto)

    en_disco = cast(
        "dict[str, object]",
        json.loads((tmp_path / "checkpoint.json").read_text(encoding="utf-8")),
    )
    eventos = cast("list[str]", en_disco["degradations"])

    assert eventos == list(degradaciones.como_tupla())
    assert "h264_nvenc" in eventos[0]
    assert "libx264" in eventos[0]


def test_el_fallback_llega_al_manifiesto_con_ambos_codecs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """El manifiesto ya tiene `degradations`; hoy solo viene del pre-vuelo.

    La degradacion de caliente se provoca dentro de la MISSA corrida, asi que el
    que la recoge es el `_write_manifest` de produccion, no un helper del test.
    """
    monkeypatch.setattr("time.monotonic", lambda: 1000.0)
    with recolectar():
        _ = _forzar_fallback()
        manifiesto = _corrida_given_clips(tmp_path, monkeypatch)

    crudo = cast("dict[str, object]", json.loads(manifiesto.read_text(encoding="utf-8")))
    todas = cast("list[str]", crudo["degradations"])

    assert "pre-vuelo: algo" in todas
    caliente = [d for d in todas if d != "pre-vuelo: algo"]
    assert len(caliente) == 1, todas
    assert "h264_nvenc" in caliente[0]
    assert "libx264" in caliente[0]


def test_un_run_sin_fallback_no_registra_degradacion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """La ausencia de un evento no se escribe como evento.

    El checkpoint se escribe DENTRO del colector, asi que si el codigo inventara
    una entrada vacia, aqui se veria.
    """
    monkeypatch.setattr("time.monotonic", lambda: 1000.0)
    with recolectar() as degradaciones:
        admin = PipelineStateManager(tmp_path)
        artefacto = tmp_path / "final.mp4"
        _ = artefacto.write_bytes(b"x")
        _ = admin.mark_started(PipelineStage.REFRAME)
        _ = admin.mark_done(PipelineStage.REFRAME, artefacto)

    en_disco = cast(
        "dict[str, object]",
        json.loads((tmp_path / "checkpoint.json").read_text(encoding="utf-8")),
    )

    assert degradaciones.como_tupla() == ()
    assert en_disco["degradations"] == []


def test_una_caida_sin_colector_no_embarca_en_silencio() -> None:
    """Fail-closed: una degradacion que no se puede registrar es un error.

    Si nadie pudiera recogerla, el artefacto diria que pidio NVENC mientras el
    archivo es libx264. Eso es peor que el bug que esta PR arregla.
    """
    runner = _FallaSoloNvenc()

    with pytest.raises(RuntimeError, match="no hay colector"):
        _ = run_ffmpeg_with_fallback(
            _ARGV_NVENC,
            render=RenderConfig(),
            error_cls=RuntimeError,
            runner=runner,
        )
