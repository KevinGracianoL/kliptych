"""NVENC encendido: el encoder detectado llega al argv de ffmpeg.

Estos tests afirman sobre el ARGV, no sobre la config. Una config con
`nvenc_available=True` que no produzca `-c:v h264_nvenc` en el comando real no
le sirve a nadie: el pipeline habria pedido la GPU y codificado en CPU.

La CLI `run` no se prueba aqui porque sus dobles viven en
`tests/test_audit_cli_run_exit_code.py`, e importarlos cruzaria modulos de test,
que el repo evita. Ese camino lo cubre `test_audit_cli_run_nvenc_llega_al_render`
en ese archivo.
"""

import json
import subprocess
from pathlib import Path
from typing import ClassVar, cast, no_type_check, override

import pytest

from kliptych.assembler import FFmpegAssembler, RenderSpec
from kliptych.config import Settings
from kliptych.contract import ContractDraft, Layout
from kliptych.degradations import recolectar
from kliptych.encoding import Invocation, RenderConfig, run_ffmpeg_with_fallback
from kliptych.environment import EnvironmentReport
from kliptych.pipeline import PieceAssembler, RunRequest, run_given_clips
from kliptych.runtime import CampaignModel, Caption
from tests.support import FakeProbe, make_asset_draft, make_draft, make_media

_ENTORNO_NVENC = EnvironmentReport(
    ffmpeg_version="N-118380",
    nvenc_available=True,
    gpu=None,
)
_ENTORNO_SIN_NVENC = EnvironmentReport(ffmpeg_version="N-118380", nvenc_available=False)


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
        del brief
        return self._borrador

    @override
    def write_caption(self, contract: object, piece: object) -> Caption:
        del contract, piece
        return Caption(caption="mira @marca #marca", hashtags=("#marca",))


class _SinteticoProbe:
    """Reemplaza la CLASE `FFprobeProbe` por un `FakeProbe` ya construido."""

    _probe: FakeProbe

    def __init__(self, probe: FakeProbe) -> None:
        self._probe = probe

    def __call__(self) -> FakeProbe:
        return self._probe


class _EspiaAssembler(PieceAssembler):
    """Espia el `render` que le pasa el pipeline y delega el argv en el real."""

    _real: FFmpegAssembler
    vistos: ClassVar[list[RenderConfig]] = []

    def __init__(self, *, render: RenderConfig | None = None) -> None:
        self._real = FFmpegAssembler(render=render)
        _EspiaAssembler.vistos.append(RenderConfig() if render is None else render)

    @override
    def assemble(self, spec: RenderSpec) -> Path:
        """Escribe el destino sin tocar ffmpeg.

        Args:
            spec: Entradas inmutables del render.

        Returns:
            La ruta del destino.
        """
        spec.destination.parent.mkdir(parents=True, exist_ok=True)
        _ = spec.destination.write_bytes(b"video")
        return spec.destination

    @override
    @override
    def render_arguments(self, spec: RenderSpec) -> tuple[str, ...]:
        del spec
        return self._real.render_arguments(_espec())

    def render_arguments_viejo(self, spec: RenderSpec) -> tuple[str, ...]:
        """Devuelve el argv real del assembler, que decide el codec.

        Args:
            spec: Entradas inmutables del render.

        Returns:
            El argv completo que produciria el render real.
        """
        del spec
        return self._real.render_arguments(_espec())


def _espec() -> RenderSpec:
    """Un `RenderSpec` minimo para pedir el argv al assembler.

    Returns:
        La especificacion con un clip y un destino temporales.
    """
    return RenderSpec(
        clip=Path("clip.mp4"),
        destination=Path("salida.mp4"),
        layout=Layout.SINGLE,
        width=1080,
        height=1920,
    )


def _encoder_de(argv: tuple[str, ...]) -> str:
    """Lee el codec de video que pide un argv.

    Args:
        argv: Argumentos de ffmpeg.

    Returns:
        El valor de ``-c:v``, o ``"ninguno"`` si no aparece.
    """
    for indice, argumento in enumerate(argv):
        if argumento == "-c:v" and indice + 1 < len(argv):
            return argv[indice + 1]
    return "ninguno"


@no_type_check
def _config_de_cli(output_format: str, entorno: EnvironmentReport) -> RenderConfig:
    """Config de render que la CLI construiria para ese entorno.

    Alcanza `_render_config_para` por atributo, no por import: es un nombre
    privado del modulo CLI y el repo no importa privados entre modulos.

    Args:
        output_format: El valor de `--format`, o cadena vacia para omitirlo.
        entorno: Deteccion del entorno con la que se construiria.

    Returns:
        La config resultante.
    """
    import kliptych.__main__ as cli

    funcion = cli._render_config_para
    return cast("RenderConfig", funcion(output_format or None, entorno))


def _reales(vistos: list[RenderConfig]) -> RenderConfig:
    """Unica config que vio el builder.

    Args:
        vistos: Las configs acumuladas por el espia.

    Returns:
        La primera, que es la de la corrida.
    """
    assert vistos, "el pipeline nunca construyo el assembler"
    return vistos[0]


def test_el_encoder_detectado_llega_a_la_config_de_render() -> None:
    """La config de render de la CLI ya no devuelve siempre el default."""
    assert _config_de_cli("", _ENTORNO_NVENC).nvenc_available is True
    assert _config_de_cli("", _ENTORNO_SIN_NVENC).nvenc_available is False
    vertical = _config_de_cli("9:16", _ENTORNO_NVENC)
    assert vertical.nvenc_available is True
    assert round(vertical.target_aspect, 4) == round(9.0 / 16.0, 4)


def test_el_assembler_honra_el_codec_de_la_config() -> None:
    """`FFmpegAssembler` ya no codifica `libx264` a mano."""
    con_nvenc = FFmpegAssembler(render=RenderConfig(nvenc_available=True))
    sin_nvenc = FFmpegAssembler(render=RenderConfig(nvenc_available=False))
    por_defecto = FFmpegAssembler()

    assert _encoder_de(con_nvenc.render_arguments(_espec())) == "h264_nvenc"
    assert _encoder_de(sin_nvenc.render_arguments(_espec())) == "libx264"
    assert _encoder_de(por_defecto.render_arguments(_espec())) == "libx264"


def test_pipeline_pasa_el_render_al_builder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`pipeline.py` reenvia la config al builder que construye.

    Sin esto, quitarle el render al constructor no cambiaria ningun resultado:
    la suite seguiria verde con NVENC sin usar.
    """
    _EspiaAssembler.vistos = []
    _ = (tmp_path / "clip.mp4").write_bytes(b"clip")
    _ = monkeypatch.setattr(
        "kliptych.pipeline.FFprobeProbe",
        _SinteticoProbe(FakeProbe(info=make_media(width=608, height=1080))),
    )
    _ = monkeypatch.setattr("kliptych.pipeline.FFmpegAssembler", _EspiaAssembler)
    _ = run_given_clips(
        model=_ModeloFijo(),
        settings=Settings.from_root(tmp_path),
        request=RunRequest(
            brief="cita del brief",
            destination=tmp_path / "package",
            environment=_ENTORNO_NVENC,
            model_version="static",
            prompt_version="p",
            caption_prompt_version="c",
            run_id="run-nvenc",
            render=RenderConfig(nvenc_available=True),
        ),
    )

    assert _reales(_EspiaAssembler.vistos).nvenc_available is True


def test_el_manifiesto_registra_el_encoder_pedido(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`render_arguments` deja de decir siempre `libx264`.

    Mide la INTENCION, no el resultado: si ffmpeg cae a libx264, eso lo anota
    `degradations` (canal de #97), no este campo. Lo que este campo garantiza es
    que el encoder pedido es el detectado y no el default.
    """
    _EspiaAssembler.vistos = []
    _ = (tmp_path / "clip.mp4").write_bytes(b"clip")
    _ = monkeypatch.setattr(
        "kliptych.pipeline.FFprobeProbe",
        _SinteticoProbe(FakeProbe(info=make_media(width=608, height=1080))),
    )
    _ = monkeypatch.setattr("kliptych.pipeline.FFmpegAssembler", _EspiaAssembler)
    resultado = run_given_clips(
        model=_ModeloFijo(),
        settings=Settings.from_root(tmp_path),
        request=RunRequest(
            brief="cita del brief",
            destination=tmp_path / "package",
            environment=_ENTORNO_NVENC,
            model_version="static",
            prompt_version="p",
            caption_prompt_version="c",
            run_id="run-nvenc",
            render=RenderConfig(nvenc_available=True),
        ),
    )

    crudo = cast(
        "dict[str, object]",
        json.loads(Path(resultado.manifest_path).read_text(encoding="utf-8")),
    )
    salidas = cast("list[object]", crudo["outputs"])
    argumentos = cast("tuple[str, ...]", cast("dict[str, object]", salidas[0])["render_arguments"])

    assert _encoder_de(argumentos) == "h264_nvenc", argumentos


def test_el_canal_de_caida_sigue_funcionando_con_nvenc_encendido() -> None:
    """Encender NVENC no apaga el guardian de #97.

    Si NVENC se pide y ffmpeg lo rechaza, la caida debe quedar registrada con
    ambos codecs. Un PR que enciende el encoder y nunca ejercita la caida no ha
    demostrado que el canal sigue vivo.
    """

    class _Falla:
        def __call__(self, argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
            del kwargs
            if "h264_nvenc" in argv:
                return subprocess.CompletedProcess(argv, 1, "", "No NVENC capable devices found")
            return subprocess.CompletedProcess(argv, 0, "", "")

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
    with recolectar() as degradaciones:
        _ = run_ffmpeg_with_fallback(
            argv,
            Invocation(
                render=RenderConfig(nvenc_available=True), error_cls=RuntimeError, runner=_Falla()
            ),
        )

    assert degradaciones.como_tupla() == ("encoder pedido h264_nvenc, usado libx264",)
