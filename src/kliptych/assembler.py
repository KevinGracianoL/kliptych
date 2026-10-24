"""Ensamblado de piezas ``given_clips`` con ffmpeg (argumentos, nunca shell).

El modo ``given_clips`` recibe clips ya cortados: el ensamblado los normaliza
a un lienzo vertical (píxeles cuadrados, SAR del clip respetado) y conserva el
audio propio del clip. El watermark opcional se superpone desde un asset local
durante todo el video, en la zona y tamaño del ``WatermarkConfig``. El
artefacto que sale de aquí es el que inspecciona el gate: nunca se valida
sobre los parámetros de entrada.

La publicación es atómica: ffmpeg escribe en un temporal hermano y solo un
render exitoso reemplaza el destino; un fallo deja intacto el artefacto previo.
"""

import contextlib
import logging
import math
import os
import subprocess
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Final, assert_never

from kliptych.contract import (
    Band,
    Layout,
    LayoutSpec,
    SplitScreenConfig,
    Watermark,
    WatermarkPosition,
)
from kliptych.encoding import (
    Invocation,
    RenderConfig,
    muted_audio_arguments,
    run_ffmpeg_with_fallback,
    video_encoder_arguments,
)
from kliptych.layout_geometry import build_layout_filter_chains

logger = logging.getLogger(__name__)

_DEFAULT_WIDTH = 1080
_DEFAULT_HEIGHT = 1920
_DEFAULT_TIMEOUT_S = 300.0
_SPLIT_AUDIO_PROBE_TIMEOUT_S = 30.0
_WATERMARK_MARGIN = 20
_STDERR_TAIL = 400
_DIMENSION_PARTS: Final[int] = 2
_DEFAULT_PROBED_DIMENSIONS: Final[tuple[int, int]] = (1920, 1080)


class AssembleError(Exception):
    """La pieza no se pudo ensamblar."""


def canvas_size(long_side: int, *, target_aspect: float) -> tuple[int, int]:
    """El lienzo que se emite para un lado largo y un aspecto objetivo dados.

    El lado CORTO sale de `_even(round(long_side / target_aspect))`, con la misma
    cuantizacion a entero par que usa `reframe.crop_size`, para que productor y
    juez compartan la misma desviacion en vez de tener cada uno la suya. Con el
    9:16 por defecto y lado largo 1920 da 1080x1920, que es lo que emitia el
    ensamblador cuando el lienzo eran constantes de modulo. Medido con 16:9 da
    1920x1080.

    Que sea una funcion y no una constante es lo que hace que `RunRequest.render`
    sea una DECLARACION de la corrida y no un adorno: si el aspecto objetivo lo
    cambia la corrida, el lienzo lo sigue.

    Args:
        long_side: El lado largo del lienzo, en pixeles.
        target_aspect: El aspecto objetivo de la corrida (ancho / alto).

    Returns:
        El par (ancho, alto) del lienzo, con el lado corto par y minimo 2.

    Raises:
        ValueError: Si `long_side` no es positivo o `target_aspect` no lo es.
    """
    if long_side <= 0:
        msg = f"lado largo de lienzo invalido: {long_side}"
        raise ValueError(msg)
    if target_aspect <= 0:
        msg = f"aspecto objetivo invalido: {target_aspect}"
        raise ValueError(msg)
    if target_aspect >= 1.0:
        # Paisaje: el ancho es el lado largo y el alto se deriva.
        return long_side, max(2, math.floor(round(long_side / target_aspect) / 2) * 2)
    # Vertical: el alto es el lado largo y el ancho se deriva.
    return max(2, math.floor(round(long_side * target_aspect) / 2) * 2), long_side


@dataclass(frozen=True, slots=True)
class RenderSpec:
    """Entradas inmutables de un render ``given_clips``.

    Agrupa la superficie de render (clip, destino, watermark, subtítulos y
    política de audio) en un único valor para que los ensambladores expongan
    una firma mínima sin perder explicitud. Con ``layout=SPLIT_SCREEN`` el
    lienzo lo define ``split_screen`` y los paneles salen de ``top_clip`` y
    ``bottom_clip`` (``clip`` se ignora pero se conserva por compatibilidad);
    el audio mezcla ambos paneles cuando los dos aportan pista, o conserva
    el único que la aporta (un panel con audio jamás queda silenciado).

    ``width`` y ``height`` son el lienzo que se EMITE, y los calcula
    `canvas_size` a partir del ``target_aspect`` de la corrida. El campo existe
    porque el lienzo es una entrada del render, no una constante del
    ensamblador: si el aspecto objetivo lo declarara la corrida, el lienzo tiene
    que seguirlo o el gate mediria una geometria que nadie pidio producir.
    """

    clip: Path
    destination: Path
    watermark: Path | None = None
    watermark_config: Watermark | None = None
    width: int = _DEFAULT_WIDTH
    height: int = _DEFAULT_HEIGHT
    subtitles: Path | None = None
    mute_audio: bool = False
    layout: Layout = Layout.SINGLE
    split_screen: SplitScreenConfig | None = None
    top_clip: Path | None = None
    bottom_clip: Path | None = None
    layout_spec: LayoutSpec | None = None
    band_clips: tuple[Path, ...] = ()


class FFmpegAssembler:
    """Ensambla clips entregados en piezas verticales usando ffmpeg."""

    def __init__(
        self,
        *,
        ffmpeg: str = "ffmpeg",
        timeout_s: float = _DEFAULT_TIMEOUT_S,
        ffprobe: str = "ffprobe",
        render: RenderConfig | None = None,
    ) -> None:
        """Configura el binario y el timeout del ensamblado.

        Args:
            ffmpeg: Nombre o ruta del binario ffmpeg.
            timeout_s: Timeout máximo del ensamblado, en segundos.
            ffprobe: Nombre o ruta del binario ffprobe para contar el audio
                de cada panel del split antes de elegir el mapa final.
            render: Config de render. Su `nvenc_available` decide el codec; sin
                ella vale el default de la clase (`libx264`), que es lo mismo que
                hacia el string que habia antes.
        """
        self._ffmpeg: str = ffmpeg
        self._timeout_s: float = timeout_s
        self._ffprobe: str = ffprobe
        self._render: RenderConfig = RenderConfig() if render is None else render

    def assemble(self, spec: RenderSpec) -> Path:
        """Ensambla un clip entregado en una pieza vertical para el gate.

        El render ocurre en un temporal hermano y se publica con un reemplazo
        atómico solo si ffmpeg termina con éxito; ante cualquier fallo el
        artefacto previo en ``destination`` queda intacto.

        Args:
            spec: Entradas inmutables del render (clip, destino, watermark,
                lienzo, subtítulos y política de audio).

        Returns:
            La ruta del artefacto ensamblado.

        Raises:
            AssembleError: Si las entradas no existen, las dimensiones son
                inválidas, el destino no se puede preparar, ffmpeg falla o
                expira.
        """
        if spec.layout is Layout.SPLIT_SCREEN:
            return self._assemble_split(spec)
        if spec.layout is not Layout.SINGLE or spec.layout_spec is not None:
            return self._assemble_layout_spec(spec)
        _require_file(spec.clip, what="clip")
        if spec.watermark is not None:
            _require_file(spec.watermark, what="watermark")
        if spec.subtitles is not None:
            _require_file(spec.subtitles, what="subtítulos")
        if spec.width <= 0 or spec.height <= 0:
            msg = f"dimensiones de lienzo inválidas: {spec.width}x{spec.height}"
            raise AssembleError(msg)
        try:
            spec.destination.parent.mkdir(parents=True, exist_ok=True)
        except OSError as error:
            msg = f"no se pudo preparar el directorio del destino {spec.destination}: {error}"
            raise AssembleError(msg) from error
        temporary = _temporary_path(spec.destination)
        argv = self._build_argv(
            clip=spec.clip,
            destination=temporary,
            watermark=spec.watermark,
            watermark_config=spec.watermark_config,
            width=spec.width,
            height=spec.height,
            subtitles=spec.subtitles,
            mute_audio=spec.mute_audio,
        )
        # Sin subtítulos el contrato de invocación no lleva `cwd`; con
        # subtítulos ffmpeg corre en el directorio de salida para que el
        # filtro `subtitles` acepte la ruta relativa del .ass en Windows.
        cwd = None if spec.subtitles is None else temporary.parent
        return self._run_and_publish(
            argv, temporary=temporary, destination=spec.destination, cwd=cwd
        )

    def _assemble_split(self, spec: RenderSpec) -> Path:
        """Ensambla dos paneles (video+video o video+imagen) en un lienzo 9:16.

        Cada panel se escala y recorta a su franja del lienzo del
        ``split_screen`` y ambas franjas se apilan con ``vstack``; una
        imagen estática entra como frame único y el framesync repite su
        último frame hasta el fin del panel más largo (sin ``-loop`` ni
        ``-shortest``: un bucle infinito bajo ``vstack`` nunca termina).
        La publicación es atómica igual que el modo simple.

        Args:
            spec: Entradas del render con ``layout=SPLIT_SCREEN``.

        Returns:
            La ruta del artefacto ensamblado.

        Raises:
            AssembleError: Si falta la configuración, los paneles o los
                archivos no existen, o ffmpeg falla o expira.
        """
        split = spec.split_screen
        if split is None:
            msg = "el layout split_screen exige split_screen con la geometría de los paneles"
            raise AssembleError(msg)
        if spec.top_clip is None or spec.bottom_clip is None:
            msg = "el layout split_screen exige los paneles top_clip y bottom_clip"
            raise AssembleError(msg)
        _require_file(spec.top_clip, what="panel superior")
        _require_file(spec.bottom_clip, what="panel inferior")
        if spec.watermark is not None:
            _require_file(spec.watermark, what="watermark")
        if spec.subtitles is not None:
            _require_file(spec.subtitles, what="subtítulos")
        try:
            spec.destination.parent.mkdir(parents=True, exist_ok=True)
        except OSError as error:
            msg = f"no se pudo preparar el directorio del destino {spec.destination}: {error}"
            raise AssembleError(msg) from error
        temporary = _temporary_path(spec.destination)
        argv = self._build_split_argv(
            top=spec.top_clip,
            bottom=spec.bottom_clip,
            split=split,
            destination=temporary,
            watermark=spec.watermark,
            watermark_config=spec.watermark_config,
            subtitles=spec.subtitles,
            mute_audio=spec.mute_audio,
        )
        cwd = None if spec.subtitles is None else temporary.parent
        return self._run_and_publish(
            argv, temporary=temporary, destination=spec.destination, cwd=cwd
        )

    def _assemble_layout_spec(self, spec: RenderSpec) -> Path:
        """Ensambla video multifranga (Dividido, Trío, Spotlight, etc.) con vstack.

        Args:
            spec: Entradas del render con layout multifranga.

        Returns:
            La ruta del artefacto final publicado.

        Raises:
            AssembleError: Si las dimensiones son inválidas, los archivos no
                existen o ffmpeg falla al procesar el ensamble.
        """
        if spec.width <= 0 or spec.height <= 0:
            msg = f"dimensiones de lienzo inválidas: {spec.width}x{spec.height}"
            raise AssembleError(msg)
        _require_file(spec.clip, what="clip")
        for band_clip in spec.band_clips:
            _require_file(band_clip, what="band clip")
        if spec.watermark is not None:
            _require_file(spec.watermark, what="watermark")
        if spec.subtitles is not None:
            _require_file(spec.subtitles, what="subtítulos")
        try:
            spec.destination.parent.mkdir(parents=True, exist_ok=True)
        except OSError as error:
            msg = f"no se pudo preparar el directorio del destino {spec.destination}: {error}"
            raise AssembleError(msg) from error
        temporary = _temporary_path(spec.destination)
        argv = self._build_layout_spec_argv(spec, destination=temporary)
        cwd = None if spec.subtitles is None else temporary.parent
        return self._run_and_publish(
            argv, temporary=temporary, destination=spec.destination, cwd=cwd
        )

    def _run_and_publish(
        self,
        argv: list[str],
        *,
        temporary: Path,
        destination: Path,
        cwd: Path | None,
    ) -> Path:
        """Ejecuta ffmpeg sobre un temporal hermano y lo publica en atómico.

        Args:
            argv: Argumentos de ffmpeg, con el temporal como destino final.
            temporary: Ruta del temporal hermano donde escribe ffmpeg.
            destination: Ruta del artefacto final, reemplazada solo en éxito.
            cwd: Directorio de trabajo de ffmpeg, o ``None`` para heredar el
                actual (sin subtítulos el contrato de invocación no lleva
                ``cwd``).

        Returns:
            La ruta del artefacto publicado.

        Raises:
            AssembleError: Si ffmpeg falla en ambos encoders o el artefacto no se
                puede publicar (el destino previo queda intacto).

        El reintento con libx264 va por `run_ffmpeg_with_fallback`, el mismo
        camino que usa el resto del pipeline: sin él, un NVENC que falla en caliente
        no deja artefacto ni degradación, y la corrida muere con 255 en máquinas
        donde el encoder está listado pero no funciona.
        """
        # El runner se resuelve aquí, en el modulo del assembler, para que los
        # dobles de los tests sigan viendo su propio `subprocess.run`.
        run_ffmpeg_with_fallback(
            argv,
            Invocation(
                render=self._render,
                temporary=temporary,
                error_cls=AssembleError,
                runner=subprocess.run,
                cwd=cwd,
                timeout_s=self._timeout_s,
            ),
        )
        try:
            _ = temporary.replace(destination)
        except OSError as error:
            _remove_quietly(temporary)
            msg = f"no se pudo publicar el artefacto en {destination}: {error}"
            raise AssembleError(msg) from error
        return destination

    def cut_exact(
        self,
        *,
        source: Path,
        destination: Path,
        start_s: float,
        duration_s: float,
    ) -> Path:
        """Corta un clip con precisión de frame usando libx264/aac y reseteando PTS a 0.0s.

        Args:
            source: Video descargado (con margen).
            destination: Ruta del artefacto cortado con precisión.
            start_s: Segundo de inicio relativo a la fuente descargada.
            duration_s: Duración exacta en segundos (end - start).

        Returns:
            La ruta del artefacto cortado con precisión.

        Raises:
            AssembleError: Si la fuente no existe, el intervalo es inválido o ffmpeg falla.
        """
        _require_file(source, what="source")
        if start_s < 0 or duration_s <= 0:
            msg = f"intervalo de corte inválido: start={start_s}, duration={duration_s}"
            raise AssembleError(msg)
        try:
            destination.parent.mkdir(parents=True, exist_ok=True)
        except OSError as error:
            msg = f"no se pudo preparar el directorio del destino {destination}: {error}"
            raise AssembleError(msg) from error
        temporary = _temporary_path(destination)
        argv = [
            self._ffmpeg,
            "-hide_banner",
            "-nostdin",
            "-v",
            "error",
            "-y",
            "-ss",
            f"{start_s:.3f}",
            "-t",
            f"{duration_s:.3f}",
            "-i",
            str(source),
            "-vf",
            "setpts=PTS-STARTPTS",
            "-map",
            "0:v:0",
            "-map",
            "0:a?",
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "20",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-movflags",
            "+faststart",
            str(temporary),
        ]
        try:
            completed = subprocess.run(
                argv,
                capture_output=True,
                text=True,
                timeout=self._timeout_s,
                check=False,
            )
        except FileNotFoundError as error:
            _remove_quietly(temporary)
            msg = f"ffmpeg no está disponible: {self._ffmpeg}"
            raise AssembleError(msg) from error
        except subprocess.TimeoutExpired as error:
            _remove_quietly(temporary)
            msg = f"ffmpeg excedió el timeout de {self._timeout_s} s"
            raise AssembleError(msg) from error
        except OSError as error:
            _remove_quietly(temporary)
            msg = f"no se pudo ejecutar ffmpeg ({self._ffmpeg}): {error}"
            raise AssembleError(msg) from error
        if completed.returncode != 0:
            _remove_quietly(temporary)
            msg = f"ffmpeg falló con código {completed.returncode}: {_tail(completed.stderr)}"
            raise AssembleError(msg)
        try:
            _ = temporary.replace(destination)
        except OSError as error:
            _remove_quietly(temporary)
            msg = f"no se pudo publicar el artefacto en {destination}: {error}"
            raise AssembleError(msg) from error
        return destination

    def render_arguments(self, spec: RenderSpec) -> tuple[str, ...]:
        """Devuelve el argv de ffmpeg que se usaría para este ensamblado.

        Es la receta de render que se registra en el manifiesto; el ensamblado
        real escribe primero en un temporal y publica al final.

        Args:
            spec: Entradas inmutables del render.

        Returns:
            El argv completo de ffmpeg, como tupla inmutable.

        Raises:
            AssembleError: Si el layout split_screen no trae su
                configuración ni sus paneles.
        """
        if spec.layout is Layout.SPLIT_SCREEN:
            if spec.split_screen is None:
                msg = "el layout split_screen exige split_screen con la geometría de los paneles"
                raise AssembleError(msg)
            if spec.top_clip is None or spec.bottom_clip is None:
                msg = "el layout split_screen exige los paneles top_clip y bottom_clip"
                raise AssembleError(msg)
            return tuple(
                self._build_split_argv(
                    top=spec.top_clip,
                    bottom=spec.bottom_clip,
                    split=spec.split_screen,
                    destination=spec.destination,
                    watermark=spec.watermark,
                    watermark_config=spec.watermark_config,
                    subtitles=spec.subtitles,
                    mute_audio=spec.mute_audio,
                )
            )
        if spec.layout is not Layout.SINGLE or spec.layout_spec is not None:
            return tuple(self._build_layout_spec_argv(spec))
        return tuple(
            self._build_argv(
                clip=spec.clip,
                destination=spec.destination,
                watermark=spec.watermark,
                watermark_config=spec.watermark_config,
                width=spec.width,
                height=spec.height,
                subtitles=spec.subtitles,
                mute_audio=spec.mute_audio,
            )
        )

    def _build_argv(
        self,
        *,
        clip: Path,
        destination: Path,
        watermark: Path | None,
        watermark_config: Watermark | None,
        width: int,
        height: int,
        subtitles: Path | None,
        mute_audio: bool,
    ) -> list[str]:
        square = "scale=trunc(iw*sar/2)*2:ih,setsar=1"
        scale = f"scale={width}:{height}:force_original_aspect_ratio=decrease"
        pad = f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:color=black"
        base = f"{square},{scale},{pad},setsar=1"
        if subtitles is None:
            clip_arg = str(clip)
            watermark_arg = str(watermark) if watermark is not None else None
            destination_arg = str(destination)
            subtitles_suffix = ""
        else:
            # ffmpeg corre con cwd en el directorio de salida: las entradas
            # viajan absolutas y el .ass en ruta relativa, que es lo único
            # que el parser del filtro `subtitles` acepta en Windows.
            clip_arg = str(clip.resolve())
            watermark_arg = str(watermark.resolve()) if watermark is not None else None
            destination_arg = str(destination.resolve())
            subtitles_suffix = (
                f",subtitles=filename='{_subtitles_filter_value(subtitles, destination)}'"
            )
        argv = [
            self._ffmpeg,
            "-hide_banner",
            "-nostdin",
            "-v",
            "error",
            "-y",
            "-i",
            clip_arg,
        ]
        if watermark is None:
            argv += ["-vf", f"{base}{subtitles_suffix}", "-map", "0:v:0", "-map", "0:a?"]
        else:
            config = (
                watermark_config
                if watermark_config is not None
                else Watermark(required=True, visible_full_video=True)
            )
            graph = _watermark_filter(base, config, canvas_width=width)
            video_label = "[v]"
            if subtitles is not None:
                graph += (
                    f";{video_label}subtitles=filename='"
                    f"{_subtitles_filter_value(subtitles, destination)}'[vout]"
                )
                video_label = "[vout]"
            argv += [
                "-i",
                str(watermark_arg),
                "-filter_complex",
                graph,
                "-map",
                video_label,
                "-map",
                "0:a?",
            ]
        if mute_audio:
            argv += list(muted_audio_arguments())
        argv += list(video_encoder_arguments(nvenc_available=self._render.nvenc_available))
        argv += [
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-movflags",
            "+faststart",
            destination_arg,
        ]
        return argv

    def _build_split_argv(
        self,
        *,
        top: Path,
        bottom: Path,
        split: SplitScreenConfig,
        destination: Path,
        watermark: Path | None,
        watermark_config: Watermark | None,
        subtitles: Path | None,
        mute_audio: bool,
    ) -> list[str]:
        """Construye el argv que apila dos paneles en un lienzo 9:16.

        Cada panel se escala con ``scale`` y se recorta con ``crop`` a su
        franja del lienzo; con ``gap`` el panel superior se extiende con
        ``pad`` negro y ambas franjas se apilan con ``vstack``. Una imagen
        estática entra como frame único y el framesync repite su último
        frame hasta el fin del panel más largo (sin ``-loop`` ni
        ``-shortest``: un bucle infinito bajo ``vstack`` nunca termina).
        El audio se elige por panel con ffprobe: mezcla ``amix`` cuando los
        dos aportan pista, o el único mapa con audio en caso contrario (un
        panel con audio jamás queda silenciado).

        Args:
            top: Panel superior (video o imagen estática).
            bottom: Panel inferior (video o imagen estática).
            split: Geometría del lienzo y los paneles.
            destination: Temporal hermano donde escribe ffmpeg.
            watermark: PNG opcional superpuesto al lienzo ya apilado.
            watermark_config: Posición y tamaño del watermark.
            subtitles: Archivo ``.ass`` opcional a quemar al final.
            mute_audio: Si es True, silencia la pista sin eliminarla.

        Returns:
            El argv completo de ffmpeg.
        """
        top_h, bottom_h = split.panel_heights()
        width = split.width
        top_chain = (
            f"scale={width}:{top_h}:force_original_aspect_ratio=increase,"
            f"crop={width}:{top_h},setsar=1"
        )
        if split.gap > 0:
            top_chain += f",pad={width}:{top_h + split.gap}:0:0:color=black"
        bottom_chain = (
            f"scale={width}:{bottom_h}:force_original_aspect_ratio=increase,"
            f"crop={width}:{bottom_h},setsar=1"
        )
        core_out = "[splitv]" if watermark is not None else "[v]"
        graph = (
            f"[0:v]{top_chain}[top];"
            f"[1:v]{bottom_chain}[bottom];"
            f"[top][bottom]vstack=inputs=2{core_out}"
        )
        if subtitles is None:
            top_arg = str(top)
            bottom_arg = str(bottom)
            watermark_arg = str(watermark) if watermark is not None else None
            destination_arg = str(destination)
        else:
            top_arg = str(top.resolve())
            bottom_arg = str(bottom.resolve())
            watermark_arg = str(watermark.resolve()) if watermark is not None else None
            destination_arg = str(destination.resolve())
        argv = [
            self._ffmpeg,
            "-hide_banner",
            "-nostdin",
            "-v",
            "error",
            "-y",
            "-i",
            top_arg,
            "-i",
            bottom_arg,
        ]
        if watermark is not None and watermark_arg is not None:
            argv += ["-i", watermark_arg]
        if watermark is None:
            video_label = "[v]"
        else:
            config = (
                watermark_config
                if watermark_config is not None
                else Watermark(required=True, visible_full_video=True)
            )
            graph += ";"
            graph += _watermark_overlay(
                wm_input="[2:v]",
                main_label="[splitv]",
                out_label="[v]",
                config=config,
                canvas_width=width,
            )
            video_label = "[v]"
        if subtitles is not None:
            graph += (
                f";{video_label}subtitles=filename='"
                f"{_subtitles_filter_value(subtitles, destination)}'[vout]"
            )
            video_label = "[vout]"
        argv += self._split_audio_tail(
            top=top,
            bottom=bottom,
            graph=graph,
            video_label=video_label,
            mute_audio=mute_audio,
        )
        argv += list(video_encoder_arguments(nvenc_available=self._render.nvenc_available))
        argv += [
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-movflags",
            "+faststart",
            destination_arg,
        ]
        return argv

    def _split_audio_tail(
        self,
        *,
        top: Path,
        bottom: Path,
        graph: str,
        video_label: str,
        mute_audio: bool,
    ) -> list[str]:
        """Arma la cola del argv split: filtro de audio, mapas y silenciado.

        Cuenta el audio real de cada panel con ffprobe y delega la elección
        en ``_split_audio_parts`` (mezcla ``amix`` con ambos, o el único mapa
        con audio): un panel con audio jamás queda silenciado por mapear
        solo el otro.

        Args:
            top: Panel superior (video o imagen estática).
            bottom: Panel inferior (video o imagen estática).
            graph: Grafo de video ya cerrado (vstack, watermark, subtítulos).
            video_label: Etiqueta del video final dentro del grafo.
            mute_audio: Si es True, silencia la pista sin eliminarla.

        Returns:
            La cola del argv desde ``-filter_complex`` hasta el mapa de
            audio, con el silenciado ya aplicado cuando corresponde.
        """
        top_has_audio = _split_panel_has_audio(
            top, ffprobe=self._ffprobe, timeout_s=_SPLIT_AUDIO_PROBE_TIMEOUT_S
        )
        bottom_has_audio = _split_panel_has_audio(
            bottom, ffprobe=self._ffprobe, timeout_s=_SPLIT_AUDIO_PROBE_TIMEOUT_S
        )
        audio_filter, audio_maps, mute_embedded = _split_audio_parts(
            top_has_audio=top_has_audio,
            bottom_has_audio=bottom_has_audio,
            mute_audio=mute_audio,
        )
        tail = ["-filter_complex", graph + audio_filter, "-map", video_label, *audio_maps]
        if mute_audio and not mute_embedded:
            tail += list(muted_audio_arguments())
        return tail

    def _build_layout_spec_argv(
        self,
        spec: RenderSpec,
        *,
        destination: Path | None = None,
    ) -> list[str]:
        target = destination if destination is not None else spec.destination
        layout_spec = spec.layout_spec or _default_layout_spec_for(spec.layout)
        band_heights = layout_spec.band_heights(spec.height)
        bands = layout_spec.bands or tuple(Band() for _ in range(len(band_heights)))
        n = len(bands)
        clips = [spec.band_clips[i] if i < len(spec.band_clips) else spec.clip for i in range(n)]
        argv, target_str = self._layout_spec_base_argv(clips, spec, target)
        gap = layout_spec.gap if n > 1 else 0
        source_dimensions = tuple(
            _probe_clip_dimensions(clip, ffprobe=self._ffprobe, timeout_s=self._timeout_s)
            for clip in clips
        )
        filter_chains = list(
            build_layout_filter_chains(
                layout_spec=LayoutSpec(layout=layout_spec.layout, bands=bands, gap=gap),
                canvas_width=spec.width,
                canvas_height=spec.height,
                source_dimensions=source_dimensions,
            )
        )
        video_label = self._layout_spec_overlay_tail(filter_chains, argv, spec, n, target)
        argv += ["-filter_complex", ";".join(filter_chains), "-map", video_label, "-map", "0:a?"]
        if spec.mute_audio:
            argv += list(muted_audio_arguments())
        argv += list(video_encoder_arguments(nvenc_available=self._render.nvenc_available))
        argv += ["-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", target_str]
        return argv

    def _layout_spec_base_argv(
        self, clips: list[Path], spec: RenderSpec, target: Path
    ) -> tuple[list[str], str]:
        if spec.subtitles is None:
            clip_args = [str(c) for c in clips]
            target_str = str(target)
        else:
            clip_args = [str(c.resolve()) for c in clips]
            target_str = str(target.resolve())
        argv = [self._ffmpeg, "-hide_banner", "-nostdin", "-v", "error", "-y"]
        for c_arg in clip_args:
            argv += ["-i", c_arg]
        return argv, target_str

    @staticmethod
    def _layout_spec_overlay_tail(
        filter_chains: list[str],
        argv: list[str],
        spec: RenderSpec,
        n: int,
        target: Path,
    ) -> str:
        if n > 1:
            stack_inputs = "".join(f"[b{i}]" for i in range(n))
            filter_chains.append(f"{stack_inputs}vstack=inputs={n}[v_stacked]")
            current_v = "[v_stacked]"
        else:
            filter_chains.append("[b0]null[v_stacked]")
            current_v = "[v_stacked]"
        if spec.watermark is not None:
            wm_path = (
                str(spec.watermark.resolve()) if spec.subtitles is not None else str(spec.watermark)
            )
            argv += ["-i", wm_path]
            config = spec.watermark_config or Watermark(required=True, visible_full_video=True)
            filter_chains.append(
                _watermark_overlay(
                    wm_input=f"[{n}:v]",
                    main_label=current_v,
                    out_label="[v_wm]",
                    config=config,
                    canvas_width=spec.width,
                )
            )
            current_v = "[v_wm]"
        if spec.subtitles is not None:
            sub_val = _subtitles_filter_value(spec.subtitles, target)
            filter_chains.append(f"{current_v}subtitles=filename='{sub_val}'[vout]")
            current_v = "[vout]"
        return current_v


def _default_layout_spec_for(layout: Layout) -> LayoutSpec:
    if layout in {Layout.DIVIDIDO, Layout.SPOTLIGHT}:
        return LayoutSpec(layout=layout, bands=(Band(), Band()))
    if layout is Layout.TRIO:
        return LayoutSpec(layout=layout, bands=(Band(), Band(), Band()))
    return LayoutSpec(layout=layout, bands=(Band(),))


def _probe_clip_dimensions(clip: Path, *, ffprobe: str, timeout_s: float) -> tuple[int, int]:
    """Lee las dimensiones del video con ffprobe o devuelve (1920, 1080) por defecto.

    Args:
        clip: Ruta del archivo de video a inspeccionar.
        ffprobe: Binario de ffprobe.
        timeout_s: Timeout de la invocación.

    Returns:
        Tupla (ancho, alto) en píxeles.
    """
    argv = [
        ffprobe,
        "-hide_banner",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height",
        "-of",
        "csv=s=x:p=0",
        str(clip),
    ]
    try:
        completed = subprocess.run(
            argv,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as error:
        logger.warning(
            "No se pudieron leer las dimensiones de %s (%s); se usa %dx%d para el recorte",
            clip,
            error,
            *_DEFAULT_PROBED_DIMENSIONS,
        )
        return _DEFAULT_PROBED_DIMENSIONS
    if completed.returncode != 0 or not completed.stdout.strip():
        logger.warning(
            "ffprobe no reportó dimensiones de %s (rc=%s); se usa %dx%d para el recorte",
            clip,
            completed.returncode,
            *_DEFAULT_PROBED_DIMENSIONS,
        )
        return _DEFAULT_PROBED_DIMENSIONS
    first_line = completed.stdout.strip().splitlines()[0]
    parts = first_line.split("x")
    if len(parts) == _DIMENSION_PARTS and parts[0].isdigit() and parts[1].isdigit():
        w, h = int(parts[0]), int(parts[1])
        if w > 0 and h > 0:
            return w, h
    logger.warning(
        "Salida de ffprobe no interpretable para %s: %r; se usa %dx%d para el recorte",
        clip,
        first_line,
        *_DEFAULT_PROBED_DIMENSIONS,
    )
    return _DEFAULT_PROBED_DIMENSIONS


def _split_panel_has_audio(path: Path, *, ffprobe: str, timeout_s: float) -> bool:
    """Indica si un panel del split aporta pista de audio (fail-closed a True).

    Cuenta las pistas de audio con ffprobe (lista de argumentos, sin shell):
    una imagen estática no reporta ninguna y un video con audio al menos una.
    Ante cualquier fallo (binario ausente, timeout o error de inspección) se
    asume que hay audio: antes una mezcla que quizá falle en ffmpeg que un
    MP4 silenciado cuando sí había audio que conservar.

    Args:
        path: Panel superior o inferior del split.
        ffprobe: Binario ffprobe para contar las pistas de audio.
        timeout_s: Timeout máximo de la inspección, en segundos.

    Returns:
        True si el panel tiene al menos una pista de audio o no se pudo
        determinar; False solo cuando ffprobe confirma que no hay ninguna.
    """
    # ffprobe no acepta `-nostdin` (falla con "Option not found"): solo
    # `-hide_banner` y `-v error` lo silencian sin romper la inspección.
    argv = [
        ffprobe,
        "-hide_banner",
        "-v",
        "error",
        "-select_streams",
        "a",
        "-show_entries",
        "stream=index",
        "-of",
        "csv=p=0",
        str(path),
    ]
    try:
        completed = subprocess.run(
            argv,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return True
    if completed.returncode != 0:
        return True
    return any(line.strip() for line in completed.stdout.splitlines())


def _split_audio_parts(
    *, top_has_audio: bool, bottom_has_audio: bool, mute_audio: bool
) -> tuple[str, list[str], bool]:
    """Elige el filtro y el mapa de audio del split según las pistas reales.

    Con audio en ambos paneles los mezcla con ``amix`` en una única pista
    (normalizados a estéreo antes de mezclar); con audio en un solo panel
    mapea ese panel para no silenciarlo; sin audio en ninguno conserva el
    mapa opcional histórico (el MP4 queda sin pista, como antes).

    Args:
        top_has_audio: Si el panel superior aporta pista de audio.
        bottom_has_audio: Si el panel inferior aporta pista de audio.
        mute_audio: Si el render debe silenciar la pista sin eliminarla.

    Returns:
        El tramo de filtro a anexar al grafo, los argumentos ``-map`` de
        audio y si el silenciado ya quedó incrustado en la mezcla (en ese
        caso el llamador no debe agregar ``-af volume=0``, que chocaría con
        el ``filter_complex`` de audio).
    """
    if top_has_audio and bottom_has_audio:
        silence = ",volume=0" if mute_audio else ""
        audio_filter = (
            ";[0:a]aformat=sample_fmts=fltp:channel_layouts=stereo[topa]"
            ";[1:a]aformat=sample_fmts=fltp:channel_layouts=stereo[bottoma]"
            f";[topa][bottoma]amix=inputs=2:duration=longest{silence}[a]"
        )
        return (audio_filter, ["-map", "[a]"], mute_audio)
    if bottom_has_audio:
        return ("", ["-map", "1:a"], False)
    if top_has_audio:
        return ("", ["-map", "0:a"], False)
    return ("", ["-map", "0:a?"], False)


def _subtitles_filter_value(subtitles: Path, output: Path) -> str:
    """Devuelve la ruta del .ass para el filtro ``subtitles`` de ffmpeg.

    ffmpeg corre con ``cwd`` en el directorio de salida, así que una ruta
    relativa funciona en Windows donde una absoluta rompe el parser del
    filtro (misma estrategia que el quemado de ``SubtitleRenderer``).

    Args:
        subtitles: Ruta del archivo ``.ass`` a quemar.
        output: Ruta del artefacto de salida (su padre es el cwd de ffmpeg).

    Returns:
        La ruta relativa en formato POSIX, o la absoluta con ``:`` escapado
        cuando no se puede relativizar.
    """
    try:
        rel = os.path.relpath(subtitles, start=output.parent)
    except (ValueError, OSError):
        return subtitles.as_posix().replace(":", "\\:")
    return Path(rel).as_posix()


def _watermark_filter(base: str, config: Watermark, *, canvas_width: int) -> str:
    """Construye el ``filter_complex`` que escala y superpone el watermark.

    El PNG viaja como segunda entrada explícita (``-i``), nunca como
    ``movie=``: así las rutas Windows con ``:`` no rompen el parser del
    grafo. El PNG se escala con ``scale`` de una sola entrada al
    ``scale_ratio`` del ancho del lienzo (altura con ``-2`` para preservar
    su aspecto en píxeles pares) y se superpone con ``overlay`` en la zona
    de ``config.position`` con el margen de seguridad. Sin filtros de doble
    entrada el framesync no puede truncar el video (``overlay`` repite el
    frame único del PNG hasta el fin del lienzo de forma determinista).

    Args:
        base: Cadena de filtros que normaliza el clip al lienzo vertical.
        config: Posición, tamaño y opacidad del watermark.
        canvas_width: Ancho del lienzo vertical, en píxeles.

    Returns:
        El grafo completo, con el video final en la etiqueta ``[v]``.
    """
    overlay = _watermark_overlay(
        wm_input="[1:v]",
        main_label="[base]",
        out_label="[v]",
        config=config,
        canvas_width=canvas_width,
    )
    return f"[0:v]{base}[base];{overlay}"


def _watermark_overlay(
    *,
    wm_input: str,
    main_label: str,
    out_label: str,
    config: Watermark,
    canvas_width: int,
) -> str:
    """Escala el PNG del watermark y lo superpone sobre el lienzo principal.

    Args:
        wm_input: Etiqueta de la entrada del PNG (p. ej. ``[1:v]``).
        main_label: Etiqueta del video principal ya normalizado.
        out_label: Etiqueta del video final con el watermark.
        config: Posición, tamaño y opacidad del watermark.
        canvas_width: Ancho del lienzo vertical, en píxeles.

    Returns:
        El tramo del grafo que produce ``out_label`` desde ``main_label``.
    """
    x, y = _overlay_xy(config.position)
    target_w = int(math.trunc(canvas_width * config.scale_ratio / 2) * 2)
    scale = f"{wm_input}format=rgba,scale={target_w}:-2[wm]"
    if config.opacity >= 1.0:
        return f"{scale};{main_label}[wm]overlay={x}:{y}{out_label}"
    return (
        f"{scale};"
        f"[wm]colorchannelmixer=aa={config.opacity}[wmf];"
        f"{main_label}[wmf]overlay={x}:{y}{out_label}"
    )


def _overlay_xy(position: WatermarkPosition) -> tuple[str, str]:
    """Devuelve las expresiones ``(x, y)`` del ``overlay`` para una posición.

    Args:
        position: Zona del lienzo exigida por el contrato.

    Returns:
        Las expresiones de ffmpeg para la esquina superior izquierda del
        watermark, con el margen de seguridad desde los bordes.
    """
    margin = _WATERMARK_MARGIN
    if position is WatermarkPosition.TOP_LEFT:
        x, y = f"{margin}", f"{margin}"
    elif position is WatermarkPosition.TOP_RIGHT:
        x, y = f"W-w-{margin}", f"{margin}"
    elif position is WatermarkPosition.BOTTOM_LEFT:
        x, y = f"{margin}", f"H-h-{margin}"
    elif position is WatermarkPosition.BOTTOM_RIGHT:
        x, y = f"W-w-{margin}", f"H-h-{margin}"
    elif position is WatermarkPosition.CENTER:
        x, y = "(W-w)/2", "(H-h)/2"
    elif position is WatermarkPosition.CENTER_TOP:
        x, y = "(W-w)/2", f"{margin}"
    elif position is WatermarkPosition.CENTER_BOTTOM:
        x, y = "(W-w)/2", f"H-h-{margin}"
    else:
        assert_never(position)
    return (x, y)


def _temporary_path(destination: Path) -> Path:
    return destination.with_name(f".{destination.stem}.part-{uuid.uuid4().hex}{destination.suffix}")


def _remove_quietly(path: Path) -> None:
    # Limpieza best-effort del temporal; el destino final nunca se toca.
    with contextlib.suppress(OSError):
        path.unlink(missing_ok=True)


def _require_file(path: Path, *, what: str) -> None:
    if not path.is_file():
        msg = f"el {what} no existe: {path}"
        raise AssembleError(msg)


def _tail(text: str) -> str:
    stripped = text.strip()
    return stripped[-_STDERR_TAIL:]
