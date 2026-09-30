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
import math
import os
import subprocess
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import assert_never

from kliptych.contract import Layout, SplitScreenConfig, Watermark, WatermarkPosition
from kliptych.encoding import muted_audio_arguments

_DEFAULT_WIDTH = 1080
_DEFAULT_HEIGHT = 1920
_DEFAULT_TIMEOUT_S = 300.0
_WATERMARK_MARGIN = 20
_STDERR_TAIL = 400


class AssembleError(Exception):
    """La pieza no se pudo ensamblar."""


@dataclass(frozen=True, slots=True)
class RenderSpec:
    """Entradas inmutables de un render ``given_clips``.

    Agrupa la superficie de render (clip, destino, watermark, subtítulos y
    política de audio) en un único valor para que los ensambladores expongan
    una firma mínima sin perder explicitud. Con ``layout=SPLIT_SCREEN`` el
    lienzo lo define ``split_screen`` y los paneles salen de ``top_clip`` y
    ``bottom_clip`` (``clip`` se ignora pero se conserva por compatibilidad);
    el audio se toma del panel superior.
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


class FFmpegAssembler:
    """Ensambla clips entregados en piezas verticales usando ffmpeg."""

    def __init__(
        self,
        *,
        ffmpeg: str = "ffmpeg",
        timeout_s: float = _DEFAULT_TIMEOUT_S,
    ) -> None:
        """Configura el binario y el timeout del ensamblado.

        Args:
            ffmpeg: Nombre o ruta del binario ffmpeg.
            timeout_s: Timeout máximo del ensamblado, en segundos.
        """
        self._ffmpeg: str = ffmpeg
        self._timeout_s: float = timeout_s

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
        if spec.layout is not Layout.SINGLE:
            assert_never(spec.layout)
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
        imagen estática entra en bucle y el render se acota al panel más
        corto con ``-shortest``. La publicación es atómica igual que el
        modo simple.

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
            AssembleError: Si ffmpeg no está disponible, expira, falla o el
                artefacto no se puede publicar (el destino previo queda
                intacto en todos los casos).
        """
        try:
            completed = (
                subprocess.run(
                    argv,
                    capture_output=True,
                    text=True,
                    timeout=self._timeout_s,
                    check=False,
                )
                if cwd is None
                else subprocess.run(
                    argv,
                    capture_output=True,
                    text=True,
                    timeout=self._timeout_s,
                    check=False,
                    cwd=cwd,
                )
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
        if spec.layout is not Layout.SINGLE:
            assert_never(spec.layout)
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
        argv += [
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
        El audio se toma del panel superior.

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
        argv += ["-filter_complex", graph, "-map", video_label, "-map", "0:a?"]
        if mute_audio:
            argv += list(muted_audio_arguments())
        argv += [
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
            destination_arg,
        ]
        return argv


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
