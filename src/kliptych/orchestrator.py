"""Orquestador end-to-end del modo ``long_video`` (fase C5).

Encadena descarga -> transcripción -> detección de momentos -> selección LLM ->
reframe 9:16 -> subtítulos -> vídeo vertical final. No implementa ninguna de las
etapas: compone los módulos de C2-C4 y traduce cualquier fallo de una etapa a
``PipelineError`` conservando la causa.

El modo Repost/UGC (``repost_mode``) reutiliza el mismo encadenado pero salta las
capas de inteligencia: no transcribe, no detecta momentos ni llama al LLM; el
vídeo completo es el segmento y solo se reframea si no es 9:16. El audio externo
y el passthrough final siguen publicándose de forma atómica.

Los artefactos finales (``source.mp4`` y ``final.mp4``) se publican de forma
atómica: ffmpeg escribe en un temporal hermano y solo un render exitoso
reemplaza el destino. Un registro de limpieza elimina todos los temporales
(``.part`` de la descarga, clips intermedios, ``.ass`` y reframes) tanto en el
camino feliz como ante error.
"""

import shutil
import subprocess
import uuid
from collections.abc import Generator, Mapping, Sequence
from contextlib import contextmanager, suppress
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Protocol

from kliptych.contract import Contract, Segment
from kliptych.download import MediaDownloader
from kliptych.encoding import (
    RenderConfig,
    audio_and_container_arguments,
    audio_injection_arguments,
    video_encoder_arguments,
)
from kliptych.moments import FFmpegMomentDetector, Moment, MomentDetector
from kliptych.reframe import FFmpegReframer, MediaPipeFaceDetector, ReframeResult
from kliptych.segment import LLMSegmentSelector, SegmentSelection, SegmentSelector
from kliptych.subtitles import SubtitleRenderer
from kliptych.transcribe import (
    FasterWhisperTranscriber,
    Transcriber,
    Transcript,
    Word,
)

_STDERR_TAIL = 400
_SOURCE_NAME = "source.mp4"
_FINAL_NAME = "final.mp4"
_FFPROBE = "ffprobe"
_TARGET_ASPECT = 9.0 / 16.0
_ASPECT_TOLERANCE = 0.05


class PipelineError(Exception):
    """El pipeline end-to-end no se pudo completar."""


@dataclass(frozen=True, slots=True)
class PipelineResult:
    """Resultado del pipeline long_video.

    Attributes:
        source: Vídeo fuente descargado, conservado en ``output_dir``.
        transcript: Transcripción word-level del vídeo; ``None`` en modo repost,
            que omite la transcripción por completo.
        moments: Momentos candidatos detectados, ordenados por puntuación;
            vacío en modo repost, que no analiza el vídeo.
        selection: Segmentos elegidos; en modo repost es un único segmento que
            cubre el vídeo completo.
        reframe: Trayectoria de recorte 9:16 del segmento procesado, o ``None``
            cuando no hizo falta reframe (el vídeo ya era 9:16 en modo repost).
        subtitles: Ruta del ``.ass`` generado; el archivo se elimina en la
            limpieza final y la ruta se conserva como procedencia. Es ``None``
            en modo repost, que no genera subtítulos.
        final_video: Vídeo vertical final con subtítulos quemados.
        cleaning: Rutas de los temporales eliminados al terminar.
    """

    source: Path
    transcript: Transcript | None
    moments: tuple[Moment, ...]
    selection: SegmentSelection
    reframe: ReframeResult | None
    subtitles: Path | None
    final_video: Path
    cleaning: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class PipelineConfig:
    """Configuración del pipeline long_video.

    Attributes:
        output_dir: Directorio donde se publican ``source.mp4`` y ``final.mp4``.
        contract: Contrato validado con las reglas y cotas de la selección.
        render: Binario, timeout y NVENC compartidos por los renders.
        face_model_path: Modelo ``.tflite`` de caras; solo se exige cuando no se
            inyecta un reframer.
        download_timeout_s: Timeout máximo de la descarga, en segundos.
        download_max_size_bytes: Tamaño máximo del vídeo descargado, en bytes.
        audio_locked: Si se debe inyectar una pista de audio externa en el vídeo
            final (modo audio obligatorio).
        audio_track_path: Pista de audio local a inyectar; mutuamente excluyente
            con ``audio_track_url``.
        audio_track_url: URL http/https de la pista de audio a descargar e
            inyectar; mutuamente excluyente con ``audio_track_path``.
        audio_mix_ratio: Peso de la pista externa: ``1.0`` la reemplaza, ``0.0``
            deja la original y un valor intermedio las mezcla.
        repost_mode: Si se activa el modo Repost/UGC: omite transcripción,
            detección de momentos y selección LLM, usa el vídeo completo como
            segmento y sólo reframea cuando el vídeo no es 9:16. Coincide con
            ``audio_locked`` para inyectar una pista externa.
    """

    output_dir: Path
    contract: Contract
    render: RenderConfig
    face_model_path: Path | None = None
    download_timeout_s: float = 600.0
    download_max_size_bytes: int = 2 * 1024**3
    audio_locked: bool = False
    audio_track_path: Path | None = None
    audio_track_url: str | None = None
    audio_mix_ratio: float = 1.0
    repost_mode: bool = False

    def __post_init__(self) -> None:
        """Valida la proporción de mezcla de la pista externa.

        Raises:
            ValueError: Si ``audio_mix_ratio`` queda fuera de ``[0.0, 1.0]``.
        """
        if not 0.0 <= self.audio_mix_ratio <= 1.0:
            msg = f"audio_mix_ratio fuera de rango [0.0, 1.0]: {self.audio_mix_ratio}"
            raise ValueError(msg)


class LongVideoModel(Protocol):
    """Modelo de runtime que el pipeline long_video consume."""

    def select_segments(self, prompt: Mapping[str, object]) -> object:
        """Selecciona los segmentos a partir del payload del prompt.

        Args:
            prompt: Payload serializable construido por el selector.

        Returns:
            La respuesta cruda del modelo: objeto, JSON en texto o bytes.
        """
        ...


class Reframer(Protocol):
    """Calcula la trayectoria 9:16 y renderiza el recorte del segmento."""

    def analyze(self, video: Path) -> ReframeResult:
        """Calcula la trayectoria de recorte 9:16 del vídeo.

        Args:
            video: Ruta del vídeo del segmento.

        Returns:
            La trayectoria de recorte con las dimensiones de origen.
        """
        ...

    def render(self, *, video: Path, destination: Path, result: ReframeResult) -> Path:
        """Renderiza el recorte 9:16 a partir de una trayectoria calculada.

        Args:
            video: Ruta del vídeo del segmento.
            destination: Ruta del artefacto recortado.
            result: Trayectoria de recorte ya calculada.

        Returns:
            La ruta del artefacto recortado.
        """
        ...


class SubtitleBurner(Protocol):
    """Genera el ``.ass`` y lo quema en el vídeo."""

    def write(self, words: Sequence[Word], destination: Path) -> Path:
        """Escribe el archivo ``.ass`` con las palabras dadas.

        Args:
            words: Palabras con marca de tiempo del segmento.
            destination: Ruta del archivo ``.ass``.

        Returns:
            La ruta del archivo escrito.
        """
        ...

    def burn(self, *, video: Path, subtitles: Path, destination: Path) -> Path:
        """Quema los subtítulos en el vídeo.

        Args:
            video: Ruta del vídeo del segmento reframeado.
            subtitles: Ruta del archivo ``.ass``.
            destination: Ruta del artefacto final.

        Returns:
            La ruta del artefacto con subtítulos quemados.
        """
        ...


@dataclass
class _CleanupRegistry:
    """Registro de temporales que se eliminan al terminar el pipeline."""

    paths: list[Path] = field(default_factory=list)

    def register(self, path: Path) -> Path:
        """Registra un temporal y devuelve la misma ruta.

        Args:
            path: Ruta del temporal a eliminar al finalizar.

        Returns:
            La misma ruta, para encadenar en la creación de temporales.
        """
        self.paths.append(path)
        return path

    def cleanup(self) -> tuple[str, ...]:
        """Elimina los temporales registrados que aún existan.

        Returns:
            Las rutas eliminadas, en el orden de registro.
        """
        removed: list[str] = []
        for path in self.paths:
            if not path.exists():
                continue
            with suppress(OSError):
                path.unlink()
            if not path.exists():
                removed.append(str(path))
        return tuple(removed)


@dataclass(frozen=True, slots=True)
class _Dependencies:
    """Dependencias externas ya resueltas para una corrida."""

    downloader: MediaDownloader
    detector: MomentDetector
    transcriber: Transcriber
    selector: SegmentSelector
    reframer: Reframer | None
    subtitle_renderer: SubtitleBurner


@dataclass(frozen=True, slots=True)
class _VideoInfo:
    """Dimensiones y duración leídas del vídeo con ffprobe."""

    width: int
    height: int
    duration_s: float


def run_long_video(
    url: str,
    *,
    model: LongVideoModel,
    config: PipelineConfig,
    detector: MomentDetector | None = None,
    transcriber: Transcriber | None = None,
    selector: SegmentSelector | None = None,
    reframer: Reframer | None = None,
    subtitle_renderer: SubtitleBurner | None = None,
) -> PipelineResult:
    """Ejecuta el pipeline long_video desde la URL hasta el vídeo final.

    Args:
        url: URL http/https del vídeo fuente.
        model: Modelo de runtime que elige los segmentos.
        config: Directorio de salida, contrato y render.
        detector: Detector de momentos; por defecto usa ffmpeg.
        transcriber: Transcriber word-level; por defecto usa faster-whisper.
        selector: Constructor y validador del prompt; por defecto el de LLM.
        reframer: Reframer 9:16; por defecto usa MediaPipe en CPU y ffmpeg.
        subtitle_renderer: Renderizador de subtítulos; por defecto usa ffmpeg.

    Returns:
        El resultado con los artefactos y las rutas de temporales limpiados.

    Raises:
        PipelineError: Si una etapa falla o no se puede construir el reframe.
    """
    try:
        config.output_dir.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        msg = f"no se pudo preparar el directorio de salida {config.output_dir}: {error}"
        raise PipelineError(msg) from error
    dependencies = _resolve_dependencies(
        config=config,
        detector=detector,
        transcriber=transcriber,
        selector=selector,
        reframer=reframer,
        subtitle_renderer=subtitle_renderer,
    )
    registry = _CleanupRegistry()
    try:
        result = _run_stages(
            url,
            model=model,
            config=config,
            dependencies=dependencies,
            registry=registry,
        )
    finally:
        cleaning = registry.cleanup()
    return replace(result, cleaning=cleaning)


def _resolve_dependencies(
    *,
    config: PipelineConfig,
    detector: MomentDetector | None,
    transcriber: Transcriber | None,
    selector: SegmentSelector | None,
    reframer: Reframer | None,
    subtitle_renderer: SubtitleBurner | None,
) -> _Dependencies:
    """Resuelve las dependencias externas, construyendo las estándar.

    Args:
        config: Configuración con los binarios, rutas y modelo de caras.
        detector: Detector inyectado, o ``None`` para usar el de ffmpeg.
        transcriber: Transcriber inyectado, o ``None`` para usar faster-whisper.
        selector: Selector inyectado, o ``None`` para usar el de LLM.
        reframer: Reframer inyectado, o ``None`` para construir el de MediaPipe.
        subtitle_renderer: Renderizador inyectado, o ``None`` para usar ffmpeg.

    Returns:
        Las dependencias listas para ejecutar el pipeline.

    Raises:
        PipelineError: Si no se puede construir el reframe por defecto.
    """
    return _Dependencies(
        downloader=MediaDownloader(
            timeout_s=config.download_timeout_s,
            max_size_bytes=config.download_max_size_bytes,
        ),
        detector=FFmpegMomentDetector() if detector is None else detector,
        transcriber=FasterWhisperTranscriber() if transcriber is None else transcriber,
        selector=LLMSegmentSelector() if selector is None else selector,
        reframer=_resolve_reframer(config, injected=reframer),
        subtitle_renderer=(
            SubtitleRenderer(render=config.render)
            if subtitle_renderer is None
            else subtitle_renderer
        ),
    )


def _default_reframer(config: PipelineConfig) -> FFmpegReframer:
    if config.face_model_path is None:
        msg = (
            "se requiere un reframer inyectado o face_model_path para el reframe 9:16 con MediaPipe"
        )
        raise PipelineError(msg)
    return FFmpegReframer(
        detector=MediaPipeFaceDetector(model_path=config.face_model_path),
        render=config.render,
    )


def _resolve_reframer(config: PipelineConfig, *, injected: Reframer | None) -> Reframer | None:
    """Resuelve el reframer sin exigirlo cuando el modo repost no lo necesite.

    Args:
        config: Configuración con el modo y el modelo de caras opcional.
        injected: Reframer inyectado por el llamador, o ``None``.

    Returns:
        El reframer inyectado, el estándar de MediaPipe, o ``None`` en modo
        repost sin ``face_model_path`` (el vídeo ya vertical no se reframea).

    Raises:
        PipelineError: Si se requiere el reframer estándar y falta
            ``face_model_path``.
    """
    if injected is not None:
        return injected
    if config.repost_mode and config.face_model_path is None:
        return None
    return _default_reframer(config)


def _require_reframer(dependencies: _Dependencies) -> Reframer:
    """Devuelve el reframer resuelto o falla si se necesita y no existe.

    Args:
        dependencies: Dependencias resueltas para la corrida.

    Returns:
        El reframer listo para analizar y renderizar.

    Raises:
        PipelineError: Si el repost necesita reframe pero no se configuró un
            reframer ni ``face_model_path``.
    """
    reframer = dependencies.reframer
    if reframer is None:
        msg = "se requiere un reframer inyectado o face_model_path para reframe 9:16 con MediaPipe"
        raise PipelineError(msg)
    return reframer


def _run_stages(
    url: str,
    *,
    model: LongVideoModel,
    config: PipelineConfig,
    dependencies: _Dependencies,
    registry: _CleanupRegistry,
) -> PipelineResult:
    source = _download(url, config=config, downloader=dependencies.downloader, registry=registry)
    if config.repost_mode:
        # Repost/UGC: sin transcripción, sin momentos y sin LLM; el vídeo
        # completo es el segmento.
        transcript: Transcript | None = None
        moments: tuple[Moment, ...] = ()
        selection = _full_video_selection(source, render=config.render)
    else:
        transcript = _transcribe(source, transcriber=dependencies.transcriber)
        moments = _detect(source, transcript=transcript, detector=dependencies.detector)
        selection = _select(
            transcript,
            moments,
            contract=config.contract,
            model=model,
            selector=dependencies.selector,
        )
    segment = _primary_segment(selection)
    passthrough = (
        config.repost_mode and not config.audio_locked and not _needs_reframe(source, config.render)
    )
    if passthrough:
        # Repost de un vídeo ya 9:16 sin audio externo: el segmento es el vídeo
        # completo y se copia sin recodificar para preservar la calidad.
        clip = _passthrough(
            source,
            output_dir=config.output_dir,
            render=config.render,
            registry=registry,
        )
    else:
        clip = _cut_segment(source, segment=segment, render=config.render, registry=registry)
    reframe: ReframeResult | None = None
    reframed: Path
    if config.repost_mode and not _needs_reframe(clip, config.render):
        # El vídeo ya es 9:16: passthrough sin tocar la pista visual.
        reframed = clip
    else:
        reframe, reframed = _reframe(
            clip,
            reframer=_require_reframer(dependencies),
            registry=registry,
        )
    with_audio = (
        _inject_audio(
            reframed,
            config=config,
            downloader=dependencies.downloader,
            registry=registry,
        )
        if config.audio_locked
        else reframed
    )
    subtitles: Path | None
    if transcript is not None:
        subtitles = _write_subtitles(
            transcript,
            segment=segment,
            renderer=dependencies.subtitle_renderer,
            output_dir=config.output_dir,
            registry=registry,
        )
        final_video = _burn(
            with_audio,
            subtitles=subtitles,
            renderer=dependencies.subtitle_renderer,
            output_dir=config.output_dir,
        )
    else:
        subtitles = None
        final_video = _publish(with_audio, output_dir=config.output_dir, registry=registry)
    return PipelineResult(
        source=source,
        transcript=transcript,
        moments=moments,
        selection=selection,
        reframe=reframe,
        subtitles=subtitles,
        final_video=final_video,
        cleaning=(),
    )


def _download(
    url: str,
    *,
    config: PipelineConfig,
    downloader: MediaDownloader,
    registry: _CleanupRegistry,
) -> Path:
    source = config.output_dir / _SOURCE_NAME
    temporary = registry.register(_temporary_path(source))
    with _translated("descarga"):
        _ = downloader.download_video(url=url, destination=temporary)
        _ = temporary.replace(source)
    return source


def _probe_video(video: Path, *, render: RenderConfig) -> _VideoInfo:
    """Lee las dimensiones y la duración de un vídeo con ffprobe.

    Args:
        video: Ruta del vídeo a sondear.
        render: Configuración con el timeout del sondeo.

    Returns:
        Las dimensiones y la duración del vídeo.

    Raises:
        PipelineError: Si ffprobe no está disponible, falla, expira o no se
            pueden leer las dimensiones o la duración.
    """
    argv = [
        _FFPROBE,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height:stream_side_data=rotation:format=duration",
        "-of",
        "default=noprint_wrappers=1",
        str(video),
    ]
    try:
        completed = subprocess.run(
            argv,
            capture_output=True,
            text=True,
            timeout=render.timeout_s,
            check=False,
        )
    except FileNotFoundError as error:
        msg = f"ffprobe no está disponible: {_FFPROBE}"
        raise PipelineError(msg) from error
    except subprocess.TimeoutExpired as error:
        msg = f"ffprobe excedió el timeout de {render.timeout_s} s"
        raise PipelineError(msg) from error
    except OSError as error:
        msg = f"no se pudo ejecutar ffprobe ({_FFPROBE}): {error}"
        raise PipelineError(msg) from error
    if completed.returncode != 0:
        msg = f"ffprobe falló con código {completed.returncode}: {_tail(completed.stderr)}"
        raise PipelineError(msg)
    fields = _probe_fields(completed.stdout)
    width = _probe_dimension(fields.get("width"), video=video)
    height = _probe_dimension(fields.get("height"), video=video)
    duration_s = _probe_duration(fields.get("duration"), video=video)
    rotation = _probe_rotation(fields.get("rotation"))
    if rotation in {90, 270, -90}:
        width, height = height, width
    if width <= 0 or height <= 0:
        msg = f"dimensiones inválidas en {video}: {width}x{height}"
        raise PipelineError(msg)
    return _VideoInfo(width=width, height=height, duration_s=duration_s)


def _probe_fields(stdout: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for line in stdout.splitlines():
        key, separator, value = line.partition("=")
        if separator:
            fields[key.strip()] = value.strip()
    return fields


def _probe_dimension(raw: str | None, *, video: Path) -> int:
    if raw is None or not raw.isdigit():
        msg = f"no se pudieron leer las dimensiones de {video}: {raw!r}"
        raise PipelineError(msg)
    return int(raw)


def _probe_duration(raw: str | None, *, video: Path) -> float:
    if raw is None:
        msg = f"no se pudo leer la duración de {video}"
        raise PipelineError(msg)
    try:
        return float(raw)
    except ValueError:
        msg = f"no se pudo leer la duración de {video}: {raw!r}"
        raise PipelineError(msg) from None


def _probe_rotation(raw: str | None) -> int:
    """Parsea el ángulo de rotación de los metadatos del stream.

    Args:
        raw: Valor crudo del campo ``rotation`` de la Display Matrix, o ``None``
            si no aparece.

    Returns:
        El ángulo de rotación en grados, o ``0`` si falta o no es un entero.
    """
    if not raw:
        return 0
    try:
        return int(raw)
    except ValueError:
        return 0


def _is_vertical(width: int, height: int) -> bool:
    # Las dimensiones vienen validadas por _probe_video: aquí son positivas.
    aspect = width / height
    return abs(aspect - _TARGET_ASPECT) <= _TARGET_ASPECT * _ASPECT_TOLERANCE


def _needs_reframe(video: Path, render: RenderConfig) -> bool:
    """Indica si el vídeo debe pasar por el reframe 9:16.

    Args:
        video: Ruta del vídeo a evaluar.
        render: Configuración con el timeout del sondeo.

    Returns:
        ``True`` si el vídeo no es 9:16 dentro de la tolerancia; ``False`` si ya
        es vertical.

    Raises:
        PipelineError: Si no se pueden leer las dimensiones del vídeo.
    """
    info = _probe_video(video, render=render)
    return not _is_vertical(info.width, info.height)


def _full_video_selection(source: Path, *, render: RenderConfig) -> SegmentSelection:
    """Construye la selección que cubre el vídeo completo (modo repost).

    Args:
        source: Ruta del vídeo fuente descargado.
        render: Configuración con el timeout del sondeo.

    Returns:
        Una selección con un único segmento ``[0, duración]``.

    Raises:
        PipelineError: Si no se puede leer la duración del vídeo o no es
            utilizable.
    """
    info = _probe_video(source, render=render)
    if info.duration_s <= 0.0:
        msg = f"el vídeo fuente no tiene duración utilizable: {source}"
        raise PipelineError(msg)
    return SegmentSelection(
        segments=(Segment(start_s=0.0, end_s=info.duration_s),),
        rationale="modo repost: el vídeo completo es el segmento",
    )


def _publish(video: Path, *, output_dir: Path, registry: _CleanupRegistry) -> Path:
    """Publica el vídeo procesado como artefacto final sin subtítulos.

    Args:
        video: Vídeo procesado (cortado, reframeado y/o con audio inyectado).
        output_dir: Directorio donde se publica ``final.mp4``.
        registry: Registro del temporal de publicación.

    Returns:
        La ruta del artefacto final.

    Raises:
        PipelineError: Si no se puede copiar o publicar el artefacto.
    """
    destination = output_dir / _FINAL_NAME
    temporary = registry.register(_temporary_path(destination))
    with _translated("publicación del vídeo final"):
        _ = shutil.copyfile(video, temporary)
        _ = temporary.replace(destination)
    return destination


def _transcribe(source: Path, *, transcriber: Transcriber) -> Transcript:
    with _translated("transcripción"):
        return transcriber.transcribe(source)


def _detect(
    source: Path,
    *,
    transcript: Transcript,
    detector: MomentDetector,
) -> tuple[Moment, ...]:
    with _translated("detección de momentos"):
        return detector.detect(source, transcript=transcript)


def _select(
    transcript: Transcript,
    moments: tuple[Moment, ...],
    *,
    contract: Contract,
    model: LongVideoModel,
    selector: SegmentSelector,
) -> SegmentSelection:
    with _translated("selección de segmentos"):
        prompt = selector.build_prompt(transcript, moments, contract)
        raw = model.select_segments(prompt)
        return selector.parse_response(raw)


def _primary_segment(selection: SegmentSelection) -> Segment:
    if not selection.segments:
        msg = "la selección no contiene segmentos"
        raise PipelineError(msg)
    return selection.segments[0]


def _cut_segment(
    source: Path,
    *,
    segment: Segment,
    render: RenderConfig,
    registry: _CleanupRegistry,
) -> Path:
    destination = registry.register(_temporary_path(source.with_name("segment.mp4")))
    argv = [
        render.ffmpeg,
        "-hide_banner",
        "-nostdin",
        "-v",
        "error",
        "-y",
        "-ss",
        _seconds(segment.start_s),
        "-t",
        _seconds(segment.end_s - segment.start_s),
        "-i",
        str(source),
        "-map",
        "0:v:0",
        "-map",
        "0:a?",
    ]
    argv += list(video_encoder_arguments(nvenc_available=render.nvenc_available))
    argv += list(audio_and_container_arguments())
    argv.append(str(destination))
    with _translated("corte del segmento"):
        _run_ffmpeg(argv, render=render)
    return destination


def _passthrough(
    video: Path,
    *,
    output_dir: Path,
    render: RenderConfig,
    registry: _CleanupRegistry,
) -> Path:
    """Copia el vídeo sin recodificar (stream copy).

    Args:
        video: Ruta del vídeo fuente ``9:16`` que se copia tal cual.
        output_dir: Directorio de salida donde se crea el temporal.
        render: Configuración con el binario y el timeout del render.
        registry: Registro del temporal para limpieza.

    Returns:
        La ruta del vídeo copiado sin recodificar.

    Raises:
        PipelineError: Si ffmpeg no está disponible o falla.
    """
    destination = registry.register(_temporary_path(output_dir / "passthrough.mp4"))
    argv = [
        render.ffmpeg,
        "-hide_banner",
        "-nostdin",
        "-v",
        "error",
        "-y",
        "-i",
        str(video),
        "-c",
        "copy",
        "-movflags",
        "+faststart",
        str(destination),
    ]
    with _translated("passthrough"):
        _run_ffmpeg(argv, render=render)
    return destination


def _reframe(
    clip: Path,
    *,
    reframer: Reframer,
    registry: _CleanupRegistry,
) -> tuple[ReframeResult, Path]:
    destination = registry.register(_temporary_path(clip.with_name("reframed.mp4")))
    with _translated("reframe 9:16"):
        result = reframer.analyze(clip)
        _ = reframer.render(video=clip, destination=destination, result=result)
    return result, destination


def _inject_audio(
    video: Path,
    *,
    config: PipelineConfig,
    downloader: MediaDownloader,
    registry: _CleanupRegistry,
) -> Path:
    """Inyecta la pista externa reemplazando o mezclando la original.

    La pista se resuelve desde ``audio_track_path`` o ``audio_track_url`` y el
    render escribe en un temporal hermano registrado para limpieza. El vídeo se
    copia y solo se recodifica el audio.

    Args:
        video: Vídeo reframeado sin la pista externa.
        config: Configuración con la pista, la proporción y el render.
        downloader: Descargador acotado para pistas entregadas por URL.
        registry: Registro de temporales para limpiar la pista y el resultado.

    Returns:
        La ruta del vídeo con la pista externa inyectada.

    Raises:
        PipelineError: Si la pista no existe, no se configuró ninguna o ffmpeg
            falla.
    """
    track = _resolve_audio_track(config, downloader=downloader, registry=registry)
    destination = registry.register(_temporary_path(video.with_name("audio_injected.mp4")))
    argv = [
        config.render.ffmpeg,
        "-hide_banner",
        "-nostdin",
        "-v",
        "error",
        "-y",
        "-i",
        str(video),
        "-i",
        str(track),
    ]
    argv += list(audio_injection_arguments(mix_ratio=config.audio_mix_ratio))
    argv.append(str(destination))
    with _translated("inyección de audio"):
        _run_ffmpeg(argv, render=config.render)
    return destination


def _resolve_audio_track(
    config: PipelineConfig,
    *,
    downloader: MediaDownloader,
    registry: _CleanupRegistry,
) -> Path:
    """Resuelve la pista de audio externa, descargándola si llega por URL.

    Args:
        config: Configuración con la ruta local o la URL de la pista.
        downloader: Descargador acotado que se usa cuando la pista es una URL.
        registry: Registro donde se anota la pista descargada como temporal.

    Returns:
        La ruta local de la pista de audio.

    Raises:
        PipelineError: Si se configuraron ambas fuentes, ninguna o la ruta no
            existe.
    """
    if config.audio_track_path is not None and config.audio_track_url is not None:
        msg = "audio_locked acepta audio_track_path o audio_track_url, no ambos"
        raise PipelineError(msg)
    if config.audio_track_path is not None:
        if not config.audio_track_path.is_file():
            msg = f"la pista de audio no existe: {config.audio_track_path}"
            raise PipelineError(msg)
        return config.audio_track_path
    if config.audio_track_url is not None:
        destination = registry.register(_temporary_path(config.output_dir / "audio_track.mp3"))
        with _translated("descarga de audio"):
            _ = downloader.download_video(url=config.audio_track_url, destination=destination)
        return destination
    msg = "audio_locked requiere audio_track_path o audio_track_url"
    raise PipelineError(msg)


def _write_subtitles(
    transcript: Transcript,
    *,
    segment: Segment,
    renderer: SubtitleBurner,
    output_dir: Path,
    registry: _CleanupRegistry,
) -> Path:
    destination = registry.register(_temporary_path(output_dir / "subtitles.ass"))
    words = _segment_words(transcript, segment)
    with _translated("subtítulos"):
        return renderer.write(words, destination)


def _burn(
    video: Path,
    *,
    subtitles: Path,
    renderer: SubtitleBurner,
    output_dir: Path,
) -> Path:
    destination = output_dir / _FINAL_NAME
    with _translated("quemado de subtítulos"):
        return renderer.burn(video=video, subtitles=subtitles, destination=destination)


def _segment_words(transcript: Transcript, segment: Segment) -> tuple[Word, ...]:
    duration_s = segment.end_s - segment.start_s
    words: list[Word] = []
    for word in transcript.words:
        if word.end_s <= segment.start_s or word.start_s >= segment.end_s:
            continue
        start_s = max(0.0, word.start_s - segment.start_s)
        end_s = min(duration_s, word.end_s - segment.start_s)
        if end_s <= start_s:
            continue
        words.append(
            Word(
                start_s=start_s,
                end_s=end_s,
                text=word.text,
                confidence=word.confidence,
                token_id=len(words),
            )
        )
    return tuple(words)


@contextmanager
def _translated(stage: str) -> Generator[None]:
    # Frontera de etapa: cualquier fallo externo se traduce a PipelineError
    # conservando la causa; los PipelineError propios se dejan pasar.
    try:
        yield
    except PipelineError:
        raise
    except Exception as error:
        msg = f"la etapa '{stage}' falló: {error}"
        raise PipelineError(msg) from error


def _run_ffmpeg(argv: list[str], *, render: RenderConfig) -> None:
    try:
        completed = subprocess.run(
            list(argv),
            capture_output=True,
            text=True,
            timeout=render.timeout_s,
            check=False,
        )
    except FileNotFoundError as error:
        msg = f"ffmpeg no está disponible: {render.ffmpeg}"
        raise PipelineError(msg) from error
    except subprocess.TimeoutExpired as error:
        msg = f"ffmpeg excedió el timeout de {render.timeout_s} s"
        raise PipelineError(msg) from error
    except OSError as error:
        msg = f"no se pudo ejecutar ffmpeg ({render.ffmpeg}): {error}"
        raise PipelineError(msg) from error
    if completed.returncode != 0:
        msg = f"ffmpeg falló con código {completed.returncode}: {_tail(completed.stderr)}"
        raise PipelineError(msg)


def _temporary_path(destination: Path) -> Path:
    return destination.with_name(f".{destination.stem}.part-{uuid.uuid4().hex}{destination.suffix}")


def _seconds(value: float) -> str:
    return f"{value:.3f}"


def _tail(text: str) -> str:
    return text.strip()[-_STDERR_TAIL:]
