"""Orquestador end-to-end del modo ``long_video`` (fase C5).

Encadena descarga -> transcripción -> detección de momentos -> selección LLM ->
reframe 9:16 -> subtítulos -> vídeo vertical final. No implementa ninguna de las
etapas: compone los módulos de C2-C4 y traduce cualquier fallo de una etapa a
``PipelineError`` conservando la causa.

Los artefactos finales (``source.mp4`` y ``final.mp4``) se publican de forma
atómica: ffmpeg escribe en un temporal hermano y solo un render exitoso
reemplaza el destino. Un registro de limpieza elimina todos los temporales
(``.part`` de la descarga, clips intermedios, ``.ass`` y reframes) tanto en el
camino feliz como ante error.
"""

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


class PipelineError(Exception):
    """El pipeline end-to-end no se pudo completar."""


@dataclass(frozen=True, slots=True)
class PipelineResult:
    """Resultado del pipeline long_video.

    Attributes:
        source: Vídeo fuente descargado, conservado en ``output_dir``.
        transcript: Transcripción word-level del vídeo.
        moments: Momentos candidatos detectados, ordenados por puntuación.
        selection: Segmentos elegidos por el modelo.
        reframe: Trayectoria de recorte 9:16 del segmento procesado.
        subtitles: Ruta del ``.ass`` generado; el archivo se elimina en la
            limpieza final y la ruta se conserva como procedencia.
        final_video: Vídeo vertical final con subtítulos quemados.
        cleaning: Rutas de los temporales eliminados al terminar.
    """

    source: Path
    transcript: Transcript
    moments: tuple[Moment, ...]
    selection: SegmentSelection
    reframe: ReframeResult
    subtitles: Path
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
    """

    output_dir: Path
    contract: Contract
    render: RenderConfig
    face_model_path: Path | None = None
    download_timeout_s: float = 600.0
    download_max_size_bytes: int = 2 * 1024**3


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
    reframer: Reframer
    subtitle_renderer: SubtitleBurner


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
        reframer=_default_reframer(config) if reframer is None else reframer,
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


def _run_stages(
    url: str,
    *,
    model: LongVideoModel,
    config: PipelineConfig,
    dependencies: _Dependencies,
    registry: _CleanupRegistry,
) -> PipelineResult:
    source = _download(url, config=config, downloader=dependencies.downloader, registry=registry)
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
    clip = _cut_segment(source, segment=segment, render=config.render, registry=registry)
    reframe, reframed = _reframe(clip, reframer=dependencies.reframer, registry=registry)
    subtitles = _write_subtitles(
        transcript,
        segment=segment,
        renderer=dependencies.subtitle_renderer,
        output_dir=config.output_dir,
        registry=registry,
    )
    final_video = _burn(
        reframed,
        subtitles=subtitles,
        renderer=dependencies.subtitle_renderer,
        output_dir=config.output_dir,
    )
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
