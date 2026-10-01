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

El modo Slideshow (``run_slideshow``) convierte una secuencia de imágenes
estáticas en un vídeo vertical continuo. Salta las capas de inteligencia (no
transcribe ni selecciona segmentos, porque no hay voz ni momentos que elegir) y
exige audio externo: sin ``audio_locked`` no arranca. Las imágenes se ensamblan
con el demuxer ``concat`` de ffmpeg, que escala y rellena cada slide a 9:16, y
el resultado se publica con la misma limpieza determinista.
"""

import json
import logging
import math
import os
import shutil
import subprocess
import tempfile
import uuid
from collections.abc import Generator, Mapping, Sequence
from contextlib import contextmanager, suppress
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Protocol, overload

from pydantic import TypeAdapter, ValidationError

from kliptych.assets import AssetError, AssetRegistry
from kliptych.contract import (
    Contract,
    Format,
    Segment,
    TimestampRange,
    contract_digest,
    contract_mutes_audio,
)
from kliptych.download import MediaDownloader
from kliptych.encoding import (
    RenderConfig,
    audio_and_container_arguments,
    audio_injection_arguments,
    muted_audio_arguments,
    run_ffmpeg_with_fallback,
    video_encoder_arguments,
)
from kliptych.hashing import sha256_canonical_json, sha256_file
from kliptych.lyrics import (
    LrcEmptyWindowError,
    LyricLine,
    LyricsError,
    SyncedLyricsProvider,
    cut_lyric_window,
    lyric_lines_to_ass,
    parse_lrc,
)
from kliptych.moments import FFmpegMomentDetector, Moment, MomentDetector
from kliptych.pipeline_state import PipelineStage, PipelineStateManager
from kliptych.reframe import FFmpegReframer, MediaPipeFaceDetector, ReframeResult
from kliptych.segment import LLMSegmentSelector, SegmentSelection, SegmentSelector
from kliptych.subtitles import SubtitleRenderer
from kliptych.transcribe import (
    FasterWhisperTranscriber,
    Transcriber,
    Transcript,
    Word,
)

logger = logging.getLogger(__name__)

_STDERR_TAIL = 400
_SOURCE_NAME = "source.mp4"
_AUDIO_TRACK_NAME = "audio_track.mp3"
_FINAL_NAME = "final.mp4"
_FFPROBE = "ffprobe"
_AUDIO_DURATION_TOLERANCE_S = 0.1
_TARGET_ASPECT = 9.0 / 16.0
_ASPECT_TOLERANCE = 0.05
_SLIDESHOW_NAME = "slideshow.mp4"
_SLIDESHOW_CONCAT_NAME = "slideshow_input.txt"
_SLIDESHOW_WIDTH = 1080
_SLIDESHOW_HEIGHT = 1920
_SLIDESHOW_FPS = "30"
_SLIDESHOW_FILTER = (
    f"scale={_SLIDESHOW_WIDTH}:{_SLIDESHOW_HEIGHT}:force_original_aspect_ratio=decrease,"
    f"pad={_SLIDESHOW_WIDTH}:{_SLIDESHOW_HEIGHT}:(ow-iw)/2:(oh-ih)/2,setsar=1"
)
_CUT_DURATION_TOLERANCE_S = 0.5


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
        source_offset_sec: Inicio absoluto (en segundos del vídeo original) de
            la ventana quirúrgica descargada; ``0.0`` sin
            ``timestamp_ranges``. Los segmentos de la selección son relativos
            a la descarga y el recorte del ``.lrc`` suma este desplazamiento
            para usar coordenadas absolutas.
    """

    source: Path
    transcript: Transcript | None
    moments: tuple[Moment, ...]
    selection: SegmentSelection
    reframe: ReframeResult | None
    subtitles: Path | None
    final_video: Path
    cleaning: tuple[str, ...]
    final_videos: tuple[Path, ...] = ()
    source_offset_sec: float = 0.0

    def __post_init__(self) -> None:
        """Inicializa final_videos con final_video si no se proporcionó."""
        if not self.final_videos and self.final_video:
            object.__setattr__(self, "final_videos", (self.final_video,))


@dataclass(frozen=True, slots=True)
class SlideshowResult:
    """Resultado del pipeline slideshow.

    Attributes:
        images: Imágenes resueltas, en orden de montaje; las descargadas desde
            URL se eliminan en la limpieza final.
        slideshow_video: Vídeo intermedio ensamblado a partir de las imágenes;
            se elimina en la limpieza final y la ruta se conserva como
            procedencia.
        final_video: Vídeo vertical final con el audio externo inyectado.
        subtitles: Siempre ``None``: el slideshow no transcribe y por tanto no
            genera subtítulos.
        cleaning: Rutas de los temporales eliminados al terminar.
        final_videos: Todos los vídeos generados en el lote.
    """

    images: tuple[Path, ...]
    slideshow_video: Path
    final_video: Path
    subtitles: Path | None
    cleaning: tuple[str, ...]
    final_videos: tuple[Path, ...] = ()

    def __post_init__(self) -> None:
        """Inicializa final_videos con final_video si no se proporcionó."""
        if not self.final_videos and self.final_video:
            object.__setattr__(self, "final_videos", (self.final_video,))


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
        watermark_path: PNG del watermark superpuesto en el render; sus bytes
            entran al fingerprint para que cualquier cambio invalide
            ``--resume``.
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
    watermark_path: Path | None = None
    lrc_path: Path | None = None
    synced_lyrics_provider: SyncedLyricsProvider | None = None
    assets: AssetRegistry | None = None

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

    def burn(
        self, *, video: Path, subtitles: Path, destination: Path, mute_audio: bool = False
    ) -> Path:
        """Quema los subtítulos en el vídeo.

        Args:
            video: Ruta del vídeo del segmento reframeado.
            subtitles: Ruta del archivo ``.ass``.
            destination: Ruta del artefacto final.
            mute_audio: Si es True, silencia la pista sin eliminarla.

        Returns:
            La ruta del artefacto con subtítulos quemados.
        """
        ...


@dataclass
class _CleanupRegistry:
    """Registro de temporales que se eliminan al terminar el pipeline."""

    temps: list[Path] = field(default_factory=list)
    artifacts: list[Path] = field(default_factory=list)

    def register(self, path: Path, *, is_artifact: bool = False) -> Path:
        """Registra un archivo y devuelve la misma ruta.

        Args:
            path: Ruta del archivo a registrar.
            is_artifact: Si es True, es un artefacto útil conservado ante
                fallos para permitir la reanudación del pipeline.

        Returns:
            La misma ruta, para encadenar en la creación de temporales.
        """
        if is_artifact:
            self.artifacts.append(path)
        else:
            self.temps.append(path)
        return path

    @property
    def paths(self) -> list[Path]:
        return [*self.temps, *self.artifacts]

    @paths.setter
    def paths(self, value: list[Path]) -> None:
        self.temps = list(value)
        self.artifacts = []

    def cleanup(self, mode: str = "all") -> tuple[str, ...]:
        """Elimina los temporales registrados según el modo.

        Args:
            mode: "all" elimina todo (éxito o reinicio forzado); "temp_only"
                elimina solo verdaderos temporales, conservando artefactos
                descargados para resume.

        Returns:
            Las rutas eliminadas, en el orden de registro.

        Raises:
            ValueError: Si el modo no es 'all' ni 'temp_only'.
        """
        if mode not in {"all", "temp_only"}:
            msg = f"modo de limpieza inválido: {mode}"
            raise ValueError(msg)
        targets = [*self.temps, *self.artifacts] if mode == "all" else list(self.temps)
        removed: list[str] = []
        for path in targets:
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


@overload
def _file_signature(path: None) -> None: ...


@overload
def _file_signature(path: Path | str) -> str: ...


def _file_signature(path: Path | str | None) -> str | None:
    """Calcula una firma determinista de un archivo local o retorna su ruta si no existe.

    Args:
        path: Ruta o URL del archivo a firmar.

    Returns:
        None si path es None; la combinación de la ruta normalizada y el sha256
        del contenido si el archivo existe en disco; o la ruta como string si no
        existe o es una URL remota.
    """
    if path is None:
        return None
    if isinstance(path, str) and path.startswith(("http://", "https://")):
        return path
    try:
        p = Path(path)
        if p.is_file():
            normalized = Path(os.path.normpath(str(p))).as_posix()
            return f"{normalized}:{sha256_file(p)}"
    except (OSError, ValueError):
        pass
    return str(path)


def compute_long_video_fingerprint(
    url: str,
    *,
    config: PipelineConfig,
) -> str:
    """Calcula el digest SHA-256 determinista de las entradas de long_video.

    Args:
        url: URL del vídeo fuente.
        config: Configuración con contrato, render y audio.

    Returns:
        El digest SHA-256 en hexadecimal.
    """
    payload: dict[str, object] = {
        "pipeline": "long_video",
        "url": _file_signature(url),
        "contract": contract_digest(config.contract),
        "face_model_path": _file_signature(config.face_model_path),
        "audio": {
            "locked": config.audio_locked,
            "track_path": _file_signature(config.audio_track_path),
            "track_url": config.audio_track_url,
            "mix_ratio": config.audio_mix_ratio,
        },
        "render": {
            "ffmpeg": config.render.ffmpeg,
            "timeout_s": config.render.timeout_s,
            "nvenc_available": config.render.nvenc_available,
        },
        "download": {
            "timestamp_ranges": [
                {"start_sec": tr.start_sec, "end_sec": tr.end_sec}
                for tr in config.contract.timestamp_ranges
            ],
            "download_section_mode": bool(config.contract.timestamp_ranges),
        },
        "repost_mode": config.repost_mode,
        "watermark": _watermark_signature(config),
    }
    lyric_sig = _lyric_signature(config)
    if lyric_sig is not None:
        payload["lyric_video"] = lyric_sig
    split_sig = _split_screen_signature(config)
    if split_sig is not None:
        payload["split_screen"] = split_sig
    return sha256_canonical_json(payload)


def _lyric_signature(config: PipelineConfig) -> dict[str, object] | None:
    """Firma el formato lyric_video, configuración y archivo .lrc para --resume.

    Args:
        config: Configuración de la corrida.

    Returns:
        Diccionario con la firma determinista de letras, o None si no aplica.
    """
    if config.contract.format is not Format.LYRIC_VIDEO and config.contract.lyric_video is None:
        return None

    lrc_file_path: Path | None = config.lrc_path
    if (
        lrc_file_path is None
        and config.contract.lyric_video is not None
        and config.contract.lyric_video.lrc_asset_id
        and config.assets is not None
    ):
        try:
            lrc_file_path = config.assets.path_for(config.contract.lyric_video.lrc_asset_id)
        except (AssetError, OSError):
            lrc_file_path = None

    lrc_sha = None
    if lrc_file_path is not None and lrc_file_path.is_file():
        lrc_sha = sha256_file(lrc_file_path)

    return {
        "format": config.contract.format.value,
        "config": (
            config.contract.lyric_video.model_dump(mode="json")
            if config.contract.lyric_video is not None
            else None
        ),
        "lrc_file": _file_signature(lrc_file_path),
        "lrc_sha256": lrc_sha,
    }


def _watermark_signature(config: PipelineConfig) -> dict[str, object]:
    """Firma la configuración y los bytes del PNG del watermark para ``--resume``.

    La configuración entra por su digest canónico (un cambio de posición o
    tamaño invalida el checkpoint) y el PNG por su firma de contenido
    (``ruta:sha256``): retocar el asset sin tocar el contrato también
    invalida las etapas dependientes.

    Args:
        config: Configuración con el contrato y la ruta opcional del PNG.

    Returns:
        El bloque determinista del watermark para el fingerprint.
    """
    return {
        "config_sha256": sha256_canonical_json(config.contract.watermark.model_dump(mode="json")),
        "png": _file_signature(config.watermark_path),
    }


def _split_screen_signature(config: PipelineConfig) -> dict[str, object] | None:
    """Firma la geometría split_screen exacta para ``--resume``.

    Los valores entran tal cual (fuentes, gap, ratio y lienzo): cualquier
    cambio invalida el checkpoint aunque el resto del contrato no se mueva.

    Args:
        config: Configuración de la corrida.

    Returns:
        El bloque determinista del split, o ``None`` sin split declarado.
    """
    if config.contract.split_screen is None:
        return None
    return config.contract.split_screen.model_dump(mode="json")


def compute_slideshow_fingerprint(
    images: Sequence[Path | str],
    *,
    config: PipelineConfig,
    slide_duration_s: float,
) -> str:
    """Calcula el digest SHA-256 determinista de las entradas de slideshow.

    Args:
        images: Secuencia de rutas o URLs de imágenes.
        config: Configuración con contrato, render y audio.
        slide_duration_s: Duración por slide.

    Returns:
        El digest SHA-256 en hexadecimal.
    """
    payload: dict[str, object] = {
        "pipeline": "slideshow",
        "images": [_file_signature(img) for img in images],
        "slide_duration_s": slide_duration_s,
        "contract": contract_digest(config.contract),
        "face_model_path": _file_signature(config.face_model_path),
        "audio": {
            "locked": config.audio_locked,
            "track_path": _file_signature(config.audio_track_path),
            "track_url": config.audio_track_url,
            "mix_ratio": config.audio_mix_ratio,
        },
        "render": {
            "ffmpeg": config.render.ffmpeg,
            "timeout_s": config.render.timeout_s,
            "nvenc_available": config.render.nvenc_available,
        },
        "watermark": _watermark_signature(config),
    }
    split_sig = _split_screen_signature(config)
    if split_sig is not None:
        payload["split_screen"] = split_sig
    return sha256_canonical_json(payload)


def _reject_split_screen(config: PipelineConfig) -> None:
    """Rechaza split_screen en pipelines de un único panel (fail-closed).

    ``run_long_video`` y ``run_slideshow`` componen un solo panel 9:16: sin
    esta guarda, un contrato con ``split_screen`` generaría un MP4 de un
    panel que el gate ``layout.geometry`` aceptaría por sus dimensiones. El
    split solo se compone en el pipeline ``given_clips``.

    Args:
        config: Configuración con el contrato de la corrida.

    Raises:
        PipelineError: Si el contrato declara ``split_screen``.
    """
    if config.contract.split_screen is not None:
        msg = (
            "split_screen exige el pipeline given_clips con composición de dos "
            "paneles; este pipeline genera un único panel que el gate "
            "layout.geometry no puede distinguir de un split real"
        )
        raise PipelineError(msg)


def _validate_audio_combination(config: PipelineConfig) -> None:
    """Rechaza la inyección externa de audio con política de silencio (fail-closed).

    ``internal_official_sound`` exige silenciar el render final (el sonido
    oficial se añade al publicar): inyectar una pista externa (``audio_locked``
    o ``audio_track_path``/``audio_track_url``) y luego silenciarla es
    contradictorio y nunca se degrada en silencio.

    Args:
        config: Configuración con el contrato y la pista externa.

    Raises:
        PipelineError: Si el contrato silencia el audio y se pidió inyección
            externa.
    """
    if contract_mutes_audio(config.contract) and (
        config.audio_locked
        or config.audio_track_path is not None
        or config.audio_track_url is not None
    ):
        msg = (
            "audio_policy 'internal_official_sound' exige silenciar el render final; "
            "es incompatible con la inyección de audio externo "
            "(audio_locked o audio_track_path/audio_track_url)"
        )
        raise PipelineError(msg)


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
    resume: bool = False,
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
        resume: Si es True, reanuda la ejecución desde el último punto de control
            sin repetir las etapas ya completadas.

    Returns:
        El resultado con los artefactos y las rutas de temporales limpiados.

    Raises:
        PipelineError: Si una etapa falla o no se puede construir el reframe.
    """
    _reject_split_screen(config)
    _validate_audio_combination(config)
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
    state = PipelineStateManager(config.output_dir)
    fingerprint = compute_long_video_fingerprint(url, config=config)
    effective_resume = resume
    if effective_resume:
        if state.is_loaded:
            if state.checkpoint.input_fingerprint != fingerprint:
                logger.warning(
                    "Entradas cambiaron respecto al checkpoint previo; reiniciando ejecución limpia"
                )
                state.reset(input_fingerprint=fingerprint)
                effective_resume = False
        else:
            state.checkpoint.input_fingerprint = fingerprint
    else:
        state.reset(input_fingerprint=fingerprint)
    try:
        result = _run_stages(
            url,
            model=model,
            config=config,
            dependencies=dependencies,
            registry=registry,
            state=state,
            resume=effective_resume,
        )
        cleaning = registry.cleanup(mode="all")
        state.mark_done(PipelineStage.COMPLETED, result.final_video)
    except Exception:
        _ = registry.cleanup(mode="temp_only")
        raise
    return replace(result, cleaning=cleaning)


def run_audio_locked(
    url: str,
    *,
    model: LongVideoModel,
    config: PipelineConfig,
    detector: MomentDetector | None = None,
    transcriber: Transcriber | None = None,
    selector: SegmentSelector | None = None,
    reframer: Reframer | None = None,
    subtitle_renderer: SubtitleBurner | None = None,
    resume: bool = False,
) -> PipelineResult:
    """Ejecuta el pipeline long_video con pista de audio externa obligatoria.

    Args:
        url: URL http/https del vídeo fuente.
        model: Modelo de runtime que elige los segmentos.
        config: Directorio de salida, contrato y render; exige
            ``audio_locked=True`` con pista configurada.
        detector: Detector de momentos; por defecto usa ffmpeg.
        transcriber: Transcriber word-level; por defecto usa faster-whisper.
        selector: Constructor y validador del prompt; por defecto el de LLM.
        reframer: Reframer 9:16; por defecto usa MediaPipe en CPU y ffmpeg.
        subtitle_renderer: Renderizador de subtítulos; por defecto usa ffmpeg.
        resume: Si es True, reanuda la ejecución desde el último punto de control
            sin repetir las etapas ya completadas.

    Returns:
        El resultado con los artefactos y las rutas de temporales limpiados.

    Raises:
        PipelineError: Si la config no activa ``audio_locked`` o una etapa falla.
    """
    if not config.audio_locked:
        msg = "run_audio_locked exige PipelineConfig con audio_locked=True"
        raise PipelineError(msg)
    return run_long_video(
        url,
        model=model,
        config=config,
        detector=detector,
        transcriber=transcriber,
        selector=selector,
        reframer=reframer,
        subtitle_renderer=subtitle_renderer,
        resume=resume,
    )


def run_repost(
    url: str,
    *,
    model: LongVideoModel,
    config: PipelineConfig,
    detector: MomentDetector | None = None,
    transcriber: Transcriber | None = None,
    selector: SegmentSelector | None = None,
    reframer: Reframer | None = None,
    subtitle_renderer: SubtitleBurner | None = None,
    resume: bool = False,
) -> PipelineResult:
    """Ejecuta el pipeline en modo Repost/UGC: vídeo completo sin inteligencia.

    Args:
        url: URL http/https del vídeo fuente.
        model: Modelo de runtime (no se usa en modo repost, que omite la
            selección LLM).
        config: Directorio de salida, contrato y render; exige
            ``repost_mode=True``.
        detector: Detector de momentos; por defecto usa ffmpeg.
        transcriber: Transcriber word-level; por defecto usa faster-whisper.
        selector: Constructor y validador del prompt; por defecto el de LLM.
        reframer: Reframer 9:16; por defecto usa MediaPipe en CPU y ffmpeg.
        subtitle_renderer: Renderizador de subtítulos; por defecto usa ffmpeg.
        resume: Si es True, reanuda la ejecución desde el último punto de control
            sin repetir las etapas ya completadas.

    Returns:
        El resultado con los artefactos y las rutas de temporales limpiados.

    Raises:
        PipelineError: Si la config no activa ``repost_mode`` o una etapa falla.
    """
    if not config.repost_mode:
        msg = "run_repost exige PipelineConfig con repost_mode=True"
        raise PipelineError(msg)
    return run_long_video(
        url,
        model=model,
        config=config,
        detector=detector,
        transcriber=transcriber,
        selector=selector,
        reframer=reframer,
        subtitle_renderer=subtitle_renderer,
        resume=resume,
    )


def run_slideshow(
    images: Sequence[Path | str],
    *,
    config: PipelineConfig,
    slide_duration_s: float = 3.0,
    resume: bool = False,
) -> SlideshowResult:
    """Convierte una secuencia de imágenes en un vídeo vertical con audio.

    Ensambla las imágenes con el demuxer ``concat`` de ffmpeg, inyecta de forma
    obligatoria la pista externa y publica ``final.mp4`` de forma atómica. No
    transcribe ni llama al LLM: el slideshow no tiene voz ni momentos que
    seleccionar.

    Args:
        images: Rutas locales o URLs http/https de las imágenes, en orden de
            montaje.
        config: Directorio de salida, render y pista de audio externa; exige
            ``audio_locked=True``.
        slide_duration_s: Duración de cada slide en segundos; debe ser positiva.
        resume: Si es True, reanuda la ejecución aprovechando imágenes o pistas
            ya descargadas.

    Returns:
        El resultado con los artefactos y las rutas de temporales limpiados.

    Raises:
        PipelineError: Si la duración no es positiva, falta ``audio_locked``,
            no hay imágenes, una imagen no existe o una etapa falla.
    """
    _reject_split_screen(config)
    _validate_audio_combination(config)
    _validate_slideshow(images, config=config, slide_duration_s=slide_duration_s)
    try:
        config.output_dir.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        msg = f"no se pudo preparar el directorio de salida {config.output_dir}: {error}"
        raise PipelineError(msg) from error
    downloader = MediaDownloader(
        timeout_s=config.download_timeout_s,
        max_size_bytes=config.download_max_size_bytes,
    )
    registry = _CleanupRegistry()
    state = PipelineStateManager(config.output_dir)
    fingerprint = compute_slideshow_fingerprint(
        images,
        config=config,
        slide_duration_s=slide_duration_s,
    )
    effective_resume = resume
    if effective_resume:
        if state.is_loaded:
            if state.checkpoint.input_fingerprint != fingerprint:
                logger.warning(
                    "Entradas cambiaron respecto al checkpoint previo; reiniciando ejecución limpia"
                )
                state.reset(input_fingerprint=fingerprint)
                effective_resume = False
        else:
            state.checkpoint.input_fingerprint = fingerprint
    else:
        state.reset(input_fingerprint=fingerprint)
    try:
        result = _run_slideshow_stages(
            images,
            config=config,
            slide_duration_s=slide_duration_s,
            downloader=downloader,
            registry=registry,
            state=state,
            resume=effective_resume,
        )
        cleaning = registry.cleanup(mode="all")
        state.mark_done(PipelineStage.COMPLETED, result.final_video)
    except Exception:
        _ = registry.cleanup(mode="temp_only")
        raise
    return replace(result, cleaning=cleaning)


def _validate_slideshow(
    images: Sequence[Path | str],
    *,
    config: PipelineConfig,
    slide_duration_s: float,
) -> None:
    """Valida las precondiciones del modo slideshow.

    Args:
        images: Imágenes declaradas por el llamador.
        config: Configuración con la política de audio.
        slide_duration_s: Duración de cada slide en segundos.

    Raises:
        PipelineError: Si la duración no es positiva, ``audio_locked`` está
            desactivado, ``audio_mix_ratio`` es menor que ``1.0`` o no hay
            imágenes.
    """
    if slide_duration_s <= 0:
        msg = f"la duración por slide debe ser positiva: {slide_duration_s}"
        raise PipelineError(msg)
    if not config.audio_locked:
        msg = "el slideshow requiere audio_locked=True"
        raise PipelineError(msg)
    if config.audio_mix_ratio < 1.0:
        msg = (
            "el slideshow no tiene pista de audio original; requiere "
            "audio_mix_ratio >= 1.0 (reemplazo total del audio)"
        )
        raise PipelineError(msg)
    if not images:
        msg = "el slideshow requiere al menos una imagen"
        raise PipelineError(msg)


def _run_slideshow_stages(
    images: Sequence[Path | str],
    *,
    config: PipelineConfig,
    slide_duration_s: float,
    downloader: MediaDownloader,
    registry: _CleanupRegistry,
    state: PipelineStateManager,
    resume: bool = False,
) -> SlideshowResult:
    resolved = _resolve_slideshow_images(
        images,
        config=config,
        downloader=downloader,
        registry=registry,
        resume=resume,
    )
    concat = _write_concat_file(
        resolved,
        slide_duration_s=slide_duration_s,
        output_dir=config.output_dir,
        registry=registry,
    )
    slideshow_video = _assemble_slideshow(
        concat,
        total_duration_s=len(resolved) * slide_duration_s,
        render=config.render,
        output_dir=config.output_dir,
        registry=registry,
    )
    with_audio = _inject_audio(
        slideshow_video,
        config=config,
        downloader=downloader,
        registry=registry,
        resume=resume,
        state=state,
    )
    final_video = _publish(
        with_audio,
        output_dir=config.output_dir,
        registry=registry,
        mute_audio=contract_mutes_audio(config.contract),
        render=config.render,
    )
    return SlideshowResult(
        images=resolved,
        slideshow_video=slideshow_video,
        final_video=final_video,
        subtitles=None,
        cleaning=(),
    )


def _resolve_slideshow_images(
    images: Sequence[Path | str],
    *,
    config: PipelineConfig,
    downloader: MediaDownloader,
    registry: _CleanupRegistry,
    resume: bool = False,
) -> tuple[Path, ...]:
    return tuple(
        _resolve_slideshow_image(
            image,
            index=index,
            config=config,
            downloader=downloader,
            registry=registry,
            resume=resume,
        )
        for index, image in enumerate(images)
    )


def _resolve_slideshow_image(
    image: Path | str,
    *,
    index: int,
    config: PipelineConfig,
    downloader: MediaDownloader,
    registry: _CleanupRegistry,
    resume: bool = False,
) -> Path:
    """Resuelve una imagen local o descarga su URL como temporal registrado.

    Args:
        image: Ruta local o URL http/https de la imagen.
        index: Posición de la imagen en la secuencia, para nombrar el temporal.
        config: Configuración con el directorio de salida y los límites.
        downloader: Descargador acotado para las imágenes entregadas por URL.
        registry: Registro donde se anota la imagen descargada como temporal.
        resume: Si es True, reutiliza la imagen si ya fue descargada.

    Returns:
        La ruta local de la imagen.

    Raises:
        PipelineError: Si la imagen resuelta no existe como archivo.
    """
    if isinstance(image, Path) or not _is_url(image):
        path = Path(image)
    else:
        destination = config.output_dir / f"slide_{index:03d}.jpg"
        if resume and destination.is_file() and destination.stat().st_size > 0:
            path = destination
        else:
            temporary = registry.register(_temporary_path(destination))
            with _translated("descarga de imagen"):
                _ = downloader.download_video(url=image, destination=temporary)
                _ = temporary.replace(destination)
            path = destination
    if not path.is_file():
        msg = f"la imagen del slideshow no existe: {path}"
        raise PipelineError(msg)
    return path


def _write_concat_file(
    images: Sequence[Path],
    *,
    slide_duration_s: float,
    output_dir: Path,
    registry: _CleanupRegistry,
) -> Path:
    destination = registry.register(_temporary_path(output_dir / _SLIDESHOW_CONCAT_NAME))
    content = _concat_file_content(images, slide_duration_s=slide_duration_s)
    with _translated("archivo de concatenación del slideshow"):
        _ = destination.write_text(content, encoding="utf-8")
    return destination


def _concat_file_content(images: Sequence[Path], *, slide_duration_s: float) -> str:
    """Construye el contenido del archivo ``concat`` del demuxer de ffmpeg.

    Cada imagen precede a su directiva ``duration`` y la última se repite al
    final: el demuxer solo retiene el último frame si vuelve a aparecer.

    Args:
        images: Imágenes en orden de montaje.
        slide_duration_s: Duración de cada slide en segundos.

    Returns:
        El contenido del archivo, terminado en salto de línea.
    """
    duration = _seconds(slide_duration_s)
    lines: list[str] = []
    for image in images:
        lines.extend((_concat_file_line(image), f"duration {duration}"))
    lines.append(_concat_file_line(images[-1]))
    return "\n".join(lines) + "\n"


def _concat_file_line(path: Path) -> str:
    # El demuxer exige comillas simples y escapa una comilla literal como
    # '\'' (cierra, escapa y reabre); las rutas usan separadores POSIX.
    escaped = path.as_posix().replace("'", "'\\''")
    return f"file '{escaped}'"


def _assemble_slideshow(
    concat: Path,
    *,
    total_duration_s: float | None = None,
    render: RenderConfig,
    output_dir: Path,
    registry: _CleanupRegistry,
) -> Path:
    destination = registry.register(_temporary_path(output_dir / _SLIDESHOW_NAME))
    argv = [
        render.ffmpeg,
        "-hide_banner",
        "-nostdin",
        "-v",
        "error",
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(concat),
    ]
    if total_duration_s is not None:
        argv.extend(["-t", _seconds(total_duration_s)])
    argv.extend(
        [
            "-vf",
            _SLIDESHOW_FILTER,
            "-r",
            _SLIDESHOW_FPS,
        ]
    )
    argv += list(video_encoder_arguments(nvenc_available=render.nvenc_available))
    argv.append(str(destination))
    with _translated("ensamblado del slideshow"):
        _run_ffmpeg(argv, render=render)
    return destination


def _is_url(value: str) -> bool:
    return value.startswith(("http://", "https://"))


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
        transcriber=(
            FasterWhisperTranscriber(language=config.contract.languages.language)
            if transcriber is None
            else transcriber
        ),
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


def _resolve_source_stage(
    url: str,
    *,
    config: PipelineConfig,
    downloader: MediaDownloader,
    registry: _CleanupRegistry,
    state: PipelineStateManager,
    resume: bool,
) -> Path:
    destination = config.output_dir / _SOURCE_NAME
    if resume and state.is_done(PipelineStage.DOWNLOAD):
        source_path = state.artifact_path(PipelineStage.DOWNLOAD) or destination
        if source_path.is_file() and source_path.stat().st_size > 0:
            if _is_url(url):
                return _revalidate_remote_source(
                    url,
                    source_path,
                    config=config,
                    downloader=downloader,
                    registry=registry,
                )
            return source_path
    try:
        source = _download(url, config=config, downloader=downloader, registry=registry)
        state.mark_done(PipelineStage.DOWNLOAD, source)
    except Exception:
        state.mark_failed(PipelineStage.DOWNLOAD)
        raise
    else:
        return source


def _revalidate_remote_source(
    url: str,
    source_path: Path,
    *,
    downloader: MediaDownloader,
    registry: _CleanupRegistry,
    config: PipelineConfig | None = None,
) -> Path:
    """Re-descarga la fuente remota y actualiza el local solo si cambió.

    El servidor es la fuente de verdad: si los bytes difieren, el archivo
    local se reemplaza y la firma de contenido posterior invalida las etapas
    hijas. Si la revalidación falla, se lanza ``PipelineError`` (fail-closed):
    una red inestable nunca publica bytes obsoletos sin conocimiento del operador.

    Args:
        url: URL http/https del vídeo fuente.
        source_path: Archivo local descargado en una corrida previa.
        downloader: Descargador acotado para re-descargar los bytes.
        registry: Registro donde se anota el temporal de revalidación.
        config: Configuración del pipeline con contrato y render (opcional).

    Returns:
        La ruta del archivo fuente local (actualizado).

    Raises:
        PipelineError: Si la re-descarga o la verificación del hash falla.
    """
    temporary = registry.register(_temporary_path(source_path))
    with _translated("revalidación de la fuente"):
        if config is not None and config.contract.timestamp_ranges:
            window_start = min(tr.start_sec for tr in config.contract.timestamp_ranges)
            window_end = max(tr.end_sec for tr in config.contract.timestamp_ranges)
            actual_download_start = max(0.0, window_start - 10.0)
            margin_dest = registry.register(
                _temporary_path(source_path.with_name("margin_source.mp4"))
            )
            _ = downloader.download_video(
                url=url,
                destination=margin_dest,
                section=(window_start, window_end),
            )
            cut_start = window_start - actual_download_start
            cut_duration = window_end - window_start
            _ = _cut_exact_ffmpeg(
                source=margin_dest,
                destination=temporary,
                start_s=cut_start,
                duration_s=cut_duration,
                render=config.render,
            )
        else:
            _ = downloader.download_video(url=url, destination=temporary)
    try:
        changed = sha256_file(temporary) != sha256_file(source_path)
    except OSError as error:
        msg = f"no se pudo verificar la fuente revalidada {source_path}: {error}"
        raise PipelineError(msg) from error
    if changed:
        logger.warning("La fuente remota cambió; actualizando el archivo local")
        _ = temporary.replace(source_path)
    return source_path


def _revalidate_remote_audio_track(
    url: str,
    track_path: Path,
    *,
    downloader: MediaDownloader,
    registry: _CleanupRegistry,
) -> Path:
    """Re-descarga la pista remota y actualiza el local solo si cambió.

    Mismo patrón fail-closed que la fuente de video: si la re-descarga
    falla, se lanza ``PipelineError`` y nunca se reutilizan bytes obsoletos
    en silencio.

    Args:
        url: URL http/https de la pista de audio.
        track_path: Archivo local descargado en una corrida previa.
        downloader: Descargador acotado para re-descargar los bytes.
        registry: Registro donde se anota el temporal de revalidación.

    Returns:
        La ruta de la pista local (actualizada).

    Raises:
        PipelineError: Si la re-descarga o la verificación del hash falla.
    """
    temporary = registry.register(_temporary_path(track_path))
    with _translated("revalidación del audio"):
        _ = downloader.download_video(url=url, destination=temporary)
    try:
        changed = sha256_file(temporary) != sha256_file(track_path)
    except OSError as error:
        msg = f"no se pudo verificar el audio revalidado {track_path}: {error}"
        raise PipelineError(msg) from error
    if changed:
        logger.warning("La pista remota cambió; actualizando el archivo local")
        _ = temporary.replace(track_path)
    return track_path


def _audio_track_signature(config: PipelineConfig) -> str | None:
    """Calcula la firma de contenido de la pista externa para --resume.

    Incluye el hash del archivo local (``audio_track_path`` o el
    ``audio_track.mp3`` descargado) para que un cambio de bytes invalide
    las etapas dependientes aunque la URL sea idéntica.

    Args:
        config: Configuración con la pista local o remota.

    Returns:
        La firma ``ruta:sha256`` del archivo, la URL si aún no se descargó,
        o None si no hay pista configurada.
    """
    if config.audio_track_path is not None:
        return _file_signature(config.audio_track_path)
    if config.audio_track_url is not None:
        downloaded = config.output_dir / _AUDIO_TRACK_NAME
        if downloaded.is_file():
            return _file_signature(downloaded)
        return config.audio_track_url
    return None


def _local_copy_needs_update(*, src: Path, dst: Path) -> bool:
    """Indica si la copia en el work_dir difiere de la fuente local.

    Args:
        src: Pista local de origen.
        dst: Copia en el work_dir.

    Returns:
        True si la copia falta, difiere o no se puede verificar (fail-closed
        hacia la copia).
    """
    if not dst.is_file():
        return True
    try:
        return sha256_file(dst) != sha256_file(src)
    except OSError:
        return True


def _sync_local_audio_copy(*, config: PipelineConfig) -> None:
    """Sincroniza la pista local en ``output_dir/audio_track.mp3``.

    Mantiene una copia en el work_dir para que ``--resume`` pueda revalidar
    sin depender solo del fingerprint global; si los bytes difieren, la copia
    se actualiza. No se registra como temporal para que la limpieza no la
    elimine, igual que la pista descargada por URL.

    Args:
        config: Configuración con la pista local.

    Raises:
        PipelineError: Si la copia no se puede actualizar.
    """
    if config.audio_track_path is None:
        return
    src = config.audio_track_path
    if not src.is_file():
        return
    dst = config.output_dir / _AUDIO_TRACK_NAME
    if not _local_copy_needs_update(src=src, dst=dst):
        return
    try:
        dst.parent.mkdir(parents=True, exist_ok=True)
        _ = shutil.copyfile(src, dst)
    except OSError as error:
        msg = f"no se pudo sincronizar la pista local en {dst}: {error}"
        raise PipelineError(msg) from error


def _check_audio_content_signature(
    *,
    config: PipelineConfig,
    state: PipelineStateManager,
    resume: bool,
) -> bool:
    """Compara la firma de la pista con el checkpoint e invalida si cambió.

    Para ``audio_track_path`` local siempre se recalcula el hash actual y se
    sincroniza la copia en el work_dir; si difiere del checkpoint o el
    checkpoint no tiene hash (fail-closed), se invalidan las etapas
    dependientes.

    Args:
        config: Configuración con la pista externa.
        state: Administrador del checkpoint persistente.
        resume: Si es True, se compara con el hash almacenado.

    Returns:
        True si se puede continuar con resume; False si el audio cambió y
        se invalidaron las etapas dependientes.
    """
    if not config.audio_locked:
        return resume
    if config.audio_track_path is not None:
        _sync_local_audio_copy(config=config)
    current = _audio_track_signature(config)
    if current is None:
        return resume
    stored = state.checkpoint.audio_content_hash
    if resume and stored != current:
        logger.warning("El audio externo cambió; invalidando etapas dependientes...")
        state.invalidate_audio_dependents()
        for pattern in ("final*.mp4", "audio_injected*.mp4"):
            for path in config.output_dir.glob(pattern):
                with suppress(OSError):
                    path.unlink(missing_ok=True)
        resume = False
    state.set_audio_content_hash(current)
    return resume


def _resolve_transcript_stage(
    source: Path,
    *,
    config: PipelineConfig,
    transcriber: Transcriber,
    state: PipelineStateManager,
    resume: bool,
) -> Transcript:
    artifact = config.output_dir / "transcript.json"
    if resume and (state.is_done(PipelineStage.TRANSCRIBE) or artifact.is_file()):
        artifact_file = state.artifact_path(PipelineStage.TRANSCRIBE) or artifact
        if artifact_file.is_file() and artifact_file.stat().st_size > 0:
            try:
                transcript = Transcript.model_validate_json(
                    artifact_file.read_text(encoding="utf-8")
                )
            except (ValidationError, ValueError, json.JSONDecodeError):
                pass
            else:
                if not state.is_done(PipelineStage.TRANSCRIBE):
                    state.mark_done(PipelineStage.TRANSCRIBE, artifact_file)
                return transcript
    try:
        transcript = _transcribe(source, transcriber=transcriber)
        _atomic_write_json(artifact, transcript.model_dump_json(indent=2))
        state.mark_done(PipelineStage.TRANSCRIBE, artifact)
    except Exception:
        state.mark_failed(PipelineStage.TRANSCRIBE)
        raise
    else:
        return transcript


def _resolve_moments_stage(
    source: Path,
    *,
    transcript: Transcript,
    config: PipelineConfig,
    detector: MomentDetector,
    state: PipelineStateManager,
    resume: bool,
) -> tuple[Moment, ...]:
    artifact = config.output_dir / "moments.json"
    adapter = TypeAdapter(tuple[Moment, ...])
    if resume and (state.is_done(PipelineStage.MOMENTS) or artifact.is_file()):
        artifact_file = state.artifact_path(PipelineStage.MOMENTS) or artifact
        if artifact_file.is_file() and artifact_file.stat().st_size > 0:
            try:
                moments = adapter.validate_json(artifact_file.read_text(encoding="utf-8"))
            except (ValidationError, ValueError, json.JSONDecodeError):
                pass
            else:
                if not state.is_done(PipelineStage.MOMENTS):
                    state.mark_done(PipelineStage.MOMENTS, artifact_file)
                return moments
    try:
        moments = _detect(source, transcript=transcript, detector=detector)
        _atomic_write_json(artifact, adapter.dump_json(moments, indent=2).decode("utf-8"))
        state.mark_done(PipelineStage.MOMENTS, artifact)
    except Exception:
        state.mark_failed(PipelineStage.MOMENTS)
        raise
    else:
        return moments


def _source_time_offset_sec(config: PipelineConfig) -> float:
    if config.contract.timestamp_ranges:
        return min(tr.start_sec for tr in config.contract.timestamp_ranges)
    return 0.0


def _contract_with_relative_timestamps(contract: Contract, *, offset_sec: float) -> Contract:
    if math.isclose(offset_sec, 0.0, abs_tol=1e-6) or not contract.timestamp_ranges:
        return contract
    rel_ranges = tuple(
        TimestampRange(
            start_sec=max(0.0, tr.start_sec - offset_sec),
            end_sec=max(0.0, tr.end_sec - offset_sec),
        )
        for tr in contract.timestamp_ranges
    )
    return contract.model_copy(update={"timestamp_ranges": rel_ranges})


def _resolve_selection_stage(
    *,
    transcript: Transcript,
    moments: tuple[Moment, ...],
    config: PipelineConfig,
    model: LongVideoModel,
    selector: SegmentSelector,
    state: PipelineStateManager,
    resume: bool,
) -> SegmentSelection:
    artifact = config.output_dir / "selection.json"
    if resume and (state.is_done(PipelineStage.SELECT) or artifact.is_file()):
        artifact_file = state.artifact_path(PipelineStage.SELECT) or artifact
        if artifact_file.is_file() and artifact_file.stat().st_size > 0:
            try:
                selection = SegmentSelection.model_validate_json(
                    artifact_file.read_text(encoding="utf-8")
                )
            except (ValidationError, ValueError, json.JSONDecodeError):
                pass
            else:
                if not state.is_done(PipelineStage.SELECT):
                    state.mark_done(PipelineStage.SELECT, artifact_file)
                return selection
    try:
        rel_contract = _contract_with_relative_timestamps(
            config.contract, offset_sec=_source_time_offset_sec(config)
        )
        selection = _select(
            transcript,
            moments,
            contract=rel_contract,
            model=model,
            selector=selector,
        )
        _atomic_write_json(artifact, selection.model_dump_json(indent=2))
        state.mark_done(PipelineStage.SELECT, artifact)
    except Exception:
        state.mark_failed(PipelineStage.SELECT)
        raise
    else:
        return selection


def _resolve_intelligence_stages(
    source: Path,
    *,
    model: LongVideoModel,
    config: PipelineConfig,
    dependencies: _Dependencies,
    state: PipelineStateManager,
    resume: bool,
) -> tuple[Transcript | None, tuple[Moment, ...], SegmentSelection]:
    if config.repost_mode:
        return None, (), _full_video_selection(source, render=config.render)
    transcript = _resolve_transcript_stage(
        source,
        config=config,
        transcriber=dependencies.transcriber,
        state=state,
        resume=resume,
    )
    moments = _resolve_moments_stage(
        source,
        transcript=transcript,
        config=config,
        detector=dependencies.detector,
        state=state,
        resume=resume,
    )
    selection = _resolve_selection_stage(
        transcript=transcript,
        moments=moments,
        config=config,
        model=model,
        selector=dependencies.selector,
        state=state,
        resume=resume,
    )
    return transcript, moments, selection


def _resolve_clip(
    source: Path,
    *,
    segment: Segment,
    config: PipelineConfig,
    registry: _CleanupRegistry,
    suffix: str = "",
) -> Path:
    passthrough = (
        config.repost_mode and not config.audio_locked and not _needs_reframe(source, config.render)
    )
    if passthrough:
        return _passthrough(
            source,
            output_dir=config.output_dir,
            render=config.render,
            registry=registry,
        )
    return _cut_segment(
        source,
        segment=segment,
        render=config.render,
        registry=registry,
        suffix=suffix,
    )


def _resolve_reframe_stage(
    clip: Path,
    *,
    config: PipelineConfig,
    dependencies: _Dependencies,
    registry: _CleanupRegistry,
    state: PipelineStateManager,
    resume: bool = False,
    suffix: str = "",
    is_primary: bool = True,
) -> tuple[ReframeResult | None, Path]:
    if config.repost_mode and not _needs_reframe(clip, config.render):
        return None, clip
    reframe_json = config.output_dir / f"reframe{suffix}.json"
    expected_reframed = config.output_dir / f"reframed{suffix}.mp4"
    recorded = state.artifact_path(PipelineStage.REFRAME) if is_primary else None
    reframed_video = _recorded_or_expected(recorded, expected_reframed)
    subtitles_already_done = is_primary and (
        state.is_done(PipelineStage.SUBTITLES) and (config.output_dir / _FINAL_NAME).is_file()
    )
    video_valid = reframed_video.is_file() and reframed_video.stat().st_size > 0
    json_valid = reframe_json.is_file() and reframe_json.stat().st_size > 0
    can_reuse = (video_valid or subtitles_already_done) and json_valid
    if resume and (state.is_done(PipelineStage.REFRAME) if is_primary else True) and can_reuse:
        try:
            reframe = ReframeResult.model_validate_json(reframe_json.read_text(encoding="utf-8"))
        except (ValidationError, ValueError, json.JSONDecodeError):
            pass
        else:
            if video_valid:
                _ = registry.register(reframed_video, is_artifact=True)
            return reframe, reframed_video
    try:
        destination = config.output_dir / f"reframed{suffix}.mp4"
        reframe, reframed = _reframe(
            clip,
            reframer=_require_reframer(dependencies),
            destination=destination,
            registry=registry,
        )
        _atomic_write_json(reframe_json, reframe.model_dump_json(indent=2))
        if is_primary:
            state.mark_done(PipelineStage.REFRAME, reframed)
    except Exception:
        if is_primary:
            state.mark_failed(PipelineStage.REFRAME)
        raise
    else:
        return reframe, reframed


def _resolve_subtitle_file(
    *,
    is_lyric: bool,
    transcript: Transcript | None,
    segment: Segment,
    dependencies: _Dependencies,
    config: PipelineConfig,
    registry: _CleanupRegistry,
    subtitles_name: str,
    source_offset_sec: float = 0.0,
) -> Path:
    if is_lyric:
        return _write_lyric_subtitles(
            segment=segment,
            config=config,
            output_dir=config.output_dir,
            registry=registry,
            subtitles_name=subtitles_name,
            source_offset_sec=source_offset_sec,
        )
    if transcript is None:
        msg = "no hay transcripción para generar subtítulos"
        raise PipelineError(msg)
    return _write_subtitles(
        transcript,
        segment=segment,
        renderer=dependencies.subtitle_renderer,
        output_dir=config.output_dir,
        registry=registry,
        subtitles_name=subtitles_name,
    )


def _resolve_subtitles_and_burn_stage(
    video: Path,
    *,
    transcript: Transcript | None,
    segment: Segment,
    dependencies: _Dependencies,
    config: PipelineConfig,
    registry: _CleanupRegistry,
    state: PipelineStateManager,
    resume: bool = False,
    suffix: str = "",
    is_primary: bool = True,
    source_offset_sec: float = 0.0,
) -> tuple[Path | None, Path]:
    final_name = f"final{suffix}.mp4"
    subtitles_name = f"subtitles{suffix}.ass"
    mute_audio = contract_mutes_audio(config.contract)
    is_lyric = config.contract.format is Format.LYRIC_VIDEO
    if transcript is None and not is_lyric:
        final_video = _publish(
            video,
            output_dir=config.output_dir,
            registry=registry,
            final_name=final_name,
            mute_audio=mute_audio,
            render=config.render,
        )
        return None, final_video
    expected_final = config.output_dir / final_name
    recorded = state.artifact_path(PipelineStage.SUBTITLES) if is_primary else None
    final_video = _recorded_or_expected(recorded, expected_final)
    if (
        resume
        and (state.is_done(PipelineStage.SUBTITLES) if is_primary else True)
        and final_video.is_file()
        and final_video.stat().st_size > 0
    ):
        cached_subtitles = config.output_dir / subtitles_name
        return cached_subtitles, final_video
    try:
        subtitles = _resolve_subtitle_file(
            is_lyric=is_lyric,
            transcript=transcript,
            segment=segment,
            dependencies=dependencies,
            config=config,
            registry=registry,
            subtitles_name=subtitles_name,
            source_offset_sec=source_offset_sec,
        )
        final = _burn(
            video,
            subtitles=subtitles,
            renderer=dependencies.subtitle_renderer,
            output_dir=config.output_dir,
            final_name=final_name,
            mute_audio=mute_audio,
        )
        if is_primary:
            state.mark_done(PipelineStage.SUBTITLES, final)
    except Exception:
        if is_primary:
            state.mark_failed(PipelineStage.SUBTITLES)
        raise
    else:
        return subtitles, final


def _render_segment(
    source: Path,
    *,
    segment: Segment,
    transcript: Transcript | None,
    config: PipelineConfig,
    dependencies: _Dependencies,
    registry: _CleanupRegistry,
    state: PipelineStateManager,
    resume: bool,
    suffix: str,
    is_primary: bool,
    source_offset_sec: float = 0.0,
) -> tuple[ReframeResult | None, Path | None, Path]:
    clip = _resolve_clip(
        source,
        segment=segment,
        config=config,
        registry=registry,
        suffix=suffix,
    )
    reframe, reframed = _resolve_reframe_stage(
        clip,
        config=config,
        dependencies=dependencies,
        registry=registry,
        state=state,
        resume=resume,
        suffix=suffix,
        is_primary=is_primary,
    )
    with_audio = (
        _inject_audio(
            reframed,
            config=config,
            downloader=dependencies.downloader,
            registry=registry,
            resume=resume,
            suffix=suffix,
            state=state,
        )
        if config.audio_locked
        else reframed
    )
    subtitles, final_video = _resolve_subtitles_and_burn_stage(
        with_audio,
        transcript=transcript,
        segment=segment,
        dependencies=dependencies,
        config=config,
        registry=registry,
        state=state,
        resume=resume,
        suffix=suffix,
        is_primary=is_primary,
        source_offset_sec=source_offset_sec,
    )
    return reframe, subtitles, final_video


def _check_source_content_signature(
    source: Path,
    *,
    config: PipelineConfig,
    state: PipelineStateManager,
    resume: bool,
) -> bool:
    source_sig = _file_signature(source)
    if (
        resume
        and state.checkpoint.source_content_hash is not None
        and state.checkpoint.source_content_hash != source_sig
    ):
        logger.warning("El contenido del video fuente cambió; invalidando etapas dependientes...")
        state.invalidate_downstream_stages()
        patterns = (
            "transcript.json",
            "moments.json",
            "selection.json",
            "reframed*.mp4",
            "reframe*.json",
            "subtitles*.ass",
            "final*.mp4",
            "clip*.mp4",
        )
        for pattern in patterns:
            for path in config.output_dir.glob(pattern):
                with suppress(OSError):
                    path.unlink(missing_ok=True)
        resume = False
    state.set_source_content_hash(source_sig)
    return resume


def _clean_leftover_segments(output_dir: Path, total_segments: int) -> None:
    for final_file in output_dir.glob("final_*.mp4"):
        stem = final_file.stem
        try:
            num = int(stem.split("_")[-1])
        except (ValueError, IndexError):
            continue
        if num >= total_segments:
            with suppress(OSError):
                final_file.unlink()
            patterns = (
                f"reframed_{num:02d}.mp4",
                f"subtitles_{num:02d}.ass",
                f"reframe_{num:02d}.json",
                f"clip_{num:02d}.mp4",
            )
            for pattern in patterns:
                with suppress(OSError):
                    (output_dir / pattern).unlink(missing_ok=True)


def _invalidate_segment_artifacts(output_dir: Path, suffix: str) -> None:
    patterns = (
        f"final{suffix}.mp4",
        f"reframed{suffix}.mp4",
        f"subtitles{suffix}.ass",
        f"reframe{suffix}.json",
        f"clip{suffix}.mp4",
    )
    for pattern in patterns:
        with suppress(OSError):
            (output_dir / pattern).unlink(missing_ok=True)


def _recorded_or_expected(recorded: Path | None, expected: Path) -> Path:
    """Devuelve el artefacto del checkpoint solo si coincide con el esperado.

    Al cambiar la cardinalidad del lote (``final.mp4`` vs ``final_00.mp4``),
    la ruta registrada pertenece al esquema viejo y debe ignorarse para que
    ``final_videos`` nunca mezcle ambos esquemas.

    Args:
        recorded: Ruta registrada en el checkpoint, o None.
        expected: Ruta esperada para la cardinalidad vigente.

    Returns:
        La ruta registrada si coincide con la esperada; la esperada si no.
    """
    if recorded is not None and recorded == expected:
        return recorded
    return expected


def _normalize_final_artifacts(output_dir: Path, total_segments: int) -> None:
    """Elimina finales con el esquema de nombres de la otra cardinalidad.

    Un lote de 1 segmento publica ``final.mp4`` y un lote múltiple publica
    ``final_00.mp4``...: al cambiar la cardinalidad en ``--resume``, el
    artefacto con el nombre viejo se elimina para que ``final_videos`` nunca
    mezcle ambos esquemas. La limpieza opera en ambas direcciones: de
    múltiple a 1 se eliminan todos los ``final_NN.mp4`` indexados (que
    ``_clean_leftover_segments`` conserva parcialmente), y de 1 a múltiple se
    eliminan los artefactos de nombre singular.

    Args:
        output_dir: Directorio de salida del pipeline.
        total_segments: Número de segmentos de la selección vigente.
    """
    if total_segments > 1:
        for name in ("final.mp4", "reframed.mp4", "reframe.json", "subtitles.ass"):
            with suppress(OSError):
                (output_dir / name).unlink(missing_ok=True)
        return
    for pattern in (
        "final_*.mp4",
        "reframed_*.mp4",
        "subtitles_*.ass",
        "reframe_*.json",
        "clip_*.mp4",
    ):
        for stale in output_dir.glob(pattern):
            with suppress(OSError):
                stale.unlink()


def _render_all_segments(
    source: Path,
    selection: SegmentSelection,
    *,
    transcript: Transcript | None,
    config: PipelineConfig,
    dependencies: _Dependencies,
    registry: _CleanupRegistry,
    state: PipelineStateManager,
    effective_resume: bool,
    source_offset_sec: float = 0.0,
) -> tuple[ReframeResult | None, Path | None, tuple[Path, ...]]:
    _ = _primary_segment(selection)
    total_segments = len(selection.segments)
    final_videos: list[Path] = []
    primary_reframe: ReframeResult | None = None
    primary_subtitles: Path | None = None

    for idx, segment in enumerate(selection.segments):
        suffix = f"_{idx:02d}" if total_segments > 1 else ""
        coords_match = (
            effective_resume
            and idx < len(state.checkpoint.segment_coords)
            and state.checkpoint.segment_coords[idx] == (segment.start_s, segment.end_s)
        )
        if effective_resume and not coords_match:
            _invalidate_segment_artifacts(config.output_dir, suffix)

        reframe, subtitles, final_video = _render_segment(
            source,
            segment=segment,
            transcript=transcript,
            config=config,
            dependencies=dependencies,
            registry=registry,
            state=state,
            resume=effective_resume and coords_match,
            suffix=suffix,
            is_primary=(idx == 0),
            source_offset_sec=source_offset_sec,
        )
        if idx == 0:
            primary_reframe = reframe
            primary_subtitles = subtitles
        final_videos.append(final_video)

    state.set_segment_coords(tuple((s.start_s, s.end_s) for s in selection.segments))
    return primary_reframe, primary_subtitles, tuple(final_videos)


def _run_stages(
    url: str,
    *,
    model: LongVideoModel,
    config: PipelineConfig,
    dependencies: _Dependencies,
    registry: _CleanupRegistry,
    state: PipelineStateManager,
    resume: bool = False,
) -> PipelineResult:
    source = _resolve_source_stage(
        url,
        config=config,
        downloader=dependencies.downloader,
        registry=registry,
        state=state,
        resume=resume,
    )
    effective_resume = _check_source_content_signature(
        source, config=config, state=state, resume=resume
    )
    transcript, moments, selection = _resolve_intelligence_stages(
        source,
        model=model,
        config=config,
        dependencies=dependencies,
        state=state,
        resume=effective_resume,
    )
    _clean_leftover_segments(config.output_dir, len(selection.segments))
    _normalize_final_artifacts(config.output_dir, len(selection.segments))
    source_offset_sec = _source_time_offset_sec(config)
    reframe, subtitles, final_videos = _render_all_segments(
        source,
        selection,
        transcript=transcript,
        config=config,
        dependencies=dependencies,
        registry=registry,
        state=state,
        effective_resume=effective_resume,
        source_offset_sec=source_offset_sec,
    )

    return PipelineResult(
        source=source,
        transcript=transcript,
        moments=moments,
        selection=selection,
        reframe=reframe,
        subtitles=subtitles,
        final_video=final_videos[0],
        cleaning=(),
        final_videos=final_videos,
        source_offset_sec=source_offset_sec,
    )


def _cut_exact_ffmpeg(
    *,
    source: Path,
    destination: Path,
    start_s: float,
    duration_s: float,
    render: RenderConfig,
) -> Path:
    argv = [
        render.ffmpeg,
        "-hide_banner",
        "-nostdin",
        "-v",
        "error",
        "-y",
        "-ss",
        _seconds(start_s),
        "-t",
        _seconds(duration_s),
        "-i",
        str(source),
        "-vf",
        "setpts=PTS-STARTPTS",
        "-map",
        "0:v:0",
        "-map",
        "0:a?",
    ]
    argv += list(video_encoder_arguments(nvenc_available=render.nvenc_available))
    argv += list(audio_and_container_arguments())
    argv.append(str(destination))
    with _translated("corte exacto de la fuente"):
        _run_ffmpeg(argv, render=render)
    info = _probe_video(destination, render=render)
    if info.duration_s <= 0.0 or abs(info.duration_s - duration_s) > _CUT_DURATION_TOLERANCE_S:
        msg = (
            f"el corte exacto produjo una duración inválida ({info.duration_s:.3f} s), "
            f"se esperaban {duration_s:.3f} s (tolerancia {_CUT_DURATION_TOLERANCE_S} s)"
        )
        raise PipelineError(msg)
    return destination


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
        if config.contract.timestamp_ranges:
            window_start = min(tr.start_sec for tr in config.contract.timestamp_ranges)
            window_end = max(tr.end_sec for tr in config.contract.timestamp_ranges)
            actual_download_start = max(0.0, window_start - 10.0)
            margin_dest = registry.register(_temporary_path(source.with_name("margin_source.mp4")))
            _ = downloader.download_video(
                url=url,
                destination=margin_dest,
                section=(window_start, window_end),
            )
            cut_start = window_start - actual_download_start
            cut_duration = window_end - window_start
            _ = _cut_exact_ffmpeg(
                source=margin_dest,
                destination=temporary,
                start_s=cut_start,
                duration_s=cut_duration,
                render=config.render,
            )
        else:
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


def _probe_audio_duration(audio: Path, *, render: RenderConfig) -> float:
    """Lee la duración de una pista de audio con ffprobe.

    Args:
        audio: Ruta de la pista de audio a sondear.
        render: Configuración con el timeout del sondeo.

    Returns:
        La duración del audio en segundos.

    Raises:
        PipelineError: Si ffprobe no está disponible, falla, expira o no se
            puede leer la duración.
    """
    argv = [
        _FFPROBE,
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1",
        str(audio),
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
    return _probe_duration(fields.get("duration"), video=audio)


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


def _publish(
    video: Path,
    *,
    output_dir: Path,
    registry: _CleanupRegistry,
    final_name: str = _FINAL_NAME,
    mute_audio: bool = False,
    render: RenderConfig | None = None,
) -> Path:
    """Publica el vídeo procesado como artefacto final sin subtítulos.

    Con ``mute_audio`` el vídeo no se copia: se re-publica con ffmpeg
    copiando el vídeo y silenciando el audio (``volume=0``), de modo que la
    pista sigue presente en el contenedor pero en silencio digital. Sin
    ``mute_audio`` conserva la copia binaria histórica.

    Args:
        video: Vídeo procesado (cortado, reframeado y/o con audio inyectado).
        output_dir: Directorio donde se publica ``final.mp4``.
        registry: Registro del temporal de publicación.
        final_name: Nombre del archivo de video final publicado.
        mute_audio: Si es True, silencia la pista sin eliminarla
            (``audio_policy=internal_official_sound``).
        render: Binario y timeout de ffmpeg; exigido cuando ``mute_audio``.

    Returns:
        La ruta del artefacto final.

    Raises:
        PipelineError: Si no se puede copiar o publicar el artefacto, si
            falta ``render`` para silenciar, o si ffmpeg falla.
    """
    destination = output_dir / final_name
    temporary = registry.register(_temporary_path(destination))
    if mute_audio:
        if render is None:
            msg = "el silenciado del audio final requiere la configuración de render"
            raise PipelineError(msg)
        argv = [
            render.ffmpeg,
            "-hide_banner",
            "-nostdin",
            "-v",
            "error",
            "-y",
            "-i",
            str(video),
            "-map",
            "0:v:0",
            "-map",
            "0:a?",
            "-c:v",
            "copy",
            *muted_audio_arguments(),
            *audio_and_container_arguments(),
            str(temporary),
        ]
        with _translated("silenciado del audio final"):
            _run_ffmpeg(argv, render=render)
            _ = temporary.replace(destination)
        return destination
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
    suffix: str = "",
) -> Path:
    destination = registry.register(_temporary_path(source.with_name(f"segment{suffix}.mp4")))
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
    expected_duration = segment.end_s - segment.start_s
    with _translated("corte del segmento"):
        _run_ffmpeg(argv, render=render)
    info = _probe_video(destination, render=render)
    if (
        info.duration_s <= 0.0
        or abs(info.duration_s - expected_duration) > _CUT_DURATION_TOLERANCE_S
    ):
        msg = (
            f"el corte del segmento produjo una duración inválida ({info.duration_s:.3f} s), "
            f"se esperaban {expected_duration:.3f} s (tolerancia {_CUT_DURATION_TOLERANCE_S} s)"
        )
        raise PipelineError(msg)
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
    destination: Path,
    registry: _CleanupRegistry,
) -> tuple[ReframeResult, Path]:
    temporary = registry.register(_temporary_path(destination))
    with _translated("reframe 9:16"):
        result = reframer.analyze(clip)
        _ = reframer.render(video=clip, destination=temporary, result=result)
        _ = temporary.replace(destination)
    _ = registry.register(destination, is_artifact=True)
    return result, destination


def _inject_audio(
    video: Path,
    *,
    config: PipelineConfig,
    downloader: MediaDownloader,
    registry: _CleanupRegistry,
    resume: bool = False,
    suffix: str = "",
    state: PipelineStateManager | None = None,
) -> Path:
    """Inyecta la pista externa reemplazando o mezclando la original.

    La pista se resuelve desde ``audio_track_path`` o ``audio_track_url`` y el
    render escribe en un temporal hermano registrado para limpieza. El vídeo se
    copia y solo se recodifica el audio. Tras resolver, la firma de contenido
    se compara con el checkpoint: si cambió, se invalidan los finales.

    Args:
        video: Vídeo reframeado sin la pista externa.
        config: Configuración con la pista, la proporción y el render.
        downloader: Descargador acotado para pistas entregadas por URL.
        registry: Registro de temporales para limpiar la pista y el resultado.
        resume: Si es True, revalida la pista ya descargada si existe.
        suffix: Sufijo opcional para nombres de archivo en lote.
        state: Checkpoint para la firma de contenido del audio.

    Returns:
        La ruta del vídeo con la pista externa inyectada.

    Raises:
        PipelineError: Si la pista no existe, no se configuró ninguna, la
            combinación con la política de audio es contradictoria o ffmpeg
            falla.
    """
    _validate_audio_combination(config)
    track = _resolve_audio_track(
        config, downloader=downloader, registry=registry, resume=resume, state=state
    )
    if state is not None:
        _ = _check_audio_content_signature(config=config, state=state, resume=resume)
    video_info = _probe_video(video, render=config.render)
    audio_duration_s = _probe_audio_duration(track, render=config.render)
    if audio_duration_s < video_info.duration_s - _AUDIO_DURATION_TOLERANCE_S:
        msg = "El audio externo es más corto que el video"
        raise PipelineError(msg)
    destination = registry.register(_temporary_path(video.with_name(f"audio_injected{suffix}.mp4")))
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
    argv += list(
        audio_injection_arguments(
            mix_ratio=config.audio_mix_ratio,
            video_duration_s=video_info.duration_s,
        )
    )
    argv.append(str(destination))
    with _translated("inyección de audio"):
        _run_ffmpeg(argv, render=config.render)
    return destination


def _resolve_audio_track(
    config: PipelineConfig,
    *,
    downloader: MediaDownloader,
    registry: _CleanupRegistry,
    resume: bool = False,
    state: PipelineStateManager | None = None,
) -> Path:
    """Resuelve la pista de audio externa, descargándola si llega por URL.

    La pista descargada persiste en ``output_dir`` (como ``source.mp4``)
    para permitir la revalidación en ``--resume``: no se registra como
    artefacto temporal para que la limpieza final no la elimine. En resume,
    una URL remota se re-descarga y compara (fail-closed, mismo patrón que
    la fuente de video); si los bytes cambiaron, el local se actualiza.

    Args:
        config: Configuración con la ruta local o la URL de la pista.
        downloader: Descargador acotado que se usa cuando la pista es una URL.
        registry: Registro donde se anota el temporal de descarga.
        resume: Si es True, revalida la pista ya descargada si existe.
        state: Administrador del checkpoint para saber si download terminó.

    Returns:
        La ruta local de la pista de audio.

    Raises:
        PipelineError: Si se configuraron ambas fuentes, ninguna, la ruta no
            existe o la revalidación remota falla.
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
        destination = config.output_dir / _AUDIO_TRACK_NAME
        download_done = state is not None and state.is_done(PipelineStage.DOWNLOAD)
        if (
            resume
            and destination.is_file()
            and destination.stat().st_size > 0
            and (state is None or download_done)
        ):
            return _revalidate_remote_audio_track(
                config.audio_track_url,
                destination,
                downloader=downloader,
                registry=registry,
            )
        temporary = registry.register(_temporary_path(destination))
        with _translated("descarga de audio"):
            _ = downloader.download_video(url=config.audio_track_url, destination=temporary)
            _ = temporary.replace(destination)
        return destination
    msg = "audio_locked requiere audio_track_path o audio_track_url"
    raise PipelineError(msg)


def _fetch_lrclib_lines(config: PipelineConfig) -> tuple[LyricLine, ...]:
    if not (config.contract.lyric_video and config.contract.lyric_video.lrclib_enabled):
        msg = "no se encontraron letras sincronizadas (.lrc no configurado o lrclib deshabilitado)"
        raise PipelineError(msg)
    if config.synced_lyrics_provider is None:
        msg = "no se encontraron letras sincronizadas (proveedor lrclib ausente)"
        raise PipelineError(msg)
    if not config.contract.lyric_video.track_name:
        msg = "se requiere track_name para consultar letras en lrclib"
        raise PipelineError(msg)
    try:
        content = config.synced_lyrics_provider.get_synced_lyrics(
            track_name=config.contract.lyric_video.track_name,
            artist_name=config.contract.lyric_video.artist_name,
        )
        return parse_lrc(content)
    except LyricsError as error:
        raise PipelineError(str(error)) from error


def _resolve_pipeline_lyric_lines(config: PipelineConfig) -> tuple[LyricLine, ...]:
    if config.lrc_path is not None:
        if not config.lrc_path.is_file():
            msg = f"el archivo .lrc especificado no existe: '{config.lrc_path}'"
            raise PipelineError(msg)
        try:
            content = config.lrc_path.read_text(encoding="utf-8")
            return parse_lrc(content)
        except (OSError, ValueError) as error:
            msg = f"no se pudo procesar el archivo .lrc en '{config.lrc_path}': {error}"
            raise PipelineError(msg) from error

    lyric_config = config.contract.lyric_video
    if lyric_config is not None and lyric_config.lrc_asset_id:
        if config.assets is None:
            msg = (
                f"no se puede resolver el asset de letras '{lyric_config.lrc_asset_id}': "
                "no hay AssetRegistry configurado"
            )
            raise PipelineError(msg)
        try:
            intact = config.assets.verify(lyric_config.lrc_asset_id)
        except (AssetError, OSError) as error:
            msg = f"error al verificar el asset de letras '{lyric_config.lrc_asset_id}': {error}"
            raise PipelineError(msg) from error
        if not intact:
            msg = (
                f"asset de letras '{lyric_config.lrc_asset_id}' "
                "ausente o con integridad comprometida"
            )
            raise PipelineError(msg)
        try:
            lrc_path = config.assets.path_for(lyric_config.lrc_asset_id)
            content = lrc_path.read_text(encoding="utf-8")
            return parse_lrc(content)
        except (AssetError, OSError, ValueError) as error:
            msg = f"no se pudo procesar el asset .lrc '{lyric_config.lrc_asset_id}': {error}"
            raise PipelineError(msg) from error

    if lyric_config is not None:
        return _fetch_lrclib_lines(config)

    msg = "formato lyric_video sin archivo ni configuración de letras .lrc"
    raise PipelineError(msg)


def _write_lyric_subtitles(
    *,
    segment: Segment,
    config: PipelineConfig,
    output_dir: Path,
    registry: _CleanupRegistry,
    subtitles_name: str = "subtitles.ass",
    source_offset_sec: float = 0.0,
) -> Path:
    raw_lines = _resolve_pipeline_lyric_lines(config)
    # El segmento es relativo a la descarga quirúrgica (que arranca en 0.0 s);
    # el .lrc vive en coordenadas absolutas del vídeo original, así que el
    # recorte suma el desplazamiento. cut_lyric_window desplaza el resultado
    # a 0.0 s: el .ass quemado queda relativo al inicio del clip.
    absolute_start = segment.start_s + source_offset_sec
    absolute_end = segment.end_s + source_offset_sec
    try:
        window_lines = cut_lyric_window(
            raw_lines,
            start_sec=absolute_start,
            end_sec=absolute_end,
        )
    except (LrcEmptyWindowError, ValueError) as err:
        raise PipelineError(str(err)) from err

    duration_s = segment.end_s - segment.start_s
    ass_content = lyric_lines_to_ass(window_lines, duration_s=duration_s)
    destination = registry.register(_temporary_path(output_dir / subtitles_name))
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        _ = destination.write_text(ass_content, encoding="utf-8")
        target = output_dir / subtitles_name
        _ = destination.replace(target)
    except OSError as error:
        msg = f"no se pudo guardar el archivo .ass en {destination}: {error}"
        raise PipelineError(msg) from error
    else:
        return target


def _write_subtitles(
    transcript: Transcript,
    *,
    segment: Segment,
    renderer: SubtitleBurner,
    output_dir: Path,
    registry: _CleanupRegistry,
    subtitles_name: str = "subtitles.ass",
) -> Path:
    destination = registry.register(_temporary_path(output_dir / subtitles_name))
    words = _segment_words(transcript, segment)
    with _translated("subtítulos"):
        return renderer.write(words, destination)


def _burn(
    video: Path,
    *,
    subtitles: Path,
    renderer: SubtitleBurner,
    output_dir: Path,
    final_name: str = _FINAL_NAME,
    mute_audio: bool = False,
) -> Path:
    destination = output_dir / final_name
    with _translated("quemado de subtítulos"):
        return renderer.burn(
            video=video, subtitles=subtitles, destination=destination, mute_audio=mute_audio
        )


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


def _run_ffmpeg(argv: Sequence[str], *, render: RenderConfig) -> None:
    run_ffmpeg_with_fallback(
        argv,
        render=render,
        error_cls=PipelineError,
        runner=subprocess.run,
    )


def _atomic_write_json(destination: Path, content: str) -> None:
    parent = destination.parent
    parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=parent, prefix=f".{destination.stem}-", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as file:
            _ = file.write(content)
        _ = Path(tmp).replace(destination)
    except BaseException:
        with suppress(OSError):
            Path(tmp).unlink()
        raise


def _temporary_path(destination: Path) -> Path:
    return destination.with_name(f".{destination.stem}.part-{uuid.uuid4().hex}{destination.suffix}")


def _seconds(value: float) -> str:
    return f"{value:.3f}"


def _tail(text: str) -> str:
    return text.strip()[-_STDERR_TAIL:]
