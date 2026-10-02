"""Reframe dinámico a 9:16 con detección de caras en CPU (MediaPipe Tasks).

El módulo convierte un video horizontal en un recorte vertical 9:16 que sigue al
sujeto: muestrea frames con ffmpeg, detecta caras con MediaPipe y calcula una
trayectoria suavizada de la ventana de recorte, que luego aplica con el filtro
``crop`` de ffmpeg.

MediaPipe corre SIEMPRE en CPU (``BaseOptions.Delegate.CPU``): no se importa
ninguna ruta GPU y este módulo no crea ningún contexto ``torch``/CUDA, de modo
que la VRAM queda libre después de la transcripción (brief §8). La API de
MediaPipe usada es la Tasks API (``FaceDetector``), única disponible en las
versiones con ruedas para Python 3.13; el modelo corto de BlazeFace
(``blaze_face_short_range``) equivale al ``model_selection=0`` de la API antigua.

``mediapipe`` y ``numpy`` son dependencias opcionales (extra ``reframe``): se
importan de forma diferida para que el módulo base se cargue sin ellas. Toda
invocación de ffmpeg usa lista de argumentos (nunca shell) y todo fallo externo
se traduce a ``ReframeError`` conservando la causa.
"""

import contextlib
import importlib
import subprocess
import time
import uuid
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, ClassVar, Protocol, Self, cast

from pydantic import BaseModel, ConfigDict, Field, model_validator

from kliptych.encoding import (
    RenderConfig,
    audio_and_container_arguments,
    run_ffmpeg_with_fallback,
    video_encoder_arguments,
)

_TARGET_ASPECT = 9.0 / 16.0
_DEFAULT_ASPECT = "9:16"
_DEFAULT_SAMPLE_FPS = 2.0
_DEFAULT_SMOOTHING = 0.35
_DEFAULT_MAX_STEP_RATIO = 0.02
_DEFAULT_MIN_CONFIDENCE = 0.3
_DEFAULT_MIN_SUPPRESSION = 0.3
_STDERR_TAIL = 400
_DIMENSION_PARTS = 2


class ReframeError(Exception):
    """El reframe no se pudo calcular o renderizar."""


class ReframeTarget(BaseModel):
    """Ventana de recorte de un frame, normalizada respecto al video fuente.

    Las cuatro coordenadas van en escala 0.0-1.0 sobre el ancho y el alto del
    VOD, no en pixeles: asi la misma trayectoria sirve para cualquier
    resolucion del mismo contenido. Los pixeles absolutos se derivan al
    construir el filtro de ffmpeg, multiplicando por las dimensiones reales
    (``ReframeResult.source_width`` / ``source_height``) y ajustando a pixeles
    pares, que exige el subsampling de ffmpeg.
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", frozen=True)

    x: float = Field(ge=0.0, le=1.0)
    y: float = Field(ge=0.0, le=1.0)
    width: float = Field(gt=0.0, le=1.0)
    height: float = Field(gt=0.0, le=1.0)

    @model_validator(mode="after")
    def _window_stays_inside_frame(self) -> Self:
        """Rechaza ventanas que se salen del frame por la derecha o por abajo.

        El limite es estricto, sin tolerancia: una ventana ``x + width <= 1.0``
        escrita en decimal (por ejemplo 0.1 + 0.9) suma exactamente 1.0 o menos
        en IEEE-754, y lo mismo para la trayectoria que produce
        ``_targets_from_faces``, asi que un epsilon aqui solo relajaria el
        limite sin absorber ningun error real.

        Returns:
            La propia ventana, si cabe dentro del frame normalizado.

        Raises:
            ValueError: Si ``x + width`` o ``y + height`` exceden 1.0.
        """
        right = self.x + self.width
        if right > 1.0:
            msg = f"la ventana se sale del frame por la derecha: x+width={right}"
            raise ValueError(msg)
        bottom = self.y + self.height
        if bottom > 1.0:
            msg = f"la ventana se sale del frame por abajo: y+height={bottom}"
            raise ValueError(msg)
        return self


class ReframeResult(BaseModel):
    """Trayectoria de recorte 9:16 calculada para un video."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", frozen=True)

    targets: tuple[ReframeTarget, ...]
    source_width: int = Field(gt=0)
    source_height: int = Field(gt=0)
    target_aspect: str = _DEFAULT_ASPECT


class FaceBox(BaseModel):
    """Caja de una cara detectada, normalizada respecto al frame."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", frozen=True)

    x: float = Field(ge=0.0, le=1.0)
    y: float = Field(ge=0.0, le=1.0)
    width: float = Field(gt=0.0, le=1.0)
    height: float = Field(gt=0.0, le=1.0)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)


@dataclass(frozen=True, slots=True)
class RgbFrame:
    """Frame RGB24 crudo, tal como lo entrega ffmpeg."""

    width: int
    height: int
    data: bytes

    def __post_init__(self) -> None:
        """Valida las dimensiones y el tamaño del buffer.

        Raises:
            ValueError: Si las dimensiones no son positivas o el buffer no mide
                ``width * height * 3`` bytes.
        """
        if self.width <= 0 or self.height <= 0:
            msg = f"dimensiones de frame inválidas: {self.width}x{self.height}"
            raise ValueError(msg)
        if len(self.data) != self.width * self.height * 3:
            msg = f"buffer RGB inválido: {len(self.data)} bytes para {self.width}x{self.height}"
            raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class SampledFrame:
    """Frame muestreado con su marca de tiempo dentro del video."""

    timestamp_s: float
    frame: RgbFrame


@dataclass(frozen=True, slots=True)
class ReframeConfig:
    """Parámetros del muestreo, suavizado y detección del reframe."""

    sample_fps: float = _DEFAULT_SAMPLE_FPS
    smoothing: float = _DEFAULT_SMOOTHING
    max_step_ratio: float = _DEFAULT_MAX_STEP_RATIO
    min_confidence: float = _DEFAULT_MIN_CONFIDENCE

    def __post_init__(self) -> None:
        """Valida los parámetros del reframe.

        Raises:
            ValueError: Si algún parámetro queda fuera de rango.
        """
        if self.sample_fps <= 0:
            msg = f"fps de muestreo inválido: {self.sample_fps}"
            raise ValueError(msg)
        if not 0.0 < self.smoothing <= 1.0:
            msg = f"suavizado inválido: {self.smoothing}"
            raise ValueError(msg)
        if not 0.0 <= self.max_step_ratio <= 1.0:
            msg = f"paso máximo inválido: {self.max_step_ratio}"
            raise ValueError(msg)
        if not 0.0 <= self.min_confidence <= 1.0:
            msg = f"confianza mínima inválida: {self.min_confidence}"
            raise ValueError(msg)


class FaceDetector(Protocol):
    """Detector de caras que consume el reframe."""

    def detect(self, frame: RgbFrame) -> tuple[FaceBox, ...]:
        """Detecta caras en un frame RGB.

        Args:
            frame: Frame RGB24 crudo.

        Returns:
            Las cajas de las caras detectadas, normalizadas al frame.
        """
        ...


class FrameSource(Protocol):
    """Fuente de frames muestreados de un video."""

    def frames(self, video: Path, *, sample_fps: float) -> Iterator[SampledFrame]:
        """Extrae frames del video a la tasa pedida.

        Args:
            video: Ruta del video fuente.
            sample_fps: Frames por segundo a muestrear.

        Yields:
            Los frames muestreados, en orden temporal.
        """
        ...


class _NumpyArray(Protocol):
    """Superficie mínima de un array de numpy que usa el reframe."""

    def reshape(self, shape: tuple[int, int, int]) -> object: ...


class _NumpyApi(Protocol):
    """Superficie mínima de numpy que usa el reframe."""

    uint8: object

    def frombuffer(self, buffer: bytes, *, dtype: object) -> _NumpyArray: ...


class _MpBoundingBox(Protocol):
    """Caja absoluta de MediaPipe, en píxeles."""

    origin_x: int
    origin_y: int
    width: int
    height: int


class _MpCategory(Protocol):
    """Categoría de una detección de MediaPipe."""

    score: float


class _MpDetection(Protocol):
    """Detección de MediaPipe."""

    bounding_box: _MpBoundingBox
    categories: Sequence[_MpCategory]


class _MpResult(Protocol):
    """Resultado de una detección de MediaPipe."""

    detections: Sequence[_MpDetection]


class _RawFaceDetector(Protocol):
    """Detector crudo de la Tasks API de MediaPipe."""

    def detect(self, image: object) -> _MpResult: ...

    def close(self) -> None: ...


@dataclass(frozen=True, slots=True)
class _MediaPipeApi:
    """Componentes de la Tasks API de MediaPipe usados por el detector."""

    make_image: Callable[[RgbFrame], object]
    base_options: Callable[..., object]
    delegate_cpu: object
    face_detector_options: Callable[..., object]
    running_mode_image: object
    create_face_detector: Callable[..., _RawFaceDetector]


class MediaPipeFaceDetector:
    """Detecta caras con la Tasks API de MediaPipe, siempre en CPU."""

    def __init__(
        self,
        *,
        model_path: Path,
        min_confidence: float = _DEFAULT_MIN_CONFIDENCE,
        min_suppression: float = _DEFAULT_MIN_SUPPRESSION,
    ) -> None:
        """Configura el detector con el modelo local de BlazeFace.

        Args:
            model_path: Ruta del modelo ``.tflite`` de detección de caras.
            min_confidence: Confianza mínima de detección, en ``(0, 1]``.
            min_suppression: Umbral de supresión de cajas solapadas, en ``(0, 1]``.

        Raises:
            ValueError: Si los umbrales no están en ``(0, 1]``.
        """
        if not 0.0 < min_confidence <= 1.0:
            msg = f"confianza mínima inválida: {min_confidence}"
            raise ValueError(msg)
        if not 0.0 < min_suppression <= 1.0:
            msg = f"supresión mínima inválida: {min_suppression}"
            raise ValueError(msg)
        self._model_path: Path = model_path
        self._min_confidence: float = min_confidence
        self._min_suppression: float = min_suppression
        self._api: _MediaPipeApi | None = None
        self._detector: _RawFaceDetector | None = None

    def detect(self, frame: RgbFrame) -> tuple[FaceBox, ...]:
        """Detecta caras en un frame RGB.

        El detector se construye una sola vez y se reutiliza entre frames; el
        modelo no se recarga en el camino caliente.

        Args:
            frame: Frame RGB24 crudo.

        Returns:
            Las cajas de las caras detectadas, normalizadas al frame.

        Raises:
            ReframeError: Si el modelo no existe o MediaPipe no está disponible.
        """
        detector = self._ensure_detector()
        image = self._ensure_api().make_image(frame)
        return _boxes_from_result(detector.detect(image), frame)

    def close(self) -> None:
        """Libera el detector de MediaPipe si fue creado."""
        detector, self._detector = self._detector, None
        if detector is not None:
            with contextlib.suppress(Exception):
                detector.close()

    def _ensure_api(self) -> _MediaPipeApi:
        if self._api is None:
            self._api = _load_mediapipe()
        return self._api

    def _ensure_detector(self) -> _RawFaceDetector:
        if self._detector is None:
            if not self._model_path.is_file():
                msg = f"el modelo de caras no existe: {self._model_path}"
                raise ReframeError(msg)
            api = self._ensure_api()
            base = api.base_options(
                model_asset_path=str(self._model_path),
                delegate=api.delegate_cpu,
            )
            options = api.face_detector_options(
                base_options=base,
                running_mode=api.running_mode_image,
                min_detection_confidence=self._min_confidence,
                min_suppression_threshold=self._min_suppression,
            )
            self._detector = api.create_face_detector(options)
        return self._detector


class FFmpegFrameSource:
    """Extrae frames RGB24 de un video con ffmpeg, en streaming."""

    def __init__(
        self,
        *,
        ffmpeg: str = "ffmpeg",
        ffprobe: str = "ffprobe",
        timeout_s: float = 600.0,
    ) -> None:
        """Configura los binarios y el timeout de la extracción.

        Args:
            ffmpeg: Nombre o ruta del binario ffmpeg.
            ffprobe: Nombre o ruta del binario ffprobe.
            timeout_s: Timeout máximo de la extracción, en segundos.

        Raises:
            ValueError: Si el timeout no es positivo.
        """
        if timeout_s <= 0:
            msg = f"timeout inválido: {timeout_s}"
            raise ValueError(msg)
        self._ffmpeg: str = ffmpeg
        self._ffprobe: str = ffprobe
        self._timeout_s: float = timeout_s

    def frames(self, video: Path, *, sample_fps: float) -> Iterator[SampledFrame]:
        """Extrae frames del video a la tasa pedida, sin cargarlos todos en memoria.

        Args:
            video: Ruta del video fuente.
            sample_fps: Frames por segundo a muestrear.

        Yields:
            Los frames muestreados, en orden temporal.

        Raises:
            ValueError: Si ``sample_fps`` no es positivo.
            ReframeError: Si el video no existe, ffprobe/ffmpeg fallan o expiran.
        """
        if sample_fps <= 0:
            msg = f"fps de muestreo inválido: {sample_fps}"
            raise ValueError(msg)
        if not video.is_file():
            msg = f"el video no existe: {video}"
            raise ReframeError(msg)
        width, height = _probe_dimensions(video, ffprobe=self._ffprobe, timeout_s=self._timeout_s)
        frame_size = width * height * 3
        argv = [
            self._ffmpeg,
            "-hide_banner",
            "-nostdin",
            "-v",
            "error",
            "-i",
            str(video),
            "-vf",
            f"fps={_num(sample_fps)}",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-",
        ]
        process = _open_process(argv, ffmpeg=self._ffmpeg)
        stdout = cast("BinaryIO", process.stdout)
        deadline = time.monotonic() + self._timeout_s
        index = 0
        try:
            while True:
                if time.monotonic() > deadline:
                    _terminate(process)
                    msg = f"ffmpeg excedió el timeout de {self._timeout_s} s al extraer frames"
                    raise ReframeError(msg)
                chunk = _read_exact(stdout, frame_size)
                if len(chunk) < frame_size:
                    break
                yield SampledFrame(
                    timestamp_s=index / sample_fps,
                    frame=RgbFrame(width=width, height=height, data=chunk),
                )
                index += 1
            _ = process.wait(timeout=self._timeout_s)
        finally:
            _close_process(process)
        if process.returncode != 0:
            msg = f"ffmpeg falló al extraer frames con código {process.returncode}"
            raise ReframeError(msg)


class FFmpegReframer:
    """Calcula y aplica el recorte dinámico 9:16 de un video."""

    def __init__(
        self,
        *,
        detector: FaceDetector,
        frame_source: FrameSource | None = None,
        config: ReframeConfig | None = None,
        render: RenderConfig | None = None,
    ) -> None:
        """Configura el detector, la fuente de frames y el render.

        Args:
            detector: Detector de caras inyectable (real o falso en tests).
            frame_source: Fuente de frames; por defecto usa ffmpeg.
            config: Parámetros de muestreo y suavizado; por defecto los estándar.
            render: Binario, timeout y NVENC del render; por defecto los estándar.
        """
        self._detector: FaceDetector = detector
        self._config: ReframeConfig = ReframeConfig() if config is None else config
        self._render: RenderConfig = RenderConfig() if render is None else render
        self._frame_source: FrameSource = (
            FFmpegFrameSource(
                ffmpeg=self._render.ffmpeg,
                timeout_s=self._render.timeout_s,
            )
            if frame_source is None
            else frame_source
        )

    def analyze(self, video: Path) -> ReframeResult:
        """Detecta caras y calcula la trayectoria de recorte 9:16.

        Args:
            video: Ruta del video fuente.

        Returns:
            La trayectoria de recorte con las dimensiones de origen.

        Raises:
            ReframeError: Si el video no existe, no se extraen frames o la
                detección falla.
        """
        if not video.is_file():
            msg = f"el video no existe: {video}"
            raise ReframeError(msg)
        faces_per_frame: list[tuple[FaceBox, ...]] = []
        source_width = 0
        source_height = 0
        for sample in self._frame_source.frames(video, sample_fps=self._config.sample_fps):
            if source_width == 0:
                source_width = sample.frame.width
                source_height = sample.frame.height
            faces_per_frame.append(self._detector.detect(sample.frame))
        if source_width == 0:
            msg = f"no se pudieron extraer frames del video: {video}"
            raise ReframeError(msg)
        targets = _targets_from_faces(
            faces_per_frame,
            source_width=source_width,
            source_height=source_height,
            config=self._config,
        )
        return ReframeResult(
            targets=targets,
            source_width=source_width,
            source_height=source_height,
        )

    def reframe(self, *, video: Path, destination: Path) -> Path:
        """Renderiza el recorte 9:16 del video siguiendo la trayectoria.

        El render ocurre en un temporal hermano y se publica con un reemplazo
        atómico solo si ffmpeg termina con éxito; ante cualquier fallo el
        artefacto previo en ``destination`` queda intacto.

        Args:
            video: Ruta del video fuente.
            destination: Ruta del artefacto final; se crean los directorios
                padre que falten.

        Returns:
            La ruta del artefacto recortado.

        Raises:
            ReframeError: Si las entradas no existen, el destino no se puede
                preparar, ffmpeg falla o expira.
        """
        result = self.analyze(video)
        return self.render(video=video, destination=destination, result=result)

    def render(self, *, video: Path, destination: Path, result: ReframeResult) -> Path:
        """Renderiza el recorte 9:16 a partir de una trayectoria ya calculada.

        Permite reutilizar la trayectoria de ``analyze`` sin volver a muestrear
        frames ni re-ejecutar la detección. El render ocurre en un temporal
        hermano y se publica con un reemplazo atómico solo si ffmpeg termina
        con éxito; ante cualquier fallo el artefacto previo en ``destination``
        queda intacto.

        Args:
            video: Ruta del video fuente.
            destination: Ruta del artefacto final; se crean los directorios
                padre que falten.
            result: Trayectoria de recorte ya calculada.

        Returns:
            La ruta del artefacto recortado.

        Raises:
            ReframeError: Si el destino no se puede preparar, ffmpeg falla o
                expira.
        """
        try:
            destination.parent.mkdir(parents=True, exist_ok=True)
        except OSError as error:
            msg = f"no se pudo preparar el directorio del destino {destination}: {error}"
            raise ReframeError(msg) from error
        temporary = _temporary_path(destination)
        argv = self._build_argv(video=video, destination=temporary, result=result)
        _run_ffmpeg(argv, temporary=temporary, render=self._render)
        try:
            _ = temporary.replace(destination)
        except OSError as error:
            _remove_quietly(temporary)
            msg = f"no se pudo publicar el artefacto en {destination}: {error}"
            raise ReframeError(msg) from error
        return destination

    def render_arguments(
        self,
        *,
        video: Path,
        destination: Path,
        result: ReframeResult,
    ) -> tuple[str, ...]:
        """Devuelve el argv de ffmpeg que se usaría para este recorte.

        Es la receta de render que se registra en el manifiesto; el recorte real
        escribe primero en un temporal y publica al final.

        Args:
            video: Ruta del video fuente.
            destination: Ruta final del artefacto.
            result: Trayectoria de recorte ya calculada.

        Returns:
            El argv completo de ffmpeg, como tupla inmutable.
        """
        return tuple(self._build_argv(video=video, destination=destination, result=result))

    def _build_argv(
        self,
        *,
        video: Path,
        destination: Path,
        result: ReframeResult,
    ) -> list[str]:
        crop = _crop_filter(
            result.targets,
            sample_fps=self._config.sample_fps,
            source_width=result.source_width,
            source_height=result.source_height,
        )
        argv = [
            self._render.ffmpeg,
            "-hide_banner",
            "-nostdin",
            "-v",
            "error",
            "-y",
            "-i",
            str(video),
            "-vf",
            crop,
            "-map",
            "0:v:0",
            "-map",
            "0:a?",
        ]
        argv += list(video_encoder_arguments(nvenc_available=self._render.nvenc_available))
        argv += list(audio_and_container_arguments())
        argv.append(str(destination))
        return argv


def _even(value: int) -> int:
    """Redondea a la baja al par no negativo más cercano.

    El clamp a 0 evita que un offset negativo sobreviva al redondeo hacia
    abajo: ``_even(-3)`` sería -4, y ffmpeg leería un ``crop`` con x/y
    negativo como un recorte fuera del frame en lugar de un offset inválido.

    Args:
        value: Valor a ajustar.

    Returns:
        El mayor par <= ``value`` que no es negativo.
    """
    floored = max(0, value)
    return floored if floored % 2 == 0 else floored - 1


def _clamp(value: float, low: float, high: float) -> float:
    return min(high, max(low, value))


def _num(value: float) -> str:
    return f"{value:g}"


def _crop_size(source_width: int, source_height: int) -> tuple[int, int]:
    if source_width <= 0 or source_height <= 0:
        msg = f"dimensiones de origen inválidas: {source_width}x{source_height}"
        raise ValueError(msg)
    if source_height * 9 <= source_width * 16:
        height = _even(source_height)
        width = _even(round(height * _TARGET_ASPECT))
    else:
        width = _even(source_width)
        height = _even(round(width / _TARGET_ASPECT))
    width = max(2, min(width, _even(source_width)))
    height = max(2, min(height, _even(source_height)))
    return width, height


def _primary_face(faces: Sequence[FaceBox], min_confidence: float) -> FaceBox | None:
    candidates = [face for face in faces if face.confidence >= min_confidence]
    if not candidates:
        return None
    return max(candidates, key=lambda face: face.width * face.height)


def _targets_from_faces(
    faces_per_frame: Sequence[tuple[FaceBox, ...]],
    *,
    source_width: int,
    source_height: int,
    config: ReframeConfig,
) -> tuple[ReframeTarget, ...]:
    geometry = _crop_geometry(source_width, source_height)
    centers = _smoothed_centers(
        faces_per_frame,
        source_width=source_width,
        source_height=source_height,
        geometry=geometry,
        config=config,
    )
    targets: list[ReframeTarget] = []
    for center_x, center_y in centers:
        half_width = geometry.width / 2.0
        half_height = geometry.height / 2.0
        offset_x = max(0, min(_even(round(center_x - half_width)), geometry.max_offset_x))
        offset_y = max(0, min(_even(round(center_y - half_height)), geometry.max_offset_y))
        targets.append(
            ReframeTarget(
                x=offset_x / source_width,
                y=offset_y / source_height,
                width=geometry.width / source_width,
                height=geometry.height / source_height,
            )
        )
    return tuple(targets)


@dataclass(frozen=True, slots=True)
class _CropGeometry:
    """Tamaño del recorte 9:16 y límites de su centro en el video fuente."""

    width: int
    height: int
    max_offset_x: int
    max_offset_y: int
    min_center_x: float
    max_center_x: float
    min_center_y: float
    max_center_y: float


def _crop_geometry(source_width: int, source_height: int) -> _CropGeometry:
    crop_width, crop_height = _crop_size(source_width, source_height)
    return _CropGeometry(
        width=crop_width,
        height=crop_height,
        max_offset_x=_even(source_width - crop_width),
        max_offset_y=_even(source_height - crop_height),
        min_center_x=crop_width / 2.0,
        max_center_x=source_width - crop_width / 2.0,
        min_center_y=crop_height / 2.0,
        max_center_y=source_height - crop_height / 2.0,
    )


def _desired_center(
    faces: Sequence[FaceBox],
    *,
    source_width: int,
    source_height: int,
    geometry: _CropGeometry,
    min_confidence: float,
) -> tuple[float, float]:
    face = _primary_face(faces, min_confidence)
    if face is None:
        desired_x = source_width / 2.0
        desired_y = source_height / 2.0
    else:
        desired_x = (face.x + face.width / 2.0) * source_width
        desired_y = (face.y + face.height / 2.0) * source_height
    return (
        _clamp(desired_x, geometry.min_center_x, geometry.max_center_x),
        _clamp(desired_y, geometry.min_center_y, geometry.max_center_y),
    )


def _smoothed_centers(
    faces_per_frame: Sequence[tuple[FaceBox, ...]],
    *,
    source_width: int,
    source_height: int,
    geometry: _CropGeometry,
    config: ReframeConfig,
) -> tuple[tuple[float, float], ...]:
    max_step = max(1.0, config.max_step_ratio * source_width)
    center_x = source_width / 2.0
    center_y = source_height / 2.0
    initialized = False
    centers: list[tuple[float, float]] = []
    for faces in faces_per_frame:
        desired_x, desired_y = _desired_center(
            faces,
            source_width=source_width,
            source_height=source_height,
            geometry=geometry,
            min_confidence=config.min_confidence,
        )
        if not initialized:
            center_x, center_y = desired_x, desired_y
            initialized = True
        else:
            next_x = config.smoothing * desired_x + (1.0 - config.smoothing) * center_x
            next_y = config.smoothing * desired_y + (1.0 - config.smoothing) * center_y
            center_x = _clamp(next_x, center_x - max_step, center_x + max_step)
            center_y = _clamp(next_y, center_y - max_step, center_y + max_step)
            center_x = _clamp(center_x, geometry.min_center_x, geometry.max_center_x)
            center_y = _clamp(center_y, geometry.min_center_y, geometry.max_center_y)
        centers.append((center_x, center_y))
    return tuple(centers)


def _linear_expression(times: Sequence[float], values: Sequence[float]) -> str:
    if not times or len(times) != len(values):
        msg = "tiempos y valores deben ser no vacíos y de igual longitud"
        raise ValueError(msg)
    if len(times) == 1:
        return _num(values[0])

    def _build_tree(i: int, j: int) -> str:
        if j - i == 1:
            start_t = times[i]
            end_t = times[j]
            start_v = values[i]
            end_v = values[j]
            if end_t <= start_t:
                return f"if(lt(t,{_num(end_t)}),{_num(start_v)},{_num(end_v)})"
            if start_v == end_v:
                return _num(start_v)
            segment = (
                f"{_num(start_v)}+({_num(end_v)}-{_num(start_v)})"
                f"*(t-{_num(start_t)})/({_num(end_t)}-{_num(start_t)})"
            )
            return f"if(lt(t,{_num(end_t)}),{segment},{_num(end_v)})"
        mid = (i + j) // 2
        mid_t = times[mid]
        left = _build_tree(i, mid)
        right = _build_tree(mid, j)
        return f"if(lt(t,{_num(mid_t)}),{left},{right})"

    return _build_tree(0, len(times) - 1)


def _denormalized_size(normalized: float, total: int) -> int:
    """Convierte una dimensión normalizada a píxeles pares dentro del frame.

    Args:
        normalized: Dimensión en escala 0.0-1.0.
        total: Ancho o alto real del video fuente.

    Returns:
        La dimensión en píxeles, par y no mayor que el frame.

    Raises:
        ValueError: Si ``total`` no es positiva.
    """
    if total <= 0:
        msg = f"dimensión de origen inválida: {total}"
        raise ValueError(msg)
    return max(2, min(_even(round(normalized * total)), _even(total)))


def _denormalized_offset(normalized: float, total: int, *, size: int) -> int:
    """Convierte un desplazamiento normalizado a píxeles pares dentro del frame.

    Args:
        normalized: Desplazamiento en escala 0.0-1.0.
        total: Ancho o alto real del video fuente.
        size: Tamaño ya desnormalizado de la ventana en esa direccion.

    Returns:
        El desplazamiento en píxeles, par y tal que ``offset + size <= total``.
    """
    limit = _even(total - size)
    if limit <= 0:
        return 0
    return max(0, min(_even(round(normalized * total)), limit))


def _crop_filter(
    targets: Sequence[ReframeTarget],
    *,
    sample_fps: float,
    source_width: int,
    source_height: int,
) -> str:
    """Construye el filtro ``crop`` de ffmpeg desde una trayectoria normalizada.

    Args:
        targets: Ventanas normalizadas de la trayectoria.
        sample_fps: Frecuencia de muestreo usada al calcular la trayectoria.
        source_width: Ancho real del video fuente.
        source_height: Alto real del video fuente.

    Returns:
        El valor completo del flag ``-vf``.

    Raises:
        ValueError: Si no hay ventanas, si no comparten tamaño o si las
            dimensiones del fuente no son positivas.
    """
    if not targets:
        msg = "no hay recortes para construir el filtro"
        raise ValueError(msg)
    width = _denormalized_size(targets[0].width, source_width)
    height = _denormalized_size(targets[0].height, source_height)
    if any(
        _denormalized_size(target.width, source_width) != width
        or _denormalized_size(target.height, source_height) != height
        for target in targets
    ):
        msg = "todos los recortes deben compartir tamaño"
        raise ValueError(msg)
    times = [index / sample_fps for index in range(len(targets))]
    offsets_x = [_denormalized_offset(target.x, source_width, size=width) for target in targets]
    offsets_y = [_denormalized_offset(target.y, source_height, size=height) for target in targets]
    expression_x = _linear_expression(times, [float(value) for value in offsets_x])
    expression_y = _linear_expression(times, [float(value) for value in offsets_y])
    return f"crop={width}:{height}:x='{expression_x}':y='{expression_y}'"


def _boxes_from_result(result: _MpResult, frame: RgbFrame) -> tuple[FaceBox, ...]:
    boxes: list[FaceBox] = []
    for detection in result.detections:
        box = detection.bounding_box
        confidence = float(detection.categories[0].score) if detection.categories else 0.0
        x = _clamp(box.origin_x / frame.width, 0.0, 1.0)
        y = _clamp(box.origin_y / frame.height, 0.0, 1.0)
        width = _clamp(box.width / frame.width, 0.0, 1.0 - x)
        height = _clamp(box.height / frame.height, 0.0, 1.0 - y)
        if width <= 0.0 or height <= 0.0:
            continue
        boxes.append(
            FaceBox(
                x=x,
                y=y,
                width=width,
                height=height,
                confidence=_clamp(confidence, 0.0, 1.0),
            )
        )
    return tuple(boxes)


def _require_api_attribute(owner: object, name: str, what: str) -> object:
    value = getattr(owner, name, None)
    if value is None:
        msg = f"mediapipe no expone {what}"
        raise ReframeError(msg)
    return cast("object", value)


def _build_mediapipe_api(
    module: object,
    tasks: object,
    vision: object,
    numpy: _NumpyApi,
) -> _MediaPipeApi:
    image_factory = cast("Callable[..., object]", _require_api_attribute(module, "Image", "Image"))
    image_format = _require_api_attribute(module, "ImageFormat", "ImageFormat")
    base_options = cast(
        "Callable[..., object]", _require_api_attribute(tasks, "BaseOptions", "BaseOptions")
    )
    face_detector = _require_api_attribute(vision, "FaceDetector", "FaceDetector")
    face_options = cast(
        "Callable[..., object]",
        _require_api_attribute(vision, "FaceDetectorOptions", "FaceDetectorOptions"),
    )
    running_mode = _require_api_attribute(vision, "RunningMode", "RunningMode")
    delegate_cpu = _require_api_attribute(
        _require_api_attribute(base_options, "Delegate", "BaseOptions.Delegate"),
        "CPU",
        "Delegate.CPU",
    )
    srgb = _require_api_attribute(image_format, "SRGB", "ImageFormat.SRGB")
    image_mode = _require_api_attribute(running_mode, "IMAGE", "RunningMode.IMAGE")
    create_detector = cast(
        "Callable[..., _RawFaceDetector]",
        _require_api_attribute(
            face_detector, "create_from_options", "FaceDetector.create_from_options"
        ),
    )

    def make_image(frame: RgbFrame) -> object:
        array = numpy.frombuffer(frame.data, dtype=numpy.uint8).reshape(
            (frame.height, frame.width, 3)
        )
        return image_factory(image_format=srgb, data=array)

    return _MediaPipeApi(
        make_image=make_image,
        base_options=base_options,
        delegate_cpu=delegate_cpu,
        face_detector_options=face_options,
        running_mode_image=image_mode,
        create_face_detector=create_detector,
    )


def _load_mediapipe() -> _MediaPipeApi:
    # Import diferido: mediapipe y numpy son dependencias opcionales.
    try:
        module = importlib.import_module("mediapipe")
        tasks = importlib.import_module("mediapipe.tasks.python")
        vision = importlib.import_module("mediapipe.tasks.python.vision")
        numpy = cast("_NumpyApi", cast("object", importlib.import_module("numpy")))
    except ImportError as error:
        msg = "mediapipe no está instalado: instala la extra 'reframe'"
        raise ReframeError(msg) from error
    return _build_mediapipe_api(module, tasks, vision, numpy)


def _probe_dimensions(video: Path, *, ffprobe: str, timeout_s: float) -> tuple[int, int]:
    argv = [
        ffprobe,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height",
        "-of",
        "csv=s=x:p=0",
        str(video),
    ]
    try:
        completed = subprocess.run(
            argv,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
    except FileNotFoundError as error:
        msg = f"ffprobe no está disponible: {ffprobe}"
        raise ReframeError(msg) from error
    except subprocess.TimeoutExpired as error:
        msg = f"ffprobe excedió el timeout de {timeout_s} s"
        raise ReframeError(msg) from error
    except OSError as error:
        msg = f"no se pudo ejecutar ffprobe ({ffprobe}): {error}"
        raise ReframeError(msg) from error
    if completed.returncode != 0:
        msg = f"ffprobe falló con código {completed.returncode}: {_tail(completed.stderr)}"
        raise ReframeError(msg)
    first_line = completed.stdout.strip().splitlines()[0] if completed.stdout.strip() else ""
    parts = first_line.split("x")
    if len(parts) != _DIMENSION_PARTS or not all(part.isdigit() for part in parts):
        msg = f"no se pudieron leer las dimensiones de {video}: {first_line!r}"
        raise ReframeError(msg)
    width, height = int(parts[0]), int(parts[1])
    if width <= 0 or height <= 0:
        msg = f"dimensiones inválidas en {video}: {width}x{height}"
        raise ReframeError(msg)
    return width, height


def _open_process(argv: Sequence[str], *, ffmpeg: str) -> subprocess.Popen[bytes]:
    try:
        return subprocess.Popen(
            list(argv),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except FileNotFoundError as error:
        msg = f"ffmpeg no está disponible: {ffmpeg}"
        raise ReframeError(msg) from error
    except OSError as error:
        msg = f"no se pudo ejecutar ffmpeg ({ffmpeg}): {error}"
        raise ReframeError(msg) from error


def _read_exact(stream: BinaryIO, size: int) -> bytes:
    chunks: list[bytes] = []
    remaining = size
    while remaining > 0:
        chunk = stream.read(remaining)
        if not chunk:
            break
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _terminate(process: subprocess.Popen[bytes]) -> None:
    with contextlib.suppress(OSError):
        process.kill()


def _close_process(process: subprocess.Popen[bytes]) -> None:
    for stream in (process.stdout, process.stderr):
        if stream is not None:
            with contextlib.suppress(OSError):
                stream.close()
    if process.poll() is None:
        _terminate(process)
        with contextlib.suppress(OSError):
            _ = process.wait()


def _run_ffmpeg(
    argv: Sequence[str],
    *,
    temporary: Path,
    render: RenderConfig,
) -> None:
    run_ffmpeg_with_fallback(
        argv,
        render=render,
        temporary=temporary,
        error_cls=ReframeError,
        runner=subprocess.run,
    )


def _temporary_path(destination: Path) -> Path:
    return destination.with_name(f".{destination.stem}.part-{uuid.uuid4().hex}{destination.suffix}")


def _remove_quietly(path: Path) -> None:
    with contextlib.suppress(OSError):
        path.unlink(missing_ok=True)


def _tail(text: str) -> str:
    return text.strip()[-_STDERR_TAIL:]
