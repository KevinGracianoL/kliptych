"""Detección de momentos candidatos por escena, energía de audio y densidad de chat.

El alcance termina en una lista rankeada de ``Moment``: no hay transcripción,
selección de segmentos ni render aquí. Las tres estrategias producen momentos
con su propia puntuación normalizada y ``detect`` las fusiona por solapamiento
temporal, de modo que una señal fuerte en varios canales pesa más que una sola.

Toda llamada a ffmpeg usa lista de argumentos (nunca shell) y se traduce a
``MomentDetectionError`` conservando la causa. ``scdet`` aporta la magnitud del
corte de escena y ``astats`` la energía RMS por frame; ninguna de las dos exige
dependencias fuera de ffmpeg.
"""

import math
import re
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import ClassVar, Protocol, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from kliptych.transcribe import Transcript

_DEFAULT_TIMEOUT_S = 300.0
_DEFAULT_WINDOW_S = 2.0
_DEFAULT_SCENE_THRESHOLD = 10.0
_DEFAULT_SCENE_CHANGE = 0.3
_DEFAULT_ENERGY_MIN_SCORE = 0.5
_DEFAULT_CHAT_MIN_SCORE = 0.5
_STDERR_TAIL = 400

_SCENE_FILTER = (
    "scdet=threshold={threshold},select='gt(scene,{change})',"
    "metadata=print:key=lavfi.scd.score:file=-"
)
_ENERGY_FILTER = (
    "astats=metadata=1:reset=1,ametadata=print:key=lavfi.astats.Overall.RMS_level:file=-"
)
_PTS_TIME = re.compile(r"pts_time:\s*([0-9]+(?:\.[0-9]+)?)")
_SCENE_SCORE = re.compile(r"lavfi\.scd\.score=(-?[0-9]+(?:\.[0-9]+)?)")
_RMS_LEVEL = re.compile(r"lavfi\.astats\.Overall\.RMS_level=(-?[0-9]+(?:\.[0-9]+)?|-inf)")


class MomentDetectionError(Exception):
    """Los momentos no se pudieron detectar."""


class MomentSource(StrEnum):
    """Estrategia que produjo un momento."""

    SCENE = "scene"
    AUDIO_ENERGY = "audio_energy"
    CHAT_DENSITY = "chat_density"
    FUSED = "fused"


_DEFAULT_WEIGHTS: Mapping[MomentSource, float] = {
    MomentSource.SCENE: 0.3,
    MomentSource.AUDIO_ENERGY: 0.4,
    MomentSource.CHAT_DENSITY: 0.3,
}


@dataclass(frozen=True, slots=True)
class DetectionConfig:
    """Umbrales y pesos de la detección de momentos."""

    window_s: float = _DEFAULT_WINDOW_S
    scene_threshold: float = _DEFAULT_SCENE_THRESHOLD
    scene_change: float = _DEFAULT_SCENE_CHANGE
    energy_min_score: float = _DEFAULT_ENERGY_MIN_SCORE
    chat_min_score: float = _DEFAULT_CHAT_MIN_SCORE
    weights: Mapping[MomentSource, float] = field(default_factory=lambda: dict(_DEFAULT_WEIGHTS))

    def __post_init__(self) -> None:
        """Valida los umbrales y pesos.

        Raises:
            ValueError: Si algún umbral o peso no es válido.
        """
        if self.window_s <= 0:
            msg = f"ventana inválida: {self.window_s}"
            raise ValueError(msg)
        if self.scene_threshold < 0:
            msg = f"umbral de escena inválido: {self.scene_threshold}"
            raise ValueError(msg)
        if not 0.0 <= self.scene_change <= 1.0:
            msg = f"cambio de escena inválido: {self.scene_change}"
            raise ValueError(msg)
        if not 0.0 <= self.energy_min_score <= 1.0:
            msg = f"puntuación mínima de energía inválida: {self.energy_min_score}"
            raise ValueError(msg)
        if not 0.0 <= self.chat_min_score <= 1.0:
            msg = f"puntuación mínima de chat inválida: {self.chat_min_score}"
            raise ValueError(msg)
        if any(weight < 0 for weight in self.weights.values()):
            msg = f"pesos negativos no permitidos: {self.weights}"
            raise ValueError(msg)


class ChatMessage(BaseModel):
    """Mensaje de chat con su marca de tiempo en el stream."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", frozen=True)

    timestamp_s: float = Field(ge=0.0)
    text: str = Field(min_length=1)


class Moment(BaseModel):
    """Momento candidato de un vídeo, con su puntuación y procedencia."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", frozen=True)

    start_s: float = Field(ge=0.0)
    end_s: float = Field(ge=0.0)
    score: float = Field(ge=0.0, le=1.0)
    source: MomentSource

    @model_validator(mode="after")
    def _bounds_are_ordered_and_finite(self) -> Self:
        if not (math.isfinite(self.start_s) and math.isfinite(self.end_s)):
            msg = f"las cotas del momento deben ser finitas: {self.start_s}, {self.end_s}"
            raise ValueError(msg)
        if self.end_s < self.start_s:
            msg = f"start_s ({self.start_s}) no puede superar end_s ({self.end_s})"
            raise ValueError(msg)
        return self


class MomentDetector(Protocol):
    """Interfaz del detector de momentos que consume el pipeline."""

    def detect(
        self,
        video: Path,
        *,
        transcript: Transcript | None = None,
        chat: Sequence[ChatMessage] | None = None,
    ) -> tuple[Moment, ...]:
        """Detecta y fusiona los momentos candidatos de un vídeo.

        Args:
            video: Ruta del vídeo fuente.
            transcript: Transcripción opcional; aporta la duración del vídeo.
            chat: Mensajes de chat opcionales; aportan la densidad por ventana.

        Returns:
            Los momentos fusionados, ordenados por puntuación descendente.

        Raises:
            MomentDetectionError: Si el vídeo no existe o ffmpeg falla.
        """
        ...


class FFmpegMomentDetector:
    """Detecta momentos con ffmpeg: escenas, energía RMS y densidad de chat."""

    def __init__(
        self,
        *,
        ffmpeg: str = "ffmpeg",
        timeout_s: float = _DEFAULT_TIMEOUT_S,
        config: DetectionConfig | None = None,
    ) -> None:
        """Configura el binario, el timeout y los umbrales de detección.

        Args:
            ffmpeg: Nombre o ruta del binario ffmpeg.
            timeout_s: Timeout máximo de cada pasada de ffmpeg, en segundos.
            config: Umbrales y pesos de detección; por defecto los estándar.

        Raises:
            ValueError: Si el timeout no es positivo.
        """
        if timeout_s <= 0:
            msg = f"timeout inválido: {timeout_s}"
            raise ValueError(msg)
        self._ffmpeg: str = ffmpeg
        self._timeout_s: float = timeout_s
        self._config: DetectionConfig = DetectionConfig() if config is None else config

    def detect(
        self,
        video: Path,
        *,
        transcript: Transcript | None = None,
        chat: Sequence[ChatMessage] | None = None,
    ) -> tuple[Moment, ...]:
        """Detecta y fusiona los momentos candidatos de un vídeo.

        Args:
            video: Ruta del vídeo fuente.
            transcript: Transcripción opcional; aporta la duración del vídeo.
            chat: Mensajes de chat opcionales; aportan la densidad por ventana.

        Returns:
            Los momentos fusionados, ordenados por puntuación descendente.

        Raises:
            MomentDetectionError: Si el vídeo no existe o ffmpeg falla.
        """
        _require_file(video)
        config = self._config
        frames = self._energy_frames(video)
        if transcript is not None and transcript.duration_s > 0:
            duration_s = transcript.duration_s
        elif frames:
            duration_s = frames[-1][0]
        else:
            duration_s = 0.0
        energy = _window_moments(
            _energy_bins(frames, config.window_s),
            window_s=config.window_s,
            duration_s=duration_s,
            min_score=config.energy_min_score,
            source=MomentSource.AUDIO_ENERGY,
        )
        scenes = _scene_moments(self._scene_events(video), duration_s=duration_s)
        chats: tuple[Moment, ...] = ()
        if chat:
            chats = _window_moments(
                _chat_bins(chat, config.window_s),
                window_s=config.window_s,
                duration_s=duration_s,
                min_score=config.chat_min_score,
                source=MomentSource.CHAT_DENSITY,
            )
        return _fuse((*scenes, *energy, *chats), weights=config.weights)

    def _energy_frames(self, video: Path) -> tuple[tuple[float, float], ...]:
        completed = self._run_ffmpeg(self._energy_argv(video))
        return _parse_rms_levels(completed.stdout)

    def _scene_events(self, video: Path) -> tuple[tuple[float, float], ...]:
        completed = self._run_ffmpeg(self._scene_argv(video))
        return _parse_scene_scores(completed.stdout)

    def _energy_argv(self, video: Path) -> list[str]:
        return [
            self._ffmpeg,
            "-hide_banner",
            "-nostdin",
            "-v",
            "error",
            "-i",
            str(video),
            "-af",
            _ENERGY_FILTER,
            "-vn",
            "-f",
            "null",
            "-",
        ]

    def _scene_argv(self, video: Path) -> list[str]:
        scene_filter = _SCENE_FILTER.format(
            threshold=self._config.scene_threshold,
            change=self._config.scene_change,
        )
        return [
            self._ffmpeg,
            "-hide_banner",
            "-nostdin",
            "-v",
            "error",
            "-i",
            str(video),
            "-vf",
            scene_filter,
            "-an",
            "-f",
            "null",
            "-",
        ]

    def _run_ffmpeg(self, argv: Sequence[str]) -> subprocess.CompletedProcess[str]:
        try:
            completed = subprocess.run(
                list(argv),
                capture_output=True,
                text=True,
                timeout=self._timeout_s,
                check=False,
            )
        except FileNotFoundError as error:
            msg = f"ffmpeg no está disponible: {self._ffmpeg}"
            raise MomentDetectionError(msg) from error
        except subprocess.TimeoutExpired as error:
            msg = f"ffmpeg excedió el timeout de {self._timeout_s} s"
            raise MomentDetectionError(msg) from error
        except OSError as error:
            msg = f"no se pudo ejecutar ffmpeg ({self._ffmpeg}): {error}"
            raise MomentDetectionError(msg) from error
        if completed.returncode != 0:
            msg = f"ffmpeg falló con código {completed.returncode}: {_tail(completed.stderr)}"
            raise MomentDetectionError(msg)
        return completed


def _parse_scene_scores(text: str) -> tuple[tuple[float, float], ...]:
    events: list[tuple[float, float]] = []
    current: float | None = None
    for line in text.splitlines():
        time_match = _PTS_TIME.search(line)
        if time_match is not None:
            current = float(time_match.group(1))
            continue
        score_match = _SCENE_SCORE.search(line)
        if score_match is not None and current is not None:
            events.append((current, float(score_match.group(1))))
            current = None
    return tuple(events)


def _parse_rms_levels(text: str) -> tuple[tuple[float, float], ...]:
    frames: list[tuple[float, float]] = []
    current: float | None = None
    for line in text.splitlines():
        time_match = _PTS_TIME.search(line)
        if time_match is not None:
            current = float(time_match.group(1))
            continue
        level_match = _RMS_LEVEL.search(line)
        if level_match is not None and current is not None:
            frames.append((current, _db_to_linear(level_match.group(1))))
            current = None
    return tuple(frames)


def _db_to_linear(raw: str) -> float:
    if raw == "-inf":
        return 0.0
    value = float(raw)
    if not math.isfinite(value):
        return 0.0
    return math.pow(10.0, value / 20.0)


def _normalize(values: Sequence[float]) -> tuple[float, ...]:
    top = max(values, default=0.0)
    if top <= 0.0:
        return tuple(0.0 for _ in values)
    return tuple(min(1.0, max(0.0, value / top)) for value in values)


def _energy_bins(
    frames: Sequence[tuple[float, float]],
    window_s: float,
) -> tuple[float, ...]:
    if not frames:
        return ()
    last_index = int(frames[-1][0] // window_s)
    sums = [0.0] * (last_index + 1)
    counts = [0] * (last_index + 1)
    for time_s, value in frames:
        index = int(time_s // window_s)
        sums[index] += value
        counts[index] += 1
    return tuple(
        sums[index] / counts[index] if counts[index] else 0.0 for index in range(len(sums))
    )


def _chat_bins(messages: Sequence[ChatMessage], window_s: float) -> tuple[float, ...]:
    if not messages:
        return ()
    last_index = int(max(message.timestamp_s for message in messages) // window_s)
    counts = [0] * (last_index + 1)
    for message in messages:
        counts[int(message.timestamp_s // window_s)] += 1
    return tuple(count / window_s for count in counts)


def _window_moments(
    bins: Sequence[float],
    *,
    window_s: float,
    duration_s: float,
    min_score: float,
    source: MomentSource,
) -> tuple[Moment, ...]:
    normalized = _normalize(bins)
    moments: list[Moment] = []
    run_start: int | None = None
    run_end = 0
    run_score = 0.0

    def flush() -> None:
        nonlocal run_start, run_score
        if run_start is None:
            return
        start_s = run_start * window_s
        end_s = (run_end + 1) * window_s
        if duration_s > 0.0:
            end_s = min(end_s, duration_s)
        if end_s > start_s:
            moments.append(Moment(start_s=start_s, end_s=end_s, score=run_score, source=source))
        run_start = None
        run_score = 0.0

    for index, score in enumerate(normalized):
        if score > 0.0 and score >= min_score:
            if run_start is None:
                run_start = index
            run_end = index
            run_score = max(run_score, score)
        else:
            flush()
    flush()
    return tuple(moments)


def _scene_moments(
    events: Sequence[tuple[float, float]],
    *,
    duration_s: float,
) -> tuple[Moment, ...]:
    if not events or duration_s <= 0.0:
        return ()
    normalized = _normalize([score for _, score in events])
    cut_scores: dict[float, float] = {}
    for (time_s, _), score in zip(events, normalized, strict=True):
        if time_s <= 0.0 or time_s >= duration_s:
            continue
        cut_scores[time_s] = max(cut_scores.get(time_s, 0.0), score)
    boundaries = [0.0, *sorted(cut_scores), duration_s]
    moments: list[Moment] = []
    for index in range(len(boundaries) - 1):
        start_s = boundaries[index]
        end_s = boundaries[index + 1]
        score = cut_scores.get(start_s, 0.0)
        if end_s <= start_s or score <= 0.0:
            continue
        moments.append(Moment(start_s=start_s, end_s=end_s, score=score, source=MomentSource.SCENE))
    return tuple(moments)


def _fuse(
    moments: Sequence[Moment],
    *,
    weights: Mapping[MomentSource, float],
) -> tuple[Moment, ...]:
    if not moments:
        return ()
    ordered = sorted(moments, key=lambda moment: (moment.start_s, moment.end_s))
    groups: list[list[Moment]] = []
    current: list[Moment] = []
    current_end = -1.0
    for moment in ordered:
        if current and moment.start_s < current_end:
            current.append(moment)
            current_end = max(current_end, moment.end_s)
        else:
            if current:
                groups.append(current)
            current = [moment]
            current_end = moment.end_s
    if current:
        groups.append(current)
    fused: list[Moment] = []
    for group in groups:
        best: dict[MomentSource, float] = {}
        for moment in group:
            if moment.source is MomentSource.FUSED:
                continue
            best[moment.source] = max(best.get(moment.source, 0.0), moment.score)
        weighted = sum(weights.get(source, 0.0) * score for source, score in best.items())
        total = sum(weights.get(source, 0.0) for source in best)
        score = min(1.0, max(0.0, weighted / total)) if total > 0.0 else 0.0
        fused.append(
            Moment(
                start_s=min(moment.start_s for moment in group),
                end_s=max(moment.end_s for moment in group),
                score=score,
                source=MomentSource.FUSED,
            )
        )
    return tuple(sorted(fused, key=lambda moment: (-moment.score, moment.start_s, moment.end_s)))


def _require_file(video: Path) -> None:
    if not video.is_file():
        msg = f"el vídeo no existe: {video}"
        raise MomentDetectionError(msg)


def _tail(text: str) -> str:
    return text.strip()[-_STDERR_TAIL:]
