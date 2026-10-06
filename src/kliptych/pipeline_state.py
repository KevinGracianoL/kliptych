"""Gestión de estado y puntos de control (checkpointing) para pipelines reanudables.

Permite registrar el progreso etapa por etapa, guardar y cargar checkpoints
en formato JSON de manera atómica, y verificar si una etapa ya fue completada
con su artefacto correspondiente para garantizar idempotencia en la reanudación.
"""

import os
import tempfile
import time
import uuid
from collections.abc import Sequence
from contextlib import suppress
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import ClassVar

from pydantic import BaseModel, ConfigDict, Field

_CHECKPOINT_FILE = "checkpoint.json"


class PipelineStage(StrEnum):
    """Etapas del ciclo de vida del pipeline."""

    DOWNLOAD = "download"
    TRANSCRIBE = "transcribe"
    MOMENTS = "moments"
    SELECT = "select"
    REFRAME = "reframe"
    SUBTITLES = "subtitles"
    COMPLETED = "completed"


class StageStatus(StrEnum):
    """Estados posibles para una etapa individual."""

    PENDING = "pending"
    DONE = "done"
    FAILED = "failed"


class PipelineCheckpoint(BaseModel):
    """Representación persistente del punto de control del pipeline."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    run_id: str
    stages: dict[str, str] = Field(default_factory=dict)
    artifacts: dict[str, str] = Field(default_factory=dict)
    created_at: str
    updated_at: str
    input_fingerprint: str | None = None
    source_content_hash: str | None = None
    audio_content_hash: str | None = None
    segment_coords: tuple[tuple[float, float], ...] = ()
    # Milisegundos que tardó cada etapa. Sin esto no se puede responder si la
    # transcripción es el cuello de botella del pipeline o si lo es el render: el
    # registro decía QUÉ se hizo y nunca CUÁNTO costó. Medido con reloj monotónico,
    # así que un salto de reloj del sistema no puede fabricar tiempos negativos.
    durations_ms: dict[str, int] = Field(default_factory=dict)


class PipelineStateManager:
    """Administrador del estado de ejecución y artefactos por etapa."""

    output_dir: Path
    checkpoint_path: Path
    checkpoint: PipelineCheckpoint
    is_loaded: bool
    # Instantes monotónicos de arranque por etapa. Vive en memoria y no se
    # persiste: `time.monotonic()` no tiene sentido entre dos procesos.
    _started_at: dict[str, float]

    def __init__(self, output_dir: Path, *, input_fingerprint: str | None = None) -> None:
        """Inicializa el administrador de estado en el directorio de salida.

        Si ya existe un ``checkpoint.json`` en el directorio, se carga
        automáticamente para permitir reanudación transparente. En caso
        contrario, se crea una nueva estructura con un ``run_id`` único.

        Args:
            output_dir: Directorio de salida donde reside el checkpoint.
            input_fingerprint: Fingerprint de las entradas del pipeline.
        """
        self.output_dir = output_dir
        self.checkpoint_path = output_dir / _CHECKPOINT_FILE
        self._started_at = {}
        existing = self._try_load_existing()
        if existing is not None:
            self.checkpoint = existing
            self.is_loaded = True
            return
        self.is_loaded = False
        now = datetime.now(UTC).isoformat()
        self.checkpoint = PipelineCheckpoint(
            run_id=uuid.uuid4().hex,
            stages={},
            artifacts={},
            created_at=now,
            updated_at=now,
            input_fingerprint=input_fingerprint,
        )

    def _try_load_existing(self) -> PipelineCheckpoint | None:
        if not self.checkpoint_path.is_file() or self.checkpoint_path.stat().st_size == 0:
            return None
        with suppress(OSError, ValueError):
            data = self.checkpoint_path.read_text(encoding="utf-8")
            return PipelineCheckpoint.model_validate_json(data)
        return None

    def reset(self, *, input_fingerprint: str | None = None) -> None:
        """Reinicia el punto de control a un estado nuevo y limpio.

        Args:
            input_fingerprint: Opcional fingerprint a persistir en el nuevo
                checkpoint limpio.
        """
        self.is_loaded = False
        self._started_at.clear()
        now = datetime.now(UTC).isoformat()
        self.checkpoint = PipelineCheckpoint(
            run_id=uuid.uuid4().hex,
            stages={},
            artifacts={},
            created_at=now,
            updated_at=now,
            input_fingerprint=input_fingerprint,
        )
        self.save()

    def set_input_fingerprint(self, fingerprint: str) -> None:
        """Asigna y persiste el fingerprint de entradas del pipeline.

        Args:
            fingerprint: El digest SHA-256 de las entradas del pipeline.
        """
        self.checkpoint.input_fingerprint = fingerprint
        self.checkpoint.updated_at = datetime.now(UTC).isoformat()
        self.save()

    def set_segment_coords(self, coords: Sequence[tuple[float, float]]) -> None:
        """Asigna y persiste las coordenadas de los segmentos seleccionados.

        Args:
            coords: Secuencia de pares (start_s, end_s).
        """
        self.checkpoint.segment_coords = tuple((float(s), float(e)) for s, e in coords)
        self.checkpoint.updated_at = datetime.now(UTC).isoformat()
        self.save()

    def set_source_content_hash(self, content_hash: str) -> None:
        """Asigna y persiste la firma de contenido del video fuente.

        Args:
            content_hash: Hash o firma del contenido del archivo fuente.
        """
        self.checkpoint.source_content_hash = content_hash
        self.checkpoint.updated_at = datetime.now(UTC).isoformat()
        self.save()

    def set_audio_content_hash(self, content_hash: str | None) -> None:
        """Asigna y persiste la firma de contenido de la pista de audio.

        Args:
            content_hash: Firma del archivo de audio externo, o None si no hay pista.
        """
        self.checkpoint.audio_content_hash = content_hash
        self.checkpoint.updated_at = datetime.now(UTC).isoformat()
        self.save()

    def invalidate_audio_dependents(self) -> None:
        """Invalida las etapas que dependen de la pista de audio externa."""
        for stage in (PipelineStage.SUBTITLES, PipelineStage.COMPLETED):
            stage_name = stage.value
            _ = self.checkpoint.stages.pop(stage_name, None)
            _ = self.checkpoint.artifacts.pop(stage_name, None)
        self.checkpoint.updated_at = datetime.now(UTC).isoformat()
        self.save()

    def invalidate_downstream_stages(self) -> None:
        """Invalida las etapas dependientes del video fuente."""
        downstream = [
            PipelineStage.TRANSCRIBE,
            PipelineStage.MOMENTS,
            PipelineStage.SELECT,
            PipelineStage.REFRAME,
            PipelineStage.SUBTITLES,
            PipelineStage.COMPLETED,
        ]
        for stage in downstream:
            stage_name = stage.value
            _ = self.checkpoint.stages.pop(stage_name, None)
            _ = self.checkpoint.artifacts.pop(stage_name, None)
        self.checkpoint.segment_coords = ()
        self.checkpoint.updated_at = datetime.now(UTC).isoformat()
        self.save()

    def is_done(self, stage: str | PipelineStage) -> bool:
        """Verifica si una etapa está marcada como completada en el checkpoint.

        Args:
            stage: Nombre de la etapa o miembro de ``PipelineStage``.

        Returns:
            ``True`` si la etapa tiene estado 'done'; ``False`` en caso contrario.
        """
        stage_name = str(stage.value if isinstance(stage, PipelineStage) else stage)
        return self.checkpoint.stages.get(stage_name) == StageStatus.DONE

    def artifact_path(self, stage: str | PipelineStage) -> Path | None:
        """Devuelve la ruta al artefacto de una etapa, si fue registrado.

        Args:
            stage: Nombre de la etapa o miembro de ``PipelineStage``.

        Returns:
            La ruta como ``Path`` o ``None`` si la etapa no tiene artefacto.
        """
        stage_name = str(stage.value if isinstance(stage, PipelineStage) else stage)
        path_str = self.checkpoint.artifacts.get(stage_name)
        if path_str is None:
            return None
        return Path(path_str)

    def mark_started(self, stage: str | PipelineStage) -> None:
        """Marca el instante de arranque de una etapa para medir su duración.

        Registrar el INICIO y no solo el final es lo que hace que el tiempo sea
        atribuible a una etapa y no al reloj de pared: dos etapas que empiezan a
        la vez siguen teniendo durations distintas.

        No persiste nada. El reloj monotónico no sobrevive a un proceso, y un
        ``started_at`` guardado en disco junto al artefacto seria una promesa que
        un resume posterior no puede cumplir.

        Args:
            stage: Nombre de la etapa o miembro de ``PipelineStage``.
        """
        stage_name = str(stage.value if isinstance(stage, PipelineStage) else stage)
        self._started_at[stage_name] = time.monotonic()

    def mark_started_if(self, stage: str | PipelineStage, *, condition: bool) -> None:
        """Arranca el cronómetro de una etapa solo si se cumple la condición.

        Las etapas de reframe y subtitulos se ejecutan una vez por segmento, pero
        el checkpoint tiene UNA sola entrada por etapa: la del segmento principal.
        Los segmentos secundarios no se cronometran porque sumarlos mediría el
        trabajo dos veces, y porque no son la pregunta.

        La condición vive aquí y no en el llamador para que las funciones de etapa
        no acumulen ramas: si el `if` estuviera en el orquestador, cada una cruzaria
        su umbral de complejidad por añadir instrumentación.

        Args:
            stage: Nombre de la etapa o miembro de ``PipelineStage``.
            condition: Si es ``False``, no arranca nada.
        """
        if condition:
            self.mark_started(stage)

    def _record_duration(self, stage: str | PipelineStage) -> None:
        """Acumula la duración de una etapa si se había marcado su arranque.

        Args:
            stage: Nombre de la etapa o miembro de ``PipelineStage``.
        """
        stage_name = str(stage.value if isinstance(stage, PipelineStage) else stage)
        started = self._started_at.pop(stage_name, None)
        if started is None:
            return
        elapsed_ms = int((time.monotonic() - started) * 1000)
        # Una etapa reanudada desde cache no llega aqui: `mark_started` no se llamo,
        # no hay arranque que fechar y por tanto no hay duracion que registrar. Es
        # la diferencia entre "tardo 0 ms" (una cifra) y "no se ejecuto" (un hecho).
        self.checkpoint.durations_ms[stage_name] = (
            self.checkpoint.durations_ms.get(stage_name, 0) + elapsed_ms
        )

    def mark_done(self, stage: str | PipelineStage, artifact: Path) -> None:
        """Marca una etapa como exitosamente completada y persiste el estado.

        Args:
            stage: Nombre de la etapa o miembro de ``PipelineStage``.
            artifact: Ruta al archivo o artefacto generado por la etapa.
        """
        stage_name = str(stage.value if isinstance(stage, PipelineStage) else stage)
        self.checkpoint.stages[stage_name] = StageStatus.DONE
        self.checkpoint.artifacts[stage_name] = str(artifact)
        self._record_duration(stage_name)
        self.checkpoint.updated_at = datetime.now(UTC).isoformat()
        self.save()

    def mark_failed(self, stage: str | PipelineStage) -> None:
        """Marca una etapa como fallida y persiste el estado.

        Args:
            stage: Nombre de la etapa o miembro de ``PipelineStage``.
        """
        stage_name = str(stage.value if isinstance(stage, PipelineStage) else stage)
        self.checkpoint.stages[stage_name] = StageStatus.FAILED
        self._record_duration(stage_name)
        self.checkpoint.updated_at = datetime.now(UTC).isoformat()
        self.save()

    def save(self) -> None:
        """Escribe el checkpoint a disco de forma atómica."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=self.output_dir, prefix=".checkpoint-", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as file:
                _ = file.write(self.checkpoint.model_dump_json(indent=2))
            _ = Path(tmp).replace(self.checkpoint_path)
        except BaseException:
            with suppress(OSError):
                Path(tmp).unlink()
            raise

    @classmethod
    def load(cls, output_dir: Path) -> "PipelineStateManager":
        """Carga un administrador de estado desde un ``checkpoint.json`` existente.

        Args:
            output_dir: Directorio donde debe existir el checkpoint.

        Returns:
            Instancia cargada con los datos del checkpoint.

        Raises:
            FileNotFoundError: Si no existe ``checkpoint.json`` en el directorio.
        """
        checkpoint_file = output_dir / _CHECKPOINT_FILE
        if not checkpoint_file.is_file() or checkpoint_file.stat().st_size == 0:
            msg = f"no se encontró checkpoint en {checkpoint_file}"
            raise FileNotFoundError(msg)
        data = checkpoint_file.read_text(encoding="utf-8")
        checkpoint = PipelineCheckpoint.model_validate_json(data)
        manager = cls(output_dir)
        manager.checkpoint = checkpoint
        return manager
