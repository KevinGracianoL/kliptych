"""Gestión de estado y puntos de control (checkpointing) para pipelines reanudables.

Permite registrar el progreso etapa por etapa, guardar y cargar checkpoints
en formato JSON de manera atómica, y verificar si una etapa ya fue completada
con su artefacto correspondiente para garantizar idempotencia en la reanudación.
"""

import os
import tempfile
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
    segment_coords: tuple[tuple[float, float], ...] = ()


class PipelineStateManager:
    """Administrador del estado de ejecución y artefactos por etapa."""

    output_dir: Path
    checkpoint_path: Path
    checkpoint: PipelineCheckpoint
    is_loaded: bool

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

    def mark_done(self, stage: str | PipelineStage, artifact: Path) -> None:
        """Marca una etapa como exitosamente completada y persiste el estado.

        Args:
            stage: Nombre de la etapa o miembro de ``PipelineStage``.
            artifact: Ruta al archivo o artefacto generado por la etapa.
        """
        stage_name = str(stage.value if isinstance(stage, PipelineStage) else stage)
        self.checkpoint.stages[stage_name] = StageStatus.DONE
        self.checkpoint.artifacts[stage_name] = str(artifact)
        self.checkpoint.updated_at = datetime.now(UTC).isoformat()
        self.save()

    def mark_failed(self, stage: str | PipelineStage) -> None:
        """Marca una etapa como fallida y persiste el estado.

        Args:
            stage: Nombre de la etapa o miembro de ``PipelineStage``.
        """
        stage_name = str(stage.value if isinstance(stage, PipelineStage) else stage)
        self.checkpoint.stages[stage_name] = StageStatus.FAILED
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
