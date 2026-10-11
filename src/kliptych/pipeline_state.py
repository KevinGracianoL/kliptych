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
from typing import Annotated, ClassVar

from pydantic import BaseModel, ConfigDict, Field

from kliptych.degradations import activas

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
    # Degradaciones EN CALIENTE de la corrida: lo que se pidio y lo que se uso
    # de verdad. Distintas de las de pre-vuelo de `EnvironmentReport`, que solo
    # ven el estado antes de arrancar. Se rellenan en cada `save()` leyendo el
    # colector activo, para que el archivo nunca pueda afirmar un encoder que no
    # codifico el artefacto.
    degradations: tuple[str, ...] = ()
    # Milisegundos que tardo cada etapa. Sin esto el registro dice QUE se hizo y
    # nunca CUANTO costo, y sin ese numero no se puede decidir si una etapa es el
    # cuello de botella: una mitigacion que suma 20 min por corrida vale distinta
    # cosa sobre un pipeline de 40 min que sobre uno de 4 horas.
    #
    # QUANTO mide, para quien lo lea sin conocer este PR:
    #   `reframe` y `subtitles` suman TODOS los segmentos de la corrida, porque
    #   N segmentos son N trabajos distintos y la pregunta es cuanto tardo el
    #   paquete. `stages` y `artifacts` los describe el segmento PRIMARIO, que es
    #   un solo archivo. El numero de segmentos se deduce de `segment_coords`
    #   mientras la corrida termina bien.
    #   Mide el COSTE DEL ARTEFACTO QUE SOBREVIVE, no el gasto total: un intento
    #   fallido o invalidado no suma. El gasto total de reloj de la corrida,
    #   reintentos incluidos, sigue en `updated_at - created_at`.
    #
    # Una etapa reanudada desde cache NO aparece: no se ejecuto. Escribir 0 seria
    # una cifra, y una cifra se lee como medicion.
    #
    # `ge=0` igual que en `RunManifest.stage_durations_ms`. Los dos archivos son el
    # registro de reproducibilidad y un tiempo negativo no es una medida extrema:
    # con reloj monotonico es imposible, asi que si aparece es un defecto. Dejar
    # el checkpoint sin la restriccion mientras el manifiesto la exige permitiria
    # que un numero imposible llegara a uno de los dos artefactos y no al otro.
    durations_ms: dict[str, Annotated[int, Field(ge=0)]] = Field(default_factory=dict)


class PipelineStateManager:
    """Administrador del estado de ejecución y artefactos por etapa."""

    output_dir: Path
    checkpoint_path: Path
    checkpoint: PipelineCheckpoint
    is_loaded: bool
    # Instantes monotónicos de arranque, por etapa. Vive en memoria a proposito:
    # `time.monotonic()` no tiene sentido entre dos procesos, asi que persistirlo
    # daria duraciones que miden el reloj, no el trabajo.
    _started_at: dict[str, float]
    # Etapas que YA arrancaron en este proceso. Distingue "segundo segmento de la
    # misma corrida, que suma" de "reanudacion, que descarta": sin este conjunto
    # la politica de coste del artefacto vivo no se puede aplicar.
    _arrancadas_en_este_proceso: set[str]

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
        self._arrancadas_en_este_proceso = set()
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
        self._arrancadas_en_este_proceso.clear()
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

    def update_segment_coord(self, index: int, coord: tuple[float, float]) -> None:
        """Registra y persiste la coordenada de UN segmento ya completado.

        Existe porque ``set_segment_coords`` solo corre al final del bucle de
        segmentos: si un segmento secundario falla, ese bucle aborta y el
        checkpoint se queda con ``segment_coords`` vacío. En el resume
        siguiente ``coords_match`` es falso para todos los segmentos y
        ``_invalidate_segment_artifacts`` borra el trabajo ya terminado.

        Persistir en el momento en que cada segmento completa hace que el
        checkpoint refleje exactamente lo que ya está en disco.

        Args:
            index: Posición del segmento en la selección.
            coord: Par ``(start_s, end_s)`` del segmento.
        """
        coords = list(self.checkpoint.segment_coords)
        par = (float(coord[0]), float(coord[1]))
        if 0 <= index < len(coords):
            coords[index] = par
        else:
            coords.append(par)
        self.checkpoint.segment_coords = tuple(coords)
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

    def _invalidate(self, stages: Sequence[PipelineStage]) -> None:
        """Borra estado, artefacto y duracion de las etapas dadas.

        `durations_ms` mide el COSTE DEL ARTEFACTO QUE SOBREVIVE. Un intento
        descartado no contribuye, y por eso su duracion se borra junto con el
        estado y el artefacto: las tres cosas describen el MISMO trabajo. Si se
        conservara, una etapa reejecutada sumaria su intento muerto con el vivo, y
        el checkpoint anunciaria una duracion sobre un archivo que ya no existe.

        La invalidacion no es la UNICA via por la que se descarta un intento.
        Reanudar tras un fallo no invalida nada, porque no cambio el contenido de
        la entrada, y aun asi el intento fallido no produjo el artefacto que
        sobrevive. Por eso `mark_started` tambien descarta, en su primer arranque
        dentro de un proceso. Las dos reglas son la misma politica aplicada a los
        dos caminos por los que un intento deja de contar.

        El total de reloj de pared de la corrida, reintentos incluidos, NO se
        pierde: sigue en la diferencia entre `created_at` y `updated_at`.

        Args:
            stages: Etapas cuyo trabajo deja de ser valido.
        """
        for stage in stages:
            stage_name = stage.value
            _ = self.checkpoint.stages.pop(stage_name, None)
            _ = self.checkpoint.artifacts.pop(stage_name, None)
            _ = self.checkpoint.durations_ms.pop(stage_name, None)

    def invalidate_audio_dependents(self) -> None:
        """Invalida las etapas que dependen de la pista de audio externa."""
        self._invalidate((PipelineStage.SUBTITLES, PipelineStage.COMPLETED))
        self.checkpoint.updated_at = datetime.now(UTC).isoformat()
        self.save()

    def invalidate_downstream_stages(self) -> None:
        """Invalida las etapas dependientes del video fuente."""
        self._invalidate(
            (
                PipelineStage.TRANSCRIBE,
                PipelineStage.MOMENTS,
                PipelineStage.SELECT,
                PipelineStage.REFRAME,
                PipelineStage.SUBTITLES,
                PipelineStage.COMPLETED,
            )
        )
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
        self._record_duration(stage_name)
        self.checkpoint.updated_at = datetime.now(UTC).isoformat()
        self.save()

    def mark_segment_done(
        self, stage: str | PipelineStage, artifact: Path, *, is_primary: bool
    ) -> None:
        """Cronometra una etapa por segmentos sin perder de vista al primario.

        `reframe` y `subtitles` corren una vez por segmento, pero el checkpoint
        tiene UNA entrada por etapa y esa entrada describe el segmento PRINCIPAL.
        Un secundario no puede llamar a `mark_done`: se quedaria con el estado y el
        artefacto de la etapa, y con ocho clips el artefacto registrado seria el
        del ultimo, no el del primario.

        El tiempo si se suma en todos: N segmentos son N trabajos distintos, y lo
        que se quiere responder es "cuanto tardo en renderizarse el paquete", no
        "cuanto tardo el primer clip".

        Args:
            stage: Etapa cronometrada.
            artifact: Artefacto del segmento, solo si es el primario.
            is_primary: Si este segmento es el dueno del estado de la etapa.
        """
        if is_primary:
            self.mark_done(stage, artifact)
        else:
            self.mark_accumulated(stage)

    def mark_segment_failed(self, stage: str | PipelineStage, *, is_primary: bool) -> None:
        """Marca el fallo de un segmento cronometrandolo, sea primario o no.

        Un intento fallido de un secundario tambien consumio tiempo real. Si no se
        contara, el registro subestimaria el coste de la etapa justo en el caso en
        que mas caro sale.

        Args:
            stage: Etapa que fallo.
            is_primary: Si el segmento fallido es el dueno del estado de la etapa.
        """
        if is_primary:
            self.mark_failed(stage)
        else:
            self.mark_accumulated(stage)

    def mark_accumulated(self, stage: str | PipelineStage) -> None:
        """Acumula la duracion de una etapa SIN tomar posesion de su estado.

        `reframe` y `subtitles` corren una vez por segmento, pero el checkpoint
        tiene UNA sola entrada por etapa, y esa entrada describe el segmento
        principal. Un segmento secundario no puede usar `mark_done`: haria que ese
        segmento fuera el dueno del estado y del artefacto de la etapa, que es
        exactamente lo que `stages` y `artifacts` significan.

        El tiempo si se suma: N segmentos son N trabajos distintos, y lo que se
        quiere responder es "cuanto tardo en renderizarse el paquete", no "cuanto
        tardo el primer clip".

        PERSISTE, y no por simetria con las otras marcas sino porque es lo unico
        que hace visible el peor caso. Un secundario que falla va seguido de un
        `raise`: si esta suma viviera solo en memoria, el proceso se desarmaria y
        el tiempo quemado por ese segmento se perderia sin dejar rastro. El
        comentario que dice "el peor caso tiene que quedar visible" seria falso
        sin este `save`.

        Args:
            stage: Nombre de la etapa o miembro de ``PipelineStage``.
        """
        stage_name = str(stage.value if isinstance(stage, PipelineStage) else stage)
        self._record_duration(stage_name)
        self.checkpoint.updated_at = datetime.now(UTC).isoformat()
        self.save()

    def mark_started(self, stage: str | PipelineStage) -> None:
        """Marca el instante de arranque de una etapa para medir su duracion.

        Registrar el INICIO y no solo el final es lo que hace el tiempo atribuible
        a una etapa: dos etapas que empiezan a la vez siguen teniendo duraciones
        distintas.

        La PRIMERA vez que una etapa arranca en este proceso descarta la duracion
        que traia del checkpoint. Esa cifra es de una corrida anterior, y bajo la
        politica de coste del artefacto vivo un intento que no produjo el
        artefacto que sobrevive no cuenta. Sin esto, reanudar tras un fallo sumaria
        el intento muerto con el vivo, que es justo lo que `_invalidate` ya evita
        cuando la invalidacion si ocurre.

        Las siguientes arrancadas de la MISMA etapa en el mismo proceso NO
        descartan nada: ahi si se esta sumando el trabajo de varios segmentos,
        que es lo que hace la cifra utilizable para pregunta cuanto tardo el
        paquete.

        No persiste el arranque. `time.monotonic()` no significa nada en otro
        proceso, asi que guardarlo seria una promesa que una reanudacion no
        podria cumplir.

        Args:
            stage: Nombre de la etapa o miembro de ``PipelineStage``.
        """
        stage_name = str(stage.value if isinstance(stage, PipelineStage) else stage)
        if stage_name not in self._arrancadas_en_este_proceso:
            _ = self.checkpoint.durations_ms.pop(stage_name, None)
            self._arrancadas_en_este_proceso.add(stage_name)
        self._started_at[stage_name] = time.monotonic()

    def _record_duration(self, stage: str) -> None:
        """Suma al registro la duracion de una etapa si se marco su arranque.

        Una etapa reanudada desde cache no llega aqui: no se llamo `mark_started`,
        no hay arranque que fechar y por tanto no hay duracion que registrar. Esa
        es la diferencia entre "tardo 0 ms", que es una cifra, y "no se ejecuto",
        que es un hecho.

        Args:
            stage: Nombre de la etapa, ya normalizado a ``str``.
        """
        started = self._started_at.pop(stage, None)
        if started is None:
            return
        elapsed_ms = int((time.monotonic() - started) * 1000)
        self.checkpoint.durations_ms[stage] = (
            self.checkpoint.durations_ms.get(stage, 0) + elapsed_ms
        )

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
        colector = activas()
        if colector is not None:
            self.checkpoint.degradations = colector.como_tupla()
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
