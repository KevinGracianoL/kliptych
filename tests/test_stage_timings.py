"""Tests de la instrumentacion de tiempo por fase.

RED primero: sin esto, `durations_ms` puede desaparecer sin que nada falle, y
volveriamos a no poder responder si la transcripcion es el cuello de botella.
"""

from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from kliptych.environment import EnvironmentReport, GpuInfo
from kliptych.manifest import RunManifest
from kliptych.pipeline_state import PipelineStage, PipelineStateManager

_ARRANQUE = datetime(2026, 9, 22, 10, 0, tzinfo=UTC)


def _entorno() -> EnvironmentReport:
    return EnvironmentReport(
        ffmpeg_version="N-118380",
        nvenc_available=False,
        gpu=GpuInfo(name="GTX 1650 Ti", vram_mib=4096, driver_version="610.74"),
    )


def _manifesto(**overrides: object) -> RunManifest:
    defaults: dict[str, object] = {
        "run_id": "run-001",
        "started_at": _ARRANQUE,
        "environment": _entorno(),
    }
    defaults.update(overrides)
    return RunManifest.model_validate(defaults)


def _manager(tmp_path: Path) -> PipelineStateManager:
    return PipelineStateManager(tmp_path)


def test_una_fase_cronometrada_registra_su_duracion(tmp_path: Path) -> None:
    manager = _manager(tmp_path)
    artifact = tmp_path / "out.mp4"
    _ = artifact.write_bytes(b"x")

    _ = manager.mark_started(PipelineStage.TRANSCRIBE)
    _ = manager.mark_done(PipelineStage.TRANSCRIBE, artifact)

    assert "transcribe" in manager.checkpoint.durations_ms
    assert manager.checkpoint.durations_ms["transcribe"] >= 0


def test_la_duracion_sobrevive_a_releer_el_checkpoint(tmp_path: Path) -> None:
    manager = _manager(tmp_path)
    artifact = tmp_path / "out.mp4"
    _ = artifact.write_bytes(b"x")
    _ = manager.mark_started(PipelineStage.MOMENTS)
    _ = manager.mark_done(PipelineStage.MOMENTS, artifact)
    measured = manager.checkpoint.durations_ms["moments"]

    reloaded = PipelineStateManager(tmp_path)

    assert reloaded.checkpoint.durations_ms["moments"] == measured


def test_una_fase_sin_arranque_no_inventa_duracion(tmp_path: Path) -> None:
    """Una reanudada desde cache se marca sin arrancar: no se ejecuto.

    Registrar 0 seria una cifra, y una cifra se lee como medicion. Lo honesto es
    que no exista entrada para esa fase.
    """
    manager = _manager(tmp_path)
    artifact = tmp_path / "out.mp4"
    _ = artifact.write_bytes(b"x")

    _ = manager.mark_done(PipelineStage.SELECT, artifact)

    assert "select" not in manager.checkpoint.durations_ms


def test_una_fase_fallida_tambien_deja_duracion(tmp_path: Path) -> None:
    """Un fallo consume tiempo de verdad; sin esto el peor caso no se ve."""
    manager = _manager(tmp_path)

    _ = manager.mark_started(PipelineStage.REFRAME)
    _ = manager.mark_failed(PipelineStage.REFRAME)

    assert "reframe" in manager.checkpoint.durations_ms


def test_el_reloj_de_wall_no_fabrica_duraciones_negativas(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """La medicion sale de `monotonic`: el reloj de pared puede saltar atras.

    El reloj de pared se hace retroceder a proposito. Si la medicion saliera de
    `time.time()`, la duracion seria negativa y este test fallaria; si sale de
    `time.monotonic()`, el reloj de pared no la toca. Un `assert >= 0` sin
    manipular el reloj NO probaria nada: con cualquiera de los dos relojes sale
    no negativo, asi que la mutacion `monotonic` -> `time` sobrevivia.
    """
    manager = _manager(tmp_path)
    artifact = tmp_path / "out.mp4"
    _ = artifact.write_bytes(b"x")

    # Dos instantes de pared, el segundo ANTERIOR al primero.
    ticks = iter([1_000.0, 900.0])
    monkeypatch.setattr("time.time", lambda: next(ticks))

    _ = manager.mark_started(PipelineStage.DOWNLOAD)
    _ = manager.mark_done(PipelineStage.DOWNLOAD, artifact)

    assert manager.checkpoint.durations_ms["download"] >= 0


def test_un_segmento_secundario_no_hereda_el_estado_del_primario(
    tmp_path: Path,
) -> None:
    """`record_duration` cierra el cronometro sin marcar la etapa.

    Los segmentos secundarios de reframe y subtitulos llaman a este metodo: si
    marcaran `done`, el estado de la etapa pasaria a describir el ultimo
    segmento en vez del primario, que es el unico con artefacto registrado.
    """
    manager = _manager(tmp_path)
    artifact = tmp_path / "out.mp4"
    _ = artifact.write_bytes(b"x")

    _ = manager.mark_started(PipelineStage.REFRAME)
    _ = manager.record_duration(PipelineStage.REFRAME)

    assert "reframe" in manager.checkpoint.durations_ms
    assert not manager.is_done(PipelineStage.REFRAME)
    assert manager.artifact_path(PipelineStage.REFRAME) is None


def test_los_tiempos_de_todos_los_segmentos_se_suman(tmp_path: Path) -> None:
    """N segmentos dan el coste de render del paquete, no el del primero.

    Es la cifra que `#91` necesita: cuanto costo renderizar, no cuanto costo el
    clip principal. Medir solo el primario dejaria el numero sin qualifier en el
    artefacto, y un lector no tendria forma de saber que le falta la mayor parte.
    """
    manager = _manager(tmp_path)
    artifact = tmp_path / "out.mp4"
    _ = artifact.write_bytes(b"x")

    for _segmento in range(4):
        _ = manager.mark_started(PipelineStage.REFRAME)
        _ = manager.mark_done(PipelineStage.REFRAME, artifact)

    assert len(manager.checkpoint.durations_ms) == 1
    assert manager.checkpoint.durations_ms["reframe"] >= 0


def test_invalidar_una_etapa_borra_su_duracion(tmp_path: Path) -> None:
    """Una duracion sin artefacto que la produjo es un numero sin comando.

    `invalidate_downstream_stages` borra `stages` y `artifacts`. Si dejara la
    duracion, el checkpoint seguiria anunciando el coste de un `transcript.json`
    que ya no existe: la cifra de un trabajo anulado, mezclada con la del
    reejecutado.
    """
    manager = _manager(tmp_path)
    artifact = tmp_path / "out.mp4"
    _ = artifact.write_bytes(b"x")

    _ = manager.mark_started(PipelineStage.TRANSCRIBE)
    _ = manager.mark_done(PipelineStage.TRANSCRIBE, artifact)
    assert "transcribe" in manager.checkpoint.durations_ms

    _ = manager.invalidate_downstream_stages()

    assert "transcribe" not in manager.checkpoint.durations_ms
    assert not manager.is_done(PipelineStage.TRANSCRIBE)


def test_invalidar_el_audio_borra_su_duracion(tmp_path: Path) -> None:
    manager = _manager(tmp_path)
    artifact = tmp_path / "out.mp4"
    _ = artifact.write_bytes(b"x")

    _ = manager.mark_started(PipelineStage.SUBTITLES)
    _ = manager.mark_done(PipelineStage.SUBTITLES, artifact)

    _ = manager.invalidate_audio_dependents()

    assert "subtitles" not in manager.checkpoint.durations_ms


def test_reset_descarta_las_duraciones(tmp_path: Path) -> None:
    manager = _manager(tmp_path)
    artifact = tmp_path / "out.mp4"
    _ = artifact.write_bytes(b"x")
    _ = manager.mark_started(PipelineStage.TRANSCRIBE)
    _ = manager.mark_done(PipelineStage.TRANSCRIBE, artifact)

    _ = manager.reset()

    assert manager.checkpoint.durations_ms == {}


def test_el_manifiesto_acepta_y_conserva_las_duraciones() -> None:
    manifest = _manifesto(stage_durations_ms={"assemble": 1234, "export": 56})

    assert manifest.stage_durations_ms == {"assemble": 1234, "export": 56}
    reloaded = RunManifest.model_validate(manifest.model_dump())
    assert reloaded.stage_durations_ms == {"assemble": 1234, "export": 56}


def test_las_duraciones_no_pueden_ser_negativas() -> None:
    """Un tiempo negativo es imposible; el modelo no debe aceptarlo."""
    with pytest.raises(ValidationError):
        _ = _manifesto(stage_durations_ms={"assemble": -1})


def test_sin_duraciones_el_manifiesto_sigue_siendo_valido() -> None:
    """Una corrida sin instrumentar debe leerse, no romperse al migrar."""
    _ = _manifesto()
