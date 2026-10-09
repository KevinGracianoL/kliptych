"""Tests del contrato de `durations_ms` en el checkpoint del orquestador.

Viven aparte de `test_resumable_pipeline.py` porque el objeto bajo prueba es el
MODELO del checkpoint y su persistencia, no el cableado del orquestador, que
aquel archivo cubre.
"""

from pathlib import Path

import pytest
from pydantic import ValidationError

from kliptych.pipeline_state import PipelineCheckpoint, PipelineStage, PipelineStateManager


def _artefacto(tmp_path: Path, nombre: str = "out.mp4") -> Path:
    ruta = tmp_path / nombre
    _ = ruta.write_bytes(b"x")
    return ruta


class RelojPinhado:
    """Reloj monotónico falso, con paso fijo por llamada.

    Los tests de duración necesitan cifras EXACTAS y no nulas. Con un reloj real,
    cuatro segmentos de microsegundos se acumulan a 0 ms y no hay forma de
    distinguir "acumulo" de "sobrescribi": ambos dan 0. Un paso fijo de 250 ms
    hace que accumulate y overwrite produzcan numeros distintos y observables.
    """

    ahora: float
    paso_s: float

    def __init__(self, inicio: float = 0.0, *, paso_s: float = 0.250) -> None:
        self.ahora = inicio
        self.paso_s = paso_s

    def __call__(self) -> float:
        self.ahora += self.paso_s
        return self.ahora


def test_el_checkpoint_rechaza_duraciones_negativas() -> None:
    """El checkpoint y el manifiesto exigen lo mismo: un negativo no es una medida.

    Con reloj monotónico no puede ocurrir, así que si aparece es un defecto del
    instrumento. Dejar el checkpoint sin la restricción que sí tiene el manifiesto
    permitiría que un número imposible llegara a uno de los dos artefactos.
    """
    checkpoint = PipelineCheckpoint(
        run_id="r1",
        stages={},
        artifacts={},
        created_at="2026-01-01T00:00:00+00:00",
        updated_at="2026-01-01T00:00:00+00:00",
    )

    with pytest.raises(ValidationError):
        _ = PipelineCheckpoint.model_validate(
            {
                "run_id": "r1",
                "created_at": "2026-01-01T00:00:00+00:00",
                "updated_at": "2026-01-01T00:00:00+00:00",
                "durations_ms": {"download": -10},
            }
        )
    assert checkpoint.durations_ms == {}


def test_el_checkpoint_admite_un_cero_legitimo() -> None:
    """Cero puede ser real; lo imposible es lo negativo, no lo raro."""
    valido = PipelineCheckpoint.model_validate(
        {
            "run_id": "r1",
            "created_at": "2026-01-01T00:00:00+00:00",
            "updated_at": "2026-01-01T00:00:00+00:00",
            "durations_ms": {"download": 0},
        }
    )

    assert valido.durations_ms == {"download": 0}


def test_la_duracion_de_un_secundario_fallido_llega_al_disco(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """El peor caso tiene que quedar en el ARTEFACTO, no en la memoria del proceso.

    Este es el test que hace falta y el que faltaba. Cuando un segmento
    secundario falla, `mark_segment_failed` va seguido de un `raise`: si la suma
    viviera solo en RAM, el proceso se desarmaria y el tiempo quemado se perderia
    sin dejar rastro. Asertar el estado en memoria no lo detecta, porque la
    memoria si lo tiene.
    """
    reloj = RelojPinhado()
    monkeypatch.setattr("time.monotonic", reloj)
    manager = PipelineStateManager(tmp_path)

    # Primario: 250 ms, queda persistido por mark_done.
    _ = manager.mark_started(PipelineStage.REFRAME)
    _ = manager.mark_done(PipelineStage.REFRAME, _artefacto(tmp_path, "primario.mp4"))
    assert PipelineStateManager.load(tmp_path).checkpoint.durations_ms["reframe"] == 250

    # Secundario que falla: 250 ms mas, y despues el proceso muere.
    _ = manager.mark_started(PipelineStage.REFRAME)
    _ = manager.mark_segment_failed(PipelineStage.REFRAME, is_primary=False)

    en_disco = PipelineStateManager.load(tmp_path).checkpoint.durations_ms

    assert en_disco["reframe"] == 500, en_disco


def test_la_duracion_acumulada_de_varios_segmentos_es_la_suma(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """La suma tiene que ser observable como suma, no como presencia.

    Un `assert clave in durations_ms` no puede fallar: pasa igual si el codigo
    acumula, si sobrescribe y si no hace nada. Con reloj real ademas el problema
    es invisible, porque cada segmento dura 0 ms y la suma de ceros es cero. El
    paso fijo hace que acumular (5 x 250) y sobrescribir (250) den numeros
    distintos, y por tanto distinguibles.
    """
    reloj = RelojPinhado()
    monkeypatch.setattr("time.monotonic", reloj)
    manager = PipelineStateManager(tmp_path)

    for indice in range(5):
        _ = manager.mark_started(PipelineStage.SUBTITLES)
        _ = manager.mark_segment_done(
            PipelineStage.SUBTITLES, _artefacto(tmp_path, f"seg{indice}.mp4"), is_primary=False
        )

    assert manager.checkpoint.durations_ms["subtitles"] == 1250


def test_el_primario_tambien_suma_a_los_secundarios(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """La sumaThrough todas las etapas, primaria incluida.

    La pregunta que responde el registro es "cuanto tardo en renderizarse el
    paquete", asi que el ultimo segmento que se cronometra tambien cuenta.
    """
    reloj = RelojPinhado()
    monkeypatch.setattr("time.monotonic", reloj)
    manager = PipelineStateManager(tmp_path)

    for indice in range(3):
        _ = manager.mark_started(PipelineStage.REFRAME)
        _ = manager.mark_segment_done(
            PipelineStage.REFRAME, _artefacto(tmp_path, f"seg{indice}.mp4"), is_primary=False
        )
    _ = manager.mark_started(PipelineStage.REFRAME)
    primario = _artefacto(tmp_path, "p.mp4")
    _ = manager.mark_segment_done(PipelineStage.REFRAME, primario, is_primary=True)

    assert manager.checkpoint.durations_ms["reframe"] == 1000
    assert manager.checkpoint.artifacts["reframe"].endswith("p.mp4")
