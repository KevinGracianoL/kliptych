"""Tests del contrato de `stage_durations_ms` en el manifiesto de corrida.

Viven aqui y no junto al resto porque el objeto bajo prueba es el MODELO del
manifiesto: que los valores que acepta y que los que no puede contener. El
cableado de `pipeline.py` se prueba en `test_pipeline.py`.
"""

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from kliptych.environment import EnvironmentReport, GpuInfo
from kliptych.manifest import RunManifest

_ARRANQUE = datetime(2026, 9, 22, 10, 0, tzinfo=UTC)


def _entorno() -> EnvironmentReport:
    return EnvironmentReport(
        ffmpeg_version="N-118380",
        nvenc_available=False,
        gpu=GpuInfo(name="GTX 1650 Ti", vram_mib=4096, driver_version="610.74"),
    )


def _manifiesto(**overrides: object) -> RunManifest:
    defaults: dict[str, object] = {
        "run_id": "run-001",
        "started_at": _ARRANQUE,
        "environment": _entorno(),
    }
    defaults.update(overrides)
    return RunManifest.model_validate(defaults)


def test_el_manifiesto_rechaza_duraciones_negativas() -> None:
    """Un tiempo negativo es imposible: el reloj que lo produce es monotonico.

    Si alguna vez apareciera, seria un defecto del instrumento y no una medida.
    Por eso el campo lo rechaza en vez de almacenarlo y dejar que alguien lo lea
    como un dato.
    """
    with pytest.raises(ValidationError):
        _ = _manifiesto(stage_durations_ms={"assemble": -1})


def test_el_manifiesto_rechaza_una_duracion_negativa_entre_varias() -> None:
    """Una sola entrada invalida el documento entero; no hay que filtrarlas."""
    with pytest.raises(ValidationError):
        _ = _manifiesto(stage_durations_ms={"assemble": 10, "export": -5})


def test_el_manifiesto_conserva_las_duraciones() -> None:
    manifest = _manifiesto(stage_durations_ms={"assemble": 1234, "export": 56})

    reloaded = RunManifest.model_validate(manifest.model_dump())

    assert reloaded.stage_durations_ms == {"assemble": 1234, "export": 56}


def test_el_manifiesto_acepta_un_cero_legitimo() -> None:
    """Cero puede ser real: una fase que se midió y fue instantanea.

    Lo que se rechaza es lo imposible, no lo raro. Confundir los dos seria
    descartar una medida valida por miedo a una invalida.
    """
    assert _manifiesto(stage_durations_ms={"assemble": 0}).stage_durations_ms == {"assemble": 0}


def test_un_manifiesto_sin_instrumentar_sigue_siendo_valido() -> None:
    """Las corridas anteriores a este campo tienen que seguir leyendose.

    Migrar el esquema no puede convertir un archivo viejo en un error de
    validacion: seria romper el registro historico para acomodar el nuevo.
    """
    assert _manifiesto().stage_durations_ms == {}
