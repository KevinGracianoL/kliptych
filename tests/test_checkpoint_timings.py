"""Tests del contrato de `durations_ms` en el checkpoint del orquestador.

Viven aparte de `test_resumable_pipeline.py` porque el objeto bajo prueba es el
MODELO del checkpoint y su persistencia, no el cableado del orquestador, que
aquel archivo cubre.
"""

import json
from pathlib import Path
from typing import cast

import pytest
from pydantic import ValidationError

from kliptych.pipeline_state import PipelineCheckpoint, PipelineStage, PipelineStateManager


def _artefacto(tmp_path: Path, nombre: str = "out.mp4") -> Path:
    ruta = tmp_path / nombre
    _ = ruta.write_bytes(b"x")
    return ruta


class RelojPinhado:
    """Reloj monotónico falso con avance EXPLICITO.

    No avanza solo al ser llamado. Un reloj que suma un paso fijo en cada
    `__call__` acopla la cifra esperada al NUMERO de llamadas internas que hace la
    etapa, no al trabajo: cualquier instrumentación futura que añadiera una
    lectura de reloj romperia un test cuyo nombre afirma comprobar la
    acumulación. Con `avanzar()` la cifra depende solo del trabajo que el test
    dice que hubo.
    """

    ahora: float

    def __init__(self, inicio: float = 0.0) -> None:
        self.ahora = inicio

    def avanzar(self, segundos: float) -> None:
        """Avanza el reloj, simulando el paso del tiempo.

        Args:
            segundos: Cuanto avanza.
        """
        self.ahora += segundos

    def __call__(self) -> float:
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
    reloj.avanzar(0.250)
    _ = manager.mark_done(PipelineStage.REFRAME, _artefacto(tmp_path, "primario.mp4"))
    assert PipelineStateManager.load(tmp_path).checkpoint.durations_ms["reframe"] == 250

    # Secundario que falla: 250 ms mas, y despues el proceso muere.
    _ = manager.mark_started(PipelineStage.REFRAME)
    reloj.avanzar(0.250)
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
    reloj avanza de forma explicita para que acumular (5 x 250 ms) y sobrescribir
    (250 ms) den numeros distintos y distinguibles.
    """
    reloj = RelojPinhado()
    monkeypatch.setattr("time.monotonic", reloj)
    manager = PipelineStateManager(tmp_path)

    for indice in range(5):
        _ = manager.mark_started(PipelineStage.SUBTITLES)
        reloj.avanzar(0.250)
        _ = manager.mark_segment_done(
            PipelineStage.SUBTITLES, _artefacto(tmp_path, f"seg{indice}.mp4"), is_primary=False
        )

    assert manager.checkpoint.durations_ms["subtitles"] == 1250


def test_la_persistencia_se_lee_del_archivo_que_el_test_elige(tmp_path: Path) -> None:
    """El archivo comprobado lo nombra el test, no el helper.

    Con el directorio por defecto `checkpoint.json` es el UNICO archivo que
    existe, y una escritura mal dirigida pasaria desapercibida. Aqui la ruta es
    explicita y se comprueba ademas que `load()` ve lo mismo que el JSON crudo, leido
    sin pasar por el helper que se intenta fiscalizar.
    """
    destino = tmp_path / "corrida-con-nombre-explicito"
    _ = destino.mkdir()
    manager = PipelineStateManager(destino)
    artefacto = destino / "final.mp4"
    _ = artefacto.write_bytes(b"x")

    _ = manager.mark_started(PipelineStage.REFRAME)
    _ = manager.mark_segment_failed(PipelineStage.REFRAME, is_primary=False)

    ruta = destino / "checkpoint.json"
    assert ruta.is_file(), f"no se escribio {ruta}"
    crudo = cast("dict[str, object]", json.loads(ruta.read_text(encoding="utf-8")))
    duraciones = cast("dict[str, int]", crudo["durations_ms"])

    assert "reframe" in duraciones
    assert PipelineStateManager.load(destino).checkpoint.durations_ms == duraciones


def test_el_primario_reejecutado_no_duplica_su_tiempo(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`reframe` se suma entre segmentos de UNA corrida, no entre corridas.

    El caso que la revision encontro: si un secundario falla, `segment_coords`
    queda vacio, la reanudacion invalida los artefactos y el primario se vuelve a
    renderizar. Ese primario produjo UN artefacto en cada corrida y solo sobrevive
    el de la segunda, asi que su tiempo de la primera no puede seguir sumando.
    """
    reloj = RelojPinhado()
    monkeypatch.setattr("time.monotonic", reloj)
    manager = PipelineStateManager(tmp_path)

    for indice in range(2):
        _ = manager.mark_started(PipelineStage.REFRAME)
        reloj.avanzar(10.0)
        _ = manager.mark_segment_done(
            PipelineStage.REFRAME, tmp_path / f"seg{indice}.mp4", is_primary=(indice == 0)
        )
    assert manager.checkpoint.durations_ms["reframe"] == 20_000

    reanudada = PipelineStateManager(tmp_path)
    for indice in range(2):
        _ = reanudada.mark_started(PipelineStage.REFRAME)
        reloj.avanzar(10.0)
        _ = reanudada.mark_segment_done(
            PipelineStage.REFRAME, tmp_path / f"r{indice}.mp4", is_primary=(indice == 0)
        )

    assert reanudada.checkpoint.durations_ms["reframe"] == 20_000


def test_los_segmentos_de_la_misma_corrida_siguen_sumando(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """La suma entre segmentos es lo que hace la cifra utilizable para #91.

    Si el reinicio borrara en cada segmento, `reframe` seria el tiempo de un solo
    clip y la pregunta de coste no tendria respuesta.
    """
    reloj = RelojPinhado()
    monkeypatch.setattr("time.monotonic", reloj)
    manager = PipelineStateManager(tmp_path)

    for indice in range(8):
        _ = manager.mark_started(PipelineStage.SUBTITLES)
        reloj.avanzar(1.0)
        _ = manager.mark_segment_done(
            PipelineStage.SUBTITLES, tmp_path / f"s{indice}.mp4", is_primary=(indice == 0)
        )

    assert manager.checkpoint.durations_ms["subtitles"] == 8_000


def test_una_etapa_reiniciada_no_hereda_el_intento_anterior(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Reanudar tras un fallo sin cambio de hash no invalida, y aun asi se descarta.

    Es el caso que `_invalidate` no cubre: el intento fallido no produjo el
    artefacto que sobrevive, asi que su tiempo no puede seguir en la cifra.
    """
    reloj = RelojPinhado()
    monkeypatch.setattr("time.monotonic", reloj)
    artefacto = tmp_path / "transcript.json"
    _ = artefacto.write_bytes(b"{}")

    primera = PipelineStateManager(tmp_path)
    _ = primera.mark_started(PipelineStage.TRANSCRIBE)
    reloj.avanzar(10.0)
    _ = primera.mark_failed(PipelineStage.TRANSCRIBE)
    assert primera.checkpoint.durations_ms["transcribe"] == 10_000

    segunda = PipelineStateManager(tmp_path)
    _ = segunda.mark_started(PipelineStage.TRANSCRIBE)
    reloj.avanzar(7.0)
    _ = segunda.mark_done(PipelineStage.TRANSCRIBE, artefacto)

    assert segunda.checkpoint.durations_ms["transcribe"] == 7_000


def test_el_tiempo_total_no_se_confunde_con_el_por_fase(tmp_path: Path) -> None:
    """`updated_at - created_at` es el total con reintentos; `durations_ms` es el reparto.

    La division de trabajo queda escrita en un test y no solo en un comentario: el
    total de reloj de pared ya existe, asi que no hace falta un campo nuevo.
    """
    manager = PipelineStateManager(tmp_path)
    artefacto = tmp_path / "final.mp4"
    _ = artefacto.write_bytes(b"x")
    _ = manager.mark_started(PipelineStage.SUBTITLES)
    _ = manager.mark_done(PipelineStage.SUBTITLES, artefacto)
    crudo = cast(
        "dict[str, object]",
        json.loads((tmp_path / "checkpoint.json").read_text(encoding="utf-8")),
    )

    assert crudo["created_at"] != crudo["updated_at"]
    assert isinstance(crudo["durations_ms"], dict)
