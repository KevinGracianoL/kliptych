"""Arnes de mutaciones para la instrumentacion de tiempo por fase.

Ejecuta cada mutacion en una COPIA desechable del repo y comprueba que la suite
la mate. Nunca toca el repo de trabajo.

    uv run python scripts/mutaciones_tiempos.py

El control del final no es decorativo: un arnes que declara muerta toda
mutacion es indistinguible de uno que funciona. Si el control no se detecta, el
resultado de este script no vale nada.
"""

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

ORIGEN = Path(r"C:\Users\Kevin\Documents\Kliptych")
DESTINO = Path(r"C:\Users\Kevin\AppData\Local\Temp\opencode\mut_tiempos")

TESTS = (
    "tests/test_checkpoint_timings.py",
    "tests/test_stage_timings.py",
    "tests/test_pipeline.py",
    "tests/test_resumable_pipeline.py",
)

# pytest no llego a ejecutarse: el comando fallo o la recoleccion se rompio.
# No es una mutacion muerta, y no se cuenta como tal.
ERROR_DE_ARRASTRES = 3


@dataclass(frozen=True)
class Mutacion:
    """Una alteracion puntual del codigo de produccion.

    `viejo` es el fragmento que se busca y `nuevo` el que se pone en su lugar.
    Una mutacion cuyo fragmento viejo ya no aparece NO se aplica: eso significa
    que el codigo cambio de forma y la mutacion dejaria de probar lo que
    pretendia.
    """

    nombre: str
    archivo: str
    viejo: str
    nuevo: str


def _mutaciones_acumulacion() -> list[Mutacion]:
    """La suma entre segmentos, y el reloj que la hace observable.

    Returns:
        Las mutaciones de acumulacion y cierre del cronometro.
    """
    return [
        Mutacion(
            nombre="durations: sobrescribe en vez de sumar",
            archivo="src/kliptych/pipeline_state.py",
            viejo="""        self.checkpoint.durations_ms[stage] = (
            self.checkpoint.durations_ms.get(stage, 0) + elapsed_ms
        )""",
            nuevo="""        self.checkpoint.durations_ms[stage] = elapsed_ms""",
        ),
        Mutacion(
            nombre="mark_accumulated no persiste en disco",
            archivo="src/kliptych/pipeline_state.py",
            viejo="""        self._record_duration(stage_name)
        self.checkpoint.updated_at = datetime.now(UTC).isoformat()
        self.save()""",
            nuevo="""        self._record_duration(stage_name)""",
        ),
        Mutacion(
            nombre="mark_accumulated no cierra el cronometro",
            archivo="src/kliptych/pipeline_state.py",
            viejo="""        self._record_duration(stage_name)
        self.checkpoint.updated_at = datetime.now(UTC).isoformat()
        self.save()

    def mark_started""",
            nuevo="""        self._record_duration(stage_name)

    def mark_started""",
        ),
        Mutacion(
            nombre="mark_segment_done no cierra el cronometro del secundario",
            archivo="src/kliptych/pipeline_state.py",
            viejo="""        if is_primary:
            self.mark_done(stage, artifact)
        else:
            self.mark_accumulated(stage)""",
            nuevo="""        if is_primary:
            self.mark_done(stage, artifact)""",
        ),
        Mutacion(
            nombre="mark_segment_failed no cierra el cronometro del secundario",
            archivo="src/kliptych/pipeline_state.py",
            viejo="""        if is_primary:
            self.mark_failed(stage)
        else:
            self.mark_accumulated(stage)""",
            nuevo="""        self.mark_failed(stage)""",
        ),
    ]


def _mutaciones_reloj() -> list[Mutacion]:
    """El reloj de pared puede saltar atras; el monotónico no.

    Returns:
        Las mutaciones de reloj, una por modulo que mide.
    """
    return [
        Mutacion(
            nombre="reloj de pared en pipeline_state",
            archivo="src/kliptych/pipeline_state.py",
            viejo="elapsed_ms = int((time.monotonic() - started) * 1000)",
            nuevo="elapsed_ms = int((time.time() - started) * 1000)",
        ),
        Mutacion(
            nombre="reloj de pared en pipeline._fase",
            archivo="src/kliptych/pipeline.py",
            viejo="elapsed_ms = int((time.monotonic() - started) * 1000)",
            nuevo="elapsed_ms = int((time.time() - started) * 1000)",
        ),
    ]


def _mutaciones_registro() -> list[Mutacion]:
    """Verdades del registro: la invalidacion, el manifiesto y las validaciones.

    Returns:
        Las mutaciones de veracidad del registro.
    """
    return [
        Mutacion(
            nombre="las invalidaciones no borran la duracion",
            archivo="src/kliptych/pipeline_state.py",
            viejo="            _ = self.checkpoint.durations_ms.pop(stage_name, None)",
            nuevo="            pass",
        ),
        Mutacion(
            nombre="el manifiesto no se reescribe si export falla",
            archivo="src/kliptych/pipeline.py",
            viejo="""        _ = _write_manifest(
            context, contract=contract, outputs=outputs, gates=gates)
        raise""",
            nuevo="""        raise""",
        ),
        Mutacion(
            nombre="el checkpoint acepta duraciones negativas",
            archivo="src/kliptych/pipeline_state.py",
            viejo="durations_ms: dict[str, Annotated[int, Field(ge=0)]] = Field(",
            nuevo="durations_ms: dict[str, int] = Field(",
        ),
        Mutacion(
            nombre="el manifiesto no anuncia la version 1.2",
            archivo="src/kliptych/manifest.py",
            viejo='manifest_version: Literal["1.1", "1.2"] = "1.2"',
            nuevo='manifest_version: Literal["1.1", "1.2"] = "1.1"',
        ),
    ]


def _mutaciones_cableado() -> list[Mutacion]:
    """Una por etapa del orquestador.

    `PipelineStateManager` sabe medir pero no mide nada por si mismo: si falta UN
    `mark_started`, esa etapa no aparece en el registro y el resto de la suite
    sigue verde. Por eso hay una mutacion por etapa y no una sola.

    Returns:
        Seis mutaciones, una por etapa del orquestador.
    """
    etapas = ("DOWNLOAD", "TRANSCRIBE", "MOMENTS", "SELECT", "REFRAME", "SUBTITLES")
    return [
        Mutacion(
            nombre=f"orchestrator sin mark_started({etapa})",
            archivo="src/kliptych/orchestrator.py",
            viejo=f"    state.mark_started(PipelineStage.{etapa})",
            nuevo="",
        )
        for etapa in etapas
    ]


def _preparar() -> Path:
    """Copia el repo a un directorio desechable, sin estado de git ni artefactos.

    Returns:
        La raiz de la copia, que el llamador usa como directorio de trabajo.
    """
    if DESTINO.exists():
        _ = shutil.rmtree(DESTINO)
    _ = shutil.copytree(
        ORIGEN,
        DESTINO,
        ignore=shutil.ignore_patterns(".git", ".venv", "runs", "__pycache__", ".pytest_cache"),
    )
    return DESTINO


def _aplicar(raiz: Path, mutacion: Mutacion) -> bool:
    """Sustituye el fragmento viejo por el nuevo, una sola vez.

    Args:
        raiz: Copia desechable.
        mutacion: La alteracion a aplicar.

    Returns:
        `False` si el fragmento viejo ya no aparece en el archivo.
    """
    objetivo = raiz / mutacion.archivo
    texto = objetivo.read_text(encoding="utf-8")
    if mutacion.viejo not in texto:
        return False
    _ = objetivo.write_text(texto.replace(mutacion.viejo, mutacion.nuevo, 1), encoding="utf-8")
    return True


def _correr_tests(raiz: Path) -> tuple[int, str]:
    """Ejecuta la suite enlazada y clasifica el resultado.

    El codigo de salida de pytest no basta: un comando que no llega a correr y
    una suite que muere por recoleccion sale ambos con 2, y reportar eso como
    "mutacion muerta" seria escribir una ausencia como un cero. Por eso se exige
    la linea de resumen de pytest; si no aparece, el resultado no es valido y se
    reporta como ERROR, que no cuenta como muerte.

    Args:
        raiz: Copia desechable donde corre la suite.

    Returns:
        Codigo de salida y la ultima linea de la salida de pytest.
    """
    proceso = subprocess.run(  # ruff: ignore[subprocess-without-shell-equals-true]
        ["uv", "run", "pytest", *TESTS, "-q", "--no-cov"],  # ruff: ignore[start-process-with-partial-path]
        cwd=raiz,
        capture_output=True,
        text=True,
        check=False,
    )
    lineas = proceso.stdout.strip().splitlines()
    con_resumen = ("passed", "error", "failed")
    resumen = next(
        (linea for linea in reversed(lineas) if any(clave in linea for clave in con_resumen)),
        "",
    )
    if not resumen:
        return ERROR_DE_ARRASTRES, "pytest no llego a ejecutarse"
    return proceso.returncode, resumen


def _control(raiz: Path) -> tuple[int, str]:
    """Rompe una asercion a proposito y comprueba que el arnes la ve.

    Args:
        raiz: Copia desechable ya creada.

    Returns:
        El codigo de salida de la suite y su resumen.
    """
    objetivo = raiz / "tests/test_checkpoint_timings.py"
    texto = objetivo.read_text(encoding="utf-8")
    roto = texto.replace("== 500, en_disco", "== 999999, en_disco")
    _ = objetivo.write_text(roto, encoding="utf-8")
    return _correr_tests(raiz)


def main() -> int:
    """Ejecuta el control y todas las mutaciones, y reporta cada veredicto.

    Returns:
        0 si ninguna mutacion sobrevive y el control se detecta; 1 en otro caso.
    """
    raiz = _preparar()
    mutaciones = (
        *_mutaciones_acumulacion(),
        *_mutaciones_reloj(),
        *_mutaciones_registro(),
        *_mutaciones_cableado(),
    )
    print(f"copia desechable: {raiz}")
    print()

    codigo, resumen = _control(raiz)
    if codigo != 0:
        print(f"CONTROL: DETECTA el fallo  rc={codigo}  {resumen}")
    else:
        print("CONTROL: NO DETECTA EL FALLO -> ARNES ROTO, el resto no vale nada")
    print()

    supervivientes = 0
    for mutacion in mutaciones:
        _ = _preparar()
        if not _aplicar(raiz, mutacion):
            print(f"{mutacion.nombre:52} NO APLICADA: el codigo ya no coincide")
            print()
            supervivientes += 1
            continue
        codigo, resumen = _correr_tests(raiz)
        if codigo == ERROR_DE_ARRASTRES:
            etiqueta = "ERROR (pytest no corrio: no cuenta como muerte)"
            supervivientes += 1
        elif codigo == 0:
            etiqueta = "SUPERVIVIENTE"
            supervivientes += 1
        else:
            etiqueta = "MUERTA"
        print(f"{mutacion.nombre:52} {etiqueta}")
        print(f"  {resumen}")
        print()

    _ = shutil.rmtree(raiz, ignore_errors=True)
    print(f"supervivientes: {supervivientes} de {len(mutaciones)}")
    return 1 if supervivientes else 0


if __name__ == "__main__":
    raise SystemExit(main())
