"""Arnes de mutaciones para la instrumentacion de tiempo por fase.

Ejecuta cada mutacion en una COPIA desechable del repo y comprueba que la suite
la mate. Nunca toca el repo de trabajo.

    uv run python scripts/mutaciones_tiempos.py

Que el arnes no escriba una ausencia como un cero es la condicion de este
script. Tres formas de hacerlo, las tres corregidas aqui:

* Una mutacion cuyo fragmento no aparece **no se aplica**, y eso es ERROR, no
  muerte. Un mutante que no se aplico no prueba nada.
* Una mutacion cuyo fragmento aparece **mas de una vez** se aplica en la
  primera, y se reporta ERROR. Con `replace(..., 1)` contra un patron repetido
  el mutante cae en el sitio que toca primero, que puede no ser el que el
  nombre declara.
* Un codigo de salida que **no es 0 y no es 1** es ERROR: pytest no llego a
  los tests (error de recoleccion, import o sintaxis). Sin su linea de resumen
  no hubo veredicto de la suite, y llamarlo "muerta" seria inventarlo.

El control del principio se corre en cada barrido y, si SOBREVIVE, todos los
resultados de ese barrido quedan anulados. Un arnes que declara muerta toda
mutacion es indistinguible de uno que funciona.
"""

import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

ORIGEN = Path(__file__).resolve().parents[1]

# La escritura del manifiesto en el camino de fallo, tal cual esta en el fuente.
# Vive aqui y no en la declaracion de la mutacion porque la linea excede el
# limite de longitud: partirla dentro del literal de la mutacion ya produjo una
# vez un mutante que no quitaba nada y se reporto como muerto.
_ESCRITURA_EN_FALLO = (
    "        _ = _write_manifest(context, contract=contract, "
    "outputs=outputs, gates=gates)\n        raise"
)

TESTS = (
    "tests/test_checkpoint_timings.py",
    "tests/test_stage_timings.py",
    "tests/test_pipeline.py",
    "tests/test_resumable_pipeline.py",
)


class Veredicto(StrEnum):
    """Lo que se puede afirmar de una mutacion. Los tres son afirmables.

    `ERROR` existe para que no haya que mentir: significa "no se puede decir",
    y no cuenta ni como muerte ni como supervivencia.
    """

    MUERTA = "MUERTA"
    SUPERVIVIENTE = "SUPERVIVIENTE"
    ERROR = "ERROR"
    NO_APLICADA = "NO APLICADA"
    DUPLICADA = "DUPLICADA"
    CONTROL_ROTO = "CONTROL ROTO"


@dataclass(frozen=True)
class Aplicacion:
    """El resultado de intentar aplicar una mutacion, con su evidencia.

    Todo es keyword-only a proposito: `Aplicacion(False, ...)` se leia como una
    respuesta cuando en realidad es una afirmacion, que es justo el modo de
    fallo que este arnes existe para evitar.
    """

    aplicada: bool = field(kw_only=True)
    ocurrencias: int = field(kw_only=True)
    detalle: str = field(kw_only=True)

    @property
    def es_error(self) -> bool:
        """Cero o mas de una ocurrencia es un error, nunca una muerte."""
        return self.ocurrencias != 1


@dataclass(frozen=True)
class Mutacion:
    """Una alteracion puntual del codigo de produccion.

    `ancla` es la firma del metodo dentro del cual debe caer el fragmento. Sin
    ella, un fragmento que se repite en el archivo se aplica en la primera
    ocurrencia, que puede ser otro metodo.

    Attributes:
        nombre: Como se reporta.
        archivo: Ruta relativa a la raiz del repo.
        ancla: Metodo o funcion donde el fragmento debe estar.
        viejo: Texto a sustituir.
        nuevo: Texto que lo sustituye.
    """

    nombre: str
    archivo: str
    ancla: str
    viejo: str
    nuevo: str

    @property
    def identidad(self) -> str:
        """Huella del efecto, para detectar dos nombres con el mismo cuerpo.

        Incluye el texto viejo y el nuevo, no solo el nuevo: seis mutaciones que
        borran una linea distinta tienen el mismo `nuevo` (vacio) y son pruebas
        distintas. Solo son la misma si apuntan al mismo sitio y make lo mismo.
        """
        return f"{self.archivo}::{self.ancla}::{self.viejo}::{self.nuevo}"


def _mutaciones_acumulacion() -> list[Mutacion]:
    """La suma entre segmentos, y el reloj que la hace observable.

    Returns:
        Las mutaciones de acumulacion y cierre del cronometro.
    """
    return [
        Mutacion(
            nombre="durations: sobrescribe en vez de sumar",
            archivo="src/kliptych/pipeline_state.py",
            ancla="def _record_duration",
            viejo="""        self.checkpoint.durations_ms[stage] = (
            self.checkpoint.durations_ms.get(stage, 0) + elapsed_ms
        )""",
            nuevo="""        self.checkpoint.durations_ms[stage] = elapsed_ms""",
        ),
        Mutacion(
            nombre="mark_accumulated no persiste en disco",
            archivo="src/kliptych/pipeline_state.py",
            ancla="def mark_accumulated",
            viejo="""        self.checkpoint.updated_at = datetime.now(UTC).isoformat()
        self.save()""",
            nuevo="""        self.checkpoint.updated_at = datetime.now(UTC).isoformat()
        _ = 0""",
        ),
        Mutacion(
            nombre="mark_accumulated no cierra el cronometro",
            archivo="src/kliptych/pipeline_state.py",
            ancla="def mark_accumulated",
            viejo="""        self._record_duration(stage_name)""",
            nuevo="""        _ = stage_name""",
        ),
        Mutacion(
            nombre="mark_segment_done no cierra el cronometro del secundario",
            archivo="src/kliptych/pipeline_state.py",
            ancla="def mark_segment_done",
            viejo="""            self.mark_accumulated(stage)""",
            nuevo="""            self.mark_accumulated""",
        ),
        Mutacion(
            nombre="mark_segment_failed no cierra el cronometro del secundario",
            archivo="src/kliptych/pipeline_state.py",
            ancla="def mark_segment_failed",
            viejo="""            self.mark_accumulated(stage)""",
            nuevo="""            self.mark_accumulated""",
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
            ancla="def _record_duration",
            viejo="elapsed_ms = int((time.monotonic() - started) * 1000)",
            nuevo="elapsed_ms = int((time.time() - started) * 1000)",
        ),
        Mutacion(
            nombre="reloj de pared en pipeline._fase",
            archivo="src/kliptych/pipeline.py",
            ancla="def _fase",
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
            ancla="def _invalidate",
            viejo="            _ = self.checkpoint.durations_ms.pop(stage_name, None)",
            nuevo="            pass",
        ),
        Mutacion(
            nombre="el manifiesto no se reescribe si export falla",
            archivo="src/kliptych/pipeline.py",
            ancla="def _resolved",
            viejo=_ESCRITURA_EN_FALLO,
            nuevo="        raise",
        ),
        Mutacion(
            nombre="el checkpoint acepta duraciones negativas",
            archivo="src/kliptych/pipeline_state.py",
            ancla="class PipelineCheckpoint",
            viejo="durations_ms: dict[str, Annotated[int, Field(ge=0)]] = Field(",
            nuevo="durations_ms: dict[str, int] = Field(",
        ),
        Mutacion(
            nombre="el manifiesto no anuncia la version 1.2",
            archivo="src/kliptych/manifest.py",
            ancla="class RunManifest",
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
            ancla="",
            viejo=f"    state.mark_started(PipelineStage.{etapa})",
            nuevo="",
        )
        for etapa in etapas
    ]


def _preparar() -> Path:
    """Copia el repo a un directorio temporal, sin git ni artefactos.

    Returns:
        La raiz de la copia, que el llamador usa como directorio de trabajo.
    """
    destino = Path(tempfile.mkdtemp(prefix="mut_tiempos-"))
    _ = shutil.copytree(
        ORIGEN,
        destino,
        dirs_exist_ok=True,
        ignore=shutil.ignore_patterns(".git", ".venv", "runs", "__pycache__", ".pytest_cache"),
    )
    return destino


def _cuerpo_de(texto: str, ancla: str) -> tuple[int, int]:
    """Acota el texto a la region que empieza en el ancla.

    El ancla debe incluir la firma (`def f` o `class F`), no una llamada: si
    fuera una llamada, la region empezaria en el primer uso de ese nombre en el
    archivo, que puede estar en otro metodo.

    Args:
        texto: Contenido completo del archivo.
        ancla: Firma del metodo, clase o funcion que contiene el fragmento.

    Returns:
        El indice de inicio de la region y su indice final, o `(-1, -1)` si el
        ancla no aparece.
    """
    if not ancla:
        return 0, len(texto)
    inicio = texto.find(ancla)
    if inicio < 0:
        return -1, -1
    sangria = texto[:inicio].rsplit("\n", 1)[-1]
    prefijo = f"\n{sangria}"
    fin = texto.find(f"{prefijo}def ", inicio + len(ancla))
    if fin < 0:
        fin = texto.find(f"{prefijo}class ", inicio + len(ancla))
    if fin < 0:
        fin = len(texto)
    return inicio, fin


def _aplicar(raiz: Path, mutacion: Mutacion) -> Aplicacion:
    """Aplica la mutacion dentro del metodo que declara, y verifica que ocurrio.

    Cero ocurrencias y mas de una son ERROR. Una no es un detalle: con
    `replace(..., 1)` un patron repetido cae en la primera aparicion, que puede
    ser otro metodo, y el arnes reportaria como muerta una mutacion que nunca
    toco lo que su nombre declara.

    Args:
        raiz: Copia desechable.
        mutacion: La alteracion a aplicar.

    Returns:
        Si se aplico, cuantas occurencias tenia y por que fallo si no.
    """
    objetivo = raiz / mutacion.archivo
    texto = objetivo.read_text(encoding="utf-8")
    inicio, fin = _cuerpo_de(texto, mutacion.ancla)
    region = texto[inicio:fin] if inicio >= 0 else ""
    ocurrencias = region.count(mutacion.viejo)
    problema = _problema_de_aplicacion(mutacion, region, ocurrencias, inicio)
    if problema is not None:
        return problema
    cambiada = region.replace(mutacion.viejo, mutacion.nuevo, 1)
    _ = objetivo.write_text(texto[:inicio] + cambiada + texto[fin:], encoding="utf-8")
    escrito = objetivo.read_text(encoding="utf-8")
    if escrito.count(mutacion.viejo) >= texto.count(mutacion.viejo):
        detalle = "el texto viejo sigue presente: la sustitucion no quito nada"
        return Aplicacion(aplicada=False, ocurrencias=1, detalle=detalle)
    return Aplicacion(aplicada=True, ocurrencias=1, detalle="aplicada y verificada")


def _problema_de_aplicacion(
    mutacion: Mutacion,
    region: str,
    ocurrencias: int,
    inicio: int,
) -> Aplicacion | None:
    """Comprueba que la mutacion se puede aplicar sin ambiguedad.

    Args:
        mutacion: La alteracion declarada.
        region: La region del ancla dentro del archivo.
        ocurrencias: Veces que el fragmento aparece en la region.
        inicio: Indice de inicio de la region, `-1` si el ancla no existe.

    Returns:
        La aplicacion fallida con su motivo, o `None` si se puede aplicar.
    """
    if inicio < 0:
        return Aplicacion(
            aplicada=False,
            ocurrencias=0,
            detalle=f"el ancla {mutacion.ancla!r} no existe en el archivo",
        )
    if ocurrencias == 0:
        return Aplicacion(
            aplicada=False, ocurrencias=0, detalle="el fragmento no aparece dentro del ancla"
        )
    if ocurrencias > 1:
        detalle = (
            f"el fragmento aparece {ocurrencias} veces dentro del ancla; "
            "replace(..., 1) caeria en la primera"
        )
        return Aplicacion(aplicada=False, ocurrencias=ocurrencias, detalle=detalle)
    if region.replace(mutacion.viejo, mutacion.nuevo, 1) == region:
        detalle = "el texto nuevo es identico al viejo"
        return Aplicacion(aplicada=False, ocurrencias=1, detalle=detalle)
    return None


def _correr_tests(raiz: Path) -> tuple[Veredicto, int, str]:
    """Ejecuta la suite y clasifica el resultado sin inventar nada.

    Solo hay veredicto de la suite si pytest imprime su linea de resumen. Un
    error de recoleccion, de import o de sintaxis sale con codigo distinto de 0
    y distinto de 1, y en ese caso la suite NO llego a los tests: eso es ERROR.

    Args:
        raiz: Copia desechable donde corre la suite.

    Returns:
        Veredicto, codigo de salida y la linea de resumen de pytest.
    """
    proceso = subprocess.run(  # ruff: ignore[subprocess-without-shell-equals-true]
        ["uv", "run", "pytest", *TESTS, "-q", "--no-cov"],  # ruff: ignore[start-process-with-partial-path]
        cwd=raiz,
        capture_output=True,
        text=True,
        check=False,
    )
    lineas = proceso.stdout.strip().splitlines()
    resumen = next(
        (linea for linea in reversed(lineas) if "passed" in linea or "failed" in linea),
        "",
    )
    if not resumen:
        return Veredicto.ERROR, proceso.returncode, "pytest no imprimio linea de resumen"
    if proceso.returncode == 0:
        return Veredicto.SUPERVIVIENTE, proceso.returncode, resumen
    if proceso.returncode == 1:
        return Veredicto.MUERTA, proceso.returncode, resumen
    return Veredicto.ERROR, proceso.returncode, f"pytest no llego a los tests: {resumen}"


def _control(raiz: Path) -> tuple[Veredicto, int, str]:
    """Rompe una asercion a proposito y comprueba que el arnes la ve.

    Args:
        raiz: Copia desechable ya creada.

    Returns:
        Veredicto, codigo de salida y resumen de la suite deliberadamente rota.
    """
    objetivo = raiz / "tests/test_checkpoint_timings.py"
    texto = objetivo.read_text(encoding="utf-8")
    roto = texto.replace("== 500, en_disco", "== 999999, en_disco")
    if roto == texto:
        return Veredicto.CONTROL_ROTO, -1, "no se encontro la asercion del control"
    _ = objetivo.write_text(roto, encoding="utf-8")
    return _correr_tests(raiz)


def main() -> int:
    """Ejecuta el control y todas las mutaciones, y reporta cada veredicto.

    Returns:
        0 solo si el control se detecta, no hay supervivientes y no hay errores.
    """
    raiz = _preparar()
    try:
        veredicto_control, codigo_control, resumen_control = _control(raiz)
        print(f"copia desechable: {raiz}")
        print()
        print(f"CONTROL: {veredicto_control}  rc={codigo_control}  {resumen_control}")
        if veredicto_control is not Veredicto.MUERTA:
            print("  el control debio morir y no murio: BARRIDO ANULADO, sin resultados")
            return 1
        print("  el control murio como debia: el arnes ve los fallos")
        print()

        declaradas = (
            *_mutaciones_acumulacion(),
            *_mutaciones_reloj(),
            *_mutaciones_registro(),
            *_mutaciones_cableado(),
        )
        vistas: set[str] = set()
        conteo = dict.fromkeys(Veredicto, 0)
        for mutacion in declaradas:
            huella = mutacion.identidad
            if huella in vistas:
                conteo[Veredicto.DUPLICADA] += 1
                print(f"{mutacion.nombre:52} {Veredicto.DUPLICADA}")
                print("  mismo efecto que otra mutacion ya ejecutada; no cuenta como evidencia")
                print()
                continue
            vistas.add(huella)

            _ = shutil.rmtree(raiz, ignore_errors=True)
            raiz = _preparar()
            aplicacion = _aplicar(raiz, mutacion)
            if not aplicacion.aplicada:
                conteo[Veredicto.ERROR] += 1
                print(f"{mutacion.nombre:52} {Veredicto.ERROR}")
                print(f"  no aplicada: {aplicacion.detalle}")
                print(f"  ocurrencias={aplicacion.ocurrencias}  aplicada=no")
                print()
                continue
            veredicto, codigo, resumen = _correr_tests(raiz)
            conteo[veredicto] += 1
            print(f"{mutacion.nombre:52} {veredicto}")
            print(f"  ocurrencias={aplicacion.ocurrencias}  aplicada=si  rc={codigo}")
            print(f"  {resumen}")
            print()

        fallidas = (
            conteo[Veredicto.SUPERVIVIENTE] + conteo[Veredicto.ERROR] + conteo[Veredicto.DUPLICADA]
        )
        resumen_conteo = " ".join(
            (
                f"muertas={conteo[Veredicto.MUERTA]}",
                f"supervivientes={conteo[Veredicto.SUPERVIVIENTE]}",
                f"errores={conteo[Veredicto.ERROR]}",
                f"duplicadas={conteo[Veredicto.DUPLICADA]}",
                f"de {len(declaradas)}",
            )
        )
        print(resumen_conteo)
        return 1 if fallidas else 0
    finally:
        _ = shutil.rmtree(raiz, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
