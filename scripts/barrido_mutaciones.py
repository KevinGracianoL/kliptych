"""Barredor de mutaciones dirigido a los comportamientos que esta PR cambia.

No usa mutmut (no corre en Windows). Aplica cada mutacion como sustitucion de
texto sobre el arbol de trabajo, corre los tests FOCALIZADOS de esa mutacion,
restaura el archivo y registra el veredicto. Un fallo de ejecucion se cuenta como
NO VERIFICADO, nunca como "superviviente": distinguir las tres categorias es lo
que hace que la tabla signifique algo.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]

# Linea de resumen que pytest emite al terminar. Su ausencia significa que pytest
# no llego a ejecutarse: sin este chequeo, un `uv` roto o un comando inexistente
# salen con el mismo codigo que un fallo de recoleccion y se reportan igual.
_RESUMEN_PYTEST = re.compile(r"\d+ (?:passed|failed|error|errors|no tests ran)\b")


class Resultado(Enum):
    """Veredicto de UNA corrida de pytest, con sus causas separadas.

    Los tres ultimos son los que se confundian antes. `ERROR` es un fallo del
    arnes, no del mutante: aborta la corrida con codigo distinto de cero para
    que un fallo de infraestructura no se lea como una tabla limpia.
    """

    PASA = "pasa"
    FALLA = "falla"
    NO_VERIFICADA = "no_verificada"
    ERROR = "error"


@dataclass(frozen=True)
class Mutacion:
    """Una sustitucion de texto y los tests que tienen que detectarla.

    Attributes:
        nombre: Que cambio del comportamiento simula.
        archivo: Ruta relativa a la raiz del repositorio.
        viejo: Texto exacto a reemplazar. No se busca "parecido": si el fuente
            se mueve, la mutacion se marca NO VERIFICADA en vez de fingir.
        nuevo: Texto que lo reemplaza.
        tests: Suites focalizadas que deben caer.
    """

    nombre: str
    archivo: str
    viejo: str
    nuevo: str
    tests: tuple[str, ...]


MUTACIONES: tuple[Mutacion, ...] = (
    Mutacion(
        nombre="gate ignora la config de la corrida y usa un default",
        archivo="src/kliptych/gate/engine.py",
        viejo="render=self._render,",
        nuevo="render=RenderConfig(),",
        tests=("tests/test_gate_declared_aspect.py", "tests/test_zero_contract.py"),
    ),
    Mutacion(
        nombre="la tolerancia se lee del default y no de context.render",
        archivo="src/kliptych/gate/checks.py",
        viejo="return context.render.aspect_tolerance",
        nuevo="return 0.05",
        tests=("tests/test_gate_declared_aspect.py",),
    ),
    Mutacion(
        nombre="split-screen compara contra 9:16 de modulo en vez del declarado",
        archivo="src/kliptych/gate/checks.py",
        viejo="matches_declared_aspect(width, height, split.aspect, tolerancia=tolerancia)",
        nuevo="matches_declared_aspect(width, height, Aspect.VERTICAL, tolerancia=tolerancia)",
        tests=("tests/test_gate_declared_aspect.py",),
    ),
    Mutacion(
        nombre="las salidas declaradas se exigen TODAS en vez de ALGUNA",
        archivo="src/kliptych/gate/checks.py",
        viejo=(
            "    return any(\n"
            "        matches_declared_aspect(width, height, aspecto, tolerancia=tolerancia)\n"
            "        for aspecto in declarados\n"
            "    )"
        ),
        nuevo=(
            "    return all(\n"
            "        matches_declared_aspect(width, height, aspecto, tolerancia=tolerancia)\n"
            "        for aspecto in declarados\n"
            "    )"
        ),
        tests=("tests/test_gate_declared_aspect.py",),
    ),
    Mutacion(
        nombre="la evidencia reduce el aspecto medido a un string reducido",
        archivo="src/kliptych/gate/checks.py",
        viejo='    return f"{width}:{height}"',
        nuevo=(
            "    reducido = Fraction(width, height).limit_denominator(9)\n"
            '    return f"{reducido.numerator}:{reducido.denominator}"'
        ),
        tests=("tests/test_gate_declared_aspect.py",),
    ),
    Mutacion(
        nombre="artifact.aspect deja de forzarse a hard en el resolver",
        archivo="src/kliptych/resolver.py",
        viejo='    rules = ["artifact.integrity", "artifact.aspect"]',
        nuevo='    rules = ["artifact.integrity"]',
        tests=(
            "tests/test_resolver.py",
            "tests/test_gate_declared_aspect.py",
            "tests/test_zero_contract.py",
        ),
    ),
    Mutacion(
        nombre="artifact.aspect desaparece de las reglas conocidas del schema",
        archivo="src/kliptych/contract/schema.py",
        viejo='    "artifact.aspect",\n',
        nuevo="",
        # La puerta real de esta invariante es
        # `tests/test_gate_checks.py::test_known_validator_rules_matches_default_validators`,
        # que existia antes que este archivo. Con solo las suites de la tabla el
        # mutante sobrevivia, no porque la invariante no estuviera cubierta sino
        # porque el ambito de la fila no incluia a quien la cubre.
        tests=(
            "tests/test_gate_checks.py",
            "tests/test_gate_declared_aspect.py",
            "tests/test_zero_contract.py",
        ),
    ),
    Mutacion(
        nombre="la predicada compartida se convierte en una comparacion exacta",
        archivo="src/kliptych/encoding.py",
        viejo="    return abs(aspect - target_aspect) <= target_aspect * tolerance",
        nuevo="    return aspect == target_aspect",
        tests=("tests/test_gate_declared_aspect.py", "tests/test_reframe_integration.py"),
    ),
)


def correr(comando: list[str]) -> Resultado:
    """Corre el comando y clasifica su codigo de salida.

    La exit code de pytest no basta para separar "el mutante murio" de "el
    comando ni llego a correr". `uv run <comando inexistente>` y un error de
    recoleccion salen ambos con 2, y con solo el codigo el primero se reportaba
    como una mutacion no verificada, que es exactamente el modo de fallo que
    esta herramienta existe para evitar: escribir una ausencia como un cero.

    Por eso se mira la salida de pytest. Si no aparece su linea de resumen, no
    fue pytest quien fallo y el resultado es ERROR, no NO VERIFICADO.

    Args:
        comando: argv fijo, sin shell.

    Returns:
        El veredicto de la corrida segun `Resultado`.
    """
    proceso = subprocess.run(  # ruff: ignore[subprocess-without-shell-equals-true]
        comando,
        cwd=RAIZ,
        capture_output=True,
        text=True,
        check=False,
    )
    salida = f"{proceso.stdout}\n{proceso.stderr}"
    if not _RESUMEN_PYTEST.search(salida):
        return Resultado.ERROR
    if proceso.returncode == 0:
        return Resultado.PASA
    if proceso.returncode == 1:
        return Resultado.FALLA
    return Resultado.NO_VERIFICADA


def seleccionar(argumentos: list[str]) -> tuple[Mutacion, ...]:
    """Resuelve los nombres pedidos contra la lista, y falla si no existen.

    Un nombre mal escrito tiene que ser un ERROR. Antes se filtraba en silencio
    y una corrida que no verifico nada imprimia `de 0` con codigo de salida 0,
    indistinguible de una corrida limpia. Ese es el bug que hizo que 39 y 34
    llegaran a un commit: un numero que nadie midio escrito como un cero.

    Args:
        argumentos: Los `sys.argv[1:]`.

    Returns:
        Las mutaciones pedidas, o todas si no hay argumentos.

    Raises:
        SystemExit: Si algun nombre pedido no esta en `MUTACIONES`.
    """
    if not argumentos:
        return MUTACIONES
    conocidos = {m.nombre for m in MUTACIONES}
    desconocidos = sorted(set(argumentos) - conocidos)
    if desconocidos:
        raise SystemExit(
            "ERROR: no existe(n) la(s) mutacion(es) "
            + ", ".join(repr(nombre) for nombre in desconocidos)
            + f". Conocidas: {len(conocidos)}."
        )
    pedido = set(argumentos)
    return tuple(m for m in MUTACIONES if m.nombre in pedido)


def main() -> int:
    mutaciones = seleccionar(sys.argv[1:])
    conteo = {"muerta": 0, "superviviente": 0, "no_verificada": 0, "error": 0}
    for mutacion in mutaciones:
        destino = RAIZ / mutacion.archivo
        original = destino.read_text(encoding="utf-8")
        if mutacion.viejo not in original:
            # El texto viejo no esta: la mutacion NO LLEGO a aplicarse, asi que
            # no se ejecuto nada. No es una mutacion superviviente ni una que
            # fallo, es una que no se corrio.
            print(f"[NO VERIFICADA] {mutacion.nombre}: el texto viejo no esta en el fuente")
            conteo["no_verificada"] += 1
            continue
        copia = Path(tempfile.gettempdir()) / f"mutmut_backup_{destino.name}"
        _ = shutil.copy2(destino, copia)
        try:
            _ = destino.write_text(
                original.replace(mutacion.viejo, mutacion.nuevo, 1),
                encoding="utf-8",
            )
            resultado = correr(
                [
                    "uv",
                    "run",
                    "pytest",
                    "--no-cov",
                    "-x",
                    "-q",
                    *mutacion.tests,
                ],
            )
        finally:
            _ = shutil.copy2(copia, destino)
            copia.unlink(missing_ok=True)

        if resultado is Resultado.FALLA:
            print(f"[MUERTA] {mutacion.nombre}")
            conteo["muerta"] += 1
        elif resultado is Resultado.PASA:
            print(f"[SUPERVIVIENTE] {mutacion.nombre}")
            conteo["superviviente"] += 1
        elif resultado is Resultado.NO_VERIFICADA:
            print(f"[NO VERIFICADA] {mutacion.nombre}: pytest salio con un codigo inesperado")
            conteo["no_verificada"] += 1
        else:
            print(f"[ERROR] {mutacion.nombre}: pytest no llego a ejecutarse (fallo del arnes)")
            conteo["error"] += 1

    print()
    resumen = (
        f"muertas={conteo['muerta']} supervivientes={conteo['superviviente']} "
        f"no_verificadas={conteo['no_verificada']} errores={conteo['error']} "
        f"de {len(mutaciones)}"
    )
    print(resumen)
    # Codigo de salida distinto de cero si algo quedo sin verificar. Una corrida
    # con una fila NO VERIFICADA o ERROR no es una corrida limpia, y quien la
    # lee desde CI tiene que poder notarlo sin leer el texto.
    return 1 if (conteo["no_verificada"] or conteo["error"]) else 0


if __name__ == "__main__":
    raise SystemExit(main())
