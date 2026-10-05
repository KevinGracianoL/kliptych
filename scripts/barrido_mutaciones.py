"""Barredor de mutaciones dirigido a los comportamientos que esta PR cambia.

No usa mutmut (no corre en Windows). Aplica cada mutacion como sustitucion de
texto sobre el arbol de trabajo, corre los tests FOCALIZADOS de esa mutacion,
restaura el archivo y registra el veredicto. Un fallo de ejecucion se cuenta como
NO VERIFICADO, nunca como "superviviente": distinguir las tres categorias es lo
que hace que la tabla signifique algo.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]


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
        tests=("tests/test_gate_declared_aspect.py", "tests/test_zero_contract.py"),
    ),
    Mutacion(
        nombre="la predicada compartida se convierte en una comparacion exacta",
        archivo="src/kliptych/encoding.py",
        viejo="    return abs(aspect - target_aspect) <= target_aspect * tolerance",
        nuevo="    return aspect == target_aspect",
        tests=("tests/test_gate_declared_aspect.py", "tests/test_reframe_integration.py"),
    ),
)


def correr(comando: list[str]) -> int | None:
    """Corre el comando y clasifica su codigo de salida.

    Args:
        comando: argv fijo, sin shell.

    Returns:
        0 si pasaron, 1 si el suite fallo (mutante muerto) y `None` si el fallo
        no fue de aserciones (mutacion no verificada, no superviviente).
    """
    proceso = subprocess.run(  # ruff: ignore[subprocess-without-shell-equals-true]
        comando,
        cwd=RAIZ,
        capture_output=True,
        text=True,
        check=False,
    )
    if proceso.returncode not in {0, 1}:
        return None
    return proceso.returncode


def main() -> int:
    if len(sys.argv) > 1:
        seleccion = set(sys.argv[1:])
        mutaciones = tuple(m for m in MUTACIONES if m.nombre in seleccion)
    else:
        mutaciones = MUTACIONES

    conteo = {"muerta": 0, "superviviente": 0, "no_verificada": 0}
    for mutacion in mutaciones:
        destino = RAIZ / mutacion.archivo
        original = destino.read_text(encoding="utf-8")
        if mutacion.viejo not in original:
            print(f"[NO VERIFICADA] {mutacion.nombre}: el texto viejo no esta")
            conteo["no_verificada"] += 1
            continue
        copia = Path(tempfile.gettempdir()) / f"mutmut_backup_{destino.name}"
        shutil.copy2(destino, copia)
        try:
            destino.write_text(
                original.replace(mutacion.viejo, mutacion.nuevo, 1),
                encoding="utf-8",
            )
            codigo = correr(
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
            shutil.copy2(copia, destino)
            copia.unlink(missing_ok=True)

        if codigo is None:
            veredicto = "NO VERIFICADA"
            conteo["no_verificada"] += 1
        elif codigo == 1:
            veredicto = "MUERTA"
            conteo["muerta"] += 1
        else:
            veredicto = "SUPERVIVIENTE"
            conteo["superviviente"] += 1
        print(f"[{veredicto}] {mutacion.nombre}")

    print()
    print(
        f"muertas={conteo['muerta']} supervivientes={conteo['superviviente']} "
        f"no_verificadas={conteo['no_verificada']} de {len(mutaciones)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
