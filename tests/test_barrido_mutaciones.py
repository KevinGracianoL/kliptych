"""El barredor de mutaciones no puede reportar AUSENCIA como exito.

Esta es la clase de bug mas cara de la sesion: un numero que nadie midio escrito
como un cero. Ocurrio tres veces (39 y 34 en un commit, dos conteos mas en el
cuerpo del PR) y la version anterior de `barrido_mutaciones.py` reproducia el
mismo patron por dentro: un nombre de mutacion mal escrito imprimia `de 0` con
codigo de salida 0, y un `uv` roto se reportaba como NO VERIFICADA en vez de como
error del arnes.

Estos tests fijan los dos casos. Si alguno puede volver a pasar en silencio, toda
tabla que produzca la herramienta es inmerecible.
"""

import subprocess
import sys
from pathlib import Path
from typing import cast

import pytest
from scripts import barrido_mutaciones as modulo
from scripts.barrido_mutaciones import (
    MUTACIONES,
    Mutacion,
    Resultado,
    correr,
    seleccionar,
)

_RAIZ = Path(__file__).resolve().parents[1]
_SCRIPT = "scripts/barrido_mutaciones.py"


def _correr_script(*argumentos: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, _SCRIPT, *argumentos],
        cwd=_RAIZ,
        capture_output=True,
        text=True,
        check=False,
    )


def test_un_nombre_inexistente_es_error_y_no_un_cero() -> None:
    """Un nombre mal escrito aborta con codigo distinto de cero y nombra la causa.

    RED: antes `seleccionar` filtraba en silencio, el filtro daba `()`, el bucle
    no corria nada y `main` devolvia 0 imprimiendo `de 0`. Indistinguible de una
    corrida limpia, que es justo el defecto.
    """
    proceso = _correr_script("esta mutacion no existe")

    assert proceso.returncode != 0, (
        "un nombre de mutacion inexistente debe fallar, no devolver exito"
    )
    assert "ERROR" in proceso.stdout + proceso.stderr
    assert "esta mutacion no existe" in proceso.stdout + proceso.stderr
    assert "de 0" not in proceso.stdout


def test_un_nombre_valido_entre_otros_desconocidos_tambien_falla() -> None:
    """La seleccion es estricta: un nombrebueno no enmascara a uno malo."""
    valido = MUTACIONES[0].nombre

    proceso = _correr_script(valido, "nombre Inventado")

    assert proceso.returncode != 0
    assert "nombre Inventado" in proceso.stdout + proceso.stderr


def test_seleccionar_admite_todas_sin_argumentos() -> None:
    """Sin argumentos se seleccionan todas, y eso si es una seleccion valida."""
    assert seleccionar([]) == MUTACIONES


def test_una_falla_de_arnes_es_error_y_no_no_verificada() -> None:
    """Un comando que no llego a correr es ERROR, no una mutacion no verificada.

    `uv run nonexistentcmd` sale con codigo 2, el mismo que un error de
    recoleccion de pytest. Sin mirar la salida de pytest, ese 2 se reportaba
    como NO VERIFICADA: una mutacion que en realidad no se probo figurando en la
    tabla como "no verificada", que es el mismo defecto con otro nombre.
    """
    resultado = correr(["uv", "run", "comando-que-no-existe-ni-vendra"])

    assert resultado is Resultado.ERROR
    assert resultado is not Resultado.NO_VERIFICADA


def test_una_mutacion_no_aplicada_cuenta_como_no_verificada_y_sale_no_cero(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Una fila NO VERIFICADA hace que la corrida entera salga con codigo != 0.

    Una mutacion cuyo texto viejo no esta en el fuente NO se ejecuto. La corrida
    no es limpia y quien la lee desde CI tiene que poder notarlo sin leer el
    texto, asi que el codigo de salida lo dice.
    """

    # Un texto viejo que no esta en el fuente: la mutacion no llega a aplicarse.
    class _ViejaDesplazada:
        """Mutacion cuyo texto viejo no aparece en ningun fuente."""

        nombre: str = "prueba: texto inexistente"
        archivo: str = "src/kliptych/gate/checks.py"
        viejo: str = "esto no esta en el fuente"
        nuevo: str = "lo que sea"
        tests: tuple[str, ...] = ("tests/test_gate_declared_aspect.py",)

    vacia: tuple[Mutacion, ...] = cast("tuple[Mutacion, ...]", (_ViejaDesplazada(),))
    monkeypatch.setattr(modulo, "MUTACIONES", vacia)
    monkeypatch.setattr(sys, "argv", ["barrido_mutaciones.py"])
    codigo = modulo.main()

    assert codigo != 0
    assert "NO VERIFICADA" in capsys.readouterr().out


def test_el_arnes_deja_el_arbol_limpio_despues_de_correr() -> None:
    """La herramienta restaura cada archivo que muta, incluso si pytest falla."""
    antes = _estado_del_arbol()
    _ = _correr_script(MUTACIONES[0].nombre)
    despues = _estado_del_arbol()

    assert antes == despues, "el barredor dejo el arbol de trabajo modificado"


def _estado_del_arbol() -> str:
    proceso = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=_RAIZ,
        capture_output=True,
        text=True,
        check=False,
    )
    return proceso.stdout
