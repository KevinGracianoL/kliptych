"""El barredor de mutaciones no puede reportar AUSENCIA como exito, ni CONFLICTOS.

Esta es la clase de bug mas cara de la sesion: un numero que nadie midio escrito
como un cero, y un mutante que sobrevivio escrito como muerto. Ocurrio tres
veces (39 y 34 en un commit, dos conteos mas en un cuerpo de PR).

La version anterior de esta suite fijaba las RUTAS DE ERROR y nada mas. El
resultado fue que roto `correr` para que devolviera siempre FALLA, para que no
sumara el contador, y las seis pruebas pasaron igual: una mutacion que de verdad
supervivia se reportaba como `[MUERTA]` con `supervivientes=0`. Estos tests
exigen el otro sentido tambien.

Tres invariantes:
  - `clasificar` es la unica fuente de verdad de que FALLA sea muerte y PASA
    supervivencia.
  - Una tabla vacia es un ERROR, nunca "de 0" con exito.
  - La corrida sale con codigo != 0 si algo sobrevivio, quedo sin verificar, o
    fallo el arnes.
"""

import subprocess
import sys
from pathlib import Path

import pytest
from scripts import barrido_mutaciones as modulo
from scripts.barrido_mutaciones import (
    MUTACIONES,
    Mutacion,
    Resultado,
    clasificar,
    correr,
    seleccionar,
)

_RAIZ = Path(__file__).resolve().parents[1]
_SCRIPT = "scripts/barrido_mutaciones.py"
_SCRIPT_RAIZ = modulo.RAIZ


def _correr_script(*argumentos: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, _SCRIPT, *argumentos],
        cwd=_RAIZ,
        capture_output=True,
        text=True,
        check=False,
    )


# Una mutacion que NO cambia comportamiento: editar un comentario no rompe
# ninguna prueba, asi que debe reportarse SUPERVIVIENTE. Es la fila de control
# que demuestra que "MUERTA" significa "la suite lo detecto" y no "la suite
# corrio".
_COMENTARIO: Mutacion = Mutacion(
    nombre="control: cambio de comentario que no debe morir",
    archivo="src/kliptych/gate/engine.py",
    viejo="    checks = [check for _, check in outcomes]",
    nuevo="    checks = [check for _, check in outcomes]  # control de supervivencia",
    tests=("tests/test_gate_declared_aspect.py",),
)

# Una mutacion que SI cambia comportamiento y que una prueba real detecta.
_CERTERO: Mutacion = Mutacion(
    nombre="control: cambio de comportamiento que debe morir",
    archivo="src/kliptych/encoding.py",
    viejo="    return abs(aspect - target_aspect) <= target_aspect * tolerance",
    nuevo="    return aspect == target_aspect",
    tests=("tests/test_gate_declared_aspect.py", "tests/test_reframe_integration.py"),
)


def _tabla(*mutaciones: Mutacion) -> tuple[Mutacion, ...]:
    return mutaciones


# ---------------------------------------------------------------- clasificacion


@pytest.mark.parametrize(
    ("resultado", "etiqueta", "contador"),
    [
        (Resultado.FALLA, "MUERTA", "muerta"),
        (Resultado.PASA, "SUPERVIVIENTE", "superviviente"),
        (Resultado.NO_VERIFICADA, "NO VERIFICADA", "no_verificada"),
        (Resultado.ERROR, "ERROR", "error"),
    ],
)
def test_clasificar_mapea_cada_veredicto_a_su_contador(
    resultado: Resultado, etiqueta: str, contador: str
) -> None:
    """Cada veredicto tiene exactamente una etiqueta y un contador.

    Este es el test que faltaba. Con la clasificacion embebida en el bucle de
    `main` no habia forma de ejercitarla sin mutar el arbol entero, y por eso
    `correr` podia devolver siempre FALLA sin que nada se-enterase.
    """
    assert clasificar(resultado) == (etiqueta, contador)


def test_un_superviviente_no_puede_contarse_como_muerta() -> None:
    """`PASA` es supervivencia. Nunca comparte etiqueta ni contador con `FALLA`."""
    muerta = clasificar(Resultado.FALLA)
    viviente = clasificar(Resultado.PASA)

    assert muerta[0] != viviente[0]
    assert muerta[1] != viviente[1]
    assert viviente == ("SUPERVIVIENTE", "superviviente")


def test_error_y_no_verificada_son_cosas_distintas() -> None:
    """`NOT VERIFIED` significa una sola cosa: se aplico y no se pudo interpretar.

    Nunca "la mutacion no existe", y nunca un fallo del arnes. Si estas dos
    colapsaran, un `uv` roto se reportaria como una fila de la tabla y el conteo
    de mutaciones reales bajaria sin que nadie lo notase.
    """
    fallo_del_arnes = clasificar(Resultado.ERROR)
    no_verificada = clasificar(Resultado.NO_VERIFICADA)

    assert fallo_del_arnes != no_verificada
    assert fallo_del_arnes[0] == "ERROR"
    assert no_verificada[0] == "NO VERIFICADA"


# ------------------------------------------- las cuatro filas, sobre el arnes real


def test_una_mutacion_que_sobrevive_se_reporta_y_sale_no_cero(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Una fila SUPERVIVIENTE es un hallazgo sobre el codigo: sale != 0.

    RED: con `correr` roto para devolver siempre FALLA, esta fila de control se
    reportaba `[MUERTA]` con `supervivientes=0` y la corrida salia con 0. Los
    seis tests de la version anterior de este archivo pasaban igual.
    """
    monkeypatch.setattr(modulo, "MUTACIONES", _tabla(_COMENTARIO, _CERTERO))
    monkeypatch.setattr(sys, "argv", ["barrido_mutaciones.py"])

    codigo = modulo.main()
    salida = capsys.readouterr().out

    assert "[SUPERVIVIENTE] control: cambio de comentario" in salida, salida
    assert "[MUERTA] control: cambio de comportamiento" in salida, salida
    assert "muertas=1 supervivientes=1" in salida, salida
    assert codigo != 0, "una corrida con un superviviente no es una corrida limpia"


def test_todas_muertas_es_el_unico_camino_a_codigo_cero(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Codigo 0 significa, exactamente, que todas las filas murieron."""
    monkeypatch.setattr(modulo, "MUTACIONES", _tabla(_CERTERO))
    monkeypatch.setattr(sys, "argv", ["barrido_mutaciones.py"])

    codigo = modulo.main()

    assert "muertas=1 supervivientes=0" in capsys.readouterr().out
    assert codigo == 0


def test_una_tabla_vacia_es_error_y_no_un_de_cero() -> None:
    """Sin mutaciones no hay corrida limpia: hay una corrida que no probo nada."""
    proceso = _correr_script_con_tabla_vacia()

    assert proceso.returncode != 0, "una tabla vacia debe fallar, no devolver exito"
    assert "ERROR" in proceso.stdout + proceso.stderr
    assert "vacia" in proceso.stdout + proceso.stderr


def test_un_nombre_inexistente_es_error_y_no_un_cero() -> None:
    """Un nombre mal escrito aborta con codigo != 0 y nombra la causa.

    RED: antes `seleccionar` filtraba en silencio, el filtro daba `()`, el bucle
    no corria nada y `main` devolvia 0 imprimiendo `de 0`. Indistinguible de una
    corrida limpia, que es justo el defecto.
    """
    proceso = _correr_script("esta mutacion no existe")

    assert proceso.returncode != 0
    assert "ERROR" in proceso.stdout + proceso.stderr
    assert "esta mutacion no existe" in proceso.stdout + proceso.stderr
    assert "de 0" not in proceso.stdout


def test_un_nombre_valido_entre_otros_desconocidos_tambien_falla() -> None:
    """La seleccion es estricta: un nombre bueno no enmascara a uno malo."""
    proceso = _correr_script(MUTACIONES[0].nombre, "nombre Inventado")

    assert proceso.returncode != 0
    assert "nombre Inventado" in proceso.stdout + proceso.stderr


def test_una_falla_de_arnes_es_error_y_no_no_verificada() -> None:
    """Un comando que no llego a correr es ERROR, no una fila de la tabla.

    `uv run nonexistentcmd` sale con codigo 2, el mismo que un error de
    recoleccion de pytest. Sin mirar la salida de pytest, ese 2 se reportaba
    como NO VERIFICADA: una mutacion que en realidad no se probo figurando como
    "medida", que es el mismo defecto con otro nombre.
    """
    resultado = correr(["uv", "run", "comando-que-no-existe-ni-vendra"])

    assert resultado is Resultado.ERROR
    assert resultado is not Resultado.NO_VERIFICADA


def test_seleccionar_admite_todas_sin_argumentos() -> None:
    """Sin argumentos se seleccionan todas, y eso si es una seleccion valida."""
    assert seleccionar([]) == MUTACIONES


def test_el_arnes_deja_el_arbol_limpio_despues_de_correr() -> None:
    """La herramienta restaura cada archivo que muta, incluso si pytest falla."""
    antes = _estado_del_arbol()
    _ = _correr_script(MUTACIONES[0].nombre)
    despues = _estado_del_arbol()

    assert antes == despues, "el barredor dejo el arbol de trabajo modificado"


def test_la_tabla_real_no_tiene_filas_vacias_ni_sin_tipo() -> None:
    """Ninguna fila puede tener un nombre vacio o un texto viejo vacio.

    Una fila con `viejo=""` haria `replace("", ...)` en todo el archivo y la
    herramienta reportaria algo que no midio. Es el mismo fallo que la tabla
    vacia, un renglon por debajo.
    """
    assert MUTACIONES, "la tabla de mutaciones esta vacia"
    for mutacion in MUTACIONES:
        assert mutacion.nombre.strip(), mutacion
        assert mutacion.viejo.strip(), mutacion
        assert mutacion.tests, mutacion
        assert (_SCRIPT_RAIZ / mutacion.archivo).is_file(), mutacion


def _correr_script_con_tabla_vacia() -> subprocess.CompletedProcess[str]:
    """Ejecuta el script con `MUTACIONES` reducido a la tabla vacia.

    Como proceso aparte y no en proceso: el codigo de salida de `SystemExit` es
    justo lo que hay que observar, y en proceso seria el del test.

    Returns:
        El proceso terminado, con su codigo de salida y su salida capturada.
    """
    codigo = (
        "import sys; "
        f"sys.path.insert(0, {str(_RAIZ)!r}); "
        "import scripts.barrido_mutaciones as m; "
        "m.MUTACIONES = (); "
        "sys.exit(m.main())"
    )
    return subprocess.run(
        [sys.executable, "-c", codigo],
        cwd=_RAIZ,
        capture_output=True,
        text=True,
        check=False,
    )


def _estado_del_arbol() -> str:
    """El estado del arbol de trabajo, para comparar antes y despues de correr.

    Returns:
        La salida de `git status --porcelain`: vacia si el arbol esta limpio.
    """
    return subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=_RAIZ,
        capture_output=True,
        text=True,
        check=False,
    ).stdout
