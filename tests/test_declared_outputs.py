"""El contrato DECLARA que formatos emite, y lo declara de forma coherente.

PR 1 de la escalera de #76. Antes el aspecto vivia en tres casas sin unsur:
el renderer tenia su constante, el gate tenia la suya, y el contrato no decia
nada. Aqui aparece la cuarta casa, la declaracion, y se hace que las demas no
puedan contradecirla.

Lo que esta PR NO hace, y por eso estos tests no renderizan nada ni exportan:
- No hay forma de declarar dos salidas en la CLI.
- Ningun pipeline consume `Contract.outputs` todavia.
- El gate sigue validando con su propia constante de 9:16 (PR 2).
- El render sigue siendo 9:16 (PR 3).

Asi que un contrato puede DECLARAR un 16:9 y el sistema entero lo ignorara sin
quejarse. Eso es correcto para esta PR y es exactamente por lo que las
siguientes existen. Lo que no se permite es que las piezas se contradigan entre
si, que es lo que estos tests comprueban.
"""

import pytest
from pydantic import ValidationError

from kliptych.contract.enums import Aspect
from kliptych.contract.schema import Contract, OutputSpec, SplitScreenConfig
from tests.support import make_contract

_VERTICAL: dict[str, object] = {"aspect": "9:16", "index": 0}
_HORIZONTAL: dict[str, object] = {"aspect": "16:9", "index": 1}
_CANVAS_16x9 = {"width": 1920, "height": 1080}


def _split(**overrides: object) -> dict[str, object]:
    """Un payload de split_screen valido, con lo que se cambie encima.

    Args:
        **overrides: Campos a sustituir en el payload por defecto.

    Returns:
        El payload, con `top_source`/`bottom_source` y el resto por defecto.
    """
    payload: dict[str, object] = {
        "top_source": "clip-top",
        "bottom_source": "clip-bottom",
        "width": 1080,
        "height": 1920,
    }
    payload.update(overrides)
    return payload


def test_a1_el_enum_expone_solo_los_formatos_soportados() -> None:
    """El conjunto de aspectos es cerrado y explicito."""
    assert [aspect.value for aspect in Aspect] == ["9:16", "16:9"]
    assert Aspect("9:16") is Aspect.VERTICAL
    assert Aspect("16:9") is Aspect.HORIZONTAL


def test_a2_cada_aspecto_expone_su_razon() -> None:
    """La razon vive en el enum, no la recalcula cada consumidor."""
    assert Aspect.VERTICAL.ratio == pytest.approx(9 / 16)
    assert Aspect.HORIZONTAL.ratio == pytest.approx(16 / 9)


@pytest.mark.parametrize("invalido", ["4:5", "9:16 ", "1:1", ""])
def test_a3_un_aspecto_fuera_del_conjunto_no_construye(invalido: str) -> None:
    """Un string raro es un error de construccion, no un render raro.

    El conjunto es cerrado a proposito: 4:5 y 1:1 son relaciones que ningun
    codificador respeta aqui, asi que admitirlas seria aceptar algo que el
    render no sabe cumplir.
    """
    with pytest.raises(ValueError, match="not a valid Aspect"):
        _ = Aspect(invalido)


def test_a4_la_razon_no_sirve_para_construir_un_aspecto() -> None:
    """El enum se busca POR VALOR, y la razon no es un valor del enum.

    `Aspect.VERTICAL.ratio` es 0.5625, que es informacion derivada, no la
    declaracion. Si `Aspect(0.5625)` devolviera algo, la razon habria pasado a
    ser una segunda forma de declarar el aspecto, con su propia fuente de
    verdad. Es justo lo que el enum cerrado quiere impedir.
    """
    with pytest.raises(ValueError, match="not a valid Aspect"):
        _ = Aspect(9 / 16)


def test_b1_el_default_es_una_salida_vertical() -> None:
    """Sin declarar nada, el contrato significa lo que siempre significar.

    El default no es "lo mismo que antes" por casualidad: es el mismo valor,
    literalmente. Si alguien cambia esta tupla, este test lo dice.
    """
    contract = make_contract()

    assert contract.outputs == (OutputSpec(aspect=Aspect.VERTICAL, index=0),)
    assert len(contract.outputs) == 1
    assert contract.outputs[0].aspect is Aspect.VERTICAL


def test_b2_se_puede_declarar_un_formato_horizontal() -> None:
    """La declaracion admite 16:9 sin tocar nada mas."""
    contract = make_contract(schema_version="1.2", outputs=[_HORIZONTAL])

    assert contract.outputs[0].aspect is Aspect.HORIZONTAL
    assert contract.outputs[0].index == 1


def test_b3_no_se_puede_declarar_ninguna_salida() -> None:
    """Una campana que no emite nada no es una campana."""
    with pytest.raises(ValidationError, match="at least 1 item"):
        _ = make_contract(schema_version="1.2", outputs=[])


def test_b4_no_se_pueden_repetir_aspectos() -> None:
    """Dos salidas del mismo formato colisionarian en el nombre de entrega.

    El error tiene que salir al construir y decir CUAL se repite. Si sale mas
    tarde, en el exportador, el mensaje habla de nombres de archivo y no de
    salidas duplicadas, que es la causa real.
    """
    repetido: dict[str, object] = {"aspect": "9:16", "index": 1}

    with pytest.raises(ValidationError) as excinfo:
        _ = make_contract(schema_version="1.2", outputs=[_VERTICAL, repetido])

    assert "repetidos" in str(excinfo.value)
    assert "9:16" in str(excinfo.value)


def test_b5_un_documento_1_1_no_puede_declarar_salidas() -> None:
    """`extra="forbid"` no cubre esto, asi que hay que cubrirlo a mano.

    `outputs` SI existe en el modelo, asi que un documento que se declara 1.1 y
    lo trae pasa la validacion de campos sin quejarse. Afirmaria dos cosas
    incompatibles: que habla el 1.1, donde el campo no existia, y que declara
    una salida que el 1.1 no puede expresar.
    """
    with pytest.raises(ValidationError) as excinfo:
        _ = make_contract(schema_version="1.1", outputs=[_HORIZONTAL])

    mensaje = str(excinfo.value)
    assert "1.1" in mensaje
    assert "outputs" in mensaje
    assert "1.2" in mensaje


def test_b6_un_documento_1_1_sin_salidas_carga_con_el_default() -> None:
    """La backwards no es un compromiso: es el camino de los 589 documentos."""
    contract = make_contract(schema_version="1.1")

    assert contract.schema_version == "1.1"
    assert contract.outputs == (OutputSpec(aspect=Aspect.VERTICAL, index=0),)


def test_b7_esta_version_emite_1_2() -> None:
    """El default es 1.2 aunque se lean 1.1.

    Esto es lo que protege al LECTOR VIEJO: un binario con 1.1 que recibe un
    documento nuevo lo rechaza por `schema_version`, con un mensaje que dice la
    causa, en vez de ignorarle `outputs` y reventar mas tarde con "campo extra
    no permitido". Si este test falla, el bump se ha deshEcho.
    """
    assert Contract.model_json_schema()["properties"]["schema_version"]["default"] == "1.2"
    assert make_contract(schema_version="1.2").schema_version == "1.2"


def test_c1_el_lienzo_se_valida_contra_su_aspecto_declarado() -> None:
    """SplitScreenConfig se auto-valida: 16:9 con HORIZONTAL pasa, con VERTICAL no.

    El lienzo declara su PROPIA geometria porque describe un solo artefacto: un
    lienzo no puede ser 9:16 y 16:9 a la vez.
    """
    horizontal = SplitScreenConfig.model_validate(_split(aspect="16:9", **_CANVAS_16x9))
    assert horizontal.aspect is Aspect.HORIZONTAL
    assert (horizontal.width, horizontal.height) == (1920, 1080)

    with pytest.raises(ValidationError) as excinfo:
        _ = SplitScreenConfig.model_validate(_split(aspect="9:16", **_CANVAS_16x9))

    assert "9:16" in str(excinfo.value)


def test_c2_el_lienzo_vertical_por_defecto_seguida_siendo_valido() -> None:
    """El caso de antes tiene que seguir funcionando sin declarar nada."""
    config = SplitScreenConfig.model_validate(_split())

    assert config.aspect is Aspect.VERTICAL
    assert (config.width, config.height) == (1080, 1920)


def test_c3_un_lienzo_vertical_no_puede_declararse_horizontal() -> None:
    """El caso inverso tambien falla, y por la misma regla.

    Antes esto lo cubria una constante de modulo; ahora lo cubre el enum. Si
    alguien reintroduce la comparacion contra 9 y 16, este test y `c1` mueren.
    """
    with pytest.raises(ValidationError, match="16:9"):
        _ = SplitScreenConfig.model_validate(_split(aspect="16:9"))


def test_d1_lienzo_y_salida_declarada_pueden_coincidir() -> None:
    """El caso coherente: una salida, un lienzo de ese formato."""
    contract = make_contract(
        schema_version="1.2",
        outputs=[_VERTICAL],
        split_screen=_split(aspect="9:16"),
    )

    assert contract.split_screen is not None
    assert contract.split_screen.aspect is contract.outputs[0].aspect


def test_d2_un_lienzo_horizontal_con_salida_vertical_se_rechaza() -> None:
    """Las dos declaraciones se reconcilian, y aqui no coinciden.

    Sin esta comprobacion el contrato declararia 9:16 y traeria un lienzo
    16:9, y nada lo notaria hasta el render, que produciria un artefacto con un
    formato que la campana no declaro. El mensaje tiene que nombrar LAS DOS.
    """
    with pytest.raises(ValidationError) as excinfo:
        _ = make_contract(
            schema_version="1.2",
            outputs=[_VERTICAL],
            split_screen=_split(aspect="16:9", **_CANVAS_16x9),
        )

    mensaje = str(excinfo.value)
    assert "16:9" in mensaje
    assert "9:16" in mensaje


def test_d3_multi_formato_con_lienzo_se_rechaza() -> None:
    """Multi-formato + split-screen es una limitacion, no un caso raro.

    El lienzo pertenece a UN formato y el modelo no dice a cual de los dos, asi
    que se rechaza en lugar de adivinar. Si alguna vez hace falta, la
    ampliacion es un campo que diga a que salida pertenece, mas esta misma
    reconciliacion.
    """
    with pytest.raises(ValidationError) as excinfo:
        _ = make_contract(
            schema_version="1.2",
            outputs=[_VERTICAL, _HORIZONTAL],
            split_screen=_split(aspect="9:16"),
        )

    mensaje = str(excinfo.value)
    assert "multi-formato" in mensaje
    assert "9:16" in mensaje
    assert "16:9" in mensaje
    assert "limitacion deliberada" in mensaje


def test_d4_multi_formato_sin_lienzo_si_carga() -> None:
    """El rechazo es del lienzo, no de multi-formato.

    Si esto fallara, el limite se habria colado en la declaracion de salidas y
    habria impedido el caso que PR 3 va a querer.
    """
    contract = make_contract(schema_version="1.2", outputs=[_VERTICAL, _HORIZONTAL])

    assert [spec.aspect for spec in contract.outputs] == [
        Aspect.VERTICAL,
        Aspect.HORIZONTAL,
    ]
    assert contract.split_screen is None
