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

import inspect
from pathlib import Path

import pytest
from pydantic import ValidationError

from kliptych.contract.enums import Aspect
from kliptych.contract.schema import (
    Contract,
    OutputSpec,
    SplitScreenConfig,
    contract_digest,
    vanilla_contract,
)
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
    assert [aspect.value for aspect in Aspect] == ["9:16", "16:9", "1:1", "4:5"]
    assert Aspect("9:16") is Aspect.VERTICAL
    assert Aspect("16:9") is Aspect.HORIZONTAL
    assert Aspect("1:1") is Aspect.SQUARE
    assert Aspect("4:5") is Aspect.PORTRAIT_4_5


@pytest.mark.parametrize("aspect", list(Aspect))
def test_a2_todo_aspecto_expone_su_razon(aspect: Aspect) -> None:
    """La razon vive en el enum, y TODO miembro la tiene.

    Parametrizado sobre `Aspect`, no sobre dos literales: anadir `SQUARE="1:1"`
    sin su entrada en la tabla tiene que fallar AQUI, no con un `KeyError`
    dentro de un `model_validator` en produccion, que pydantic no envuelve.
    """
    assert aspect.ratio > 0
    ancho, alto = aspect.value.split(":")
    assert aspect.ratio == pytest.approx(int(ancho) / int(alto))


def test_a2b_las_razones_conocidas_son_las_declaradas() -> None:
    """El enum no crece en silencio, y la razon sale de enteros exactos."""
    assert Aspect.VERTICAL.numerator == 9
    assert Aspect.VERTICAL.denominator == 16
    assert Aspect.VERTICAL.ratio == pytest.approx(9 / 16)

    assert Aspect.HORIZONTAL.numerator == 16
    assert Aspect.HORIZONTAL.denominator == 9
    assert Aspect.HORIZONTAL.ratio == pytest.approx(16 / 9)

    assert Aspect.SQUARE.numerator == 1
    assert Aspect.SQUARE.denominator == 1
    assert Aspect.SQUARE.ratio == pytest.approx(1.0)

    assert Aspect.PORTRAIT_4_5.numerator == 4
    assert Aspect.PORTRAIT_4_5.denominator == 5
    assert Aspect.PORTRAIT_4_5.ratio == pytest.approx(0.8)


@pytest.mark.parametrize("invalido", ["2:3", "16:10", "9:16 ", ""])
def test_a3_un_aspecto_fuera_del_conjunto_no_construye(invalido: str) -> None:
    """Un string raro es un error de construccion, no un render raro.

    El conjunto es cerrado a proposito: 2:3 y 16:10 no estan declarados, asi que
    admitirlos seria aceptar algo que el render no sabe cumplir.
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


def test_b3b_el_indice_es_una_etiqueta_no_un_identificador() -> None:
    """Dos formatos con el mismo `index` cargan; no son una colision todavia.

    El `index` es una etiqueta ordenada, no una clave unica, asi que repetirlo
    entre formatos distintos es legal. Lo que no puede ser es repetir el ASPECTO,
    porque ahi si dos piezas identicas compartirian el nombre de entrega. La
    unicidad real del nombre la pondra PR 3, que hoy nombra las piezas
    `{campana}-{plataforma}-{NN}` sin el formato.
    """
    contrato = make_contract(
        schema_version="1.2",
        outputs=[{"aspect": "9:16", "index": 0}, {"aspect": "16:9", "index": 0}],
    )

    assert [spec.index for spec in contrato.outputs] == [0, 0]
    assert [spec.aspect for spec in contrato.outputs] == [
        Aspect.VERTICAL,
        Aspect.HORIZONTAL,
    ]


def test_b3c_el_indice_no_puede_ser_negativo() -> None:
    """Un indice negativo no es una posicion, es un nombre de archivo raro."""
    with pytest.raises(ValidationError, match="greater than or equal to 0"):
        _ = make_contract(schema_version="1.2", outputs=[{"aspect": "9:16", "index": -1}])


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


def test_b6b_un_1_1_puede_traer_el_default_explicito() -> None:
    """La veda es por valor, no por presencia, y esto es el motivo.

    `model_dump` emite siempre todos los campos, y un contrato 1.1 tiene que
    sobrevivir a su propio round-trip: `tests/support.py:with_platform` revalida
    el dump en vez de copiar el modelo, asi que el documento que llega al
    validador SI trae `outputs` aunque nadie lo declarara. Un chequeo por
    presencia rechazaria eso y, con el, media suite.
    """
    contrato = make_contract(schema_version="1.1")

    recargado = Contract.model_validate(contrato.model_dump(mode="json"))

    assert recargado.outputs == contrato.outputs
    assert recargado.schema_version == "1.1"


def test_b8_el_digest_no_mueve_por_declarar_el_default() -> None:
    """Un contrato ya grabado tiene que conservar su hash.

    `outputs` no existia en 1.1, asi que un contrato 1.1 que no lo declara no lo
    tiene en su dump. `contract_digest` lo poda cuando vale el default, que es
    el patron que ya usan `audio_policy` y los ajustes de watermark.

    La comparacion es 1.1 contra 1.1 a proposito: `schema_version` entra en el
    digest, asi que un 1.1 nunca puede igualar a un 1.2 y comparar eso solo
    probaria que el campo existe.
    """
    sin_declarar = contract_digest(make_contract(schema_version="1.1"))
    declarando_el_default = contract_digest(
        make_contract(schema_version="1.1", outputs=[_VERTICAL])
    )

    assert sin_declarar == declarando_el_default


def test_b8c_el_digest_de_un_1_1_es_el_de_siempre() -> None:
    """El valor golden, medido sobre `main` antes de este PR.

    Sin este ancla, la poda de `outputs` puede desaparecer y mover el hash de
    todo contrato 1.1 sin que ningun test se entere: nada mas mide el digest.
    Con ella, el hash que hay grabado en los manifiestos es un contrato, no una
    funcion del dia.
    """
    assert contract_digest(make_contract(schema_version="1.1")) == (
        "b307aa91c957f7ae38849c2e21f3e67bc4891ad948cc7c7e5e7814e94b09e006"
    )


def test_b8b_declarar_salidas_de_verdad_si_mueve_el_digest() -> None:
    """La poda no puede tragarse la declaracion: si no, no detectaria el cambio."""
    base = contract_digest(make_contract())
    dual = contract_digest(make_contract(schema_version="1.2", outputs=[_VERTICAL, _HORIZONTAL]))

    assert base != dual


def test_b9_el_digest_sobrevive_al_round_trip() -> None:
    """Rehidratar un contrato no puede cambiar su identidad."""
    contrato = make_contract(schema_version="1.2", outputs=[_VERTICAL])

    recargado = Contract.model_validate(contrato.model_dump(mode="json"))

    assert contract_digest(recargado) == contract_digest(contrato)


def test_b7_esta_version_emite_1_2() -> None:
    """El default es 1.2 aunque se lean 1.1.

    Esto es lo que protege al LECTOR VIEJO: un binario con 1.1 que recibe un
    documento nuevo lo rechaza por `schema_version`, con un mensaje que dice la
    causa, en vez de ignorarle `outputs` y reventar mas tarde con "campo extra
    no permitido". Si este test falla, el bump se ha deshEcho.
    """
    assert Contract.model_json_schema()["properties"]["schema_version"]["default"] == "1.2"
    assert make_contract(schema_version="1.2").schema_version == "1.2"


def test_b7b_el_contrato_de_referencia_emite_1_2() -> None:
    """El default tiene que llegar al codigo que construye contratos de verdad.

    `vanilla_contract` es el camino del modo zero-contract, donde no hay brief
    ni LLM: el contrato lo sintetiza el codigo. Comprobar el default del modelo
    JSON no basta, porque se puede declarar 1.2 en el modelo y seguir emitiendo
    1.1 desde ahi.
    """
    assert vanilla_contract(campaign_id="camp-test").schema_version == "1.2"


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


def test_c1b_el_lienzo_tiene_techo() -> None:
    """Un entero gigante se rechaza por el campo, no por un crash del validador.

    El techo existia porque el cociente `width / height` lanzaba `OverflowError`
    sobre enteros sin cota, y pydantic no envolvia ese crash. Desde #80 la
    validacion multiplica en lugar de dividir, asi que el `OverflowError` ya no
    aplica: el techo sigue rechazando el documento absurdo, ahora por la propia
    restriccion de campo.
    """
    with pytest.raises(ValidationError, match="less than or equal to 65536"):
        _ = SplitScreenConfig.model_validate(_split(width=10**400, height=2))


def test_c1c_cuadrado_1_1_valida_contra_su_lienzo() -> None:
    """1:1 con un lienzo cuadrado pasa, y uno vertical no."""
    square = SplitScreenConfig.model_validate(_split(aspect="1:1", width=1080, height=1080))
    assert square.aspect is Aspect.SQUARE
    assert (square.width, square.height) == (1080, 1080)

    with pytest.raises(ValidationError) as excinfo:
        _ = SplitScreenConfig.model_validate(_split(aspect="1:1", width=1080, height=1920))
    assert "1:1" in str(excinfo.value)


def test_c1d_retrato_4_5_valida_contra_su_lienzo() -> None:
    """4:5 con un lienzo 864x1080 pasa, y uno cuadrado no."""
    portrait = SplitScreenConfig.model_validate(_split(aspect="4:5", width=864, height=1080))
    assert portrait.aspect is Aspect.PORTRAIT_4_5
    assert (portrait.width, portrait.height) == (864, 1080)

    with pytest.raises(ValidationError) as excinfo:
        _ = SplitScreenConfig.model_validate(_split(aspect="4:5", width=1080, height=1080))
    assert "4:5" in str(excinfo.value)


def test_c1f_no_hay_tolerancia_de_coma_flotante_en_el_validador() -> None:
    """La exactitud es estructural: el validador no compara cocientes.

    Con `_MAX_CANVAS_PX` en 65536 la minima desviacion posible del cociente
    `width / height` es 1/(16*65529) = 9.5378e-07 (alcanzada en 36860x65529),
    casi 1000 veces por encima de la eps de 1e-9 que habia: ningun documento
    dentro del techo colaba por ella. Pero la eps seguia siendo una comparacion
    de floats esperando a que alguien subiera el techo, y el issue #80 pide
    eliminarla, no solo neutralizarla.

    Este test es el gate: si alguien reintroduce `_SPLIT_ASPECT_EPS`, un
    `abs(...) > ...` o un `.ratio` dentro del validador, falla AQUI.
    """
    source_file = Path(inspect.getfile(SplitScreenConfig))
    source = source_file.read_text(encoding="utf-8")
    validator_start = source.index("def _canvas_and_panels_are_viable")
    validator_end = source.index("\n    def ", validator_start + 10)
    validator = source[validator_start:validator_end]

    assert "_SPLIT_ASPECT_EPS" not in validator
    assert ".ratio" not in validator, "el validador sigue comparando el cociente flotante"
    assert "abs(" not in validator, "el validador sigue usando una comparacion con eps"
    assert "* self.aspect.denominator != self.height * self.aspect.numerator" in validator


def test_c1e_la_validacion_es_cruzamiento_entero_exacto() -> None:
    """La pertenencia del lienzo se decide con enteros, sin tolerancia.

    El cociente `width / height` es inexacto en coma flotante, asi que comparar
    con una eps deja la exactitud en manos de un numero magico. La multiplicacion
    cruzada no: 1098x1952 satisface 9:16 exacto y 1100x1952 no, y ninguno de los
    dos depende de una tolerancia.
    """
    # 9:16 exacto: 1098*16 == 1952*9 == 17568.
    exacto = SplitScreenConfig.model_validate(_split(aspect="9:16", width=1098, height=1952))
    assert exacto.aspect is Aspect.VERTICAL

    # Un lienzo que no satisface la igualdad se rechaza, sea cual sea la eps.
    with pytest.raises(ValidationError) as excinfo:
        _ = SplitScreenConfig.model_validate(_split(aspect="9:16", width=1100, height=1952))
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
    """El caso coherente: una salida, un lienzo de ESE formato.

    No es tautologico aunque las dos declaraciones digan 9:16: la regla podria
    comparar mal y aun asi dejar pasar este caso. Lo que lo hace util es que
    recorre las DOS ramas de la reconciliacion (una sola salida, aspecto
    coincidente) y exige que el lienzo sobreviva con su geometria intacta.
    """
    contract = make_contract(
        schema_version="1.2",
        outputs=[_VERTICAL],
        split_screen=_split(aspect="9:16"),
    )

    assert contract.split_screen is not None
    assert contract.split_screen.aspect is Aspect.VERTICAL
    assert (contract.split_screen.width, contract.split_screen.height) == (1080, 1920)


def test_d1b_un_lienzo_horizontal_tambien_puede_coincidir() -> None:
    """La reconciliacion no esta cableada a 9:16.

    Aqui el lienzo y la salida son los dos HORIZONTAL, y aun asi tienen que
    cargar. Si la comparacion se hubiera escrito contra el valor crudo "9:16" en
    vez de contra `outputs[0].aspect`, este contrato pasaria o fallaria por el
    motivo equivocado.
    """
    contract = make_contract(
        schema_version="1.2",
        outputs=[_HORIZONTAL],
        split_screen=_split(aspect="16:9", **_CANVAS_16x9),
    )

    assert contract.split_screen is not None
    assert contract.split_screen.aspect is Aspect.HORIZONTAL


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
