"""Validador opt-in de brand safety con LLM, indexado por rule_id.

- ``brand.safety``: solo se activa si el contrato prohíbe explícitamente
  la controversia o exige brand safety (menciones en ``prohibitions``);
  en campañas estilo CB20 el evaluador ni se invoca y el resultado es
  ``pass``.

Con la regla activa, el texto de la pieza (caption, hashtags y
``subtitle_text``) se somete a un evaluador LLM inyectado: riesgo
detectado → ``manual_review``; texto limpio → ``pass``. Fail-closed: sin
evaluador configurado, con excepción/timeout del evaluador o con
respuesta inválida, el resultado es ``manual_review`` (jamás un ``pass``
silencioso). El orquestador inyecta el evaluador real con
``make_brand_safety_validator``; el catálogo por defecto no trae LLM y
por tanto exige revisión humana cuando la regla está activa.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, cast, runtime_checkable

from kliptych.gate.models import CheckOutcome, CheckStatus, GateContext
from kliptych.gate.text import normalize_text

if TYPE_CHECKING:
    from collections.abc import Mapping

    from kliptych.contract import Contract
    from kliptych.gate.models import Piece

_RULE = "brand.safety"

BRAND_SAFETY_MENTIONS: tuple[str, ...] = (
    "controvers",
    "polemic",
    "brand safety",
    "seguridad de marca",
    "toxic",
    "sin groserias",
    "contenido apto para marcas",
    "apto para todo publico",
    "family friendly",
    "nada de lenguaje ofensivo",
    "sin lenguaje ofensivo",
    "sin insultos",
    "no insultos",
    "no profanity",
    "sin malas palabras",
    "malas palabras",
    "prohibido contenido nsfw",
    "nsfw",
    "sin temas sensibles",
    "temas sensibles",
    "no politica ni religion",
    "no contenido para adultos",
    "contenido para adultos",
)
_BRAND_SAFETY_MENTIONS = BRAND_SAFETY_MENTIONS

_BRAND_SAFETY_SYSTEM_PROMPT = (
    "Eres el evaluador de brand safety de Kliptych. Recibes los textos de "
    "una pieza (caption, hashtags y subtítulos del directo) y devuelves "
    "ÚNICAMENTE un JSON con la forma "
    '{"risk": bool, "categories": [str], "reason": str|null}: "risk" es true '
    "si los textos conllevan controversia, toxicidad, odio o riesgo para la "
    "marca; "
    '"categories" lista las categorías detectadas (vacía sin '
    'riesgo); "reason" explica el veredicto en una frase o null.'
)


class BrandSafetyError(Exception):
    """El riesgo de brand safety no se pudo evaluar con el LLM."""


@dataclass(frozen=True, slots=True)
class BrandSafetyAssessment:
    """Veredicto del evaluador sobre los textos de la pieza."""

    risk: bool
    categories: tuple[str, ...] = ()
    reason: str | None = None


Assessor = Callable[[str], BrandSafetyAssessment]
"""Evalúa los textos de la pieza y devuelve su veredicto de riesgo."""


@runtime_checkable
class ChatJsonModel(Protocol):
    """Backend chat con respuesta JSON, como el runtime OpenAI-compatible."""

    def chat_json(self, *, system_prompt: str, user_content: str) -> object:
        """Envía prompt y contenido de usuario esperando un objeto JSON."""
        ...


_ChatJsonModel = ChatJsonModel


def check_brand_safety(context: GateContext, *, assess: Assessor | None = None) -> CheckOutcome:
    """Verifica el riesgo de brand safety solo si la campaña lo exige.

    Args:
        context: Contexto resuelto del gate.
        assess: Evaluador LLM inyectado; ``None`` exige revisión humana
            cuando la regla está activa (fail-closed).

    Returns:
        PASS si la campaña no exige brand safety o el evaluador no ve
        riesgo; MANUAL_REVIEW si ve riesgo, si no hay evaluador o si el
        evaluador falla o devuelve algo inválido.
    """
    if not _brand_safety_required(context.contract):
        return CheckOutcome(
            status=CheckStatus.PASS,
            evidence={"rule": _RULE, "active": False, "reason": "la campaña no exige brand safety"},
        )
    if assess is None:
        return CheckOutcome(
            status=CheckStatus.MANUAL_REVIEW,
            evidence={
                "rule": _RULE,
                "active": True,
                "reason": (
                    "la campaña exige brand safety pero no hay evaluador LLM "
                    "configurado; requiere revisión humana"
                ),
            },
        )
    try:
        assessment = assess(_safety_text(context.piece))
    except Exception as error:
        return CheckOutcome(
            status=CheckStatus.MANUAL_REVIEW,
            evidence={
                "rule": _RULE,
                "active": True,
                "reason": (
                    "el evaluador de brand safety falló "
                    f"({type(error).__name__}); requiere revisión humana"
                ),
            },
        )
    if assessment.risk:
        return CheckOutcome(
            status=CheckStatus.MANUAL_REVIEW,
            evidence={
                "rule": _RULE,
                "active": True,
                "risk": True,
                "categories": list(assessment.categories),
                "reason": assessment.reason,
            },
        )
    return CheckOutcome(
        status=CheckStatus.PASS,
        evidence={"rule": _RULE, "active": True, "risk": False},
    )


def make_brand_safety_validator(assess: Assessor) -> Callable[[GateContext], CheckOutcome]:
    """Construye el validador ``brand.safety`` con un evaluador inyectado.

    Args:
        assess: Evaluador LLM (o doble de tests) sobre los textos de la pieza.

    Returns:
        El validador listo para el catálogo del gate.
    """

    def _check(context: GateContext) -> CheckOutcome:
        return check_brand_safety(context, assess=assess)

    return _check


def make_model_assessor(model: _ChatJsonModel) -> Assessor:
    """Adapta un backend chat con JSON al evaluador de brand safety.

    Args:
        model: Backend con ``chat_json`` (p. ej. el runtime OpenAI-compatible).

    Returns:
        El evaluador que pregunta al modelo y valida su respuesta; ante
        JSON inválido lanza ``BrandSafetyError`` (el validador lo traduce
        a ``manual_review``, jamás a ``pass``).
    """

    def _assess(text: str) -> BrandSafetyAssessment:
        payload = model.chat_json(
            system_prompt=_BRAND_SAFETY_SYSTEM_PROMPT,
            user_content=json.dumps({"texts": text}, ensure_ascii=False, sort_keys=True),
        )
        return parse_brand_safety_response(payload)

    return _assess


def parse_brand_safety_response(payload: object) -> BrandSafetyAssessment:
    """Valida la respuesta JSON del LLM como veredicto de riesgo.

    Args:
        payload: Valor JSON ya parseado del primer choice del backend.

    Returns:
        El veredicto con riesgo, categorías y motivo.

    Raises:
        BrandSafetyError: Si la respuesta no tiene la forma exigida
            (``risk`` bool obligatorio; ``categories`` lista de str y
            ``reason`` str o null cuando aparecen).
    """
    if not isinstance(payload, dict):
        msg = f"respuesta de brand safety no es un objeto: {type(payload).__name__}"
        raise BrandSafetyError(msg)
    mapping: Mapping[str, object] = cast("dict[str, object]", payload)
    risk = mapping.get("risk")
    if not isinstance(risk, bool):
        msg = "respuesta de brand safety sin 'risk' booleano"
        raise BrandSafetyError(msg)
    categories = _parse_categories(mapping.get("categories"))
    reason = mapping.get("reason")
    if reason is not None and not isinstance(reason, str):
        msg = "respuesta de brand safety con 'reason' no textual"
        raise BrandSafetyError(msg)
    return BrandSafetyAssessment(risk=risk, categories=categories, reason=reason)


def _parse_categories(raw: object) -> tuple[str, ...]:
    """Valida las categorías de riesgo de la respuesta del LLM.

    Args:
        raw: Valor crudo de ``categories`` (o ``None`` si ausente).

    Returns:
        Las categorías como tupla (vacía si ausente).

    Raises:
        BrandSafetyError: Si no es una lista de str.
    """
    if raw is None:
        return ()
    if not isinstance(raw, list):
        msg = "respuesta de brand safety con 'categories' no lista de str"
        raise BrandSafetyError(msg)
    cleaned: list[str] = []
    for item in cast("list[object]", raw):
        if not isinstance(item, str):
            msg = "respuesta de brand safety con 'categories' no lista de str"
            raise BrandSafetyError(msg)
        cleaned.append(item)
    return tuple(cleaned)


def _brand_safety_required(contract: Contract) -> bool:
    """Indica si el contrato exige brand safety en sus campos o prohibiciones.

    Args:
        contract: Contrato validado de la campaña.

    Returns:
        True si contract.brand_safety_required es True o si alguna prohibición
        menciona controversia, toxicidad o brand safety explícitos.
    """
    if contract.brand_safety_required:
        return True
    haystack = normalize_text(" ".join(contract.prohibitions))
    return any(normalize_text(mention) in haystack for mention in _BRAND_SAFETY_MENTIONS)


def _safety_text(piece: Piece) -> str:
    """Junta los textos de la pieza para el evaluador.

    Args:
        piece: Pieza con caption, hashtags y subtítulos del directo.

    Returns:
        Los textos unidos por líneas (subtítulos solo si existen).
    """
    parts = [piece.caption, *piece.hashtags]
    if piece.subtitle_text:
        parts.append(piece.subtitle_text)
    return "\n".join(parts)
