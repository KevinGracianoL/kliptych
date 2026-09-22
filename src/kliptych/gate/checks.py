"""Validadores deterministas del gate, indexados por rule_id.

Catálogo de reglas que el contrato puede declarar:

- ``artifact.integrity``: el artefacto existe y su hash es calculable.
- ``artifact.video_stream``: el artefacto tiene pista de video.
- ``assets.required``: los assets obligatorios están registrados e íntegros.
- ``audio.present``: el artefacto tiene pista de audio.
- ``caption.first_line``: el caption abre con la primera línea exigida.
- ``caption.forbidden``: no aparecen términos prohibidos en el caption.
- ``caption.required_hashtag``: están los hashtags obligatorios.
- ``caption.required_mention``: están las menciones obligatorias.
- ``duration.min`` / ``duration.max``: duración dentro del rango.
- ``subtitles.spelling_lock``: spelling exacto en subtítulos.

Reglas declaradas sin validador registrado jamás pasan: el motor las marca
``unsupported`` (o ``manual_review`` si el contrato las clasificó así).
"""

from collections.abc import Callable
from dataclasses import dataclass, field

from kliptych.assets import AssetNotFoundError, AssetRegistry
from kliptych.contract import Contract, Format, PlatformRules
from kliptych.gate.models import CheckStatus, MediaInfo, Piece


@dataclass(frozen=True, slots=True)
class CheckOutcome:
    """Resultado interno de un validador, sin el id de la regla."""

    status: CheckStatus
    evidence: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class GateContext:
    """Datos resueltos que recibe cada validador."""

    contract: Contract
    rules: PlatformRules
    piece: Piece
    artifact_sha256: str | None
    media: MediaInfo | None
    assets: AssetRegistry


Validator = Callable[[GateContext], CheckOutcome]


def _pass(**evidence: object) -> CheckOutcome:
    return CheckOutcome(status=CheckStatus.PASS, evidence=dict(evidence))


def _fail(**evidence: object) -> CheckOutcome:
    return CheckOutcome(status=CheckStatus.FAIL, evidence=dict(evidence))


def _unsupported(reason: str) -> CheckOutcome:
    return CheckOutcome(status=CheckStatus.UNSUPPORTED, evidence={"reason": reason})


def check_artifact_integrity(context: GateContext) -> CheckOutcome:
    """Verifica que el artefacto exista y su hash sea calculable.

    Args:
        context: Contexto resuelto del gate.

    Returns:
        PASS con el sha256 del artefacto, o FAIL si no se pudo leer.
    """
    if context.artifact_sha256 is None:
        return _fail(reason="el artefacto no existe o no se pudo leer")
    return _pass(sha256=context.artifact_sha256)


def check_duration_min(context: GateContext) -> CheckOutcome:
    """Verifica que la duración del artefacto no baje del mínimo.

    Args:
        context: Contexto resuelto del gate.

    Returns:
        PASS/FAIL con la cota y la duración medida, o UNSUPPORTED sin probe.
    """
    return _duration_outcome(context, context.rules.duration.min_s, minimum=True)


def check_duration_max(context: GateContext) -> CheckOutcome:
    """Verifica que la duración del artefacto no supere el máximo.

    Args:
        context: Contexto resuelto del gate.

    Returns:
        PASS/FAIL con la cota y la duración medida, o UNSUPPORTED sin probe.
    """
    return _duration_outcome(context, context.rules.duration.max_s, minimum=False)


def check_audio_present(context: GateContext) -> CheckOutcome:
    """Verifica que el artefacto tenga pista de audio.

    Args:
        context: Contexto resuelto del gate.

    Returns:
        PASS si hay pista de audio; FAIL si no; UNSUPPORTED sin probe.
    """
    if context.media is None:
        return _unsupported("no se pudo inspeccionar el artefacto")
    if context.media.has_audio:
        return _pass(has_audio=True)
    return _fail(has_audio=False)


def check_video_stream(context: GateContext) -> CheckOutcome:
    """Verifica que el artefacto tenga pista de video.

    Args:
        context: Contexto resuelto del gate.

    Returns:
        PASS/FAIL según la pista de video; UNSUPPORTED sin probe o si el
        contrato no es de formato video.
    """
    if context.contract.format is not Format.VIDEO:
        return _unsupported("la validación de video no aplica a este formato")
    if context.media is None:
        return _unsupported("no se pudo inspeccionar el artefacto")
    if context.media.has_video:
        return _pass(width=context.media.width, height=context.media.height)
    return _fail(has_video=False)


def check_required_mentions(context: GateContext) -> CheckOutcome:
    """Verifica las menciones obligatorias del caption.

    Args:
        context: Contexto resuelto del gate.

    Returns:
        PASS si todas las menciones (``required_mentions`` más
        ``caption_rules.must_mention``) aparecen; FAIL con las faltantes.
    """
    required = [
        *context.rules.required_mentions,
        *context.rules.caption_rules.must_mention,
    ]
    missing = [mention for mention in required if mention not in context.piece.caption]
    evidence: dict[str, object] = {"required": required, "missing": missing}
    return CheckOutcome(
        status=CheckStatus.FAIL if missing else CheckStatus.PASS,
        evidence=evidence,
    )


def check_required_hashtags(context: GateContext) -> CheckOutcome:
    """Verifica los hashtags obligatorios.

    Args:
        context: Contexto resuelto del gate.

    Returns:
        PASS si cada hashtag aparece en el caption o en el campo hashtags
        (comparación insensible a mayúsculas); FAIL con los faltantes.
    """
    available = {_normalize_hashtag(tag) for tag in context.piece.hashtags}
    caption = context.piece.caption.lower()
    missing = [
        tag
        for tag in context.rules.required_hashtags
        if _normalize_hashtag(tag) not in available and tag.lower() not in caption
    ]
    evidence: dict[str, object] = {
        "required": list(context.rules.required_hashtags),
        "missing": missing,
    }
    return CheckOutcome(
        status=CheckStatus.FAIL if missing else CheckStatus.PASS,
        evidence=evidence,
    )


def check_forbidden_terms(context: GateContext) -> CheckOutcome:
    """Verifica que no aparezcan términos prohibidos en el caption.

    Args:
        context: Contexto resuelto del gate.

    Returns:
        PASS si no aparece ningún término de ``caption_rules.forbidden``;
        FAIL con los términos encontrados.
    """
    caption = context.piece.caption.lower()
    found = [term for term in context.rules.caption_rules.forbidden if term.lower() in caption]
    evidence: dict[str, object] = {
        "forbidden": list(context.rules.caption_rules.forbidden),
        "found": found,
    }
    return CheckOutcome(
        status=CheckStatus.FAIL if found else CheckStatus.PASS,
        evidence=evidence,
    )


def check_first_line(context: GateContext) -> CheckOutcome:
    """Verifica que el caption abra con la primera línea exigida.

    Args:
        context: Contexto resuelto del gate.

    Returns:
        PASS si no hay exigencia o la primera línea la contiene; FAIL en
        caso contrario.
    """
    required = context.rules.caption_rules.first_line
    if required is None:
        return _pass(required=None)
    first_line = context.piece.caption.splitlines()[0].strip() if context.piece.caption else ""
    evidence: dict[str, object] = {"required": required, "first_line": first_line}
    return CheckOutcome(
        status=CheckStatus.PASS if required.strip() in first_line else CheckStatus.FAIL,
        evidence=evidence,
    )


def check_spelling_locks(context: GateContext) -> CheckOutcome:
    """Verifica el spelling exacto de los locks en los subtítulos.

    Args:
        context: Contexto resuelto del gate.

    Returns:
        PASS si no hay subtítulos o todos los locks aparecen literalmente;
        FAIL con los locks faltantes.
    """
    if context.piece.subtitle_text is None:
        return _pass(reason="la pieza no tiene subtítulos que revisar")
    missing = [
        lock for lock in context.contract.spelling_locks if lock not in context.piece.subtitle_text
    ]
    evidence: dict[str, object] = {
        "locks": list(context.contract.spelling_locks),
        "missing": missing,
    }
    return CheckOutcome(
        status=CheckStatus.FAIL if missing else CheckStatus.PASS,
        evidence=evidence,
    )


def check_required_assets(context: GateContext) -> CheckOutcome:
    """Verifica que los assets obligatorios estén registrados e íntegros.

    Args:
        context: Contexto resuelto del gate.

    Returns:
        PASS si todos los assets requeridos existen con hash intacto; FAIL
        con los ausentes y los manipulados.
    """
    missing: list[str] = []
    tampered: list[str] = []
    for asset in context.contract.assets.required:
        try:
            _ = context.assets.get(asset.asset_id)
        except AssetNotFoundError:
            missing.append(asset.asset_id)
            continue
        if not context.assets.verify(asset.asset_id):
            tampered.append(asset.asset_id)
    evidence: dict[str, object] = {"missing": missing, "tampered": tampered}
    return CheckOutcome(
        status=CheckStatus.FAIL if missing or tampered else CheckStatus.PASS,
        evidence=evidence,
    )


DEFAULT_VALIDATORS: dict[str, Validator] = {
    "artifact.integrity": check_artifact_integrity,
    "artifact.video_stream": check_video_stream,
    "assets.required": check_required_assets,
    "audio.present": check_audio_present,
    "caption.first_line": check_first_line,
    "caption.forbidden": check_forbidden_terms,
    "caption.required_hashtag": check_required_hashtags,
    "caption.required_mention": check_required_mentions,
    "duration.max": check_duration_max,
    "duration.min": check_duration_min,
    "subtitles.spelling_lock": check_spelling_locks,
}


def _duration_outcome(context: GateContext, bound: int | None, *, minimum: bool) -> CheckOutcome:
    if context.media is None:
        return _unsupported("no se pudo inspeccionar el artefacto")
    if bound is None:
        return _pass(bound=None)
    duration = context.media.duration_s
    within = duration >= bound if minimum else duration <= bound
    label = "min_s" if minimum else "max_s"
    evidence: dict[str, object] = {label: bound, "duration_s": duration}
    return CheckOutcome(
        status=CheckStatus.PASS if within else CheckStatus.FAIL,
        evidence=evidence,
    )


def _normalize_hashtag(tag: str) -> str:
    return tag.strip().lstrip("#").lower()
