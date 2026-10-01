"""Resolutor ContractDraft -> Contract: normaliza, clasifica y no inventa.

Política de resolución:

- Un campo obligatorio sin valor resuelto (``missing``) o en conflicto
  (``conflict``) produce un issue bloqueante: no se construye contrato y el
  resultado es ``MANUAL_REVIEW``. Sin ``mode``/``format`` el resultado es
  ``NEW_ARCHETYPE`` (no hay ruta de pipeline).
- Los defaults del schema solo relajan (``audio_rule=any``, ``attribution=none``,
  ``link_in_bio=false``, listas vacías, watermark no exigido): no agregan
  restricciones, por eso no se reportan como issues.
- Una restricción activa sin clasificar en ``rules`` se clasifica como ``hard``
  (fail-closed: se declaró, se exige) y se reporta como ``rule_defaulted``.
- Las reglas base del gate (``artifact.integrity`` siempre,
  ``artifact.video_stream`` para formato video y ``audio.present`` cuando alguna
  plataforma exige audio) se fuerzan como ``hard``: el brief (§5.2/5.4) las
  considera checks del núcleo, no reglas de campaña, y un draft no puede
  degradarlas. Si el draft las clasificaba en otra categoría, se reporta
  ``rule_defaulted``.
- Los assets obligatorios se resuelven contra el ``AssetRegistry``; si no se
  pueden registrar, el resultado es ``MANUAL_REVIEW``. Un asset opcional que no
  se puede resolver se descarta y se reporta (no bloquea).
"""

import math
import re
import unicodedata
from collections.abc import Sequence
from contextlib import suppress
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol, cast

from pydantic import BaseModel, ValidationError

from kliptych.assets import AssetError, AssetNotFoundError, AssetRegistry
from kliptych.contract import (
    AnalyticsProofRequired,
    AssetBundle,
    AssetDraft,
    AssetRef,
    Attribution,
    AttributionDraft,
    AttributionType,
    AudioPolicy,
    AudioRule,
    CaptionRules,
    Confidence,
    Contract,
    ContractDraft,
    DurationRange,
    FieldCandidate,
    Format,
    GeoTarget,
    GlobalRestrictions,
    Languages,
    LinkRules,
    LyricConfig,
    MinViewsForPayout,
    Mode,
    OfficialAudio,
    Platform,
    PlatformDraft,
    PlatformRules,
    RuleSet,
    Segment,
    SourceEvidence,
    SplitScreenConfig,
    SplitScreenDraft,
    TimestampRange,
    UnmappedRule,
    Watermark,
    WatermarkPosition,
    active_restriction_rules,
)
from kliptych.contract.base import ContractBase
from kliptych.gate.brand_safety import BRAND_SAFETY_MENTIONS
from kliptych.gate.text import normalize_text


class ResolutionStatus(StrEnum):
    """Resultado de resolver un draft."""

    RESOLVED = "resolved"
    MANUAL_REVIEW = "manual_review"
    NEW_ARCHETYPE = "new_archetype"


class IssueCode(StrEnum):
    """Motivo de un issue de resolución."""

    MISSING_REQUIRED = "missing_required"
    CONFLICT = "conflict"
    UNRESOLVED_ASSET = "unresolved_asset"
    OPTIONAL_ASSET_DROPPED = "optional_asset_dropped"
    RULE_DEFAULTED = "rule_defaulted"
    INVALID_CONTRACT = "invalid_contract"
    MODE_NOT_IMPLEMENTED = "mode_not_implemented"
    UNSAFE_ASSET_SCOPE = "unsafe_asset_scope"


_BLOCKING_CODES = frozenset(
    {
        IssueCode.MISSING_REQUIRED,
        IssueCode.CONFLICT,
        IssueCode.UNRESOLVED_ASSET,
        IssueCode.INVALID_CONTRACT,
    }
)


class ResolutionIssue(ContractBase):
    """Hallazgo del resolutor, con campo afectado y detalle."""

    code: IssueCode
    field: str
    detail: str


class ResolutionResult(ContractBase):
    """Resultado de la resolución: contrato (o ``None``) más issues."""

    status: ResolutionStatus
    contract: Contract | None = None
    issues: tuple[ResolutionIssue, ...] = ()

    @property
    def resolved(self) -> bool:
        """Indica si se produjo un contrato listo para el gate.

        Returns:
            ``True`` solo cuando hay contrato y estado ``resolved``.
        """
        return self.status is ResolutionStatus.RESOLVED and self.contract is not None


@dataclass(frozen=True, slots=True)
class _RuleContext:
    platforms: dict[Platform, PlatformRules]
    global_restrictions: GlobalRestrictions
    format_: Format
    brief_text: str | None = None
    split_screen: SplitScreenConfig | None = None


class _ConfidenceCarrier(Protocol):
    @property
    def confidence(self) -> Confidence: ...


class ProvenanceError(ValueError):
    """Fallo mecánico en la verificación de procedencia de una cita del brief."""


def _collect_all_evidence(node: object) -> list[SourceEvidence]:
    evidences: list[SourceEvidence] = []
    if isinstance(node, SourceEvidence):
        evidences.append(node)
        return evidences
    if isinstance(node, BaseModel):
        for field_name in type(node).model_fields:
            field_value = cast("object", getattr(node, field_name))
            if field_value is not None:
                evidences.extend(_collect_all_evidence(field_value))
    elif isinstance(node, (list, tuple)):
        for item in cast("Sequence[object]", node):
            evidences.extend(_collect_all_evidence(item))
    elif isinstance(node, dict):
        for value in cast("dict[object, object]", node).values():
            evidences.extend(_collect_all_evidence(value))
    return evidences


def verify_provenance(draft: ContractDraft, brief_text: str) -> None:
    """Verifica mecánicamente que cada cita y sus offsets existan en el brief.

    Args:
        draft: Borrador de contrato con evidencias textuales.
        brief_text: Texto crudo del brief original.

    Raises:
        ProvenanceError: Si alguna cita está fabricada (no existe en brief_text)
            o si los offsets no coinciden exactamente con la cita en brief_text.
    """
    for evidence in _collect_all_evidence(draft):
        quote = evidence.quote
        if quote not in brief_text:
            msg = f"Cita fabricada: {quote!r} no existe en el texto del brief"
            raise ProvenanceError(msg)
        start = getattr(evidence, "start_char", getattr(evidence, "start", None))
        end = getattr(evidence, "end_char", getattr(evidence, "end", None))
        if start is not None and end is not None:
            if start < 0 or end < start or end > len(brief_text):
                msg = (
                    f"Offsets fuera de rango [{start}:{end}] para cita {quote!r} "
                    f"en brief de longitud {len(brief_text)}"
                )
                raise ProvenanceError(msg)
            if brief_text[start:end] != quote:
                msg = (
                    f"Offsets desalineados: brief[{start}:{end}] es "
                    f"{brief_text[start:end]!r}, se esperaba {quote!r}"
                )
                raise ProvenanceError(msg)
        elif start is not None and end is None:
            if start < 0 or start + len(quote) > len(brief_text):
                msg = (
                    f"Offset de inicio fuera de rango [{start}] para cita {quote!r} "
                    f"en brief de longitud {len(brief_text)}"
                )
                raise ProvenanceError(msg)
            if brief_text[start : start + len(quote)] != quote:
                msg = (
                    f"Offsets desalineados: brief[{start}:{start + len(quote)}] es "
                    f"{brief_text[start : start + len(quote)]!r}, se esperaba {quote!r}"
                )
                raise ProvenanceError(msg)


def resolve_contract(
    draft: ContractDraft,
    *,
    registry: AssetRegistry,
    brief_text: str | None = None,
) -> ResolutionResult:
    """Convierte un ``ContractDraft`` en un ``Contract`` validado.

    Args:
        draft: Salida cruda del extractor, con evidencia por campo.
        registry: Registro de assets del workspace para resolver assets.
        brief_text: Texto crudo del brief original para verificación mecánica
            de procedencia. Si se proporciona, cada cita y offset en ``draft``
            se verifican contra este texto.

    Returns:
        El resultado con el contrato resuelto (si aplica) y los issues
        encontrados.
    """
    if brief_text is not None:
        verify_provenance(draft, brief_text)

    issues: list[ResolutionIssue] = _collect_conflicts(draft)

    mode = _required_value(draft.mode, "mode", issues)
    format_val = _value(draft.format)
    if format_val is None:
        if _brief_mentions_lyric_video(brief_text):
            format_ = Format.LYRIC_VIDEO
        else:
            format_ = _required_value(draft.format, "format", issues)
    else:
        format_ = format_val
    if mode is None or format_ is None:
        status = (
            ResolutionStatus.MANUAL_REVIEW
            if any(issue.code is IssueCode.CONFLICT for issue in issues)
            else ResolutionStatus.NEW_ARCHETYPE
        )
        return ResolutionResult(status=status, issues=tuple(issues))

    try:
        contract = _build_contract(
            draft, registry, mode=mode, format_=format_, issues=issues, brief_text=brief_text
        )
    except ValidationError as error:
        issues.append(
            ResolutionIssue(
                code=IssueCode.INVALID_CONTRACT,
                field="contract",
                detail=str(error),
            )
        )
        return ResolutionResult(status=ResolutionStatus.MANUAL_REVIEW, issues=tuple(issues))
    if contract is None:
        return ResolutionResult(status=ResolutionStatus.MANUAL_REVIEW, issues=tuple(issues))
    return ResolutionResult(
        status=ResolutionStatus.RESOLVED, contract=contract, issues=tuple(issues)
    )


def _resolve_brand_safety(
    draft: ContractDraft,
    brief_text: str | None,
    prohibitions: list[str],
) -> tuple[bool, str | None, list[str]]:
    brand_safety_indicated = (
        _brief_indicates_brand_safety(brief_text, prohibitions)
        or _value(draft.brand_safety) is True
        or _value(draft.brand_safety_required) is True
    )
    brand_safety_required = (
        _value(draft.brand_safety_required) is True
        or _value(draft.brand_safety) is True
        or brand_safety_indicated
    )
    brand_safety_citation = _value(draft.brand_safety_citation)
    if brand_safety_citation is None and brand_safety_required:
        if (
            draft.brand_safety_required is not None
            and draft.brand_safety_required.evidence is not None
        ):
            brand_safety_citation = draft.brand_safety_required.evidence.quote
        elif draft.brand_safety is not None and draft.brand_safety.evidence is not None:
            brand_safety_citation = draft.brand_safety.evidence.quote
        elif draft.prohibitions is not None and draft.prohibitions.evidence is not None:
            brand_safety_citation = draft.prohibitions.evidence.quote
        elif prohibitions:
            brand_safety_citation = prohibitions[0]
    if brand_safety_indicated or brand_safety_required:
        prohibitions_text = normalize_text(" ".join(prohibitions))
        if not any(normalize_text(m) in prohibitions_text for m in BRAND_SAFETY_MENTIONS):
            prohibitions = [*prohibitions, "brand safety"]
    return brand_safety_required, brand_safety_citation, prohibitions


_SECONDS_PER_MINUTE = 60.0
_SECONDS_PER_HOUR = 3600.0
_TIME_PARTS_MM_SS = 2
_TIME_PARTS_HH_MM_SS = 3

_CLOCK_RE = re.compile(r"^[0-9]+(:[0-5][0-9]){1,2}(\.[0-9]+)?$")
_NUMERIC_SEC_RE = re.compile(r"^[0-9]+(\.[0-9]+)?$")


def _parse_clock_seconds(parts: list[str]) -> float | None:
    try:
        nums = [float(p) for p in parts]
    except ValueError:
        return None
    if len(nums) == _TIME_PARTS_MM_SS:
        return nums[0] * _SECONDS_PER_MINUTE + nums[1]
    if len(nums) == _TIME_PARTS_HH_MM_SS:
        return nums[0] * _SECONDS_PER_HOUR + nums[1] * _SECONDS_PER_MINUTE + nums[2]
    return None


def parse_timestamp_seconds(value: float | str) -> float | None:
    """Convierte un valor de tiempo a float seconds con validación estricta ASCII.

    Args:
        value: Valor numérico o texto en formato segundos, 'MM:SS' o 'HH:MM:SS'.

    Returns:
        Segundos como float finito no negativo, o None si el valor es inválido.
    """
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        val = float(value)
        return val if math.isfinite(val) and val >= 0.0 else None
    cleaned = value.strip().rstrip("sS").strip()
    if not cleaned:
        return None
    val: float | None = None
    if _NUMERIC_SEC_RE.match(cleaned):
        with suppress(ValueError):
            val = float(cleaned)
    elif _CLOCK_RE.match(cleaned):
        val = _parse_clock_seconds(cleaned.split(":"))
    return val if val is not None and math.isfinite(val) and val >= 0.0 else None


_TIME_PATTERN = r"(?:[0-9]{1,2}:)?[0-9]{1,2}:[0-9]{2}(?:\.[0-9]+)?"
_SEC_PATTERN = r"[0-9]+(?:\.[0-9]+)?\s*s?"
_TIME_SEP = r"(?:-|[\u2013\u2014]|\ba\b|\bal\b|\bto\b|\bhasta\b)"
_RANGE_RE = re.compile(
    rf"(?P<start>{_TIME_PATTERN}|{_SEC_PATTERN})\s*{_TIME_SEP}\s*(?P<end>{_TIME_PATTERN}|{_SEC_PATTERN})",
    re.IGNORECASE,
)
_CUTOFF_LABEL_RE = re.compile(
    r"\b(?:timestamps?|cortes?|cortar\s+de(?:l)?|corte\s+de(?:l)?|minutos?|segmentos?|fragmentos?|marcas?|clip\s+from)\b",
    re.IGNORECASE,
)
_EXCLUDED_CONTEXT_RE = re.compile(
    r"\b(?:durar|duracion|duración|duration|horario|schedule|horas?)\b",
    re.IGNORECASE,
)
_CLAUSE_SPLIT_RE = re.compile(r"[\r\n;]+|(?<=\S)\.\s+")


def _is_valid_cutoff_clause(clause: str) -> bool:
    cleaned = clause.strip()
    if not cleaned or _EXCLUDED_CONTEXT_RE.search(cleaned):
        return False
    return bool(_CUTOFF_LABEL_RE.search(cleaned))


def _extract_clause_ranges(
    clause: str,
    seen: set[tuple[float, float]],
    ranges: list[TimestampRange],
    issues: list[ResolutionIssue] | None,
) -> None:
    for match in _RANGE_RE.finditer(clause):
        start = parse_timestamp_seconds(match.group("start"))
        end = parse_timestamp_seconds(match.group("end"))
        if start is None or end is None:
            continue
        if start >= end:
            if issues is not None:
                issues.append(
                    ResolutionIssue(
                        code=IssueCode.INVALID_CONTRACT,
                        field="timestamp_ranges",
                        detail=f"rango temporal invertido en el brief: start={start} >= end={end}",
                    )
                )
        elif (start, end) not in seen:
            seen.add((start, end))
            ranges.append(TimestampRange(start_sec=start, end_sec=end))


def _extract_timestamp_ranges_from_text(
    brief_text: str, issues: list[ResolutionIssue] | None = None
) -> list[TimestampRange]:
    ranges: list[TimestampRange] = []
    seen: set[tuple[float, float]] = set()

    for line in brief_text.splitlines():
        for clause in _CLAUSE_SPLIT_RE.split(line):
            if _is_valid_cutoff_clause(clause):
                _extract_clause_ranges(clause.strip(), seen, ranges, issues)
    return ranges


def _extract_candidate_value(
    candidate: FieldCandidate[float | str] | dict[str, object] | float | str | None,
) -> float | str | None:
    if isinstance(candidate, FieldCandidate):
        val = candidate.value
        return val if isinstance(val, (int, float, str)) and not isinstance(val, bool) else None
    if isinstance(candidate, dict):
        val = candidate.get("value")
        return val if isinstance(val, (int, float, str)) and not isinstance(val, bool) else None
    if isinstance(candidate, (int, float, str)) and not isinstance(candidate, bool):
        return candidate
    return None


def _extract_candidate_quote(bound: object) -> str | None:
    if isinstance(bound, FieldCandidate):
        return bound.evidence.quote if bound.evidence else None
    if isinstance(bound, dict):
        d = cast("dict[str, object]", bound)
        ev = d.get("evidence")
        if isinstance(ev, dict):
            ev_dict = cast("dict[str, object]", ev)
            quote = ev_dict.get("quote")
            if isinstance(quote, str):
                return quote
        elif isinstance(ev, SourceEvidence):
            return ev.quote
    return None


def _extract_candidate_confidence(bound: object) -> object:
    if isinstance(bound, FieldCandidate):
        return bound.confidence
    if isinstance(bound, dict):
        d = cast("dict[str, object]", bound)
        return d.get("confidence")
    return None


def _check_confidence_provenance(
    bound: object,
    issues: list[ResolutionIssue],
    *,
    field: str = "timestamp_ranges",
) -> bool:
    conf = _extract_candidate_confidence(bound)
    if conf is None or (isinstance(conf, str) and not conf.strip()):
        issues.append(
            ResolutionIssue(
                code=IssueCode.MISSING_REQUIRED,
                field=field,
                detail=f"campo '{field}' sin confianza explícita",
            )
        )
        return False

    conf_str = str(conf).lower()
    if "conflict" in conf_str:
        issues.append(
            ResolutionIssue(
                code=IssueCode.CONFLICT,
                field=field,
                detail=f"campo '{field}' en conflicto",
            )
        )
        return False
    if "missing" in conf_str:
        issues.append(
            ResolutionIssue(
                code=IssueCode.MISSING_REQUIRED,
                field=field,
                detail=f"campo '{field}' con confianza 'missing'",
            )
        )
        return False
    if conf_str not in {"explicit", "inferred"}:
        issues.append(
            ResolutionIssue(
                code=IssueCode.INVALID_CONTRACT,
                field=field,
                detail=f"campo '{field}' con confianza inválida: {conf!r}",
            )
        )
        return False
    return True


def _check_evidence_provenance(
    bound: object,
    brief_text: str | None,
    issues: list[ResolutionIssue],
    *,
    field: str = "timestamp_ranges",
) -> bool:
    if isinstance(bound, dict):
        d = cast("dict[str, object]", bound)
        if "citation" in d or "quote" in d:
            issues.append(
                ResolutionIssue(
                    code=IssueCode.MISSING_REQUIRED,
                    field=field,
                    detail=f"campo '{field}' usa alias prohibido ('citation' o 'quote') en la raíz",
                )
            )
            return False
    quote = _extract_candidate_quote(cast("object", bound))
    if quote is None or not quote.strip():
        issues.append(
            ResolutionIssue(
                code=IssueCode.MISSING_REQUIRED,
                field=field,
                detail=f"campo '{field}' sin evidencia textual obligatoria",
            )
        )
        return False

    if brief_text is not None and quote not in brief_text:
        issues.append(
            ResolutionIssue(
                code=IssueCode.INVALID_CONTRACT,
                field=field,
                detail=f"cita de '{field}' no encontrada en el brief: {quote!r}",
            )
        )
        return False
    return True


def _check_candidate_provenance(
    bound: object,
    brief_text: str | None,
    issues: list[ResolutionIssue],
    *,
    field: str = "timestamp_ranges",
) -> bool:
    conf_ok = _check_confidence_provenance(bound, issues, field=field)
    ev_ok = _check_evidence_provenance(bound, brief_text, issues, field=field)
    return conf_ok and ev_ok


_QUOTE_TIME_RE = re.compile(rf"\b(?:{_TIME_PATTERN}|{_SEC_PATTERN})\b")


def _extract_ordered_times(text: str) -> list[float]:
    times: list[float] = []
    for match in _QUOTE_TIME_RE.finditer(text):
        token = match.group(0).strip()
        parsed = parse_timestamp_seconds(token)
        if parsed is not None:
            times.append(parsed)
    return times


def _extract_parsed_times(text: str) -> set[float]:
    return set(_extract_ordered_times(text))


def _check_numeric_correspondence(
    start_bound: object,
    end_bound: object,
    s: float,
    e: float,
    issues: list[ResolutionIssue],
) -> bool:
    start_quote = _extract_candidate_quote(start_bound)
    end_quote = _extract_candidate_quote(end_bound)
    start_times: set[float] = _extract_parsed_times(start_quote) if start_quote else set()
    end_times: set[float] = _extract_parsed_times(end_quote) if end_quote else set()

    has_s = any(math.isclose(t, s, abs_tol=1e-3) for t in start_times)
    has_e = any(math.isclose(t, e, abs_tol=1e-3) for t in end_times)
    if not (has_s and has_e):
        issues.append(
            ResolutionIssue(
                code=IssueCode.INVALID_CONTRACT,
                field="timestamp_ranges",
                detail=(
                    f"marcas temporales start={s} end={e} no se corresponden con la cita: "
                    f"start_quote={start_quote!r}, end_quote={end_quote!r}"
                ),
            )
        )
        return False

    if (
        start_quote is not None
        and end_quote is not None
        and start_quote.strip() == end_quote.strip()
    ):
        ordered = _extract_ordered_times(start_quote)
        in_order = any(
            math.isclose(ordered[i], s, abs_tol=1e-3) and math.isclose(ordered[j], e, abs_tol=1e-3)
            for i in range(len(ordered))
            for j in range(i + 1, len(ordered))
        )
        if not in_order:
            issues.append(
                ResolutionIssue(
                    code=IssueCode.INVALID_CONTRACT,
                    field="timestamp_ranges",
                    detail=(
                        f"marcas temporales start={s} end={e} desordenadas en la cita compartida: "
                        f"{start_quote!r}"
                    ),
                )
            )
            return False

    return True


def _resolve_timestamp_ranges(
    draft: ContractDraft,
    brief_text: str | None,
    issues: list[ResolutionIssue],
) -> tuple[TimestampRange, ...]:
    ranges: list[TimestampRange] = []
    for item in draft.timestamp_ranges:
        start_bound = item.start_sec
        end_bound = item.end_sec
        if start_bound is None or end_bound is None:
            issues.append(
                ResolutionIssue(
                    code=IssueCode.MISSING_REQUIRED,
                    field="timestamp_ranges",
                    detail="rango temporal incompleto: falta start_sec o end_sec",
                )
            )
            continue

        start_valid = _check_candidate_provenance(start_bound, brief_text, issues)
        end_valid = _check_candidate_provenance(end_bound, brief_text, issues)
        if not (start_valid and end_valid):
            continue

        start_raw = _extract_candidate_value(start_bound)
        end_raw = _extract_candidate_value(end_bound)
        if start_raw is None or end_raw is None:
            issues.append(
                ResolutionIssue(
                    code=IssueCode.INVALID_CONTRACT,
                    field="timestamp_ranges",
                    detail=f"rango temporal sin valor válido: start={start_raw}, end={end_raw}",
                )
            )
            continue

        s = parse_timestamp_seconds(start_raw)
        e = parse_timestamp_seconds(end_raw)
        if s is not None and e is not None and 0.0 <= s < e:
            if not _check_numeric_correspondence(start_bound, end_bound, s, e, issues):
                continue
            ranges.append(TimestampRange(start_sec=s, end_sec=e))
        else:
            issues.append(
                ResolutionIssue(
                    code=IssueCode.INVALID_CONTRACT,
                    field="timestamp_ranges",
                    detail=f"rango temporal inválido: start={start_raw}, end={end_raw}",
                )
            )
    if not ranges and brief_text and not issues:
        ranges.extend(_extract_timestamp_ranges_from_text(brief_text, issues=issues))
    return tuple(ranges)


def _extract_candidate_str(
    candidate: FieldCandidate[str] | dict[str, object] | str | None,
) -> str | None:
    if isinstance(candidate, FieldCandidate):
        return candidate.value
    if isinstance(candidate, dict):
        val = candidate.get("value")
        return str(val) if isinstance(val, (str, int, float)) else None
    if isinstance(candidate, str):
        return candidate
    return None


def _extract_candidate_bool(candidate: object) -> bool | None:
    if isinstance(candidate, FieldCandidate):
        cand = cast("FieldCandidate[bool]", candidate)
        return cand.value
    if isinstance(candidate, dict):
        d = cast("dict[str, object]", candidate)
        val = d.get("value")
        return bool(val) if isinstance(val, bool) else None
    if isinstance(candidate, bool):
        return candidate
    return None


def _verify_lrc_asset_candidate(
    lrc_asset_id: str | None,
    registry: AssetRegistry,
    issues: list[ResolutionIssue],
) -> bool:
    if not lrc_asset_id:
        return True
    try:
        intact = registry.verify(lrc_asset_id)
    except AssetNotFoundError:
        issues.append(
            ResolutionIssue(
                code=IssueCode.UNRESOLVED_ASSET,
                field="lyric_video.lrc_asset_id",
                detail=f"el asset de letras '{lrc_asset_id}' no existe en el registro",
            )
        )
        return False
    except (AssetError, OSError) as error:
        issues.append(
            ResolutionIssue(
                code=IssueCode.UNRESOLVED_ASSET,
                field="lyric_video.lrc_asset_id",
                detail=f"error al verificar el asset de letras '{lrc_asset_id}': {error}",
            )
        )
        return False
    if not intact:
        issues.append(
            ResolutionIssue(
                code=IssueCode.UNRESOLVED_ASSET,
                field="lyric_video.lrc_asset_id",
                detail=f"el asset de letras '{lrc_asset_id}' tiene integridad comprometida",
            )
        )
        return False
    return True


def _resolve_lyric_video(
    draft: ContractDraft,
    format_: Format,
    brief_text: str | None,
    registry: AssetRegistry,
    issues: list[ResolutionIssue],
) -> LyricConfig | None:
    lyric_draft = draft.lyric_video
    if lyric_draft is None:
        if format_ is Format.LYRIC_VIDEO:
            return LyricConfig(lrclib_enabled=True)
        return None

    fields_to_check: list[tuple[str, object]] = [
        ("lrc_asset_id", lyric_draft.lrc_asset_id),
        ("track_name", lyric_draft.track_name),
        ("artist_name", lyric_draft.artist_name),
        ("lrclib_enabled", lyric_draft.lrclib_enabled),
    ]

    has_error = False
    for _, candidate_bound in fields_to_check:
        if candidate_bound is not None and not _check_candidate_provenance(
            candidate_bound, brief_text, issues, field="lyric_video"
        ):
            has_error = True

    if has_error:
        return None

    lrc_asset_id = _extract_candidate_str(lyric_draft.lrc_asset_id)
    if not _verify_lrc_asset_candidate(lrc_asset_id, registry, issues):
        return None

    track_name = _extract_candidate_str(lyric_draft.track_name)
    artist_name = _extract_candidate_str(lyric_draft.artist_name)
    lrclib_enabled_raw = _extract_candidate_bool(lyric_draft.lrclib_enabled)
    lrclib_enabled = True if lrclib_enabled_raw is None else lrclib_enabled_raw

    try:
        return LyricConfig(
            lrc_asset_id=lrc_asset_id,
            track_name=track_name,
            artist_name=artist_name,
            lrclib_enabled=lrclib_enabled,
        )
    except ValidationError as err:
        issues.append(
            ResolutionIssue(
                code=IssueCode.INVALID_CONTRACT,
                field="lyric_video",
                detail=str(err),
            )
        )
        return None


_SPLIT_NUMBER_RE = re.compile(r"[0-9]+(?:\.[0-9]+)?")
_SPLIT_PERCENT_SCALE = 100.0
_SPLIT_NUMERIC_TOLERANCE = 1e-6


def _extract_split_int(bound: object) -> int | None:
    """Extrae un entero del candidato, rechazando booleanos.

    Args:
        bound: Candidato con evidencia, dict crudo o valor directo.

    Returns:
        El entero declarado, o ``None`` si no hay valor entero válido.
    """
    if isinstance(bound, FieldCandidate):
        value = cast("object", bound.value)
        return value if isinstance(value, int) and not isinstance(value, bool) else None
    if isinstance(bound, dict):
        raw = cast("dict[str, object]", bound).get("value")
        return raw if isinstance(raw, int) and not isinstance(raw, bool) else None
    if isinstance(bound, int) and not isinstance(bound, bool):
        return bound
    return None


def _extract_split_float(bound: object) -> float | None:
    """Extrae un flotante del candidato, rechazando booleanos.

    Args:
        bound: Candidato con evidencia, dict crudo o valor directo.

    Returns:
        El flotante declarado, o ``None`` si no hay valor numérico válido.
    """
    if isinstance(bound, FieldCandidate):
        value = cast("object", bound.value)
    elif isinstance(bound, dict):
        value = cast("dict[str, object]", bound).get("value")
    elif isinstance(bound, (int, float)) and not isinstance(bound, bool):
        return float(bound)
    else:
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return None


def _quote_numbers(quote: str) -> list[float]:
    """Extrae los literales numéricos de una cita del brief.

    Args:
        quote: Cita textual del brief.

    Returns:
        Los números mencionados, en orden de aparición.
    """
    return [float(match.group(0)) for match in _SPLIT_NUMBER_RE.finditer(quote)]


def _quote_mentions_number(quote: str, value: float) -> bool:
    """Indica si la cita menciona un valor numérico dentro de la tolerancia.

    Args:
        quote: Cita textual del brief.
        value: Valor declarado por el candidato.

    Returns:
        True si algún número de la cita coincide con el valor.
    """
    return any(
        math.isclose(number, value, abs_tol=_SPLIT_NUMERIC_TOLERANCE)
        for number in _quote_numbers(quote)
    )


def _check_split_source_quote(bound: object, value: str, issues: list[ResolutionIssue]) -> bool:
    """Exige que la fuente aparezca en su propia cita, no en la de otro campo.

    Args:
        bound: Candidato del campo (o ``None`` si no se declaró).
        value: Valor resuelto de la fuente.
        issues: Hallazgos del resolutor; se amplía en sitio.

    Returns:
        True si el campo no se declaró o su valor está en su propia cita.
    """
    if bound is None:
        return True
    quote = _extract_candidate_quote(bound)
    if quote is None or value not in quote:
        issues.append(
            ResolutionIssue(
                code=IssueCode.INVALID_CONTRACT,
                field="split_screen",
                detail=(f"la fuente {value!r} no aparece en su propia cita: {quote!r}"),
            )
        )
        return False
    return True


def _check_split_number_quote(bound: object, value: float, issues: list[ResolutionIssue]) -> bool:
    """Exige correspondencia numérica entre el valor y su propia cita.

    Args:
        bound: Candidato del campo (o ``None`` si no se declaró).
        value: Valor numérico resuelto.
        issues: Hallazgos del resolutor; se amplía en sitio.

    Returns:
        True si el campo no se declaró o su cita menciona el valor.
    """
    if bound is None:
        return True
    quote = _extract_candidate_quote(bound)
    if quote is None or not _quote_mentions_number(quote, value):
        issues.append(
            ResolutionIssue(
                code=IssueCode.INVALID_CONTRACT,
                field="split_screen",
                detail=(f"el valor {value} no se corresponde con su propia cita: {quote!r}"),
            )
        )
        return False
    return True


def _check_split_ratio_quote(bound: object, value: float, issues: list[ResolutionIssue]) -> bool:
    """Exige que la cita mencione el ratio como fracción o como porcentaje.

    Args:
        bound: Candidato del campo (o ``None`` si no se declaró).
        value: Fracción resuelta del panel superior (p. ej. ``0.5``).
        issues: Hallazgos del resolutor; se amplía en sitio.

    Returns:
        True si el campo no se declaró o su cita menciona la fracción o su
        porcentaje equivalente (p. ej. ``50`` para ``0.5``).
    """
    if bound is None:
        return True
    quote = _extract_candidate_quote(bound)
    if quote is None or not (
        _quote_mentions_number(quote, value)
        or _quote_mentions_number(quote, value * _SPLIT_PERCENT_SCALE)
    ):
        issues.append(
            ResolutionIssue(
                code=IssueCode.INVALID_CONTRACT,
                field="split_screen",
                detail=(f"el ratio {value} no se corresponde con su propia cita: {quote!r}"),
            )
        )
        return False
    return True


def _check_split_own_quotes(
    split_draft: SplitScreenDraft,
    config: SplitScreenConfig,
    issues: list[ResolutionIssue],
) -> bool:
    """Valida cada campo declarado contra su propia cita, nunca contra la unión.

    Args:
        split_draft: Borrador con los candidatos y sus citas.
        config: Configuración ya construida con los valores resueltos.
        issues: Hallazgos del resolutor; se amplía en sitio.

    Returns:
        True si cada campo declarado se corresponde con su propia cita.
    """
    valid = True
    if not _check_split_source_quote(split_draft.top_source, config.top_source, issues):
        valid = False
    if not _check_split_source_quote(split_draft.bottom_source, config.bottom_source, issues):
        valid = False
    if not _check_split_number_quote(split_draft.gap, float(config.gap), issues):
        valid = False
    if not _check_split_ratio_quote(split_draft.panel_ratio, config.panel_ratio, issues):
        valid = False
    if not _check_split_number_quote(split_draft.width, float(config.width), issues):
        valid = False
    if not _check_split_number_quote(split_draft.height, float(config.height), issues):
        valid = False
    return valid


def _split_config_kwargs(
    split_draft: SplitScreenDraft,
    issues: list[ResolutionIssue],
) -> dict[str, object] | None:
    """Extrae las fuentes y los numéricos opcionales del split o falla en cerrado.

    Args:
        split_draft: Borrador con los candidatos y sus citas.
        issues: Hallazgos del resolutor; se amplía en sitio.

    Returns:
        Los argumentos para ``SplitScreenConfig``, o ``None`` si faltan las
        fuentes o un campo declarado no trae un valor válido.
    """
    top_source = _extract_candidate_str(split_draft.top_source)
    bottom_source = _extract_candidate_str(split_draft.bottom_source)
    if top_source is None or bottom_source is None:
        issues.append(
            ResolutionIssue(
                code=IssueCode.MISSING_REQUIRED,
                field="split_screen",
                detail="split_screen exige top_source y bottom_source con valor textual",
            )
        )
        return None
    kwargs: dict[str, object] = {"top_source": top_source, "bottom_source": bottom_source}
    optionals: list[tuple[str, object, bool]] = [
        ("gap", split_draft.gap, True),
        ("panel_ratio", split_draft.panel_ratio, False),
        ("width", split_draft.width, True),
        ("height", split_draft.height, True),
    ]
    for name, bound, is_int in optionals:
        if bound is None:
            continue
        value = _extract_split_int(bound) if is_int else _extract_split_float(bound)
        if value is None:
            issues.append(
                ResolutionIssue(
                    code=IssueCode.INVALID_CONTRACT,
                    field="split_screen",
                    detail=f"campo '{name}' declarado sin valor numérico válido",
                )
            )
            return None
        kwargs[name] = value
    return kwargs


def _resolve_split_screen(
    draft: ContractDraft,
    brief_text: str | None,
    issues: list[ResolutionIssue],
    *,
    mode: Mode,
) -> SplitScreenConfig | None:
    """Resuelve el split_screen con procedencia estricta T6.

    Cada campo declarado exige confianza válida y una cita textual presente
    en el brief (los alias sueltos ``citation``/``quote`` en la raíz se
    rechazan); cada valor se valida contra su propia cita (las fuentes como
    subcadena, los numéricos por correspondencia, el ratio también como
    porcentaje) y cualquier fallo es bloqueante (MANUAL_REVIEW).

    Fail-closed por modo: solo ``given_clips`` compone dos paneles. Cualquier
    otro modo genera un único panel 9:16 que el gate ``layout.geometry`` no
    puede distinguir de un split real, así que el split declarado fuera de
    ``given_clips`` se rechaza como contrato inválido antes de renderizar.

    Args:
        draft: Salida cruda del extractor, con evidencia por campo.
        brief_text: Texto crudo del brief original, o ``None``.
        issues: Hallazgos del resolutor; se amplía en sitio.
        mode: Modo de campaña ya resuelto; solo ``given_clips`` admite split.

    Returns:
        La configuración resuelta, o ``None`` sin split declarado, con modo
        sin soporte o ante cualquier fallo de procedencia o validación.
    """
    split_draft = draft.split_screen
    if split_draft is None:
        return None
    fields: list[tuple[str, object]] = [
        ("top_source", split_draft.top_source),
        ("bottom_source", split_draft.bottom_source),
        ("gap", split_draft.gap),
        ("panel_ratio", split_draft.panel_ratio),
        ("width", split_draft.width),
        ("height", split_draft.height),
    ]
    has_error = False
    if mode is not Mode.GIVEN_CLIPS:
        issues.append(
            ResolutionIssue(
                code=IssueCode.INVALID_CONTRACT,
                field="split_screen",
                detail=(
                    "split_screen solo se compone en el modo given_clips; "
                    f"el modo {mode.value} genera un único panel que el gate "
                    "layout.geometry no puede distinguir de un split real"
                ),
            )
        )
        has_error = True
    for _, bound in fields:
        if bound is not None and not _check_candidate_provenance(
            bound, brief_text, issues, field="split_screen"
        ):
            has_error = True
    if has_error:
        return None
    kwargs = _split_config_kwargs(split_draft, issues)
    if kwargs is None:
        return None
    try:
        config = SplitScreenConfig.model_validate(kwargs)
    except ValidationError as err:
        issues.append(
            ResolutionIssue(
                code=IssueCode.INVALID_CONTRACT,
                field="split_screen",
                detail=str(err),
            )
        )
        return None
    if not _check_split_own_quotes(split_draft, config, issues):
        return None
    return config


@dataclass(frozen=True, slots=True)
class _ResolvedSecondary:
    official_audio: OfficialAudio | None
    unmapped: tuple[UnmappedRule, ...]
    segments: tuple[Segment, ...]
    geo_target: GeoTarget | None
    timestamp_ranges: tuple[TimestampRange, ...]
    lyric_video: LyricConfig | None
    split_screen: SplitScreenConfig | None


def _resolve_secondary_fields(
    draft: ContractDraft,
    platforms: dict[Platform, PlatformRules],
    mode: Mode,
    *,
    format_: Format,
    brief_text: str | None,
    registry: AssetRegistry,
    issues: list[ResolutionIssue],
) -> _ResolvedSecondary:
    return _ResolvedSecondary(
        official_audio=_resolve_official_audio(draft, platforms, issues),
        unmapped=_resolve_unmapped(draft, issues),
        segments=_resolve_segments(draft, mode, issues),
        geo_target=_resolve_geo_target(draft, issues),
        timestamp_ranges=_resolve_timestamp_ranges(draft, brief_text, issues),
        lyric_video=_resolve_lyric_video(draft, format_, brief_text, registry, issues),
        split_screen=_resolve_split_screen(draft, brief_text, issues, mode=mode),
    )


def _build_contract(
    draft: ContractDraft,
    registry: AssetRegistry,
    *,
    mode: Mode,
    format_: Format,
    issues: list[ResolutionIssue],
    brief_text: str | None = None,
) -> Contract | None:
    campaign_id = _required_value(draft.campaign_id, "campaign_id", issues)
    platforms = _resolve_platforms(draft, issues)
    languages = _resolve_languages(draft, issues)
    watermark = _resolve_watermark(draft)
    assets = _resolve_assets(draft, registry, issues)
    spelling_locks = _values(draft.spelling_locks)
    prohibitions = _values(draft.prohibitions)
    brand_safety_required, brand_safety_citation, prohibitions = _resolve_brand_safety(
        draft, brief_text, prohibitions
    )
    hook_keyword = _value(draft.hook_keyword)
    secondary = _resolve_secondary_fields(
        draft,
        platforms,
        mode,
        format_=format_,
        brief_text=brief_text,
        registry=registry,
        issues=issues,
    )
    if secondary.lyric_video is not None and secondary.lyric_video.lrc_asset_id:
        lrc_id = secondary.lyric_video.lrc_asset_id
        if not any(a.asset_id == lrc_id for a in assets.required):
            try:
                ref = registry.get(lrc_id)
                assets.required.append(ref)
            except AssetNotFoundError:
                pass
    rules = _resolve_rules(
        draft,
        context=_RuleContext(
            platforms=platforms,
            format_=format_,
            global_restrictions=GlobalRestrictions(
                watermark_required=watermark.required,
                watermark_visible_full_video=watermark.visible_full_video,
                has_required_assets=bool(assets.required),
                spelling_locks=tuple(spelling_locks),
                prohibitions=tuple(prohibitions),
                audio_policy=_value(draft.audio_policy),
                hook_keyword=hook_keyword,
                brand_safety_required=brand_safety_required,
            ),
            brief_text=brief_text,
            split_screen=secondary.split_screen,
        ),
        unmapped=secondary.unmapped,
        issues=issues,
    )

    if _has_blocking(issues) or campaign_id is None or languages is None:
        return None

    return Contract(
        campaign_id=campaign_id,
        format=format_,
        mode=mode,
        platforms=platforms,
        languages=languages,
        official_audio=secondary.official_audio,
        audio_policy=_value(draft.audio_policy),
        watermark=watermark,
        spelling_locks=spelling_locks,
        prohibitions=prohibitions,
        hook_keyword=hook_keyword,
        brand_safety_required=brand_safety_required,
        brand_safety_citation=brand_safety_citation,
        unmapped=secondary.unmapped,
        timestamp_ranges=secondary.timestamp_ranges,
        lyric_video=secondary.lyric_video,
        split_screen=secondary.split_screen,
        rules=rules,
        assets=assets,
        segments=secondary.segments,
        geo_target=secondary.geo_target,
        min_views_for_payout=MinViewsForPayout(value=_value(draft.min_views_for_payout)),
        analytics_proof_required=AnalyticsProofRequired(
            value=bool(_value(draft.analytics_proof_required))
        ),
    )


def _resolve_unmapped(
    draft: ContractDraft,
    issues: list[ResolutionIssue],
) -> tuple[UnmappedRule, ...]:
    resolved: list[UnmappedRule] = []
    for index, item in enumerate(draft.unmapped):
        rule = _value(item.rule)
        quote = _value(item.quote)
        if rule and quote:
            resolved.append(UnmappedRule(rule=rule, quote=quote))
        elif rule or quote:
            issues.append(
                ResolutionIssue(
                    code=IssueCode.MISSING_REQUIRED,
                    field=f"unmapped[{index}]",
                    detail="unmapped exige rule y quote",
                )
            )
    return tuple(resolved)


def _value[T](candidate: FieldCandidate[T] | None) -> T | None:
    return None if candidate is None else candidate.value


def _values[T](candidate: FieldCandidate[list[T]] | None) -> list[T]:
    value = _value(candidate)
    return [] if value is None else list(value)


def _required_value[T](
    candidate: FieldCandidate[T] | None,
    field: str,
    issues: list[ResolutionIssue],
) -> T | None:
    if candidate is not None and candidate.value is not None:
        return candidate.value
    code = (
        IssueCode.CONFLICT
        if candidate is not None and candidate.confidence is Confidence.CONFLICT
        else IssueCode.MISSING_REQUIRED
    )
    detail = (
        "candidato en conflicto sin resolver"
        if code is IssueCode.CONFLICT
        else "campo obligatorio sin valor resuelto en el draft"
    )
    issues.append(ResolutionIssue(code=code, field=field, detail=detail))
    return None


def _is_conflict(candidate: _ConfidenceCarrier | None) -> bool:
    return candidate is not None and candidate.confidence is Confidence.CONFLICT


def _note_conflict(
    candidate: _ConfidenceCarrier | None,
    field: str,
    issues: list[ResolutionIssue],
) -> None:
    if _is_conflict(candidate):
        issues.append(
            ResolutionIssue(
                code=IssueCode.CONFLICT,
                field=field,
                detail="candidato en conflicto sin resolver",
            )
        )


def _collect_conflicts(draft: ContractDraft) -> list[ResolutionIssue]:
    issues: list[ResolutionIssue] = []
    _collect_top_level_conflicts(draft, issues)
    _collect_platform_conflicts(draft, issues)
    _collect_global_conflicts(draft, issues)
    _collect_asset_conflicts(draft, issues)
    _collect_unmapped_conflicts(draft, issues)
    return issues


def _collect_top_level_conflicts(
    draft: ContractDraft,
    issues: list[ResolutionIssue],
) -> None:
    for field, draft_candidate in (
        ("spelling_locks", draft.spelling_locks),
        ("prohibitions", draft.prohibitions),
        ("audio_policy", draft.audio_policy),
        ("hook_keyword", draft.hook_keyword),
        ("hook_window_seconds", draft.hook_window_seconds),
        ("brand_safety", draft.brand_safety),
        ("brand_safety_required", draft.brand_safety_required),
        ("brand_safety_citation", draft.brand_safety_citation),
        ("min_views_for_payout", draft.min_views_for_payout),
        ("analytics_proof_required", draft.analytics_proof_required),
    ):
        _note_conflict(draft_candidate, field, issues)


def _collect_unmapped_conflicts(
    draft: ContractDraft,
    issues: list[ResolutionIssue],
) -> None:
    for index, item in enumerate(draft.unmapped):
        _note_conflict(item.rule, f"unmapped[{index}].rule", issues)
        _note_conflict(item.quote, f"unmapped[{index}].quote", issues)


def _collect_platform_conflicts(
    draft: ContractDraft,
    issues: list[ResolutionIssue],
) -> None:
    for platform, platform_draft in draft.platforms.items():
        prefix = f"platforms.{platform}"
        _note_conflict(platform_draft.audio_rule, f"{prefix}.audio_rule", issues)
        _note_conflict(platform_draft.required_hashtags, f"{prefix}.required_hashtags", issues)
        _note_conflict(platform_draft.required_mentions, f"{prefix}.required_mentions", issues)
        if platform_draft.duration is not None:
            _note_conflict(platform_draft.duration.min_s, f"{prefix}.duration.min_s", issues)
            _note_conflict(platform_draft.duration.max_s, f"{prefix}.duration.max_s", issues)
        if platform_draft.caption_rules is not None:
            caption = platform_draft.caption_rules
            _note_conflict(caption.must_mention, f"{prefix}.caption_rules.must_mention", issues)
            _note_conflict(caption.first_line, f"{prefix}.caption_rules.first_line", issues)
            _note_conflict(caption.forbidden, f"{prefix}.caption_rules.forbidden", issues)
        if platform_draft.attribution is not None:
            _note_conflict(platform_draft.attribution.type, f"{prefix}.attribution.type", issues)
            _note_conflict(platform_draft.attribution.value, f"{prefix}.attribution.value", issues)
        if platform_draft.link_rules is not None:
            _note_conflict(
                platform_draft.link_rules.link_in_bio,
                f"{prefix}.link_rules.link_in_bio",
                issues,
            )


def _collect_global_conflicts(
    draft: ContractDraft,
    issues: list[ResolutionIssue],
) -> None:
    if draft.languages is not None:
        _note_conflict(draft.languages.subtitles, "languages.subtitles", issues)
        _note_conflict(draft.languages.voice, "languages.voice", issues)
        _note_conflict(draft.languages.language, "languages.language", issues)
    if draft.official_audio is not None:
        _note_conflict(draft.official_audio.tiktok_url, "official_audio.tiktok_url", issues)
        _note_conflict(draft.official_audio.instagram_url, "official_audio.instagram_url", issues)
    if draft.watermark is not None:
        _note_conflict(draft.watermark.required, "watermark.required", issues)
        _note_conflict(draft.watermark.asset_id, "watermark.asset_id", issues)
        _note_conflict(draft.watermark.visible_full_video, "watermark.visible_full_video", issues)
        _note_conflict(draft.watermark.position, "watermark.position", issues)
        _note_conflict(draft.watermark.scale_ratio, "watermark.scale_ratio", issues)
        _note_conflict(draft.watermark.opacity, "watermark.opacity", issues)
        _note_conflict(draft.watermark.min_width_ratio, "watermark.min_width_ratio", issues)
    if draft.rules is not None:
        _note_conflict(draft.rules.hard, "rules.hard", issues)
        _note_conflict(draft.rules.recommended, "rules.recommended", issues)
        _note_conflict(draft.rules.manual_review, "rules.manual_review", issues)
    if draft.geo_target is not None:
        _note_conflict(draft.geo_target.country, "geo_target.country", issues)
        _note_conflict(draft.geo_target.min_pct, "geo_target.min_pct", issues)


def _collect_asset_conflicts(
    draft: ContractDraft,
    issues: list[ResolutionIssue],
) -> None:
    if draft.assets is None:
        return
    for index, asset in enumerate(draft.assets.required):
        _collect_single_asset_conflicts(asset, f"assets.required[{index}]", issues)
    for index, asset in enumerate(draft.assets.optional):
        _collect_single_asset_conflicts(asset, f"assets.optional[{index}]", issues)


def _collect_single_asset_conflicts(
    draft: AssetDraft,
    field: str,
    issues: list[ResolutionIssue],
) -> None:
    for name, draft_candidate in (
        ("asset_id", draft.asset_id),
        ("kind", draft.kind),
        ("uri", draft.uri),
        ("origin", draft.origin),
        ("license", draft.license),
    ):
        _note_conflict(draft_candidate, f"{field}.{name}", issues)


def _resolve_platforms(
    draft: ContractDraft,
    issues: list[ResolutionIssue],
) -> dict[Platform, PlatformRules]:
    if not draft.platforms:
        issues.append(
            ResolutionIssue(
                code=IssueCode.MISSING_REQUIRED,
                field="platforms",
                detail="el draft no declara ninguna plataforma",
            )
        )
        return {}
    return {
        platform: _resolve_platform(platform_draft, platform, issues)
        for platform, platform_draft in draft.platforms.items()
    }


def _resolve_platform(
    draft: PlatformDraft,
    platform: Platform,
    issues: list[ResolutionIssue],
) -> PlatformRules:
    prefix = f"platforms.{platform}"
    duration = draft.duration
    caption = draft.caption_rules
    link_rules = draft.link_rules
    return PlatformRules(
        duration=DurationRange(
            min_s=_value(duration.min_s) if duration is not None else None,
            max_s=_value(duration.max_s) if duration is not None else None,
        ),
        caption_rules=CaptionRules(
            must_mention=_values(caption.must_mention) if caption is not None else [],
            first_line=_value(caption.first_line) if caption is not None else None,
            forbidden=_values(caption.forbidden) if caption is not None else [],
        ),
        audio_rule=_value(draft.audio_rule) or AudioRule.ANY,
        required_hashtags=_values(draft.required_hashtags),
        required_mentions=_values(draft.required_mentions),
        attribution=_resolve_attribution(draft.attribution, prefix, issues),
        link_rules=LinkRules(
            link_in_bio=bool(_value(link_rules.link_in_bio)) if link_rules is not None else False
        ),
    )


def _resolve_attribution(
    draft: AttributionDraft | None,
    prefix: str,
    issues: list[ResolutionIssue],
) -> Attribution:
    if draft is None:
        return Attribution(type=AttributionType.NONE)
    attribution_type = _value(draft.type)
    value = _value(draft.value)
    if attribution_type is None:
        if value and not _is_conflict(draft.type):
            issues.append(
                ResolutionIssue(
                    code=IssueCode.MISSING_REQUIRED,
                    field=f"{prefix}.attribution.type",
                    detail="hay value de atribución pero falta el tipo",
                )
            )
        return Attribution(type=AttributionType.NONE)
    if attribution_type is AttributionType.NONE:
        return Attribution(type=AttributionType.NONE)
    if not value:
        if not _is_conflict(draft.value):
            issues.append(
                ResolutionIssue(
                    code=IssueCode.MISSING_REQUIRED,
                    field=f"{prefix}.attribution.value",
                    detail=f"atribución '{attribution_type}' sin value",
                )
            )
        return Attribution(type=AttributionType.NONE)
    return Attribution(type=attribution_type, value=value)


def _resolve_languages(
    draft: ContractDraft,
    issues: list[ResolutionIssue],
) -> Languages | None:
    languages = draft.languages
    if languages is None:
        issues.append(
            ResolutionIssue(
                code=IssueCode.MISSING_REQUIRED,
                field="languages",
                detail="el draft no declara idiomas",
            )
        )
        return None
    source = _required_value(languages.source, "languages.source", issues)
    caption = _required_value(languages.caption, "languages.caption", issues)
    if source is None or caption is None:
        return None
    return Languages(
        source=source,
        subtitles=_value(languages.subtitles),
        caption=caption,
        voice=_value(languages.voice),
        language=_value(languages.language),
    )


def _resolve_watermark(draft: ContractDraft) -> Watermark:
    watermark = draft.watermark
    position = _value(watermark.position) if watermark is not None else None
    return Watermark(
        required=bool(_value(watermark.required)) if watermark is not None else False,
        asset_id=_value(watermark.asset_id) if watermark is not None else None,
        visible_full_video=(
            bool(_value(watermark.visible_full_video)) if watermark is not None else False
        ),
        position=position if position is not None else WatermarkPosition.TOP_RIGHT,
        scale_ratio=_resolve_ratio(
            watermark.scale_ratio if watermark is not None else None, default=0.20
        ),
        opacity=_resolve_ratio(watermark.opacity if watermark is not None else None, default=1.0),
        min_width_ratio=_resolve_ratio(
            watermark.min_width_ratio if watermark is not None else None, default=0.05
        ),
    )


def _resolve_ratio(candidate: FieldCandidate[float] | None, *, default: float) -> float:
    value = _value(candidate)
    return default if value is None else value


def _resolve_official_audio(
    draft: ContractDraft,
    platforms: dict[Platform, PlatformRules],
    issues: list[ResolutionIssue],
) -> OfficialAudio | None:
    draft_audio = draft.official_audio
    tiktok_url = _value(draft_audio.tiktok_url) if draft_audio is not None else None
    instagram_url = _value(draft_audio.instagram_url) if draft_audio is not None else None
    if tiktok_url or instagram_url:
        return OfficialAudio(tiktok_url=tiktok_url, instagram_url=instagram_url)
    needs_audio = any(
        rules.audio_rule is AudioRule.OFFICIAL_REQUIRED for rules in platforms.values()
    )
    if needs_audio:
        issues.append(
            ResolutionIssue(
                code=IssueCode.MISSING_REQUIRED,
                field="official_audio",
                detail="audio_rule 'official_required' sin URLs de audio oficial",
            )
        )
    return None


def _resolve_assets(
    draft: ContractDraft,
    registry: AssetRegistry,
    issues: list[ResolutionIssue],
) -> AssetBundle:
    assets = draft.assets
    if assets is None:
        return AssetBundle()
    required = [
        ref
        for index, asset_draft in enumerate(assets.required)
        if (
            ref := _resolve_asset(
                asset_draft,
                registry,
                f"assets.required[{index}]",
                issues,
                dropped_code=None,
            )
        )
        is not None
    ]
    optional = [
        ref
        for index, asset_draft in enumerate(assets.optional)
        if (
            ref := _resolve_asset(
                asset_draft,
                registry,
                f"assets.optional[{index}]",
                issues,
                dropped_code=IssueCode.OPTIONAL_ASSET_DROPPED,
            )
        )
        is not None
    ]
    return AssetBundle(required=required, optional=optional)


def _resolve_asset(
    draft: AssetDraft,
    registry: AssetRegistry,
    field: str,
    issues: list[ResolutionIssue],
    *,
    dropped_code: IssueCode | None,
) -> AssetRef | None:
    asset_id = _value(draft.asset_id) or None
    kind = _value(draft.kind) or None
    uri = _value(draft.uri) or None
    origin = _value(draft.origin) or None
    if asset_id is None or kind is None or uri is None or origin is None:
        missing = [
            name
            for name, value, draft_candidate in (
                ("asset_id", asset_id, draft.asset_id),
                ("kind", kind, draft.kind),
                ("uri", uri, draft.uri),
                ("origin", origin, draft.origin),
            )
            if value is None and not _is_conflict(draft_candidate)
        ]
        if missing:
            issues.append(
                ResolutionIssue(
                    code=dropped_code or IssueCode.MISSING_REQUIRED,
                    field=field,
                    detail=f"asset sin campos: {', '.join(missing)}",
                )
            )
        return None
    if asset_id in registry.assets:
        existing = registry.get(asset_id)
        try:
            canonical = registry.canonical_uri(uri)
        except (AssetError, OSError, ValidationError) as error:
            issues.append(
                ResolutionIssue(
                    code=dropped_code or IssueCode.UNRESOLVED_ASSET,
                    field=field,
                    detail=str(error),
                )
            )
            return None
        if existing.uri == canonical:
            return existing
        issues.append(
            ResolutionIssue(
                code=dropped_code or IssueCode.UNRESOLVED_ASSET,
                field=field,
                detail=f"asset_id '{asset_id}' ya registrado con otra uri",
            )
        )
        return None
    try:
        return registry.register(
            asset_id=asset_id,
            kind=kind,
            uri=uri,
            origin=origin,
            license=_value(draft.license),
        )
    except (AssetError, OSError, ValidationError) as error:
        issues.append(
            ResolutionIssue(
                code=dropped_code or IssueCode.UNRESOLVED_ASSET,
                field=field,
                detail=str(error),
            )
        )
        return None


def _resolve_segments(
    draft: ContractDraft,
    mode: Mode,
    issues: list[ResolutionIssue],
) -> tuple[Segment, ...]:
    segments_draft = draft.segments
    if mode is not Mode.LONG_VIDEO:
        return ()
    if segments_draft is None or not segments_draft.segments:
        issues.append(
            ResolutionIssue(
                code=IssueCode.MISSING_REQUIRED,
                field="segments",
                detail="el modo long_video exige al menos un segmento",
            )
        )
        return ()

    segments: list[Segment] = []
    for index, candidate in enumerate(segments_draft.segments):
        conflicted = [
            name
            for name, bound in (("start_s", candidate.start_s), ("end_s", candidate.end_s))
            if _is_conflict(bound)
        ]
        if conflicted:
            issues.extend(
                ResolutionIssue(
                    code=IssueCode.CONFLICT,
                    field=f"segments[{index}].{name}",
                    detail="candidato en conflicto sin resolver",
                )
                for name in conflicted
            )
            continue
        start = _value(candidate.start_s)
        end = _value(candidate.end_s)
        if start is None or end is None:
            issues.append(
                ResolutionIssue(
                    code=IssueCode.MISSING_REQUIRED,
                    field=f"segments[{index}]",
                    detail="segmento sin start_s o end_s resueltos",
                )
            )
            continue
        try:
            segments.append(Segment(start_s=start, end_s=end))
        except ValidationError as error:
            issues.append(
                ResolutionIssue(
                    code=IssueCode.INVALID_CONTRACT,
                    field=f"segments[{index}]",
                    detail=str(error),
                )
            )
    return tuple(segments)


def _base_rules(context: _RuleContext) -> list[str]:
    rules = ["artifact.integrity"]
    if context.format_ in {Format.VIDEO, Format.LYRIC_VIDEO}:
        rules.append("artifact.video_stream")
    if context.format_ is Format.LYRIC_VIDEO:
        rules.append("subtitles.spelling_lock")
    if any(platform.audio_rule is not AudioRule.ANY for platform in context.platforms.values()):
        rules.append("audio.present")
    if context.global_restrictions.has_required_assets:
        rules.append("assets.required")
    if context.global_restrictions.watermark_required:
        rules.append(
            "watermark.full_video"
            if context.global_restrictions.watermark_visible_full_video
            else "watermark.present"
        )
    if context.split_screen is not None:
        rules.append("layout.geometry")
    return rules


def _resolve_rules(
    draft: ContractDraft,
    *,
    context: _RuleContext,
    unmapped: tuple[UnmappedRule, ...] = (),
    issues: list[ResolutionIssue],
) -> RuleSet:
    draft_rules = draft.rules
    hard = _values(draft_rules.hard) if draft_rules is not None else []
    recommended = _values(draft_rules.recommended) if draft_rules is not None else []
    manual_review = _values(draft_rules.manual_review) if draft_rules is not None else []
    classified = {*hard, *recommended, *manual_review}
    base_rules = _base_rules(context)
    for rule_id in base_rules:
        if rule_id not in hard:
            hard.append(rule_id)
        if rule_id in recommended:
            recommended.remove(rule_id)
            issues.append(
                ResolutionIssue(
                    code=IssueCode.RULE_DEFAULTED,
                    field=f"rules.{rule_id}",
                    detail="regla obligatoria reclasificada como hard (fail-closed)",
                )
            )
        if rule_id in manual_review:
            manual_review.remove(rule_id)
            issues.append(
                ResolutionIssue(
                    code=IssueCode.RULE_DEFAULTED,
                    field=f"rules.{rule_id}",
                    detail="regla obligatoria reclasificada como hard (fail-closed)",
                )
            )
    _force_policy_review_when_official_without_policy(
        context,
        classified=classified,
        manual_review=manual_review,
        issues=issues,
    )
    _force_policy_review_when_brief_mentions_official_sound(
        context.brief_text,
        audio_policy=context.global_restrictions.audio_policy,
        classified=classified,
        manual_review=manual_review,
        issues=issues,
    )
    _apply_platform_restrictions(
        context,
        classified=classified,
        hard=hard,
        manual_review=manual_review,
        issues=issues,
    )
    if (
        context.global_restrictions.brand_safety_required
        or _brief_indicates_brand_safety(
            context.brief_text, context.global_restrictions.prohibitions
        )
        or _value(draft.brand_safety) is True
        or _value(draft.brand_safety_required) is True
    ) and "brand.safety" not in classified:
        classified.add("brand.safety")
        hard.append("brand.safety")
        issues.append(
            ResolutionIssue(
                code=IssueCode.RULE_DEFAULTED,
                field="rules.brand.safety",
                detail="el brief indica brand safety; se clasifica como hard (fail-closed)",
            )
        )
    for entry in unmapped:
        if entry.rule not in classified:
            classified.add(entry.rule)
            manual_review.append(entry.rule)
            issues.append(
                ResolutionIssue(
                    code=IssueCode.RULE_DEFAULTED,
                    field=f"rules.{entry.rule}",
                    detail="requisito sin validador clasificado como manual_review",
                )
            )
    return RuleSet(hard=hard, recommended=recommended, manual_review=manual_review)


_MANUAL_REVIEW_DEFAULTS: frozenset[str] = frozenset(
    {
        "audio.official_track",
        "audio.official_selection",
        "audio.rule",
        "audio.own_clip",
        "audio.no_trending",
        "audio.policy",
        "attribution.required",
        "attribution.present",
        "link.in_bio",
        "link_rules.link_in_bio",
    }
)


def _force_policy_review_when_official_without_policy(
    context: _RuleContext,
    *,
    classified: set[str],
    manual_review: list[str],
    issues: list[ResolutionIssue],
) -> None:
    """Clasifica ``audio.policy`` como manual_review si falta la política (fail-closed).

    Cuando alguna plataforma exige sonido oficial (``official_required``) pero
    el draft no resolvió ``audio_policy``, el contrato no puede pasar en
    silencio sin política ni mute: la regla ``audio.policy`` exige revisión
    humana en el gate.

    Args:
        context: Plataformas y restricciones globales ya resueltas.
        classified: Reglas ya clasificadas; se amplía en sitio.
        manual_review: Reglas de revisión manual; se amplía en sitio.
        issues: Hallazgos del resolutor; se amplía en sitio.
    """
    needs_official = any(
        rules.audio_rule is AudioRule.OFFICIAL_REQUIRED for rules in context.platforms.values()
    )
    if not needs_official or context.global_restrictions.audio_policy is not None:
        return
    if "audio.policy" in classified:
        return
    classified.add("audio.policy")
    manual_review.append("audio.policy")
    issues.append(
        ResolutionIssue(
            code=IssueCode.RULE_DEFAULTED,
            field="rules.audio.policy",
            detail=(
                "audio_rule 'official_required' sin audio_policy resuelta; "
                "se clasifica como manual_review (fail-closed)"
            ),
        )
    )


_OFFICIAL_SOUND_MENTIONS: tuple[str, ...] = (
    "official sound",
    "official audio",
    "sonido oficial",
    "audio oficial",
)

_LYRIC_VIDEO_MENTIONS: tuple[str, ...] = (
    "lyric video",
    "video con letra",
    "letra official",
    "letra oficial",
    ".lrc",
)


def _normalize_mention_text(text: str) -> str:
    """Normaliza el brief para detectar menciones de sonido oficial.

    Descompone con NFKD, elimina diacríticos y pliega a minúsculas: así
    "SONIDO OFICIAL" u "Official Sound" coinciden con las menciones
    canónicas sin falsos negativos por mayúsculas o tildes.

    Args:
        text: Texto crudo del brief.

    Returns:
        El texto normalizado para búsqueda de subcadenas.
    """
    decomposed = unicodedata.normalize("NFKD", text)
    stripped = "".join(char for char in decomposed if not unicodedata.combining(char))
    return stripped.casefold()


def _brief_mentions_official_sound(brief_text: str | None) -> bool:
    """Indica si el brief menciona textualmente el sonido oficial.

    Args:
        brief_text: Texto crudo del brief original, o ``None`` si no se
            proporcionó.

    Returns:
        True si alguna mención canónica aparece en el texto normalizado.
    """
    if not brief_text:
        return False
    normalized = _normalize_mention_text(brief_text)
    return any(mention in normalized for mention in _OFFICIAL_SOUND_MENTIONS)


def _brief_mentions_lyric_video(brief_text: str | None) -> bool:
    """Indica si el brief menciona explícitamente formato lyric video.

    Args:
        brief_text: Texto crudo del brief original, o ``None``.

    Returns:
        True si alguna mención canónica aparece en el texto normalizado.
    """
    if not brief_text:
        return False
    normalized = _normalize_mention_text(brief_text)
    return any(mention in normalized for mention in _LYRIC_VIDEO_MENTIONS)


def _brief_indicates_brand_safety(
    brief_text: str | None,
    prohibitions: Sequence[str],
) -> bool:
    combined: list[str] = list(prohibitions)
    if brief_text:
        combined.append(brief_text)
    if not combined:
        return False
    haystack = normalize_text(" ".join(combined))
    return any(normalize_text(mention) in haystack for mention in BRAND_SAFETY_MENTIONS)


def _force_policy_review_when_brief_mentions_official_sound(
    brief_text: str | None,
    *,
    audio_policy: AudioPolicy | None,
    classified: set[str],
    manual_review: list[str],
    issues: list[ResolutionIssue],
) -> None:
    """Clasifica ``audio.policy`` como manual_review si el brief lo menciona (fail-closed).

    Aunque ninguna plataforma declare ``official_required``, un brief que
    menciona textualmente el sonido oficial ("official sound", "sonido
    oficial", "audio oficial", "official audio") sin ``audio_policy``
    resuelta no puede pasar en silencio: la regla ``audio.policy`` exige
    revisión humana en el gate.

    Args:
        brief_text: Texto crudo del brief original, o ``None``.
        audio_policy: Política de audio ya resuelta del draft.
        classified: Reglas ya clasificadas; se amplía en sitio.
        manual_review: Reglas de revisión manual; se amplía en sitio.
        issues: Hallazgos del resolutor; se amplía en sitio.
    """
    if audio_policy is not None:
        return
    if "audio.policy" in classified:
        return
    if not _brief_mentions_official_sound(brief_text):
        return
    classified.add("audio.policy")
    manual_review.append("audio.policy")
    issues.append(
        ResolutionIssue(
            code=IssueCode.RULE_DEFAULTED,
            field="rules.audio.policy",
            detail=(
                "el brief menciona el sonido oficial sin audio_policy resuelta; "
                "se clasifica como manual_review (fail-closed)"
            ),
        )
    )


def _apply_platform_restrictions(
    context: _RuleContext,
    *,
    classified: set[str],
    hard: list[str],
    manual_review: list[str],
    issues: list[ResolutionIssue],
) -> None:
    for platform, platform_rules in context.platforms.items():
        for rule_id in active_restriction_rules(platform_rules, context.global_restrictions):
            if rule_id in classified:
                continue
            classified.add(rule_id)
            if rule_id in _MANUAL_REVIEW_DEFAULTS:
                if rule_id not in manual_review:
                    manual_review.append(rule_id)
                detail = (
                    f"{platform}: restricción activa sin clasificar en el brief; "
                    "se clasifica como manual_review"
                )
            else:
                if rule_id not in hard:
                    hard.append(rule_id)
                detail = (
                    f"{platform}: restricción activa sin clasificar en el brief; "
                    "se clasifica como hard (fail-closed)"
                )
            issues.append(
                ResolutionIssue(
                    code=IssueCode.RULE_DEFAULTED,
                    field=f"rules.{rule_id}",
                    detail=detail,
                )
            )


def _resolve_geo_target(
    draft: ContractDraft,
    issues: list[ResolutionIssue],
) -> GeoTarget | None:
    geo = draft.geo_target
    if geo is None:
        return None
    country = _value(geo.country)
    min_pct = _value(geo.min_pct)
    if country is None and min_pct is None:
        return None
    missing = [
        name
        for name, value, draft_candidate in (
            ("country", country, geo.country),
            ("min_pct", min_pct, geo.min_pct),
        )
        if value is None and not _is_conflict(draft_candidate)
    ]
    if missing:
        issues.append(
            ResolutionIssue(
                code=IssueCode.MISSING_REQUIRED,
                field="geo_target",
                detail=f"geo_target incompleto: faltan {', '.join(missing)}",
            )
        )
        return None
    if country is None or min_pct is None:
        return None
    return GeoTarget(country=country, min_pct=min_pct)


def _has_blocking(issues: Sequence[ResolutionIssue]) -> bool:
    return any(issue.code in _BLOCKING_CODES for issue in issues)
