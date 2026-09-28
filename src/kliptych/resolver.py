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

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol, cast

from pydantic import BaseModel, ValidationError

from kliptych.assets import AssetError, AssetRegistry
from kliptych.contract import (
    AnalyticsProofRequired,
    AssetBundle,
    AssetDraft,
    AssetRef,
    Attribution,
    AttributionDraft,
    AttributionType,
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
    MinViewsForPayout,
    Mode,
    OfficialAudio,
    Platform,
    PlatformDraft,
    PlatformRules,
    RuleSet,
    Segment,
    SourceEvidence,
    Watermark,
    active_restriction_rules,
)
from kliptych.contract.base import ContractBase


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
    format_ = _required_value(draft.format, "format", issues)
    if mode is None or format_ is None:
        status = (
            ResolutionStatus.MANUAL_REVIEW
            if any(issue.code is IssueCode.CONFLICT for issue in issues)
            else ResolutionStatus.NEW_ARCHETYPE
        )
        return ResolutionResult(status=status, issues=tuple(issues))

    try:
        contract = _build_contract(draft, registry, mode=mode, format_=format_, issues=issues)
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


def _build_contract(
    draft: ContractDraft,
    registry: AssetRegistry,
    *,
    mode: Mode,
    format_: Format,
    issues: list[ResolutionIssue],
) -> Contract | None:
    campaign_id = _required_value(draft.campaign_id, "campaign_id", issues)
    platforms = _resolve_platforms(draft, issues)
    languages = _resolve_languages(draft, issues)
    watermark = _resolve_watermark(draft)
    assets = _resolve_assets(draft, registry, issues)
    spelling_locks = _values(draft.spelling_locks)
    prohibitions = _values(draft.prohibitions)
    official_audio = _resolve_official_audio(draft, platforms, issues)
    segments = _resolve_segments(draft, mode, issues)
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
            ),
        ),
        issues=issues,
    )
    geo_target = _resolve_geo_target(draft, issues)

    if _has_blocking(issues) or campaign_id is None or languages is None:
        return None

    return Contract(
        campaign_id=campaign_id,
        format=format_,
        mode=mode,
        platforms=platforms,
        languages=languages,
        official_audio=official_audio,
        watermark=watermark,
        spelling_locks=spelling_locks,
        prohibitions=prohibitions,
        rules=rules,
        assets=assets,
        segments=segments,
        geo_target=geo_target,
        min_views_for_payout=MinViewsForPayout(value=_value(draft.min_views_for_payout)),
        analytics_proof_required=AnalyticsProofRequired(
            value=bool(_value(draft.analytics_proof_required))
        ),
    )


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
    return issues


def _collect_top_level_conflicts(
    draft: ContractDraft,
    issues: list[ResolutionIssue],
) -> None:
    for field, draft_candidate in (
        ("spelling_locks", draft.spelling_locks),
        ("prohibitions", draft.prohibitions),
        ("min_views_for_payout", draft.min_views_for_payout),
        ("analytics_proof_required", draft.analytics_proof_required),
    ):
        _note_conflict(draft_candidate, field, issues)


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
    return Watermark(
        required=bool(_value(watermark.required)) if watermark is not None else False,
        asset_id=_value(watermark.asset_id) if watermark is not None else None,
        visible_full_video=(
            bool(_value(watermark.visible_full_video)) if watermark is not None else False
        ),
    )


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
    if context.format_ is Format.VIDEO:
        rules.append("artifact.video_stream")
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
    return rules


def _resolve_rules(
    draft: ContractDraft,
    *,
    context: _RuleContext,
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
    _apply_platform_restrictions(
        context,
        classified=classified,
        hard=hard,
        manual_review=manual_review,
        issues=issues,
    )
    return RuleSet(hard=hard, recommended=recommended, manual_review=manual_review)


_MANUAL_REVIEW_DEFAULTS: frozenset[str] = frozenset(
    {
        "audio.official_track",
        "audio.official_selection",
        "audio.rule",
        "audio.own_clip",
        "audio.no_trending",
        "attribution.required",
        "attribution.present",
        "link.in_bio",
        "link_rules.link_in_bio",
    }
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
