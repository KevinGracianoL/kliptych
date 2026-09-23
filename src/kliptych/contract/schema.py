"""Contrato validado (schema v1.0): restricciones duras que obedecen pipeline y gate."""

from typing import Annotated, Literal, Self

from pydantic import AwareDatetime, Field, StringConstraints, model_validator

from kliptych.contract.base import ContractBase
from kliptych.contract.enums import AttributionType, AudioRule, Format, Mode, Platform

ENFORCEMENT = "post_publication_manual"
_RULE_ID = Annotated[str, StringConstraints(pattern=r"^[a-z][a-z0-9_]*(\.[a-z][a-z0-9_]*)+$")]
_SHA256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
_MIME = Annotated[str, StringConstraints(pattern=r"^[a-z]+/[a-z0-9][a-z0-9.+-]*$")]
_MENTION = Annotated[str, StringConstraints(pattern=r"^@\S+$")]
_HASHTAG = Annotated[str, StringConstraints(pattern=r"^#\S+$")]


class DurationRange(ContractBase):
    """Rango de duración permitido, en segundos."""

    min_s: int | None = Field(default=None, ge=0)
    max_s: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def _bounds_are_ordered(self) -> Self:
        if self.min_s is not None and self.max_s is not None and self.min_s > self.max_s:
            msg = f"min_s ({self.min_s}) no puede superar max_s ({self.max_s})"
            raise ValueError(msg)
        return self


class CaptionRules(ContractBase):
    """Reglas de caption para una plataforma."""

    must_mention: list[_MENTION] = Field(default_factory=list)
    first_line: str | None = None
    forbidden: list[str] = Field(default_factory=list)


class Attribution(ContractBase):
    """Atribución requerida (tag o URL) para una plataforma."""

    type: AttributionType
    value: str | None = None

    @model_validator(mode="after")
    def _value_matches_type(self) -> Self:
        if self.type is not AttributionType.NONE and not self.value:
            msg = f"attribution de tipo '{self.type}' exige un value"
            raise ValueError(msg)
        return self


class LinkRules(ContractBase):
    """Reglas de enlaces para una plataforma."""

    link_in_bio: bool = False


class PlatformRules(ContractBase):
    """Restricciones de una plataforma concreta."""

    duration: DurationRange = Field(default_factory=DurationRange)
    caption_rules: CaptionRules = Field(default_factory=CaptionRules)
    audio_rule: AudioRule = AudioRule.ANY
    required_hashtags: list[_HASHTAG] = Field(default_factory=list)
    required_mentions: list[_MENTION] = Field(default_factory=list)
    attribution: Attribution = Field(default_factory=lambda: Attribution(type=AttributionType.NONE))
    link_rules: LinkRules = Field(default_factory=LinkRules)


class Languages(ContractBase):
    """Idiomas del material: fuente, subtítulos, caption y voz."""

    source: str = Field(min_length=1)
    subtitles: str | None = None
    caption: str = Field(min_length=1)
    voice: str | None = None


class OfficialAudio(ContractBase):
    """URLs del audio oficial por plataforma."""

    tiktok_url: str | None = None
    instagram_url: str | None = None

    @model_validator(mode="after")
    def _at_least_one_url(self) -> Self:
        if not self.tiktok_url and not self.instagram_url:
            msg = "official_audio exige al menos una URL"
            raise ValueError(msg)
        return self


class Watermark(ContractBase):
    """Configuración de watermark de la campaña."""

    required: bool
    asset_id: str | None = None
    visible_full_video: bool


class RuleSet(ContractBase):
    """Clasificación de cada regla como hard, recommended o manual_review."""

    hard: list[_RULE_ID] = Field(default_factory=list)
    recommended: list[_RULE_ID] = Field(default_factory=list)
    manual_review: list[_RULE_ID] = Field(default_factory=list)

    @model_validator(mode="after")
    def _rules_are_classified_once(self) -> Self:
        all_ids = [*self.hard, *self.recommended, *self.manual_review]
        duplicates = sorted({rule for rule in all_ids if all_ids.count(rule) > 1})
        if duplicates:
            msg = f"reglas duplicadas en más de una categoría: {duplicates}"
            raise ValueError(msg)
        return self


class AssetRef(ContractBase):
    """Referencia a un asset resuelto, con hash y procedencia."""

    asset_id: str = Field(min_length=1)
    kind: str = Field(min_length=1)
    uri: str = Field(min_length=1)
    sha256: _SHA256
    size_bytes: int = Field(ge=0)
    mime: _MIME
    origin: str = Field(min_length=1)
    license: str | None = None
    resolved_at: AwareDatetime


class AssetBundle(ContractBase):
    """Assets requeridos y opcionales del contrato."""

    required: list[AssetRef] = Field(default_factory=list)
    optional: list[AssetRef] = Field(default_factory=list)

    @model_validator(mode="after")
    def _asset_ids_are_unique(self) -> Self:
        ids = [asset.asset_id for asset in (*self.required, *self.optional)]
        duplicates = sorted({asset_id for asset_id in ids if ids.count(asset_id) > 1})
        if duplicates:
            msg = f"asset_id duplicado en el bundle: {duplicates}"
            raise ValueError(msg)
        return self


class GeoTarget(ContractBase):
    """Recordatorio de geo-targeting; se verifica post-publicación, no en el gate."""

    country: str = Field(pattern=r"^[A-Z]{2}$")
    min_pct: int = Field(ge=1, le=100)
    enforcement: Literal["post_publication_manual"] = ENFORCEMENT


class MinViewsForPayout(ContractBase):
    """Umbral de views para payout; recordatorio post-publicación."""

    value: int | None = Field(default=None, ge=0)
    enforcement: Literal["post_publication_manual"] = ENFORCEMENT


class AnalyticsProofRequired(ContractBase):
    """Prueba de analytics requerida; recordatorio post-publicación."""

    value: bool = False
    enforcement: Literal["post_publication_manual"] = ENFORCEMENT


class GlobalRestrictions(ContractBase):
    """Restricciones globales que, con las de plataforma, activan rule_ids."""

    watermark_required: bool = False
    watermark_visible_full_video: bool = False
    has_required_assets: bool = False
    spelling_locks: tuple[str, ...] = ()
    prohibitions: tuple[str, ...] = ()


class Contract(ContractBase):
    """Contrato validado y normalizado que consumen pipeline y gate."""

    schema_version: Literal["1.0"] = "1.0"
    campaign_id: str = Field(min_length=1)
    format: Format
    mode: Mode
    platforms: dict[Platform, PlatformRules] = Field(min_length=1)
    languages: Languages
    official_audio: OfficialAudio | None = None
    watermark: Watermark
    spelling_locks: list[str] = Field(default_factory=list)
    prohibitions: list[str] = Field(default_factory=list)
    rules: RuleSet
    assets: AssetBundle
    geo_target: GeoTarget | None = None
    min_views_for_payout: MinViewsForPayout = Field(default_factory=MinViewsForPayout)
    analytics_proof_required: AnalyticsProofRequired = Field(default_factory=AnalyticsProofRequired)

    @model_validator(mode="after")
    def _mode_matches_format(self) -> Self:
        if (self.mode is Mode.SLIDESHOW) != (self.format is Format.SLIDESHOW):
            msg = "mode 'slideshow' y format 'slideshow' deben aparecer juntos"
            raise ValueError(msg)
        return self

    @model_validator(mode="after")
    def _official_audio_is_present_when_required(self) -> Self:
        needs_audio = any(
            rules.audio_rule is AudioRule.OFFICIAL_REQUIRED for rules in self.platforms.values()
        )
        if needs_audio and self.official_audio is None:
            msg = "audio_rule 'official_required' exige official_audio con al menos una URL"
            raise ValueError(msg)
        return self

    @model_validator(mode="after")
    def _declared_restrictions_are_classified(self) -> Self:
        classified = {*self.rules.hard, *self.rules.recommended, *self.rules.manual_review}
        global_restrictions = GlobalRestrictions(
            watermark_required=self.watermark.required,
            watermark_visible_full_video=self.watermark.visible_full_video,
            has_required_assets=bool(self.assets.required),
            spelling_locks=tuple(self.spelling_locks),
            prohibitions=tuple(self.prohibitions),
        )
        missing: set[str] = set()
        for platform, rules in self.platforms.items():
            missing.update(
                f"{platform}.{rule_id}"
                for rule_id in active_restriction_rules(rules, global_restrictions)
                if rule_id not in classified
            )
        if missing:
            msg = (
                "restricciones declaradas sin regla clasificada en rules: "
                f"{sorted(missing)}; cada regla debe ser hard, recommended o manual_review"
            )
            raise ValueError(msg)
        return self


def active_restriction_rules(
    rules: PlatformRules,
    global_restrictions: GlobalRestrictions,
) -> list[str]:
    """Deriva los rule_ids que activa una plataforma según sus restricciones.

    Args:
        rules: Restricciones declaradas para una plataforma.
        global_restrictions: Restricciones globales de la campaña.

    Returns:
        Los rule_ids del catálogo del gate que la plataforma exige; el
        contrato debe clasificarlos en ``rules`` para que el gate los
        evalúe en vez de ignorarlos en silencio.

    Nota:
        ``watermark.present`` cubre el watermark exigido sin cobertura total y
        aún no tiene validador mecánico (fase B): clasificado en ``rules``, el
        gate lo marca ``unsupported``.  ``audio_rule``, ``attribution`` y
        ``link_rules.link_in_bio`` tampoco tienen rule_id en el catálogo del
        gate (fases C/D); hasta que lo tengan, el contrato puede declararlos
        sin clasificación y el gate no los evalúa.
    """
    active: list[str] = []
    if rules.duration.min_s is not None:
        active.append("duration.min")
    if rules.duration.max_s is not None:
        active.append("duration.max")
    if rules.caption_rules.first_line is not None:
        active.append("caption.first_line")
    if rules.caption_rules.forbidden or global_restrictions.prohibitions:
        active.append("caption.forbidden")
    if rules.caption_rules.must_mention or rules.required_mentions:
        active.append("caption.required_mention")
    if rules.required_hashtags:
        active.append("caption.required_hashtag")
    if global_restrictions.has_required_assets:
        active.append("assets.required")
    watermark_rule = _watermark_rule(global_restrictions)
    if watermark_rule is not None:
        active.append(watermark_rule)
    if global_restrictions.spelling_locks:
        active.append("subtitles.spelling_lock")
    return active


def _watermark_rule(global_restrictions: GlobalRestrictions) -> str | None:
    if not global_restrictions.watermark_required:
        return None
    if global_restrictions.watermark_visible_full_video:
        return "watermark.full_video"
    return "watermark.present"
