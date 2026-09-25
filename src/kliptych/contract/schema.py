"""Contrato validado (schema v1.1): restricciones duras que obedecen pipeline y gate.

El contrato v1.1 incorpora ``segments`` para el modo ``long_video``: la lista de
cortes temporales seleccionados del vídeo fuente, obligatoria en ese modo y
prohibida en el resto.
"""

import math
from collections.abc import Sequence
from typing import Annotated, Literal, Self

from pydantic import AwareDatetime, Field, StringConstraints, field_validator, model_validator

from kliptych.contract.base import ContractBase
from kliptych.contract.enums import AttributionType, AudioRule, Format, Mode, Platform
from kliptych.hashing import sha256_canonical_json
from kliptych.naming import is_safe_segment

ENFORCEMENT = "post_publication_manual"
_RULE_ID = Annotated[str, StringConstraints(pattern=r"^[a-z][a-z0-9_]*(\.[a-z][a-z0-9_]*)+$")]
_SHA256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
_MIME = Annotated[str, StringConstraints(pattern=r"^[a-z]+/[a-z0-9][a-z0-9.+-]*$")]
_MENTION = Annotated[str, StringConstraints(pattern=r"^@\S+$")]
Hashtag = Annotated[str, StringConstraints(pattern=r"^#\S+$")]


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
    required_hashtags: list[Hashtag] = Field(default_factory=list)
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

    @field_validator("asset_id")
    @classmethod
    def _asset_id_is_safe(cls, asset_id: str | None) -> str | None:
        if asset_id is not None and not is_safe_segment(asset_id):
            msg = "asset_id no es un segmento de ruta seguro"
            raise ValueError(msg)
        return asset_id


class Segment(ContractBase):
    """Segmento temporal seleccionado de un vídeo fuente (modo long_video)."""

    start_s: float = Field(ge=0)
    end_s: float = Field(gt=0)

    @field_validator("start_s", "end_s")
    @classmethod
    def _bounds_are_finite(cls, value: float) -> float:
        if not math.isfinite(value):
            msg = f"las cotas del segmento deben ser finitas, no {value}"
            raise ValueError(msg)
        return value

    @model_validator(mode="after")
    def _ordered(self) -> Self:
        if self.end_s <= self.start_s:
            msg = f"start_s ({self.start_s}) debe ser menor que end_s ({self.end_s})"
            raise ValueError(msg)
        return self


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

    @field_validator("asset_id")
    @classmethod
    def _asset_id_is_safe(cls, asset_id: str) -> str:
        if not is_safe_segment(asset_id):
            msg = "asset_id no es un segmento de ruta seguro"
            raise ValueError(msg)
        return asset_id


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

    schema_version: Literal["1.1"] = "1.1"
    campaign_id: str = Field(min_length=1, max_length=64)
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
    segments: tuple[Segment, ...] = ()
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
        aliases: dict[str, tuple[str, ...]] = {
            "audio.official_track": (
                "audio.official_track",
                "audio.official_selection",
                "audio.rule",
            ),
            "attribution.required": ("attribution.required", "attribution.present"),
            "link.in_bio": ("link.in_bio", "link_rules.link_in_bio"),
        }
        missing: set[str] = set()
        for platform, rules in self.platforms.items():
            for rule_id in active_restriction_rules(rules, global_restrictions):
                allowed = aliases.get(rule_id, (rule_id,))
                if not any(alias in classified for alias in allowed):
                    missing.add(f"{platform}.{rule_id}")
        if missing:
            msg = (
                "restricciones declaradas sin regla clasificada en rules: "
                f"{sorted(missing)}; cada regla debe ser hard, recommended o manual_review"
            )
            raise ValueError(msg)
        return self

    @model_validator(mode="after")
    def _segments_valid_for_mode(self) -> Self:
        if self.mode is Mode.LONG_VIDEO:
            if not self.segments:
                msg = "el modo long_video exige al menos un segmento"
                raise ValueError(msg)
        elif self.segments:
            msg = "solo el modo long_video admite segmentos"
            raise ValueError(msg)
        return self


# `resolved_at` es procedencia de la resolución (reloj), no identidad del
# contrato lógico: excluirlo mantiene el digest estable entre corridas.
_VOLATILE_RESOLUTION_FIELDS: dict[str, dict[str, dict[str, set[str]]]] = {
    "assets": {
        "required": {"__all__": {"resolved_at"}},
        "optional": {"__all__": {"resolved_at"}},
    }
}


def contract_digest(contract: Contract) -> str:
    """Calcula el hash canónico del contrato lógico.

    Excluye metadatos volátiles de resolución (``resolved_at`` de los assets):
    el mismo contrato resuelto dos veces produce el mismo digest.

    Args:
        contract: Contrato validado.

    Returns:
        El digest sha256 en hexadecimal.
    """
    return sha256_canonical_json(
        contract.model_dump(mode="json", exclude=_VOLATILE_RESOLUTION_FIELDS)
    )


def _platform_restriction_rules(
    rules: PlatformRules,
    prohibitions: Sequence[str],
) -> list[str]:
    active: list[str] = []
    if rules.duration.min_s is not None:
        active.append("duration.min")
    if rules.duration.max_s is not None:
        active.append("duration.max")
    if rules.caption_rules.first_line is not None:
        active.append("caption.first_line")
    if rules.caption_rules.forbidden or prohibitions:
        active.append("caption.forbidden")
    if rules.caption_rules.must_mention or rules.required_mentions:
        active.append("caption.required_mention")
    if rules.required_hashtags:
        active.append("caption.required_hashtag")
    if rules.audio_rule is AudioRule.OFFICIAL_REQUIRED:
        active.append("audio.official_track")
    if rules.attribution.type is not AttributionType.NONE:
        active.append("attribution.required")
    if rules.link_rules.link_in_bio:
        active.append("link.in_bio")
    return active


def _global_restriction_rules(global_restrictions: GlobalRestrictions) -> list[str]:
    active: list[str] = []
    if global_restrictions.has_required_assets:
        active.append("assets.required")
    watermark_rule = _watermark_rule(global_restrictions)
    if watermark_rule is not None:
        active.append(watermark_rule)
    if global_restrictions.spelling_locks:
        active.append("subtitles.spelling_lock")
    return active


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
    """
    return [
        *_platform_restriction_rules(rules, global_restrictions.prohibitions),
        *_global_restriction_rules(global_restrictions),
    ]


def _watermark_rule(global_restrictions: GlobalRestrictions) -> str | None:
    if not global_restrictions.watermark_required:
        return None
    if global_restrictions.watermark_visible_full_video:
        return "watermark.full_video"
    return "watermark.present"
