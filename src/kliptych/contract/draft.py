"""ContractDraft: salida cruda del extractor, con evidencia por campo.

No se usa para renderizar: el resolutor lo normaliza a ``Contract`` y solo
entonces el pipeline y el gate lo consumen.
"""

from typing import Literal

from pydantic import Field

from kliptych.contract.base import ContractBase
from kliptych.contract.enums import (
    AttributionType,
    AudioPolicy,
    AudioRule,
    Format,
    Mode,
    Platform,
    WatermarkPosition,
)
from kliptych.contract.evidence import FieldCandidate


class DurationDraft(ContractBase):
    """Duración propuesta, con evidencia por cota."""

    min_s: FieldCandidate[int] | None = None
    max_s: FieldCandidate[int] | None = None


class CaptionRulesDraft(ContractBase):
    """Reglas de caption propuestas, con evidencia por regla."""

    must_mention: FieldCandidate[list[str]] | None = None
    first_line: FieldCandidate[str] | None = None
    forbidden: FieldCandidate[list[str]] | None = None


class AttributionDraft(ContractBase):
    """Atribución propuesta, con evidencia por subcampo."""

    type: FieldCandidate[AttributionType] | None = None
    value: FieldCandidate[str] | None = None


class LinkRulesDraft(ContractBase):
    """Reglas de enlace propuestas, con evidencia."""

    link_in_bio: FieldCandidate[bool] | None = None


class PlatformDraft(ContractBase):
    """Restricciones propuestas para una plataforma."""

    duration: DurationDraft | None = None
    caption_rules: CaptionRulesDraft | None = None
    audio_rule: FieldCandidate[AudioRule] | None = None
    required_hashtags: FieldCandidate[list[str]] | None = None
    required_mentions: FieldCandidate[list[str]] | None = None
    attribution: AttributionDraft | None = None
    link_rules: LinkRulesDraft | None = None


class LanguagesDraft(ContractBase):
    """Idiomas propuestos, con evidencia por campo."""

    source: FieldCandidate[str] | None = None
    subtitles: FieldCandidate[str] | None = None
    caption: FieldCandidate[str] | None = None
    voice: FieldCandidate[str] | None = None
    language: FieldCandidate[str] | None = None


class OfficialAudioDraft(ContractBase):
    """URLs de audio oficial propuestas, con evidencia por plataforma."""

    tiktok_url: FieldCandidate[str] | None = None
    instagram_url: FieldCandidate[str] | None = None


class WatermarkDraft(ContractBase):
    """Watermark propuesto, con evidencia por campo."""

    required: FieldCandidate[bool] | None = None
    asset_id: FieldCandidate[str] | None = None
    visible_full_video: FieldCandidate[bool] | None = None
    position: FieldCandidate[WatermarkPosition] | None = None
    scale_ratio: FieldCandidate[float] | None = None
    opacity: FieldCandidate[float] | None = None
    min_width_ratio: FieldCandidate[float] | None = None


class SegmentDraft(ContractBase):
    """Segmento temporal propuesto, con evidencia por cota."""

    start_s: FieldCandidate[float] | None = None
    end_s: FieldCandidate[float] | None = None


class SegmentsDraft(ContractBase):
    """Segmentos propuestos para el modo long_video."""

    segments: list[SegmentDraft] = Field(default_factory=list, max_length=1000)


class RuleSetDraft(ContractBase):
    """Clasificación propuesta de reglas, con evidencia por categoría."""

    hard: FieldCandidate[list[str]] | None = None
    recommended: FieldCandidate[list[str]] | None = None
    manual_review: FieldCandidate[list[str]] | None = None


class AssetDraft(ContractBase):
    """Asset referenciado en el brief; hash y tamaño los resuelve el registry."""

    asset_id: FieldCandidate[str] | None = None
    kind: FieldCandidate[str] | None = None
    uri: FieldCandidate[str] | None = None
    origin: FieldCandidate[str] | None = None
    license: FieldCandidate[str] | None = None


class AssetsDraft(ContractBase):
    """Assets requeridos y opcionales propuestos."""

    required: list[AssetDraft] = Field(default_factory=list)
    optional: list[AssetDraft] = Field(default_factory=list)


class GeoTargetDraft(ContractBase):
    """Geo-targeting propuesto; el enforcement lo agrega el resolutor."""

    country: FieldCandidate[str] | None = None
    min_pct: FieldCandidate[int] | None = None


class UnmappedRuleDraft(ContractBase):
    """Requisito del brief sin validador mecánico propuesto con su cita."""

    rule: FieldCandidate[str] | None = None
    quote: FieldCandidate[str] | None = None


class ContractDraft(ContractBase):
    """Salida cruda del LLM: todo campo es opcional y lleva su evidencia."""

    schema_version: Literal["1.1"] = "1.1"
    campaign_id: FieldCandidate[str] | None = None
    format: FieldCandidate[Format] | None = None
    mode: FieldCandidate[Mode] | None = None
    platforms: dict[Platform, PlatformDraft] = Field(default_factory=dict)
    languages: LanguagesDraft | None = None
    official_audio: OfficialAudioDraft | None = None
    audio_policy: FieldCandidate[AudioPolicy] | None = None
    watermark: WatermarkDraft | None = None
    spelling_locks: FieldCandidate[list[str]] | None = None
    prohibitions: FieldCandidate[list[str]] | None = None
    hook_keyword: FieldCandidate[str] | None = None
    hook_window_seconds: FieldCandidate[float] | None = None
    brand_safety: FieldCandidate[bool] | None = None
    unmapped: list[UnmappedRuleDraft] = Field(default_factory=list)
    rules: RuleSetDraft | None = None
    assets: AssetsDraft | None = None
    segments: SegmentsDraft | None = None
    geo_target: GeoTargetDraft | None = None
    min_views_for_payout: FieldCandidate[int] | None = None
    analytics_proof_required: FieldCandidate[bool] | None = None
