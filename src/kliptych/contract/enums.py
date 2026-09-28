"""Enumeraciones del contrato de campaña."""

from enum import StrEnum


class Format(StrEnum):
    """Formato de la pieza."""

    VIDEO = "video"
    SLIDESHOW = "slideshow"


class Mode(StrEnum):
    """Modo de campaña que activa el pipeline."""

    GIVEN_CLIPS = "given_clips"
    LONG_VIDEO = "long_video"
    AUDIO_LOCKED = "audio_locked"
    REPOST_UGC = "repost_ugc"
    SLIDESHOW = "slideshow"


class Platform(StrEnum):
    """Plataformas soportadas por el contrato."""

    TIKTOK = "tiktok"
    INSTAGRAM_REELS = "instagram_reels"
    YOUTUBE_SHORTS = "youtube_shorts"
    X = "x"


class AudioRule(StrEnum):
    """Regla de audio de una plataforma."""

    OWN_CLIP = "own_clip"
    OFFICIAL_REQUIRED = "official_required"
    NO_TRENDING = "no_trending"
    ANY = "any"


class AudioPolicy(StrEnum):
    """Política de audio de la campaña."""

    INTERNAL_OFFICIAL_SOUND = "internal_official_sound"
    ORIGINAL_AUDIO = "original_audio"
    ANY_AUDIO = "any_audio"


class AttributionType(StrEnum):
    """Tipo de atribución requerida."""

    TAG = "tag"
    URL = "url"
    NONE = "none"


class RuleStrength(StrEnum):
    """Fuerza con la que el contrato clasifica cada regla."""

    HARD = "hard"
    RECOMMENDED = "recommended"
    MANUAL_REVIEW = "manual_review"
