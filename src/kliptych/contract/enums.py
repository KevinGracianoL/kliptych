"""Enumeraciones del contrato de campaña."""

from enum import StrEnum


class Format(StrEnum):
    """Formato de la pieza."""

    VIDEO = "video"
    SLIDESHOW = "slideshow"
    LYRIC_VIDEO = "lyric_video"


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


class Aspect(StrEnum):
    """Relacion de aspecto que el contrato declara para su salida.

    Es un conjunto CERRADO a proposito. Un float admitiria 1,7777, que ningun
    codificador respeta y que ningun check podria comparar con exactitud; un
    enum hace que el conjunto soportado sea explicito y que un aspecto no
    soportado sea un error de construccion y no un render silenciosamente
    equivocado. Cualquier otra relacion (1:1, 4:5) se anade aqui de forma
    explicita cuando haga falta.

    La razon la expone cada miembro para que ningun consumidor la vuelva a
    escribir: el unico lugar que conoce el 9:16 y el 16:9 es este enum.

    Las dimensiones son ENTERAS y se guardan como tal. La razon derivada sigue
    siendo un float, pero la pertenencia y la validacion de lienzo se hacen con
    `numerator`/`denominator` por multiplicacion cruzada, que es exacta: un
    cociente de enteros no lo es, y una tolerancia fija sobre el deja pasar
    lienzos que difieren del declarado por un pixel.
    """

    VERTICAL = "9:16"
    HORIZONTAL = "16:9"
    SQUARE = "1:1"
    PORTRAIT_4_5 = "4:5"

    @property
    def numerator(self) -> int:
        """El ancho de la razon, como entero exacto.

        Returns:
            El numerador de la relacion ancho:alto.
        """
        return _ASPECT_DIMENSIONS[self][0]

    @property
    def denominator(self) -> int:
        """El alto de la razon, como entero exacto.

        Returns:
            El denominador de la relacion ancho:alto.
        """
        return _ASPECT_DIMENSIONS[self][1]

    @property
    def ratio(self) -> float:
        """La relacion ancho/alto que declara este aspecto.

        La tabla se indexa por MIEMBRO, no por el string del valor, para que
        anadir un miembro sin su razon falle en el test que recorre `Aspect` y
        no con un `KeyError` dentro de un validador, que pydantic no envuelve.

        Returns:
            La razon como float.
        """
        return self.numerator / self.denominator


_ASPECT_DIMENSIONS: dict["Aspect", tuple[int, int]] = {
    Aspect.VERTICAL: (9, 16),
    Aspect.HORIZONTAL: (16, 9),
    Aspect.SQUARE: (1, 1),
    Aspect.PORTRAIT_4_5: (4, 5),
}


class WatermarkPosition(StrEnum):
    """Zona del lienzo donde se superpone el watermark."""

    CENTER_BOTTOM = "center_bottom"
    CENTER_TOP = "center_top"
    CENTER = "center"
    TOP_RIGHT = "top_right"
    TOP_LEFT = "top_left"
    BOTTOM_RIGHT = "bottom_right"
    BOTTOM_LEFT = "bottom_left"


class RuleStrength(StrEnum):
    """Fuerza con la que el contrato clasifica cada regla."""

    HARD = "hard"
    RECOMMENDED = "recommended"
    MANUAL_REVIEW = "manual_review"


class Layout(StrEnum):
    """Disposición de los paneles en el lienzo vertical."""

    SINGLE = "single"
    SPLIT_SCREEN = "split_screen"
    VERTICAL = "vertical"
    HORIZONTAL = "horizontal"
    DIVIDIDO = "dividido"
    TRIO = "trio"
    SPOTLIGHT = "spotlight"


class CaptionCapitalization(StrEnum):
    """Estilo de capitalización para los subtítulos karaoke."""

    ORIGINAL = "original"
    UPPERCASE = "uppercase"
    LOWERCASE = "lowercase"


class SubtitleLanguage(StrEnum):
    """Idioma de los subtítulos soportado en el contrato."""

    ORIGINAL = "original"
    ENGLISH = "en"
    SPANISH = "es"
    PORTUGUESE = "pt"
    FRENCH = "fr"
    GERMAN = "de"
    ITALIAN = "it"
    JAPANESE = "ja"
