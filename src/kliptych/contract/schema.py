"""Contrato validado (schema v1.2, lee v1.1): restricciones duras que obedecen pipeline y gate.

El contrato v1.1 incorpora ``segments`` para el modo ``long_video``: la lista de
cortes temporales seleccionados del vídeo fuente, obligatoria en ese modo y
prohibida en el resto.

v1.2 anade ``outputs``, la declaracion explicita de los formatos que la campana
emite. El significado no cambia: ausente significa un unico formato 9:16, como
siempre. Lo que cambia es que ahora se puede DECIR, que es el primer peldano
para renderizar mas de un formato.
"""

import math
from collections.abc import Sequence
from typing import Annotated, Literal, Self, cast

from pydantic import AwareDatetime, Field, StringConstraints, field_validator, model_validator

from kliptych.contract.base import ContractBase
from kliptych.contract.enums import (
    Aspect,
    AttributionType,
    AudioPolicy,
    AudioRule,
    Format,
    Mode,
    Platform,
    WatermarkPosition,
)
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
    language: str | None = Field(
        default=None,
        pattern=r"^[a-z]{2,3}(-[a-z]{2})?$",
        description=(
            "Locale explícito de transcripción (p. ej. 'es', 'en', 'pt-br'); "
            "None conserva la autodetección histórica de faster-whisper"
        ),
    )


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
    """Configuración de watermark de la campaña.

    ``position`` fija la zona del lienzo donde el ensamblado superpone el
    PNG y donde el gate lo busca; ``scale_ratio`` es el ancho del watermark
    relativo al ancho del video; ``opacity`` atenúa el PNG al componer (con
    suelo en 0.15: por debajo el logo es inverificable y el contrato se
    rechaza); y ``min_width_ratio`` es el ancho mínimo relativo que el gate
    acepta en la detección (nunca mayor que ``scale_ratio``).
    """

    required: bool
    asset_id: str | None = None
    visible_full_video: bool
    position: WatermarkPosition = WatermarkPosition.TOP_RIGHT
    scale_ratio: float = Field(default=0.20, ge=0.0, le=1.0)
    opacity: float = Field(default=1.0, ge=0.15, le=1.0)
    min_width_ratio: float = Field(default=0.05, ge=0.0, le=1.0)

    @field_validator("asset_id")
    @classmethod
    def _asset_id_is_safe(cls, asset_id: str | None) -> str | None:
        if asset_id is not None and not is_safe_segment(asset_id):
            msg = "asset_id no es un segmento de ruta seguro"
            raise ValueError(msg)
        return asset_id

    @model_validator(mode="after")
    def _scale_covers_minimum_width(self) -> Self:
        if self.scale_ratio < self.min_width_ratio:
            msg = (
                f"scale_ratio ({self.scale_ratio}) no puede ser menor que "
                f"min_width_ratio ({self.min_width_ratio})"
            )
            raise ValueError(msg)
        return self


WatermarkConfig = Watermark


_SPLIT_ASPECT_EPS = 1e-9
_SPLIT_EVEN_PX = 2
_SPLIT_MIN_PANEL_PX = 2
# Techo del lienzo. Ademas de que ningun encoder llega aqui, acota el cociente
# `width / height`: con enteros arbitrariamente grandes la division launcha
# `OverflowError`, que pydantic no envuelve, en vez de un `ValidationError`.
_MAX_CANVAS_PX = 65_536


class SplitScreenConfig(ContractBase):
    """Composición en dos paneles (superior + inferior).

    El lienzo tiene la geometria que declara ``aspect`` (9:16 por defecto,
    1080x1920) dividido en dos paneles horizontales: ``panel_ratio`` es la
    fracción del alto útil (alto menos ``gap``) que ocupa el panel superior;
    ``gap`` es la franja negra entre paneles, en píxeles pares. ``top_source`` y
    ``bottom_source`` son los asset_id de cada panel (video o imagen).
    """

    top_source: str = Field(min_length=1)
    bottom_source: str = Field(min_length=1)
    gap: int = Field(default=0, ge=0)
    panel_ratio: float = Field(default=0.5, gt=0, lt=1)
    width: int = Field(default=1080, gt=0, le=_MAX_CANVAS_PX)
    height: int = Field(default=1920, gt=0, le=_MAX_CANVAS_PX)
    # El lienzo se valida contra el ASPECTO DECLARADO, no contra una constante.
    # Antes compara con 9 y 16 de modulo, asi que el unico lienzo posible era
    # 9:16 y ningun contrato podia declarar otra cosa. El default sigue siendo
    # VERTICAL, de modo que un contrato que no dice nada se comporta EXACTAMENTE
    # como antes.
    # El gate valida este lienzo contra el ASPECTO DECLARADO AQUI
    # (`gate/checks.py:check_split_screen_geometry`), no contra una constante de
    # modulo, asi que un lienzo 16:9 con un artefacto 16:9 pasa. No es alcanzable
    # desde la CLI: el draft no expone `aspect`, asi que todo contrato resuelto
    # llega con VERTICAL.
    aspect: Aspect = Aspect.VERTICAL

    @field_validator("top_source", "bottom_source")
    @classmethod
    def _source_is_safe(cls, source: str) -> str:
        if not is_safe_segment(source):
            msg = "split source no es un segmento de ruta seguro"
            raise ValueError(msg)
        return source

    def panel_heights(self) -> tuple[int, int]:
        """Devuelve el alto de cada panel, en píxeles pares.

        El panel superior se redondea hacia abajo al par más cercano y el
        inferior absorbe el resto del alto útil: la suma siempre cubre el
        lienzo menos el ``gap`` y ambos quedan aptos para yuv420p.

        Returns:
            El par ``(alto_superior, alto_inferior)`` en píxeles.
        """
        available = self.height - self.gap
        top = int(math.floor(available * self.panel_ratio / _SPLIT_EVEN_PX) * _SPLIT_EVEN_PX)
        return (top, available - top)

    @model_validator(mode="after")
    def _canvas_and_panels_are_viable(self) -> Self:
        if self.width % _SPLIT_EVEN_PX != 0 or self.height % _SPLIT_EVEN_PX != 0:
            msg = f"el lienzo split_screen exige dimensiones pares, no {self.width}x{self.height}"
            raise ValueError(msg)
        declarado = self.aspect.ratio
        if abs(self.width / self.height - declarado) > _SPLIT_ASPECT_EPS:
            msg = (
                f"el lienzo split_screen exige el aspecto declarado "
                f"{self.aspect.value}, no {self.width}x{self.height}"
            )
            raise ValueError(msg)
        if self.gap % _SPLIT_EVEN_PX != 0:
            msg = f"el gap split_screen exige píxeles pares, no {self.gap}"
            raise ValueError(msg)
        if not self.gap < self.height:
            msg = f"el gap ({self.gap}) no puede cubrir el alto ({self.height})"
            raise ValueError(msg)
        top, bottom = self.panel_heights()
        if top < _SPLIT_MIN_PANEL_PX or bottom < _SPLIT_MIN_PANEL_PX:
            msg = f"los paneles split_screen son inviables: superior={top} px, inferior={bottom} px"
            raise ValueError(msg)
        return self


class LyricConfig(ContractBase):
    """Configuración de video de letras (format lyric_video)."""

    lrc_asset_id: str | None = None
    track_name: str | None = None
    artist_name: str | None = None
    lrclib_enabled: bool = True

    @field_validator("lrc_asset_id")
    @classmethod
    def _asset_id_is_safe(cls, asset_id: str | None) -> str | None:
        if asset_id is not None and not is_safe_segment(asset_id):
            msg = "lrc_asset_id no es un segmento de ruta seguro"
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


class TimestampRange(ContractBase):
    """Rango de marcas temporales mandatorio extraído del brief o solicitado."""

    start_sec: float = Field(ge=0)
    end_sec: float = Field(gt=0)

    @field_validator("start_sec", "end_sec")
    @classmethod
    def _bounds_are_finite(cls, value: float) -> float:
        if not math.isfinite(value):
            msg = f"las cotas del rango temporal deben ser finitas, no {value}"
            raise ValueError(msg)
        return value

    @model_validator(mode="after")
    def _ordered(self) -> Self:
        if not (0.0 <= self.start_sec < self.end_sec):
            msg = (
                f"se exige 0 <= start_sec < end_sec, pero "
                f"start_sec={self.start_sec}, end_sec={self.end_sec}"
            )
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
    audio_policy: AudioPolicy | None = None
    hook_keyword: str | None = None
    brand_safety_required: bool = False


class UnmappedRule(ContractBase):
    """Requisito del brief sin validador mecánico, con su cita textual.

    El gate no puede verificarlo con código: lo marca ``manual_review``
    con la cita para que un humano lo revise contra el brief original.
    """

    rule: str = Field(min_length=1)
    quote: str = Field(min_length=1)


KNOWN_VALIDATOR_RULES: frozenset[str] = frozenset(
    {
        # Este catalogo de rule_ids vive en el modulo de contrato, y no en el
        # gate, por una dependencia que conviene no romper: `unmapped` rechaza
        # como colision una regla que este名录ada aqui, asi que el contrato
        # necesita conocer el catalogo para validar su propia documentacion. Si
        # se mueve al gate, el contrato deja de poder validar ese caso y la
        # colision pasa por alto.
        #
        # `artifact.aspect` verifica que el artefacto satisface alguna de las
        # salidas que el contrato declara. Llega junto a `outputs` en el
        # esquema 1.2. Un documento 1.1 no declara la regla porque no declaro
        # salidas, pero el gate si lo mide igual: su conjunto efectivo es la
        # entrada vertical por defecto, y para un conjunto de un elemento
        # "pertenece a sus salidas" y "es 9:16" son la misma pregunta.
        "artifact.aspect",
        "artifact.integrity",
        "artifact.video_stream",
        "assets.required",
        "audio.present",
        "audio.policy",
        "audio.silence",
        "brand.safety",
        "caption.first_line",
        "caption.forbidden",
        "caption.required_hashtag",
        "caption.required_mention",
        "duration.max",
        "duration.min",
        "hook.keyword",
        "layout.geometry",
        "subtitles.spelling_lock",
        "watermark.full_video",
        "watermark.present",
    }
)


class OutputSpec(ContractBase):
    """Un formato que el contrato dice que hay que emitir.

    El `index` es una etiqueta ORDENADA dentro del conjunto de salidas, no un
    identificador unico. El par (plataforma, formato) es un producto cartesiano:
    una campana con dos plataformas y dos salidas emite cuatro artefactos por
    span. El nombre de entrega lo pondra PR 3, que hoy sigue siendo
    `{campana}-{plataforma}-{NN}` y no incluye el formato.
    """

    aspect: Aspect = Aspect.VERTICAL
    index: int = Field(default=0, ge=0)


_DEFAULT_OUTPUTS: tuple[OutputSpec, ...] = (OutputSpec(aspect=Aspect.VERTICAL, index=0),)


class Contract(ContractBase):
    """Contrato validado y normalizado que consumen pipeline y gate."""

    # "1.2" por una razon SINTACTICA, no semantica, y va en las DOS
    # direcciones. El DEFAULT es "1.2": esta version EMITE documentos 1.2, y
    # como ContractBase prohibe campos extra, un binario con "1.1" que lea uno
    # de esos lo RECHAZA con "schema_version no soportado" en vez de ignorarle
    # `outputs` y fallar mas tarde con un "campo extra no permitido" que no
    # dice nada. El conjunto permitido incluye "1.1" porque esta version LEE
    # documentos 1.1 sin cambiar su significado: al faltar `outputs`, el
    # default vertical significa exactamente lo que siempre significo.
    #
    # NO "endurecer" este Literal a un solo valor. Ponerlo en Literal["1.2"]
    # parece mas estricto pero rompe los documentos 1.1 que hay, que son
    # validos: lo que protege al lector viejo es el DEFAULT, no la lista.
    schema_version: Literal["1.1", "1.2"] = "1.2"
    campaign_id: str = Field(min_length=1, max_length=64)
    format: Format
    outputs: tuple[OutputSpec, ...] = Field(default=_DEFAULT_OUTPUTS, min_length=1)
    mode: Mode
    platforms: dict[Platform, PlatformRules] = Field(min_length=1)
    languages: Languages
    official_audio: OfficialAudio | None = None
    audio_policy: AudioPolicy | None = Field(
        default=None,
        description=(
            "Política de audio de la campaña; 'internal_official_sound' activa "
            "la regla 'audio.policy' (revisión manual obligatoria)"
        ),
    )
    watermark: Watermark
    spelling_locks: list[str] = Field(default_factory=list)
    prohibitions: list[str] = Field(default_factory=list)
    hook_keyword: str | None = Field(default=None, min_length=1, max_length=140)
    brand_safety_required: bool = False
    brand_safety_citation: str | None = None
    unmapped: tuple[UnmappedRule, ...] = ()
    timestamp_ranges: tuple[TimestampRange, ...] = ()
    lyric_video: LyricConfig | None = None
    split_screen: SplitScreenConfig | None = None
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
    def _outputs_are_distinct(self) -> Self:
        """Los aspectos declarados tienen que ser distintos.

        `min_length=1` ya rechaza la tupla vacia, pero no puede expresar "sin
        repetidos". Un contrato que declara (VERTICAL, VERTICAL) produciria dos
        piezas identicas, con el mismo piece_id y la misma carpeta de entrega, y
        el exportador lo rechazaria en cerrado por colision de nombres: el
        defecto se detecta tarde y con un mensaje que no senala la causa. Falla
        aqui, al construir, diciendo cual se repite.

        Returns:
            ``self``, sin cambios.

        Raises:
            ValueError: si un aspecto declarado se repite.
        """
        vistos = [spec.aspect for spec in self.outputs]
        repetidos = {a for a in vistos if vistos.count(a) > 1}
        if repetidos:
            msg = f"aspectos de salida repetidos: {sorted(a.value for a in repetidos)}"
            raise ValueError(msg)
        return self

    @model_validator(mode="after")
    def _schema_version_matches_outputs(self) -> Self:
        """Un documento 1.1 no puede declarar SALIDAS que difieran del default.

        `extra="forbid"` no ayuda aqui: `outputs` SI existe en el modelo, asi que
        un documento que dice 1.1 y ademas trae `outputs` se valida sin quejarse
        de nada. Pero estaria afirmando dos cosas incompatibles a la vez: que
        habla el 1.1, donde el campo no existia, y que declara una salida que el
        1.1 no puede expresar. Un documento que se contradice no se acepta en
        silencio: o se corrige a "1.2", o se quita `outputs`.

        La comparacion es por VALOR y no por presencia a proposito. Un 1.1 que
        trae el default se ACEPTA, porque `model_dump` emite siempre todos los
        campos y un contrato 1.1 tiene que sobrevivir a su propio round-trip
        (`tests/support.py:with_platform` revalida en vez de copiar el modelo).
        Un chequeo por presencia rechazaria ese round-trip y, con el, media suite.

        Returns:
            ``self``, sin cambios.

        Raises:
            ValueError: si el documento se declara 1.1 y declara salidas que no
                son el default.
        """
        if self.schema_version == "1.1" and self.outputs != _DEFAULT_OUTPUTS:
            declarados = ", ".join(spec.aspect.value for spec in self.outputs)
            msg = (
                f"un documento schema_version 1.1 no puede declarar outputs "
                f"({declarados}): ese campo no existia en 1.1. Pon schema_version "
                f'"1.2" para declarar salidas, o quita `outputs`.'
            )
            raise ValueError(msg)
        return self

    @model_validator(mode="after")
    def _split_screen_matches_outputs(self) -> Self:
        """El lienzo split_screen tiene que encajar con las salidas declaradas.

        Sin esta comprobacion el contrato tiene DOS declaraciones de aspecto que
        nadie reconcilia: `SplitScreenConfig.aspect` dice la geometria de ESTE
        artefacto y `Contract.outputs` dice que emite la campana. Un contrato
        podria declarar outputs=(HORIZONTAL,) y traer un lienzo 9:16, y nada lo
        notaria hasta el render, que produciria un artefacto con un formato que
        la campana no declaro. El issue #76 existe porque el aspecto tenia
        demasiadas casas independientes; esta es la que las une.

        Cuando hay un unico formato, el lienzo tiene que coincidir con el.
        Cuando hay varios, el modelo todavia no sabe a cual pertenece el lienzo,
        asi que se rechaza en lugar de adivinar. El limite es deliberado: si
        alguna vez hace falta, la ampliacion es un campo que diga a que salida
        pertenece el lienzo, mas esta misma reconciliacion.

        Returns:
            ``self``, sin cambios.

        Raises:
            ValueError: si la campana declara varios formatos y trae un lienzo,
                o si el aspecto del lienzo no esta entre las salidas declaradas.
        """
        if self.split_screen is None:
            return self
        declarados = [spec.aspect for spec in self.outputs]
        if len(declarados) > 1:
            detalle = ", ".join(aspect.value for aspect in declarados)
            msg = (
                f"una campana multi-formato ({detalle}) no puede declarar un "
                f"lienzo split_screen: el lienzo describe un unico formato y este "
                f"modelo no dice a cual de los {len(declarados)} pertenece. "
                f"Es una limitacion deliberada, no un defecto pendiente"
            )
            raise ValueError(msg)
        declarado = next(iter(declarados))
        if self.split_screen.aspect is not declarado:
            msg = (
                f"el lienzo split_screen declara el aspecto "
                f"{self.split_screen.aspect.value}, que no esta en las salidas "
                f"declaradas ({declarado.value})"
            )
            raise ValueError(msg)
        return self

    @field_validator("hook_keyword")
    @classmethod
    def _hook_keyword_is_meaningful(cls, hook_keyword: str | None) -> str | None:
        if hook_keyword is not None and not hook_keyword.strip():
            msg = "hook_keyword no puede ser vacío o solo espacios"
            raise ValueError(msg)
        return hook_keyword

    @field_validator("unmapped")
    @classmethod
    def _unmapped_rules_do_not_collide_with_catalog(
        cls, unmapped: tuple[UnmappedRule, ...]
    ) -> tuple[UnmappedRule, ...]:
        normalized_known = {rule.strip().lower() for rule in KNOWN_VALIDATOR_RULES}
        for entry in unmapped:
            if entry.rule.strip().lower() in normalized_known:
                msg = (
                    f"la regla no mapeada '{entry.rule}' colisiona con un validador "
                    "conocido en el catálogo del gate"
                )
                raise ValueError(msg)
        return unmapped

    @model_validator(mode="after")
    def _audio_policy_compatible_with_mode(self) -> Self:
        """Exige la matriz audio_rule x audio_policy a nivel de contrato.

        Combinaciones válidas:

        - ``official_required`` x ``internal_official_sound``: el render se
          silencia y el sonido oficial se añade al publicar.
        - ``any`` / ``own_clip`` / ``no_trending`` x ``internal_official_sound``:
          el mute aplica igual al publicar; la pista del render queda inaudible.
        - Cualquier ``audio_rule`` x ``original_audio`` / ``any_audio`` / None:
          se conserva el audio del render.

        El modo ``audio_locked`` inyecta una pista externa, lo contrario de
        silenciar: es incompatible con ``internal_official_sound`` (la pista
        externa también se rechaza en el orquestador).

        Returns:
            El contrato validado.

        Raises:
            ValueError: Si el modo inyecta audio externo con política de silencio.
        """
        if (
            self.mode is Mode.AUDIO_LOCKED
            and self.audio_policy is AudioPolicy.INTERNAL_OFFICIAL_SOUND
        ):
            msg = (
                "mode 'audio_locked' inyecta una pista de audio externa, incompatible con "
                "audio_policy 'internal_official_sound' (el render debe silenciarse)"
            )
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
            audio_policy=self.audio_policy,
            hook_keyword=self.hook_keyword,
            brand_safety_required=self.brand_safety_required,
        )
        aliases = RULE_ID_ALIASES
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
    def _brand_safety_citation_present_when_required(self) -> Self:
        if self.brand_safety_required and (
            not self.brand_safety_citation or not self.brand_safety_citation.strip()
        ):
            msg = "brand_safety_required=True exige brand_safety_citation no vacía"
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


# Las reglas que el modo zero-contract declara sin brief. Cada una esta por una
# razon concreta, y borrarla cuesta un fallo silencioso:
#
# - `artifact.integrity`: sin esta, `_derive_status` devuelve PASSED sobre cero
#   checks y el gate aprueba en vacio una pieza que no existe.
# - `artifact.video_stream`: para probar que el MP4 TIENE video. Sin ella, un
#   contenedor sin pista de video pasaria.
# - `artifact.aspect`: el modo zero-contract declara `outputs` = un 9:16 (el
#   default de `_DEFAULT_OUTPUTS`, sin brief que diga otra cosa), asi que su
#   salida DEBERIA ser vertical y el artefacto tiene que medir como tal.
#   No esta aqui porque el render emita 1080x1920: no lo emite. El reframe
#   cuantiza a entero par, asi que 1920x1080 -> 608x1080 (0,08 % de 9:16) y
#   1280x720 -> 404x720 (0,25 %). Por eso el gate verifica con
#   `encoding.aspect_matches` y la tolerancia de la corrida, no con una
#   comparacion exacta.
#   Y no es redundante con `artifact.video_stream`: aquella prueba que HAY video,
#   esta que tiene la geometria que el contrato declaro.
_ZERO_CONTRACT_AV_RULES = (
    "artifact.aspect",
    "artifact.integrity",
    "artifact.video_stream",
)
_ZERO_CONTRACT_LANGUAGE = "es"


def vanilla_contract(
    *,
    campaign_id: str,
    platform: Platform = Platform.TIKTOK,
    language: str = _ZERO_CONTRACT_LANGUAGE,
) -> Contract:
    """Construye el contrato "vainilla" del modo zero-contract, sin brief ni LLM.

    Es el contrato que se usa cuando el operador solo aporta una ``--url``: no
    hay brief que extraer, así que no se llama al LLM y se sintetiza aquí el
    mínimo válido.

    Declara como reglas ``hard`` la integridad estándar del artefacto:
    ``artifact.integrity`` (que exista y sea legible) y ``artifact.video_stream``
    (que tenga flujo de video). Sin ellas el gate aprobaría en vacío incluso con
    el artefacto ausente, porque ``_derive_status`` devuelve ``PASSED`` sobre
    cero checks.

    No declara reglas de audio. La plataforma usa ``PlatformRules()``, cuyo
    ``audio_rule`` es ``ANY``: no hay audio que exigir, y el resolver solo añade
    ``audio.present`` cuando alguna plataforma exige audio. Declararla aquí
    sería una aserción que el gate no puede sostener.

    Lo que el contrato NO declara es precisamente lo que no se puede saber sin
    un brief: sin prohibiciones léxicas, sin ``brand_safety_required``, sin
    bounds de duración, sin menciones ni hashtags, y sin watermark. El audio
    original se conserva declarando ``AudioPolicy.ORIGINAL_AUDIO``, que no
    silencia la pista.

    Args:
        campaign_id: Identificador de campaña; debe ser un segmento de ruta
            seguro porque se usa como nombre de directorio en la entrega.
        platform: Plataforma única que declara el contrato.
        language: Locale de fuente y caption.

    Returns:
        Un ``Contract`` válido, sin reglas de campaña.

    Raises:
        ValueError: Si ``campaign_id`` está vacío. El schema no valida que sea
            un segmento de ruta seguro ni que ``language`` sea un locale: el
            llamador de la CLI deriva el identificador de un hash, así que
            ninguno de los dos valores alcanza una ruta.
    """
    if not campaign_id:
        msg = "vanilla_contract exige un campaign_id no vacío"
        raise ValueError(msg)
    return Contract(
        campaign_id=campaign_id,
        format=Format.VIDEO,
        mode=Mode.REPOST_UGC,
        platforms={platform: PlatformRules()},
        languages=Languages(source=language, caption=language),
        watermark=Watermark(required=False, visible_full_video=False),
        rules=RuleSet(hard=list(_ZERO_CONTRACT_AV_RULES)),
        assets=AssetBundle(),
        audio_policy=AudioPolicy.ORIGINAL_AUDIO,
        brand_safety_required=False,
    )


def prompt_languages(contract: Contract) -> dict[str, object]:
    """Idiomas del contrato para prompts de LLM, sin el locale de transcripción.

    ``languages.language`` es un ajuste del motor de transcripción, no contexto
    de redacción ni de selección (el prompt de segmentos ya lleva el idioma
    detectado en la transcripción): excluirlo mantiene estables las claves de
    replay de captions grabados y los prompts ya fijados.

    Args:
        contract: Contrato validado.

    Returns:
        El volcado de idiomas sin la clave ``language``.
    """
    dump = contract.languages.model_dump(mode="json")
    if "language" in dump:
        del dump["language"]
    return dump


def contract_mutes_audio(contract: Contract) -> bool:
    """Indica si el contrato exige silenciar el render final de cada pieza.

    Solo ``audio_policy=internal_official_sound`` silencia: el sonido oficial
    se añade en la publicación y el MP4 debe llevar la pista presente pero en
    silencio digital. Las demás políticas (o su ausencia) conservan el audio.

    Args:
        contract: Contrato validado de la campaña.

    Returns:
        True si el render final debe aplicar el filtro de silenciado.
    """
    return contract.audio_policy is AudioPolicy.INTERNAL_OFFICIAL_SOUND


def contract_digest(contract: Contract) -> str:
    """Calcula el hash canónico del contrato lógico.

    Excluye metadatos volátiles de resolución (``resolved_at`` de los assets):
    el mismo contrato resuelto dos veces produce el mismo digest. Los
    opcionales añadidos tras v1.1 (``languages.language``, ``audio_policy``,
    ``outputs``) se excluyen cuando no se declaran, para no mover el hash de
    contratos ya grabados; al declararlos sí entran al digest e invalidan la
    caché de ``--resume``.

    Args:
        contract: Contrato validado.

    Returns:
        El digest sha256 en hexadecimal.
    """
    dump: dict[str, object] = contract.model_dump(mode="json", exclude=_VOLATILE_RESOLUTION_FIELDS)
    _prune_unset_options(contract, dump)
    return sha256_canonical_json(dump)


def _prune_unset_options(contract: Contract, dump: dict[str, object]) -> None:
    """Elimina del dump los opcionales no declarados.

    El modelo dice qué se declaró; el dump solo se poda en lo mecánico para
    no mover el hash de contratos ya grabados.

    Args:
        contract: Contrato validado que indica los opcionales declarados.
        dump: Volcado JSON del contrato cuyo digest se va a calcular; se
            modifica en sitio.
    """
    if contract.languages.language is None:
        languages = cast("dict[str, object]", dump["languages"])
        if "language" in languages:
            del languages["language"]
    if contract.audio_policy is None:
        _ = dump.pop("audio_policy", None)
    if contract.outputs == _DEFAULT_OUTPUTS:
        # `outputs` llego en v1.2, asi que un contrato ya grabado que no lo
        # declara no lo tiene en su dump. Podarlo cuando vale el default es lo
        # que mantiene el digest identico entre versiones; al declarar salidas
        # de verdad si entran, y con eso invalidan la cache de `--resume`.
        _ = dump.pop("outputs", None)
    if contract.hook_keyword is None:
        _ = dump.pop("hook_keyword", None)
    if not contract.unmapped:
        _ = dump.pop("unmapped", None)
    if not contract.brand_safety_required:
        _ = dump.pop("brand_safety_required", None)
    if contract.brand_safety_citation is None:
        _ = dump.pop("brand_safety_citation", None)
    if not contract.timestamp_ranges:
        _ = dump.pop("timestamp_ranges", None)
    _prune_unset_media_options(contract, dump)
    _prune_default_watermark_options(dump)


_DEFAULT_WATERMARK_JSON: dict[str, object] = Watermark(
    required=False, visible_full_video=False
).model_dump(mode="json")
_WATERMARK_TUNABLE_KEYS: tuple[str, ...] = (
    "position",
    "scale_ratio",
    "opacity",
    "min_width_ratio",
)


def _prune_unset_media_options(contract: Contract, dump: dict[str, object]) -> None:
    """Elimina del dump los bloques multimedia no declarados.

    Args:
        contract: Contrato validado que indica los opcionales declarados.
        dump: Volcado JSON del contrato cuyo digest se va a calcular; se
            modifica en sitio.
    """
    if contract.lyric_video is None:
        _ = dump.pop("lyric_video", None)
    if contract.split_screen is None:
        _ = dump.pop("split_screen", None)


def _prune_default_watermark_options(dump: dict[str, object]) -> None:
    """Elimina del dump los ajustes de watermark que conservan el valor por defecto.

    Los campos de posición y tamaño (fase sprint 2) no existían en contratos
    ya grabados: podarlos cuando valen lo mismo que el defecto mantiene el
    digest estable entre versiones; al declararlos sí entran al digest e
    invalidan la caché de ``--resume``.

    Args:
        dump: Volcado JSON del contrato cuyo digest se va a calcular; se
            modifica en sitio.
    """
    raw = dump.get("watermark")
    if not isinstance(raw, dict):
        return
    watermark = cast("dict[str, object]", raw)
    for key in _WATERMARK_TUNABLE_KEYS:
        if watermark.get(key) == _DEFAULT_WATERMARK_JSON.get(key):
            _ = watermark.pop(key, None)


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
    active.extend(_audio_restriction_rules(rules))
    if rules.attribution.type is not AttributionType.NONE:
        active.append("attribution.required")
    if rules.link_rules.link_in_bio:
        active.append("link.in_bio")
    return active


def _audio_restriction_rules(rules: PlatformRules) -> list[str]:
    """Deriva la regla de audio que exige la plataforma, si declara alguna.

    Args:
        rules: Restricciones declaradas para una plataforma.

    Returns:
        El rule_id de audio correspondiente, o lista vacía con ``any``.
    """
    if rules.audio_rule is AudioRule.OFFICIAL_REQUIRED:
        return ["audio.official_track"]
    if rules.audio_rule is AudioRule.OWN_CLIP:
        return ["audio.own_clip"]
    if rules.audio_rule is AudioRule.NO_TRENDING:
        return ["audio.no_trending"]
    return []


def _global_restriction_rules(global_restrictions: GlobalRestrictions) -> list[str]:
    active: list[str] = []
    if global_restrictions.has_required_assets:
        active.append("assets.required")
    watermark_rule = _watermark_rule(global_restrictions)
    if watermark_rule is not None:
        active.append(watermark_rule)
    if global_restrictions.spelling_locks:
        active.append("subtitles.spelling_lock")
    if global_restrictions.hook_keyword:
        active.append("hook.keyword")
    if global_restrictions.audio_policy is AudioPolicy.INTERNAL_OFFICIAL_SOUND:
        active.extend(("audio.policy", "audio.silence"))
    if global_restrictions.brand_safety_required:
        active.append("brand.safety")
    return active


# Regla canonica -> rule_ids que el contrato acepta como su clasificacion. Un
# operador (o un modelo) puede escribir cualquiera de los alias, asi que quien
# decida algo a partir de un rule_id debe canonizarlo antes. Es la unica fuente:
# el validador de clasificacion y el predicado de aplicabilidad del gate la
# comparten.
RULE_ID_ALIASES: dict[str, tuple[str, ...]] = {
    "audio.official_track": (
        "audio.official_track",
        "audio.official_selection",
        "audio.rule",
    ),
    "attribution.required": ("attribution.required", "attribution.present"),
    "link.in_bio": ("link.in_bio", "link_rules.link_in_bio"),
}

_ALIAS_TO_CANONICAL: dict[str, str] = {
    alias: canonical for canonical, aliases in RULE_ID_ALIASES.items() for alias in aliases
}


def canonical_rule_id(rule_id: str) -> str:
    """Devuelve el ``rule_id`` canonico de una regla o de uno de sus alias.

    Args:
        rule_id: Identificador tal y como aparece en ``contract.rules``.

    Returns:
        El identificador canonico, o el mismo ``rule_id`` si no tiene alias.
    """
    return _ALIAS_TO_CANONICAL.get(rule_id, rule_id)


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
