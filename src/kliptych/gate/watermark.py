"""Validadores mecánicos del watermark con OpenCV, indexados por rule_id.

- ``watermark.present``: el logo aparece al menos en una muestra del video.
- ``watermark.full_video``: el logo aparece en todas las muestras,
  distribuidas con densidad temporal por la duración: si desaparece a
  mitad o al final del clip, el resultado es ``fail``.

Las muestras se leen con una sola captura ``cv2.VideoCapture`` (un ``open``
y un seek por instante, sin un proceso ffmpeg por frame). Cada frame se
compara con el PNG de referencia reescalado al tamaño esperado del render
(con máscara alfa cuando el PNG la trae). La ubicación se busca con
``TM_SQDIFF`` enmascarado (estable, sin NaN ni inf en parches planos) y la
similitud se puntúa con la correlación cruzada normalizada calculada
explícitamente en la caja esperada: al restar las medias locales, la mezcla
del render (``opacidad * logo + (1 - opacidad) * fondo``) es una
transformación afín de la intensidad y puntúa igual con cualquier opacidad,
así que un logo semitransparente (opacidad mínima del contrato: 0.15) se
detecta igual que uno opaco sobre fondos planos. (El mapa de
``TM_CCOEFF_NORMED`` enmascarado de OpenCV es numéricamente inestable
—valores fuera de [-1, 1], ±inf— y su argmax global no es fiable; por eso
la NCC se calcula a mano en una sola caja.) Los parches planos del fondo
(varianza cero) no correlacionan: la muestra falla en vez de producir un
falso positivo.

Los templates uniformes (varianza ~0 bajo la máscara, p. ej. un logo
blanco plano) dejan a CCOEFF sin varianza que correlacionar: esas muestras
se verifican por contraste de borde en la posición esperada (media
interior del logo frente al anillo de fondo que lo rodea), con un umbral
que escala con la opacidad esperada del contrato. Sin watermark ambas
medias son el mismo fondo y el contraste es ~0.

Sobre fondos texturizados un template multicolor semitransparente
(opacidad < 0.7) no alcanza el umbral NCC de 0.75 aunque el logo esté
presente (NCC ~0.37 a opacidad 0.3 sobre ``testsrc2``, frente a ~0.0 sin
watermark). Para esos casos la muestra exige señal consistente del
contorno alfa: NCC sobre el suelo de textura más coincidencia de
gradientes (magnitud Sobel del frame frente a la del template, bajo la
máscara). Sin señal en algún frame (NCC bajo el suelo de ausencia o fuera
de zona) el resultado es ``fail``; con señal consistente pero sin llegar
al umbral fuerte el resultado es ``manual_review`` (revisión humana,
fail-closed). La 4.ª ruta de W1-bis permite un ``pass`` directo cuando la
opacidad es < 0.7 en un template no uniforme si la correlación NCC es
>= 0.22 y la coincidencia de bordes alfa (alpha-edge matching) alcanza
el umbral fuerte >= 0.90.

La muestra exige tres condiciones: similitud sobre el umbral (correlación
o contraste de borde según el template), posición en la zona del contrato
(con margen desde los bordes) y ancho relativo sobre ``min_width_ratio``.
El template se reescala al tamaño esperado del render (``scale_ratio`` del
ancho del frame): un logo más pequeño o más grande que el contratado no
correlaciona y falla.

Fail-closed: sin PNG resoluble, sin video legible, sin duración medible,
sin cv2 o con cualquier muestra no evaluable, el resultado es ``fail``
(jamás ``pass`` ni ``unsupported``); la ruta semitransparente devuelve
``manual_review`` (jamás ``pass`` silencioso sin contorno fuerte).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, assert_never, cast

from kliptych.assets import AssetError
from kliptych.contract import Watermark, WatermarkPosition
from kliptych.gate.models import CheckOutcome, CheckStatus, GateContext

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    import cv2
    import numpy as np

_PRESENT_FRACTIONS: tuple[float, ...] = (0.10, 0.50, 0.90)
_FULL_VIDEO_MIN_SAMPLES = 12
_FULL_VIDEO_MAX_SAMPLES = 120
_FULL_VIDEO_SAMPLES_PER_SECOND = 4
_MATCH_THRESHOLD = 0.75
_MARGIN = 20
_GRAY_IMAGE_DIMS = 2
_COLOR_IMAGE_DIMS = 3
_RGB_CHANNELS = 3
_RGBA_CHANNELS = 4
_UNIFORM_TEMPLATE_STD = 2.0
_EDGE_NOISE_FLOOR = 5.0
_EDGE_OPACITY_GAIN = 12.0
_EDGE_RING_PAD = 8
_NCC_MIN_DENOMINATOR = 1e-6
_NO_SIGNAL_NCC = 0.18
_TRANSLUCENT_MIN_NCC = 0.22
_TRANSLUCENT_MIN_EDGE = 0.35
_TRANSLUCENT_STRONG_EDGE = 0.90
_TRANSLUCENT_MAX_OPACITY = 0.7


class WatermarkError(Exception):
    """El watermark no se pudo evaluar en una muestra."""


@dataclass(frozen=True, slots=True)
class _Template:
    """PNG de referencia preparado para ``matchTemplate``."""

    gray: cv2.typing.MatLike
    mask: cv2.typing.MatLike | None
    width: int
    height: int


@dataclass(frozen=True, slots=True)
class _Ready:
    """Entradas verificadas para evaluar las muestras de un artefacto."""

    template: _Template
    duration: float
    video: Path
    config: Watermark


@dataclass(frozen=True, slots=True)
class Sample:
    """Resultado de evaluar una muestra temporal del video."""

    t_s: float
    matched: bool
    correlation: float
    x: int
    y: int
    expected_x: float
    expected_y: float
    reason: str | None = None
    metric: Literal["ccoeff", "edge"] = "ccoeff"
    edge_score: float | None = None

    def evidence(self) -> dict[str, object]:
        """Serializa la muestra para la evidencia del check.

        ``correlation`` guarda la correlación CCOEFF (métrica
        ``ccoeff``) o el contraste de borde en niveles de gris
        (métrica ``edge``, para templates uniformes). ``edge_score``
        guarda la coincidencia de gradientes del contorno alfa
        (métrica ``ccoeff`` sobre fondos texturizados).

        Returns:
            El dict con tiempo, correlación, posición detectada y esperada.
        """
        detail: dict[str, object] = {
            "t_s": round(self.t_s, 3),
            "matched": self.matched,
            "metric": self.metric,
            "correlation": round(self.correlation, 4),
            "loc": [self.x, self.y],
            "expected_loc": [round(self.expected_x, 1), round(self.expected_y, 1)],
        }
        if self.reason is not None:
            detail["reason"] = self.reason
        if self.edge_score is not None:
            detail["edge_score"] = round(self.edge_score, 4)
        return detail


def _pass(**evidence: object) -> CheckOutcome:
    return CheckOutcome(status=CheckStatus.PASS, evidence=dict(evidence))


def _fail(**evidence: object) -> CheckOutcome:
    return CheckOutcome(status=CheckStatus.FAIL, evidence=dict(evidence))


def _review(**evidence: object) -> CheckOutcome:
    return CheckOutcome(status=CheckStatus.MANUAL_REVIEW, evidence=dict(evidence))


def check_watermark_present(context: GateContext) -> CheckOutcome:
    """Verifica que el watermark aparezca al menos en una muestra del video.

    Args:
        context: Contexto resuelto del gate.

    Returns:
        PASS si alguna muestra correlaciona en la zona y tamaño exigidos;
        FAIL en caso contrario o ante cualquier fallo de evaluación.
    """
    return _check_watermark(context, full_video=False)


def check_watermark_full_video(context: GateContext) -> CheckOutcome:
    """Verifica que el watermark esté presente durante todo el video.

    Todas las muestras distribuidas por la duración deben correlacionar en
    la zona y tamaño exigidos; si el logo desaparece a mitad o al final del
    clip, el resultado es FAIL.

    Args:
        context: Contexto resuelto del gate.

    Returns:
        PASS si todas las muestras cumplen; FAIL en caso contrario o ante
        cualquier fallo de evaluación.
    """
    return _check_watermark(context, full_video=True)


def _check_watermark(context: GateContext, *, full_video: bool) -> CheckOutcome:
    rule = "watermark.full_video" if full_video else "watermark.present"
    if not context.contract.watermark.required:
        return _pass(rule=rule, reason="la campaña no exige watermark")
    ready = _prepare_evaluation(context, rule)
    if isinstance(ready, CheckOutcome):
        return ready
    if full_video:
        timestamps = _full_video_timestamps(ready.duration)
    else:
        timestamps = [fraction * ready.duration for fraction in _PRESENT_FRACTIONS]
    try:
        samples = _evaluate_samples(ready, timestamps)
    except WatermarkError as error:
        return _fail(rule=rule, reason=str(error))
    return _verdict(rule, ready.config, samples, full_video=full_video)


def _prepare_evaluation(context: GateContext, rule: str) -> _Ready | CheckOutcome:
    """Verifica las precondiciones del chequeo sin evaluar muestras.

    Args:
        context: Contexto resuelto del gate.
        rule: Id de la regla en evaluación, para la evidencia.

    Returns:
        Las entradas verificadas, o el FAIL fail-closed correspondiente.
    """
    try:
        template = _load_template(context)
    except WatermarkError as error:
        return _fail(rule=rule, reason=str(error))
    duration = context.media.duration_s if context.media is not None else None
    if duration is None or duration <= 0:
        return _fail(rule=rule, reason="no se pudo medir la duración del artefacto")
    video = context.piece.artifact_path
    if not video.is_file():
        return _fail(rule=rule, reason="el artefacto no existe; no se pudo verificar el watermark")
    return _Ready(
        template=template,
        duration=duration,
        video=video,
        config=context.contract.watermark,
    )


def _verdict(
    rule: str, config: Watermark, samples: list[Sample], *, full_video: bool
) -> CheckOutcome:
    """Deriva el veredicto de las muestras evaluadas.

    Además del paso fuerte (todas/alguna muestra sobre el umbral, según el
    modo), reconoce la ruta semitransparente sobre fondos texturizados:
    señal consistente del contorno sin llegar al umbral fuerte devuelve
    ``manual_review`` (o ``pass`` con contorno muy fuerte); sin señal en
    algún frame el resultado es ``fail``.

    Args:
        rule: Id de la regla en evaluación, para la evidencia.
        config: Watermark exigido por el contrato.
        samples: Muestras evaluadas, en orden temporal.
        full_video: Si todas deben cumplir (``full_video``) o basta una
            (``present``).

    Returns:
        PASS si se cumple el modo exigido; MANUAL_REVIEW con señal
        semitransparente consistente; FAIL en caso contrario.
    """
    matched = [sample for sample in samples if sample.matched]
    evidence: dict[str, object] = {
        "rule": rule,
        "mode": "all" if full_video else "any",
        "threshold": _MATCH_THRESHOLD,
        "opacity": config.opacity,
        "position": config.position.value,
        "scale_ratio": config.scale_ratio,
        "min_width_ratio": config.min_width_ratio,
        "samples": [sample.evidence() for sample in samples],
    }
    if full_video:
        return _verdict_full_video(config, samples, matched, evidence)
    return _verdict_present(config, samples, matched, evidence)


def _verdict_full_video(
    config: Watermark,
    samples: list[Sample],
    matched: list[Sample],
    evidence: dict[str, object],
) -> CheckOutcome:
    """Deriva el veredicto cuando el watermark debe cubrir todo el video.

    Args:
        config: Watermark exigido por el contrato.
        samples: Muestras evaluadas, en orden temporal.
        matched: Muestras que superaron la vía fuerte.
        evidence: Evidencia base ya serializada.

    Returns:
        PASS si todas las muestras cumplen; MANUAL_REVIEW (o PASS) con
        señal semitransparente consistente; FAIL en caso contrario.
    """
    if len(matched) == len(samples):
        return _pass(**evidence)
    translucent = _translucent_status(config, samples, full_video=True)
    if translucent is not None:
        return _translucent_outcome(translucent, evidence)
    missing = [sample.t_s for sample in samples if not sample.matched]
    return _fail(
        **evidence,
        missing_timestamps=[round(stamp, 3) for stamp in missing],
        reason="el watermark falta en alguna muestra del video",
    )


def _verdict_present(
    config: Watermark,
    samples: list[Sample],
    matched: list[Sample],
    evidence: dict[str, object],
) -> CheckOutcome:
    """Deriva el veredicto cuando basta con que el watermark aparezca una vez.

    Args:
        config: Watermark exigido por el contrato.
        samples: Muestras evaluadas, en orden temporal.
        matched: Muestras que superaron la vía fuerte.
        evidence: Evidencia base ya serializada.

    Returns:
        PASS si alguna muestra cumple; MANUAL_REVIEW (o PASS) con señal
        semitransparente; FAIL en caso contrario.
    """
    if matched:
        return _pass(
            **evidence,
            matched_timestamps=[round(sample.t_s, 3) for sample in matched],
        )
    translucent = _translucent_status(config, samples, full_video=False)
    if translucent is not None:
        return _translucent_outcome(translucent, evidence)
    return _fail(**evidence, reason="el watermark no aparece en ninguna muestra del video")


def _translucent_outcome(status: CheckStatus, evidence: dict[str, object]) -> CheckOutcome:
    """Construye el resultado de la ruta semitransparente.

    Args:
        status: PASS con contorno muy fuerte o MANUAL_REVIEW consistente.
        evidence: Evidencia base ya serializada.

    Returns:
        El PASS con decisión documentada o el MANUAL_REVIEW fail-closed.
    """
    if status is CheckStatus.PASS:
        return _pass(**evidence, decision="translucent-strong-edge")
    return _review(
        **evidence,
        decision="translucent-consistent",
        reason=(
            "el watermark semitransparente deja señal consistente del contorno "
            "sobre el fondo texturizado sin alcanzar el umbral fuerte; "
            "requiere revisión humana"
        ),
    )


def _translucent_status(
    config: Watermark, samples: list[Sample], *, full_video: bool
) -> CheckStatus | None:
    """Evalúa la ruta semitransparente sobre fondos texturizados.

    Solo aplica con templates estructurados (métrica ``ccoeff``) y opacidad
    contratada bajo ``_TRANSLUCENT_MAX_OPACITY``: en ese régimen la mezcla
    del render atenúa la NCC sin borrar los bordes del contorno alfa. Cada
    muestra con señal exige NCC sobre el suelo de textura y coincidencia
    de gradientes, ambas medidas en la caja esperada del contrato (esa
    medición localizada ya es la evidencia de posición: el argmin global
    de ``_locate`` no es fiable sobre fondos texturizados y no se usa
    aquí). Sin señal en algún frame (modo ``full_video``) o en todos
    (modo ``present``) no hay revisión: es ``fail`` aguas arriba.

    Args:
        config: Watermark exigido por el contrato.
        samples: Muestras evaluadas, en orden temporal.
        full_video: Si todas deben tener señal o basta una.

    Returns:
        PASS con contorno muy fuerte, MANUAL_REVIEW con señal consistente,
        o ``None`` si la ruta no aplica.
    """
    if config.opacity >= _TRANSLUCENT_MAX_OPACITY:
        return None
    if not samples or any(sample.metric != "ccoeff" for sample in samples):
        return None
    if full_video:
        return _translucent_all(samples)
    return _translucent_any(samples)


def _translucent_all(samples: list[Sample]) -> CheckStatus | None:
    """Evalúa la señal semitransparente cuando todas las muestras la exigen.

    Args:
        samples: Muestras evaluadas, en orden temporal.

    Returns:
        PASS si todas tienen contorno muy fuerte, MANUAL_REVIEW si todas
        tienen señal consistente, o ``None`` si alguna no da señal.
    """
    if any(not _has_translucent_signal(sample) for sample in samples):
        return None
    if all(_has_strong_edge(sample) for sample in samples):
        return CheckStatus.PASS
    return CheckStatus.MANUAL_REVIEW


def _translucent_any(samples: list[Sample]) -> CheckStatus | None:
    """Evalúa la señal semitransparente cuando basta una muestra con señal.

    Args:
        samples: Muestras evaluadas, en orden temporal.

    Returns:
        PASS si alguna señal tiene contorno muy fuerte, MANUAL_REVIEW si
        hay al menos una señal consistente, o ``None`` sin señales.
    """
    signals = [sample for sample in samples if _has_translucent_signal(sample)]
    if not signals:
        return None
    if any(_has_strong_edge(sample) for sample in signals):
        return CheckStatus.PASS
    return CheckStatus.MANUAL_REVIEW


def _has_strong_edge(sample: Sample) -> bool:
    """Indica si la coincidencia de gradientes supera el umbral estricto.

    Args:
        sample: Muestra con coincidencia de gradientes medida.

    Returns:
        True si el contorno alfa coincide con fuerza de ``pass`` directo.
    """
    return sample.edge_score is not None and sample.edge_score >= _TRANSLUCENT_STRONG_EDGE


def _has_translucent_signal(sample: Sample) -> bool:
    """Indica si una muestra conserva señal del contorno alfa en la caja esperada.

    Una muestra ya emparejada por la vía fuerte cuenta como señal: la ruta
    semitransparente solo relaja el umbral NCC, nunca la coincidencia del
    contorno. Sin watermark el fondo texturizado no correlaciona en la
    caja esperada (NCC ~0.0) ni coincide en gradientes (~0.0); con el logo
    en otra posición la caja esperada también contiene solo fondo y la
    muestra no da señal (``fail`` aguas arriba).

    Args:
        sample: Muestra con similitud y coincidencia de gradientes medidas
            en la caja esperada del contrato.

    Returns:
        True si la muestra tiene NCC sobre el suelo de textura y
        coincidencia de gradientes sobre su umbral, en la caja esperada.
    """
    if sample.metric != "ccoeff":
        return False
    if sample.correlation < _TRANSLUCENT_MIN_NCC:
        return False
    return sample.edge_score is not None and sample.edge_score >= _TRANSLUCENT_MIN_EDGE


def _load_template(context: GateContext) -> _Template:
    """Carga el PNG de referencia del registry con su máscara alfa.

    Args:
        context: Contexto resuelto del gate.

    Returns:
        El template en grises con su máscara (o ``None`` sin canal alfa).

    Raises:
        WatermarkError: Si la campaña no declara asset, el PNG no se puede
            resolver o leer, o cv2 no está disponible.
    """
    asset_id = context.contract.watermark.asset_id
    if asset_id is None:
        msg = "la campaña exige watermark pero no declara su asset_id"
        raise WatermarkError(msg)
    try:
        template_path = context.assets.path_for(asset_id)
    except (AssetError, OSError) as error:
        msg = f"no se pudo resolver el PNG del watermark '{asset_id}': {error}"
        raise WatermarkError(msg) from error
    if not template_path.is_file():
        msg = f"el PNG del watermark no existe: {template_path}"
        raise WatermarkError(msg)
    try:
        import cv2
    except ImportError as error:
        msg = "opencv (cv2) no está disponible; no se pudo verificar el watermark"
        raise WatermarkError(msg) from error
    image = cv2.imread(str(template_path), cv2.IMREAD_UNCHANGED)
    if image is None:
        msg = f"el PNG del watermark no se pudo decodificar: {template_path}"
        raise WatermarkError(msg)
    shape = cast("tuple[int, ...]", image.shape)
    if image.ndim == _GRAY_IMAGE_DIMS:
        return _Template(gray=image, mask=None, width=shape[1], height=shape[0])
    channels = shape[2] if image.ndim == _COLOR_IMAGE_DIMS else 0
    if channels == _RGBA_CHANNELS:
        split = cv2.split(image)
        gray = cv2.cvtColor(cv2.merge(split[0:3]), cv2.COLOR_BGR2GRAY)
        return _Template(gray=gray, mask=split[3], width=shape[1], height=shape[0])
    if channels == _RGB_CHANNELS:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return _Template(gray=gray, mask=None, width=shape[1], height=shape[0])
    msg = f"el PNG del watermark tiene un formato no soportado: {template_path}"
    raise WatermarkError(msg)


def _full_video_timestamps(duration: float) -> list[float]:
    """Segundos a muestrear para ``watermark.full_video``, con densidad temporal.

    Cinco fracciones fijas (10/30/50/70/90%) dejan huecos ciegos: un bache
    de 0.3 s entre dos fracciones pasaba inadvertido. En su lugar se
    muestrean ``_FULL_VIDEO_SAMPLES_PER_SECOND`` instantes por segundo
    (paso <= 0.25 s), centrados en celdas uniformes de principio a fin del
    video, con un mínimo de muestras para clips cortos y un tope para
    clips largos. Todo bache de al menos un paso de muestreo contiene una
    muestra y falla.

    Args:
        duration: Duración medida del artefacto, en segundos.

    Returns:
        Los segundos a muestrear, en orden temporal.
    """
    count = min(
        _FULL_VIDEO_MAX_SAMPLES,
        max(_FULL_VIDEO_MIN_SAMPLES, math.ceil(duration * _FULL_VIDEO_SAMPLES_PER_SECOND)),
    )
    return [duration * (index + 0.5) / count for index in range(count)]


def _evaluate_samples(ready: _Ready, timestamps: Sequence[float]) -> list[Sample]:
    """Evalúa las muestras temporales del video con una sola captura.

    Abre el artefacto una vez con ``cv2.VideoCapture`` y busca cada
    instante (un seek por muestra, sin procesos ffmpeg por frame): un clip
    de 20 s con 80 muestras se evalúa en segundos en vez de minutos.

    Args:
        ready: Entradas verificadas de la evaluación.
        timestamps: Segundos del video a muestrear, en orden temporal.

    Returns:
        Una muestra por instante, en orden temporal.
    """
    capture = _open_capture(ready.video)
    try:
        samples: list[Sample] = []
        for t_s in timestamps:
            frame = _read_frame_at(capture, ready.video, t_s)
            shape = cast("tuple[int, ...]", frame.shape)
            samples.append(_match_sample(frame, (shape[1], shape[0]), ready, t_s))
        return samples
    finally:
        _ = capture.release()


def _open_capture(video: Path) -> cv2.VideoCapture:
    """Abre el artefacto para lectura de frames con seek (fail-closed).

    Args:
        video: Ruta del artefacto a verificar.

    Returns:
        La captura abierta, lista para buscar instantes.

    Raises:
        WatermarkError: Si cv2 no está disponible o el video no se pudo abrir.
    """
    import cv2

    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        msg = f"el video no se pudo abrir para lectura de frames: {video}"
        raise WatermarkError(msg)
    return capture


def _read_frame_at(capture: cv2.VideoCapture, video: Path, t_s: float) -> cv2.typing.MatLike:
    """Lee el frame de un instante con seek sobre la captura abierta.

    Args:
        capture: Captura abierta del artefacto.
        video: Ruta del artefacto, para los errores.
        t_s: Segundo del video a muestrear.

    Returns:
        El frame en grises ya decodificado.

    Raises:
        WatermarkError: Si el seek falla, el frame no se pudo leer o quedó vacío.
    """
    import cv2

    if not capture.set(cv2.CAP_PROP_POS_MSEC, t_s * 1000.0):
        msg = f"no se pudo buscar t={t_s:.3f}s en {video}"
        raise WatermarkError(msg)
    ok, frame = capture.read()
    decoded = cast("cv2.typing.MatLike | None", frame)
    if not ok or decoded is None:
        msg = f"el frame en t={t_s:.3f}s no se pudo leer de {video}"
        raise WatermarkError(msg)
    gray = (
        cv2.cvtColor(decoded, cv2.COLOR_BGR2GRAY) if decoded.ndim == _COLOR_IMAGE_DIMS else decoded
    )
    shape = cast("tuple[int, ...]", gray.shape)
    if len(shape) < _GRAY_IMAGE_DIMS or shape[0] <= 0 or shape[1] <= 0:
        msg = f"el frame en t={t_s:.3f}s quedó vacío tras decodificar"
        raise WatermarkError(msg)
    return gray


@dataclass(frozen=True, slots=True)
class _Expected:
    """Caja esperada del render para una muestra, con su instante."""

    frame_w: int
    frame_h: int
    width: int
    height: int
    x: float
    y: float
    t_s: float


def _match_sample(
    frame: cv2.typing.MatLike, frame_size: tuple[int, int], ready: _Ready, t_s: float
) -> Sample:
    """Compara un frame decodificado contra el template y veredea la muestra.

    La opacidad esperada viaja en ``ready.config`` (contrato): con
    templates estructurados la NCC enmascarada ya es invariante a ella
    (mezcla afín); con templates uniformes el contraste de borde se exige
    contra un umbral que escala con esa opacidad.

    Args:
        frame: Frame en grises ya decodificado.
        frame_size: (ancho, alto) del frame, en píxeles.
        ready: Entradas verificadas de la evaluación.
        t_s: Segundo del video muestreado, para la evidencia.

    Returns:
        La muestra con su veredicto y evidencia.

    Raises:
        WatermarkError: Si cv2 no está disponible, el tamaño esperado es
            degenerado o la máscara quedó vacía.
    """
    frame_w, frame_h = frame_size
    expected_w, expected_h = _expected_template_size(
        frame_w, ready.template.width, ready.template.height, ready.config.scale_ratio
    )
    if expected_w <= 0 or expected_h <= 0 or expected_w > frame_w or expected_h > frame_h:
        msg = f"tamaño esperado degenerado {expected_w}x{expected_h} en t={t_s:.3f}s"
        raise WatermarkError(msg)
    resized, mask = _resized_template(ready.template, expected_w, expected_h)
    expected = _Expected(
        frame_w,
        frame_h,
        expected_w,
        expected_h,
        *expected_top_left(frame_w, frame_h, expected_w, expected_h, ready.config.position),
        t_s,
    )
    if _template_std(resized, mask) < _UNIFORM_TEMPLATE_STD:
        sample = _match_uniform(frame, resized, mask, expected)
    else:
        sample = _match_structured(frame, resized, mask, expected)
    failure = _sample_failure(sample, frame_w, frame_h, expected_w, ready.config)
    if failure is None:
        return sample
    return Sample(
        t_s=sample.t_s,
        matched=False,
        correlation=sample.correlation,
        x=sample.x,
        y=sample.y,
        expected_x=sample.expected_x,
        expected_y=sample.expected_y,
        reason=failure,
        metric=sample.metric,
        edge_score=sample.edge_score,
    )


def _match_uniform(
    frame: cv2.typing.MatLike,
    resized: cv2.typing.MatLike,
    mask: cv2.typing.MatLike | None,
    expected: _Expected,
) -> Sample:
    """Verifica un template uniforme por contraste de borde en la caja esperada.

    Args:
        frame: Frame en grises ya decodificado.
        resized: Template reescalado al tamaño esperado del render.
        mask: Máscara alfa reescalada (o ``None`` sin canal alfa).
        expected: Caja esperada del render con su instante.

    Returns:
        La muestra con el contraste de borde como similitud.
    """
    contrast = _edge_contrast(frame, resized, mask, round(expected.x), round(expected.y))
    return Sample(
        t_s=expected.t_s,
        matched=True,
        correlation=0.0 if contrast is None else contrast,
        x=round(expected.x),
        y=round(expected.y),
        expected_x=expected.x,
        expected_y=expected.y,
        metric="edge",
    )


def _match_structured(
    frame: cv2.typing.MatLike,
    resized: cv2.typing.MatLike,
    mask: cv2.typing.MatLike | None,
    expected: _Expected,
) -> Sample:
    """Verifica un template estructurado: NCC y gradientes en la caja esperada.

    Args:
        frame: Frame en grises ya decodificado.
        resized: Template reescalado al tamaño esperado del render.
        mask: Máscara alfa reescalada (o ``None`` sin canal alfa).
        expected: Caja esperada del render con su instante.

    Returns:
        La muestra con NCC, coincidencia de gradientes y posición detectada.
    """
    x, y = _locate(frame, resized, mask, expected.t_s)
    score = _masked_ncc(frame, resized, mask, round(expected.x), round(expected.y))
    edge = _gradient_match(frame, resized, mask, round(expected.x), round(expected.y))
    return Sample(
        t_s=expected.t_s,
        matched=True,
        correlation=0.0 if score is None else score,
        x=x,
        y=y,
        expected_x=expected.x,
        expected_y=expected.y,
        metric="ccoeff",
        edge_score=edge,
    )


def _locate(
    frame: cv2.typing.MatLike,
    resized: cv2.typing.MatLike,
    mask: cv2.typing.MatLike | None,
    t_s: float,
) -> tuple[int, int]:
    """Localiza el template en el frame con SQDIFF enmascarado.

    SQDIFF crudo suma diferencias cuadráticas sin normalizar por energía
    local: no produce NaN ni inf en parches planos u oscuros y su argmin
    es estable. Solo ubica la detección (criterio de zona); la puntuación
    invariante a la opacidad la calcula ``_masked_ncc`` en la caja
    esperada.

    Args:
        frame: Frame en grises ya decodificado.
        resized: Template reescalado al tamaño esperado del render.
        mask: Máscara alfa reescalada (o ``None`` sin canal alfa).
        t_s: Segundo del video muestreado, para los errores.

    Returns:
        La (x, y) del mejor ajuste.

    Raises:
        WatermarkError: Si cv2 no está disponible o la máscara quedó vacía.
    """
    import cv2

    match = cv2.matchTemplate(frame, resized, cv2.TM_SQDIFF, mask=mask)
    count = cv2.countNonZero(mask) if mask is not None else _pixel_count(resized)
    if count <= 0:
        msg = f"máscara vacía en t={t_s:.3f}s"
        raise WatermarkError(msg)
    _, _, location, _ = cv2.minMaxLoc(match)
    return int(location[0]), int(location[1])


def _masked_ncc(
    frame: cv2.typing.MatLike,
    resized: cv2.typing.MatLike,
    mask: cv2.typing.MatLike | None,
    x: int,
    y: int,
) -> float | None:
    """Correlación cruzada normalizada en la caja esperada, con máscara.

    Se calcula explícitamente (medias y covarianza con ``meanStdDev`` y
    el producto en float32) en vez de leer el mapa de
    ``TM_CCOEFF_NORMED``: la variante enmascarada de OpenCV es
    numéricamente inestable (valores fuera de [-1, 1], ±inf) y su argmax
    global no es fiable. Al restar las medias locales, la mezcla afín del
    render (``opacidad * logo + (1 - opacidad) * fondo``) puntúa igual con
    cualquier opacidad.

    Args:
        frame: Frame en grises ya decodificado.
        resized: Template reescalado al tamaño esperado del render.
        mask: Máscara alfa reescalada (o ``None`` sin canal alfa).
        x: Columna esperada de la esquina superior izquierda.
        y: Fila esperada de la esquina superior izquierda.

    Returns:
        La NCC en [-1, 1], o ``None`` si la caja quedó degenerada o sin
        varianza medible (fail-closed aguas arriba).
    """
    frame_array = cast("np.ndarray[tuple[int, ...], np.dtype[np.uint8]]", frame)
    template_array = cast("np.ndarray[tuple[int, ...], np.dtype[np.uint8]]", resized)
    mask_array = (
        None if mask is None else cast("np.ndarray[tuple[int, ...], np.dtype[np.uint8]]", mask)
    )
    frame_h, frame_w = frame_array.shape[0], frame_array.shape[1]
    box = _clamp_box((frame_w, frame_h), (template_array.shape[1], template_array.shape[0]), x, y)
    if box is None:
        return None
    x0, y0, x1, y1 = box
    patch = frame_array[y0:y1, x0:x1]
    template_crop = template_array[y0 - y : y0 - y + (y1 - y0), x0 - x : x0 - x + (x1 - x0)]
    mask_crop = (
        None
        if mask_array is None
        else mask_array[y0 - y : y0 - y + (y1 - y0), x0 - x : x0 - x + (x1 - x0)]
    )
    return _crops_ncc(patch, template_crop, mask_crop)


def _clamp_box(
    frame_size: tuple[int, int], box_size: tuple[int, int], x: int, y: int
) -> tuple[int, int, int, int] | None:
    """Recorta la caja esperada al frame.

    Args:
        frame_size: (ancho, alto) del frame, en píxeles.
        box_size: (ancho, alto) de la caja, en píxeles.
        x: Columna esperada de la esquina superior izquierda.
        y: Fila esperada de la esquina superior izquierda.

    Returns:
        La (x0, y0, x1, y1) recortada, o ``None`` si quedó degenerada.
    """
    frame_w, frame_h = frame_size
    box_w, box_h = box_size
    x0 = min(max(x, 0), frame_w - 1)
    y0 = min(max(y, 0), frame_h - 1)
    x1 = min(x0 + box_w, frame_w)
    y1 = min(y0 + box_h, frame_h)
    if x1 <= x0 or y1 <= y0:
        return None
    return (x0, y0, x1, y1)


def _crops_ncc(
    patch: cv2.typing.MatLike,
    template: cv2.typing.MatLike,
    mask: cv2.typing.MatLike | None,
) -> float | None:
    """NCC enmascarada entre dos recortes del mismo tamaño.

    Args:
        patch: Recorte del frame en grises.
        template: Recorte del template en grises.
        mask: Máscara alfa del recorte (o ``None`` sin canal alfa).

    Returns:
        La NCC en [-1, 1], o ``None`` sin varianza medible en el
        denominador.
    """
    import cv2

    template_mean, template_std = cv2.meanStdDev(template, mask=mask)
    frame_mean, frame_std = cv2.meanStdDev(patch, mask=mask)
    denominator = _scalar(template_std) * _scalar(frame_std)
    if denominator < _NCC_MIN_DENOMINATOR:
        return None
    product = cv2.multiply(patch, template, dtype=cv2.CV_32F)
    covariance = cv2.mean(product, mask=mask)[0] - _scalar(template_mean) * _scalar(frame_mean)
    return max(-1.0, min(1.0, covariance / denominator))


def _gradient_match(
    frame: cv2.typing.MatLike,
    resized: cv2.typing.MatLike,
    mask: cv2.typing.MatLike | None,
    x: int,
    y: int,
) -> float | None:
    """Coincidencia de gradientes del contorno alfa en la caja esperada.

    Compara la magnitud Sobel del frame con la del template (NCC bajo la
    máscara): la mezcla semitransparente del render atenúa la correlación
    de intensidades sobre fondos texturizados pero conserva los bordes del
    contorno alfa, mientras que el fondo solo no guarda relación con esos
    bordes (gradiente ~0.0 frente a >= 0.70 con watermark a opacidad 0.3
    sobre ``testsrc2``).

    Args:
        frame: Frame en grises ya decodificado.
        resized: Template reescalado al tamaño esperado del render.
        mask: Máscara alfa reescalada (o ``None`` sin canal alfa).
        x: Columna esperada de la esquina superior izquierda.
        y: Fila esperada de la esquina superior izquierda.

    Returns:
        La NCC de magnitudes de gradiente en [-1, 1], o ``None`` si la
        caja quedó degenerada o sin varianza medible.
    """
    frame_array = cast("np.ndarray[tuple[int, ...], np.dtype[np.uint8]]", frame)
    template_array = cast("np.ndarray[tuple[int, ...], np.dtype[np.uint8]]", resized)
    mask_array = (
        None if mask is None else cast("np.ndarray[tuple[int, ...], np.dtype[np.uint8]]", mask)
    )
    frame_h, frame_w = frame_array.shape[0], frame_array.shape[1]
    box = _clamp_box((frame_w, frame_h), (template_array.shape[1], template_array.shape[0]), x, y)
    if box is None:
        return None
    x0, y0, x1, y1 = box
    patch = frame_array[y0:y1, x0:x1]
    template_crop = template_array[y0 - y : y0 - y + (y1 - y0), x0 - x : x0 - x + (x1 - x0)]
    mask_crop = (
        None
        if mask_array is None
        else mask_array[y0 - y : y0 - y + (y1 - y0), x0 - x : x0 - x + (x1 - x0)]
    )
    return _crops_ncc(_gradient_magnitude(patch), _gradient_magnitude(template_crop), mask_crop)


def _gradient_magnitude(image: cv2.typing.MatLike) -> cv2.typing.MatLike:
    """Magnitud del gradiente Sobel de una imagen en grises.

    Args:
        image: Imagen de un solo canal.

    Returns:
        La magnitud ``sqrt(gx² + gy²)`` en float32, del mismo tamaño.
    """
    import cv2

    grad_x = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3)
    return cv2.magnitude(grad_x, grad_y)


def _scalar(matrix: cv2.typing.MatLike) -> float:
    """Extrae el escalar de una matriz 1x1 de OpenCV.

    Indexar ``MatLike`` (unión de ``Mat`` y ``ndarray``) degrada a ``Any``
    en los stubs: el cast a ndarray tipado deja el escalar con tipo
    declarado para el chequeo estricto.

    Args:
        matrix: Matriz 1x1 (media o desviación de ``meanStdDev``).

    Returns:
        El escalar como float.
    """
    array = cast("np.ndarray[tuple[int, ...], np.dtype[np.float64]]", matrix)
    return cast("float", array[0, 0])


def _pixel_count(image: cv2.typing.MatLike) -> int:
    """Cuenta los píxeles de una imagen en grises.

    Args:
        image: Imagen de un solo canal.

    Returns:
        El número de píxeles (ancho por alto).
    """
    shape = cast("tuple[int, ...]", image.shape)
    return shape[1] * shape[0]


def _template_std(resized: cv2.typing.MatLike, mask: cv2.typing.MatLike | None) -> float:
    """Mide la desviación del template bajo la máscara, en niveles de gris.

    Un template uniforme (p. ej. un logo blanco plano) deja a CCOEFF sin
    varianza que correlacionar: por debajo de ``_UNIFORM_TEMPLATE_STD`` la
    muestra se verifica por contraste de borde en vez de por correlación.

    Args:
        resized: Template reescalado al tamaño esperado del render.
        mask: Máscara alfa reescalada (o ``None`` sin canal alfa).

    Returns:
        La desviación estándar de los píxeles visibles del template.
    """
    import cv2

    _, stddev = cv2.meanStdDev(resized, mask=mask)
    return _scalar(stddev)


def _edge_contrast(
    frame: cv2.typing.MatLike,
    resized: cv2.typing.MatLike,
    mask: cv2.typing.MatLike | None,
    x: int,
    y: int,
) -> float | None:
    """Mide el contraste interior/exterior del logo en la posición esperada.

    Compara la media interior del logo (bajo la máscara erosionada, o de
    toda la caja sin canal alfa) con la media del anillo de fondo que lo
    rodea: sin watermark ambas son el mismo fondo y el contraste es ~0
    (ruido de compresión); con watermark difieren en ``opacidad * |logo -
    fondo|``. El anillo se calcula por diferencia de sumas para no
    construir máscaras del tamaño del frame.

    Args:
        frame: Frame en grises ya decodificado.
        resized: Template reescalado al tamaño esperado del render.
        mask: Máscara alfa reescalada (o ``None`` sin canal alfa).
        x: Columna esperada de la esquina superior izquierda.
        y: Fila esperada de la esquina superior izquierda.

    Returns:
        El |interior - exterior| en niveles de gris, o ``None`` si la caja
        quedó degenerada, la máscara vacía o el anillo vacío (fail-closed
        aguas arriba).
    """
    frame_array = cast("np.ndarray[tuple[int, ...], np.dtype[np.uint8]]", frame)
    template_array = cast("np.ndarray[tuple[int, ...], np.dtype[np.uint8]]", resized)
    mask_array = (
        None if mask is None else cast("np.ndarray[tuple[int, ...], np.dtype[np.uint8]]", mask)
    )
    frame_h, frame_w = frame_array.shape[0], frame_array.shape[1]
    box = _clamp_box((frame_w, frame_h), (template_array.shape[1], template_array.shape[0]), x, y)
    if box is None:
        return None
    x0, y0, x1, y1 = box
    crop = frame_array[y0:y1, x0:x1]
    interior = _interior_mean(
        crop,
        None if mask_array is None else mask_array[0 : y1 - y0, 0 : x1 - x0],
    )
    if interior is None:
        return None
    exterior = _surrounding_mean(frame_array, x0, y0, x1, y1)
    if exterior is None:
        return None
    return abs(interior - exterior)


def _interior_mean(
    crop: np.ndarray[tuple[int, ...], np.dtype[np.uint8]],
    mask: np.ndarray[tuple[int, ...], np.dtype[np.uint8]] | None,
) -> float | None:
    """Media del interior del logo en un recorte.

    Bajo la máscara erosionada cuando hay canal alfa (el borde
    semitransparente del reescalado no contamina la media); de todo el
    recorte sin canal alfa.

    Args:
        crop: Recorte del frame en la caja esperada.
        mask: Máscara alfa del recorte (o ``None`` sin canal alfa).

    Returns:
        La media interior, o ``None`` si la máscara quedó vacía.
    """
    import cv2

    if mask is None:
        return cv2.mean(crop)[0]
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    inner = cv2.erode(mask, kernel)
    if cv2.countNonZero(inner) == 0:
        inner = mask
    if cv2.countNonZero(inner) == 0:
        return None
    return cv2.mean(crop, mask=inner)[0]


def _surrounding_mean(
    frame: np.ndarray[tuple[int, ...], np.dtype[np.uint8]],
    x0: int,
    y0: int,
    x1: int,
    y1: int,
) -> float | None:
    """Media del anillo de fondo alrededor de la caja (x0, y0, x1, y1).

    Por diferencia de sumas —suma de la vecindad acolchada menos suma de
    la caja, entre los píxeles del anillo— para no construir máscaras del
    tamaño del frame.

    Args:
        frame: Frame en grises ya decodificado.
        x0: Columna izquierda de la caja, recortada al frame.
        y0: Fila superior de la caja, recortada al frame.
        x1: Columna derecha (excluida) de la caja, recortada al frame.
        y1: Fila inferior (excluida) de la caja, recortada al frame.

    Returns:
        La media del anillo, o ``None`` si quedó vacío.
    """
    import cv2

    pad = _EDGE_RING_PAD
    frame_h, frame_w = frame.shape[0], frame.shape[1]
    outer = frame[
        max(y0 - pad, 0) : min(y1 + pad, frame_h),
        max(x0 - pad, 0) : min(x1 + pad, frame_w),
    ]
    inner = frame[y0:y1, x0:x1]
    ring_pixels = int(outer.size) - int(inner.size)
    if ring_pixels <= 0:
        return None
    return (cv2.sumElems(outer)[0] - cv2.sumElems(inner)[0]) / ring_pixels


def _edge_threshold(opacity: float) -> float:
    """Contraste de borde mínimo exigido, según la opacidad esperada.

    El borde visible escala con la opacidad (``opacidad * |logo -
    fondo|``): a menor opacidad se exige menos, pero nunca por debajo del
    suelo de ruido de compresión.

    Args:
        opacity: Opacidad del watermark según el contrato.

    Returns:
        El contraste mínimo en niveles de gris.
    """
    return max(_EDGE_NOISE_FLOOR, _EDGE_OPACITY_GAIN * opacity)


def _sample_failure(
    sample: Sample, frame_w: int, frame_h: int, expected_w: int, config: Watermark
) -> str | None:
    """Aplica los tres criterios de la muestra en orden de evidencia.

    Args:
        sample: Muestra con similitud y posición detectada.
        frame_w: Ancho del frame, en píxeles.
        frame_h: Alto del frame, en píxeles.
        expected_w: Ancho esperado del logo, en píxeles.
        config: Watermark exigido por el contrato.

    Returns:
        El motivo del rechazo, o ``None`` si la muestra cumple similitud,
        tamaño mínimo y zona.
    """
    similarity = _similarity_failure(sample, config)
    if similarity is not None:
        return similarity
    ratio = expected_w / frame_w
    if ratio < config.min_width_ratio:
        return f"ancho relativo {ratio:.3f} bajo el mínimo {config.min_width_ratio}"
    if not sample_within_zone(sample, frame_w, frame_h):
        return "el watermark no está en la zona exigida por el contrato"
    return None


def _similarity_failure(sample: Sample, config: Watermark) -> str | None:
    """Aplica el criterio de similitud según la métrica de la muestra.

    La correlación CCOEFF es invariante a la opacidad (mezcla afín) y se
    exige contra el umbral fijo; el contraste de borde de templates
    uniformes escala con la opacidad y se exige contra un umbral que
    escala con la opacidad esperada del contrato.

    Args:
        sample: Muestra con similitud y métrica (``ccoeff`` o ``edge``).
        config: Watermark exigido por el contrato.

    Returns:
        El motivo del rechazo por similitud, o ``None`` si la supera.
    """
    if sample.metric == "edge":
        threshold = _edge_threshold(config.opacity)
        if sample.correlation < threshold:
            return (
                f"contraste de borde {sample.correlation:.1f} bajo el umbral "
                f"{threshold:.1f} (opacidad {config.opacity})"
            )
        return None
    if sample.correlation < _MATCH_THRESHOLD:
        return f"correlación {sample.correlation:.3f} bajo el umbral {_MATCH_THRESHOLD}"
    return None


def _resized_template(
    template: _Template, width: int, height: int
) -> tuple[cv2.typing.MatLike, cv2.typing.MatLike | None]:
    """Reescala el template al tamaño esperado del render.

    Args:
        template: PNG de referencia ya cargado.
        width: Ancho esperado en el frame, en píxeles.
        height: Alto esperado en el frame, en píxeles.

    Returns:
        El template y su máscara reescalados (la máscara puede ser None).
    """
    import cv2

    shrink = width < template.width or height < template.height
    interpolation = cv2.INTER_AREA if shrink else cv2.INTER_LINEAR
    resized = cv2.resize(template.gray, (width, height), interpolation=interpolation)
    mask = (
        None
        if template.mask is None
        else cv2.resize(template.mask, (width, height), interpolation=interpolation)
    )
    return resized, mask


def _expected_template_size(
    frame_w: int, template_w: int, template_h: int, scale_ratio: float
) -> tuple[int, int]:
    """Calcula el tamaño esperado del logo con la misma aritmética del render.

    Replica el ``scale=W:-2`` del ensamblado (ancho objetivo par por
    truncado, alto con el aspecto del PNG): si el render usó otro tamaño,
    la similitud cae y la muestra falla.

    Args:
        frame_w: Ancho del frame, en píxeles.
        template_w: Ancho nativo del PNG, en píxeles.
        template_h: Alto nativo del PNG, en píxeles.
        scale_ratio: Ancho relativo exigido por el contrato.

    Returns:
        El (ancho, alto) esperado en el frame, en píxeles pares.
    """
    width = int(math.trunc(frame_w * scale_ratio / 2) * 2)
    height = int(math.trunc(width * template_h / template_w / 2) * 2)
    return width, height


def expected_top_left(
    frame_w: int,
    frame_h: int,
    template_w: int,
    template_h: int,
    position: WatermarkPosition,
) -> tuple[float, float]:
    """Calcula la esquina superior izquierda exigida para una posición.

    Replica el ``overlay`` del ensamblado con el mismo margen de seguridad.

    Args:
        frame_w: Ancho del frame, en píxeles.
        frame_h: Alto del frame, en píxeles.
        template_w: Ancho esperado del logo, en píxeles.
        template_h: Alto esperado del logo, en píxeles.
        position: Zona del lienzo exigida por el contrato.

    Returns:
        Las coordenadas (x, y) esperadas de la esquina superior izquierda.
    """
    if position is WatermarkPosition.TOP_LEFT:
        top_left = (float(_MARGIN), float(_MARGIN))
    elif position is WatermarkPosition.TOP_RIGHT:
        top_left = (float(frame_w - template_w - _MARGIN), float(_MARGIN))
    elif position is WatermarkPosition.BOTTOM_LEFT:
        top_left = (float(_MARGIN), float(frame_h - template_h - _MARGIN))
    elif position is WatermarkPosition.BOTTOM_RIGHT:
        top_left = (float(frame_w - template_w - _MARGIN), float(frame_h - template_h - _MARGIN))
    elif position is WatermarkPosition.CENTER:
        top_left = ((frame_w - template_w) / 2, (frame_h - template_h) / 2)
    elif position is WatermarkPosition.CENTER_TOP:
        top_left = ((frame_w - template_w) / 2, float(_MARGIN))
    elif position is WatermarkPosition.CENTER_BOTTOM:
        top_left = ((frame_w - template_w) / 2, float(frame_h - template_h - _MARGIN))
    else:
        assert_never(position)
    return top_left


def sample_within_zone(sample: Sample, frame_w: int, frame_h: int) -> bool:
    """Indica si la detección cae en la zona exigida, con tolerancia de render.

    Args:
        sample: Muestra con posición detectada y esperada.
        frame_w: Ancho del frame, en píxeles.
        frame_h: Alto del frame, en píxeles.

    Returns:
        True si ambas coordenadas están dentro de la tolerancia (2% de la
        dimensión, mínimo 8 px).
    """
    tolerance_x = max(8.0, 0.02 * frame_w)
    tolerance_y = max(8.0, 0.02 * frame_h)
    return abs(sample.x - sample.expected_x) <= tolerance_x and (
        abs(sample.y - sample.expected_y) <= tolerance_y
    )
