"""Validadores mecánicos del watermark con OpenCV, indexados por rule_id.

- ``watermark.present``: el logo aparece al menos en una muestra del video.
- ``watermark.full_video``: el logo aparece en todas las muestras,
  distribuidas por la duración (10/30/50/70/90%): si desaparece a mitad o al
  final del clip, el resultado es ``fail``.

Cada muestra extrae un frame con ffmpeg (lista de argumentos, sin shell;
``-ss`` después de ``-i`` para seek exacto) y lo compara con el PNG de
referencia mediante ``cv2.matchTemplate`` (con máscara alfa cuando el PNG
la trae). La muestra exige tres condiciones: correlación sobre el umbral,
posición en la zona del contrato (con margen desde los bordes) y ancho
relativo sobre ``min_width_ratio``. El template se reescala al tamaño
esperado del render (``scale_ratio`` del ancho del frame): un logo más
pequeño o más grande que el contratado no correlaciona y falla.

Fail-closed: sin PNG resoluble, sin video legible, sin duración medible,
sin ffmpeg/cv2 o con cualquier muestra no evaluable, el resultado es
``fail`` (jamás ``pass`` ni ``unsupported``).
"""

from __future__ import annotations

import math
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, assert_never, cast

from kliptych.assets import AssetError
from kliptych.contract import Watermark, WatermarkPosition
from kliptych.gate.models import CheckOutcome, CheckStatus, GateContext

if TYPE_CHECKING:
    import cv2

_PRESENT_FRACTIONS: tuple[float, ...] = (0.10, 0.50, 0.90)
_FULL_VIDEO_FRACTIONS: tuple[float, ...] = (0.10, 0.30, 0.50, 0.70, 0.90)
_MATCH_THRESHOLD = 0.75
_MARGIN = 20
_FRAME_TIMEOUT_S = 30.0
_MAX_PIXEL_SQDIFF = 255.0 * 255.0
_GRAY_IMAGE_DIMS = 2
_COLOR_IMAGE_DIMS = 3
_RGB_CHANNELS = 3
_RGBA_CHANNELS = 4


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
    ffmpeg: str
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

    def evidence(self) -> dict[str, object]:
        """Serializa la muestra para la evidencia del check.

        Returns:
            El dict con tiempo, correlación, posición detectada y esperada.
        """
        detail: dict[str, object] = {
            "t_s": round(self.t_s, 3),
            "matched": self.matched,
            "correlation": round(self.correlation, 4),
            "loc": [self.x, self.y],
            "expected_loc": [round(self.expected_x, 1), round(self.expected_y, 1)],
        }
        if self.reason is not None:
            detail["reason"] = self.reason
        return detail


def _pass(**evidence: object) -> CheckOutcome:
    return CheckOutcome(status=CheckStatus.PASS, evidence=dict(evidence))


def _fail(**evidence: object) -> CheckOutcome:
    return CheckOutcome(status=CheckStatus.FAIL, evidence=dict(evidence))


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
    fractions = _FULL_VIDEO_FRACTIONS if full_video else _PRESENT_FRACTIONS
    try:
        samples = _evaluate_samples(ready, fractions)
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
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        return _fail(rule=rule, reason="ffmpeg no está disponible; no se pudo extraer frames")
    return _Ready(
        template=template,
        duration=duration,
        video=video,
        ffmpeg=ffmpeg,
        config=context.contract.watermark,
    )


def _verdict(
    rule: str, config: Watermark, samples: list[Sample], *, full_video: bool
) -> CheckOutcome:
    """Deriva el veredicto de las muestras evaluadas.

    Args:
        rule: Id de la regla en evaluación, para la evidencia.
        config: Watermark exigido por el contrato.
        samples: Muestras evaluadas, en orden temporal.
        full_video: Si todas deben cumplir (``full_video``) o basta una
            (``present``).

    Returns:
        PASS si se cumple el modo exigido; FAIL en caso contrario.
    """
    matched = [sample for sample in samples if sample.matched]
    evidence: dict[str, object] = {
        "rule": rule,
        "mode": "all" if full_video else "any",
        "threshold": _MATCH_THRESHOLD,
        "position": config.position.value,
        "scale_ratio": config.scale_ratio,
        "min_width_ratio": config.min_width_ratio,
        "samples": [sample.evidence() for sample in samples],
    }
    if full_video:
        if len(matched) == len(samples):
            return _pass(**evidence)
        missing = [sample.t_s for sample in samples if not sample.matched]
        return _fail(
            **evidence,
            missing_timestamps=[round(stamp, 3) for stamp in missing],
            reason="el watermark falta en alguna muestra del video",
        )
    if matched:
        return _pass(
            **evidence,
            matched_timestamps=[round(sample.t_s, 3) for sample in matched],
        )
    return _fail(**evidence, reason="el watermark no aparece en ninguna muestra del video")


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


def _evaluate_samples(ready: _Ready, fractions: tuple[float, ...]) -> list[Sample]:
    """Evalúa las muestras temporales distribuidas por la duración.

    Args:
        ready: Entradas verificadas de la evaluación.
        fractions: Fracciones de la duración a muestrear.

    Returns:
        Una muestra por fracción, en orden temporal.
    """
    with tempfile.TemporaryDirectory(prefix="kliptych-wm-") as tmpdir:
        tmp = Path(tmpdir)
        return [
            _evaluate_frame(ready, tmp / f"frame-{index:02d}.png", fraction * ready.duration)
            for index, fraction in enumerate(fractions)
        ]


def _evaluate_frame(ready: _Ready, frame_path: Path, t_s: float) -> Sample:
    """Evalúa una muestra temporal: similitud, zona y tamaño.

    Args:
        ready: Entradas verificadas de la evaluación.
        frame_path: Ruta del PNG temporal para el frame extraído.
        t_s: Segundo del video a muestrear.

    Returns:
        La muestra con su veredicto y evidencia.

    Raises:
        WatermarkError: Si el frame no se pudo extraer o decodificar.
    """
    try:
        import cv2
    except ImportError as error:
        msg = "opencv (cv2) no está disponible; no se pudo verificar el watermark"
        raise WatermarkError(msg) from error
    _extract_frame(ready.ffmpeg, ready.video, t_s, frame_path)
    frame = cv2.imread(str(frame_path), cv2.IMREAD_GRAYSCALE)
    if frame is None:
        msg = f"el frame en t={t_s:.3f}s no se pudo decodificar"
        raise WatermarkError(msg)
    shape = cast("tuple[int, ...]", frame.shape)
    return _match_sample(frame, (shape[1], shape[0]), ready, t_s)


def _match_sample(
    frame: cv2.typing.MatLike, frame_size: tuple[int, int], ready: _Ready, t_s: float
) -> Sample:
    """Compara un frame decodificado contra el template y veredea la muestra.

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
    correlation, x, y = _best_match(frame, ready.template, expected_w, expected_h, t_s)
    expected_x, expected_y = expected_top_left(
        frame_w, frame_h, expected_w, expected_h, ready.config.position
    )
    sample = Sample(
        t_s=t_s,
        matched=True,
        correlation=correlation,
        x=x,
        y=y,
        expected_x=expected_x,
        expected_y=expected_y,
    )
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
    )


def _best_match(
    frame: cv2.typing.MatLike, template: _Template, width: int, height: int, t_s: float
) -> tuple[float, int, int]:
    """Busca el template en el frame con SQDIFF enmascarado (acotado [0, 1]).

    SQDIFF crudo con máscara suma diferencias cuadráticas sobre los píxeles
    visibles. Sin normalización por energía local no hay 0/0 ni cocientes
    infinitos en regiones planas u oscuras (el fallo de CCORR_NORMED con
    máscara); la similitud se normaliza con la constante N·255², acotada en
    [0, 1] por construcción (1 es idéntico).

    Args:
        frame: Frame en grises ya decodificado.
        template: PNG de referencia ya cargado.
        width: Ancho esperado en el frame, en píxeles.
        height: Alto esperado en el frame, en píxeles.
        t_s: Segundo del video muestreado, para los errores.

    Returns:
        La (similitud, x, y) del mejor ajuste.

    Raises:
        WatermarkError: Si cv2 no está disponible o la máscara quedó vacía.
    """
    import cv2

    resized, mask = _resized_template(template, width, height)
    match = cv2.matchTemplate(frame, resized, cv2.TM_SQDIFF, mask=mask)
    count = cv2.countNonZero(mask) if mask is not None else width * height
    if count <= 0:
        msg = f"máscara vacía en t={t_s:.3f}s"
        raise WatermarkError(msg)
    minimum, _, location, _ = cv2.minMaxLoc(match)
    correlation = 1.0 - math.sqrt(min(1.0, minimum / (count * _MAX_PIXEL_SQDIFF)))
    return float(correlation), int(location[0]), int(location[1])


def _sample_failure(
    sample: Sample, frame_w: int, frame_h: int, expected_w: int, config: Watermark
) -> str | None:
    """Aplica los tres criterios de la muestra en orden de evidencia.

    Args:
        sample: Muestra con correlación y posición detectada.
        frame_w: Ancho del frame, en píxeles.
        frame_h: Alto del frame, en píxeles.
        expected_w: Ancho esperado del logo, en píxeles.
        config: Watermark exigido por el contrato.

    Returns:
        El motivo del rechazo, o ``None`` si la muestra cumple correlación,
        tamaño mínimo y zona.
    """
    if sample.correlation < _MATCH_THRESHOLD:
        return f"correlación {sample.correlation:.3f} bajo el umbral {_MATCH_THRESHOLD}"
    ratio = expected_w / frame_w
    if ratio < config.min_width_ratio:
        return f"ancho relativo {ratio:.3f} bajo el mínimo {config.min_width_ratio}"
    if not sample_within_zone(sample, frame_w, frame_h):
        return "el watermark no está en la zona exigida por el contrato"
    return None


def _extract_frame(ffmpeg: str, video: Path, t_s: float, destination: Path) -> None:
    """Extrae un frame exacto del video con ffmpeg (fail-closed).

    Args:
        ffmpeg: Binario ffmpeg resuelto en el PATH.
        video: Ruta del artefacto a verificar.
        t_s: Segundo del video a muestrear (seek exacto tras ``-i``).
        destination: Ruta del PNG a escribir.

    Raises:
        WatermarkError: Si ffmpeg falla, expira o no deja el frame.
    """
    argv = [
        ffmpeg,
        "-hide_banner",
        "-nostdin",
        "-v",
        "error",
        "-i",
        str(video),
        "-ss",
        f"{t_s:.3f}",
        "-frames:v",
        "1",
        str(destination),
    ]
    try:
        completed = subprocess.run(
            argv,
            capture_output=True,
            text=True,
            timeout=_FRAME_TIMEOUT_S,
            check=False,
        )
    except FileNotFoundError as error:
        msg = f"ffmpeg no está disponible: {error}"
        raise WatermarkError(msg) from error
    except subprocess.TimeoutExpired as error:
        msg = f"la extracción del frame en t={t_s:.3f}s excedió {_FRAME_TIMEOUT_S} s"
        raise WatermarkError(msg) from error
    except OSError as error:
        msg = f"no se pudo ejecutar ffmpeg: {error}"
        raise WatermarkError(msg) from error
    if completed.returncode != 0:
        tail = completed.stderr.strip()[-200:]
        msg = f"ffmpeg falló extrayendo el frame en t={t_s:.3f}s: {tail}"
        raise WatermarkError(msg)
    if not destination.is_file() or destination.stat().st_size == 0:
        msg = f"ffmpeg no dejó el frame en t={t_s:.3f}s"
        raise WatermarkError(msg)


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
