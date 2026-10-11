"""Seguimiento de caras acotado a una region declarada (Sprint A3.2).

La trampa que estos pruebas fijan es de ESPACIO DE COORDENADAS: si los frames
entregados al detector estan recortados a la region, MediaPipe devuelve cajas
en coordenadas del RECORTE. Normalizarlas contra el tamano del recorte produce
coordenadas que apuntan al sitio equivocado, y ffmpeg recorta donde se le pide
sin fallar: el defecto llega al usuario como un crop desencajado.

Por eso todas las aserciones de targets se hacen contra coordenadas de FUENTE.
"""

import io
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import pytest

from kliptych import reframe
from kliptych.reframe import (
    FaceBox,
    FFmpegFrameSource,
    FFmpegReframer,
    ReframeConfig,
    ReframeError,
    ReframeTarget,
    RgbFrame,
    SampledFrame,
)

_boxes_from_result = cast(
    "Callable[..., tuple[FaceBox, ...]]",
    getattr(reframe, "_boxes_from_result"),  # ruff: ignore[get-attr-with-constant] - acceso por nombre, igual que test_reframe
)

_SOURCE_WIDTH = 640
_SOURCE_HEIGHT = 480
# Region declarada: la mitad derecha de la mitad inferior del frame.
_REGION = ReframeTarget(x=0.25, y=0.25, width=0.5, height=0.5)
# Recorte en pixeles pares que le corresponde sobre 640x480.
_REGION_W = 320
_REGION_H = 240


class _RegionFrameSource:
    """Fuente doble que entrega frames YA recortados a la region declarada.

    Registra la region con la que se la invoco y los frames que entrega, para
    poder afirmar que el detector nunca vio pixeles de fuera.
    """

    def __init__(self, samples: Sequence[SampledFrame]) -> None:
        self._samples: list[SampledFrame] = list(samples)
        self.sample_fps: float | None = None
        self.requested_regions: list[ReframeTarget | None] = []

    def frames(
        self,
        video: Path,
        *,
        sample_fps: float,
        region: ReframeTarget | None = None,
    ) -> Iterator[SampledFrame]:
        _ = video
        self.sample_fps = sample_fps
        self.requested_regions.append(region)
        yield from self._samples


class _RegionDetector:
    """Detector doble que DEVUELVE lo que el normalizador de producción calcula.

    Las cajas se declaran en coordenadas del FRAME que el detector ve, se
    convierten a píxeles de ese frame y se pasan por ``_boxes_from_result`` con
    la región recibida. Así el remapeo bajo prueba es el de producción, no una
    reimplementación del test: si el remapeo se rompe, este doble devuelve las
    coordenadas equivocadas y las aserciones de targets fallan con una x
    desplazada.
    """

    def __init__(self, boxes: Sequence[tuple[FaceBox, ...]]) -> None:
        self._boxes: list[tuple[FaceBox, ...]] = list(boxes)
        self.calls: int = 0
        self.frames_seen: list[RgbFrame] = []
        self.regions_seen: list[ReframeTarget | None] = []

    def detect(
        self,
        frame: RgbFrame,
        *,
        region: ReframeTarget | None = None,
    ) -> tuple[FaceBox, ...]:
        self.frames_seen.append(frame)
        self.regions_seen.append(region)
        declared = self._boxes[self.calls] if self.calls < len(self._boxes) else ()
        self.calls += 1
        return _boxes_from_result(_as_mp_result(declared, frame), frame, region=region)


@dataclass
class _FakeBox:
    origin_x: int
    origin_y: int
    width: int
    height: int


@dataclass
class _FakeCategory:
    score: float


@dataclass
class _FakeDetection:
    bounding_box: _FakeBox
    categories: Sequence[_FakeCategory]


@dataclass
class _FakeResult:
    detections: Sequence[_FakeDetection]


@dataclass
class _Completed:
    """Sustituto de ``subprocess.CompletedProcess`` sin importar subprocess."""

    returncode: int
    stdout: str
    stderr: str


def _as_mp_result(boxes: Sequence[FaceBox], frame: RgbFrame) -> _FakeResult:
    """Convierte cajas normalizadas al frame en un resultado crudo de MediaPipe.

    Args:
        boxes: Cajas normalizadas al frame que vio el detector.
        frame: Frame que se entregó al detector.

    Returns:
        Un resultado crudo con las cajas en píxeles del frame.
    """
    return _FakeResult(
        detections=[
            _FakeDetection(
                bounding_box=_FakeBox(
                    origin_x=round(box.x * frame.width),
                    origin_y=round(box.y * frame.height),
                    width=round(box.width * frame.width),
                    height=round(box.height * frame.height),
                ),
                categories=[_FakeCategory(score=box.confidence)],
            )
            for box in boxes
        ]
    )


def _frame(width: int, height: int) -> RgbFrame:
    return RgbFrame(width=width, height=height, data=b"\x00" * (width * height * 3))


def _source_frame(region: ReframeTarget | None = _REGION) -> SampledFrame:
    """Frame de 640x480, o su recorte a la región si se declara.

    Args:
        region: Región declarada, o ``None`` para el frame completo.

    Returns:
        El frame muestreado con la geometría que le corresponde.
    """
    if region is None:
        return SampledFrame(timestamp_s=0.0, frame=_frame(_SOURCE_WIDTH, _SOURCE_HEIGHT))
    return SampledFrame(
        timestamp_s=0.0,
        frame=_frame(_REGION_W, _REGION_H),
        region=region,
        source_width=_SOURCE_WIDTH,
        source_height=_SOURCE_HEIGHT,
    )


def _video(tmp_path: Path) -> Path:
    path = tmp_path / "video.mp4"
    _ = path.write_bytes(b"video")
    return path


def _completed(stdout: str = "", stderr: str = "", returncode: int = 0) -> _Completed:
    return _Completed(returncode=returncode, stdout=stdout, stderr=stderr)


class _FakeProcess:
    def __init__(self, data: bytes, *, returncode: int = 0) -> None:
        self.stdout: io.BytesIO = io.BytesIO(data)
        self.stderr: io.BytesIO = io.BytesIO(b"")
        self.returncode: int | None = None
        self._final: int = returncode

    def wait(self, timeout: float | None = None) -> int:
        _ = timeout
        self.returncode = self._final
        return self._final

    def poll(self) -> int | None:
        return self.returncode


def test_bounded_region_maps_detection_to_source_coordinates(tmp_path: Path) -> None:
    """Una caja dentro de la region produce un target en coordenadas de FUENTE.

    La caja esta en el borde izquierdo del recorte (0.0 normalizado al recorte),
    que en el fuente es x=0.25. Si el remapeo falta, esa misma caja se lee como
    x=0.0 del fuente y el crop se va al extremo izquierdo: el test falla con una
    x DESPLAZADA, no con una excepcion.
    """
    region = _REGION
    detector = _RegionDetector([(FaceBox(x=0.0, y=0.0, width=0.2, height=0.2, confidence=0.9),)])
    source = _RegionFrameSource([_source_frame(region)])
    config = ReframeConfig(bounded_region=region)

    result = FFmpegReframer(detector=detector, frame_source=source, config=config).analyze(
        _video(tmp_path)
    )

    assert result.source_width == _SOURCE_WIDTH
    assert result.source_height == _SOURCE_HEIGHT
    assert detector.regions_seen == [region]
    assert detector.frames_seen[0].width == _REGION_W
    assert detector.frames_seen[0].height == _REGION_H

    # Centro de la caja en el fuente: 0.25 + 0.1*0.5 = 0.30 -> 192 px.
    # _offset = centro - ancho_de_crop/2 = 192 - 135 = 57 -> par -> 56.
    assert result.targets[0].x == pytest.approx(56 / _SOURCE_WIDTH)
    assert result.targets[0].x > 0.0


def test_bounded_region_never_feeds_detector_pixels_outside(tmp_path: Path) -> None:
    """El detector solo ve el recorte y el target resultante cae dentro de la region.

    Es la clase de fallo que se elimina: la cara del juego detras del streamer no
    esta en los pixeles entregados, asi que no puede producir ninguna deteccion.
    """
    region = _REGION
    detector = _RegionDetector([(FaceBox(x=0.5, y=0.5, width=0.2, height=0.2, confidence=0.9),)])
    source = _RegionFrameSource([_source_frame(region)])

    result = FFmpegReframer(
        detector=detector, frame_source=source, config=ReframeConfig(bounded_region=region)
    ).analyze(_video(tmp_path))

    assert all(
        frame.width == _REGION_W and frame.height == _REGION_H for frame in detector.frames_seen
    )
    target = result.targets[0]
    centre_x = target.x + target.width / 2.0
    centre_y = target.y + target.height / 2.0
    assert region.x <= centre_x <= region.x + region.width
    assert region.y <= centre_y <= region.y + region.height


def test_bounded_region_none_reproduces_full_frame_output(tmp_path: Path) -> None:
    """Una region que cubre el frame completo da exactamente el resultado de None.

    Es la prueba de que el camino por defecto no cambia: misma deteccion, mismo
    target, con y sin region declarada.
    """
    boxes = [(FaceBox(x=0.8, y=0.4, width=0.1, height=0.1, confidence=0.9),)]
    full = _source_frame(None)
    whole_frame = ReframeTarget(x=0.0, y=0.0, width=1.0, height=1.0)

    without_region = FFmpegReframer(
        detector=_RegionDetector(boxes),
        frame_source=_RegionFrameSource([full]),
        config=ReframeConfig(),
    ).analyze(_video(tmp_path))

    with_whole_frame = FFmpegReframer(
        detector=_RegionDetector(boxes),
        frame_source=_RegionFrameSource([full]),
        config=ReframeConfig(bounded_region=whole_frame),
    ).analyze(_video(tmp_path))

    assert without_region.targets == with_whole_frame.targets
    assert without_region.targets[0].x > 0.0


def test_bounded_region_without_detections_matches_no_face_path(tmp_path: Path) -> None:
    """Sin detecciones dentro de la region, el resultado es el del camino sin caras.

    No hay expansion silenciosa a frame completo: la region declarada se respeta
    (el detector sigue viendo solo el recorte) y el target es el mismo que produce
    el camino sin caras de siempre.
    """
    region = _REGION
    no_faces: list[tuple[FaceBox, ...]] = [()]
    full = _source_frame(None)

    detector = _RegionDetector(no_faces)
    bounded = FFmpegReframer(
        detector=detector,
        frame_source=_RegionFrameSource([_source_frame(region)]),
        config=ReframeConfig(bounded_region=region),
    ).analyze(_video(tmp_path))

    reference = FFmpegReframer(
        detector=_RegionDetector(no_faces),
        frame_source=_RegionFrameSource([full]),
        config=ReframeConfig(),
    ).analyze(_video(tmp_path))

    assert bounded.targets == reference.targets
    assert all(
        frame.width == _REGION_W and frame.height == _REGION_H for frame in detector.frames_seen
    )
    assert detector.regions_seen == [region]


def test_declared_region_is_never_silently_ignored(tmp_path: Path) -> None:
    """Una fuente que entrega el frame completo pese a la región declarada FALLA.

    Es la prohibición explicita del manifiesto: una región declarada que se
    expande en silencio es una regla declarada ignorándose. Sin este test, una
    fuente de frames que perdiese el ``region=`` haría que el detector viese el
    frame completo y reintrodujese los falsos positivos que se vienen a eliminar,
    sin que nada se quejase.
    """
    region = _REGION
    detector = _RegionDetector([(FaceBox(x=0.0, y=0.0, width=0.2, height=0.2, confidence=0.9),)])
    source = _RegionFrameSource([_source_frame(None)])

    with pytest.raises(ReframeError, match="acotada declarada no se aplic"):
        _ = FFmpegReframer(
            detector=detector,
            frame_source=source,
            config=ReframeConfig(bounded_region=region),
        ).analyze(_video(tmp_path))

    assert detector.calls == 0


def test_frame_source_crops_frames_to_bounded_region(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """La fuente real recorta con ffmpeg ANTES de entregar pixeles.

    Es el ahorro de CPU que pide el manifiesto: MediaPipe recibe solo el
    fragmento, no el frame completo.
    """
    video = _video(tmp_path)

    def run(argv: list[str], **kwargs: object) -> object:
        _ = (argv, kwargs)
        return _completed(stdout=f"{_SOURCE_WIDTH}x{_SOURCE_HEIGHT}")

    monkeypatch.setattr("kliptych.reframe.subprocess.run", run)
    captured: list[list[str]] = []

    def popen(argv: list[str], **kwargs: object) -> object:
        _ = kwargs
        captured.append(list(argv))
        return _FakeProcess(b"")

    monkeypatch.setattr("kliptych.reframe.subprocess.Popen", popen)

    samples = list(
        FFmpegFrameSource().frames(
            video, sample_fps=2.0, region=ReframeTarget(x=0.25, y=0.25, width=0.5, height=0.5)
        )
    )

    assert samples == []
    assert len(captured) == 1
    argv = captured[0]
    vf = argv[argv.index("-vf") + 1]
    assert "crop=320:240:160:120" in vf
    assert "fps=2" in vf
