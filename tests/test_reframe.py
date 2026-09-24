"""Tests unitarios de reframe 9:16: MediaPipe y ffmpeg simulados, cero red.

La detección se inyecta como cajas controladas y la extracción/render de ffmpeg
se simulan; ningún test ejecuta MediaPipe ni ffmpeg reales (eso vive en
``test_reframe_integration.py``).
"""

import io
import subprocess
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import pytest
from pydantic import ValidationError

from kliptych import reframe
from kliptych.encoding import RenderConfig
from kliptych.reframe import (
    FaceBox,
    FFmpegFrameSource,
    FFmpegReframer,
    MediaPipeFaceDetector,
    ReframeConfig,
    ReframeError,
    ReframeResult,
    ReframeTarget,
    RgbFrame,
    SampledFrame,
)

_Call = tuple[list[str], dict[str, object]]
_FakeRun = Callable[..., subprocess.CompletedProcess[str]]


def _private(name: str) -> object:
    return cast("object", getattr(reframe, name))


_even = cast("Callable[[int], int]", _private("_even"))
_crop_size = cast("Callable[[int, int], tuple[int, int]]", _private("_crop_size"))
_primary_face = cast(
    "Callable[[Sequence[FaceBox], float], FaceBox | None]", _private("_primary_face")
)
_linear_expression = cast(
    "Callable[[Sequence[float], Sequence[float]], str]", _private("_linear_expression")
)
_crop_filter = cast("Callable[..., str]", _private("_crop_filter"))
_targets_from_faces = cast(
    "Callable[..., tuple[ReframeTarget, ...]]", _private("_targets_from_faces")
)
_boxes_from_result = cast("Callable[..., tuple[FaceBox, ...]]", _private("_boxes_from_result"))
_probe_dimensions = cast("Callable[..., tuple[int, int]]", _private("_probe_dimensions"))
_load_mediapipe = cast("Callable[[], object]", _private("_load_mediapipe"))


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


class _FakeRawDetector:
    def __init__(self, result: _FakeResult, closed: list[bool]) -> None:
        self._result: _FakeResult = result
        self._closed: list[bool] = closed

    def detect(self, image: object) -> _FakeResult:
        _ = image
        return self._result

    def close(self) -> None:
        self._closed.append(True)


@dataclass
class _FakeMediaPipeApi:
    make_image: Callable[[RgbFrame], object]
    base_options: Callable[..., object]
    delegate_cpu: object
    face_detector_options: Callable[..., object]
    running_mode_image: object
    create_face_detector: Callable[..., object]


class _FakeImageFormat:
    SRGB: str = "srgb"


class _FakeRunningMode:
    IMAGE: str = "IMAGE"


class _FakeBaseOptions:
    class Delegate:
        CPU: str = "CPU"


class _FakeBaseOptionsNoDelegate:
    pass


class _FakeImage:
    def __init__(self, **kwargs: object) -> None:
        self.kwargs: dict[str, object] = kwargs


class _FakeFaceOptions:
    def __init__(self, **kwargs: object) -> None:
        self.kwargs: dict[str, object] = kwargs


class _FakeFaceDetectorClass:
    @staticmethod
    def create_from_options(options: object) -> object:
        return ("detector", options)


class _FakeNumpyArray:
    @staticmethod
    def reshape(shape: tuple[int, int, int]) -> object:
        return ("array", shape)


class _FakeNumpy:
    uint8: str = "uint8"

    @staticmethod
    def frombuffer(buffer: bytes, *, dtype: object) -> _FakeNumpyArray:
        _ = (buffer, dtype)
        return _FakeNumpyArray()


class _FakeMediaPipeModule:
    Image: type[_FakeImage] = _FakeImage
    ImageFormat: type[_FakeImageFormat] = _FakeImageFormat


class _FakeTasksModule:
    BaseOptions: type[_FakeBaseOptions] = _FakeBaseOptions


class _FakeTasksNoDelegateModule:
    BaseOptions: type[_FakeBaseOptionsNoDelegate] = _FakeBaseOptionsNoDelegate


class _FakeVisionModule:
    FaceDetector: type[_FakeFaceDetectorClass] = _FakeFaceDetectorClass
    FaceDetectorOptions: type[_FakeFaceOptions] = _FakeFaceOptions
    RunningMode: type[_FakeRunningMode] = _FakeRunningMode


def _module_loader(modules: dict[str, object]) -> Callable[[str], object]:
    def load(name: str) -> object:
        return modules[name]

    return load


def _loader_returning(value: object) -> Callable[[str], object]:
    def load(name: str) -> object:
        _ = name
        return value

    return load


def _api_loader(value: object) -> Callable[[], object]:
    def load() -> object:
        return value

    return load


class _FakeDetector:
    def __init__(self, boxes: Sequence[tuple[FaceBox, ...]]) -> None:
        self._boxes: list[tuple[FaceBox, ...]] = list(boxes)
        self.calls: int = 0

    def detect(self, frame: RgbFrame) -> tuple[FaceBox, ...]:
        _ = frame
        result = self._boxes[self.calls] if self.calls < len(self._boxes) else ()
        self.calls += 1
        return result


class _FakeFrameSource:
    def __init__(self, samples: Sequence[SampledFrame]) -> None:
        self._samples: list[SampledFrame] = list(samples)
        self.sample_fps: float | None = None

    def frames(self, video: Path, *, sample_fps: float) -> Iterator[SampledFrame]:
        _ = video
        self.sample_fps = sample_fps
        yield from self._samples


class _FakeProcess:
    def __init__(self, data: bytes, *, returncode: int = 0, stderr: bytes = b"") -> None:
        self.stdout: io.BytesIO = io.BytesIO(data)
        self.stderr: io.BytesIO = io.BytesIO(stderr)
        self.returncode: int | None = None
        self._final: int = returncode

    def wait(self, timeout: float | None = None) -> int:
        _ = timeout
        self.returncode = self._final
        return self._final

    def poll(self) -> int | None:
        return self.returncode

    def kill(self) -> None:
        self.returncode = -9


def _frame(width: int = 640, height: int = 480) -> RgbFrame:
    return RgbFrame(width=width, height=height, data=b"\x00" * (width * height * 3))


def _sample(index: int, width: int = 640, height: int = 480) -> SampledFrame:
    return SampledFrame(timestamp_s=index / 2.0, frame=_frame(width, height))


def _face(x: float, y: float, width: float, height: float, confidence: float = 0.9) -> FaceBox:
    return FaceBox(x=x, y=y, width=width, height=height, confidence=confidence)


def _video(tmp_path: Path) -> Path:
    path = tmp_path / "video.mp4"
    _ = path.write_bytes(b"video")
    return path


def _model(tmp_path: Path) -> Path:
    path = tmp_path / "face.tflite"
    _ = path.write_bytes(b"model")
    return path


def _completed(
    *, stdout: str = "", stderr: str = "", returncode: int = 0
) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(
        args=["ffprobe"], returncode=returncode, stdout=stdout, stderr=stderr
    )


def _run_returning(*, stdout: str = "", stderr: str = "", returncode: int = 0) -> _FakeRun:
    def run(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        _ = (argv, kwargs)
        return _completed(stdout=stdout, stderr=stderr, returncode=returncode)

    return run


def _popen_returning(process: object) -> Callable[..., object]:
    def popen(*args: object, **kwargs: object) -> object:
        _ = (args, kwargs)
        return process

    return popen


def _raising_run(exc: BaseException) -> _FakeRun:
    def run(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        _ = (argv, kwargs)
        raise exc

    return run


def test_even_rounds_down_to_even() -> None:
    assert _even(3) == 2
    assert _even(4) == 4


def test_crop_size_landscape_uses_full_height() -> None:
    assert _crop_size(1920, 1080) == (608, 1080)


def test_crop_size_vertical_uses_full_width() -> None:
    assert _crop_size(1080, 1920) == (1080, 1920)


def test_crop_size_small_landscape_is_exact_aspect() -> None:
    width, height = _crop_size(640, 480)
    assert (width, height) == (270, 480)
    assert width / height == pytest.approx(9 / 16, abs=0.001)


def test_crop_size_rejects_non_positive() -> None:
    with pytest.raises(ValueError, match="origen"):
        _ = _crop_size(0, 100)


def test_primary_face_picks_largest_above_confidence() -> None:
    small = _face(0.0, 0.0, 0.1, 0.1, confidence=0.9)
    large = _face(0.0, 0.0, 0.4, 0.4, confidence=0.5)
    assert _primary_face([small, large], 0.4) is large


def test_primary_face_filters_by_confidence() -> None:
    low = _face(0.0, 0.0, 0.5, 0.5, confidence=0.2)
    assert _primary_face([low], 0.5) is None


def test_primary_face_none_when_empty() -> None:
    assert _primary_face([], 0.3) is None


def test_targets_without_faces_center_the_crop() -> None:
    targets = _targets_from_faces(
        [(), ()],
        source_width=1920,
        source_height=1080,
        config=ReframeConfig(),
    )
    assert all(target.x == 656 and target.y == 0 for target in targets)
    assert all(target.width == 608 and target.height == 1080 for target in targets)


def test_targets_follow_face_to_the_right() -> None:
    faces = [(_face(0.7, 0.3, 0.2, 0.3),)]
    targets = _targets_from_faces(
        faces,
        source_width=1920,
        source_height=1080,
        config=ReframeConfig(),
    )
    assert targets[0].x > 656


def test_targets_clamp_to_source_bounds() -> None:
    faces = [(_face(0.98, 0.98, 0.02, 0.02),)]
    targets = _targets_from_faces(
        faces,
        source_width=1920,
        source_height=1080,
        config=ReframeConfig(),
    )
    assert targets[0].x == 1312
    assert targets[0].y == 0


def test_targets_limit_step_between_frames() -> None:
    faces = [(_face(0.15, 0.4, 0.05, 0.05),), (_face(0.78, 0.4, 0.05, 0.05),)]
    config = ReframeConfig()
    targets = _targets_from_faces(
        faces,
        source_width=1920,
        source_height=1080,
        config=config,
    )
    max_step = config.max_step_ratio * 1920
    assert abs(targets[1].x - targets[0].x) <= max_step + 2


def test_targets_enforce_vertical_aspect() -> None:
    faces = [(), (_face(0.4, 0.4, 0.2, 0.2),)]
    targets = _targets_from_faces(
        faces,
        source_width=1920,
        source_height=1080,
        config=ReframeConfig(),
    )
    for target in targets:
        assert target.width / target.height == pytest.approx(9 / 16, abs=0.01)


def test_linear_expression_single_value() -> None:
    assert _linear_expression([0.0], [100.0]) == "100"


def test_linear_expression_interpolates() -> None:
    expression = _linear_expression([0.0, 1.0], [0.0, 10.0])
    assert expression == "if(lt(t,1),0+(10-0)*(t-0)/(1-0),10)"


def test_linear_expression_duplicate_times_hold() -> None:
    assert _linear_expression([0.0, 0.0], [5.0, 9.0]) == "if(lt(t,0),5,9)"


def test_linear_expression_empty_raises() -> None:
    with pytest.raises(ValueError, match="longitud"):
        _ = _linear_expression([], [])


def test_linear_expression_mismatched_raises() -> None:
    with pytest.raises(ValueError, match="longitud"):
        _ = _linear_expression([0.0, 1.0], [0.0])


def test_crop_filter_serializes_single_target() -> None:
    targets = (ReframeTarget(x=0, y=0, width=608, height=1080),)
    assert _crop_filter(targets, sample_fps=2.0) == "crop=608:1080:x='0':y='0'"


def test_crop_filter_empty_raises() -> None:
    with pytest.raises(ValueError, match="recortes"):
        _ = _crop_filter((), sample_fps=2.0)


def test_crop_filter_mismatched_sizes_raise() -> None:
    targets = (
        ReframeTarget(x=0, y=0, width=608, height=1080),
        ReframeTarget(x=0, y=0, width=600, height=1080),
    )
    with pytest.raises(ValueError, match="tamaño"):
        _ = _crop_filter(targets, sample_fps=2.0)


def test_boxes_from_result_normalizes_and_clamps() -> None:
    result = _FakeResult(
        detections=[
            _FakeDetection(_FakeBox(64, 48, 128, 96), [_FakeCategory(0.9)]),
        ]
    )
    boxes = _boxes_from_result(result, _frame(640, 480))
    assert boxes == (_face(0.1, 0.1, 0.2, 0.2, confidence=0.9),)


def test_boxes_from_result_skips_zero_size_and_missing_categories() -> None:
    result = _FakeResult(
        detections=[
            _FakeDetection(_FakeBox(0, 0, 0, 10), []),
            _FakeDetection(_FakeBox(10, 10, 20, 20), []),
        ]
    )
    boxes = _boxes_from_result(result, _frame(640, 480))
    assert len(boxes) == 1
    assert boxes[0].confidence == pytest.approx(0.0)


def test_boxes_from_result_empty() -> None:
    assert _boxes_from_result(_FakeResult(detections=[]), _frame()) == ()


def test_probe_dimensions_parses_csv(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("kliptych.reframe.subprocess.run", _run_returning(stdout="1920x1080\n"))
    assert _probe_dimensions(Path("video.mp4"), ffprobe="ffprobe", timeout_s=5.0) == (1920, 1080)


def test_probe_dimensions_invalid_output_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("kliptych.reframe.subprocess.run", _run_returning(stdout="n/a"))
    with pytest.raises(ReframeError, match="dimensiones"):
        _ = _probe_dimensions(Path("video.mp4"), ffprobe="ffprobe", timeout_s=5.0)


def test_probe_dimensions_nonzero_returncode_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "kliptych.reframe.subprocess.run",
        _run_returning(returncode=1, stderr="boom"),
    )
    with pytest.raises(ReframeError, match="ffprobe falló"):
        _ = _probe_dimensions(Path("video.mp4"), ffprobe="ffprobe", timeout_s=5.0)


def test_probe_dimensions_missing_binary_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "kliptych.reframe.subprocess.run", _raising_run(FileNotFoundError("ffprobe"))
    )
    with pytest.raises(ReframeError, match="no está disponible"):
        _ = _probe_dimensions(Path("video.mp4"), ffprobe="ffprobe", timeout_s=5.0)


def test_probe_dimensions_timeout_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "kliptych.reframe.subprocess.run",
        _raising_run(subprocess.TimeoutExpired(cmd="ffprobe", timeout=5.0)),
    )
    with pytest.raises(ReframeError, match="timeout"):
        _ = _probe_dimensions(Path("video.mp4"), ffprobe="ffprobe", timeout_s=5.0)


def test_probe_dimensions_os_error_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("kliptych.reframe.subprocess.run", _raising_run(OSError("permiso")))
    with pytest.raises(ReframeError, match="no se pudo ejecutar"):
        _ = _probe_dimensions(Path("video.mp4"), ffprobe="ffprobe", timeout_s=5.0)


def test_frame_source_streams_frames(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("kliptych.reframe.subprocess.run", _run_returning(stdout="4x2"))
    data = b"\x01" * 24 + b"\x02" * 24
    process = _FakeProcess(data)
    monkeypatch.setattr("kliptych.reframe.subprocess.Popen", _popen_returning(process))
    samples = list(FFmpegFrameSource().frames(_video(tmp_path), sample_fps=2.0))
    assert [sample.timestamp_s for sample in samples] == [0.0, 0.5]
    assert samples[0].frame.data == b"\x01" * 24
    assert samples[1].frame.data == b"\x02" * 24


def test_frame_source_rejects_non_positive_fps() -> None:
    with pytest.raises(ValueError, match="fps"):
        _ = list(FFmpegFrameSource().frames(Path("video.mp4"), sample_fps=0.0))


def test_frame_source_missing_video_raises(tmp_path: Path) -> None:
    with pytest.raises(ReframeError, match="video no existe"):
        _ = list(FFmpegFrameSource().frames(tmp_path / "falta.mp4", sample_fps=2.0))


def test_frame_source_missing_ffmpeg_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("kliptych.reframe.subprocess.run", _run_returning(stdout="4x2"))

    def raising_popen(*args: object, **kwargs: object) -> object:
        _ = (args, kwargs)
        message = "ffmpeg"
        raise FileNotFoundError(message)

    monkeypatch.setattr("kliptych.reframe.subprocess.Popen", raising_popen)
    with pytest.raises(ReframeError, match="no está disponible"):
        _ = list(FFmpegFrameSource().frames(_video(tmp_path), sample_fps=2.0))


def test_frame_source_nonzero_exit_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("kliptych.reframe.subprocess.run", _run_returning(stdout="4x2"))
    process = _FakeProcess(b"\x00" * 24, returncode=1)
    monkeypatch.setattr("kliptych.reframe.subprocess.Popen", _popen_returning(process))
    with pytest.raises(ReframeError, match="falló al extraer"):
        _ = list(FFmpegFrameSource().frames(_video(tmp_path), sample_fps=2.0))


def test_frame_source_timeout_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("kliptych.reframe.subprocess.run", _run_returning(stdout="4x2"))
    process = _FakeProcess(b"\x00" * 24)
    monkeypatch.setattr("kliptych.reframe.subprocess.Popen", _popen_returning(process))
    ticks = iter([0.0, 1e9])
    monkeypatch.setattr("kliptych.reframe.time.monotonic", lambda: next(ticks))
    with pytest.raises(ReframeError, match="timeout"):
        _ = list(FFmpegFrameSource().frames(_video(tmp_path), sample_fps=2.0))


def test_mediapipe_detector_uses_cpu_delegate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    recorded: dict[str, object] = {}
    closed: list[bool] = []
    model = _model(tmp_path)
    result = _FakeResult(
        detections=[_FakeDetection(_FakeBox(64, 48, 128, 96), [_FakeCategory(0.9)])]
    )

    def make_image(frame: RgbFrame) -> object:
        return ("image", frame.width)

    def base_options(**kwargs: object) -> object:
        recorded["base_options"] = kwargs
        return object()

    def face_options(**kwargs: object) -> object:
        recorded["face_options"] = kwargs
        return object()

    def create_face_detector(options: object) -> _FakeRawDetector:
        _ = options
        return _FakeRawDetector(result, closed)

    api = _FakeMediaPipeApi(
        make_image=make_image,
        base_options=base_options,
        delegate_cpu="CPU",
        face_detector_options=face_options,
        running_mode_image="IMAGE",
        create_face_detector=create_face_detector,
    )
    monkeypatch.setattr("kliptych.reframe._load_mediapipe", _api_loader(api))
    detector = MediaPipeFaceDetector(model_path=model)
    boxes = detector.detect(_frame(640, 480))
    assert recorded["base_options"] == {"model_asset_path": str(model), "delegate": "CPU"}
    assert boxes == (_face(0.1, 0.1, 0.2, 0.2, confidence=0.9),)
    detector.close()
    assert closed == [True]


def test_mediapipe_detector_reuses_single_detector(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    created: list[object] = []
    result = _FakeResult(detections=[])

    def make_image(frame: RgbFrame) -> object:
        _ = frame
        return object()

    def base_options(**kwargs: object) -> object:
        _ = kwargs
        return object()

    def face_options(**kwargs: object) -> object:
        _ = kwargs
        return object()

    def create(options: object) -> _FakeRawDetector:
        created.append(options)
        return _FakeRawDetector(result, [])

    api = _FakeMediaPipeApi(
        make_image=make_image,
        base_options=base_options,
        delegate_cpu="CPU",
        face_detector_options=face_options,
        running_mode_image="IMAGE",
        create_face_detector=create,
    )
    monkeypatch.setattr("kliptych.reframe._load_mediapipe", _api_loader(api))
    detector = MediaPipeFaceDetector(model_path=_model(tmp_path))
    _ = detector.detect(_frame())
    _ = detector.detect(_frame())
    assert len(created) == 1


def test_mediapipe_detector_missing_model_raises(tmp_path: Path) -> None:
    detector = MediaPipeFaceDetector(model_path=tmp_path / "falta.tflite")
    with pytest.raises(ReframeError, match="modelo"):
        _ = detector.detect(_frame())


def test_mediapipe_detector_close_without_detector_is_noop(tmp_path: Path) -> None:
    MediaPipeFaceDetector(model_path=_model(tmp_path)).close()


def test_mediapipe_detector_rejects_bad_confidence(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="confianza"):
        _ = MediaPipeFaceDetector(model_path=_model(tmp_path), min_confidence=0.0)


def test_mediapipe_detector_rejects_bad_suppression(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="supresión"):
        _ = MediaPipeFaceDetector(model_path=_model(tmp_path), min_suppression=1.5)


def test_load_mediapipe_without_package_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_import(name: str) -> object:
        raise ImportError(name)

    monkeypatch.setattr("kliptych.reframe.importlib.import_module", fake_import)
    with pytest.raises(ReframeError, match="no está instalado"):
        _ = _load_mediapipe()


def test_load_mediapipe_missing_api_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("kliptych.reframe.importlib.import_module", _loader_returning(object()))
    with pytest.raises(ReframeError, match="no expone"):
        _ = _load_mediapipe()


def test_load_mediapipe_missing_cpu_delegate_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    modules: dict[str, object] = {
        "mediapipe": _FakeMediaPipeModule,
        "mediapipe.tasks.python": _FakeTasksNoDelegateModule,
        "mediapipe.tasks.python.vision": _FakeVisionModule,
        "numpy": _FakeNumpy,
    }
    monkeypatch.setattr("kliptych.reframe.importlib.import_module", _module_loader(modules))
    with pytest.raises(ReframeError, match=r"BaseOptions\.Delegate"):
        _ = _load_mediapipe()


def test_load_mediapipe_builds_api(monkeypatch: pytest.MonkeyPatch) -> None:
    modules: dict[str, object] = {
        "mediapipe": _FakeMediaPipeModule,
        "mediapipe.tasks.python": _FakeTasksModule,
        "mediapipe.tasks.python.vision": _FakeVisionModule,
        "numpy": _FakeNumpy,
    }
    monkeypatch.setattr("kliptych.reframe.importlib.import_module", _module_loader(modules))
    api = cast("_FakeMediaPipeApi", _load_mediapipe())
    image = cast("_FakeImage", api.make_image(_frame(2, 2)))
    assert image.kwargs["image_format"] == "srgb"


def test_reframer_analyze_follows_faces(tmp_path: Path) -> None:
    detector = _FakeDetector([(_face(0.8, 0.4, 0.1, 0.1),), (_face(0.8, 0.4, 0.1, 0.1),)])
    source = _FakeFrameSource([_sample(0), _sample(1)])
    reframer = FFmpegReframer(detector=detector, frame_source=source)
    result = reframer.analyze(_video(tmp_path))
    assert isinstance(result, ReframeResult)
    assert result.source_width == 640
    assert result.source_height == 480
    assert result.target_aspect == "9:16"
    assert len(result.targets) == 2
    assert result.targets[0].x > 0
    assert source.sample_fps == ReframeConfig().sample_fps


def test_reframer_analyze_missing_video_raises(tmp_path: Path) -> None:
    reframer = FFmpegReframer(detector=_FakeDetector([]), frame_source=_FakeFrameSource([]))
    with pytest.raises(ReframeError, match="video no existe"):
        _ = reframer.analyze(tmp_path / "falta.mp4")


def test_reframer_analyze_without_frames_raises(tmp_path: Path) -> None:
    reframer = FFmpegReframer(detector=_FakeDetector([]), frame_source=_FakeFrameSource([]))
    with pytest.raises(ReframeError, match="frames"):
        _ = reframer.analyze(_video(tmp_path))


def test_reframer_reframe_publishes_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_run(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        _ = kwargs
        _ = Path(argv[-1]).write_bytes(b"reframed")
        return _completed()

    monkeypatch.setattr("kliptych.reframe.subprocess.run", fake_run)
    reframer = FFmpegReframer(
        detector=_FakeDetector([(_face(0.5, 0.5, 0.2, 0.2),)]),
        frame_source=_FakeFrameSource([_sample(0)]),
    )
    destination = tmp_path / "out" / "reframed.mp4"
    result = reframer.reframe(video=_video(tmp_path), destination=destination)
    assert result == destination
    assert destination.read_bytes() == b"reframed"


def test_reframer_reframe_failure_leaves_destination_intact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "kliptych.reframe.subprocess.run",
        _run_returning(returncode=1, stderr="boom"),
    )
    reframer = FFmpegReframer(
        detector=_FakeDetector([(_face(0.5, 0.5, 0.2, 0.2),)]),
        frame_source=_FakeFrameSource([_sample(0)]),
    )
    destination = tmp_path / "out.mp4"
    _ = destination.write_bytes(b"previo")
    with pytest.raises(ReframeError, match="falló con código 1"):
        _ = reframer.reframe(video=_video(tmp_path), destination=destination)
    assert destination.read_bytes() == b"previo"


def test_reframer_render_arguments_use_nvenc(tmp_path: Path) -> None:
    reframer = FFmpegReframer(
        detector=_FakeDetector([(_face(0.5, 0.5, 0.2, 0.2),)]),
        frame_source=_FakeFrameSource([_sample(0)]),
        render=RenderConfig(nvenc_available=True),
    )
    result = reframer.analyze(_video(tmp_path))
    argv = reframer.render_arguments(
        video=_video(tmp_path), destination=tmp_path / "out.mp4", result=result
    )
    assert "h264_nvenc" in argv
    assert "crop=" in argv[argv.index("-vf") + 1]


def test_reframer_render_arguments_fall_back(tmp_path: Path) -> None:
    reframer = FFmpegReframer(
        detector=_FakeDetector([(_face(0.5, 0.5, 0.2, 0.2),)]),
        frame_source=_FakeFrameSource([_sample(0)]),
    )
    result = reframer.analyze(_video(tmp_path))
    argv = reframer.render_arguments(
        video=_video(tmp_path), destination=tmp_path / "out.mp4", result=result
    )
    assert "libx264" in argv


def test_reframer_builds_default_frame_source() -> None:
    reframer = FFmpegReframer(detector=_FakeDetector([]))
    attribute = "_frame_source"
    source = cast("object", getattr(reframer, attribute))
    assert isinstance(source, FFmpegFrameSource)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"sample_fps": 0.0}, "fps"),
        ({"smoothing": 0.0}, "suavizado"),
        ({"max_step_ratio": 1.5}, "paso"),
        ({"min_confidence": 1.5}, "confianza"),
    ],
)
def test_reframe_config_rejects_invalid(kwargs: dict[str, float], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        _ = ReframeConfig(**kwargs)


def test_reframe_target_rejects_negative_x() -> None:
    with pytest.raises(ValidationError):
        _ = ReframeTarget(x=-1, y=0, width=608, height=1080)


def test_reframe_target_rejects_zero_width() -> None:
    with pytest.raises(ValidationError):
        _ = ReframeTarget(x=0, y=0, width=0, height=1080)


def test_reframe_target_rejects_extra_fields() -> None:
    with pytest.raises(ValidationError):
        _ = ReframeTarget.model_validate({"x": 0, "y": 0, "width": 608, "height": 1080, "extra": 1})


def test_reframe_target_is_frozen() -> None:
    target = ReframeTarget(x=0, y=0, width=608, height=1080)
    with pytest.raises(ValidationError):
        target.x = 5


def test_reframe_result_rejects_non_positive_source() -> None:
    with pytest.raises(ValidationError):
        _ = ReframeResult(targets=(), source_width=0, source_height=1080)


def test_rgb_frame_rejects_bad_dimensions() -> None:
    with pytest.raises(ValueError, match="dimensiones"):
        _ = RgbFrame(width=0, height=1, data=b"")


def test_rgb_frame_rejects_bad_buffer_length() -> None:
    with pytest.raises(ValueError, match="buffer"):
        _ = RgbFrame(width=2, height=2, data=b"\x00")


def test_face_box_rejects_out_of_range() -> None:
    with pytest.raises(ValidationError):
        _ = FaceBox(x=1.5, y=0.0, width=0.1, height=0.1)


def test_face_box_rejects_zero_width() -> None:
    with pytest.raises(ValidationError):
        _ = FaceBox(x=0.0, y=0.0, width=0.0, height=0.1)


def test_frame_source_rejects_invalid_timeout() -> None:
    with pytest.raises(ValueError, match="timeout"):
        _ = FFmpegFrameSource(timeout_s=0.0)
