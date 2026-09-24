"""Tests de pipeline reanudable con puntos de control (checkpointing).

Verifica que el estado del pipeline se persista tras cada etapa, que al
reanudar se salten las etapas ya completadas (idempotencia a nivel de etapa),
que los artefactos útiles se conserven ante fallos y que los temporales
puros se eliminen deterministamente.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Protocol, cast

import pytest

from kliptych import orchestrator
from kliptych.contract import Contract, Segment
from kliptych.encoding import RenderConfig
from kliptych.moments import (
    ChatMessage,
    Moment,
    MomentDetectionError,
    MomentSource,
)
from kliptych.orchestrator import (
    PipelineConfig,
    PipelineError,
    PipelineResult,
    run_long_video,
)
from kliptych.pipeline_state import (
    PipelineStage,
    PipelineStateManager,
)
from kliptych.reframe import (
    ReframeResult,
    ReframeTarget,
)
from kliptych.segment import (
    SegmentSelection,
)
from kliptych.transcribe import (
    Transcript,
    TranscriptionError,
    Word,
)

_URL = "https://example.com/video"


class _Registry(Protocol):
    paths: list[Path]

    def register(self, path: Path, *, is_artifact: bool = False) -> Path: ...

    def cleanup(self, mode: str = "all") -> tuple[str, ...]: ...


def _private(name: str) -> object:
    return cast("object", getattr(orchestrator, name))


_cleanup_registry = cast("Callable[[], _Registry]", _private("_CleanupRegistry"))


def _contract() -> Contract:
    return Contract.model_validate(
        {
            "schema_version": "1.1",
            "campaign_id": "camp-resume",
            "format": "video",
            "mode": "long_video",
            "platforms": {
                "tiktok": {
                    "duration": {"min_s": 1, "max_s": 10},
                    "caption_rules": {"must_mention": [], "first_line": None, "forbidden": []},
                    "audio_rule": "any",
                    "required_hashtags": [],
                    "required_mentions": [],
                    "attribution": {"type": "none", "value": None},
                    "link_rules": {"link_in_bio": False},
                }
            },
            "languages": {"source": "es", "subtitles": None, "caption": "es", "voice": None},
            "official_audio": None,
            "watermark": {"required": False, "asset_id": None, "visible_full_video": False},
            "spelling_locks": [],
            "prohibitions": [],
            "rules": {
                "hard": ["duration.min", "duration.max"],
                "recommended": [],
                "manual_review": [],
            },
            "assets": {"required": [], "optional": []},
            "segments": [{"start_s": 0.0, "end_s": 1.0}],
            "geo_target": None,
            "min_views_for_payout": {"value": None, "enforcement": "post_publication_manual"},
            "analytics_proof_required": {"value": False, "enforcement": "post_publication_manual"},
        }
    )


def _transcript() -> Transcript:
    words = (
        Word(start_s=0.0, end_s=0.5, text="hola", confidence=0.9, token_id=0),
        Word(start_s=0.5, end_s=1.0, text="mundo", confidence=0.9, token_id=1),
    )
    return Transcript(words=words, language="es", duration_s=1.0, text="hola mundo")


def _moment() -> Moment:
    return Moment(start_s=0.0, end_s=1.0, score=0.8, source=MomentSource.FUSED)


def _selection() -> SegmentSelection:
    return SegmentSelection(segments=(Segment(start_s=0.0, end_s=1.0),), rationale="resumen")


def _reframe_result() -> ReframeResult:
    return ReframeResult(
        targets=(ReframeTarget(x=0, y=0, width=134, height=240),),
        source_width=320,
        source_height=240,
    )


@dataclass
class _CallCounter:
    download_calls: int = 0
    transcribe_calls: int = 0
    detect_calls: int = 0
    select_calls: int = 0
    reframe_calls: int = 0
    subtitles_calls: int = 0

    download_error: Exception | None = None
    transcribe_error: Exception | None = None
    detect_error: Exception | None = None
    reframe_error: Exception | None = None
    subtitles_error: Exception | None = None


class _TrackedDownloader:
    _counter: _CallCounter

    def __init__(self, counter: _CallCounter) -> None:
        self._counter = counter

    def download_video(
        self, *, url: str, destination: Path, format_selector: str | None = None
    ) -> Path:
        _ = (url, format_selector)
        self._counter.download_calls += 1
        if self._counter.download_error is not None:
            raise self._counter.download_error
        _ = destination.write_bytes(b"source_video_bytes")
        return destination


class _TrackedTranscriber:
    _counter: _CallCounter

    def __init__(self, counter: _CallCounter) -> None:
        self._counter = counter

    def transcribe(self, audio: Path) -> Transcript:
        _ = audio
        self._counter.transcribe_calls += 1
        if self._counter.transcribe_error is not None:
            raise self._counter.transcribe_error
        return _transcript()


class _TrackedDetector:
    _counter: _CallCounter

    def __init__(self, counter: _CallCounter) -> None:
        self._counter = counter

    def detect(
        self,
        video: Path,
        *,
        transcript: Transcript | None = None,
        chat: Sequence[ChatMessage] | None = None,
    ) -> tuple[Moment, ...]:
        _ = (video, transcript, chat)
        self._counter.detect_calls += 1
        if self._counter.detect_error is not None:
            raise self._counter.detect_error
        return (_moment(),)


class _TrackedModel:
    def select_segments(self, prompt: object) -> object:
        _ = (self, prompt)
        return {"segments": [{"start_s": 0.0, "end_s": 1.0}], "rationale": "resumen"}


class _TrackedSelector:
    _counter: _CallCounter

    def __init__(self, counter: _CallCounter) -> None:
        self._counter = counter

    def build_prompt(
        self, transcript: object, moments: object, contract: object
    ) -> dict[str, object]:
        _ = (transcript, moments, contract)
        self._counter.select_calls += 1
        return {}

    def parse_response(self, raw: object) -> SegmentSelection:
        _ = (self, raw)
        return _selection()


class _TrackedReframer:
    _counter: _CallCounter

    def __init__(self, counter: _CallCounter) -> None:
        self._counter = counter

    def analyze(self, video: Path) -> ReframeResult:
        _ = video
        self._counter.reframe_calls += 1
        if self._counter.reframe_error is not None:
            raise self._counter.reframe_error
        return _reframe_result()

    def render(self, *, video: Path, destination: Path, result: ReframeResult) -> Path:
        _ = (self, video, result)
        _ = destination.write_bytes(b"reframed_bytes")
        return destination


class _TrackedSubtitleRenderer:
    _counter: _CallCounter

    def __init__(self, counter: _CallCounter) -> None:
        self._counter = counter

    def write(self, words: Sequence[Word], destination: Path) -> Path:
        _ = words
        self._counter.subtitles_calls += 1
        if self._counter.subtitles_error is not None:
            raise self._counter.subtitles_error
        _ = destination.write_text("Dialogue: 0,0:00:00.00,0:00:01.00", encoding="utf-8")
        return destination

    def burn(self, *, video: Path, subtitles: Path, destination: Path) -> Path:
        _ = (self, video, subtitles)
        _ = destination.write_bytes(b"final_video_bytes")
        return destination


@dataclass(frozen=True, slots=True)
class _Injectables:
    model: _TrackedModel
    detector: _TrackedDetector
    transcriber: _TrackedTranscriber
    selector: _TrackedSelector
    reframer: _TrackedReframer
    subtitle_renderer: _TrackedSubtitleRenderer


def _mock_ffmpeg() -> object:
    def run(argv: list[str], **kwargs: object) -> object:
        _ = kwargs
        # escribe el destino para comandos ffmpeg simulados
        out_path = Path(argv[-1])
        out_path.parent.mkdir(parents=True, exist_ok=True)
        _ = out_path.write_bytes(b"ffmpeg_output")
        return SimpleNamespace(
            args=argv,
            returncode=0,
            stdout="width=320\nheight=240\nduration=2.0\n",
            stderr="",
        )

    return run


def _setup_pipeline(
    monkeypatch: pytest.MonkeyPatch, counter: _CallCounter, tmp_path: Path
) -> tuple[PipelineConfig, _Injectables]:
    monkeypatch.setattr("kliptych.orchestrator.subprocess.run", _mock_ffmpeg())

    def downloader_factory(*, timeout_s: float, max_size_bytes: int) -> _TrackedDownloader:
        _ = (timeout_s, max_size_bytes)
        return _TrackedDownloader(counter)

    monkeypatch.setattr("kliptych.orchestrator.MediaDownloader", downloader_factory)

    config = PipelineConfig(
        output_dir=tmp_path / "out",
        contract=_contract(),
        render=RenderConfig(),
    )
    deps = _Injectables(
        model=_TrackedModel(),
        detector=_TrackedDetector(counter),
        transcriber=_TrackedTranscriber(counter),
        selector=_TrackedSelector(counter),
        reframer=_TrackedReframer(counter),
        subtitle_renderer=_TrackedSubtitleRenderer(counter),
    )
    return config, deps


def _run(config: PipelineConfig, deps: _Injectables, *, resume: bool = False) -> PipelineResult:
    return run_long_video(
        _URL,
        config=config,
        model=deps.model,
        detector=deps.detector,
        transcriber=deps.transcriber,
        selector=deps.selector,
        reframer=deps.reframer,
        subtitle_renderer=deps.subtitle_renderer,
        resume=resume,
    )


def test_pipeline_state_manager_init_and_save(tmp_path: Path) -> None:
    out_dir = tmp_path / "out"
    mgr = PipelineStateManager(out_dir)
    assert not mgr.is_done(PipelineStage.DOWNLOAD)
    assert mgr.artifact_path(PipelineStage.DOWNLOAD) is None

    mgr.mark_done(PipelineStage.DOWNLOAD, out_dir / "source.mp4")
    assert mgr.is_done(PipelineStage.DOWNLOAD)
    assert mgr.artifact_path(PipelineStage.DOWNLOAD) == out_dir / "source.mp4"

    # Verificamos persistencia inmediata en disco
    loaded = PipelineStateManager.load(out_dir)
    assert loaded.is_done(PipelineStage.DOWNLOAD)
    assert loaded.artifact_path(PipelineStage.DOWNLOAD) == out_dir / "source.mp4"
    assert loaded.checkpoint.stages["download"] == "done"


def test_pipeline_state_manager_mark_failed(tmp_path: Path) -> None:
    out_dir = tmp_path / "out"
    mgr = PipelineStateManager(out_dir)
    mgr.mark_failed(PipelineStage.TRANSCRIBE)
    assert not mgr.is_done(PipelineStage.TRANSCRIBE)

    loaded = PipelineStateManager.load(out_dir)
    assert not loaded.is_done(PipelineStage.TRANSCRIBE)
    assert loaded.checkpoint.stages["transcribe"] == "failed"


def test_pipeline_state_manager_load_missing_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        _ = PipelineStateManager.load(tmp_path / "nonexistent")


def test_cleanup_registry_modes(tmp_path: Path) -> None:
    registry = _cleanup_registry()
    temp_file = tmp_path / "temp.part-123"
    artifact_file = tmp_path / "downloaded_artifact.mp4"

    _ = temp_file.write_bytes(b"temp")
    _ = artifact_file.write_bytes(b"artifact")

    _ = registry.register(temp_file, is_artifact=False)
    _ = registry.register(artifact_file, is_artifact=True)

    # Modo temp_only: conserva artefactos, elimina sólo verdaderos temporales
    removed = registry.cleanup(mode="temp_only")
    assert str(temp_file) in removed
    assert not temp_file.exists()
    assert artifact_file.exists()

    # Modo all: elimina también artefactos
    removed_all = registry.cleanup(mode="all")
    assert str(artifact_file) in removed_all
    assert not artifact_file.exists()


def test_checkpoint_tracks_completed_stages(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    counter = _CallCounter()
    config, deps = _setup_pipeline(monkeypatch, counter, tmp_path)

    result = _run(config, deps, resume=False)
    assert isinstance(result, PipelineResult)

    checkpoint_file = config.output_dir / "checkpoint.json"
    assert checkpoint_file.is_file()

    mgr = PipelineStateManager.load(config.output_dir)
    for stage in [
        PipelineStage.DOWNLOAD,
        PipelineStage.TRANSCRIBE,
        PipelineStage.MOMENTS,
        PipelineStage.SELECT,
        PipelineStage.REFRAME,
        PipelineStage.SUBTITLES,
        PipelineStage.COMPLETED,
    ]:
        assert mgr.is_done(stage), f"Stage {stage} debería estar completada"
        assert mgr.artifact_path(stage) is not None


def test_resume_skips_completed_stages(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    counter = _CallCounter()
    counter.detect_error = MomentDetectionError("fallo intencional en momentos")
    config, deps = _setup_pipeline(monkeypatch, counter, tmp_path)

    # Primer intento: falla en detect
    with pytest.raises(PipelineError):
        _ = _run(config, deps, resume=False)

    # Se ejecutaron download y transcribe antes del fallo
    assert counter.download_calls == 1
    assert counter.transcribe_calls == 1
    assert counter.detect_calls == 1

    # Verificamos estado en checkpoint: download y transcribe completados, moments fallido
    mgr = PipelineStateManager.load(config.output_dir)
    assert mgr.is_done(PipelineStage.DOWNLOAD)
    assert mgr.is_done(PipelineStage.TRANSCRIBE)
    assert not mgr.is_done(PipelineStage.MOMENTS)

    # Segundo intento: reanudar tras resolver el fallo
    counter.detect_error = None
    result = _run(config, deps, resume=True)
    assert isinstance(result, PipelineResult)

    # download y transcribe NO deben haberse vuelto a ejecutar (call count sigue siendo 1)
    assert counter.download_calls == 1
    assert counter.transcribe_calls == 1
    assert counter.detect_calls == 2


def test_failed_pipeline_keeps_artifacts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    counter = _CallCounter()
    counter.transcribe_error = TranscriptionError("fallo en transcripción")
    config, deps = _setup_pipeline(monkeypatch, counter, tmp_path)

    with pytest.raises(PipelineError):
        _ = _run(config, deps, resume=False)

    # El video descargado debe persistir para poder reanudar
    source_file = config.output_dir / "source.mp4"
    assert source_file.is_file()
    assert source_file.read_bytes() == b"source_video_bytes"

    # Los archivos temporales .part- no deben quedar huérfanos
    leftover_parts = [p.name for p in config.output_dir.iterdir() if ".part-" in p.name]
    assert leftover_parts == []


def test_successful_pipeline_cleans_all(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    counter = _CallCounter()
    config, deps = _setup_pipeline(monkeypatch, counter, tmp_path)

    result = _run(config, deps, resume=False)
    assert isinstance(result, PipelineResult)

    # Todos los temporales limpios
    leftover_parts = [p.name for p in config.output_dir.iterdir() if ".part-" in p.name]
    assert leftover_parts == []
    assert len(result.cleaning) > 0


def test_idempotent_transcription(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    counter = _CallCounter()
    config, deps = _setup_pipeline(monkeypatch, counter, tmp_path)

    # Pre-creamos transcript y checkpoint simulando descarga y transcripción ya hechas
    config.output_dir.mkdir(parents=True, exist_ok=True)
    source = config.output_dir / "source.mp4"
    _ = source.write_bytes(b"existing_source")

    transcript_file = config.output_dir / "transcript.json"
    _ = transcript_file.write_text(_transcript().model_dump_json(indent=2), encoding="utf-8")

    mgr = PipelineStateManager(config.output_dir)
    mgr.mark_done(PipelineStage.DOWNLOAD, source)
    mgr.mark_done(PipelineStage.TRANSCRIBE, transcript_file)

    # Ejecutamos con resume=True
    result = _run(config, deps, resume=True)
    assert isinstance(result, PipelineResult)

    # Transcriber nunca fue llamado gracias a la idempotencia
    assert counter.transcribe_calls == 0
    assert result.transcript == _transcript()


def test_corrupted_checkpoint_falls_back_to_new(tmp_path: Path) -> None:
    out_dir = tmp_path / "out"
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_file = out_dir / "checkpoint.json"
    _ = checkpoint_file.write_text("invalid json content", encoding="utf-8")

    mgr = PipelineStateManager(out_dir)
    assert mgr.checkpoint.stages == {}
    assert not mgr.is_done(PipelineStage.DOWNLOAD)


def test_cleanup_registry_invalid_mode_and_paths_setter(tmp_path: Path) -> None:
    registry = _cleanup_registry()
    path_a = tmp_path / "a.tmp"
    _ = path_a.write_bytes(b"a")
    registry.paths = [path_a]
    assert registry.paths == [path_a]

    with pytest.raises(ValueError, match="modo de limpieza"):
        _ = registry.cleanup(mode="invalid_mode")


def test_idempotent_moments_and_selection(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    counter = _CallCounter()
    config, deps = _setup_pipeline(monkeypatch, counter, tmp_path)

    config.output_dir.mkdir(parents=True, exist_ok=True)
    source = config.output_dir / "source.mp4"
    _ = source.write_bytes(b"existing_source")

    transcript_file = config.output_dir / "transcript.json"
    _ = transcript_file.write_text(_transcript().model_dump_json(indent=2), encoding="utf-8")

    moments_file = config.output_dir / "moments.json"
    _ = moments_file.write_text(f"[{_moment().model_dump_json()}]", encoding="utf-8")

    selection_file = config.output_dir / "selection.json"
    _ = selection_file.write_text(_selection().model_dump_json(indent=2), encoding="utf-8")

    # Ejecutamos con resume=True sin registrar previamente en checkpoint
    # (los archivos existen en disco)
    result = _run(config, deps, resume=True)
    assert isinstance(result, PipelineResult)

    assert counter.download_calls == 1  # download run because not in checkpoint
    assert counter.transcribe_calls == 0  # transcript.json exists on disk
    assert counter.detect_calls == 0  # moments.json exists on disk
    assert counter.select_calls == 0  # selection.json exists on disk
