"""Tests de pipeline reanudable con puntos de control (checkpointing).

Verifica que el estado del pipeline se persista tras cada etapa, que al
reanudar se salten las etapas ya completadas (idempotencia a nivel de etapa),
que los artefactos útiles se conserven ante fallos y que los temporales
puros se eliminen deterministamente.
"""

import os
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Protocol, cast, override

import pytest

from kliptych import orchestrator
from kliptych.__main__ import main
from kliptych.assets import AssetRegistry
from kliptych.campaign_manager import CampaignManager, CampaignOutcome
from kliptych.campaign_types import Campaign, CampaignStatus
from kliptych.contract import AudioPolicy, Contract, Segment
from kliptych.encoding import RenderConfig
from kliptych.gate import Gate
from kliptych.gc import clean_temporary_directories
from kliptych.intelligence import Archetype, ArchetypeClassification, CampaignClassifier
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
    SlideshowResult,
    compute_long_video_fingerprint,
    run_long_video,
    run_slideshow,
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
from tests.support import FakeProbe, make_media

if TYPE_CHECKING:
    from kliptych.git_proposals import ProposalEngine

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
    burn_mutes: list[bool] = field(default_factory=list)

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

    def burn(
        self, *, video: Path, subtitles: Path, destination: Path, mute_audio: bool = False
    ) -> Path:
        _ = (self, video, subtitles)
        self._counter.burn_mutes.append(mute_audio)
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
        if argv and argv[0] == "ffprobe":
            return SimpleNamespace(
                args=argv,
                returncode=0,
                stdout="width=320\nheight=240\nduration=2.0\n",
                stderr="",
            )
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

    # Segundo intento: reanudar tras resolver el fallo. La fuente remota se
    # revalida (una descarga más con los mismos bytes) pero las etapas
    # completadas no se re-ejecutan.
    counter.detect_error = None
    result = _run(config, deps, resume=True)
    assert isinstance(result, PipelineResult)

    assert counter.download_calls == 2
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
    mgr.set_input_fingerprint(compute_long_video_fingerprint(_URL, config=config))
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


def test_corrupt_or_empty_artifacts_trigger_reexecution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    counter = _CallCounter()
    config, deps = _setup_pipeline(monkeypatch, counter, tmp_path)
    config.output_dir.mkdir(parents=True, exist_ok=True)

    # Creamos artefactos vacíos o corruptos
    source = config.output_dir / "source.mp4"
    _ = source.write_bytes(b"")  # 0 bytes

    transcript_file = config.output_dir / "transcript.json"
    _ = transcript_file.write_text("{invalid json", encoding="utf-8")

    moments_file = config.output_dir / "moments.json"
    _ = moments_file.write_text("not json", encoding="utf-8")

    selection_file = config.output_dir / "selection.json"
    _ = selection_file.write_text("", encoding="utf-8")

    mgr = PipelineStateManager(config.output_dir)
    mgr.set_input_fingerprint(compute_long_video_fingerprint(_URL, config=config))
    mgr.mark_done(PipelineStage.DOWNLOAD, source)
    mgr.mark_done(PipelineStage.TRANSCRIBE, transcript_file)
    mgr.mark_done(PipelineStage.MOMENTS, moments_file)
    mgr.mark_done(PipelineStage.SELECT, selection_file)

    result = _run(config, deps, resume=True)
    assert isinstance(result, PipelineResult)

    # Debido a la corrupción o tamaño 0, todas las etapas deben haberse re-ejecutado
    assert counter.download_calls == 1
    assert counter.transcribe_calls == 1
    assert counter.detect_calls == 1
    assert counter.select_calls == 1


def test_slideshow_images_protected_from_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("kliptych.orchestrator.subprocess.run", _mock_ffmpeg())
    fake_img = tmp_path / "downloaded.jpg"
    _ = fake_img.write_bytes(b"image_bytes")
    track = tmp_path / "track.mp3"
    _ = track.write_bytes(b"audio_bytes")

    class _MockDownloader:
        def __init__(self, *, timeout_s: float, max_size_bytes: int) -> None:
            _ = (timeout_s, max_size_bytes)

        @staticmethod
        def download_video(
            *, url: str, destination: Path, format_selector: str | None = None
        ) -> Path:
            _ = (url, format_selector)
            destination.parent.mkdir(parents=True, exist_ok=True)
            _ = destination.write_bytes(fake_img.read_bytes())
            return destination

    monkeypatch.setattr("kliptych.orchestrator.MediaDownloader", _MockDownloader)

    config = PipelineConfig(
        output_dir=tmp_path / "slideshow_out",
        contract=_contract(),
        render=RenderConfig(),
        audio_locked=True,
        audio_track_path=track,
    )
    result = run_slideshow(
        ["https://example.com/slide1.jpg"],
        config=config,
        slide_duration_s=2.0,
    )
    assert isinstance(result, SlideshowResult)
    # EDR-001: las imágenes en result.images deben conservarse, no eliminarse
    assert len(result.images) == 1
    assert result.images[0].is_file()
    assert result.images[0].stat().st_size > 0


def test_audio_track_caching_and_stable_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    counter = _CallCounter()
    config, deps = _setup_pipeline(monkeypatch, counter, tmp_path)
    config = replace(
        config,
        audio_locked=True,
        audio_track_url="https://example.com/audio.mp3",
    )

    audio_downloads = 0

    class _TrackedAudioDownloader:
        def __init__(self, *, timeout_s: float, max_size_bytes: int) -> None:
            _ = (timeout_s, max_size_bytes)

        @staticmethod
        def download_video(
            *, url: str, destination: Path, format_selector: str | None = None
        ) -> Path:
            nonlocal audio_downloads
            _ = (url, format_selector)
            destination.parent.mkdir(parents=True, exist_ok=True)
            _ = destination.write_bytes(b"audio_bytes")
            if "audio" in url:
                audio_downloads += 1
            else:
                counter.download_calls += 1
            return destination

    monkeypatch.setattr("kliptych.orchestrator.MediaDownloader", _TrackedAudioDownloader)

    # Primer intento
    result = _run(config, deps, resume=False)
    assert isinstance(result, PipelineResult)
    assert audio_downloads == 1

    # La pista persiste en ruta estable para --resume
    audio_path = config.output_dir / "audio_track.mp3"
    assert audio_path.is_file()
    assert audio_path.read_bytes() == b"audio_bytes"

    # Segundo intento con resume=True: se revalida (una descarga más)
    # pero con los mismos bytes se reutilizan las etapas
    _ = _run(config, deps, resume=True)
    assert audio_downloads == 2
    assert audio_path.read_bytes() == b"audio_bytes"


def test_reframe_and_subtitles_idempotency_on_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    counter = _CallCounter()
    config, deps = _setup_pipeline(monkeypatch, counter, tmp_path)

    # Primer intento completo
    result1 = _run(config, deps, resume=False)
    assert isinstance(result1, PipelineResult)
    assert counter.reframe_calls == 1
    assert counter.subtitles_calls == 1

    # Segundo intento con resume=True
    result2 = _run(config, deps, resume=True)
    assert isinstance(result2, PipelineResult)
    # EDR-006: no deben haberse vuelto a llamar
    assert counter.reframe_calls == 1
    assert counter.subtitles_calls == 1


def test_reframe_and_subtitles_reexecute_if_corrupted_or_empty(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    counter = _CallCounter()
    config, deps = _setup_pipeline(monkeypatch, counter, tmp_path)

    # Primer intento completo
    _ = _run(config, deps, resume=False)
    assert counter.reframe_calls == 1
    assert counter.subtitles_calls == 1

    # Corrompemos el artefacto de reframe
    reframe_json = config.output_dir / "reframe.json"
    if reframe_json.exists():
        _ = reframe_json.write_text("corrupted", encoding="utf-8")
    reframed_video = config.output_dir / "reframed.mp4"
    if reframed_video.exists():
        _ = reframed_video.write_bytes(b"")

    final_video = config.output_dir / "final.mp4"
    _ = final_video.write_bytes(b"")

    # Segundo intento con resume=True
    _ = _run(config, deps, resume=True)
    assert counter.reframe_calls == 2
    assert counter.subtitles_calls == 2


class _DummyClassifier(CampaignClassifier):
    @override
    def classify(self, brief: str, contract: Contract) -> ArchetypeClassification:
        _ = (brief, contract)
        return ArchetypeClassification(
            archetype=Archetype.KNOWN,
            rationale="dummy",
        )


class _DummyProposalEngine:
    @staticmethod
    def propose(campaign: object, contract: object) -> object:
        _ = (campaign, contract)
        return None


class _MockVideoOrchestrator:
    def __init__(self, final_video: Path | None = None) -> None:
        self.long_video_kwargs: list[dict[str, object]] = []
        self.slideshow_kwargs: list[dict[str, object]] = []
        self.final_video: Path = final_video if final_video is not None else Path("final.mp4")

    def run_long_video(self, url: str, **kwargs: object) -> PipelineResult:
        self.long_video_kwargs.append({"url": url, **kwargs})
        return PipelineResult(
            source=Path("source.mp4"),
            transcript=None,
            moments=(),
            selection=SegmentSelection(
                segments=(Segment(start_s=0.0, end_s=1.0),), rationale="resumen"
            ),
            reframe=None,
            subtitles=None,
            final_video=self.final_video,
            cleaning=(),
        )

    def run_slideshow(self, images: Sequence[Path], **kwargs: object) -> SlideshowResult:
        self.slideshow_kwargs.append({"images": images, **kwargs})
        return SlideshowResult(
            images=tuple(images),
            slideshow_video=self.final_video,
            final_video=self.final_video,
            subtitles=None,
            cleaning=(),
        )


def test_campaign_manager_propagates_resume_flag(tmp_path: Path) -> None:
    video_path = tmp_path / "final.mp4"
    _ = video_path.write_bytes(b"video payload")
    video_orchestrator = _MockVideoOrchestrator(final_video=video_path)
    probe = FakeProbe(info=make_media(duration_s=10.0, has_video=True, has_audio=True))
    gate = Gate(probe=probe)
    proposal_engine = cast("ProposalEngine", cast("object", _DummyProposalEngine()))
    manager = CampaignManager(
        classifier=_DummyClassifier(),
        proposal_engine=proposal_engine,
        video_orchestrator=video_orchestrator,
        gate=gate,
        destination=tmp_path / "delivery",
        assets=AssetRegistry(tmp_path),
    )
    campaign = Campaign(
        campaign_id="test-camp",
        brief="brief",
        status=CampaignStatus.PENDING,
        contract=_contract(),
    )

    # Modo long_video con resume=True
    outcome = manager.process(campaign, mode="long_video", url=_URL, resume=True)
    assert outcome.status == CampaignStatus.COMPLETED
    assert len(video_orchestrator.long_video_kwargs) == 1
    assert video_orchestrator.long_video_kwargs[0].get("resume") is True

    # Modo slideshow con resume=True
    outcome_slide = manager.process(
        campaign, mode="slideshow", images=[tmp_path / "img.jpg"], resume=True
    )
    assert outcome_slide.status == CampaignStatus.COMPLETED
    assert len(video_orchestrator.slideshow_kwargs) == 1
    assert video_orchestrator.slideshow_kwargs[0].get("resume") is True


def test_cli_resume_and_restart_flags(tmp_path: Path) -> None:
    brief = tmp_path / "brief.txt"
    _ = brief.write_text("test brief", encoding="utf-8")

    class _FakeManager:
        def __init__(self) -> None:
            self.last_resume: bool | None = None

        def process(
            self,
            campaign: Campaign,
            *,
            mode: str = "long_video",
            url: str | None = None,
            images: Sequence[Path] | None = None,
            resume: bool = False,
            destination: Path | None = None,
            approve_manual_review: bool = False,
            **kwargs: object,
        ) -> CampaignOutcome:
            _ = (campaign, mode, url, images, destination, approve_manual_review, kwargs)
            self.last_resume = resume
            return CampaignOutcome(
                campaign_id="test",
                archetype=Archetype.KNOWN,
                status=CampaignStatus.COMPLETED,
            )

    fake_mgr = _FakeManager()
    # --resume flag
    code = main(
        ["campaign", str(brief), "--out", str(tmp_path / "out"), "--resume"],
        manager=fake_mgr,
    )
    assert code == 0
    assert fake_mgr.last_resume is True

    # --restart flag
    code = main(
        ["campaign", str(brief), "--out", str(tmp_path / "out"), "--restart"],
        manager=fake_mgr,
    )
    assert code == 0
    assert fake_mgr.last_resume is False

    # Ambos flags a la vez -> error 2 (mutualmente excluyentes)
    with pytest.raises(SystemExit) as exc_info:
        _ = main(
            ["campaign", str(brief), "--out", str(tmp_path / "out"), "--resume", "--restart"],
            manager=fake_mgr,
        )
    assert exc_info.value.code == 2


def test_gc_clean_temporary_directories_and_cli(tmp_path: Path) -> None:
    campaigns = tmp_path / "campaigns"
    campaigns.mkdir(parents=True, exist_ok=True)

    # Subcarpetas temporales
    old_tmp = campaigns / "tmp_old_dir"
    old_tmp.mkdir()
    _ = (old_tmp / "file.tmp").write_bytes(b"data")

    old_dot_tmp = campaigns / ".tmp-old_part"
    old_dot_tmp.mkdir()

    new_tmp = campaigns / "tmp_new_dir"
    new_tmp.mkdir()

    # Carpetas y archivos no temporales protegidos
    fixtures_dir = campaigns / "fixtures"
    fixtures_dir.mkdir()
    _ = (fixtures_dir / "test.json").write_bytes(b"{}")

    pending_dir = campaigns / "pending"
    pending_dir.mkdir()

    variations_file = campaigns / "variations.md"
    _ = variations_file.write_text("# Variations", encoding="utf-8")

    # Modificamos mtime para simular antigüedad (10 días atrás)
    ten_days_ago = time.time() - (10 * 86400)

    os.utime(old_tmp, (ten_days_ago, ten_days_ago))
    os.utime(old_dot_tmp, (ten_days_ago, ten_days_ago))

    # Ejecutamos comando CLI clean
    code = main(["clean", "--days", "7", "--root", str(campaigns)])
    assert code == 0

    # Las carpetas viejas fueron eliminadas
    assert not old_tmp.exists()
    assert not old_dot_tmp.exists()

    # Las carpetas y archivos protegidos o recientes siguen existiendo
    assert new_tmp.exists()
    assert fixtures_dir.exists()
    assert (fixtures_dir / "test.json").exists()
    assert pending_dir.exists()
    assert variations_file.exists()


def test_gc_nested_and_dry_run_and_errors(tmp_path: Path) -> None:
    # Error en days negativo
    with pytest.raises(ValueError, match="mayor o igual a 0"):
        _ = clean_temporary_directories(tmp_path, days=-1.0)

    # Directorio inexistente devuelve tupla vacía
    assert clean_temporary_directories(tmp_path / "nonexistent", days=0) == ()

    # Subdirectorio anidado en carpeta no protegida
    private_dir = tmp_path / "private"
    private_dir.mkdir()
    nested_tmp = private_dir / "tmp_nested"
    nested_tmp.mkdir()

    # Dry-run identifica pero no elimina
    found = clean_temporary_directories(tmp_path, days=0.0, dry_run=True)
    assert nested_tmp in found
    assert nested_tmp.exists()

    # Ejecución normal elimina
    removed = clean_temporary_directories(tmp_path, days=0.0, dry_run=False)
    assert nested_tmp in removed
    assert not nested_tmp.exists()


def test_cli_clean_edge_cases(tmp_path: Path) -> None:
    # Root inexistente
    code_nonexistent = main(["clean", "--root", str(tmp_path / "missing")])
    assert code_nonexistent == 0

    # Días negativos -> error 1
    code_invalid = main(["clean", "--days", "-2", "--root", str(tmp_path)])
    assert code_invalid == 1


def test_cli_run_subcommand_resume_and_restart_flags(tmp_path: Path) -> None:
    brief = tmp_path / "brief.txt"
    _ = brief.write_text("test brief", encoding="utf-8")

    with pytest.raises(SystemExit) as exc_info:
        _ = main(["run", str(brief), "--out", str(tmp_path / "out"), "--resume", "--restart"])
    assert exc_info.value.code == 2


def test_resume_invalidates_cache_on_fingerprint_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    counter = _CallCounter()
    config, deps = _setup_pipeline(monkeypatch, counter, tmp_path)

    # 1. Primera ejecución limpia
    result1 = _run(config, deps, resume=False)
    assert isinstance(result1, PipelineResult)
    assert counter.download_calls == 1

    # Verificar que el checkpoint persistió con el fingerprint
    mgr1 = PipelineStateManager.load(config.output_dir)
    initial_fp = mgr1.checkpoint.input_fingerprint
    assert initial_fp is not None
    assert mgr1.is_done(PipelineStage.COMPLETED)

    # 2. Modificamos un parámetro de entrada (audio_mix_ratio de 1.0 a 0.5)
    config_modified = replace(config, audio_mix_ratio=0.5)

    # 3. Segunda ejecución con resume=True
    result2 = _run(config_modified, deps, resume=True)
    assert isinstance(result2, PipelineResult)

    # Verificamos que el fingerprint cambió y las etapas se re-ejecutaron
    mgr2 = PipelineStateManager.load(config.output_dir)
    assert mgr2.checkpoint.input_fingerprint != initial_fp
    assert counter.download_calls == 2


def _muted_contract() -> Contract:
    base = _contract()
    rules = base.rules.model_copy(
        update={"manual_review": [*base.rules.manual_review, "audio.policy"]}
    )
    return base.model_copy(
        update={"audio_policy": AudioPolicy.INTERNAL_OFFICIAL_SOUND, "rules": rules}
    )


def test_resume_regenerates_muted_final_after_policy_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    counter = _CallCounter()
    config, deps = _setup_pipeline(monkeypatch, counter, tmp_path)

    # 1. Primera ejecución limpia con audio audible.
    result1 = _run(config, deps, resume=False)
    assert isinstance(result1, PipelineResult)
    assert counter.burn_mutes == [False]

    mgr1 = PipelineStateManager.load(config.output_dir)
    initial_fp = mgr1.checkpoint.input_fingerprint
    assert initial_fp is not None
    assert mgr1.is_done(PipelineStage.COMPLETED)

    # 2. La campaña cambia su política a sonido oficial interno.
    muted_config = replace(config, contract=_muted_contract())

    # 3. Reanudar invalida el checkpoint previo y regenera el final silenciado.
    result2 = _run(muted_config, deps, resume=True)
    assert isinstance(result2, PipelineResult)
    mgr2 = PipelineStateManager.load(config.output_dir)
    assert mgr2.checkpoint.input_fingerprint != initial_fp
    assert counter.download_calls == 2
    assert counter.burn_mutes == [False, True]


def test_resume_invalidates_cache_when_audio_file_content_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    counter = _CallCounter()
    config, deps = _setup_pipeline(monkeypatch, counter, tmp_path)
    track = tmp_path / "audio.mp3"
    _ = track.write_bytes(b"audio content v1")
    config = replace(config, audio_locked=True, audio_track_path=track)

    # 1. Primera ejecución limpia
    result1 = _run(config, deps, resume=False)
    assert isinstance(result1, PipelineResult)
    assert counter.download_calls == 1

    # Verificar que el checkpoint persistió con el fingerprint
    mgr1 = PipelineStateManager.load(config.output_dir)
    initial_fp = mgr1.checkpoint.input_fingerprint
    assert initial_fp is not None
    assert mgr1.is_done(PipelineStage.COMPLETED)

    # 2. Modificamos el contenido del archivo de audio (mismo path, diferentes bytes)
    _ = track.write_bytes(b"audio content v2 changed")

    # 3. Segunda ejecución con resume=True
    result2 = _run(config, deps, resume=True)
    assert isinstance(result2, PipelineResult)

    # Verificamos que el fingerprint cambió y las etapas se re-ejecutaron
    mgr2 = PipelineStateManager.load(config.output_dir)
    assert mgr2.checkpoint.input_fingerprint != initial_fp
    assert counter.download_calls == 2


def test_resume_binds_secondary_segments_and_removes_leftovers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verifica que resume invalide segmentos cuyas coordenadas cambiaron y limpie sobrantes."""
    counter = _CallCounter()
    config, deps = _setup_pipeline(monkeypatch, counter, tmp_path)

    # Run 1: dos segmentos (0.0, 5.0) y (10.0, 15.0)
    selection1 = SegmentSelection(
        segments=(
            Segment(start_s=0.0, end_s=5.0),
            Segment(start_s=10.0, end_s=15.0),
        ),
        rationale="r1",
    )

    def _parse1(_raw: object) -> SegmentSelection:
        return selection1

    monkeypatch.setattr(deps.selector, "parse_response", _parse1)

    result1 = _run(config, deps, resume=False)
    assert len(result1.final_videos) == 2
    f0 = config.output_dir / "final_00.mp4"
    f1 = config.output_dir / "final_01.mp4"
    assert f0.is_file()
    assert f1.is_file()
    # Guardamos contenido inicial de final_01 para detectar si se re-renderizó
    f0_bytes_initial = b"final_video_run1_seg0"
    _ = f0.write_bytes(f0_bytes_initial)
    f1_bytes_initial = b"final_video_run1"
    _ = f1.write_bytes(f1_bytes_initial)

    # Creamos un sobrante final_02.mp4 de una supuesta corrida anterior con 3 segmentos
    leftover = config.output_dir / "final_02.mp4"
    _ = leftover.write_bytes(b"leftover_final_02")

    # Run 2: segmento 0 idéntico (0.0, 5.0), segmento 1 con coordenadas cambiadas (12.0, 17.0)
    selection2 = SegmentSelection(
        segments=(
            Segment(start_s=0.0, end_s=5.0),
            Segment(start_s=12.0, end_s=17.0),
        ),
        rationale="r2",
    )

    def _parse2(_raw: object) -> SegmentSelection:
        return selection2

    monkeypatch.setattr(deps.selector, "parse_response", _parse2)
    _ = (config.output_dir / "selection.json").write_text(
        selection2.model_dump_json(indent=2), encoding="utf-8"
    )

    result2 = _run(config, deps, resume=True)
    assert len(result2.final_videos) == 2
    assert result2.final_videos == (f0, f1)

    # El sobrante final_02 debe haber sido eliminado
    assert not leftover.exists()

    # final_00 debió reutilizarse intacto
    assert f0.read_bytes() == f0_bytes_initial

    # final_01 debe haberse re-renderizado porque sus coordenadas cambiaron
    assert f1.read_bytes() != f1_bytes_initial


def test_resume_invalidates_downstream_stages_when_source_content_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verifica que resume invalide etapas descendentes si los bytes remotos cambian.

    El servidor es la fuente de verdad: en --resume se re-descarga la URL y,
    si los bytes difieren, el local se actualiza y las etapas hijas se
    re-ejecutan.
    """
    counter = _CallCounter()
    config, deps = _setup_pipeline(monkeypatch, counter, tmp_path)

    # 1. Primera corrida completa
    result1 = _run(config, deps, resume=False)
    assert isinstance(result1, PipelineResult)
    assert counter.transcribe_calls == 1
    assert counter.reframe_calls == 1
    assert counter.subtitles_calls == 1

    source_file = config.output_dir / "source.mp4"
    assert source_file.is_file()

    # 2. El servidor cambia los bytes del video fuente entre corridas
    def downloader_factory_v2(*, timeout_s: float, max_size_bytes: int) -> _TrackedDownloader:
        _ = (timeout_s, max_size_bytes)

        class _ChangedDownloader(_TrackedDownloader):
            @override
            def download_video(
                self, *, url: str, destination: Path, format_selector: str | None = None
            ) -> Path:
                _ = (url, format_selector)
                self._counter.download_calls += 1
                _ = destination.write_bytes(b"completely_new_source_video_bytes_v2")
                return destination

        return _ChangedDownloader(counter)

    monkeypatch.setattr("kliptych.orchestrator.MediaDownloader", downloader_factory_v2)

    # 3. Segunda corrida con resume=True
    result2 = _run(config, deps, resume=True)
    assert isinstance(result2, PipelineResult)

    # El local se actualizó con los bytes del servidor y las etapas
    # dependientes se re-ejecutaron por el cambio de contenido
    assert counter.download_calls == 2
    assert source_file.read_bytes() == b"completely_new_source_video_bytes_v2"
    assert counter.transcribe_calls == 2
    assert counter.reframe_calls == 2
    assert counter.subtitles_calls == 2
