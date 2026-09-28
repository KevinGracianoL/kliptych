"""Tests del idioma explícito de transcripción (Fix 2 Sprint 1).

Sin ``language``, faster-whisper autodetecta el idioma y el contenido en
español puede sufrir alucinaciones o deriva de idioma. Estos tests exigen que
el contrato declare el locale, que llegue al motor y que un cambio de idioma
invalide la caché de ``--resume`` sin romper los hashes ya grabados.
"""

import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, cast

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

import pytest

from kliptych import orchestrator
from kliptych import transcribe as transcribe_module
from kliptych.contract import Contract, contract_digest
from kliptych.encoding import RenderConfig
from kliptych.orchestrator import PipelineConfig, compute_long_video_fingerprint
from kliptych.transcribe import FasterWhisperTranscriber, Transcriber, TranscriptionError
from tests.support import make_contract


@dataclass
class _FakeWord:
    start: float
    end: float
    word: str
    probability: float


@dataclass
class _FakeSegment:
    words: list[_FakeWord] | None


@dataclass
class _FakeInfo:
    language: str = "es"
    duration: float = 5.0


class _FakeEngine:
    """Motor falso que registra si recibió el idioma explícito."""

    def __init__(self, words: list[_FakeWord] | None = None) -> None:
        self._words: list[_FakeWord] = list(words or [])
        self.calls: list[Mapping[str, object]] = []

    def transcribe(self, audio: str, **kwargs: object) -> tuple[list[_FakeSegment], _FakeInfo]:
        _ = audio
        self.calls.append(dict(kwargs))
        return [_FakeSegment(list(self._words))], _FakeInfo()


class _LegacyEngine:
    """Motor antiguo sin parámetro ``language``: solo autodetección."""

    def __init__(self, words: list[_FakeWord] | None = None) -> None:
        self._words: list[_FakeWord] = list(words or [])

    def transcribe(
        self, audio: str, *, word_timestamps: bool = False
    ) -> tuple[list[_FakeSegment], _FakeInfo]:
        _ = (audio, word_timestamps)
        return [_FakeSegment(list(self._words))], _FakeInfo()


class _ResolvedDeps(Protocol):
    """Superficie de las dependencias resueltas que usan los tests."""

    transcriber: Transcriber


def _private(name: str) -> object:
    return cast("object", getattr(orchestrator, name))


_resolve_dependencies = cast("Callable[..., _ResolvedDeps]", _private("_resolve_dependencies"))


def _install(monkeypatch: pytest.MonkeyPatch, engine: object) -> None:
    transcribe_module.clear_model_cache()

    def whisper_model(
        model_size_or_path: str, *, device: str, compute_type: str, num_workers: int
    ) -> object:
        _ = (model_size_or_path, device, compute_type, num_workers)
        return engine

    module = types.SimpleNamespace(WhisperModel=whisper_model)
    monkeypatch.setitem(sys.modules, "faster_whisper", module)


def _audio(tmp_path: Path) -> Path:
    path = tmp_path / "audio.wav"
    _ = path.write_bytes(b"RIFF")
    return path


def test_language_reaches_faster_whisper(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    engine = _FakeEngine(words=[_FakeWord(0.0, 0.5, " hola", 0.9)])
    _install(monkeypatch, engine)
    transcript = FasterWhisperTranscriber(language="es").transcribe(_audio(tmp_path))
    assert transcript.text == "hola"
    assert engine.calls == [{"word_timestamps": True, "language": "es"}]


def test_unset_language_keeps_legacy_call(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    engine = _LegacyEngine(words=[_FakeWord(0.0, 0.5, " hola", 0.9)])
    _install(monkeypatch, engine)
    transcript = FasterWhisperTranscriber().transcribe(_audio(tmp_path))
    assert transcript.text == "hola"


@pytest.mark.parametrize("language", ["es", "en", "pt-br"])
def test_supported_language_codes_accepted(language: str) -> None:
    assert FasterWhisperTranscriber(language=language) is not None


@pytest.mark.parametrize("language", ["", "ES", "e", "español", "e1", "es-ES!"])
def test_invalid_language_codes_rejected(language: str) -> None:
    with pytest.raises(ValueError, match="idioma"):
        _ = FasterWhisperTranscriber(language=language)


def test_contract_accepts_language() -> None:
    contract = make_contract(language="es")
    assert contract.languages.language == "es"


def test_contract_rejects_invalid_language() -> None:
    with pytest.raises(ValueError, match="pattern"):
        _ = make_contract(language="ES")


def test_unset_language_keeps_recorded_digest() -> None:
    assert contract_digest(make_contract()) == contract_digest(make_contract(language=None))


def test_declared_language_changes_digest() -> None:
    assert contract_digest(make_contract(language="es")) != contract_digest(make_contract())
    assert contract_digest(make_contract(language="es")) != contract_digest(
        make_contract(language="en")
    )


def test_contract_without_language_key_matches_with_none() -> None:
    data: dict[str, object] = make_contract(language=None).model_dump(mode="json")
    languages = cast("dict[str, object]", data["languages"])
    _ = languages.pop("language", None)
    restored = Contract.model_validate(data)
    assert contract_digest(restored) == contract_digest(make_contract(language=None))


def _config(tmp_path: Path, *, language: str | None) -> PipelineConfig:
    return PipelineConfig(
        output_dir=tmp_path / "out",
        contract=make_contract(language=language),
        render=RenderConfig(),
        face_model_path=tmp_path / "face.tflite",
    )


def test_language_change_invalidates_resume_fingerprint(tmp_path: Path) -> None:
    before = compute_long_video_fingerprint(
        "https://example.com/video.mp4", config=_config(tmp_path, language=None)
    )
    after = compute_long_video_fingerprint(
        "https://example.com/video.mp4", config=_config(tmp_path, language="es")
    )
    assert before != after


def test_same_language_keeps_resume_fingerprint(tmp_path: Path) -> None:
    first = compute_long_video_fingerprint(
        "https://example.com/video.mp4", config=_config(tmp_path, language="es")
    )
    second = compute_long_video_fingerprint(
        "https://example.com/video.mp4", config=_config(tmp_path, language="es")
    )
    assert first == second


def test_orchestrator_builds_transcriber_with_contract_language(tmp_path: Path) -> None:
    dependencies = _resolve_dependencies(
        config=_config(tmp_path, language="es"),
        detector=None,
        transcriber=None,
        selector=None,
        reframer=None,
        subtitle_renderer=None,
    )
    transcriber = dependencies.transcriber
    assert isinstance(transcriber, FasterWhisperTranscriber)
    assert transcriber.language == "es"


def test_orchestrator_default_transcriber_autodetects_without_language(
    tmp_path: Path,
) -> None:
    dependencies = _resolve_dependencies(
        config=_config(tmp_path, language=None),
        detector=None,
        transcriber=None,
        selector=None,
        reframer=None,
        subtitle_renderer=None,
    )
    transcriber = dependencies.transcriber
    assert isinstance(transcriber, FasterWhisperTranscriber)
    assert transcriber.language is None


def test_transcript_words_untouched_by_language(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    engine = _FakeEngine(
        words=[_FakeWord(0.0, 0.5, " hola", 0.8), _FakeWord(0.6, 1.2, " mundo", 0.7)]
    )
    _install(monkeypatch, engine)
    transcript = FasterWhisperTranscriber(language="es").transcribe(_audio(tmp_path))
    assert [(w.text, w.confidence) for w in transcript.words] == [
        ("hola", 0.8),
        ("mundo", 0.7),
    ]
    assert transcript.language == "es"


def test_missing_engine_still_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    transcribe_module.clear_model_cache()
    monkeypatch.setitem(sys.modules, "faster_whisper", None)
    with pytest.raises(TranscriptionError):
        _ = FasterWhisperTranscriber(language="es").transcribe(_audio(tmp_path))
