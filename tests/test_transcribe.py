"""Tests unitarios de transcripción sin red ni descarga del modelo real.

Se inyecta un módulo ``faster_whisper`` falso en ``sys.modules`` para que
``FasterWhisperTranscriber`` cargue un motor controlado; ningún test descarga
el modelo real ni toca la red.
"""

import sys
import types
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import pytest
from pydantic import ValidationError

from kliptych.transcribe import (
    FasterWhisperTranscriber,
    Transcript,
    TranscriptionError,
    Word,
)


@dataclass
class FakeWord:
    start: float
    end: float
    word: str
    probability: float


class FakeSegment:
    def __init__(self, words: Sequence[FakeWord] | None) -> None:
        self.words: list[FakeWord] | None = None if words is None else list(words)


@dataclass
class FakeInfo:
    language: str = "es"
    duration: float = 5.0


class FakeEngine:
    """Motor falso que registra la llamada y devuelve palabras controladas."""

    def __init__(
        self,
        *,
        words: Sequence[FakeWord] = (),
        language: str = "es",
        duration: float = 5.0,
        error: BaseException | None = None,
        words_is_none: bool = False,
    ) -> None:
        self._words: list[FakeWord] = list(words)
        self._language: str = language
        self._duration: float = duration
        self._error: BaseException | None = error
        self._words_is_none: bool = words_is_none
        self.constructed: dict[str, object] = {}
        self.audio: str | None = None
        self.word_timestamps: bool | None = None

    def transcribe(
        self,
        audio: str,
        *,
        word_timestamps: bool = False,
    ) -> tuple[list[FakeSegment], FakeInfo]:
        self.audio = audio
        self.word_timestamps = word_timestamps
        if self._error is not None:
            raise self._error
        words: list[FakeWord] | None = None if self._words_is_none else self._words
        return [FakeSegment(words)], FakeInfo(self._language, self._duration)


def _install_engine(monkeypatch: pytest.MonkeyPatch, engine: FakeEngine) -> FakeEngine:
    def whisper_model(
        model_size_or_path: str,
        *,
        device: str,
        compute_type: str,
        num_workers: int,
    ) -> FakeEngine:
        engine.constructed = {
            "model_size_or_path": model_size_or_path,
            "device": device,
            "compute_type": compute_type,
            "num_workers": num_workers,
        }
        return engine

    module = types.SimpleNamespace(WhisperModel=whisper_model)
    monkeypatch.setitem(sys.modules, "faster_whisper", module)
    return engine


def _audio(tmp_path: Path) -> Path:
    path = tmp_path / "audio.wav"
    _ = path.write_bytes(b"RIFF")
    return path


def _word(start: float, end: float, text: str, probability: float = 0.9) -> FakeWord:
    return FakeWord(start, end, text, probability)


def test_default_constructor_uses_small_int8(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    engine = _install_engine(monkeypatch, FakeEngine(words=[_word(0.0, 0.5, " hola")]))
    transcriber = FasterWhisperTranscriber()
    _ = transcriber.transcribe(_audio(tmp_path))
    assert engine.constructed == {
        "model_size_or_path": "small",
        "device": "auto",
        "compute_type": "int8",
        "num_workers": 1,
    }


@pytest.mark.parametrize("model_size", ["", "gigante", "large-v3"])
def test_invalid_model_size_rejected(model_size: str) -> None:
    with pytest.raises(ValueError, match="modelo"):
        _ = FasterWhisperTranscriber(model_size=model_size)


@pytest.mark.parametrize("compute_type", ["", "float64", "int4"])
def test_invalid_compute_type_rejected(compute_type: str) -> None:
    with pytest.raises(ValueError, match="cuantización"):
        _ = FasterWhisperTranscriber(compute_type=compute_type)


def test_invalid_device_rejected() -> None:
    with pytest.raises(ValueError, match="dispositivo"):
        _ = FasterWhisperTranscriber(device="tpu")


def test_non_positive_workers_rejected() -> None:
    with pytest.raises(ValueError, match="workers"):
        _ = FasterWhisperTranscriber(num_workers=0)


def test_missing_audio_fails_before_loading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    engine = _install_engine(monkeypatch, FakeEngine())
    transcriber = FasterWhisperTranscriber()
    with pytest.raises(TranscriptionError, match="no existe"):
        _ = transcriber.transcribe(tmp_path / "falta.wav")
    assert engine.constructed == {}


def test_transcribe_returns_valid_transcript(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = _install_engine(
        monkeypatch,
        FakeEngine(
            words=[_word(0.0, 0.5, " hola", 0.8), _word(0.6, 1.2, " mundo", 0.7)],
            language="es",
            duration=1.5,
        ),
    )
    transcript = FasterWhisperTranscriber().transcribe(_audio(tmp_path))
    assert isinstance(transcript, Transcript)
    assert engine.word_timestamps is True
    assert transcript.language == "es"
    assert transcript.duration_s == pytest.approx(1.5)
    assert transcript.text == "hola mundo"
    assert [(w.start_s, w.end_s, w.text, w.confidence, w.token_id) for w in transcript.words] == [
        (0.0, 0.5, "hola", 0.8, 0),
        (0.6, 1.2, "mundo", 0.7, 1),
    ]


def test_token_ids_are_sequential_from_zero(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    words = [_word(index, index + 0.4, f" w{index}") for index in range(5)]
    _ = _install_engine(monkeypatch, FakeEngine(words=words))
    transcript = FasterWhisperTranscriber().transcribe(_audio(tmp_path))
    assert [word.token_id for word in transcript.words] == [0, 1, 2, 3, 4]


def test_empty_words_raise(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _ = _install_engine(monkeypatch, FakeEngine(words=[]))
    with pytest.raises(TranscriptionError, match="no se detectaron palabras"):
        _ = FasterWhisperTranscriber().transcribe(_audio(tmp_path))


def test_missing_segment_words_raise(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _ = _install_engine(monkeypatch, FakeEngine(words_is_none=True))
    with pytest.raises(TranscriptionError, match="no se detectaron palabras"):
        _ = FasterWhisperTranscriber().transcribe(_audio(tmp_path))


def test_blank_word_text_is_skipped(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _ = _install_engine(
        monkeypatch,
        FakeEngine(words=[_word(0.0, 0.4, "   "), _word(0.5, 0.9, " eco")]),
    )
    transcript = FasterWhisperTranscriber().transcribe(_audio(tmp_path))
    assert [word.text for word in transcript.words] == ["eco"]
    assert transcript.text == "eco"


def test_engine_error_becomes_transcription_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _ = _install_engine(monkeypatch, FakeEngine(error=RuntimeError("sin cuBLAS")))
    with pytest.raises(TranscriptionError) as excinfo:
        _ = FasterWhisperTranscriber().transcribe(_audio(tmp_path))
    assert isinstance(excinfo.value.__cause__, RuntimeError)


_RELEASE_ATTR = "_release_model"
_MODEL_ATTR = "_model"


def _release_hook() -> Callable[[FasterWhisperTranscriber], None]:
    return cast(
        "Callable[[FasterWhisperTranscriber], None]",
        getattr(FasterWhisperTranscriber, _RELEASE_ATTR),
    )


def _model_of(transcriber: FasterWhisperTranscriber) -> object:
    return cast("object", getattr(transcriber, _MODEL_ATTR))


def test_release_model_runs_on_success(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _ = _install_engine(monkeypatch, FakeEngine(words=[_word(0.0, 0.5, " hola")]))
    transcriber = FasterWhisperTranscriber()
    calls: list[None] = []
    original = _release_hook()

    def spy(self: FasterWhisperTranscriber) -> None:
        calls.append(None)
        original(self)

    monkeypatch.setattr(FasterWhisperTranscriber, "_release_model", spy)
    _ = transcriber.transcribe(_audio(tmp_path))
    assert len(calls) == 1
    assert _model_of(transcriber) is None


def test_release_model_runs_on_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _ = _install_engine(monkeypatch, FakeEngine(error=RuntimeError("falla")))
    transcriber = FasterWhisperTranscriber()
    calls: list[None] = []
    original = _release_hook()

    def spy(self: FasterWhisperTranscriber) -> None:
        calls.append(None)
        original(self)

    monkeypatch.setattr(FasterWhisperTranscriber, "_release_model", spy)
    with pytest.raises(TranscriptionError):
        _ = transcriber.transcribe(_audio(tmp_path))
    assert len(calls) == 1
    assert _model_of(transcriber) is None


def test_release_model_empties_cuda_cache_when_available(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _ = _install_engine(monkeypatch, FakeEngine(words=[_word(0.0, 0.5, " hola")]))
    emptied: list[None] = []

    class FakeCuda:
        @staticmethod
        def is_available() -> bool:
            return True

        @staticmethod
        def empty_cache() -> None:
            emptied.append(None)

    class FakeTorch:
        cuda: type[FakeCuda] = FakeCuda

    monkeypatch.setitem(sys.modules, "torch", FakeTorch)
    _ = FasterWhisperTranscriber().transcribe(_audio(tmp_path))
    assert len(emptied) == 1


def test_release_model_skips_cache_without_cuda(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _ = _install_engine(monkeypatch, FakeEngine(words=[_word(0.0, 0.5, " hola")]))
    emptied: list[None] = []

    class FakeCuda:
        @staticmethod
        def is_available() -> bool:
            return False

        @staticmethod
        def empty_cache() -> None:
            emptied.append(None)

    class FakeTorch:
        cuda: type[FakeCuda] = FakeCuda

    monkeypatch.setitem(sys.modules, "torch", FakeTorch)
    _ = FasterWhisperTranscriber().transcribe(_audio(tmp_path))
    assert emptied == []


def test_release_model_survives_missing_torch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _ = _install_engine(monkeypatch, FakeEngine(words=[_word(0.0, 0.5, " hola")]))
    monkeypatch.setitem(sys.modules, "torch", None)
    transcript = FasterWhisperTranscriber().transcribe(_audio(tmp_path))
    assert transcript.text == "hola"


def test_missing_faster_whisper_becomes_transcription_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "faster_whisper", None)
    with pytest.raises(TranscriptionError):
        _ = FasterWhisperTranscriber().transcribe(_audio(tmp_path))


def test_module_without_whisper_model_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "faster_whisper", types.SimpleNamespace())
    with pytest.raises(TranscriptionError, match="WhisperModel"):
        _ = FasterWhisperTranscriber().transcribe(_audio(tmp_path))


def test_release_model_survives_torch_without_cuda(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _ = _install_engine(monkeypatch, FakeEngine(words=[_word(0.0, 0.5, " hola")]))
    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace())
    transcript = FasterWhisperTranscriber().transcribe(_audio(tmp_path))
    assert transcript.text == "hola"


def test_word_negative_start_rejected() -> None:
    with pytest.raises(ValidationError):
        _ = Word(start_s=-0.1, end_s=0.5, text="hola", confidence=0.5, token_id=0)


def test_word_confidence_above_one_rejected() -> None:
    with pytest.raises(ValidationError):
        _ = Word(start_s=0.0, end_s=0.5, text="hola", confidence=1.5, token_id=0)


def test_word_negative_token_id_rejected() -> None:
    with pytest.raises(ValidationError):
        _ = Word(start_s=0.0, end_s=0.5, text="hola", confidence=0.5, token_id=-1)


def test_word_empty_text_rejected() -> None:
    with pytest.raises(ValidationError):
        _ = Word(start_s=0.0, end_s=0.5, text="", confidence=0.5, token_id=0)


_VALID_WORD = Word(start_s=0.0, end_s=0.5, text="hola", confidence=0.5, token_id=0)


def test_transcript_extra_fields_rejected() -> None:
    with pytest.raises(ValidationError):
        _ = Transcript.model_validate(
            {
                "words": (_VALID_WORD,),
                "language": "es",
                "duration_s": 1.0,
                "text": "hola",
                "extra": "no",
            }
        )


def test_transcript_empty_language_rejected() -> None:
    with pytest.raises(ValidationError):
        _ = Transcript(words=(_VALID_WORD,), language="", duration_s=1.0, text="hola")


def test_transcript_empty_text_rejected() -> None:
    with pytest.raises(ValidationError):
        _ = Transcript(words=(_VALID_WORD,), language="es", duration_s=1.0, text="")
