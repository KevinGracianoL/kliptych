"""Fix 4: CUDA en Windows, fallo ruidoso y caché del modelo."""

import logging
import sys
import types
from pathlib import Path
from typing import TYPE_CHECKING, cast

import pytest

if TYPE_CHECKING:
    from collections.abc import Callable

from kliptych import transcribe as transcribe_module
from kliptych.transcribe import FasterWhisperTranscriber, TranscriptionError
from tests.test_transcribe import FakeEngine, FakeWord


def _audio(tmp_path: Path) -> Path:
    path = tmp_path / "audio.wav"
    _ = path.write_bytes(b"RIFF")
    return path


def _private(name: str) -> object:
    return cast("object", getattr(transcribe_module, name))


def _clear_cache() -> None:
    cache_obj = _private("_MODEL_CACHE")
    if isinstance(cache_obj, dict):
        cache_obj.clear()


def _install_factory(monkeypatch: pytest.MonkeyPatch, factory: object) -> None:
    _clear_cache()
    module = types.SimpleNamespace(WhisperModel=factory)
    monkeypatch.setitem(sys.modules, "faster_whisper", module)


def _make_word(start: float, end: float, text: str) -> FakeWord:
    return FakeWord(start, end, text, 0.9)


def _fake_engine(*, with_words: bool = True) -> object:
    if with_words:
        return FakeEngine(words=[_make_word(0.0, 0.5, " hola")])
    return FakeEngine(words=[])


def _register_dlls() -> None:
    func = cast("Callable[[], None]", _private("_register_windows_cuda_dlls"))
    func()


def test_model_cache_reuses_instance(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, object]] = []

    def factory(
        model_size_or_path: str, *, device: str, compute_type: str, num_workers: int
    ) -> object:
        calls.append(
            {
                "model_size_or_path": model_size_or_path,
                "device": device,
                "compute_type": compute_type,
                "num_workers": num_workers,
            }
        )
        return _fake_engine()

    _install_factory(monkeypatch, factory)
    transcriber = FasterWhisperTranscriber()
    _ = transcriber.transcribe(_audio(tmp_path))
    _ = transcriber.transcribe(_audio(tmp_path))
    assert len(calls) == 1
    _clear_cache()


def test_cuda_requested_failure_is_loud(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def failing_factory(
        model_size_or_path: str, *, device: str, compute_type: str, num_workers: int
    ) -> object:
        _ = (model_size_or_path, compute_type, num_workers)
        if device == "cuda":
            msg = "cuBLAS missing"
            raise RuntimeError(msg)
        return _fake_engine()

    _install_factory(monkeypatch, failing_factory)
    transcriber = FasterWhisperTranscriber(device="cuda")
    with pytest.raises(TranscriptionError, match="CUDA solicitado"):
        _ = transcriber.transcribe(_audio(tmp_path))
    _clear_cache()


def test_auto_fallback_warns_and_uses_cpu(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    seen: list[str] = []

    def flaky_factory(
        model_size_or_path: str, *, device: str, compute_type: str, num_workers: int
    ) -> object:
        _ = (model_size_or_path, compute_type, num_workers)
        seen.append(device)
        if device in {"auto", "cuda"}:
            msg = "CUDA init failed"
            raise RuntimeError(msg)
        return _fake_engine()

    _install_factory(monkeypatch, flaky_factory)
    transcriber = FasterWhisperTranscriber(device="auto")
    with caplog.at_level(logging.WARNING, logger="kliptych.transcribe"):
        transcript = transcriber.transcribe(_audio(tmp_path))
    assert transcript.text == "hola"
    assert "cpu" in seen
    assert any(
        "CUDA" in record.message or "cuda" in record.message.lower() for record in caplog.records
    )
    _clear_cache()


def test_windows_cuda_dll_registration(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fake_venv = tmp_path / "venv"
    cublas = fake_venv / "Lib" / "site-packages" / "nvidia" / "cublas" / "bin"
    cudnn = fake_venv / "Lib" / "site-packages" / "nvidia" / "cudnn" / "bin"
    _ = cublas.mkdir(parents=True)
    _ = cudnn.mkdir(parents=True)
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(sys, "prefix", str(fake_venv))
    added: list[str] = []

    def _record(directory: str) -> object:
        added.append(directory)
        return object()

    monkeypatch.setattr("os.add_dll_directory", _record, raising=False)
    _register_dlls()
    assert str(cublas) in added or str(cudnn) in added
    _clear_cache()
