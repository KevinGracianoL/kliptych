"""Fix 4: CUDA en Windows, fallo ruidoso y caché del modelo."""

import logging
import os
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


class _CodedOSError(OSError):
    winerror: int

    def __init__(self, message: str, code: int) -> None:
        super().__init__(message)
        self.winerror = code


def _audio(tmp_path: Path) -> Path:
    path = tmp_path / "audio.wav"
    _ = path.write_bytes(b"RIFF")
    return path


def _private(name: str) -> object:
    return cast("object", getattr(transcribe_module, name))


def _clear_cache() -> None:
    transcribe_module.clear_model_cache()


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


def _patch_symlinks() -> None:
    func = cast("Callable[[], None]", _private("_patch_windows_symlinks"))
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


def test_model_cache_separates_num_workers(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
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
    _ = FasterWhisperTranscriber(num_workers=1).transcribe(_audio(tmp_path))
    _ = FasterWhisperTranscriber(num_workers=2).transcribe(_audio(tmp_path))
    assert len(calls) == 2
    _clear_cache()


def test_clear_model_cache_empties_entries(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def factory(
        model_size_or_path: str, *, device: str, compute_type: str, num_workers: int
    ) -> object:
        _ = (model_size_or_path, device, compute_type, num_workers)
        return _fake_engine()

    _install_factory(monkeypatch, factory)
    _ = FasterWhisperTranscriber().transcribe(_audio(tmp_path))
    cache = cast("dict[tuple[str, str, str, int], object]", _private("_MODEL_CACHE"))
    assert len(cache) == 1
    transcribe_module.clear_model_cache()
    assert len(cache) == 0


def test_windows_symlink_fallback_copies_file_on_winerror_1314(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(sys, "platform", "win32")
    src = tmp_path / "src.txt"
    dst = tmp_path / "dst.txt"
    _ = src.write_bytes(b"model-bytes")

    def _raising_symlink(src_arg: object, dst_arg: object, *args: object, **kwargs: object) -> None:
        _ = (src_arg, dst_arg, args, kwargs)
        msg = "symlink privilege not held"
        raise _CodedOSError(msg, 1314)

    monkeypatch.setattr(os, "symlink", _raising_symlink)
    _patch_symlinks()
    dst.symlink_to(src)
    assert dst.is_file()
    assert dst.read_bytes() == b"model-bytes"


def test_windows_symlink_fallback_copies_dir_on_winerror_1314(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(sys, "platform", "win32")
    src_dir = tmp_path / "src_dir"
    dst_dir = tmp_path / "dst_dir"
    _ = (src_dir / "nested").mkdir(parents=True)
    _ = (src_dir / "nested" / "blob.bin").write_bytes(b"blob")

    def _raising_symlink(src_arg: object, dst_arg: object, *args: object, **kwargs: object) -> None:
        _ = (src_arg, dst_arg, args, kwargs)
        msg = "symlink privilege not held"
        raise _CodedOSError(msg, 1314)

    monkeypatch.setattr(os, "symlink", _raising_symlink)
    _patch_symlinks()
    dst_dir.symlink_to(src_dir, target_is_directory=True)
    assert (dst_dir / "nested" / "blob.bin").is_file()


def test_windows_symlink_reraises_non_1314(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "platform", "win32")

    def _raising_other(src_arg: object, dst_arg: object, *args: object, **kwargs: object) -> None:
        _ = (src_arg, dst_arg, args, kwargs)
        msg = "other failure"
        raise _CodedOSError(msg, 1)

    monkeypatch.setattr(os, "symlink", _raising_other)
    _patch_symlinks()
    with pytest.raises(OSError, match="other failure"):
        (tmp_path / "b").symlink_to(tmp_path / "a")
