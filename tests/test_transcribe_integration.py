"""Integración de transcripción con el modelo real de faster-whisper.

El módulo se salta entero si faster-whisper no está instalado. La señal de
entrada es voz sintética generada localmente (SAPI en Windows): no se descarga
audio de la red. El modelo ``small`` se descarga/cachea la primera vez que se
ejecuta, fuera de CI, donde la extra ``transcription`` no se instala.

La inferencia se pide explícitamente en ``cpu``: ``auto`` elige CUDA y falla si
faltan las DLLs de runtime (``cublas64_12.dll``), ausentes en esta máquina. La
medición de VRAM en GPU requiere esas librerías y se documenta aparte.
"""

import importlib.util
import os
import subprocess
import sys
from pathlib import Path
from typing import cast

import pytest

from kliptych.transcribe import FasterWhisperTranscriber, Transcript

_HAS_FASTER_WHISPER = importlib.util.find_spec("faster_whisper") is not None

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not _HAS_FASTER_WHISPER,
        reason="faster-whisper no instalado",
    ),
]

_MODEL_ATTR = "_model"

_SPEECH = "the quick brown fox jumps over the lazy dog"
_TTS_TIMEOUT_S = 120

_TTS_SCRIPT = (
    "Add-Type -AssemblyName System.Speech; "
    "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
    "$s.SetOutputToWaveFile($env:KLIPTYCH_TTS_OUT); "
    "$s.Speak($env:KLIPTYCH_TTS_TEXT); "
    "$s.Dispose()"
)


def _synthesize_speech(path: Path) -> Path:
    if sys.platform != "win32":
        pytest.skip("síntesis de voz SAPI solo disponible en Windows")
    env = {
        **os.environ,
        "KLIPTYCH_TTS_OUT": str(path),
        "KLIPTYCH_TTS_TEXT": _SPEECH,
    }
    completed = subprocess.run(
        ["powershell", "-NoProfile", "-NonInteractive", "-Command", _TTS_SCRIPT],
        capture_output=True,
        text=True,
        timeout=_TTS_TIMEOUT_S,
        check=False,
        env=env,
    )
    if completed.returncode != 0 or not path.is_file():
        pytest.skip(f"síntesis de voz no disponible: {completed.stderr.strip()}")
    return path


def test_real_transcription_produces_words(tmp_path: Path) -> None:
    audio = _synthesize_speech(tmp_path / "speech.wav")
    transcript = FasterWhisperTranscriber(device="cpu").transcribe(audio)
    assert isinstance(transcript, Transcript)
    assert transcript.words
    assert transcript.language
    assert transcript.duration_s > 0
    for word in transcript.words:
        assert word.start_s >= 0.0
        assert word.end_s >= word.start_s
        assert word.text
        assert 0.0 <= word.confidence <= 1.0
    assert [word.token_id for word in transcript.words] == list(range(len(transcript.words)))
    assert transcript.text == " ".join(word.text for word in transcript.words)


def test_memory_released_after_transcription(tmp_path: Path) -> None:
    audio = _synthesize_speech(tmp_path / "speech.wav")
    transcriber = FasterWhisperTranscriber(device="cpu")
    _ = transcriber.transcribe(audio)
    assert cast("object", getattr(transcriber, _MODEL_ATTR)) is None
