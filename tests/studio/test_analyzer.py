"""Behavioral tests for deterministic, render-free Studio analysis."""

from pathlib import Path
from typing import final, override

import pytest

from kliptych.studio.analyzer import AnalysisEngineError, AnalysisRequest, analyze_project
from kliptych.transcribe import Transcriber, Transcript, Word


@final
class _Downloader:
    def __init__(self, content: bytes) -> None:
        self.content: bytes = content
        self.calls: int = 0

    def download_video(self, *, url: str, destination: Path) -> Path:
        assert url == "https://example.com/video"
        self.calls += 1
        _ = destination.write_bytes(self.content)
        return destination


@final
class _Transcriber:
    def __init__(self, transcript: Transcript) -> None:
        self.transcript: Transcript = transcript
        self.calls: int = 0

    def transcribe(self, audio: Path) -> Transcript:
        assert audio.is_file()
        self.calls += 1
        return self.transcript


def _word(start_s: float, end_s: float, text: str, token_id: int) -> Word:
    return Word(start_s=start_s, end_s=end_s, text=text, confidence=0.9, token_id=token_id)


def _transcript() -> Transcript:
    words = (
        _word(0.0, 1.0, "uno", 0),
        _word(1.0, 2.0, "dos", 1),
        _word(2.0, 3.0, "tres", 2),
        _word(3.0, 4.0, "cuatro", 3),
    )
    return Transcript(words=words, language="es", duration_s=4.0, text="uno dos tres cuatro")


def test_analyzer_publishes_deterministic_candidate_and_excerpt(tmp_path: Path) -> None:
    request = AnalysisRequest(
        url="https://example.com/video", project_dir=tmp_path / "project", project_name="My clip"
    )

    result = analyze_project(
        request,
        downloader=_Downloader(b"source"),
        transcriber=_Transcriber(_transcript()),
    )

    assert result.project_name == "My clip"
    assert len(result.candidates) == 1
    candidate = result.candidates[0]
    assert candidate.recommended_start_ms == 0
    assert candidate.recommended_end_ms == 4000
    assert candidate.excerpt == "uno dos tres cuatro"
    assert (request.project_dir / "source.mp4").read_bytes() == b"source"
    assert (request.project_dir / "transcript.json").is_file()
    assert (request.project_dir / "candidates.json").is_file()
    assert (request.project_dir / "analysis.json").is_file()


def test_candidate_identity_is_stable_for_same_source_and_words(tmp_path: Path) -> None:
    first = analyze_project(
        AnalysisRequest(
            url="https://example.com/video",
            project_dir=tmp_path / "first",
            project_name="First",
        ),
        downloader=_Downloader(b"source"),
        transcriber=_Transcriber(_transcript()),
    )
    second = analyze_project(
        AnalysisRequest(
            url="https://example.com/video",
            project_dir=tmp_path / "second",
            project_name="Second",
        ),
        downloader=_Downloader(b"source"),
        transcriber=_Transcriber(_transcript()),
    )

    assert first.candidates[0].candidate_id == second.candidates[0].candidate_id


def test_empty_transcript_fails_and_removes_new_project(tmp_path: Path) -> None:
    empty = Transcript(words=(), language="es", duration_s=0.0, text="[empty]")
    request = AnalysisRequest(
        url="https://example.com/video", project_dir=tmp_path / "project", project_name="My clip"
    )

    with pytest.raises(AnalysisEngineError, match="transcript_empty") as excinfo:
        _ = analyze_project(
            request,
            downloader=_Downloader(b"source"),
            transcriber=_Transcriber(empty),
        )

    assert excinfo.value.code == "transcript_empty"
    assert not request.project_dir.exists()


def test_transcription_failure_does_not_fallback_to_full_video(tmp_path: Path) -> None:
    class _FailingTranscriber(Transcriber):
        @override
        def transcribe(self, audio: Path) -> Transcript:
            assert audio.is_file()
            msg = "ASR failure"
            raise RuntimeError(msg)

    request = AnalysisRequest(
        url="https://example.com/video", project_dir=tmp_path / "project", project_name="My clip"
    )

    with pytest.raises(AnalysisEngineError, match="transcription_failed") as excinfo:
        _ = analyze_project(
            request,
            downloader=_Downloader(b"source"),
            transcriber=_FailingTranscriber(),
        )

    assert excinfo.value.code == "transcription_failed"
    assert not request.project_dir.exists()


def test_reanalysis_reuses_compatible_transcript_and_candidates(tmp_path: Path) -> None:
    request = AnalysisRequest(
        url="https://example.com/video", project_dir=tmp_path / "project", project_name="My clip"
    )
    first = analyze_project(
        request,
        downloader=_Downloader(b"source"),
        transcriber=_Transcriber(_transcript()),
    )
    downloader = _Downloader(b"source")
    transcriber = _Transcriber(_transcript())

    second = analyze_project(request, downloader=downloader, transcriber=transcriber)

    assert downloader.calls == 1
    assert transcriber.calls == 0
    assert second.project_id == first.project_id
    assert second.candidates == first.candidates


def test_reanalysis_retranscribes_when_source_hash_changes(tmp_path: Path) -> None:
    request = AnalysisRequest(
        url="https://example.com/video", project_dir=tmp_path / "project", project_name="My clip"
    )
    _ = analyze_project(
        request,
        downloader=_Downloader(b"first source"),
        transcriber=_Transcriber(_transcript()),
    )
    transcriber = _Transcriber(_transcript())

    _ = analyze_project(
        request,
        downloader=_Downloader(b"changed source"),
        transcriber=transcriber,
    )

    assert transcriber.calls == 1
