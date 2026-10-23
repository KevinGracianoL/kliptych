"""Render-free Track B1 analysis using download, ASR, and local segmentation."""

import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from kliptych.hashing import sha256_canonical_json, sha256_file
from kliptych.moments import Moment
from kliptych.segmenter import segment_by_speech_rate
from kliptych.studio.models import (
    ANALYSIS_SCHEMA_VERSION,
    AnalysisCandidate,
    AnalysisDocument,
    AnalysisProvenance,
    AnalysisSource,
    CandidateCache,
)
from kliptych.studio.project_store import open_project_store
from kliptych.transcribe import Transcriber, Transcript

TRANSCRIPTION_VERSION = "1"
SEGMENTER_VERSION = "1"
_SOURCE_NAME = "source.mp4"
_TRANSCRIPT_NAME = "transcript.json"
_CANDIDATES_NAME = "candidates.json"


class AnalysisEngineError(Exception):
    """The deterministic analysis engine could not produce an analysis."""

    def __init__(self, code: str, message: str) -> None:
        """Create an error suitable for later translation into an NDJSON event."""
        super().__init__(f"{code}: {message}")
        self.code: str = code
        self.message: str = message


class AnalysisCancelled(KeyboardInterrupt):
    """Cooperative cancellation propagated without translating it to a failure."""


class _Downloader(Protocol):
    def download_video(self, *, url: str, destination: Path) -> Path:
        """Download a source VOD to the requested stage location."""
        ...


class AnalysisProgress(Protocol):
    """Receives one known analysis stage after that stage becomes active."""

    def __call__(self, stage: str, /) -> None:
        """Receive a stage identifier."""
        ...


@dataclass(frozen=True, slots=True)
class AnalysisRequest:
    """Inputs to a single deterministic local Studio analysis."""

    url: str
    project_dir: Path
    project_name: str


def analyze_project(
    request: AnalysisRequest,
    *,
    downloader: _Downloader,
    transcriber: Transcriber,
    progress: AnalysisProgress | None = None,
) -> AnalysisDocument:
    """Create one complete Studio analysis without rendering media.

    Args:
        request: URL and project identity selected by Studio.
        downloader: Bounded source downloader.
        transcriber: Word-level ASR boundary.
        progress: Optional callback invoked when each analysis stage begins.

    Returns:
        The committed analysis document.

    """
    store = open_project_store(request.project_dir)
    project_id = (
        store.existing_analysis.project_id if store.existing_analysis else str(uuid.uuid4())
    )
    with store.stage() as stage:
        _progress(progress, "downloading")
        source = _download_source(
            request.url,
            destination=stage.directory / _SOURCE_NAME,
            downloader=downloader,
        )
        source_hash = sha256_file(source)
        _progress(progress, "transcribing")
        transcript = _resolve_transcript(
            source=source,
            source_hash=source_hash,
            existing=store.existing_analysis,
            cached=request.project_dir / _TRANSCRIPT_NAME,
            transcriber=transcriber,
        )
        _progress(progress, "finding_candidates")
        candidates = _resolve_candidates(
            transcript=transcript,
            source_hash=source_hash,
            existing=store.existing_analysis,
            cached=request.project_dir / _CANDIDATES_NAME,
        )
        document = AnalysisDocument(
            schema_version=ANALYSIS_SCHEMA_VERSION,
            project_id=project_id,
            project_name=request.project_name,
            source=AnalysisSource(url=request.url, sha256=source_hash),
            provenance=AnalysisProvenance(
                transcription_version=TRANSCRIPTION_VERSION,
                segmenter_version=SEGMENTER_VERSION,
            ),
            candidates=candidates,
        )
        stage.write_text(_TRANSCRIPT_NAME, transcript.model_dump_json(indent=2))
        stage.write_text(_CANDIDATES_NAME, _candidates_json(candidates, source_hash=source_hash))
        stage.write_analysis(document)
        return document


def _progress(callback: AnalysisProgress | None, stage: str) -> None:
    if callback is not None:
        callback(stage)


def _download_source(url: str, *, destination: Path, downloader: _Downloader) -> Path:
    try:
        downloaded = downloader.download_video(url=url, destination=destination)
    except AnalysisCancelled:
        raise
    except Exception as error:
        code = "source_download_failed"
        message = "The source video could not be downloaded."
        raise AnalysisEngineError(code, message) from error
    if downloaded != destination or not destination.is_file() or destination.stat().st_size == 0:
        code = "source_download_failed"
        message = "The source video was not written safely."
        raise AnalysisEngineError(code, message)
    return destination


def _transcribe(source: Path, *, transcriber: Transcriber) -> Transcript:
    try:
        transcript = transcriber.transcribe(source)
    except AnalysisCancelled:
        raise
    except Exception as error:
        code = "transcription_failed"
        message = "The source video could not be transcribed."
        raise AnalysisEngineError(code, message) from error
    if not transcript.words:
        code = "transcript_empty"
        message = "Transcription produced no timestamped words."
        raise AnalysisEngineError(code, message)
    return transcript


def _resolve_transcript(
    *,
    source: Path,
    source_hash: str,
    existing: AnalysisDocument | None,
    cached: Path,
    transcriber: Transcriber,
) -> Transcript:
    if (
        existing is not None
        and cached.is_file()
        and existing.source.sha256 == source_hash
        and existing.provenance.transcription_version == TRANSCRIPTION_VERSION
    ):
        try:
            return Transcript.model_validate_json(cached.read_text(encoding="utf-8"))
        except ValueError:
            pass
    return _transcribe(source, transcriber=transcriber)


def _candidates(transcript: Transcript, *, source_hash: str) -> tuple[AnalysisCandidate, ...]:
    moments = segment_by_speech_rate(transcript.words)
    candidates = tuple(
        _candidate(moment, transcript=transcript, source_hash=source_hash) for moment in moments
    )
    return tuple(
        sorted(candidates, key=lambda candidate: (-candidate.score, candidate.candidate_id))
    )


def _resolve_candidates(
    *,
    transcript: Transcript,
    source_hash: str,
    existing: AnalysisDocument | None,
    cached: Path,
) -> tuple[AnalysisCandidate, ...]:
    if (
        existing is not None
        and cached.is_file()
        and existing.source.sha256 == source_hash
        and existing.provenance.segmenter_version == SEGMENTER_VERSION
    ):
        try:
            cache = CandidateCache.model_validate_json(cached.read_text(encoding="utf-8"))
        except ValueError:
            pass
        else:
            if cache.source_sha256 == source_hash and cache.segmenter_version == SEGMENTER_VERSION:
                return cache.candidates
    return _candidates(transcript, source_hash=source_hash)


def _candidate(moment: Moment, *, transcript: Transcript, source_hash: str) -> AnalysisCandidate:
    start_ms = round(moment.start_s * 1000)
    end_ms = round(moment.end_s * 1000)
    excerpt = " ".join(
        word.text.strip()
        for word in transcript.words
        if word.end_s > moment.start_s and word.start_s < moment.end_s and word.text.strip()
    )
    candidate_id = sha256_canonical_json(
        {
            "schema_version": ANALYSIS_SCHEMA_VERSION,
            "source_sha256": source_hash,
            "start_ms": start_ms,
            "end_ms": end_ms,
        }
    )
    return AnalysisCandidate(
        candidate_id=candidate_id,
        recommended_start_ms=start_ms,
        recommended_end_ms=end_ms,
        score=moment.score,
        excerpt=excerpt,
    )


def _candidates_json(candidates: Sequence[AnalysisCandidate], *, source_hash: str) -> str:
    cache = CandidateCache(
        schema_version=ANALYSIS_SCHEMA_VERSION,
        source_sha256=source_hash,
        segmenter_version=SEGMENTER_VERSION,
        candidates=tuple(candidates),
    )
    return cache.model_dump_json(indent=2)
