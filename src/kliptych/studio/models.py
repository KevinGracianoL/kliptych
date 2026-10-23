"""Versioned documents shared by Track B1 storage and later Studio clients."""

from typing import ClassVar, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

ANALYSIS_SCHEMA_VERSION = "1"


class AnalysisSource(BaseModel):
    """Source URL and content identity of an analysis."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", frozen=True)

    url: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class AnalysisProvenance(BaseModel):
    """Versions used to decide cache compatibility."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", frozen=True)

    transcription_version: str = Field(min_length=1)
    segmenter_version: str = Field(min_length=1)


class AnalysisCandidate(BaseModel):
    """Immutable local-segmenter recommendation shown by Studio."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", frozen=True)

    candidate_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    recommended_start_ms: int = Field(ge=0)
    recommended_end_ms: int = Field(gt=0)
    score: float = Field(ge=0.0, le=1.0)
    excerpt: str = Field(min_length=1)

    @model_validator(mode="after")
    def _has_ordered_interval(self) -> Self:
        if self.recommended_end_ms <= self.recommended_start_ms:
            msg = "recommended_end_ms must be greater than recommended_start_ms"
            raise ValueError(msg)
        return self


class AnalysisDocument(BaseModel):
    """Canonical completed B1 analysis result."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = Field(min_length=1)
    project_id: str = Field(min_length=1)
    project_name: str = Field(min_length=1)
    source: AnalysisSource
    provenance: AnalysisProvenance
    candidates: tuple[AnalysisCandidate, ...]


class CandidateCache(BaseModel):
    """Versioned private cache emitted by the local speech-rate segmenter."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = Field(min_length=1)
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    segmenter_version: str = Field(min_length=1)
    candidates: tuple[AnalysisCandidate, ...]
