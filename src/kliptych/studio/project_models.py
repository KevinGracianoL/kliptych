"""Validated, immutable human-editable state for a B2 Studio project."""

import uuid
from enum import StrEnum
from typing import ClassVar, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from kliptych.contract.enums import CaptionCapitalization, Layout, WatermarkPosition
from kliptych.contract.schema import Band, LayoutSpec, NormalizedRect

PROJECT_SCHEMA_VERSION = "1"
_SHA256_PATTERN = r"^[0-9a-f]{64}$"
_VERTICAL_LAYOUT = Layout.VERTICAL
_SOURCE_NAME = "source.mp4"


class ProjectStatus(StrEnum):
    """Whether a variant can produce preview or final-render work."""

    ACTIVE = "active"
    REQUIRES_REVIEW = "requires_review"


class VariantOrigin(BaseModel):
    """Immutable B1 evidence binding carried by a human-created variant."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", frozen=True)

    source_candidate_id: str = Field(pattern=_SHA256_PATTERN)
    analysis_sha256: str = Field(pattern=_SHA256_PATTERN)
    transcript_sha256: str = Field(pattern=_SHA256_PATTERN)
    source_sha256: str = Field(pattern=_SHA256_PATTERN)


class VariantRange(BaseModel):
    """A selected time range validated against measured source duration."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", frozen=True)

    start_ms: int = Field(ge=0)
    end_ms: int = Field(gt=0)
    source_duration_ms: int = Field(gt=0)

    @model_validator(mode="after")
    def _fits_source(self) -> Self:
        if self.end_ms <= self.start_ms or self.end_ms > self.source_duration_ms:
            msg = "variant range must fit source_duration_ms"
            raise ValueError(msg)
        return self


class AbsorbNeighbor(StrEnum):
    """The adjacent position selected to cover a deleted interior interval."""

    PREVIOUS = "previous"
    NEXT = "next"


class LayoutPosition(BaseModel):
    """One validated composition interval in a variant timeline."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", frozen=True)

    position_id: uuid.UUID
    start_ms: int = Field(ge=0)
    end_ms: int = Field(gt=0)
    layout_spec: LayoutSpec

    @model_validator(mode="after")
    def _has_positive_duration_and_explicit_bands(self) -> Self:
        if self.end_ms <= self.start_ms:
            msg = "position end_ms must be greater than start_ms"
            raise ValueError(msg)
        if not self.layout_spec.bands:
            msg = "layout positions require explicit bands"
            raise ValueError(msg)
        if any(band.source != _SOURCE_NAME for band in self.layout_spec.bands):
            msg = "B2.1 layout bands must reference source.mp4"
            raise ValueError(msg)
        return self


class VariantWatermark(BaseModel):
    """Local watermark asset binding for one editable variant."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", frozen=True)

    enabled: bool = False
    asset_id: uuid.UUID | None = None
    asset_sha256: str | None = None
    source_name: str | None = None
    position: WatermarkPosition = WatermarkPosition.BOTTOM_RIGHT


class Variant(BaseModel):
    """An editable human variant derived from an immutable source candidate."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", frozen=True)

    variant_id: uuid.UUID
    origin: VariantOrigin
    name: str = Field(min_length=1)
    start_ms: int = Field(ge=0)
    end_ms: int = Field(gt=0)
    status: ProjectStatus = ProjectStatus.ACTIVE
    watermark: VariantWatermark = Field(default_factory=VariantWatermark)
    captions_enabled: bool = False
    caption_highlight_color: str = "&H000000FF"
    caption_capitalization: CaptionCapitalization = CaptionCapitalization.ORIGINAL
    layout_timeline: tuple[LayoutPosition, ...]

    @model_validator(mode="after")
    def _timeline_covers_variant_exactly(self) -> Self:
        if not self.layout_timeline:
            msg = "variant layout_timeline must contain one position"
            raise ValueError(msg)
        if self.end_ms <= self.start_ms:
            msg = "variant end_ms must be greater than start_ms"
            raise ValueError(msg)
        first, *rest = self.layout_timeline
        if len({position.position_id for position in self.layout_timeline}) != len(
            self.layout_timeline
        ):
            msg = "layout_timeline position IDs must be unique"
            raise ValueError(msg)
        if first.start_ms != self.start_ms or self.layout_timeline[-1].end_ms != self.end_ms:
            msg = "layout_timeline must cover the variant range exactly"
            raise ValueError(msg)
        previous = first
        for current in rest:
            if previous.end_ms != current.start_ms:
                msg = "layout_timeline positions must have continuous coverage"
                raise ValueError(msg)
            previous = current
        return self


class ProjectDocument(BaseModel):
    """Versioned editable Studio state bound to B1 evidence digests."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = Field(pattern=r"^1$")
    project_id: str = Field(min_length=1)
    analysis_sha256: str = Field(pattern=_SHA256_PATTERN)
    transcript_sha256: str = Field(pattern=_SHA256_PATTERN)
    source_sha256: str = Field(pattern=_SHA256_PATTERN)
    source_duration_ms: int = Field(gt=0)
    variants: tuple[Variant, ...] = ()

    @model_validator(mode="after")
    def _variants_fit_source_duration(self) -> Self:
        if len({variant.variant_id for variant in self.variants}) != len(self.variants):
            msg = "project variant IDs must be unique"
            raise ValueError(msg)
        if any(
            variant.status is ProjectStatus.ACTIVE and variant.end_ms > self.source_duration_ms
            for variant in self.variants
        ):
            msg = "active variant range exceeds source_duration_ms"
            raise ValueError(msg)
        return self

    def require_review_if_evidence_changed(
        self,
        *,
        analysis_sha256: str,
        transcript_sha256: str,
        source_sha256: str,
    ) -> Self:
        """Return active variants marked for review when any evidence digest changes."""
        if (
            self.analysis_sha256 == analysis_sha256
            and self.transcript_sha256 == transcript_sha256
            and self.source_sha256 == source_sha256
        ):
            return self
        variants = tuple(
            variant.model_copy(update={"status": ProjectStatus.REQUIRES_REVIEW})
            for variant in self.variants
        )
        return self.model_copy(
            update={
                "analysis_sha256": analysis_sha256,
                "transcript_sha256": transcript_sha256,
                "source_sha256": source_sha256,
                "variants": variants,
            }
        )

    def mark_variants_requires_review(self) -> Self:
        """Preserve historical evidence while blocking all variants for review.

        Returns:
            This project with every variant marked requires_review.
        """
        return self.model_copy(
            update={
                "variants": tuple(
                    variant.model_copy(update={"status": ProjectStatus.REQUIRES_REVIEW})
                    for variant in self.variants
                )
            }
        )


def create_initial_variant(
    *,
    name: str,
    selection: VariantRange,
    origin: VariantOrigin,
) -> Variant:
    """Create a full-range Vertical variant with declared source geometry.

    Returns:
        A new active variant with one explicit-band Vertical position.

    """
    spec = LayoutSpec(
        layout=_VERTICAL_LAYOUT,
        bands=(Band(source=_SOURCE_NAME, source_rect=NormalizedRect()),),
    )
    position = LayoutPosition(
        position_id=uuid.uuid4(),
        start_ms=selection.start_ms,
        end_ms=selection.end_ms,
        layout_spec=spec,
    )
    return Variant(
        variant_id=uuid.uuid4(),
        origin=origin,
        name=name,
        start_ms=selection.start_ms,
        end_ms=selection.end_ms,
        layout_timeline=(position,),
    )


def split_position(variant: Variant, *, position_id: uuid.UUID, at_ms: int) -> Variant:
    """Split one timeline position, duplicating its validated geometry.

    Returns:
        A variant with two adjacent positions replacing the selected position.

    Raises:
        ValueError: If the position is missing or the cut lies on its boundary.
    """
    positions = list(variant.layout_timeline)
    index = next(
        (i for i, position in enumerate(positions) if position.position_id == position_id), None
    )
    if index is None:
        msg = "position_id is not part of the variant timeline"
        raise ValueError(msg)
    original = positions[index]
    if at_ms <= original.start_ms or at_ms >= original.end_ms:
        msg = "split must be strictly inside the selected position"
        raise ValueError(msg)
    positions[index : index + 1] = [
        original.model_copy(update={"end_ms": at_ms}),
        LayoutPosition(
            position_id=uuid.uuid4(),
            start_ms=at_ms,
            end_ms=original.end_ms,
            layout_spec=original.layout_spec,
        ),
    ]
    return variant.model_copy(update={"layout_timeline": tuple(positions)})


def duplicate_variant(variant: Variant, *, name: str) -> Variant:
    """Copy a variant's human decisions with fresh variant and position IDs.

    Returns:
        The independent duplicate with the requested name.
    """
    positions = tuple(
        LayoutPosition(
            position_id=uuid.uuid4(),
            start_ms=position.start_ms,
            end_ms=position.end_ms,
            layout_spec=position.layout_spec,
        )
        for position in variant.layout_timeline
    )
    return Variant(
        variant_id=uuid.uuid4(),
        origin=variant.origin,
        name=name,
        start_ms=variant.start_ms,
        end_ms=variant.end_ms,
        status=variant.status,
        watermark=variant.watermark,
        captions_enabled=variant.captions_enabled,
        caption_highlight_color=variant.caption_highlight_color,
        caption_capitalization=variant.caption_capitalization,
        layout_timeline=positions,
    )


def delete_position(
    variant: Variant,
    *,
    position_id: uuid.UUID,
    absorb_neighbor: AbsorbNeighbor | None,
) -> Variant:
    """Delete one position while preserving continuous variant coverage.

    Returns:
        A variant whose adjacent position covers the removed interval.

    Raises:
        ValueError: If the position is absent, is the only one, or an interior
            deletion has no explicit absorbing neighbor.
    """
    positions = list(variant.layout_timeline)
    index = next(
        (i for i, position in enumerate(positions) if position.position_id == position_id), None
    )
    if index is None:
        msg = "position_id is not part of the variant timeline"
        raise ValueError(msg)
    if len(positions) == 1:
        msg = "the only layout position cannot be deleted"
        raise ValueError(msg)
    deleted = positions[index]
    if index == 0:
        positions[1] = positions[1].model_copy(update={"start_ms": deleted.start_ms})
    elif index == len(positions) - 1 or absorb_neighbor is AbsorbNeighbor.PREVIOUS:
        positions[index - 1] = positions[index - 1].model_copy(update={"end_ms": deleted.end_ms})
    elif absorb_neighbor is AbsorbNeighbor.NEXT:
        positions[index + 1] = positions[index + 1].model_copy(
            update={"start_ms": deleted.start_ms}
        )
    else:
        msg = "interior deletion requires absorb_neighbor"
        raise ValueError(msg)
    del positions[index]
    return variant.model_copy(update={"layout_timeline": tuple(positions)})
