"""Behavioral tests for the editable B2 Studio project model."""

from uuid import UUID

import pytest

from kliptych.contract.enums import Layout
from kliptych.contract.schema import Band, LayoutSpec, NormalizedRect
from kliptych.studio.project_models import (
    AbsorbNeighbor,
    LayoutPosition,
    ProjectDocument,
    ProjectStatus,
    Variant,
    VariantOrigin,
    VariantRange,
    create_initial_variant,
    delete_position,
    duplicate_variant,
    split_position,
)

_PROJECT_ID = "11111111-1111-1111-1111-111111111111"
_CANDIDATE_ID = "a" * 64
_HASH = "b" * 64


def _project(*, variants: tuple[Variant, ...] = ()) -> ProjectDocument:
    return ProjectDocument(
        schema_version="1",
        project_id=_PROJECT_ID,
        analysis_sha256=_HASH,
        transcript_sha256=_HASH,
        source_sha256=_HASH,
        source_duration_ms=60_000,
        variants=variants,
    )


def _initial_variant(*, start_ms: int, end_ms: int) -> Variant:
    return create_initial_variant(
        name="Variant 1",
        selection=VariantRange(
            start_ms=start_ms,
            end_ms=end_ms,
            source_duration_ms=60_000,
        ),
        origin=VariantOrigin(
            source_candidate_id=_CANDIDATE_ID,
            analysis_sha256=_HASH,
            transcript_sha256=_HASH,
            source_sha256=_HASH,
        ),
    )


def test_initial_variant_covers_its_full_range_with_explicit_vertical_band() -> None:
    variant = _initial_variant(start_ms=10_000, end_ms=30_000)

    assert isinstance(variant.variant_id, UUID)
    assert variant.status is ProjectStatus.ACTIVE
    assert len(variant.layout_timeline) == 1
    position = variant.layout_timeline[0]
    assert position.start_ms == 10_000
    assert position.end_ms == 30_000
    assert position.layout_spec.layout.value == "vertical"
    assert len(position.layout_spec.bands) == 1
    assert position.layout_spec.bands[0].source == "source.mp4"


def test_project_rejects_variant_outside_measured_source_duration() -> None:
    variant = create_initial_variant(
        name="Variant 1",
        selection=VariantRange(
            start_ms=10_000,
            end_ms=70_000,
            source_duration_ms=70_000,
        ),
        origin=VariantOrigin(
            source_candidate_id=_CANDIDATE_ID,
            analysis_sha256=_HASH,
            transcript_sha256=_HASH,
            source_sha256=_HASH,
        ),
    )

    with pytest.raises(ValueError, match="source_duration_ms"):
        _ = _project(variants=(variant,))


def test_split_duplicates_layout_and_preserves_exact_coverage() -> None:
    original = _initial_variant(start_ms=10_000, end_ms=30_000)

    split = split_position(
        original, position_id=original.layout_timeline[0].position_id, at_ms=20_000
    )

    assert len(split.layout_timeline) == 2
    left, right = split.layout_timeline
    assert left.start_ms == 10_000
    assert left.end_ms == 20_000
    assert right.start_ms == 20_000
    assert right.end_ms == 30_000
    assert left.position_id != right.position_id
    assert left.layout_spec == right.layout_spec


def test_split_rejects_position_boundary() -> None:
    variant = _initial_variant(start_ms=10_000, end_ms=30_000)

    with pytest.raises(ValueError, match="strictly inside"):
        _ = split_position(
            variant, position_id=variant.layout_timeline[0].position_id, at_ms=10_000
        )


def test_project_requires_review_when_bound_analysis_evidence_changes() -> None:
    variant = _initial_variant(start_ms=10_000, end_ms=30_000)
    project = _project(variants=(variant,))

    reviewed = project.require_review_if_evidence_changed(
        analysis_sha256="c" * 64,
        transcript_sha256=_HASH,
        source_sha256=_HASH,
    )

    assert reviewed.variants[0].status is ProjectStatus.REQUIRES_REVIEW
    assert reviewed.variants[0].origin.analysis_sha256 == _HASH
    assert reviewed.variants[0].origin.transcript_sha256 == _HASH
    assert reviewed.variants[0].origin.source_sha256 == _HASH


def test_delete_interior_position_requires_explicit_absorbing_neighbor() -> None:
    original = _initial_variant(start_ms=10_000, end_ms=40_000)
    first_split = split_position(
        original,
        position_id=original.layout_timeline[0].position_id,
        at_ms=20_000,
    )
    timeline = split_position(
        first_split,
        position_id=first_split.layout_timeline[1].position_id,
        at_ms=30_000,
    )
    middle = timeline.layout_timeline[1]

    with pytest.raises(ValueError, match="absorb_neighbor"):
        _ = delete_position(timeline, position_id=middle.position_id, absorb_neighbor=None)

    deleted = delete_position(
        timeline,
        position_id=middle.position_id,
        absorb_neighbor=AbsorbNeighbor.PREVIOUS,
    )

    assert [(position.start_ms, position.end_ms) for position in deleted.layout_timeline] == [
        (10_000, 30_000),
        (30_000, 40_000),
    ]


def test_duplicate_variant_gives_variant_and_positions_fresh_ids() -> None:
    variant = _initial_variant(start_ms=10_000, end_ms=30_000)

    duplicate = duplicate_variant(variant, name="Variant 2")

    assert duplicate.variant_id != variant.variant_id
    assert duplicate.name == "Variant 2"
    assert duplicate.origin == variant.origin
    assert duplicate.layout_timeline[0].position_id != variant.layout_timeline[0].position_id
    assert duplicate.layout_timeline[0].layout_spec == variant.layout_timeline[0].layout_spec


def test_duplicate_variant_rejects_invalid_name() -> None:
    variant = _initial_variant(start_ms=10_000, end_ms=30_000)

    with pytest.raises(ValueError, match="at least 1 character"):
        _ = duplicate_variant(variant, name="")


@pytest.mark.parametrize("source", [None, "other.mp4"])
def test_position_rejects_band_that_is_not_the_project_source(source: str | None) -> None:
    with pytest.raises(ValueError, match=r"source\.mp4"):
        _ = LayoutPosition(
            position_id=UUID("11111111-1111-1111-1111-111111111111"),
            start_ms=10_000,
            end_ms=20_000,
            layout_spec=LayoutSpec(
                layout=Layout.VERTICAL,
                bands=(Band(source=source, source_rect=NormalizedRect()),),
            ),
        )
