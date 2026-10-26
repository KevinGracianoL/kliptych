# ADR 0002: Track B2 Project and Layout Timeline Contract

- **Status:** Accepted
- **Date:** 2026-10-23
- **Decision owners:** Kliptych product and engineering

## Context

Track B1 produces immutable `analysis.json` evidence: source provenance and
automatic candidate clips. Track B2 introduces creative edits without rewriting
that evidence: user variants, editable ranges, layouts over time, preview state,
basic caption controls, and optional local watermark assets.

The intended editor can apply different compositions during a single clip.
Therefore a candidate does not own one static layout. A user-created variant
owns a continuous timeline of validated layout positions.

This ADR defines the persistent boundary before implementing `project.json`, the
preview command, or the Tauri UI.

## Decision

### Immutable analysis and editable project state

`analysis.json` remains immutable automatic evidence. `project.json` is a
separate, versioned, atomically-written document containing human decisions. It
never rewrites a candidate's recommended range, score, excerpt, or ASR
transcript.

`project.json` records the SHA-256 of the canonical `analysis.json`, the
canonical `transcript.json`, and the source content that it was created against:

```json
{
  "schema_version": "1",
  "project_id": "<opaque stable identifier from analysis.json>",
  "analysis_sha256": "<canonical analysis.json hash>",
  "transcript_sha256": "<canonical transcript.json hash>",
  "source_sha256": "<source.mp4 hash>",
  "variants": []
}
```

The analysis digest detects candidate and segmenter-result changes. The
transcript digest detects ASR evidence changes even when the source, analysis
provenance, and candidate list are otherwise identical. The source digest
detects a changed VOD even when its URL is the same.

`project_id` remains the opaque stable identifier emitted by B1. It is not
required to be a UUID. User-created `variant_id` and `position_id` are UUIDs.

### Variants

A variant is an editable, named selection derived from one automatic candidate.
It has a random UUID because it is mutable human state, unlike a deterministic
candidate identity.

```json
{
  "variant_id": "<uuid>",
  "source_candidate_id": "<deterministic candidate id>",
  "name": "Variant 1",
  "start_ms": 30040,
  "end_ms": 51780,
  "status": "active",
  "layout_timeline": []
}
```

A variant may select any valid range inside the measured duration of
`source.mp4`; it does not alter the source candidate. The final render remains
responsible for applying the relevant duration policy.

The variant count is bounded by a validated implementation policy. This ADR does
not freeze a numeric limit: any permanent value requires a reproducible storage,
preview, and usability measurement on target hardware before it becomes product
policy. The UI creates `Variant 1`, `Variant 2`, and so on; names are editable.
Creating a variant duplicates the active variant's range, timeline, caption
settings, and watermark settings with fresh IDs. Deleting a variant requires
confirmation and deletes only that variant's derived previews and asset
references, never the candidate or another variant.

### Layout timeline

Each variant owns a bounded ordered `layout_timeline`. A position has a random
UUID, an interval, and a validated `LayoutSpec`. The implementation policy for
the maximum number of positions requires a reproducible measurement before it
becomes a permanent numeric limit:

```json
{
  "position_id": "<uuid>",
  "start_ms": 30040,
  "end_ms": 38000,
  "layout_spec": {
    "layout": "vertical",
    "bands": [
      {
        "source": "source.mp4",
        "source_rect": {
          "x": 0.0,
          "y": 0.0,
          "width": 1.0,
          "height": 1.0
        }
      }
    ]
  }
}
```

The timeline has exact continuous coverage:

```text
first.start_ms = variant.start_ms
last.end_ms = variant.end_ms
position[i].end_ms = position[i + 1].start_ms
each start_ms < end_ms
no gaps and no overlaps
```

Opening a candidate creates one initial Vertical position covering the complete
variant range. A user may split the active position at the playhead only when
the cut is strictly inside the active position. Splitting copies the existing
`LayoutSpec` and crops to both positions and assigns a fresh `position_id` to
the latter position. Invalid boundary cuts are rejected without changing the
timeline. B2.1 intentionally does not impose a numeric minimum position
duration: a permanent threshold requires reproducible preview/render measurement
on target hardware before it becomes policy.

The only position cannot be deleted. Deleting the first position extends the
next backward; deleting the last extends the previous forward. Deleting an
interior position requires the user to choose whether the preceding or following
neighbor absorbs its interval. These rules preserve coverage without silently
guessing an artistic decision.

### Layout presets and source reuse

All composition presets are available: Vertical, Dividido, Trío, Spotlight,
Centrado, and Horizontal. Their backend layout values remain the existing
`Layout` enum values. B2 persists an explicit `Band` for every visible band,
including one-band presets; it does not rely on the assembler's empty-band
defaults because the canvas must display declared geometry.

`Centrado` is a UI preset, not a new backend `Layout`: it creates
`Layout.VERTICAL` with a centered initial crop. This avoids a UI-only layout
value that the assembler cannot apply.

For a B1 one-source project, every band references the same `source.mp4`, but
each `source_rect` is independent. The UI labels this as multiple crops of the
same VOD, not as multiple source files. The existing assembler requires one
input clip per band, so preview/final-render integration must explicitly repeat
the same validated source input for each band. A `Band.source` declaration alone
does not currently create duplicate assembler inputs.

`source_rect` values are normalized and must remain inside the unit rectangle:

```text
0 <= x, y
x + width <= 1
y + height <= 1
width and height satisfy a validated minimum-size policy
```

During drag/resize, the canvas updates a local overlay only. On pointer release,
the UI sends the complete layout position for backend validation, persists the
validated result, and requests an exact preview. Changing presets requires
confirmation because it replaces the position's current crops with valid preset
defaults.

### Analysis change review

Variants retain their work when B1 reanalysis changes source, analysis, or
transcript evidence, but enter `requires_review` and cannot generate a preview
or final render.

This retention is a **B2.1 implementation prerequisite**, not behavior supplied
by the current B1 store. B1 currently replaces its project directory as one
analysis transaction. Before `project.json`, assets, or previews are persisted,
the project store must be extended and tested to preserve editable state across
an analysis replacement, or to atomically migrate that state to the replacement
directory. B2 must not write `project.json` until that invariant is implemented.

- If `source_sha256` changes, Studio never migrates automatically. The user must
  consciously create or migrate work over the changed source. If editable state
  exists and the active source is changed, missing, or link-like, B1 reanalysis
  aborts before migrating `project.json`, assets, or previews.
- If `transcript_sha256` changes, Studio marks variants `requires_review` even
  when the candidate list did not change, because captions and their previews
  derive from ASR evidence.
- If the source SHA-256 is unchanged but `analysis_sha256` changes and the
  candidate no longer exists, Studio offers candidates whose ranges overlap the
  variant range. The user explicitly selects a replacement candidate or marks
  the variant as a manual selection over the same source.

No variant is deleted merely because reanalysis changed.

### Preview state

Preview files are derived cache, not source of truth. A preview record includes
its relative path and a canonical fingerprint of every visual input:

```json
{
  "preview_path": "previews/<variant-id>/<position-id>.jpg",
  "preview_fingerprint": "<sha256 of canonical visual inputs>",
  "frame_time_ms": 34000,
  "width": 720,
  "height": 1280
}
```

The preview is valid only when its fingerprint matches the current variant,
position, watermark, caption style, source digest, transcript digest, and
preview implementation version. Changing a layout preset, crop, caption
setting, watermark setting, source, or transcript invalidates affected previews.
The active preview is generated at the active timeline/playhead timestamp after
the user releases a crop gesture.
Changing a preset removes only the derived preview records/files for the changed
position; it never deletes previews belonging to another position or variant.

### Basic caption settings

B2 supports only the existing `CaptionStyleConfig` controls:

```text
enabled
highlight color
capitalization: Aa / AA / aa
```

`captions_enabled` is a Studio variant field; it is not a field of
`CaptionStyleConfig`. When enabled, the variant contains a validated override
limited to `highlight_color` and `capitalization`; every other caption-style
field retains its inherited contract/default value. The B2 preview may show a
clearly labelled representative sample; it does not claim to be karaoke timing.
The original ASR transcript remains immutable. Translation is out of scope for
Kliptych Studio. Future caption correction must be attributed human override
data, not a rewrite of ASR evidence.

### B2.5 caption style presets and bounded overrides

B2.5 replaces the B2 restriction above for variants that explicitly opt into a
caption style. It does not turn Studio into a free-form typography editor. A
variant persists one optional `caption_style` object:

```json
{
  "preset": "clean",
  "size": "medium",
  "vertical_position": "bottom",
  "horizontal_alignment": "center",
  "outline": "thin",
  "shadow": "none"
}
```

`caption_style` is absent for a B2.4 project or variant. Absence preserves the
existing inherited B2.4 rendering exactly; migration never silently changes a
human's existing captions. Selecting a B2.5 preset writes the complete,
validated object atomically to `project.json` and invalidates only derived
previews of that variant.

The permitted enum values are deliberately closed:

```text
preset:               clean | bold | boxed
size:                 small | medium | large
vertical_position:    top | middle | bottom
horizontal_alignment: left | center | right
outline:              none | thin | thick
shadow:               none | soft | strong
```

The existing per-variant `captions_enabled`, `caption_highlight_color`, and
`caption_capitalization` fields remain independent. `highlight_color` and
capitalization therefore keep their B2.3 semantics after a B2.5 style is
selected. No B2.5 control accepts a free font family, raw ASS field, arbitrary
color, free coordinate, margin, box padding, or radius.

The backend resolves the persisted enums into one full immutable ASS style;
the renderer never accepts style fragments supplied by the UI. The mapping is
versioned with the implementation and included in the preview fingerprint.
The presets define the base typography and colors; size, alignment, outline,
and shadow selections replace their corresponding preset values. Vertical and
horizontal positions map to ASS's nine canonical alignment cells, never to
numeric user coordinates. `boxed` uses ASS `BorderStyle=3`, an opaque box
around each dialogue line; it is explicitly not a full-width, independently
sized, or rounded box.

All B2.5 presets use the one packaged family `Noto Sans Regular`. The project
ships the unmodified official `NotoSans-Regular.ttf` under SIL Open Font
License 1.1 together with its `OFL.txt`; the vendored artifact's source URL and
SHA-256 are recorded beside the asset. The renderer passes the controlled font
directory through the FFmpeg/libass `subtitles` filter's `fontsdir` option.
This avoids host-font dependence on Linux, Windows, and macOS. A missing,
modified, symlinked, or non-regular packaged font is a render error, never a
fallback to a host font.

The B2.5 implementation must validate all enum values at the project-store
boundary, include the complete resolved B2.5 style and renderer implementation
version in cache fingerprints, and cover these contracts with model,
persistence, CLI, Tauri, ASS, and temporal-preview tests. The interface shows
presets first and keeps the five bounded advanced controls secondary; it never
claims that a static UI sample is a rendered karaoke preview.

### Watermark settings

For a zero-contract project, watermark is disabled by default. A user may import
a local PNG. The backend validates it, copies it to a controlled project path,
and records immutable asset provenance:

```text
assets/watermark/<asset-id>.png
```

```json
{
  "watermark": {
    "enabled": true,
    "asset_id": "<uuid>",
    "asset_sha256": "<sha256>",
    "source_name": "brand.png",
    "position": "bottom_right"
  }
}
```

`position` is exclusively the existing seven-zone `WatermarkPosition` enum:
`center_top`, `center`, `center_bottom`, `top_left`, `top_right`,
`bottom_left`, and `bottom_right`. The UI never stores free watermark
coordinates. In a campaign project the contract remains authoritative: Studio
exposes only allowed zones and the backend validates them before preview and
render.

## Consequences

- A candidate can support many intentional compositions without losing analysis
  provenance.
- Preview and final render can consume identical validated timeline geometry.
- User work survives reanalysis but cannot silently render against changed
  evidence.
- UI drag motion stays local and inexpensive; backend work begins only on
  pointer release or explicit timeline operation.
- B2 requires a project store extension for atomic `project.json`, local assets,
  and preview cache lifecycle.

## Out of scope

- Tauri application scaffolding;
- final video render of the timeline;
- transcript rewriting or translation;
- free subtitle font families, raw ASS styles, arbitrary colors, free text
  coordinates, full-width/rounded caption boxes, or unbounded typography
  controls;
- free watermark coordinates;
- multiple independent source video files;
- unbounded variants or layout positions; permanent numeric limits require
  reproducible target-hardware evidence.
