# ADR 0003: Track B3 Studio Final Render and Export Contract

- **Status:** Accepted
- **Date:** 2026-10-28
- **Decision owners:** Kliptych product and engineering

## Context

Track B1 introduced headless analysis emitting `analysis.json`. Track B2 added
human creative editing: continuous composition timelines (`LayoutTimeline`),
custom rectangular crops, local PNG watermarks (7 zones), attributed human
caption overrides, temporal motion previews (B2.4b), and typography presets
(B2.5).

Studio operators now need to export their completed variants to publishable
deliverables. In traditional clip editors, rendering is often a monolithic
batch step that blocks the UI and lacks rigorous compliance verification.

Track B3 defines the contract for lazy, on-demand final rendering and export of
individual variants directly from Kliptych Studio, retaining:

- strict local processing and zero network latency;
- the single source of truth for geometry (`LayoutSpec` and `source_rect`);
- identical visual output between preview and final export (Invariant 9);
- hardware-aware encoding (NVENC acceleration with logged CPU fallback on target
  GTX 1650 Ti / Turing TU117 hardware);
- fail-closed gate compliance verification (Invariant 3).

## Decision

### 1. Process Boundary & CLI Command

Studio invokes final rendering via the Python CLI child-process protocol:

```text
kliptych studio render-final --project-dir <project-dir> --variant-id <uuid> [--out <destination>]
```

- **Namespace:** Lives under the `studio` command group alongside `inspect`,
  `preview-motion`, and `set-captions`, maintaining project isolation and
  preventing ambient unversioned invocations.
- **Output Protocol:** Emits structured JSON events to `stdout` with
  `schema_version: "1"`:
  ```json
  {
    "schema_version": "1",
    "type": "render_completed",
    "variant_id": "<uuid>",
    "output_path": "delivery/Variante 1.mp4",
    "duration_ms": 21000,
    "width": 1080,
    "height": 1920,
    "gate_verdict": "pass",
    "encoder_used": "h264_nvenc"
  }
  ```
- **Error Reporting:** Returns exit code `1` and emits an error event on any
  failure:
  ```json
  {
    "schema_version": "1",
    "type": "error",
    "error_code": "gate_rejected",
    "message": "Gate check failed: aspect ratio mismatch"
  }
  ```
- **Cancellation:** Supports cooperative termination when Tauri terminates the
  child process upon user cancellation, deterministically removing incomplete
  temporary files.

### 2. Multi-Position Timeline Assembly

A variant can define multiple composition positions in its `layout_timeline`
(e.g., 0:00–0:08 in Vertical and 0:08–0:21 in Dividido).

Final rendering assembles the complete timeline into a single, continuous video:

1. **Video Chains:** For each position in `layout_timeline`, an FFmpeg filter
   chain extracts the declared bands from `source.mp4`, applies the exact
   normalized `source_rect`, stacks multi-band layouts (`vstack`), and trims the
   interval `[start_ms, end_ms]`.
2. **Visual Concatenation:** Positions are concatenated in chronological order.
3. **Audio Preservation:** Audio from `source.mp4` for the variant's range
   `[variant.start_ms, variant.end_ms]` is trimmed and muxed without re-encoding
   or desynchronization.
4. **Watermark Burn-in:** If enabled, the validated local PNG asset is overlaid
   at the target zone from the 7-zone `WatermarkPosition` enum.
5. **Captions Burn-in:** If enabled, karaoke SubStation Alpha subtitles are
   rendered:
   - using the resolved B2.5 preset (`clean`, `bold`, `boxed`) and secondary
     overrides;
   - using the packaged `Noto Sans Regular` font via `fontsdir`;
   - incorporating B2.4a human caption overrides and bounds clipping.

### 3. Fail-Closed Gate Evaluation

Before an exported file is declared completed, the backend runs the project's
rigorous compliance gate (`Gate.evaluate`):

- **Geometry & Aspect:** Confirms exact target aspect ratio (`9:16`, `1080x1920`
  or declared output).
- **Duration Integrity:** Confirms duration matches the human-selected interval
  `[start_ms, end_ms]`.
- **Audio & Video Integrity:** Confirms presence of valid audio/video streams,
  detects unintended black frames or prolonged silence.
- **Fail-Closed Policy:** If the Gate returns `FAIL`, the deliverable is not
  published, temporary files are cleaned up, and Studio reports the exact gate
  failure to the operator.

### 4. Destination & File Naming

- **Default Path:** `<project-dir>/delivery/<sanitized-variant-name>.mp4`
  (e.g., `runs/<project-id>/delivery/Variante 1.mp4`).
- **Custom Output (`--out`):** When the operator specifies an explicit target
  file via Tauri's native file dialog, the deliverable is published to that path.
- **Atomic Publishing:** The render is written to a temporary sibling
  `.tmp.mp4` and atomically published via `.replace()` only after successful
  Gate verification.

## Rejected Alternatives

- **Batch export of all candidates:** Rejected. As measured in Track B1,
  rendering 12 clips eagerly wastes GPU time and VRAM. Lazy, single-clip export
  ensures sub-minute turnaround for the creator's chosen variant.
- **Bypassing the Gate for human edits:** Rejected. Invariant 3 dictates zero
  exceptions in the Gate. Human crop edits must still satisfy video compliance
  before publication.
- **Separate per-position clips:** Rejected. Creators expect a single stitched
  clip ready for upload, not separate files requiring external NLE assembly.
- **Embedded HTTP API server:** Rejected. Retains the simple, process-isolated
  NDJSON child-process contract proven in B1 and B2.

## Consequences & Verification

- Creators can produce final, publishable vertical videos directly from
  Kliptych Studio with single-click simplicity.
- The pipeline contract remains headless, testable via CLI, and independent of
  the Tauri shell.
- Verification command:
  ```bash
  uv run kliptych studio render-final --project-dir <path> --variant-id <uuid>
  ```
  Expected outcome: Exit code 0, deliverable in `delivery/`, and full gate pass.
