# ADR 0005: Track B5 Studio Interactive Video Playback and Timeline Scrubbing Contract

- **Status:** Accepted
- **Date:** 2026-10-28
- **Decision owners:** Kliptych product and engineering

## Context

Tracks B1–B4 established the headless CLI, editable project models, typography presets,
fail-closed final export, and streaming ingest.

However, in the canvas workspace, the source frame is currently represented as a
static JPEG snapshot requested on-demand via `kliptych preview`. Adjusting split
points (`split-position`) or verifying motion across intervals requires generating
temporal previews or manually typing millisecond offsets.

VOD sources can exceed 5–10 GB in size. Loading uncompressed frames or entire video files
into memory violates the strict hardware budget (24 GB RAM, GTX 1650 Ti 4 GB VRAM)
and causes webview stutter.

Track B5 defines the contract for real-time interactive video playback, playhead
scrubbing, and temporal position synchronization directly inside Kliptych Studio.

## Decision

### 1. Zero-Copy Media Streaming via Tauri Asset Protocol & HTTP Range

Studio leverages Tauri 2's `protocol-asset` feature with native HTTP `Range: bytes=`
support:

- **Configuration:** `Cargo.toml` enables `features = ["protocol-asset"]`, and
  `tauri.conf.json` activates `security.assetProtocol`.
- **Streaming URL:** The webview converts local source paths via `convertFileSrc`:
  ```typescript
  const sourceVideoUrl = convertFileSrc(`${projectDir}/source.mp4`);
  ```
- **Byte-Range Streaming:** WebKitGTK / WebView2 / WKWebView issues standard HTTP 206
  partial requests to Tauri's internal asset server (`http-range`), allowing sub-10ms
  seeks anywhere in a 10 GB file with bounded memory consumption (< 150 MB buffer).

### 2. High-Frequency Scrubbing & GPU Performance Contract

In accordance with `better-ui` spatial and motion contracts:

- **State Separation:** High-frequency playhead movements (pointer drag, `timeupdate`,
  scrubber slider) are bound directly to `videoRef.current.currentTime` and a local
  DOM counter using `tabular-nums` and `font-mono`.
- **Render Throttle:** React state re-renders are suppressed during active pointer
  drags and committed only on release (`onPointerUp`), preventing full-tree layout
  thrashing.
- **Loop Bounding:** Playback is automatically bounded to the active variant or
  position window `[position.start_ms, position.end_ms]`, preventing accidental drift
  into unrelated sections of the source VOD.

### 3. Integrated Video Canvas & Crop Overlay

The editor canvas replaces the static JPEG image with an interactive `<video>` element
sharing the exact same container geometry:

- The crop rectangles (`localBands`) remain overlaid directly on top of the moving
  video stream, enabling live visual verification of framing as subjects move.
- A playback control bar provides:
  - Play / Pause toggle with spacebar keyboard shortcut and standard accessible states;
  - Scrubbing slider mapping normalized `0.0 .. 1.0` to `[start_ms, end_ms]`;
  - Current playhead readout in `mm:ss.ms` (`font-mono tabular-nums`);
  - "Cortar en playhead" button setting `splitAtMs` to the exact playhead timestamp.

## Consequences & Verification

- Creators can preview and scrub through multi-gigabyte VODs with instantaneous
  responsiveness and zero memory bloat.
- Split points can be placed visually at natural speech pauses or scene transitions.
- Verification command:
  ```bash
  npm --prefix studio run check
  ```
  Expected outcome: TypeScript 0 errors, Vite build successful, Rust check and test pass.
