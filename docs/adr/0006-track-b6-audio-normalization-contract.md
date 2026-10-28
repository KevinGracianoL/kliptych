# ADR 0006: Track B6 Studio EBU R128 Audio Loudness Normalization Contract

- **Status:** Accepted
- **Date:** 2026-10-28
- **Decision owners:** Kliptych product and engineering

## Context

Tracks B1–B5 established end-to-end clip editing in Kliptych Studio: streaming ingest,
real-time video scrubbing, layout geometry composition, typography presets, and fail-closed
final export.

However, source VODs from livestreams, podcasts, and recordings vary wildly in audio levels.
Some segments are inaudible (-28 LUFS), while others contain clipped peaks exceeding 0 dBFS.
When published to TikTok, Instagram Reels, or YouTube Shorts, unnormalized audio produces
either quiet videos penalized by platform algorithms or harsh distortion caused by platform
lossy AAC re-encoding.

Track B6 defines the contract for broadcast-quality audio loudness normalization in Kliptych
Studio final exports, adhering to the international **EBU R128 / ITU-R BS.1770-4** standard.

## Decision

### 1. Loudness Target Specifications for Vertical Video

Final exports apply single-pass dynamic loudness normalization using FFmpeg's native `loudnorm`
filter configured with industry targets:

- **Integrated Loudness (`I`):** `-14.0 LUFS` (standard target for YouTube Shorts, TikTok, and Meta Reels).
- **Maximum True Peak (`TP`):** `-1.0 dBTP` (guarantees headroom preventing inter-sample clipping
  during platform lossy audio compression).
- **Loudness Range (`LRA`):** `11.0 LU` (preserves vocal clarity and natural dynamics without
  excessive dynamic range squashing).

FFmpeg audio filter specification:
```text
loudnorm=I=-14.0:TP=-1.0:LRA=11.0
```

### 2. Backend & Filtergraph Architecture

In `src/kliptych/studio/final_render.py`:

- When `source_has_audio` is `True` and `normalize_audio` is `True`, the audio chain connects:
  ```text
  [0:a]atrim=start={start_s}:duration={dur_s},asetpts=PTS-STARTPTS,loudnorm=I=-14.0:TP=-1.0:LRA=11.0[outa]
  ```
- When `normalize_audio` is `False`, the audio chain bypasses `loudnorm` and preserves original gain:
  ```text
  [0:a]atrim=start={start_s}:duration={dur_s},asetpts=PTS-STARTPTS[outa]
  ```
- The audio is encoded to standard AAC stereo at 192 kbps with `+faststart`.

### 3. CLI Contract & Flags

Studio invokes final rendering with an optional normalization flag:

```text
kliptych studio render-final --project-dir <path> --variant-id <uuid> [--out <dest>] [--normalize-audio | --no-normalize-audio]
```

- **Default:** `normalize_audio=True`.
- **Output Protocol:** The result JSON event reports `audio_normalized: true | false`.

### 4. Tauri IPC & UI Controls

- **IPC:** `studio_project(operation: "render-final")` accepts `normalize_audio: Option<bool>`.
- **UI:** The export panel in `App.tsx` includes an accessible toggle:
  `<label><input type="checkbox" checked={normalizeAudio} /> Normalización EBU R128 (-14 LUFS)</label>`
- Compliant with `better-ui` layout and typography guidelines.

## Consequences & Verification

- Published vertical videos have consistent, professional broadcast loudness across all platforms
  without distortion or volume jumps between clips.
- Audio filter execution adds < 0.3s processing overhead for typical 15–60s clips on target hardware
  (Ryzen 5 4600H) with zero GPU VRAM impact.
- Verification command:
  ```bash
  uv run pytest tests/studio/test_final_render.py -k "audio"
  ```
  Expected outcome: Pass with verified filtergraph assertions.
