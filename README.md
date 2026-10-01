# Kliptych

Feed it a campaign brief; it delivers publication-ready vertical clips and ruthlessly blocks anything that breaks the brand's rules

[![CI](https://github.com/KevinGracianoL/kliptych/actions/workflows/ci.yml/badge.svg)](https://github.com/KevinGracianoL/kliptych/actions) [![Release](https://img.shields.io/github/v/release/KevinGracianoL/kliptych?color=2563eb&logo=github&logoColor=white)](https://github.com/KevinGracianoL/kliptych/releases) ![Tests](https://img.shields.io/badge/tests-%E2%89%A51%2C500_passed-10b981?logo=pytest&logoColor=white) ![Coverage](https://img.shields.io/badge/coverage-%E2%89%A590%25_(CI_gate)-0d9488)
![OS](https://img.shields.io/badge/OS-ubuntu_%7C_windows-334155?logo=githubactions&logoColor=white) ![GPU](https://img.shields.io/badge/GPU-CUDA_12_%C2%B7_Faster--Whisper_%C2%B7_MediaPipe_%C2%B7_NVENC-76b900?logo=nvidia&logoColor=white) ![Python](https://img.shields.io/badge/python-3.13%2B-3776ab?logo=python&logoColor=white) ![Architecture](https://img.shields.io/badge/architecture-fail--closed_gate-d97706)

Content Rewards / UGC campaigns only pay when every contractual rule holds:
exact brand spelling, mandatory mentions and hashtags, duration bounds, audio
requirements, watermark coverage. Human review does not scale to dozens of
cuts, and an LLM verdict is not evidence.

Kliptych converts a chaotic campaign brief into a validated vertical-video
delivery package through a versioned JSON contract, a GPU pipeline
(download, transcription, moments, 9:16 reframe, karaoke subtitles, hardware
render), and a fail-closed compliance gate.

What it does NOT do: it never auto-publishes, never lets an LLM clear a hard
rule, and never ships a partial delivery — a rejected piece aborts the whole
batch.

![Demo](assets/samples/given-clips-sample.mp4)

## Quickstart

Requires Python 3.13+ and [`uv`](https://docs.astral.sh/uv/).

```sh
uv sync
uv sync --extra transcription --extra reframe  # CUDA + MediaPipe
uv run kliptych env                             # toolchain, GPU, NVENC
uv run kliptych ingest campaigns/fixtures/given-clips/brief.md
uv run kliptych run campaigns/fixtures/given-clips/brief.md \
  --out runs/demo-pass/package \
  --recorded campaigns/fixtures/given-clips/recorded
```

Accepted run prints `"outcome": "exported"`; a fixture missing a mandatory
mention prints `"outcome": "blocked"` and exports nothing.

## Modes

| Mode | Plain language |
|------|----------------|
| `long_video` | Full chain from a long URL: downloads the best moments, cuts them, and renders vertical clips with subtitles. |
| `audio_locked` | Same as above but dubs a mandatory external audio track over the video (slideshows require it). |
| `repost_ugc` | Reuses an already-approved clip: no re-transcription, just reframe, metadata cleanup, and caption rewrite. |
| `slideshow` | Turns still images into a video paced against the external audio track. |

## Why the Gate rejects

Any single failed check rejects the piece, and anything the gate cannot
mechanically prove pauses for explicit human approval instead of passing
silently — see [docs/gate.md](docs/gate.md).

## Learn more

- [Architecture — 4 Engineering Pillars](docs/architecture.md)
- [Benchmarks — GTX 1650 Ti, 4 GB VRAM](docs/benchmarks.md)
- [Gate — Fail-Closed Compliance Engine](docs/gate.md)
- [Operations — CLI 2-Step Workflow](docs/operations.md)

<details><summary>Technical detail: fail-closed export barrier</summary>

`_derive_status` maps any measured `FAIL` to `REJECTED`, whatever the rule
severity. The exporter (`_is_piece_exportable`) ships only `PASSED` pieces,
or `PENDING_REVIEW` with zero `FAIL`s plus `--approve-manual-review
--approved-by "<operator>"`. Batch export is atomic via staging + rename:
one rejection aborts the delivery, never a partial shipment.

</details>

<details><summary>Technical detail: constrained-VRAM GPU orchestration</summary>

One GPU stage at a time: `faster-whisper` (`int8` CUDA, in-memory cache)
transcribes and releases VRAM, MediaPipe `FaceDetector` (CPU delegate, zero
VRAM) drives the 9:16 reframe, FFmpeg `h264_nvenc` renders with `libx264`
fallback. Checkpoints carry SHA-256 fingerprints so `--resume` regenerates
exactly the stale stage. Full numbers in
[docs/benchmarks.md](docs/benchmarks.md).

</details>

<details><summary>Technical detail: human-in-the-loop approval</summary>

Unprovable rules resolve to `MANUAL_REVIEW`, never `PASS`. The run pauses at
`PENDING_REVIEW` (exit 1, no `delivery/`), and resumes only with an
attributed signature that is sealed into the delivery record next to
`brief_sha256` / `contract_sha256`. Full workflow in
[docs/operations.md](docs/operations.md).

</details>

## Project Status

Phases A–F plus Sprints 1–3 implemented, each executable end-to-end. Full
history in [docs/architecture.md](docs/architecture.md); `campaigns/private/`
and `runs/` are git-ignored and CI fixtures are synthetic.

## License

Proprietary. All rights reserved. Author: Kevin Graciano.
