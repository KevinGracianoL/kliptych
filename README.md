# Kliptych

Processing hours of video by hand is exhausting, but giving an AI total control over your code is terrifying. Kliptych was born to solve this: it does the heavy lifting of video processing, but is forbidden from merging code without your signature. Designed as an intelligent, headless campaign engine, it converts unstructured marketing briefs into fully validated, platform-compliant vertical video packages backed by mechanical proof.

---

## What It Does

Kliptych bridges the gap between chaotic creative direction and production-grade video delivery through four specialized execution modes orchestrated by campaign intelligence:

- **Long Video Engine (`run_long_video`)**: Automatically downloads source streams (`yt-dlp`/`streamlink`), transcribes speech with word-level timestamps (`faster-whisper`), detects high-impact moments through multimodal scene, audio energy, and chat density analysis, selects optimal cuts via LLM, and reframes subjects dynamically to 9:16 vertical video with burned karaoke subtitles (`.ass`).
- **Audio Locked Mode (`audio_locked=True`)**: Enforces specific viral soundtracks or voiceovers over video content using `ffmpeg amix`, guaranteeing calibrated volume ducking, sample-rate normalization, and strict track duration matching.
- **Repost / UGC Mode (`repost_mode=True`)**: Bypasses heavy inference layers entirely for ready-to-publish assets. Automatically detects and handles mobile orientation metadata, applying lossless passthrough (`-c copy`) without re-encoding whenever source media is already 9:16.
- **Slideshow Mode (`run_slideshow`)**: Assembles static visual assets (`JPG`/`PNG`) into paced, compliant 9:16 vertical reels using ffmpeg's `concat` demuxer, mandating synchronized external audio with zero unnecessary transcription overhead.
- **Campaign Intelligence & Self-Proposing Engine**: Evaluates incoming briefs against strict Pydantic schemas, classifying requirements into `KNOWN`, `KNOWN_WITH_VARIATION`, or `NEW_ARCHETYPE`. When new campaign archetypes appear, the system safely halts video generation and authors a structured GitHub Pull Request proposing the new configuration.

---

## Architecture

The Kliptych pipeline is organized into modular, decoupled layers with strict separation between business rules, inference runtimes, and external tooling:

```
[ Unstructured Brief ]
        │
        ▼
┌─────────────────────────────────┐
│ 1. Ingest & Contract Resolution │ ➔ Normalized JSON Contract with Text Evidence
└─────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────┐
│ 2. Campaign Intelligence Engine │ ➔ KNOWN / KNOWN_WITH_VARIATION / NEW_ARCHETYPE
└─────────────────────────────────┘
   │                      │
   │ (KNOWN)              │ (NEW_ARCHETYPE)
   ▼                      ▼
┌───────────────────────┐ ┌──────────────────────────────────────┐
│ 3. Media Acquisition  │ │ Safe Halt: ProposalEngine opens a PR │
│    & Transcription    │ │ (Zero Auto-Merge: awaits human sign) │
└───────────────────────┘ └──────────────────────────────────────┘
   │
   ▼
┌─────────────────────────────────┐
│ 4. Moment Detection & Selection │ ➔ Audio Energy + Scene Cuts + Chat Density
└─────────────────────────────────┘
   │
   ▼
┌─────────────────────────────────┐
│ 5. MediaPipe CPU 9:16 Reframe   │ ➔ Face Tracking (Zero VRAM Contention)
└─────────────────────────────────┘
   │
   ▼
┌─────────────────────────────────┐
│ 6. Subtitles & HW Render        │ ➔ NVENC / CPU Fallback + .ass Subtitles
└─────────────────────────────────┘
   │
   ▼
┌─────────────────────────────────┐
│ 7. Fail-Closed Compliance Gate  │ ➔ Mechanical ffprobe Validation
└─────────────────────────────────┘
        │
        ▼
[ Validated Delivery Package ] (Atomic Publish via Path.replace)
```

1. **Ingest & Resolution (Phases A & B)**: Multi-format parsing (TXT, Markdown, PDF, DOCX, stdin) resolving into typed contracts with mandatory field-level evidence.
2. **Campaign Routing & Intelligence (Phase E)**: Safe branching that separates known workflows from architectural evolutions.
3. **Media Acquisition & Ingestion (Phase C2)**: Bounded downloads with hard byte limits, explicit timeouts, and private network rejection.
4. **Speech & Moment Analysis (Phase C3)**: Local Whisper transcription (`small` int8) combined with multi-signal candidate scoring and structured LLM extraction.
5. **Computer Vision & Reframe (Phase C4)**: Real-time subject tracking using MediaPipe on CPU, preventing GPU memory starvation during rendering.
6. **Subtitles & Video Assembly (Phases C4 & D)**: High-performance rendering pipeline with `.ass` karaoke burning and NVENC acceleration.
7. **Compliance Gate (Phase A & F)**: Fail-closed mechanical inspection verifying video duration, dimensions, audio layout, and required caption metadata before atomic disk commit.

---

## Key Features

- 🛡️ **Zero Auto-Merge Guarantee**: The AI proposes code; humans dispose. When Kliptych discovers new campaign variants, it authors a git branch, generates schemas, logs variations, and opens a GitHub PR. It contains zero methods to approve or merge pull requests—the human engineer retains ultimate production control.
- 💾 **Atomic Checkpointing & Resumability**: Every pipeline stage commits state atomically (`mkstemp` + `Path.replace()`), surviving power loss and `kill -9`. On `--resume`, the engine validates `st_size > 0` and Pydantic schemas; corrupt artifacts (0 bytes or truncated JSON) trigger automatic single-stage regeneration.
- 🎮 **Zero VRAM Contention (4GB Target)**: MediaPipe runs strictly on CPU, while `faster-whisper` (int8) and `h264_nvenc` use GPU sequentially, NOT simultaneously. Explicit `cuda.empty_cache()` is called after transcription completes before NVENC rendering begins, preventing OOM on 4GB VRAM.
- ⚡ **Zero `shell=True` Security**: Every external invocation (`ffmpeg`, `ffprobe`, `yt-dlp`, `git`, `gh`) passes arguments as explicit lists (`list[str]`). Shell interpolation and injection vectors are completely eliminated from the codebase.
- 🧪 **97.42% Test Coverage & Hermetic CI**: Tested across 988 unit and integration tests under strict `-ra --cov-fail-under=90` enforcement. The CI suite runs with complete network isolation using loopback fixtures and recorded model replays.
- 🧹 **Cleanup Registry & Storage Lifecycle**: On success, `_CleanupRegistry` purges ALL intermediate files; on failure, heavy artifacts (downloads, transcripts) are preserved for instant `--resume` recovery. The `gc.py` module (`kliptych clean --days N`) acts as a TTL-based garbage collector for abandoned failed runs.
- 🧰 **Production CLI & Centralized Observability**: Full structured command suite (`env`, `ingest`, `run`, `campaign`, `clean`) with configurable logging levels (`--verbose`, `--quiet`) and automated disk garbage collection with TTL pruning.

---

## Checkpointing & Resumability

High-volume video processing cannot afford to start from scratch when interrupted by system reboots, network drops, or unhandled exceptions. Kliptych implements an atomic, self-healing checkpointing and resume mechanism:

- **Atomic State Writes**: Every pipeline stage writes its state checkpoints (`checkpoint.json`) and intermediate JSON manifests to a sibling temporary file via `tempfile.mkstemp` before committing them via `Path.replace()`. Sudden interruptions (`kill -9`, power loss) cannot corrupt or leave partially written state files on disk.
- **Integrity Validation on Resume**: When resumed with `--resume`, the engine validates stage artifacts before skipping execution: it verifies physical file existence and non-zero size (`st_size > 0`), and executes Pydantic schema validation (`model_validate_json`).
- **Targeted Single-Stage Regeneration**: If an artifact is corrupt (0 bytes, truncated JSON, or schema mismatch), only that individual stage is automatically invalidated and regenerated; all previously verified valid artifacts are safely reused.

---

## VRAM Lifecycle (Zero VRAM Contention)

Kliptych is explicitly engineered to operate deterministically on consumer GPUs with a strict **4GB VRAM target**:

- **Strict CPU/GPU Separation**: MediaPipe face detection and facial landmark tracking run strictly on the CPU, eliminating GPU memory footprint during subject tracking and reframe analysis.
- **Sequential GPU Execution**: Heavy GPU workloads—speech transcription via `faster-whisper` (`small` int8) and hardware video encoding via `h264_nvenc`—use the GPU sequentially, NOT simultaneously.
- **Explicit Memory Flush**: Explicit `cuda.empty_cache()` is called immediately after transcription completes, purging allocated VRAM before NVENC rendering begins. This avoids concurrent allocations and completely prevents Out-of-Memory (OOM) failures on 4GB targets.

---

## Cleanup Registry & Storage Lifecycle

Kliptych coordinates temporary file management and failure recovery across three distinct tiers:

- **Deterministic Success Purge**: On pipeline success, `_CleanupRegistry` purges ALL intermediate files (`.part` downloads, temporary audio/video clips, `.ass` subtitles, intermediate reframe renders) in `finally` blocks. Final delivery packages are published atomically via `Path.replace()`.
- **Targeted Failure Preservation**: When a pipeline stage fails, `_CleanupRegistry` selectively purges ephemeral scratch files while deliberately preserving heavy artifacts (downloaded source media, extracted audio tracks, transcripts) for instant `--resume` recovery without re-downloading or re-transcribing.
- **TTL Garbage Collection (`gc.py`)**: Abandoned failed runs or orphaned temporary workspaces are managed through `kliptych clean --days N` (implemented in `gc.py`), a TTL-based garbage collector that scans directory trees and safely reclaims disk space based on directory mtime thresholds.

---

## Quick Start

### Installation

Kliptych uses [uv](https://docs.astral.sh/uv/) for deterministic Python 3.13 virtual environment management:

```sh
# Clone and sync dependencies (including optional transcription and reframe tools)
uv sync --all-extras

# Verify environment and hardware acceleration (FFmpeg, NVENC, GPU)
uv run kliptych env
```

### Ingesting Campaign Briefs

Extract structured contracts with text evidence from local Markdown, PDF, DOCX, or stdin:

```sh
uv run kliptych ingest campaigns/fixtures/given-clips/brief.md
```

### End-to-End Pipeline Execution (`given_clips` Walkthrough)

Kliptych ships with self-contained, reproducible sample fixtures.

#### 1. Accepted Case (All Compliance Checks Pass)

```sh
uv run kliptych run campaigns/fixtures/given-clips/brief.md \
  --out runs/demo-pass/package \
  --recorded campaigns/fixtures/given-clips/recorded
```

```json
{
  "run_id": "20260923T050421195926Z",
  "outcome": "exported",
  "contract_sha256": "1ccafe8d5e0fcf45100e6eb947239742cd69b8cbb987640285ebaf738e622989",
  "package": "runs\\demo-pass\\package",
  "manifest": "runs\\20260923T050421195926Z\\run_manifest.json",
  "exported": ["clip-01"],
  "rejected": []
}
```

The exported package contains the verified video clip, `delivery_report.json` with full probe evidence, and localized platform metadata (captions, tags, audio guidelines).

#### 2. Rejected Case (Fail-Closed Gate)

When brief requirements are violated (e.g., missing required sponsor mentions `@marca`), the fail-closed gate halts delivery:

```sh
uv run kliptych run campaigns/fixtures/given-clips-rejected/brief.md \
  --out runs/demo-rejected/package \
  --recorded campaigns/fixtures/given-clips-rejected/recorded
```

```json
{
  "run_id": "20260923T050425560749Z",
  "outcome": "blocked",
  "contract_sha256": "1ccafe8d5e0fcf45100e6eb947239742cd69b8cbb987640285ebaf738e622989",
  "package": "runs\\demo-rejected\\package",
  "manifest": "runs\\20260923T050425560749Z\\run_manifest.json",
  "exported": [],
  "rejected": [
    {
      "piece_id": "clip-01",
      "platform": "tiktok",
      "reason": "el gate no aprobó la pieza (rejected): caption.required_mention"
    }
  ]
}
```

### Running Intelligent Campaigns

Execute full classification, checkpointing, and routing:

```sh
# Run with automatic stage resumption
uv run kliptych campaign campaigns/fixtures/given-clips/brief.md \
  --out runs/campaign-output \
  --mode long_video \
  --resume
```

### Maintenance and Garbage Collection

Prune temporary pipeline artifacts older than a given TTL:

```sh
uv run kliptych clean --days 7 --root campaigns
```

---

## Project Status

All foundational milestones (Fases A through F) are fully implemented, verified with end-to-end integration tests, and backed by evidence:

- [x] **Fase A — Foundation & Gate Core**: Pydantic v1.1 schemas, deterministic fail-closed gate (`ffprobe`), asset registry with SHA-256 integrity, and reproducible run manifests.
- [x] **Fase B — Walking Skeleton (`given_clips`)**: End-to-end headless pipeline, recorded model replays, atomic package exporter, and rejection reporting.
- [x] **Fase C — Long Video Processing Engine**: Bounded media download (`yt-dlp`), word-level Whisper transcription (`small` int8), multi-modal moment detection, LLM selection, MediaPipe CPU 9:16 dynamic reframe, styled karaoke subtitles (`.ass`), and NVENC/CPU rendering.
- [x] **Fase D — Special Operational Modes**: Audio Locked mixing with proportional volume control (`ffmpeg amix`), Repost/UGC passthrough with orientation normalization (`-c copy`), and Slideshow video assembly.
- [x] **Fase E — Campaign Intelligence & Self-Proposals**: Multi-archetype classifier (`KNOWN` / `KNOWN_WITH_VARIATION` / `NEW_ARCHETYPE`), CampaignManager orchestrator, and Git Proposal Engine operating under the strict Zero Auto-Merge rule.
- [x] **Fase F — Operational Maturity**: Structured CLI subcommands, centralized logging architecture, deterministic state checkpointing & resumption (`--resume`/`--restart`), and automated TTL garbage collection.

---

## Development

Kliptych enforces strict static analysis, complete type safety, and rigorous test coverage gates.

### Local Gate Commands

Every change must pass the full local gate suite before opening a PR:

```sh
# Code formatting check
uv run ruff format --check .

# Linting with strict ruleset
uv run ruff check .

# Static type checking in strict mode
uv run basedpyright

# Full test suite with coverage report (minimum 90% required)
uv run pytest
```

### LLM Runtime Configuration

When running against live models (outside of recorded test fixtures), configure the OpenAI-compatible environment:

```sh
export KLIPTYCH_LLM_BASE_URL="https://<endpoint>/v1"
export KLIPTYCH_LLM_API_KEY="<secret>"
export KLIPTYCH_LLM_MODEL="<model-identifier>"
```

### Git Hooks & Data Privacy

To enable automated pre-push branch protection:

```sh
git config core.hooksPath .githooks
```

- `campaigns/private/` and `runs/` are strictly ignored by git.
- CI fixtures are 100% synthetic or anonymized.
- Zero secrets, credentials, or personal identifiers in commits or logs.

---

## License

Proprietary. All rights reserved. Author: Kevin Graciano.
