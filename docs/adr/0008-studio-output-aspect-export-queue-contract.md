# ADR 0008: Studio output aspect and persistent export queue

- **Status:** Accepted
- **Date:** 2026-11-02
- **Decision owners:** Kevin Graciano and Kliptych engineering

## Context and scope

The first Studio delivery must include horizontal composition and a persistent
batch export queue, not defer them to a later release. The operator confirmed:
continue after individual failures, retain the queue across restarts, and create
a copy rather than replace a variant when changing its output aspect.

This extends ADR 0002's editable project boundary and supersedes ADR 0003's
rejection of explicit batch selection. It does not introduce eager rendering of
all automatic candidates or weaken the Gate. Translation and additional output
aspects are out of scope. This decision does not authorize a push or release.

## Output aspect and compatibility

- A variant has one `output_aspect`, restricted to existing `Aspect.VERTICAL`
  (`9:16`) and `Aspect.HORIZONTAL` (`16:9`). Layout presets and output aspects
  remain separate concepts.
- New project writes use schema `2`. Read schema `1` without changing existing
  vertical semantics; a legacy document cannot declare a horizontal variant.
  Upgrade only during an explicit write, not merely by opening a project.
- A format copy retains evidence, selected times, normalized crops, captions,
  watermark and review status, with fresh variant and position IDs. The original
  remains unchanged. Its previews cannot be reused by the copy.
- Source editing coordinates remain normalized. Shared canvas sizing drives
  JPEG previews, temporal previews, ASS layout and final assembly: 720×1280 /
  1080×1920 vertically and 1280×720 / 1920×1080 horizontally.
- Include output aspect in derived-preview fingerprints and validate cache
  dimensions. Gate evaluation declares the chosen aspect in the existing
  contract output specification; do not modify or bypass `src/kliptych/gate/`.

## Persisted queue and revision binding

- Keep a versioned `export_queue.json` alongside `project.json`; preserve it
  and previously published `delivery/` artifacts during analysis replacement.
  Persist each transition atomically, reject
  symlinks/junctions and malformed or unknown-version state.
- Each job records a UUID, variant ID/name, canonical revision digest, explicit
  audio normalization, controlled unique relative output path, state, and an
  actionable result/error. No secrets, transcript content or absolute user paths
  need to be stored in the queue.
- Revision digest includes the complete variant and source/analysis/transcript
  evidence binding. Changed, removed or review-required variants are rejected
  before render; re-enqueue the desired revision explicitly.
- States: `pending -> running -> completed | failed`; abandoned `running` jobs
  become `interrupted` only when no worker owns them. Explicit retry returns
  `failed`/`interrupted` jobs to pending after revalidation. Stale revisions must
  be enqueued afresh, never implicitly rebound.
- A per-project worker lease rejects concurrent workers. Queue state mutations
  must also be serialized to avoid losing enqueue/stop updates during a render.
- The worker processes jobs in order, one at a time, and calls the existing
  final-render service. Failure of one job is recorded and processing continues.
  Each file requires the complete Gate pass before publication. All successes,
  partial failures and interruptions are reported honestly.
- Destination names incorporate job identity. Never overwrite an existing
  output, including on retry or recovery after publication but before the
  completion state was persisted. That ambiguous case requires user review.
  Reject links below the trusted project/output root, but allow an otherwise
  valid project reached through a symlinked ancestor (including OS path aliases).
  No-overwrite publication requires a filesystem supporting same-filesystem
  hardlinks; unsupported filesystems fail explicitly rather than overwrite.
- `Stop after current` is persisted, does not kill the active render and leaves
  remaining jobs pending. Starting/resuming the queue is explicit; reopening
  Studio does not automatically run recovered jobs.
- SIGINT/SIGTERM during a render interrupts the current job and stops the worker;
  remaining jobs stay pending. A queue-read error stays local to the panel and
  must not prevent opening a different project.
- While the queue worker runs, Studio cannot start another final render,
  analysis or expensive preview. Editing saved decisions may continue; revision
  validation protects pending jobs from silently rendering new edits.

## UI and process boundary

The UI selects active variants, adds them to the queue, starts/resumes work,
requests stop-after-current and explicitly retries eligible failures. It shows
empty/loading/error states, persisted per-job results and aggregate counts.
The backend is authoritative for jobs, revision checks and state transitions.
Tauri invokes argument-list CLI operations without a shell and must not block
its UI event loop while the worker runs. A live status refresh is read-only;
it cannot misclassify a running worker as interrupted.

## Alternatives rejected

- A CSS-only horizontal frame: does not change assembly, ASS or Gate semantics.
- Editing the original on format change: destroys an intentional composition.
- Browser-only or in-memory queues: lose work on app restart.
- Rendering the latest mutable variant: silently exports a different revision.
- Parallel exports: violate the sequential GPU-stage constraint.
- Batch-wide rollback: discards independently verified successful exports.
- Automatic retries or overwrite-on-recovery: risk duplicate/destructive work.

## Acceptance and verification

Use RED-first behavioral tests for legacy parsing, schema/aspect rejection,
lossless format copies, both canvas dimensions, aspect-specific cache identity,
ASS dimensions, final Gate aspect declaration and mismatched-output rejection.
Queue tests cover sequential execution, failure continuation, stop, restart,
stale revisions, review status, worker contention, atomic write failures,
malformed state, path/link rejection and no-overwrite recovery.

Commands (record literal results; these are plans, not measured outcomes):

```sh
uv run pytest tests/studio -q
~/.config/opencode/scripts/verify-gates.sh
npm --prefix studio run check
cargo test --manifest-path studio/src-tauri/Cargo.toml
```

Independent `reviewer` review is required for the complete change. Exercise the
actual desktop flow and render synthetic media in both aspects through FFmpeg
and the real Gate. Hardware measurements require the actual input and literal
commands/results before any latency, RAM or VRAM claim; no such claim is made
by this ADR. Installer validation on each supported OS remains a release gate.
