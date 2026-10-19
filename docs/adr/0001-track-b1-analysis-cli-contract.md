# ADR 0001: Track B1 Analysis CLI Contract

- **Status:** Accepted
- **Date:** 2026-10-18
- **Decision owners:** Kliptych product and engineering

## Context

Track A provides a headless Python pipeline. Track B introduces Kliptych Studio,
a local desktop application built with Tauri. Studio needs to list candidate
clips before it renders any final video, so the Python engine needs a stable,
machine-readable analysis boundary.

The first Studio user is an individual creator who pastes a source URL. They
choose a local project directory and name before analysis begins. Campaign
briefs, contract editing, previews, final rendering, and the Tauri application
itself are not part of B1.

The contract must preserve existing guarantees:

- all processing is local after source retrieval;
- compliance remains owned by the Python backend and its fail-closed gate;
- normalized `LayoutSpec` remains the only geometry model when preview and
  rendering are added later;
- incomplete work must never be presented as a completed analysis;
- cached artifacts are reused only when their provenance is verifiable.

## Decision

### Process boundary

Studio starts the Python CLI as a local child process:

```text
kliptych analyze --url <source-url> --project-dir <project-directory> \
  --project-name <project-name>
```

The child-process protocol is used for B1. A persistent localhost HTTP service
is explicitly out of scope. This keeps the first integration surface small and
does not introduce ports, request authentication, server lifecycle, or a second
state model before Studio requires one.

`stdout` is a versioned NDJSON protocol: exactly one JSON object per line and
nothing else. Human-readable logs and diagnostics go to `stderr`. Studio must
not infer state from log text.

Each NDJSON event includes:

```json
{
  "schema_version": "1",
  "run_id": "<uuid>",
  "type": "progress",
  "stage": "transcribing",
  "percent": null,
  "message": "Transcribing source video"
}
```

The `stage` field is always present. `percent` is optional and is emitted only
when the producing stage knows it; the user interface must not fabricate a
percentage. Initial B1 stages are `downloading`, `transcribing`,
`finding_candidates`, `completed`, `failed`, and `cancelled`.

The child-process exit status is also part of the protocol, including when a
terminal failure prevents the final NDJSON event from being flushed:

- `0`: analysis completed, including a valid result with zero candidates;
- `1`: analysis failed;
- `2`: controlled user cancellation.

The `analyze` command must translate argument-parse and invocation errors to
exit status `1`. Python `argparse` normally uses status `2` for invalid CLI
syntax, but status `2` is reserved here exclusively for a cancellation initiated
by Studio; otherwise an invocation error with no final event could be displayed
as a user cancellation.

An error event has a stable code, the failing stage, and a safe user-facing
message. It does not contain stack traces, credentials, or source command
arguments with secrets:

```json
{
  "schema_version": "1",
  "run_id": "<uuid>",
  "type": "error",
  "stage": "downloading",
  "error_code": "source_download_failed",
  "message": "The source video could not be downloaded."
}
```

### Project directory and identity

Studio chooses a project name and resolves directory collisions before invoking
the CLI. The name crosses the B1 boundary explicitly through `--project-name`;
it is not derived implicitly from a filesystem path. If a desired directory is
occupied, Studio creates the first available predictable suffix, for example
`my-clip-001`, then `my-clip-002`. Python does not invent alternate locations
and rejects a project directory that is not safe for the requested operation.

For a new project, `--project-dir` must not exist when `analyze` starts. Python
creates it itself and records that it owns the directory for this invocation.
For reanalysis, `--project-dir` must be an existing Kliptych project with a
validated active `analysis.json`. Any other pre-existing directory, including an
empty directory or one containing unrelated files, is rejected without writes
or deletion. Python never recursively deletes a directory it did not create for
the current new-project invocation.

Each project has two identifiers:

- `project_id`: an internal UUID stable across renames;
- `project_name`: a human-readable, editable name.

The directory name is not the project identity. URLs are not project identities:
the same source can be analyzed in distinct projects.

The intended durable layout is:

```text
<project-directory>/
├── source.mp4
├── transcript.json
├── moments.json
├── analysis.json
├── project.json       # introduced by B2 for user edits
├── previews/          # introduced by B2
└── delivery/          # introduced by final render work
```

Only `analysis.json` is a B1 analysis result. `project.json` is not created in
B1; B2 may introduce a separate editable project-state artifact so future user
decisions do not overwrite the automatic analysis.

### Canonical analysis result

On success, `analyze` publishes one validated, versioned, atomic
`analysis.json`. It is the canonical persistent result; NDJSON communicates
process state but is not the durable source of truth.

The initial schema contains:

```json
{
  "schema_version": "1",
  "project_id": "<uuid>",
  "project_name": "My clip",
  "source": {
    "url": "https://example.invalid/video",
    "sha256": "<sha256-of-complete-source>"
  },
  "provenance": {
    "transcription_version": "<registered-transcription-version>",
    "segmenter_version": "<registered-segmenter-version>"
  },
  "candidates": [
    {
      "candidate_id": "<deterministic-hash>",
      "recommended_start_ms": 80000,
      "recommended_end_ms": 115000,
      "score": 0.91,
      "excerpt": "Transcript text in the recommended interval."
    }
  ]
}
```

The source URL is stored locally in the analysis result to make source
provenance visible and support a deliberate retry. It is potentially sensitive
local data and must not be emitted in diagnostics unnecessarily.

Time values in this boundary are integer milliseconds. The user interface is
responsible for formatting them; Python, Rust, and TypeScript must not exchange
floating-point seconds through this protocol.

`score` is the existing normalized `Moment.score`: a finite floating-point
number in the inclusive range `[0.0, 1.0]`. Studio may format it as a percentage
for display, but it must retain the raw value in the protocol and must not treat
it as an independently calibrated confidence.

The segmenter retains its current policy: it selects up to twelve qualifying
moments. `analysis.json` exposes those candidates in descending score order,
even though internal pipeline artifacts may preserve a chronological order.

`candidate_id` is a deterministic hash of the analysis schema/version and the
candidate's source-derived identity, including its recommended interval. It is
not a ranking number and does not become random when the same analysis is
reproduced.

A successful analysis with no qualifying candidates is valid:

```json
{
  "schema_version": "1",
  "project_id": "<uuid>",
  "project_name": "My clip",
  "source": {
    "url": "https://example.invalid/video",
    "sha256": "<sha256-of-complete-source>"
  },
  "provenance": {
    "transcription_version": "<registered-transcription-version>",
    "segmenter_version": "<registered-segmenter-version>"
  },
  "candidates": []
}
```

It emits `completed`, not `failed`.

### Cache reuse and atomic publication

On reanalysis, complete caches may be reused only when:

1. the complete source video matches its recorded SHA-256; and
2. the registered `transcription_version` and `segmenter_version` in the active
   analysis provenance are compatible with the versions requested for this run.

If either check fails, processing restarts at the first incompatible stage.
URL string equality alone is insufficient because remote content can change.

Reanalysis writes artifacts into staging and replaces the active analysis only
after all required stages complete and `analysis.json` validates. A failed or
cancelled reanalysis preserves the previously valid active analysis and removes
the temporary staging work.

For a brand-new project, failure or user cancellation removes only the project
directory created by this invocation, including otherwise complete caches. This
makes cancellation mean “undo this new attempt” rather than leave an ambiguous
partially-created project, without permitting deletion of a pre-existing path.

### Cancellation

Studio cancels `analyze` by sending a controlled termination signal to the child
process. Python handles the signal, cleans staging, emits a `cancelled` NDJSON
event when process state permits, and exits with a documented non-success code.
A forceful kill is reserved for unrecoverable process termination and does not
define successful cancellation behavior.

## Deferred B2 design direction (non-normative)

Candidature is intended to remain immutable analysis evidence. The product
direction accepted during B1 discovery is that B2 should store user edits
separately in `project.json`, rather than overwriting `analysis.json`:

```json
{
  "selection_id": "<uuid>",
  "source_candidate_id": "<candidate-id>",
  "start_ms": 70000,
  "end_ms": 115000
}
```

This direction records that an automatic recommendation and a user's creative
selection are different facts. B2 must validate and adopt its own schema before
creating this file. It must also make a separate evidence-backed duration
decision. B1 preserves rather than rewrites existing policy: the local
segmenter currently emits technical candidates within its own 2–60 second
window, while the zero-contract contract currently applies a 3 second minimum
and no contractual maximum. The final render backend remains responsible for
enforcement.

## Consequences

- B1 can be tested from a terminal without starting Tauri.
- The UI receives structured progress and errors without parsing logs.
- A future persistent service remains possible, but must preserve this domain
  contract or deliberately version it.
- Existing pipeline checkpoints and implementation artifacts are not a public UI
  API.
- B2 must use the same validated `LayoutSpec` model that the assembler consumes;
  it may not implement independent pixel geometry.

## Out of scope

- Tauri/Rust/TypeScript application scaffolding;
- preview frames or temporal previews;
- final rendering and delivery export;
- campaign briefs, contract generation, and compliance approval UI;
- editable time ranges and `project.json` creation or schema adoption;
- changing the current maximum of twelve automatic candidates;
- a local HTTP service.
