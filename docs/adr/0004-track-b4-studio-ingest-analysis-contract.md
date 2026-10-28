# ADR 0004: Track B4 Studio Ingest and Analysis Streaming Contract

- **Status:** Accepted
- **Date:** 2026-10-28
- **Decision owners:** Kliptych product and engineering

## Context

Track B1 established the headless `kliptych analyze` CLI command, emitting versioned
`analysis.json` and streaming NDJSON progress stages (`downloading`, `transcribing`,
`finding_candidates`, `completed`).

Tracks B2 and B3 built the desktop editor experience for project inspection, layout
composition, caption typography presets, temporal motion previews, and fail-closed
final export. However, opening a project in Studio currently requires the operator to
run `kliptych analyze` externally via a terminal before clicking "Abrir proyecto".

Track B4 closes this usability gap by integrating the end-to-end ingest and analysis
workflow directly into the Kliptych Studio desktop shell, maintaining:

- zero network latency post-download and local, deterministic ASR/segmentation;
- non-blocking streaming progress in the desktop UI using Tauri 2 IPC channels;
- cooperative cancellation cleanly aborting subprocesses without orphan stage files;
- seamless transition from analysis completion into editable project state.

## Decision

### 1. Tauri IPC Streaming Protocol

Studio coordinates ingest and analysis via a dedicated Tauri command:

```rust
#[tauri::command]
async fn analyze_url(
    state: tauri::State<'_, BackendPath>,
    on_event: tauri::ipc::Channel<Value>,
    url: String,
    project_dir: String,
    project_name: String,
) -> Result<Value, String>
```

- **Streaming Channel:** Consumes `tauri::ipc::Channel<Value>` to stream stdout
  lines from `kliptych analyze` directly to the React webview in real time without
  polling or arbitrary timers.
- **Child Process Execution:** Spawns `kliptych analyze --url <url> --project-dir <dir> --project-name <name>`
  with piped stdout and stderr.
- **Event Forwarding:** Reads lines sequentially via `BufReader`. For each valid
  JSON event line, it emits the parsed `Value` through `on_event.send()`.
- **Exit Status Handing:**
  - Code `0`: Resolves `Ok(completed_event)`.
  - Code `2`: Resolves `Err("Análisis cancelado.")`.
  - Code `1` or failure: Resolves `Err(error_message)` extracted from the error payload.

### 2. UI State Machine & Progress Visualizer

The Studio welcome view provides two distinct entry points:
1. **Abrir proyecto existente:** Preserves existing folder selection for pre-analyzed projects.
2. **Nuevo proyecto desde URL:** Opens an inline creation form:
   - Source URL input (`placeholder="https://www.youtube.com/watch?v=..."`);
   - Project Name input (defaults to sanitized URL or "Mi Proyecto");
   - Destination Directory picker (using Tauri's native `open({ directory: true })` dialog).

During analysis, the UI renders a dedicated non-blocking progress card (`better-ui` compliant):
- **Stage Badges:** Highlights current phase:
  - `downloading`: "Descargando fuente..."
  - `transcribing`: "Transcribiendo audio (Whisper)..."
  - `finding_candidates`: "Segmentando momentos por densidad de habla..."
- **Cancellation:** An explicit "Cancelar" button sends a cancellation request,
  triggering graceful termination of the running child process.
- **Auto-Inspection:** Upon receiving the `completed` event, the UI automatically
  inspects and initializes the editable state, seamlessly transitioning to the
  editor workspace with detected candidates displayed in the sidebar.

### 3. Cooperative Cancellation & Subprocess Hygiene

- When the operator cancels from the UI or closes the window during an active analysis,
  the backend terminates the child process using `SIGINT`/`SIGTERM` (or `taskkill`
  tree kill on Windows).
- Python's `_cmd_studio_analyze` signal handler intercepts the signal, raises
  `AnalysisCancelled`, cleans up staged files in the project store, emits a
  `type: "cancelled"` NDJSON event, and exits with code 2.

## Consequences & Verification

- Operators can go from an external video link to completed vertical short exports
  entirely within Kliptych Studio without touching the command line.
- The Python CLI contract (`kliptych analyze`) remains completely unchanged and
  testable in headless environments.
- Verification command:
  ```bash
  npm --prefix studio run check
  ```
  Expected outcome: TypeScript 0 errors, Vite build successful, Cargo check pass.
