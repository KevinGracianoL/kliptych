# ADR 0007: Track B7 Kliptych Studio Packaging and Multi-Platform Release Contract

- **Status:** Accepted
- **Date:** 2026-10-28
- **Decision owners:** Kliptych product and engineering

## Context

Tracks B1–B6 established end-to-end functionality for Kliptych Studio: streaming ingest,
real-time video scrubbing, layout geometry composition, typography presets, fail-closed
final export, and EBU R128 audio normalization.

However, running the desktop application currently requires a developer toolchain (Node.js,
Rust, Cargo, and Tauri CLI). For creators and non-developer operators, Studio must ship as
self-contained, standard operating system installers across Linux, Windows, and macOS.

Track B7 defines the contract for desktop release packaging, automated GitHub Actions CI/CD
publishing, and deterministic local backend resolution.

## Decision

### 1. Multi-Platform Installer Targets

Studio configures Tauri 2 bundler targets in `studio/src-tauri/tauri.conf.json`:

- **Linux (`ubuntu-latest` / glibc):**
  - `.deb`: Standard Debian package with explicit runtime dependencies:
    `libwebkit2gtk-4.1-0`, `libayatana-appindicator3-1`, `librsvg2-2`.
  - `.AppImage`: Portable single-binary distribution.
- **Windows (`windows-latest`):**
  - `.msi`: Enterprise/Windows Installer target using WiX v4.
  - `.exe`: Lightweight installer using NSIS.
- **macOS (`macos-latest` / Universal / Apple Silicon):**
  - `.dmg`: Drag-and-drop disk image with Applications shortcut.
  - `.app`: Signed application bundle.

### 2. Cascading Backend Binary Resolution

Studio relies on the local Python CLI `kliptych` for heavy processing. To minimize
setup friction, `studio/src-tauri/src/lib.rs` implements automated cascading binary
resolution on startup:

1. **Explicit Setting:** Stored user override from the UI dialog (`set_backend_path`).
2. **Sibling Directory:** Looks beside the running Studio executable (`exe_dir.join("kliptych")`
   or `exe_dir.join("kliptych.exe")`) for portable installations.
3. **User Binaries:** Checks standard user installation paths:
   - Linux / macOS: `~/.local/bin/kliptych`, `~/.cargo/bin/kliptych`.
   - Windows: `%USERPROFILE%\.local\bin\kliptych.exe`, `%USERPROFILE%\AppData\Local\Programs\kliptych\kliptych.exe`.
4. **Environment PATH:** Resolves via system path lookup (`which` / `where`).
5. **Fallback:** Falls back to `"kliptych"`.

### 3. Automated Release Pipeline (`.github/workflows/release.yml`)

Release generation is automated via GitHub Actions:

- **Trigger:** Pushed Git version tags matching `v*` (e.g., `v0.1.0`).
- **Matrix:** Parallel compilation across `ubuntu-24.04`, `windows-latest`, and `macos-latest`.
- **Packaging Action:** Uses the official `tauri-apps/tauri-action` to build Vite client
  assets, compile release binaries with LTO optimization, package installers, and publish
  a draft GitHub Release with attached assets.

## Consequences & Verification

- Creators can download and install Kliptych Studio like any standard desktop software.
- The desktop shell automatically detects the Python engine without manual terminal setup.
- Verification command:
  ```bash
  npm --prefix studio run check
  ```
  Expected outcome: 0 errors across TypeScript, Vite, cargo fmt, and cargo check.
