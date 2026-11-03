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
    `libwebkit2gtk-4.1-0`, `libayatana-appindicator3-1`, `librsvg2-2`,
    `gstreamer1.0-plugins-good`, `gstreamer1.0-plugins-bad`, `gstreamer1.0-libav`.
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

### Linux media runtime correction (2026-11-03)

Opening the synthetic Studio project on EndeavourOS produced a grey WebView.
The application log recorded `GStreamer element autoaudiosink not found` and
the corresponding WebKit process ended. A direct GStreamer factory probe returned
`autoaudiosink: false`; after an authorized installation of `gst-plugins-good`,
the same probe returned `autoaudiosink: true`. This is a native playback-runtime
dependency, not evidence of a corrupt project or an FFmpeg render failure.

- Debian packages declare `gstreamer1.0-plugins-good`,
  `gstreamer1.0-plugins-bad` and `gstreamer1.0-libav` as runtime dependencies.
  They provide MP4 demuxing, H.264 parsing, audio output and H.264/AAC decoding.
- AppImage enables `bundle.linux.appimage.bundleMediaFramework`. Per the
  [official Tauri guide](https://v2.tauri.app/distribute/appimage/#multimedia-support-via-gstreamer),
  its Ubuntu build host must have the required media plugins installed.
- CI and release packaging share one Linux-dependency action. It installs the
  runtime package list from the Debian metadata rather than maintaining a
  separate media-package policy, then verifies the GStreamer factories
  `autoaudiosink`, `qtdemux`, `h264parse`, `avdec_h264` and `avdec_aac`.
- Source installations document the corresponding Arch/EndeavourOS and Debian
  prerequisites. No automatic privileged installation occurs inside Studio.
- Do not suppress the crash by muting/removing audio or silently disabling video.
  Do not use a GPU-renderer workaround without evidence of a GPU failure.
- Config regressions cover required runtime dependencies, AppImage inclusion,
  and use of the shared action by both workflows. Native project reopening is a
  separate verification; factory presence alone does not prove the whole UI flow.

No project/CLI schema changes, new Python/Rust dependencies, resource measurements
or installer-verification claims are introduced by this correction.

- Creators can download and install Kliptych Studio like any standard desktop software.
- The desktop shell automatically detects the Python engine without manual terminal setup.
- Verification command:
  ```bash
  npm --prefix studio run check
  ```
  Expected outcome: 0 errors across TypeScript, Vite, cargo fmt, and cargo check.
