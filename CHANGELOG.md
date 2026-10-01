# Changelog

## [1.1.0] — 2026-09-30

### Added
- **Sprint 1 — Hardening:** real `subtitle_text` wiring, Whisper language
  handling, audio policy gate, fail-closed resume with `approved-by`
  attribution, CUDA hardening
- **Sprint 2 — Brand safety:** dynamic watermarks with OpenCV validator,
  forbidden phrases, LLM brand-safety evaluator (fail-closed to
  `manual_review`), hook validation (CB22), P1 performance pass
- **Sprint 3 — New formats:** `Format.LYRIC_VIDEO` (`.lrc`/lrclib, absolute
  offsets, ASS burn-in), `Layout.SPLIT_SCREEN` with gate geometry checks,
  T6 strict provenance, surgical timestamps, spelling locks

### Reliability
- Atomic batch export preserved across all new formats and layouts
- Test coverage kept above the 90% CI gate

## [1.0.0] — 2026-09-24

### Added
- **Fase C — Motor de video largo:** Download (yt-dlp/streamlink), Transcripción (Whisper small int8), Detección de momentos, Selección LLM, Reframe 9:16 (MediaPipe CPU), Subtítulos (.ass), Render NVENC/CPU
- **Fase D — Modos especiales:** Audio Locked, Repost/UGC, Slideshow
- **Fase E — Inteligencia de campañas:** Clasificación KNOWN/KNOWN_WITH_VARIATION/NEW_ARCHETYPE, Motor de propuestas PR (Zero Auto-Merge), CampaignManager
- **Fase F — Madurez operativa:** CLI estructurada, Logging centralizado, Checkpointing/Reanudación, Garbage Collector

### Security
- Zero shell=True en todo el proyecto
- Zero Auto-Merge: la IA propone, el humano dispone
- Aislamiento total de red en CI

### Reliability
- Escritura atómica de state checkpoints
- Validación de integridad de artefactos (st_size > 0 + schema)
- Test coverage 97%+
