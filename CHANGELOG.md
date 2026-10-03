# Changelog

## [Unreleased]

### Added
- **Sprint A1 — Modo zero-contract:** `campaign --url <URL> --out <DIR>` procesa
  un vídeo sin `brief.md`. Sin brief no hay nada que extraer, así que no se
  llama al LLM y el contrato se sintetiza en memoria con `vanilla_contract`
  (`brand_safety_required=False`, sin prohibiciones léxicas,
  `audio_policy=original_audio`). El modo se fuerza a `repost_ugc`, el único
  camino del orquestador que no selecciona segmentos con el LLM, y no exige
  `KLIPTYCH_LLM_*`.
- `campaign_id` sin brief se deriva de `brief_key(url)[:12]`: determinista y
  sin colisiones entre URLs distintas en el directorio de entrega.
- El contrato zero-contract declara solo `artifact.integrity` y
  `artifact.video_stream` en `rules.hard`. No declara reglas de audio: la
  plataforma usa `audio_rule=any`, así que no hay audio que exigir ni que
  verificar, y el resolver tampoco lo haría. No están ahí por decoración:
  `_derive_status` devuelve `passed` sobre cero checks, y sin ellas el gate
  aprobaría en vacío incluso con el artefacto ausente.

### Changed
- `Campaign.brief` admite vacío; vacío es la señal del modo zero-contract.
  `campaign_id` sigue exigiendo valor porque nombra el directorio de entrega.
- `build_ytdlp_argv` usa el selector con techo anti-4K por defecto en lugar de
  `None`, así que las llamadas externas heredan el techo.
- `run_repost` acepta `model` ausente: el modo repost no consulta el modelo.

### Fixed
- `_even` ya no puede devolver un offset negativo al redondear a la baja.

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
