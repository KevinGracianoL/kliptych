# Changelog

## [Unreleased]

### Added
- **Sprint A1 — Modo zero-contract:** `campaign --url <URL> --out <DIR>` procesa
  un vídeo sin `brief.md`. Sin brief no hay nada que extraer, así que no se
  llama al LLM y el contrato se sintetiza en memoria con `vanilla_contract`
  (`brand_safety_required=False`, sin prohibiciones léxicas,
  `audio_policy=original_audio`). El modo se fuerza a `repost_ugc`, el camino
  que omite la selección de segmentos con el LLM y solo necesita la URL, y no
  exige `KLIPTYCH_LLM_*`.
- `campaign_id` sin brief se deriva de `brief_key(url)[:12]`: es determinista,
  la misma URL produce siempre el mismo identificador. Es el sha256 de la URL
  truncado a 12 dígitos hexadecimales, 48 bits, así que dos URLs distintas
  pueden acabar en el mismo directorio de entrega con probabilidad baja, no con
  certeza nula.
- El contrato zero-contract declara solo `artifact.integrity` y
  `artifact.video_stream` en `rules.hard`. No declara reglas de audio: la
  plataforma usa `audio_rule=any`, así que no hay audio que exigir ni que
  verificar, y el resolver tampoco lo haría. No están ahí por decoración:
  `_derive_status` devuelve `passed` sobre cero checks, y sin ellas el gate
  aprobaría en vacío incluso con el artefacto ausente.

### Changed
- **Gate de audio verificable (D2):** `check_audio_present` solo emite `PASS`
  cuando el audio se ha comprobado de verdad, es decir con `audio_rule=own_clip`
  mediante `media.has_audio`. Ese camino no cambia: sigue siendo idéntico byte a
  byte. Antes, `audio_rule` de `no_trending` y de `official_required` devolvía
  `PASS` con evidencia `{"has_audio": true}`, una atestestación de audio
  verificado que nadie había comprobado; ahora devuelven `MANUAL_REVIEW` con
  evidencia que dice literalmente que no se verificó (`verifiable: false`) y qué
  validador falta. `any` devuelve `UNSUPPORTED` con el motivo escrito.
- **Consecuencia para el operador:** en campañas `no_trending` u
  `official_required`, el gate **ya no pasa solo**. La pieza para en
  `pending_review` y necesita firma (`--approve-manual-review --approved-by`).
  Antes salía sin revisión de nadie. Es el sentido del cambio: una regla dura que
  no se puede verificar no puede atestiguar que se verificó.
- **Scoping por plataforma:** los `rule_id` de `contract.rules` son globales,
  pero `audio.present` solo aplica a las plataformas que exigen audio. El motor
  (`_rule_applies`) omite el check cuando la plataforma de la pieza declara
  `audio_rule=any`, en vez de emitir un estado.
  - **ANTES de este cambio:** `audio.present` con `audio_rule=any` devolvía
    `PASS` con evidencia `{"audio_rule": "any"}`, una atestestación de que el
    audio estaba verificado sin haber comprobado nada.
  - **DESPUÉS, sin el scoping:** el check nuevo devuelve `UNSUPPORTED` ("aquí no
    hay nada que verificar"), lo que en una regla dura deriva en
    `GateStatus.UNSUPPORTED`: inexportable y sin salida humana, por una regla que
    no le aplicaba a esa plataforma.
  - **DESPUÉS, con el scoping:** para la plataforma `any` no se emite ningún
    outcome. Ese es el estado final.
- **Lo que ese scoping NO arregla, y por qué:** una plataforma `any` en una
  campaña donde otra declara `no_trending` **sigue parando en `pending_review`**
  y sigue necesitando firma. No es el mismo defecto un nivel más arriba, y es
  preexistente (idéntico en `main`):
  - `audio.no_trending` no está en `DEFAULT_VALIDATORS`, así que no ejecuta
    ningún check.
  - Lo que fuerza el `pending_review` es `_derive_status` leyendo
    `RuleStrength.MANUAL_REVIEW` del `contract.rules.manual_review`, que es
    **global**, mientras que las reglas son por plataforma.
  - El predicado `_rule_applies` de este cambio arregla el **despacho de
    checks**; no toca la **agregación de fuerzas**. Mismo defecto, un nivel más
    arriba, y por eso el `any` no queda libre.
  El resultado exportable de esa pieza es el mismo que antes de este cambio: lo
  que desaparece es la atestestación falsa en `gate.json`, no el freno. Corregir
  el freno es otro arreglo y no entra aquí. Como sobre-bloquea y no
  sub-bloquea, no es un agujero de cumplimiento.
- No cambia: los checks con requisitos vacíos siguen ejecutándose y siguen
  emitiendo su evidencia (por ejemplo `caption.required_mention` con
  `{"required": [], "missing": []}`). Un conjunto de requisitos vacío no es lo
  mismo que una regla no aplicable, y esa distinción está escrita en el motor
  para que nadie la generalice por error.
- No cambia: `own_clip`, `_derive_status`, el resolver y el exportador.
- `Campaign.brief` admite vacío; vacío es la señal del modo zero-contract.
  `campaign_id` sigue exigiendo valor porque nombra el directorio de entrega.
- `run_repost` acepta `model` ausente: el modo repost no consulta el modelo.
- `build_ytdlp_argv` y el clamp de `_even` **no** se listan aquí: entraron en
  `main` con #52 y no forman parte de este trabajo.

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
