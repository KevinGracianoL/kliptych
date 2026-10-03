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
- **Scoping por plataforma para las reglas de AUDIO (`#59`, `#57`).**
  `contract.rules` es global: una sola declaración cubre todas las plataformas.
  Los requisitos de audio, no. El predicado `rule_applies` del motor es la única
  tabla de aplicabilidad, y ahora cubre `audio.present`, `audio.own_clip`,
  `audio.no_trending` y `audio.official_track`. **Este PR NO toca el
  exportador**, ni ningún recordatorio: ver el punto de "#57" más abajo.

  - **LA INVARIANTE, que es lo que gobierna todo lo demás:** una regla
    DECLARADA que no es APLICABLE es un estado explícito y REGISTRADO, nunca
    silencio. Hay tres casos y solo tres:
    - **A:** la regla aplica a esta plataforma → se despacha normal.
    - **B:** no aplica aquí pero aplica a otra plataforma del contrato → se omite
      para esta pieza y no se registra nada. Es la regla de esa otra.
    - **C:** está declarada y no aplica a NINGUNA plataforma → el contrato es
      incoherente y se despacha **igual que antes del scoping**, con su
      resultado idéntico al de `main` y una evidencia extra que lo dice.
  - **Por qué el caso C no se "arregla" con un `UNSUPPORTED` fijo:** porque
    `_derive_status` solo escala `UNSUPPORTED` cuando la fuerza es `hard`. Una
    regla `recommended` huérfana con `UNSUPPORTED` fijo pasaría de `rejected` a
    `passed`, que es un fail-open. Se comprobó: esa implementación movía 2 de
    las 12 celdas hacia más exportable, y por eso se descartó.
  - **Medido, las 12 combinaciones de fuerza × status, caso C:** las 12 quedan
    **sin cambio** frente a `main`. Cero fail-open.
  - **Medido, caso B:** una plataforma con `audio_rule=any` en una campaña con
    `no_trending` pasa de `pending_review`/`blocked` a `passed`/`exported` sin
    firma. 5 de las 12 celdas cambian, todas hacia más exportabilidad, y todas
    sobre una plataforma que no declaró la regla. Ese es el cambio pedido.
  - **Qué NO cambia:** una plataforma que declara la regla sigue exigiendo su
    firma, y también con `own_clip` y `official_required`. Un `FAIL` sigue
    rechazando con y sin aprobación. Una regla dura sin validador sigue
    bloqueando como `unsupported`, también con firma.
  - **`audio.no_trending` sigue sin validador, a propósito.** Es `manual_review`:
    significa que decide una persona, y eso no se automatiza. **No se le añade
    un validador para "arreglarla".** El motor **sintetiza** el check a partir de
    la fuerza, así que `PENDING_REVIEW` lo fuerzan dos cosas a la vez: la fuerza
    y el status. Filtrar solo una no habría arreglado `#59`.
  - **Alias.** `RULE_ID_ALIASES` es ahora la única fuente de alias, y la usan el
    validador del schema y el predicado del motor. Sin canonizar, un alias
    (`audio.official_selection`, `audio.rule`) esquivaba el scoping.
  - **Un test de `#58` cambia de significado aquí, a propósito.**
    `test_audio_rule_is_scoped_per_platform` afirmaba que `audio.no_trending` se
    emitía para la plataforma con `audio_rule=any`: documentaba este defecto
    bajo un nombre de scoping. Ahora afirma que **no** se emite. La aserción
    nueva es más fuerte: fija la ausencia del outcome. El defecto era del `#58`.
  - **LO QUE NO SE ARREGLA, y no conviene venderlo como arreglado:**
    - Solo las reglas de **audio**. `attribution.required`,
      `attribution.present`, `link.in_bio` y `link_rules.link_in_bio` siguen
      declarándose globalmente y siguen frenando a plataformas que no las
      piden; medido: `pending_review` con fuerza `manual_review` y `unsupported`
      con fuerza `hard`. Es el mismo defecto que `#59`, y está en #63.
    - **`#57` sigue abierto y este PR no lo toca.** El exportador queda
    exactamente como estaba. Se intentó acotar sus recordatorios y se revirtió,
    por dos motivos medidos: acotar a las piezas exportadas deja sin avisos un
    paquete bloqueado y rompe `test_report_lists_post_publication_reminders`, y
    acotar a nivel de contrato solo puede suprimir el caso C, que es justo
    cuando el aviso es cierto. El caso B necesita saber qué piezas hay en el
    paquete, y `DeliveryReport.reminders` es una lista plana sin ese contexto.
    Hacerlo por pieza es un cambio de schema de `DeliveryReport`, diferido a la
    lista de piezas de Studio. No se afirma aquí ningún cambio de
    comportamiento de recordatorios.
- **Se quita una aserción de reloj de pared de la suite de correctitud
  (`#56`).** Tres cosas distintas, que conviene no mezclar:

  1. **La propiedad de correctitud del watermark NO cambia.** El test sigue
     afirmando `CheckStatus.PASS` sobre 20s de vídeo real con watermark
     (4.893.366 bytes, 540x960, 20.000000s). Esa aserción sigue siendo
     load-bearing: mutando el check para que devuelva `FAIL` sobre un vídeo que
     sí lleva watermark, el test falla. Lo que se elimina es solo la afirmación
     de tiempo, `elapsed < 15.0`, y su `time.monotonic()`.

  2. **Se quita el umbral de reloj, con sus números medidos.** El mismo commit
     tardó 15.42s en un runner de Windows y pasó en el run de 13 minutos antes.
     Medido aquí cinco veces, cronometrando solo el check sobre vídeos de 20s
     distintos: 4.99s, 3.94s, 3.91s, 3.93s, 3.95s → min 3.91s, max 4.99s, media
     4.15s, spread 1.28x. Headroom sobre el umbral de 15.0s: **3.62x** con la
     media, **3.00x** con la peor muestra. El umbral nunca fue cerrado; lo que
     pasó es que el runner de CI era ~3.9x más lento que la mejor muestra local
     y consumió todo el margen de golpe. Ese es el diagnóstico, y es distinto
     del "umbral ajustado". Ojo al comparar la cifra equivocada: el test aislado
     tarda ~9.6s de reloj pero solo cronometra ~4s, porque generar el vídeo de
     20s ocurre antes de arrancar el cronómetro.

  3. **El coste de `check_watermark_full_video` queda SIN MEDIR, no "va bien".**
     No se afirma ninguna mejora de rendimiento porque no se midió ninguna. Es
     una pérdida de cobertura declarada: hasta que exista `kliptych bench`, que
     todavía no existe (Sprint A3), nadie vigila ese coste. Loergyó un gate que
     fallaba por ruido, que era peor que no tenerlo.

  Nota para quien lea el test después: es una aserción **positiva** (vídeo con
  watermark → `PASS`), así que por construcción **no** detecta que el check
  sobre-pase; mutarlo para que devuelva `PASS` siempre lo deja en verde. Esa
  dirección la cubren los 10 tests negativos del mismo archivo
  (`test_textured_wrong_position_fails`, `test_verdict_without_signal_fails`, los
  fail-closed), que sí fallaron bajo esa mutación.

  Por qué se quita en vez de relajar: un gate no determinista entrena al equipo
  a re-ejecutar, y eso destruye la señal de todos los demás. Cuando CI falla, la
  primera hipótesis pasa a ser "el flaky" y no "mi cambio". Subir el umbral sin
  medir cambiaría una alarma ruidosa por una alarma muda, y el margen seguiría
  siendo del runner y no del código.
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
