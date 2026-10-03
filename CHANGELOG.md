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
- **Las reglas declaradas globalmente solo afectan a las plataformas que las
  declararon (`#59`, `#57`).** `contract.rules` es global: una sola declaración
  cubre todas las plataformas del contrato. Lo que describe es por plataforma,
  y dos consumidores lo leían como si fuera global.
  - **Qué estaba bloqueado y ahora no:** una plataforma con `audio_rule=any` en
    una campaña donde otra declara `no_trending` exigía una firma que no había
    pedido. Medido: esa pieza salía `blocked` sin firma y ahora sale `exported`
    sin firma, con gate `passed` en vez de `pending_review`.
  - **Qué NO cambia:** cualquier plataforma que declare la regla sigue
    exigiendo su firma. Un `FAIL` sigue rechazando, con y sin aprobación. Una
    regla dura sin validador sigue bloqueando como `unsupported`, también con
    firma. Verificado con mutaciones: invertir los predicados mata el test de
    no-regresión, así que el scope no puede degradarse en sobre-aplicación.
  - **Una sola tabla de aplicabilidad.** El predicado `rule_applies` del motor
    es la única fuente y ahora cubre también `audio.own_clip`,
    `audio.no_trending` y `audio.official_track`. El exportador consulta ese
    MISMO predicado para sus recordatorios; no hay un segundo mecanismo.
  - **`audio.no_trending` sigue sin validador a propósito.** Es una regla
    `manual_review`: `manual_review` significa que decide una persona, y eso no
    se automatiza. No emite evidencia mecánica y **no se le añade un validador**
    para "arreglarla". Lo que se corrige es que no pueda exigir revisión a una
    plataforma que no declaró esa regla de audio.
  - **El motor SINTETIZA un check `manual_review` para una regla
    `manual_review` sin validador registrado**, en vez de no ejecutar nada. Por
    eso `PENDING_REVIEW` lo fuerzan **dos** cosas a la vez: la fuerza de la
    regla y el status del check. Filtrar solo una de las dos no habría arreglado
    `#59`. Queda escrito para el siguiente.
  - **Un test de `#58` cambia de significado aquí, a propósito.**
    `test_audio_rule_is_scoped_per_platform` afirmaba que `audio.no_trending`
    se emitía para la plataforma con `audio_rule=any`: es decir, documentaba
    este defecto bajo un nombre de scoping. Ahora afirma que **no** se emite y
    que el gate queda en `passed`. Es un cambio deliberado, no deriva: la
    aserción nueva es más fuerte, porque fija la ausencia del outcome. El
    defecto era del `#58`, donde se aprobó ese test sin preguntar si las reglas
    `manual_review` globales también tenían aplicabilidad por plataforma.
  - **Lo que NO se arregla:** `reminders` es plano por informe. Los
    recordatorios `manual_review` son por plataforma en intención pero planos en
    forma, así que un contrato mixto sigue mostrando el aviso. Hacerlos por
    pieza es un cambio de schema de `DeliveryReport`, y queda diferido a la lista
    de piezas de Studio, que es el primer consumidor real. Es ruido consultivo,
    no un agujero de cumplimiento: el gate ya es correcto. `#57` sigue abierto.
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
