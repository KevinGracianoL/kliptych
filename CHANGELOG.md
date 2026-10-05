# Changelog

## [Unreleased]

### Changed
- **El gate compara el aspecto medido contra el DECLARADO con la tolerancia del
  render de ESA corrida.** Antes lo comparaba contra un 9:16 fijo en modulo.
  Gate y render llaman ahora a la MISMA funcion, `encoding.aspect_matches`, que
  es la predicada que decide si un video hay que reencuadrar. No se copia el
  numero: una tolerancia en dos sitios es el mismo defecto que la constante 9:16
  que esta PR quita, y divergiria en silencio la primera vez que cambiase la
  cuantizacion.
  La tolerancia se lee de `context.render`, la config EFECTIVA de la corrida, y
  llega a `GateContext` por el campo `render`. Antes se construia un
  `RenderConfig()` nuevo en el validador, lo cual devolvia el DEFAULT de la clase:
  una corrida que renderizo con `aspect_tolerance=0.001` era juzgada con 0.05.
  Medir contra un default en vez de contra una declaracion es la misma forma que
  el `outputs` por defecto de un 1.1.
- **Por que una tolerancia y no una comparacion exacta.** El reframe cuantiza a
  entero par y `round` es half-to-even, asi que un alto de 1080 da
  `round(607.5) = 608`. Medido con `crop_size` del propio repo:
  1920x1080 -> 608x1080 (0,08 % de 9:16), 1280x720 -> 404x720 (0,25 %),
  1080x1900 -> 1068x1900 (0,07 %). **No siempre** es exacto, pero a veces si:
  720x1280, 1080x1920 y 1152x2048 caen exactos, porque 9:16 tiene
  representacion entera par. "El render nunca puede emitir 9:16" seria falso; lo
  cierto es que la cuantizacion no lo GARANTIZA, y por eso se usa la tolerancia
  que el render ya declaraba.
  Con un 1e-9 el gate rechazaba la salida del propio render. Eso es lo que hacia
  una version anterior de esta PR, y lo encontro una revision adversarial
  ejecutando el crop de verdad, no leyendo el codigo.
- **Donde si termina: la banda sigue valiendo hasta ~24 px de lado menor.** La
  cuantizacion al par se come el margen en origenes pequenos y el recorte sale de
  la banda: 16x16 -> 8x16, a un 11 % de 9:16, que es un video de 2 px de ancho.
  Ahi el rechazo del gate es CORRECTO y el defecto es del PRODUCTOR, que deberia
  negarse a construir ese recorte. Medido con `crop_size` y la predicada real: el
  lado menor mas grande entre recortes rechazados es 20 px (`20x38`), y sobre una
  rejilla de origenes 2..1200 x 2..1200 no hay ni un rechazo con ambos lados
  >= 24. Con ambos lados >= 40 tampoco, y a 22x40 el error es 2,2 %, o sea que la
  tolerancia NO hay que tocarla. En #84, con el detalle. No se arregla aqui:
  cambiar el productor es otro PR con su propio recorrido end-to-end.
- **Es una relajacion frente a comparar a 1e-9, y se declara como tal.** No es una
  perdida de rigor: es la retirada de un desacuerdo que la comprobacion anterior
  no podia detectar. Un 16:9 genuino esta a un 216 % de desviacion relativa y
  sigue fallando.
- **El gate valida el aspecto que el CONTRATO DECLARA**, en vez de comparar
  contra un 9:16 fijo en modulo. `_SPLIT_ASPECT_W`/`_SPLIT_ASPECT_H` han
  desaparecido de `gate/checks.py`; el aspecto sale de `SplitScreenConfig.aspect`
  y la aritmetica vive en un solo sitio, el enum `Aspect`. Antes, un contrato que
  declarase 16:9 no podia pasar ni con un artefacto 16:9 perfecto, y el mensaje
  de FAIL le atribuia al contrato algo que el contrato habia dicho bien.
- **Un artefacto cuya geometria no se puede medir devuelve ahora UNSUPPORTED en
  vez de FAIL** en el validador de split-screen. Es un cambio de comportamiento
  visible, asi que conviene decir exactamente que significa y que no significa:
  `FAIL` es "el artefacto esta mal" y `UNSUPPORTED` es "no pude verificarlo". Un
  archivo que ffprobe no pudo abrir es un problema de herramienta o de entorno, no
  de contenido, asi que "rejected" mandaba al operador a depurar lo que no estaba
  roto.
  El caso "ffprobe abrio el archivo y dijo que no hay pista de video" sigue siendo
  FAIL, porque ahi HUBO medicion. Es lo que ya hacen `artifact.video_stream` y
  `audio.present`, y devolver UNSUPPORTED en los dos casos dejaba el mismo
  `gate.json` con dos diagnosticos contradictorios para el mismo artefacto.
- **UNSUPPORTED ahora bloquea con CUALQUIER fuerza de regla, no solo con `hard`.**
  Este es un fail-open que ya existia en `main`, anterior a esta PR, y que solo
  ahora es alcanzable desde un validador de geometria. `_derive_status` escalaba
  `UNSUPPORTED` solo cuando la fuerza era `hard`, asi que un `UNSUPPORTED` en
  `recommended` o `manual_review` caia al final y devolvia `passed`: un artefacto
  que NADIE pudo verificar salia aprobado y exportable, incluso con firma de
  revision manual.
  El aviso estaba escrito en un comentario del propio motor, dos funciones mas
  arriba, y un test lo FIJABA como comportamiento correcto
  (`test_recommended_rule_with_unsupported_check_is_unaffected`, de #57, que
  afirmaba `PASSED` y `EXPORTED` para justo este caso). No era invisible: estaba
  aprobado. El fail-open se encontro cuando un revisor ejecuto mutaciones sobre
  `_derive_status`, no porque existiera un test que lo detectara.
  El brief (§5.1) dice "Si no hay validador para una regla **dura**, el resultado
  es UNSUPPORTED o MANUAL_REVIEW, jamás PASS": la cláusula "jamás PASS" está
  acotada a las reglas duras, así que esto NO es aplicar el brief, es un
  endurecimiento deliberado más allá de §5.1.
  El motivo no es de norma sino de salida: `PENDING_REVIEW` es una salida del
  gate, y la firma que habilita es una atestación de una persona. Una persona no
  puede atestiguar una geometría que no se midió, así que "no verificable por
  nadie, ni por máquina ni por humano" no puede ser un estado firmable.
  Lo que no cambia: `FAIL` sigue gainandole, y el orden de escalado es el mismo
  (FAIL -> UNSUPPORTED -> revision manual -> passed).
- La escalera de #76 sigue en orden: **PR 1 declaro, PR 2 valida, PR 3
  renderiza**. Solo PR 3 produce un segundo artefacto. Esta PR no entrega ninguna
  capacidad visible.

### Added
- **Regla `artifact.aspect`**: el artefacto tiene que satisfacer AL MENOS una de
  las salidas que el contrato declara. Vive en su propio validador, y no dentro
  del de split-screen, porque un contrato multi-formato no puede declarar lienzo
  (limitacion deliberada de PR 1) y anidada ahi la comprobacion no existiria
  justo en el caso para el que existe.
  Se mide tambien en un 1.1, contra su default vertical documentado. Aqui hay
  que ser exacto sobre que hacia `main`, porque la primera version de esta
  entrada exagero: `main` rechazaba un 1.1 con artefacto 16:9 **solo cuando el
  contrato declaraba `split_screen`**, a traves de `layout.geometry`. Un
  contrato resuelto SIN lienzo, que es el caso normal, no tenia ninguna regla que
  midiera el aspecto: medido, `main` le daba `passed` a un 16:9. Ese es
  exactamente el perfil de D1, una regla que falta donde deberia estar.
  Lo que cambia aqui es que el motivo es el correcto (que el contrato dijo 9:16
  y el artefacto es 16:9, no "las dimensiones no coinciden con un lienzo que
  nadie declaro") y que el caso sin lienzo esta cubierto. La exencion por
  version que existio en un borrador intermedio queda fuera de este CHANGELOG a
  proposito: nunca estuvo en `main`, y atribuirle un defecto a un commit que no
  lo contiene manda al siguiente a auditar el codigo equivocado.
  Un contrato que NO declara la regla sigue sin ejecutarla; los dos unicos
  productores de `RuleSet` (`resolver._base_rules` y `vanilla_contract`) la
  declaran, asi que toda corrida de produccion la tiene.
  `declared_by_version` queda en la evidencia como campo de PROCEDENCIA: dice que
  el requisito vino de un default de version y no de una declaracion.
- `artifact.aspect` se fuerza a `hard` en `resolver._base_rules`, con el mismo
  issue code fail-closed que el resto de reglas por defecto. No es necesario para
  la seguridad ahora que UNSUPPORTED bloquea con cualquier fuerza; es para que una
  regla de geometria no sea segura solo por casualidad del motor.
- **La regla zero-contract declara ahora `artifact.aspect`.** `vanilla_contract`
  emite `outputs` = 9:16 por default, o sea que SI declara la geometria, pero no
  declaraba la regla que la verifica: un artefacto 16:9 en el modo zero-contract
  pasaba con gate `passed`. Es la forma exacta de D1, una regla que falta donde
  deberia estar y el gate leyendo la ausencia como "no hay nada que comprobar".
- Una regla declarada que no se puede verificar mecanicamente es UNSUPPORTED,
  nunca PASS. **La clase NO queda cerrada con este PR:** `watermark.present`,
  `watermark.full_video` y `audio.silence` siguen conflacionando "el probe no
  pudo responder" con "el probe respondio que no". Se mide en #83 con su entrada
  exacta y no se arregla aqui, porque cambiarlo pasaria REJECTED a UNSUPPORTED en
  toda campana con watermark, que es un cambio visible en validadores ajenos a
  esta PR.
- **Sprint A2.2 - Segmentador local determinista:** el modo personal (zero-contract)
  emitia un unico segmento con el video completo. Ahora transcribe y trocea las
  palabras en varios segmentos, elegidos por palabras por segundo y silencios.
Medido sobre un VOD real de 70 min (una clase), en un excerpt de 10 min. Con
  el umbral de silencio en 0,6 s: 87 tramos crudos, 13 descartados y **74
  segmentos** de mediana 5,0 s, de los cuales 45 duraban menos de 6 s. Ese era el
  problema: no faltaba troceado, sobraba.
  Con el umbral en 2,0 s y el techo de 12 momentos: **12 segmentos**, min 8,6 s,
  mediana 23,3 s, max 59,8 s, y ninguno por debajo de 8 s. El umbral sale del
  histograma de huecos de ese mismo excerpt y de un barrido de umbrales; no es un
  ajuste de gusto. La seleccion se midio aparte: 0,37 ms de media, un
  0,0007 % del tiempo de transcripcion.
- La seleccion es mecanica y determinista: no hay LLM, ni API, ni red. La misma
  transcripcion produce siempre los mismos segmentos, y la lista sale en orden
  cronologico; el score viaja en cada segmento para quien quiera ordenar por
  interes.
- El modo personal ahora transcribe y, por tanto, ahora produce subtitulos karaoke
  que antes no producia.
- La duracion minima la impone la seleccion, NO el gate: el contrato zero-contract
  no declara regla de duracion y el gate no la verifica (#67).
- El limite de duracion lo fija el contrato cuando lo declara, y el segmentador
  usa los suyos solo cuando el contrato calla.
- `MAX_GAP_S` se compara contra una resta de flotantes, asi que un hueco de "0,6 s"
  es aproximado en el orden de 1e-16. La salida sigue siendo determinista: lo
  unico difuso es el significado del numero en una medida de cero. No se anadio
  redondeo para taparlo.
- El operador recibe el recuento de lo que se descarto y por que: cuantos tramos
  calificaron, cuantos se entregaron, cuantos se dejaron por el techo, cuantos
  por los suelos y cuantos clips se recortaron por duracion. Sin eso, "12 tramo(s)"
  es una operacion PARCIAL informada como completa.
- La exportacion en lote sigue siendo atomica, asi que un clip que falle deja el
  paquete entero sin entregar. El render perezoso por clip (B1) elimina este
  escalon y es la resolucion prevista.


### Changed
- **Sprint A2.1 - Karaoke por linea:** el subtitulo de karaoke escribia un
  evento `Dialogue` por palabra, de modo que en pantalla se veia una sola palabra
  a la vez. Ahora agrupa las palabras en una linea, emite un unico evento con un
  `{\k}` por palabra, y el resalte barre esa linea.
- Una palabra a la que se le asignan `0` centisegundos **no se dibuja**: medido
  con libass, un `{\k0}` no renderiza nada, sin error ni aviso. De ahi que el
  suelo sea de `1` centisegundo, y de ahi que el reparto mueva centisegundos
  entre palabras en lugar de anadirlos.
- Una palabra de menos de 10 ms recibe un resalte perceptualmente instantaneo
  pero tecnicamente valido, porque la alternativa es que no aparezca nunca.
- El `.ass` emitido es coherente consigo mismo en centisegundos: la suma de los
  `{\k}` es igual a `End - Start`. Las marcas ASS son de centisegundo, así que
  contra un extremo redondeado por separado puede diferir en `1` centisegundo.
  Eso es el suelo del formato, no un defecto a corregir.
- El reparto es por resto mayor sobre ranuras (de inicio a siguiente inicio, y la
  ultima hasta el fin). Con redondeo independiente, una linea de 8 palabras se
  desviaba 3 centisegundos, y la desviacion crece con el numero de palabras.

Evidencia de que el barrido progresa y no es una sola transicion: en un unico
evento `Dialogue`, la parte ya cantada es siempre un prefijo cuyo borde derecho
avanza, y su centroide x crece de forma monotona y es distinto en cada estado
medido (213,2 -> 326,5 -> 355,7 en una linea de 720x1280). Con solo contar
pixeles esa tabla no se distingue de "la linea entera se encendio a la mitad".

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
  (`rule_applies`) omite el check cuando la plataforma de la pieza declara
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
  - El predicado `rule_applies` de este cambio arregla el **despacho de
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
- No cambia en el arreglo anterior: `_derive_status`, el resolver y el
  exportador.
- **Scoping por plataforma para las reglas de audio (`#59`).**
  `contract.rules` es global: una sola declaración cubre todas las plataformas.
  Los requisitos de audio, no. Una plataforma con `audio_rule=any` quedaba
  frenada por la regla de audio de otra plataforma del mismo contrato.

  **Invariante:** una regla declarada que no es aplicable es un estado
  explícito y registrado, nunca silencio. El caso en que la regla no la declara
  ninguna plataforma tiene dos formas, y no se comportan igual:

  - Una regla **sin validador registrado** se comporta como en `main`: mismo
    estado, mismo gate y misma exportación. La evidencia gana una clave que
    dice que ninguna plataforma la declara.
  - Una regla **con validador registrado** es más estricta que `main`, porque
    `main` la omitía en silencio y exportaba la pieza.

  Las reglas de audio sin validador son `audio.own_clip`, `audio.no_trending` y
  `audio.official_track`; la única con validador es `audio.present`. `RULE_ID_ALIASES`
  es la lista de alias del contrato, y el resolver deriva de ella.

  `test_audio_rule_is_scoped_per_platform` (del PR anterior) afirmaba que
  `audio.no_trending` se emitía para la plataforma con `audio_rule=any`, es
  decir documentaba este defecto bajo un nombre de scoping. Ahora afirma que no
  se emite.

  **Diferido, y nada de esto se arregla aquí:**

  - Recordatorios de revisión manual por plataforma: `reminders` es plano por
    informe. El exportador no cambia. #57.
  - `attribution.required`, `attribution.present`, `link.in_bio` y
    `link_rules.link_in_bio` también son por plataforma en intención y siguen
    frenando a quien no las pide. #63.
  - Un banco de pruebas que enumere la matriz de aplicabilidad. PR aparte.

  La evidencia de todo esto está en la discusión del PR, no aquí.
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