====================================================================
KLIPTYCH — Brief de proyecto v2 (para agente de código)
====================================================================
Autor/owner: Kevin Graciano
Repo: PRIVADO, nombre "kliptych".
Dominio: el proyecto es específicamente para Content Rewards. Al ser repo
privado, el dominio SÍ puede mencionarse en código, docs, commits e issues.
Ya NO aplica la regla de ocultar palabras como "clip", "shorts", "rewards",
"viral", etc. (esa regla existía solo si el repo era público).

Única regla dura de datos que se conserva:
- No subir tokens, credenciales, cookies ni datos personales innecesarios.
- No subir material de campaña que no deba conservarse.
- Fixtures de CI sintéticos o anonimizados.
- campaigns/private/ y runs/ NO se commitean (ver sección 13).

Compatibilidad requerida: el repo se desarrolla y orquesta con opencode,
Claude Code y Codex de forma intercambiable. Esto es HERRAMIENTA DE DESARROLLO,
capa distinta al LLM de runtime del pipeline (ver sección 9).

GUI: OPCIONAL y al final. Todo el core corre headless (CLI/API).

--------------------------------------------------------------------
0. CAMBIOS DE ESTA VERSIÓN (v2) RESPECTO A v1
--------------------------------------------------------------------
1) Descripción corregida (sección 1): ya NO promete crear canales ni subir
   contenido. La publicación es 100% manual del owner.
2) Se elimina la contradicción de naming (repo privado).
3) El GATE de compliance se adelanta: núcleo desde la fase A, un validador
   por capacidad, y el gate final se ejecuta sobre el ARTEFACTO terminado
   (no se deja para el final del proyecto).
4) El extractor debe devolver EVIDENCIA por campo; sin evidencia textual no
   se rellena con suposiciones.
5) El contrato se versiona (schema_version) y las reglas se vuelven
   específicas por plataforma.
6) Primer objetivo: walking skeleton end-to-end (modo given_clips), y desde
   ahí la plataforma completa por fases A-F. No se implementa todo de una vez.
7) Los tests NUNCA llaman a modelos reales: respuestas grabadas, fixtures
   golden y contract tests.
8) Reproducibilidad (run_manifest.json), seguridad de red/shell y política
   explícita de VRAM para la laptop.
9) La revisión de cada PR la hace el mismo agente que programa, pero como
   TAREA SEPARADA, con contexto limpio, mediante la skill
   "evidence-driven-review" (revisión multifocal con validación adversarial;
   secciones 12 y 15).

--------------------------------------------------------------------
1. QUÉ ES KLIPTYCH
--------------------------------------------------------------------
Kliptych transforma briefs y materiales de referencia en paquetes de entrega
listos para publicación. Genera las piezas, captions, hashtags, tags,
instrucciones de audio y reportes de compliance; la publicación permanece
bajo control manual del owner.

Un LLM ("capa 0") lee el brief caótico y lo normaliza a un CONTRATO en JSON
versionado, con evidencia por campo. Todo el pipeline obedece ese contrato
como conjunto de restricciones duras. Nada sale sin pasar el GATE de
compliance (fail-closed).

--------------------------------------------------------------------
2. ALCANCE Y NO-ALCANCE
--------------------------------------------------------------------
ENTRA:
- Ingesta de briefs y materiales de referencia.
- Extracción del contrato con evidencia por campo.
- Clasificación KNOWN / KNOWN_WITH_VARIATION / NEW_ARCHETYPE.
- Los 5 modos de campaña (sección 3), implementados por fases.
- Generación de piezas: videos verticales, slideshows, captions, hashtags,
  menciones, instrucciones de audio y metadata.
- Gate de compliance por pieza, sobre el artefacto final.
- Paquete de entrega organizado por campaña/plataforma, con metadata al lado
  (caption final, hashtags, tags, audio a usar) + reporte del gate.
- CLI/API headless; GUI opcional al final.

NO ENTRA (y no debe prometerse en ningún doc/README):
- Crear cuentas/canales (las redes no lo permiten por API).
- Publicar por API (TikTok/Instagram exigen permisos y pagos; además el owner
  sube a mano para control total).
- Verificar condiciones post-publicación (views para payout, geo real,
  analytics): quedan como recordatorios del exportador, nunca como checks
  del gate (ver sección 5).
- Garantizar que el owner seleccione correctamente el audio nativo durante la
  subida manual; el gate solo verifica el archivo de audio entregado.

--------------------------------------------------------------------
3. LOS 5 ARQUETIPOS DE CAMPAÑA (derivados de ejemplos reales)
--------------------------------------------------------------------
El diseño NO es un pipeline lineal. Son modos que se activan según el contrato:

A) CLIP-DE-MATERIAL-DADO: te entregan clips ya cortados; solo ensamblas
   (duración mínima, audio del propio clip, hashtags/tags fijos).
B) CLIP-DE-VIDEO-LARGO: te dan una lista de videos largos (p.ej. YouTube) y
   hay que ENCONTRAR los momentos y cortarlos. Reglas de contenido estrictas
   (menciones obligatorias, spelling exacto de la marca en subtítulos, mostrar
   el producto completo, prohibiciones específicas).
C) AUDIO-OBLIGATORIO: audio oficial forzado + audible, footage temático,
   geo-targeting (p.ej. 60% de una región), texto en idioma nativo natural.
D) REPOST/UGC: repost desde un banco aprobado con watermark visible TODO el
   video + anti-duplicado obligatorio (limpiar metadata, reframe, caption
   propio).
E) SLIDESHOW (no es video): slides + screenshots de chat, CTA/teléfono en la
   primera línea del caption, link en bio. Sub-pipeline aparte del motor de
   video.

Modo streams: NO es un arquetipo aparte; es una variante del modo B para
streams (densidad de chat como detector de momentos). Va integrado en el
motor de video largo (fase C).

--------------------------------------------------------------------
4. EL CONTRATO (schema JSON) — corazón del proyecto
--------------------------------------------------------------------
La capa 0 normaliza cualquier brief a este spec (validar con pydantic).
El contrato se versiona y las reglas se organizan POR PLATAFORMA.

4.1 Dos artefactos distintos:
- ContractDraft: salida cruda del LLM, con evidencia por campo. NO se usa
  para renderizar.
- Contract: draft validado y normalizado, con evidencia resuelta. Es el que
  consumen pipeline y gate.

4.2 Evidencia por campo (obligatoria en el draft):
{
  "duration": {
    "min_s": 8,
    "evidence": {
      "quote": "El video debe durar como mínimo ocho segundos",
      "start": 421,
      "end": 472,
      "location": "brief.md#l12"
    },
    "confidence": "explicit"
  }
}
Cada campo extraído lleva:
- value
- source_quote / source_location
- confidence: "explicit" | "inferred" | "missing" | "conflict"
- status: "explicit" | "inferred" | "missing" | "conflict"

REGLA DURA: si un campo obligatorio no tiene evidencia textual, NO se rellena
con una suposición. Queda null/UNKNOWN y produce MANUAL_REVIEW o
NEW_ARCHETYPE. Un contrato contaminado con alucinaciones haría que el gate
valide una campaña inventada por el modelo.

4.3 Esquema Contract (v1.1):
{
  "schema_version": "1.1",
  "campaign_id": str,
  "format": "video" | "slideshow",
  "mode": "given_clips" | "long_video" | "audio_locked" | "repost_ugc" | "slideshow",
  "platforms": {
    "tiktok": {
      "duration": {"min_s": int|null, "max_s": int|null},
      "caption_rules": {"must_mention": [str], "first_line": str|null, "forbidden": [str]},
      "audio_rule": "own_clip" | "official_required" | "no_trending" | "any",
      "required_hashtags": [str],
      "required_mentions": [str],
      "attribution": {"type": "tag"|"url"|"none", "value": str|null},
      "link_rules": {"link_in_bio": bool}
    },
    "instagram_reels": { ... },
    "youtube_shorts": { ... },
    "x": { ... }
  },
  "languages": {
    "source": str,
    "subtitles": str|null,
    "caption": str,
    "voice": str|null
  },
  "official_audio": {"tiktok_url": str|null, "instagram_url": str|null} | null,
  "watermark": {"required": bool, "asset_id": str|null, "visible_full_video": bool},
  "spelling_locks": [str],
  "prohibitions": [str],
  "rules": {
    "hard": ["caption.required_mention", "duration.min", "watermark.full_video"],
    "recommended": ["caption.tone"],
    "manual_review": ["audio.official_selection"]
  },
  "rule_strength": "cada regla debe estar clasificada como hard, recommended o manual_review",
  "assets": {
    "required": [{
      "asset_id": str, "kind": str, "uri": str,
      "sha256": str, "size_bytes": int, "mime": str,
      "origin": str, "license": str|null, "resolved_at": str
    }],
    "optional": [ ... misma forma ... ]
  },
  "segments": [{"start_s": float, "end_s": float}],
  "geo_target": {"country": str, "min_pct": int, "enforcement": "post_publication_manual"} | null,
  "min_views_for_payout": {"value": int|null, "enforcement": "post_publication_manual"},
  "analytics_proof_required": {"value": bool, "enforcement": "post_publication_manual"}
}

Notas de diseño:
- `segments` es obligatorio (al menos un segmento) cuando `mode="long_video"` y
  está prohibido en cualquier otro modo: cada segmento exige `start_s >= 0` y
  `end_s > start_s`, ambos finitos.
- Un asset sin sha256 permite que el mismo asset_id apunte a otro archivo
  después. El hash es obligatorio.
- Los campos con enforcement "post_publication_manual" (geo_target,
  min_views_for_payout, analytics_proof_required) son metadatos de
  recordatorio: el exportador los lista para el owner, el gate NO los evalúa.
- Si hay conflicto entre plataformas, se resuelve por plataforma; si el
  conflicto es interno (dos reglas duras incompatibles), el contrato se marca
  en conflicto y va a MANUAL_REVIEW.
- Nota de implementación (schema v1.1): `required_hashtags`, `required_mentions`
  y `caption_rules.must_mention` exigen el prefijo `#`/`@` respectivamente; el
  gate compara con frontera de token e ignora mayúsculas. Cada restricción
  declarada en la plataforma debe estar clasificada en `rules` (hard,
  recommended o manual_review); si no, el contrato no valida. Un watermark
  exigido se clasifica como `watermark.full_video` (si debe cubrir todo el
  video) o `watermark.present` (cobertura parcial); ninguno tiene validador
  mecánico aún (fase B), así que el gate los marca `UNSUPPORTED` hasta
  entonces. `audio_rule`, `attribution` y `link_rules.link_in_bio` todavía no
  tienen rule_id en el catálogo del gate (fases C/D) y no exigen clasificación
  aún.

Los 5 ejemplos ya analizados sirven como TEST FIXTURES del extractor.
Como el repo es privado, también se permiten fixtures reales anonimizados
(ver sección 13):
- Slideshow / chat-style, TikTok, teléfono en 1ª línea, link en bio.
- Clip de material dado, min 8s, audio propio (sin trending), tag fijo,
  prueba de analytics tras umbral de views.
- Clip de videos largos de YouTube: menciones obligatorias, spelling exacto
  en subs, mostrar producto completo, varias prohibiciones.
- Audio oficial obligatorio + audible, 60% audiencia regional, español natural.
- Repost UGC, IG Reels only, watermark visible todo el video, anti-duplicado.

--------------------------------------------------------------------
5. GATE DE COMPLIANCE (lo que más dinero salva)
--------------------------------------------------------------------
Filosofía: fail-closed. Una pieza no se exporta si no pasa todos sus checks.
El gate se ejecuta SOBRE EL ARTEFACTO FINAL (no sobre los parámetros con los
que se intentó generarlo). Overlays, reframe, zoom y re-encodes pueden cortar
el watermark, ocultar el producto o recortar subtítulos: por eso se valida lo
que realmente sale.

5.1 Estructura del resultado (nunca solo true/false):
{
  "status": "rejected",
  "checks": [
    {
      "id": "caption.required_mention",
      "status": "fail",
      "evidence": {"required": "@marca", "found": false}
    }
  ],
  "artifact_sha256": "...",
  "contract_sha256": "..."
}

Estados permitidos: PASS | FAIL | MANUAL_REVIEW | UNSUPPORTED.
REGLA DURA: nunca puede existir un "pass" porque no había ningún validador
aplicable. Si no hay validador para una regla dura, el resultado es
UNSUPPORTED o MANUAL_REVIEW, jamás PASS.

5.2 Qué se valida con código (determinista) y qué no:
Determinista (obligatorio implementarlo como código, no como LLM):
- hashtags, menciones, palabras prohibidas
- spelling de spelling_locks en subtítulos
- primera línea del caption / CTA / teléfono / link en bio
- duración, resolución, formato, existencia de pista de audio
- timestamps y duración de segmentos dentro de los límites del video
- audio audible (ffmpeg: loudness con ebur128 / silencedetect)
- archivo final íntegro y reproducible (ffprobe + hashes)

Requiere evidencia mecánica específica (definir mecanismo o MANUAL_REVIEW):
- watermark durante todo el video (p.ej. muestreo de frames + template
  matching/hash perceptual; si no se implementa -> MANUAL_REVIEW)
- audio oficial correcto (p.ej. fingerprint tipo chromaprint contra el asset
  oficial; si no se implementa -> MANUAL_REVIEW)
- producto visible completo y presencia de persona/elemento visual
  (modelo de visión o MANUAL_REVIEW explícito)
REGLA: un LLM no puede aprobar una regla dura. Un gate que permite aprobar
reglas sin evidencia mecánica es un gate decorativo.

5.3 Validadores por modo:
- given_clips: duración, audio propio, hashtags/tags fijos, watermark.
- long_video: todo lo anterior + segmento dentro de límites del video fuente
  + menciones en audio/subtítulos + producto completo.
- audio_locked: audio oficial usado y audible + geo (recordatorio manual).
- repost_ugc: watermark todo el video + anti-duplicado aplicado + metadata
  limpia + caption propio.
- slideshow: tipo de pieza, primera línea, link en bio, orden de slides.

5.4 Gate core temprano (NO esperar al final del proyecto):
Desde el núcleo (fase A) existe un gate que ya valida: duración, assets
obligatorios, hashtags, menciones, términos prohibidos, primera línea,
idioma, formato, rangos de timestamps, presencia de subtítulos e integridad
del archivo. Cada PR que produzca o transforme una pieza agrega su validador.
El "gate final" es solo la composición de todos los validadores.

5.5 Prueba de que el gate sirve:
El README debe mostrar dos resultados reales:
Caso aceptado:
  PASS — 3 piezas exportadas — 12/12 checks pasaron
Caso rechazado:
  REJECTED — caption.required_mention — falta la mención obligatoria @marca
  — la pieza no fue exportada
Y el CI debe tener fixtures intencionalmente rotos: caption sin la marca,
hashtag faltante, duración fuera de rango, spelling incorrecto, watermark
ausente, audio incorrecto y segmento fuera de los límites del video.
La prueba importante no es que el gate pase una pieza correcta: es que
RECHAZE una pieza incorrecta.

--------------------------------------------------------------------
6. EXTRACTOR DE CONTRATO (capa 0)
--------------------------------------------------------------------
- El LLM convierte el brief en ContractDraft (sección 4.2) con evidencia.
- Un validador pydantic comprueba estructura y campos obligatorios.
- Un resolutor convierte ContractDraft -> Contract (normalización por
  plataforma, conflictos, defaults SOLO cuando el brief los autoriza).
- Campo obligatorio sin evidencia -> null/UNKNOWN -> MANUAL_REVIEW o
  NEW_ARCHETYPE (nunca inventar).
- El extractor se escribe GENÉRICO (para cualquier brief), no "para estos 5".
  Cada campaña nueva se usa como fixture: si parsea bien confirma robustez;
  si rompe, señala exactamente qué arquetipo/campo falta (posible PR nuevo).

--------------------------------------------------------------------
7. CLASIFICACIÓN DE CAMPAÑA NUEVA (fase E)
--------------------------------------------------------------------
Tras extraer el contrato, el modelo decide si es CONOCIDA o NUEVA:
- KNOWN: encaja limpio en un arquetipo; procesar normal.
- KNOWN_WITH_VARIATION: valor nuevo dentro de un campo existente; procesar y
  registrar en campaigns/variations.md.
- NEW_ARCHETYPE: el brief exige algo que el contrato no puede representar.
  DETENERSE. (a) guardar fixture en campaigns/pending/; (b) abrir issue
  describiendo qué campo/modo falta; (c) proponer PR (schema + modo + fixture
  + test) para review del owner. NUNCA mergear solo.
Emitir veredicto legible: {classification, arquetipo, campos_no_cubiertos[],
accion_tomada, pr_propuesto?}.

El owner revisa cada PR: el modelo nunca modifica código en silencio. Ante
algo nuevo: PARA, REPORTA y PROPONE.

--------------------------------------------------------------------
8. STACK Y HARDWARE OBJETIVO
--------------------------------------------------------------------
Laptop: Ryzen 5 4600H, GTX 1650Ti Mobile 4GB VRAM, 24GB RAM.
Cuello de botella: VRAM (4GB). As bajo la manga: NVENC en la 1650Ti.

- Descarga video:     yt-dlp (YouTube, Twitch VOD, Kick, X) + streamlink
- Descarga chat:      chat-downloader (chat-replay de streams)
- Transcripción:      faster-whisper CON word-level timestamps.
                      small int8 = ruta garantizada en 4GB.
                      medium = EXPERIMENTAL: no afirmar que cabe hasta medirlo
                      en este hardware concreto (comando + resultado medido).
- Detección momentos: PySceneDetect + energía de audio (ffmpeg) + densidad de
                      chat (msgs/seg) para streams. Un LLM lee el transcript
                      y elige los N mejores según contrato.
- Reframe 9:16:       MediaPipe (CPU, ligero), crop dinámico.
- Subtítulos karaoke: timestamps -> .ass (libass) -> quemado ffmpeg
- Render:             ffmpeg h264_nvenc (usa la 1650Ti). Libera CPU.
- Cerebro/LLM:        API vía opencode go (NO correr LLM grande local con 4GB).
                      Respaldo offline mecánico: Qwen2.5-3B Q4 en CPU.
- Anti-duplicado:     ffmpeg -map_metadata -1 + re-encode + micro-variación
                      (zoom/color leve). NUNCA llamarlo "pasar desapercibido".
                      Descripción correcta: "normalización de metadata y
                      transformación de formato para preparar material
                      autorizado y evitar colisiones accidentales de archivos".
                      El gate corre DESPUÉS de estas transformaciones.
- Slideshow:          sub-pipeline aparte (HTML->PNG con Playwright o PIL).

Política de VRAM (una etapa GPU a la vez):
1) usar GPU para una etapa; 2) liberar memoria; 3) pasar a la siguiente;
4) si hay OOM, degradar a modelo menor o CPU; 5) registrar la degradación en
el run manifest. Nunca fusionar etapas GPU "para ahorrar tiempo" sin medir.

Pipeline en una línea:
yt-dlp/streamlink -> faster-whisper -> PySceneDetect + audio/chat -> LLM elige
y escribe -> reframe MediaPipe -> subs .ass -> ffmpeg NVENC -> GATE compliance
-> carpeta de entrega (el owner sube a mano).

--------------------------------------------------------------------
9. RUNTIME LLM vs HERRAMIENTAS DE DESARROLLO
--------------------------------------------------------------------
Son dos capas distintas y no se mezclan:
A) Herramientas de desarrollo: opencode / Claude Code / Codex construyen y
   mantienen el repo. Cualquiera de las tres debe poder trabajar aquí.
B) Runtime LLM: Kliptych necesita su propia interfaz, independiente del
   agente que desarrolla:

   class CampaignModel(Protocol):
       def extract_contract(self, brief: str) -> ContractDraft: ...
       def select_segments(self, transcript: Transcript) -> SegmentSelection: ...
       def write_caption(self, contract: Contract, piece: Piece) -> Caption: ...

   El primer backend es UNO solo. Los demás se agregan cuando exista una
   razón real (no por simetría).
   Requisitos: timeout, retry con backoff y política explícita de fallback.
   Si el modelo offline produce salida de menor confianza, se registra como
   low_confidence en el manifiesto; no se hace pasar por equivalente.
   Los tests NUNCA llaman a modelos reales: respuestas grabadas, fixtures
   golden y contract tests.

--------------------------------------------------------------------
10. REPRODUCIBILIDAD Y SEGURIDAD
--------------------------------------------------------------------
10.1 Cada corrida genera runs/<run_id>/run_manifest.json con:
- hash del brief; hash de todos los assets (sha256)
- schema_version del contrato; versión del modelo; versión del prompt
- versión de Whisper; versión de ffmpeg; argumentos de render
- hardware detectado (GPU, VRAM, degradaciones)
- hashes de los outputs
- resultado completo del gate

10.2 Seguridad:
- límite de tamaño de descargas y timeout en toda descarga
- protección contra path traversal al resolver rutas de assets
- PROHIBIDO construir comandos shell concatenando texto del LLM
  (usar listas de argumentos, nunca shell=True con interpolación)
- sandbox o directorio temporal por corrida; limpieza determinista
- limpieza de secretos y URLs con credenciales antes de loggear
- allowlist de URLs permitidas para descarga
- manejo explícito de DRM o recursos no descargables (fallar claro, no
  reintentar infinito)

--------------------------------------------------------------------
11. ROADMAP POR FASES (plataforma completa, construida por entregas)
--------------------------------------------------------------------
Meta: la plataforma completa (5 modos, clasificador, 3 backends si se
justifican, GUI opcional). Estrategia: diseñar todo, implementar por fases,
cada fase deja algo ejecutable. No construir 5 productos a la vez.

FASE A — Núcleo
- esqueleto, configuración, logging, detección de entorno (NVENC/VRAM/ffmpeg)
- CI desde el día 1 (lint + type + test)
- contrato pydantic v1.0 + ContractDraft con evidencia + validación
- asset registry con sha256/MIME/origen/licencia
- run_manifest.json
- ingestores de briefs locales (.docx, PDF, texto/markdown) y texto pegado;
  Google Docs y Notion por exportación manual, sin OAuth ni APIs (issue #5)
- interfaz CampaignModel + UN backend + fixtures grabados
- exportador común (estructura de carpetas por campaña/plataforma)
- gate común (checks deterministas base)

FASE B — Primer modo funcional (walking skeleton)
- modo given_clips end-to-end: brief -> contrato -> ensamblado -> caption/
  hashtags/tags/audio -> watermark -> gate -> paquete de entrega
- pruebas de rechazo con fixtures rotos
- PRIMERA DEMO completa: fixture entra y sale como pieza válida con reporte
- README con los dos casos (PASS y REJECTED)

FASE C — Motor de video largo (modo estrella)
- descarga/ingesta (yt-dlp, streamlink, chat-downloader)
- transcripción word-level (faster-whisper small int8)
- detección de momentos: escenas + energía + densidad de chat
- selección de segmentos por LLM con salida estructurada
- reframe 9:16 (MediaPipe)
- subtítulos karaoke (.ass + ffmpeg)
- render NVENC/CPU con degradación registrada
- gate sobre el artefacto final (video completo, no parámetros)

FASE D — Modos especiales
- audio_locked; repost_ugc; slideshow (sub-pipeline aparte)

FASE E — Inteligencia de campañas
- clasificador KNOWN / KNOWN_WITH_VARIATION / NEW_ARCHETYPE
- campaigns/pending/ + variations.md + propuesta de PR
- nunca modificar código sin aprobación del owner

FASE F — Operación
- API headless / orquestador; CLI completa
- reanudación de jobs, logs, manifiestos
- GUI opcional encima de la API

Orden de construcción: A -> B -> C -> D -> E -> F, en orden, sin saltar.
Diferencia con v1: el primer recorrido end-to-end (A+B) existe ANTES de
construir el motor de video largo. Nada se marca como implementado si solo
está representado en el schema.

--------------------------------------------------------------------
12. ESTRATEGIA DE PRs Y REVISIÓN
--------------------------------------------------------------------
PRs pequeños, pero no artificialmente pequeños: cada PR cierra una capacidad
verificable. Ejemplos:
- feat: valida el contrato con evidencia textual
- feat: exporta un paquete de entrega para given_clips
- feat: rechaza captions sin menciones obligatorias
- feat: registra hashes de assets y outputs
- feat: genera segmentos word-level dentro de los límites
- feat: renderiza subtítulos karaoke en formato vertical

Cada PR deja: código + tests + fixture + documentación + criterio de
aceptación + evidencia de ejecución (comando y salida). CI verde obligatorio
(lint + type + test) antes de merge.

Ciclo de revisión (mismo agente puede programar y revisar, pero no en la
misma fase mental; una auto-revisión inmediata es complaciente):
1) el agente implementa el PR;
2) ejecuta tests y CI;
3) termina su turno;
4) terminado el PR y con el CI verde, el owner pide la revisión y se invoca
   la skill "evidence-driven-review" sobre el PR en la misma sesión; los
   revisores se lanzan como subagentes con contexto limpio, independientes
   entre sí y de quien implementó;
5) la skill fija un snapshot inmutable (base_sha/head_sha), despliega
   revisores independientes por frente (corrección funcional, contratos e
   integraciones, seguridad, pruebas), somete cada candidato a validación
   adversarial por un validador distinto del descubridor y cierra con mapa
   de cobertura;
6) devuelve hallazgos confirmados con evidencia, severidad y corrección
   mínima, más limitaciones. La skill no aprueba nada y en modo report no
   publica: publicar comentarios exige autorización explícita del owner;
7) el agente corrige los hallazgos y ejecuta una re-revisión del delta con
   la misma skill (clasifica los hallazgos previos como persistentes,
   corregidos, reintroducidos o no verificables);
8) el owner aprueba y se pasa al siguiente PR.

Prohibido: saltar etapas en silencio, mezclar refactors con features en el
mismo PR, dejar un modo marcado como implementado cuando solo está en el
schema, o mergear un PR de NEW_ARCHETYPE sin aprobación.

--------------------------------------------------------------------
13. FIXTURES Y CARPETAS
--------------------------------------------------------------------
- campaigns/fixtures/   briefs pequeños, sintéticos o anonimizados (SÍ a Git)
- campaigns/private/    material real de campaña (NUNCA a Git)
- campaigns/pending/    briefs de NEW_ARCHETYPE esperando decisión (anonimizar)
- campaigns/variations.md  log de variaciones KNOWN_WITH_VARIATION
- assets/samples/       archivos libres de derechos (SÍ a Git)
- runs/                 resultados locales de cada corrida (NUNCA a Git)

.gitignore debe cubrir: videos completos de campaña, credenciales, cookies,
capturas con teléfonos, datos de cuentas, información de pago y cualquier
documento que no se pueda redistribuir.

--------------------------------------------------------------------
14. CRITERIOS DE ACEPTACIÓN
--------------------------------------------------------------------
MVP (fases A+B):
- un brief de Content Rewards entra, se convierte en contrato con evidencia,
  y produce un paquete de entrega para given_clips.
- el gate acepta una pieza correcta y RECHAZA piezas con caption sin marca,
  hashtag faltante, duración fuera de rango, spelling incorrecto, watermark
  ausente, audio incorrecto y segmento fuera de límites.
- README con ambos casos reales (PASS y REJECTED) y evidencia de ejecución.

Demo estrella (fase C):
> video largo -> transcript -> selección de momentos -> short vertical
> subtitulado -> caption -> gate aceptado/rechazado -> carpeta de entrega.

Venta del proyecto (frase oficial):
> Kliptych es un pipeline headless que convierte un brief de campaña en
> piezas de video mediante un contrato JSON versionado. Un gate fail-closed
> valida el artefacto final con evidencia mecánica antes de entregarlo para
> publicación manual.

--------------------------------------------------------------------
15. CONVENCIONES DE TRABAJO
--------------------------------------------------------------------
- Cada PR exige una revisión dual: al terminar el PR y con el CI verde, el
  agente ejecuta la skill "evidence-driven-review" en la misma sesión
  (revisión multifocal con subagentes de contexto limpio y validación
  adversarial de cada candidato; modo report por defecto), y luego el owner
  hace la validación y aprobación final. Si la herramienta no soporta
  subagentes ni la skill, se aplica el procedimiento de forma manual y se
  declara que no hubo independencia real.
- CI verde obligatorio (lint + type + test) antes de merge.
- Fixtures de campaña en campaigns/; cada campaña nueva es un test.
- Sin secretos en el repo; fixtures sintéticos donde haya datos personales.
- No marcar como implementado nada que no tenga corrida end-to-end.
- Toda afirmación de rendimiento/VRAM/compatibilidad requiere medición en el
  hardware objetivo (comando + resultado), nunca estimaciones.
- Comentarios solo cuando agreguen información; nada de código muerto.
====================================================================
