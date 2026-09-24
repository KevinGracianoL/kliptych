# Kliptych

**Kliptych es un Intelligent Campaign Engine with Self-Proposing Code.** No se
limita a procesar video: descarga, transcribe, detecta momentos, reframea a
9:16 y, sobre todo, **evalúa y clasifica cada campaña** para proponer sus propias
actualizaciones de configuración como Pull Requests bajo una política estricta
de **Zero Auto-Merge**.

> Kliptych convierte un brief de campaña en piezas de video vertical mediante un
> contrato JSON versionado. Un gate fail-closed valida el artefacto final con
> evidencia mecánica antes de entregarlo para publicación manual.

## What It Does

- 🧠 **Descarga acotada** de media y chat (`yt-dlp`, `streamlink`,
  `chat-downloader`) con timeout explícito y cota dura de bytes; URL validada
  contra destinos locales/privados.
- 🧠 **Transcripción word-level** con `faster-whisper` (`small` int8, VRAM-safe)
  y detección de momentos por escena, energía de audio y densidad de chat; un
  LLM elige los segmentos según contrato.
- **Reframe a 9:16** con MediaPipe en CPU (sigue al sujeto) y **subtítulos
  karaoke** `.ass` quemados con `h264_nvenc` o `libx264`.
- **Clasificación de campañas** (`KNOWN` / `KNOWN_WITH_VARIATION` /
  `NEW_ARCHETYPE`) y **propuesta de cambios de código/config como Pull
  Requests** de GitHub (nunca fusiona — el owner aprueba).

## Operational Modes (Fases C + D)

Todos los modos publican sus artefactos finales de forma atómica y comparten el
mismo registro de limpieza determinista.

### Long Video — `run_long_video`

Encadena el pipeline completo:

```
URL → download → transcribe → moments → LLM select → reframe 9:16
    → subtitles (.ass) → final vertical video
```

Compone los módulos de C2–C4 y traduce cualquier fallo de una etapa a
`PipelineError` conservando la causa.

### Audio Locked — `audio_locked=True`

Inyecta o reemplaza una pista de audio externa (música viral o voiceover) sobre
el video. La mezcla usa `ffmpeg amix` con `duration=first`, normalización de
sample rate (`aformat=sample_fmts=fltp:channel_layouts=stereo`) y volumen
proporcional:

- `audio_mix_ratio = 1.0` → la pista externa **reemplaza** la original
  (`-map 1:a:0` + `-shortest`).
- `0.0 < ratio < 1.0` → mezcla proporcional (`1 - ratio` original, `ratio`
  externa).

Acepta `audio_track_path` o `audio_track_url` (mutuamente excluyentes).

### Repost / UGC — `repost_mode=True`

Salta por completo las capas de inteligencia: **no transcribe, no detecta
momentos, no llama al LLM**. El video completo es el segmento único. Maneja la
rotación de metadata de grabaciones móviles y usa **passthrough `-c copy`**
cuando el video ya es 9:16 (sin re-encode). Coincide con `audio_locked=True`
para inyectar una pista externa.

### Slideshow — `run_slideshow`

Ensambla una secuencia de imágenes (`JPG`/`PNG`, locales o URL) en un video
vertical continuo con el demuxer `concat` de ffmpeg, escalando y rellenando cada
slide a 9:16. **Exige audio externo**: sin `audio_locked=True` no arranca, y
requiere `audio_mix_ratio >= 1.0` (el audio es obligatorio). No transcribe
porque no hay voz ni momentos que seleccionar.

## Campaign Intelligence (Fase E)

Antes de renderizar, Kliptych clasifica el brief y su contrato validado
(`intelligence.py`) y enruta según el resultado (`campaign_manager.py`):

| Arquetipo | Significado | Acción |
| --- | --- | --- |
| `KNOWN` | Encaja en un arquetipo existente | Renderiza el video normalmente |
| `KNOWN_WITH_VARIATION` | Valores nuevos en campos existentes | Renderiza + registra la variación + propone PR |
| `NEW_ARCHETYPE` | No representable por el schema actual | Propone PR + **detiene** el procesamiento de video |

La invariante es estructural: el motor de video solo se referencia dentro de la
rama `KNOWN`. Un `NEW_ARCHETYPE` queda en `MANUAL_REVIEW` y **nunca** toca el
pipeline.

## 🛡️ The Iron Rule: Zero Auto-Merge

El sistema proponer, pero **jamás** fusiona. Concretamente:

- Crea una rama Git a partir de `main`.
- Genera la configuración (`JSON`) del nuevo arquetipo y una entrada Markdown en
  el log de variaciones.
- Abre un Pull Request en GitHub vía CLI `gh` o vía API REST (con timeout
  explícito y cota de bytes de respuesta).
- **NUNCA** mergea: no existen métodos `merge_pull_request`, `auto_merge` ni
  `approve` en ningún proveedor ni en el motor. El flujo termina al devolver la
  URL del PR.
- El **owner humano** debe aprobar y fusionar manualmente.

## Architecture & Invariants

- 🛡️ **Zero `shell=True`**: toda invocación externa usa **lista de argumentos**
  (`list[str]`), nunca interpolación de shell. No hay `os.system` ni
  `subprocess` con `shell=True` en `src/`.
- **_CleanupRegistry**: cada temporal (`.part`, `.ass`, `concat.txt`, clips
  intermedios, reframes) se registra y se elimina en un `finally`, tanto en el
  camino feliz como ante error. Un archivo preexistente nunca se toca.
- **Aislamiento de red en tests**: `FakeTransport`, proveedores inyectados y
  HTTP en loopback. El CI no hace ninguna llamada de red real.
- **Seguridad de tipos**: `StrEnum` para todos los estados y modelos Pydantic
  `frozen=True` con `extra="forbid"` en las fronteras de dominio.
- **Higiene de VRAM**: una etapa GPU a la vez; los modelos se liberan al
  terminar (`gc.collect()` + `cuda.empty_cache()`), incluso ante error. MediaPipe
  y el reframe corren siempre en CPU.
- **Publicación atómica**: `temp` + `Path.replace` para todos los artefactos
  finales; solo un render exitoso reemplaza el destino.

## Project Status

Fases A (núcleo), B (walking skeleton `given_clips`), C (motor de video largo),
D (modos especiales: audio locked, repost/UGC, slideshow) y E (clasificación de
campañas + propuestas Git) **completas**. Nada se considera implementado si no
tiene una corrida end-to-end o test de integración; este README se actualiza solo
con evidencia real.

- [x] Bootstrap: uv + Python 3.13, ruff, basedpyright, pytest y CI.
- [x] Contrato pydantic v1.1 (incluye `segments` para long_video).
- [x] Resolutor `ContractDraft` → `Contract` (normaliza, clasifica y marca
  `MANUAL_REVIEW`/`NEW_ARCHETYPE` sin inventar).
- [x] Asset registry con sha256/MIME/origen/licencia; run manifest reproducible.
- [x] Detección de entorno (ffmpeg/NVENC/GPU) y CLI `kliptych env`.
- [x] Gate core determinista fail-closed y exportador de paquetes de entrega.
- [x] Walking skeleton `given_clips` end-to-end (ver demo abajo).
- [x] **Fase C** — pipeline long_video: descarga acotada, transcripción
  word-level, detección de momentos, selección LLM, reframe 9:16, subtítulos
  karaoke y publicación atómica (`tests/test_orchestrator_integration.py`).
- [x] **Fase D** — audio locked (`tests/test_audio_locked.py`), repost/UGC
  (`tests/test_repost_ugc.py`) y slideshow (`tests/test_slideshow.py`), con sus
  tests de integración sin red.
- [x] **Fase E** — clasificación `KNOWN`/`KNOWN_WITH_VARIATION`/`NEW_ARCHETYPE`
  (`tests/test_intelligence.py`), routing maestro
  (`tests/test_campaign_manager.py`) y motor de propuestas con Zero Auto-Merge
  (`tests/test_git_proposals.py`).

Fase F (operación/API/GUI) pendiente.

### Demo del walking skeleton (`given_clips`)

El fixture `campaigns/fixtures/given-clips/` entra como brief y sale como
paquete de entrega validado por el gate. El clip sintético de muestra vive en
`assets/samples/given-clips-sample.mp4` (libre de derechos; regenerable con
`ffmpeg -y -v error -f lavfi -i color=c=blue:s=320x240:d=9 -f lavfi -i
sine=frequency=440:duration=9 -shortest -c:v libx264 -pix_fmt yuv420p -c:a aac
-movflags +faststart assets/samples/given-clips-sample.mp4`) y los captions se
reproducen desde `recorded/` sin llamar a un modelo real.

Caso aceptado — 8/8 checks pasaron:

```sh
uv run kliptych run campaigns/fixtures/given-clips/brief.md \
  --out runs/demo-pass/package \
  --recorded campaigns/fixtures/given-clips/recorded
```

```json
{
  "run_id": "20260923T050421195926Z",
  "outcome": "exported",
  "contract_sha256": "1ccafe8d5e0fcf45100e6eb947239742cd69b8cbb987640285ebaf738e622989",
  "package": "runs\\demo-pass\\package",
  "manifest": "runs\\20260923T050421195926Z\\run_manifest.json",
  "exported": ["clip-01"],
  "rejected": []
}
```

Caso rechazado — `caption.required_mention` FAIL (`missing: ["@marca"]`), la
pieza no se exporta:

```sh
uv run kliptych run campaigns/fixtures/given-clips-rejected/brief.md \
  --out runs/demo-rejected/package \
  --recorded campaigns/fixtures/given-clips-rejected/recorded
```

```json
{
  "run_id": "20260923T050425560749Z",
  "outcome": "blocked",
  "contract_sha256": "1ccafe8d5e0fcf45100e6eb947239742cd69b8cbb987640285ebaf738e622989",
  "package": "runs\\demo-rejected\\package",
  "manifest": "runs\\20260923T050425560749Z\\run_manifest.json",
  "exported": [],
  "rejected": [
    {
      "piece_id": "clip-01",
      "platform": "tiktok",
      "reason": "el gate no aprob\u00f3 la pieza (rejected): caption.required_mention"
    }
  ]
}
```

El paquete rechazado solo contiene `delivery_report.json` con el gate completo
de la pieza; `runs/` no se commitea (política de datos).

## Desarrollo

Requiere [uv](https://docs.astral.sh/uv/) (gestiona Python 3.13 por sí mismo).

```sh
uv sync
uv run ruff format --check .
uv run ruff check .
uv run basedpyright
uv run pytest
```

Doble cero en CI antes de mergear: lint + type + test, en Linux y Windows.

### Runtime LLM

El primer backend habla con un endpoint OpenAI-compatible configurado por el
operador (nunca texto del brief):

```sh
KLIPTYCH_LLM_BASE_URL=https://<endpoint>/v1
KLIPTYCH_LLM_API_KEY=<secreto>
KLIPTYCH_LLM_MODEL=<modelo>
```

Los tests y las demos nunca llaman modelos reales: reproducen respuestas
grabadas con `RecordedModel` desde `campaigns/fixtures/*/recorded/`.

### Ingesta de briefs

`kliptych ingest <ruta>` acepta texto/markdown, PDF y DOCX exportados
localmente; `kliptych ingest -` lee un brief pegado por stdin. Google Docs y
Notion se integran exportando el documento o pegando el texto: el pipeline es
100% local, sin OAuth, tokens ni APIs en la nube (issue #5).

Todo cambio entra por rama + PR con CI verde; `main` no recibe pushes directos
(hook local en `.githooks/pre-push`; activar una vez con
`git config core.hooksPath .githooks`).

## Política de datos

- `campaigns/private/` y `runs/` nunca se commitean.
- Fixtures de CI sintéticos o anonimizados.
- Sin tokens, credenciales, cookies ni datos personales innecesarios.
