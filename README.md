# Kliptych

Pipeline headless que convierte briefs de campaña en paquetes de entrega
mediante un contrato JSON versionado. Un gate fail-closed valida el artefacto
final con evidencia mecánica antes de entregarlo para publicación manual.

> Kliptych es un pipeline headless que convierte un brief de campaña en piezas
> de video mediante un contrato JSON versionado. Un gate fail-closed valida el
> artefacto final con evidencia mecánica antes de entregarlo para publicación
> manual.

## Alcance

ENTRA:

- Ingesta de briefs y materiales de referencia.
- Extracción de un contrato JSON versionado con evidencia textual por campo.
- Generación de piezas: videos verticales, slideshows, captions, hashtags,
  menciones, instrucciones de audio y metadata.
- Gate de compliance por pieza, ejecutado sobre el artefacto final.
- Paquete de entrega organizado por campaña/plataforma con reporte del gate.
- CLI/API headless. GUI opcional y al final.

NO ENTRA (no se promete en ningún doc):

- Crear cuentas o canales (las plataformas no lo permiten por API).
- Publicar por API. La publicación es manual y bajo control del owner.
- Verificar condiciones post-publicación (views para payout, geo real,
  analytics): son recordatorios del exportador, nunca checks del gate.
- Garantizar la selección manual del audio nativo durante la subida; el gate
  solo verifica el archivo de audio entregado.

## Estado

Fases A (núcleo) y B (walking skeleton `given_clips`) completas. Nada se
considera implementado si no tiene una corrida end-to-end; este README se
actualiza solo con evidencia real.

- [x] Bootstrap: uv + Python 3.13, ruff, basedpyright, pytest y CI.
- [x] Contrato pydantic v1.0 con evidencia por campo.
- [x] Resolutor `ContractDraft` → `Contract` (normaliza, clasifica y marca `MANUAL_REVIEW`/`NEW_ARCHETYPE` sin inventar).
- [x] Asset registry con sha256/MIME/origen/licencia.
- [x] Run manifest reproducible.
- [x] Detección de entorno (ffmpeg/NVENC/GPU) y CLI `kliptych env`.
- [x] Gate core determinista fail-closed.
- [x] Interfaz `CampaignModel` + primer backend (OpenAI-compatible) con fixtures grabados.
- [x] Exportador de paquetes de entrega (solo piezas que pasan el gate) con recordatorios post-publicación.
- [x] Ingesta de briefs locales (texto/markdown, PDF, DOCX) con hash normalizado y CLI `kliptych ingest`.
- [x] Google Docs y Notion por exportación manual: archivo exportado o texto pegado (`kliptych ingest -`); sin OAuth ni APIs (issue #5).
- [x] Walking skeleton `given_clips` end-to-end: brief → contrato → ensamblado vertical → caption → gate → paquete, con CLI `kliptych run` y demo PASS/REJECTED.
- [x] Fixtures rotos en CI: caption sin mención, hashtag faltante, duración fuera de rango, audio ausente, término prohibido, spelling sin subtítulos y watermark sin validador son rechazados por el gate (segmento fuera de límites: Fase C).

Fases siguientes: C (motor de video largo), D (modos especiales), E
(clasificación de campañas), F (operación/API/GUI).

## Demo del walking skeleton (`given_clips`)

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

(`manifest` es la ruta absoluta que imprime la CLI, recortada aquí a la raíz
del repo; `contract_sha256` es estable entre corridas del mismo brief: la
proyección canónica excluye metadatos volátiles de resolución.)

El paquete contiene `demo-given-clips/tiktok/clip-01.mp4`,
`clip-01.metadata.json` (caption `Mira este clip de @marca y sigue el #marca`,
hashtags `["#marca"]`, audio `own_clip`), `clip-01.gate.json` y
`delivery_report.json`; el manifiesto registra hashes, versiones de prompt y el
resultado completo del gate en `runs/<run_id>/run_manifest.json`.

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
de la pieza; `runs/` no se commitea (política de datos). Los assets bajo
`campaigns/private/` solo se aceptan desde el brief de la propia campaña;
material de otra campaña o de `runs/` se rechaza en el pipeline (fail-closed).
El CI cubre además los rechazos por hashtag faltante, duración fuera de rango,
audio ausente, término prohibido, spelling sin subtítulos y watermark sin
validador (`tests/test_pipeline.py`).

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
