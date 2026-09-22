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

Fase A (núcleo) en construcción. Nada se considera implementado si no tiene una
corrida end-to-end; este README se actualiza solo con evidencia real.

- [x] Bootstrap: uv + Python 3.13, ruff, basedpyright, pytest y CI.
- [x] Contrato pydantic v1.0 con evidencia por campo.
- [x] Asset registry con sha256/MIME/origen/licencia.
- [ ] Run manifest reproducible.
- [ ] Gate core determinista.
- [ ] Ingestores de briefs (Google Docs, Notion, .docx, PDF, texto).
- [ ] Interfaz `CampaignModel` + primer backend con fixtures grabados.
- [ ] Exportador de paquetes de entrega.

Fases siguientes: B (walking skeleton `given_clips`), C (motor de video largo),
D (modos especiales), E (clasificación de campañas), F (operación/API/GUI).

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

## Política de datos

- `campaigns/private/` y `runs/` nunca se commitean.
- Fixtures de CI sintéticos o anonimizados.
- Sin tokens, credenciales, cookies ni datos personales innecesarios.
