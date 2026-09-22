# AGENTS.md — Kliptych

Instrucciones para agentes de código (opencode, Claude Code, Codex) que
trabajen en este repo. El contexto completo del proyecto vive en
`docs/brief_v2.md`; este archivo resume las reglas operativas.

## Contrato de trabajo

- Diseñar todo, implementar por fases A→F en orden, sin saltar. Cada fase deja
  algo ejecutable.
- Nada se marca como implementado si no tiene corrida end-to-end. No vale solo
  estar representado en el schema.
- TDD: test rojo primero para cada comportamiento nuevo.
- PRs pequeños pero no artificialmente pequeños: cada PR cierra una capacidad
  verificable con código + tests + fixture + documentación + criterio de
  aceptación + evidencia de ejecución.
- CI verde obligatorio (lint + type + test) antes de merge.
- Prohibido mergear sin aprobación del owner, mezclar refactors con features,
  o modificar código/contratos para "hacer pasar" un brief nuevo. Ante algo
  nuevo: PARAR, REPORTAR y PROPONER.

## Gates locales

```sh
uv run ruff format --check .
uv run ruff check .
uv run basedpyright
uv run pytest
```

Sin `# noqa`, `# type: ignore`, tests saltados ni umbrales relajados para pasar
un gate. Se arregla el código.

## Reglas duras de datos

- `campaigns/private/` y `runs/` NO se commitean.
- Fixtures de CI sintéticos o anonimizados.
- Sin tokens, credenciales, cookies, datos personales ni información de pago.
- Nunca construir comandos shell concatenando texto del LLM; listas de
  argumentos, jamás `shell=True` con interpolación.
- Toda afirmación de rendimiento/VRAM/compatibilidad requiere medición en el
  hardware objetivo (comando + resultado), nunca estimaciones.

## Proceso de revisión

Al terminar un PR y con el CI verde, el owner pide la revisión: se invoca la
skill `evidence-driven-review` en modo report (subagentes con contexto limpio,
validación adversarial, snapshot base/head). Los hallazgos confirmados se
corrigen y se hace re-revisión del delta. La skill no aprueba ni publica nada.
