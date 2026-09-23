# AGENTS.md — Kliptych

Instrucciones para agentes de código (opencode, Claude Code, Codex) que
trabajen en este repo. El contexto completo del proyecto vive en
`docs/brief_v2.md`; este archivo resume las reglas operativas.

## Contrato de trabajo

- Todo cambio entra por rama + PR con CI verde; `main` no recibe pushes
  directos (hook local en `.githooks/pre-push`, activar con
  `git config core.hooksPath .githooks`).
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

**La skill `evidence-driven-review` ya no se invoca commit por commit ni fase por
fase: es un gasto brutal de tokens y contamina el contexto.** El flujo es:

1. El agente implementa el PR completo (puede agrupar varios commits en la rama).
2. Verifica que el CI local esté verde (lint + type + test).
3. Empuja la rama y abre el PR con todos los commits agrupados.
4. Verifica que el CI remoto reporte verde.
5. **Se detiene por completo y avisa al owner.**
6. Solo el owner da la orden explícita de disparar `evidence-driven-review` sobre
   el PR completo (snapshot base/head del PR, no commits sueltos).

La skill se lanza una sola vez sobre el PR cerrado, nunca sobre estados intermedios.
Los hallazgos confirmados se corrigen y se hace re-revisión del delta. La skill no
aprueba ni publica nada.
