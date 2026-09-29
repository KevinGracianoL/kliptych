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

## Roles de agentes

- **LongCat 2.0** es el orquestador: diseña, delega al programmer, verifica gates, abre PRs y reporta. No implementa código y no hace code reviews.
- **Gemini 3.8 Flash High via OmniRoute (programmer)** es el agente de implementación: ejecuta tareas de código delegadas por el orquestador. No hace code reviews.
- **Nemotron 3 Ultra 550B via OmniRoute (`reviewer`)** es el agente de revisión: ejecuta la skill `evidence-driven-review` sobre PRs completos (snapshot base/head, nunca commit por commit).
- El orquestador no implementa código directamente; delega al programmer la implementación y al reviewer las reviews, y verifica los resultados.

## Higiene de ramas

- Toda rama local y remota debe ser eliminada inmediatamente después de que su Pull Request haya sido mergeado a main.

## Reglas inquebrantables para módulos de descarga (C2)

1. **Subprocess**: lista de argumentos, nunca `shell=True` con interpolación.
2. **Aislamiento de red**: tests con fixtures locales, cero internet en CI.
3. **Límites**: timeout explícito + cota dura de bytes post-descarga.
4. **Alcance**: el PR de descarga NO incluye transcripción ni ensamblado; termina en la descarga del artefacto.

## Skill invocation

- La skill `evidence-driven-review` se invoca **una sola vez** sobre el PR completo (snapshot base/head), nunca commit por commit ni fase por fase.

## Proceso de revisión

Al terminar un PR y con el CI verde, el owner pide la revisión: `reviewer`
(Nemotron 3 Ultra 550B via OmniRoute) invoca la skill `evidence-driven-review` en modo
report (subagentes con contexto limpio, validación adversarial, snapshot
base/head). Los hallazgos confirmados se corrigen y se hace re-revisión del
delta. La skill no aprueba ni publica nada.
