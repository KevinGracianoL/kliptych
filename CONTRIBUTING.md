# Contributing — Kliptych

Operating rules for contributors (human or agent) working in this repo.
Full project context lives in `docs/brief_v2.md`; this file summarizes
the enforceable workflow.

## Work contract

- Every change enters via branch + PR with green CI; `main` receives no
  direct pushes (local hook in `.githooks/pre-push`, enable with
  `git config core.hooksPath .githooks`).
- Design everything; implement in phases A→F in order, without skipping.
  Each phase leaves something executable.
- Nothing counts as implemented without an end-to-end run. Schema-only
  representation does not qualify.
- TDD: red test first for every new behavior.
- PRs small but not artificially small: each PR closes one verifiable
  capability with code + tests + fixture + documentation + acceptance
  criterion + execution evidence.
- Green CI (lint + type + test) is mandatory before merge.
- Do not merge without owner approval, mix refactors with features,
  or modify code/contracts to "make pass" a new brief. On encountering
  something new: STOP, REPORT, and PROPOSE.

## Local gates

```sh
uv run ruff format --check .
uv run ruff check .
uv run basedpyright
uv run pytest
```

No `# noqa`, `# type: ignore`, skipped tests, or relaxed thresholds to
pass a gate. Fix the code.

## Data rules

- `campaigns/private/` and `runs/` are NEVER committed.
- CI fixtures are synthetic or anonymized.
- No tokens, credentials, cookies, personal data, or payment information.
- Never build shell commands by concatenating model output; use argument
  lists, never `shell=True` with interpolation.
- Every performance/VRAM/compatibility claim requires a measurement on
  the target hardware (command + result), never estimates.

## Roles

- **Maintainer**: designs the work, reviews every PR over the full base/head snapshot, verifies the gates, and merges. Reviews nothing it wrote itself.
- **Contributor**: works on one verifiable capability per PR.

No one merges their own work without review.

## Branch hygiene

- Every local and remote branch must be deleted immediately after its
  pull request has been merged into main.

## Hard rules for download modules (C2)

1. **Subprocess**: argument lists, never `shell=True` with interpolation.
2. **Network isolation**: tests use local fixtures, zero internet in CI.
3. **Limits**: explicit timeout + hard post-download byte cap.
4. **Scope**: the download PR does NOT include transcription or assembly;
   it ends at the downloaded artifact.

## Review process

When a PR is complete with green CI, the owner requests review: the
reviewer runs a single evidence-driven review over the full PR (base/head
snapshot, independent validation). Confirmed
findings are fixed and the delta is re-reviewed. Reviews report only;
they approve and publish nothing.
