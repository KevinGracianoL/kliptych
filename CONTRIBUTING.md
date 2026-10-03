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

## Comparing against another revision

To compare behaviour against another revision, use a separate worktree with its
own environment:

```sh
git worktree add ../kliptych-base <base_sha>
cd ../kliptych-base && uv sync
```

**Then prove which code you loaded before quoting any difference.** The editable
install makes `PYTHONPATH` unreliable: a `PYTHONPATH` pointed at the worktree
still imports the package from the main checkout, because the editable finder
wins over path order. A comparison built on that silently runs the same code
twice and "confirms" whatever you expected.

```sh
uv run python -c "import kliptych.gate.engine as e; print(e.__file__)"
```

If the printed path is not inside the worktree you created, the comparison is
void.

## Verify that an operation did what it reported

A silent no-op is worse than a failure, because it removes the signal.

- An edit tool that errors on no-match beats a string replace that prints `ok`.
- A harness reporting 0 violations must prove it BUILT the case it claims, not
  only that it ran.
- A count quoted in a PR body must be recomputable from a command in the repo.
- A kill count measured before a code change is not a measurement of the current
  code.

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
