# Operations — CLI 2-Step Workflow with Human Approval

Rules with no mechanical validator (official-audio identity, full-video
watermark, on-screen product presence, unmapped brief requirements,
brand-safety risk) resolve to `MANUAL_REVIEW`, never to a silent `PASS`.
Affected pieces pause at `PENDING_REVIEW` and export only with an explicit
operator signature.

## Step 1 — Run pauses at `PENDING_REVIEW`, exits non-zero, exports nothing

```sh
uv run kliptych campaign campaigns/fixtures/long-video/brief.md \
  --out runs/demo-review \
  --mode long_video \
  --url "https://www.youtube.com/watch?v=<id>"
# exit code 1 — pieces awaiting manual review, delivery/ not written
```

## Step 2 — Operator reviews the pending rules, then resumes with attribution

```sh
uv run kliptych campaign campaigns/fixtures/long-video/brief.md \
  --out runs/demo-review \
  --mode long_video \
  --url "https://www.youtube.com/watch?v=<id>" \
  --resume --approve-manual-review --approved-by "Kevin Graciano"
# exit code 0 — exports to runs/demo-review/delivery/
```

Omitting `--approved-by` is a hard error (exit 1).

## Delivery record — approval sealed next to provenance

```json
{
  "manually_approved_rules": ["audio.official_selection", "watermark.full_video"],
  "approved_by": "Kevin Graciano",
  "approved_at_utc": "2026-09-25T02:14:00+00:00",
  "brief_sha256": "1ccafe8d5e0fcf45100e6eb947239742cd69b8cbb987640285ebaf738e622989"
}
```

(`manually_approved_rules`, `approved_by`, `approved_at_utc` are written by
the exporter per piece; `brief_sha256` and `contract_sha256` are sealed in
`run_manifest.json` alongside gate results, render arguments, model versions,
and hardware/degradation info.)
