# Gate — Fail-Closed Compliance Engine

## `_derive_status` + Export Barrier

`_derive_status` (`src/kliptych/gate/engine.py`) maps **any** measured
`CheckOutcome.FAIL` to `GateStatus.REJECTED` — regardless of whether the rule
was classified `hard`, `recommended`, or `manual_review`. A second barrier
lives in the exporter (`src/kliptych/exporter.py`): `_is_piece_exportable`
refuses every piece except `PASSED`, or `PENDING_REVIEW` with explicit manual
approval **and** zero `FAIL` checks. Batch export is all-or-nothing atomic
(staging directory + atomic rename in `_publish`); a rejected piece aborts
the whole delivery, never a partial shipment.

The validator catalog is mechanical and closed (`KNOWN_VALIDATOR_RULES`,
17 rule ids). Declared rules without a registered validator never pass
silently: they resolve to `MANUAL_REVIEW` when the contract classified them
so or cited them, otherwise `UNSUPPORTED` — and `UNSUPPORTED` on a `hard`
rule blocks the piece. Requirements with no mechanical validator at all are
declared explicitly as `contract.unmapped[]` (`rule` + verbatim `quote`); the
engine emits one aggregate `rules.unmapped` check carrying every citation,
`MANUAL_REVIEW` by default, `FAIL` when an unmapped id collides with a known
validator.

## Compliance Catalog — Mechanical Validators (`src/kliptych/gate/checks.py`)

| Rule id | Verdict shape |
|---------|---------------|
| `artifact.integrity`, `artifact.video_stream`, `assets.required`, `audio.present` | Presence/integrity probes; missing or unmeasurable fails closed. |
| `audio.policy` | `internal_official_sound` → `MANUAL_REVIEW`, always. Never `PASS`. Any other policy → `PASS`. |
| `audio.silence` | Only active under `internal_official_sound`. Requires exactly one audio track (ffprobe count; 0 or 2+ → `FAIL` without measuring), then `volumedetect max_volume <= -80.0 dB` on the rendered MP4 (`-map 0:a:0`). Any measurement failure → `FAIL`. |
| `watermark.present` / `watermark.full_video` | OpenCV frame-by-frame `matchTemplate` against the contract PNG at the contracted `WatermarkConfig` zone (7 positions: `top_left`, `top_right`, `bottom_left`, `bottom_right`, `center_top`, `center_bottom`, `center`), contracted `scale_ratio` (default 0.20), opacity-aware scoring (contract floor 0.15) with alpha-edge/gradient fallback. `present` needs one matching sample; `full_video` needs all density-distributed samples (4/s, 12–120). Strong translucent signal without full confidence → `MANUAL_REVIEW`; anything else that cannot be proven → `FAIL`. Single `VideoCapture` pass, no per-frame ffmpeg processes. |
| `caption.required_mention`, `caption.required_hashtag`, `caption.first_line` | Token-boundary, case-insensitive matching against caption/hashtags. |
| `caption.forbidden` | Union of `caption_rules.forbidden` + campaign `prohibitions`, NFKD-normalized, word-boundary. Published-text hit → `FAIL`; spoken-only (`subtitle_text`) hit → `MANUAL_REVIEW`. |
| `subtitles.spelling_lock` | Exact brand spelling in subtitles; lyric-video pieces additionally checked against the cut `.lrc` window (identical token sequence and time-interval concordance with the burned ASS; missing `ass_path` → `FAIL`, no substitution). |
| `hook.keyword` (CB22) | Opening keyword must appear in timed `subtitle_segments` or `screen_text_segments` with `start_s <= 3.0 s` (NFKD-normalized). Missing, late, or untimed (plain `subtitle_text` only) → `FAIL`. First-3 s `volumedetect` is attached as an informational note and never changes the verdict. |
| `brand.safety` | Opt-in only: active when the campaign requires it (`brand_safety_required` + citation, or matching `prohibitions`). With the rule active, an injected LLM assessor decides — risk → `MANUAL_REVIEW`, clean → `PASS`. No assessor, assessor exception/timeout, or invalid response → `MANUAL_REVIEW` (fail-closed, never `PASS`). Inactive → `PASS` without invoking any model. |
| `layout.geometry` | Split-screen canvases must satisfy the contracted geometry (even 9:16 canvas, even gap, viable panels); split composition outside the supported pipeline → `FAIL`. |
| `duration.min` / `duration.max` | `ffprobe` duration inside bounds; unmeasurable → `UNSUPPORTED`, never `PASS`. |
| `rules.unmapped` | Aggregate over `contract.unmapped[]` citations: `MANUAL_REVIEW` with every `rule` + `quote` in evidence; `FAIL` when an unmapped id collides with the known-validator catalog. |

Authorship decides the verdict where it matters: `caption.forbidden` is
`FAIL` when a prohibited term appears in account-published text (caption,
hashtags) and `MANUAL_REVIEW` when it appears only in streamer-spoken
`subtitle_text`. An LLM is never permitted to clear a hard rule.
