# Studio domain glossary

## Output aspect

The width-to-height ratio of an entire editable variant: `9:16` or `16:9`.
It is independent of a layout preset: the `horizontal` layout can still be
letterboxed inside a vertical output. Every timeline position uses the variant's
one output aspect. Final dimensions are 1080×1920 or 1920×1080; preview dimensions
are 720×1280 or 1280×720. These are output specifications, not performance claims.

## Format copy

A new variant with fresh variant/position IDs and the requested output aspect.
It retains the original's evidence binding, times, normalized crops, captions,
watermark and review state. It does not change the original or reuse its previews.

## Export job

A persisted request to export a particular revision of a variant, its evidence
binding and audio-normalization setting to a unique destination. A revision is a
canonical digest of the complete variant and current project evidence. It is
not a snapshot that authorizes rendering changed or missing source files.

## Export queue

An ordered, project-local collection of explicitly selected export jobs. Only
one job runs at a time. Each output is published independently after the complete
Gate passes; the queue is not an atomic campaign delivery. A failed job does not
prevent independent jobs from being attempted, and partial success is never
labelled total success.

## Stop after current

A cooperative queue stop: the current job finishes normally and no next job
starts. Pending jobs remain persisted. This is not cancellation of the current
FFmpeg process and does not label an unfinished artifact as completed.

## Interrupted job

A job persisted as running whose owning worker has ended without recording a
terminal result. Recovery never silently retries it: explicit user action is
required. Previously published files are never overwritten by a retry.
