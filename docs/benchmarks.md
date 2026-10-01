# Benchmarks — GTX 1650 Ti, 4,096 MiB VRAM

Measured host (`uv run kliptych env`): `NVIDIA GeForce GTX 1650 Ti`,
`vram_mib: 4096`, `nvenc_available: true`. Transcription in `int8` CUDA;
reframe via MediaPipe CPU delegate; render via `h264_nvenc`.

## Table A — CUDA Transcription Scaling

| Model | init (s) | 30s warm/cold (s) | 180s warm/cold (s) | Words (180s) | Marginal slope | Real-time factor | Peak VRAM |
|-------|----------|-------------------|---------------------|--------------|----------------|------------------|-----------|
| small | 2.0 | 2.47 / 2.81 | 11.82 / 12.08 | 569 | 0.062s/s | ~16x | 848 MiB (80% free) |
| large-v3-turbo | 4.4 | 3.86 / 3.99 | 18.01 / 18.06 | 561 | 0.094s/s | ~10.6x | 1,522 MiB (63% free) |

Both models stay well under the 4 GiB ceiling with headroom for the NVENC
stage that follows.

## Table B — E2E Pipeline (3-min 1080p VOD → 4 Portrait 9:16 Clips)

| Stage | Time (s) | VRAM (MiB) | Encoder |
|-------|----------|------------|---------|
| Transcribe CUDA | 13.0 | 486→494 | CUDA |
| Moments | 5.5 | 494 | — |
| MediaPipe Reframe 9:16 (4 segs) | 17.4 | 593 | h264_nvenc |
| ASS Subtitles Burn NVENC (4 segs) | 4.4 | 593 | h264_nvenc |
| Gate + Export | <1 | — | — |
