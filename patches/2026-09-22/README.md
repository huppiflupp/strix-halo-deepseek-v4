# Patch series: llama.cpp Vulkan changes measured in STRIX-HALO-KERNELS.md

Base: llama.cpp master `ec9281505` (2026-09-21) + PR #27952 (int8 coopmat matmul, branch as of 2026-09-21, commits `aa338ea88..df9bcc16a`).
Apply with `git am` on top of that base. **Every change sits behind an environment variable and is off by default.** These are lab
patches, not PRs: they were written with an AI coding assistant, measured on one machine (Radeon 8060S / gfx1151, RADV 26.2.3) and
never tested on another GPU. Use at your own risk; check correctness on your model before you trust a speed-up.

| Patch | What | Switch | Result on this machine |
|---|---|---|---|
| 0001 | the Vulkan part of PR #27703 (contiguous f16 KV copy, by Nathanw1014) + 32-wide subgroups for prompt attention + V scratch stored transposed + two experiments (KV chunking, fused gate_up for gpt-oss) | `GGML_VK_LAB_FA_WAVE32=1`, `GGML_VK_LAB_FA_VT=1` (+ `GGML_VK_LAB_FA_CHUNK_MB`, `…_SPLIT_MAJOR`, fused gate_up: no gain) | prompt at 32 k: gpt-oss-120b +25 %, Qwen3-30B-A3B +49 %; KLD vs ROCm unchanged |
| 0002 | packed dense matmul: operands re-laid tile by tile, 4 x 4 tiles per wave, packed weights kept | `GGML_VK_LAB_MM_PACKED=2` (`=1` repacks every call; `GGML_VK_LAB_MM_CACHE_MB` caps the cache) | 32-38 TFLOPS vs 14-21; Qwen3.8-27B prompt +13...+50 %; **costs an f16 copy of all dense weights** (29 -> 76 GiB) |
| 0003 | tiled transpose-concat for the delta-net conv state — **kernel by Nathanw1014 (strix-halo-llamacpp)**, ported unchanged; plus two attention layout experiments (no gain) | `GGML_VK_LAB_CONCAT_T=1` (`GGML_VK_LAB_FA_KT`, `GGML_VK_LAB_FA_TILES`: no gain) | Qwen3.6-35B-A3B pp2048 +64 %, bit-identical |
| 0004 | compile-time phase-removal guards in `flash_attn_cm1.comp` for timing | `#define LAB_SKIP n` (0 = unchanged) | timing only — results are wrong when != 0 |
| 0005 | packed matmul loads B tiles column-major for k <= 6144 | `GGML_VK_LAB_MM_BCOL=1` | +4 % on top of 0002, bit-identical |

Correctness method and all raw numbers: `../../STRIX-HALO-KERNELS.md`, `../../messungen/2026-09-22/`. Licence: llama.cpp's MIT licence.
