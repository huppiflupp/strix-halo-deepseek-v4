# HIP MMVQ thresholds on gfx1151 (llama.cpp PR #28613): the switch point depends on matrix width

Measured on 2026-09-19 and 2026-09-25. Raw data, scripts and analysis:
[`messungen/2026-09-25/hip-mmvq-28613/`](messungen/2026-09-25/hip-mmvq-28613/). (Like
[GPT-OSS-120B.md](GPT-OSS-120B.md), this file is in English on purpose.)

| | |
|---|---|
| Machine | AMD Ryzen AI MAX+ 395, Radeon 8060S (gfx1151, RDNA3.5, 40 CUs), 128 GB unified, 85 W |
| System | Nobara 44, kernel 7.2.6, ROCm 7.1 (Fedora packages) |
| llama.cpp | master `ec9281505` against the same commit plus the PR commit cherry-picked (`eb50b32e8`), HIP, `-DAMDGPU_TARGETS=gfx1151` |

## What the PR does

[ggml-org/llama.cpp#28613](https://github.com/ggml-org/llama.cpp/pull/28613) adds a per-type table
for RDNA3.5 to `ggml_cuda_should_use_mmvq`. The table decides up to which batch size ne11 the
matrix-vector kernel (MMVQ) is used before HIP switches to the tiled matrix kernel (MMQ). Master
uses MMVQ up to ne11 = 8 for every type. The PR switches earlier: Q4_K after 2, Q3_K and Q6_K
after 3, Q2_K and Q8_0 after 4, Q4_0 after 6.

## Earlier end-to-end results (2026-09-19)

These were posted to the PR at the time.

- Qwen3.8-27B Q4_K_M (dense), `llama-batched-bench`, S_TG: npl 4 **−13 %** (37.7 → 32.7 t/s), npl 6
  ±0, npl 8 **+20 %** (44.5 → 53.2). Qwen3.6-35B-A3B IQ4_XS (MoE): no change beyond noise.
  Log: [`e2e-2026-09-19/issues2-20260919-0053.log`](messungen/2026-09-25/hip-mmvq-28613/e2e-2026-09-19/issues2-20260919-0053.log).
- End-to-end sweep requested by a reviewer: 16 types of LFM2.5-8B-A1B, `llama-bench -p 1024 -ub 4,8,128,256,512 -d 32768`,
  `--pure` quants. The PR gains nothing beyond noise. It loses up to 15 % where it switches early
  (Q3_K ub 4 −15 %, Q2_K ub 8 −12 %). Table:
  [`e2e-2026-09-19/sweep28613-vergleich.md`](messungen/2026-09-25/hip-mmvq-28613/e2e-2026-09-19/sweep28613-vergleich.md).

## The question on the PR

A reviewer found the same switch slower on a narrow matrix and faster on a wide one (q2_K, K = 5120,
ne11 = 8, on his GPU: ne01 = 1024 +27 %, ne01 = 17408 −13 %). He asked whether a simple rule
explains it. On gfx1151 the two shapes give
([`simon-two-shapes.txt`](messungen/2026-09-25/hip-mmvq-28613/simon-two-shapes.txt)):

| shape | master | PR | |
|---|---|---|---|
| ne01 = 1024 | 40.5 µs | 91.2 µs | **2.3× slower** |
| ne01 = 17408 | 562.6 µs | 563.0 µs | unchanged |

## Width sweep

`test-backend-ops perf -o MUL_MAT -b ROCm0 --test-file`, K = 5120, 6 types × ne01 ∈ {512, 1024,
2048, 4096, 8192, 17408} × ne11 1–8 = 288 shapes
([`width-sweep.txt`](messungen/2026-09-25/hip-mmvq-28613/width-sweep/width-sweep.txt)).

**Method.**
- 6 rounds; each round runs every shape on both builds. The order alternates ABBA (master first
  in odd rounds, PR first in even ones), so drift hits both builds alike.
- Before each block a guard waited until the 1-minute load was below 1.5 and no foreign process
  used more than 30 % CPU. A monitor logged foreign CPU load every 2 s from the second
  block on; the maximum during any block was 4 % of one core. For the first block, the load log shows
  only the benchmark itself among the top processes ([`waechter.log`](messungen/2026-09-25/hip-mmvq-28613/width-sweep/waechter.log),
  [`monitor2.log`](messungen/2026-09-25/hip-mmvq-28613/width-sweep/monitor2.log)).
  Reason: an unrelated simulation had run on the machine during an earlier, discarded attempt.
- Per shape: the PR/master time ratio of each round, the median over the 6 rounds, a 95 % bootstrap
  interval, and an exact two-sided Wilcoxon signed-rank test (n = 6, smallest possible p = 0.031)
  ([`auswerten.py`](messungen/2026-09-25/hip-mmvq-28613/width-sweep/auswerten.py)).
- **Null control:** at ne11 = 1 both builds run the identical kernel. There, 2 of 36 shapes still
  pass the significance test, with 6 % and 9 %. **Effects below about 10 % should therefore not be
  read as real.** Everything the conclusions rest on is far larger.

**Result.** Cells show the median time ratio PR / master (> 1 = PR slower). \* = 95 % interval
excludes 1, Wilcoxon p ≤ 0.05 and |effect| ≥ 5 %. Bold = significant and at least 10 % slower.

**q4_0** (PR: MMVQ up to ne11 = 6, master: up to 8)

| ne01 \ ne11 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|
| 512 | 1.02 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | **2.40**\* | **2.08**\* |
| 1024 | 1.00 | 1.00 | 1.00 | 1.00 | 1.01 | 1.01 | **1.49**\* | **1.48**\* |
| 2048 | 1.00 | 0.98 | **1.12**\* | 0.92 | 0.99 | 1.00 | **1.42**\* | **1.24**\* |
| 4096 | 1.00 | 0.99 | 0.99 | 1.04 | 1.01 | 1.00 | **1.11**\* | 0.97 |
| 8192 | 1.01 | 1.01 | 0.99 | 0.99 | 1.00 | 1.01 | **1.10**\* | 0.98 |
| 17408 | 0.99 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.99 | 0.90\* |

**q8_0** (PR: MMVQ up to ne11 = 4, master: up to 8)

| ne01 \ ne11 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|
| 512 | 0.99 | 1.00 | 1.00 | 1.00 | **2.39**\* | **2.24**\* | **2.03**\* | **1.76**\* |
| 1024 | 0.98 | 0.98 | 0.97 | 0.97 | **1.69**\* | **1.50**\* | **1.35**\* | **1.19**\* |
| 2048 | 0.97 | 0.98 | 0.98 | 0.98 | **1.30**\* | **1.13**\* | 1.00 | 0.89\* |
| 4096 | 0.97 | 1.01 | 1.01 | 1.00 | **1.11**\* | 0.97 | 0.85\* | 0.75\* |
| 8192 | 1.01 | 1.02 | 1.00 | 1.00 | 1.02 | 1.00 | 1.01 | 0.97 |
| 17408 | 1.00 | 1.00 | 1.01 | 1.00 | 1.06\* | 1.05\* | 1.04 | 1.03 |

**q2_K** (PR: MMVQ up to ne11 = 4, master: up to 8)

| ne01 \ ne11 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|
| 512 | 1.00 | 1.00 | 1.00 | 1.00 | **5.11**\* | **4.45**\* | **3.88**\* | **3.42**\* |
| 1024 | 1.00 | 0.99 | 0.96 | 1.00 | **3.01**\* | **2.61**\* | **2.39**\* | **2.12**\* |
| 2048 | 0.94\* | 0.98 | 0.98 | 1.02 | **2.08**\* | **1.85**\* | **1.67**\* | **1.42**\* |
| 4096 | 0.91\* | 0.99 | 0.99 | 0.99 | **1.45**\* | **1.30**\* | **1.15**\* | 1.03 |
| 8192 | 0.96 | 1.01 | 1.02 | 0.98 | **1.51**\* | **1.27**\* | 1.09\* | 0.97 |
| 17408 | 1.00 | 1.00 | 1.01 | 0.98 | **1.50**\* | **1.24**\* | 1.05 | 0.94\* |

**q3_K** (PR: MMVQ up to ne11 = 3, master: up to 8)

| ne01 \ ne11 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|
| 512 | 1.00 | 1.00 | 0.99 | **5.66**\* | **4.89**\* | **4.45**\* | **3.97**\* | **3.58**\* |
| 1024 | 0.99 | 0.96 | 0.97 | **3.20**\* | **2.86**\* | **2.51**\* | **2.30**\* | **1.96**\* |
| 2048 | 1.02 | 0.97 | 0.98 | **2.23**\* | **1.89**\* | **1.79**\* | **1.67**\* | **1.38**\* |
| 4096 | 0.95 | 1.00 | 0.98 | **1.73**\* | **1.50**\* | **1.33**\* | **1.21**\* | 1.08\* |
| 8192 | 0.98 | 0.99 | 0.99 | **1.79**\* | **1.50**\* | **1.34**\* | **1.22**\* | 1.09\* |
| 17408 | 1.01 | 1.03 | 1.00 | **1.84**\* | **1.56**\* | **1.38**\* | **1.15**\* | 1.01 |

**q4_K** (PR: MMVQ up to ne11 = 2, master: up to 8)

| ne01 \ ne11 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|
| 512 | 1.00 | 1.02 | **2.84**\* | **2.51**\* | **2.22**\* | **2.00**\* | **1.79**\* | **1.54**\* |
| 1024 | 1.00 | 0.96 | **1.98**\* | **1.70**\* | **1.47**\* | **1.31**\* | **1.19**\* | 1.06\* |
| 2048 | 0.98 | 0.98 | **1.25**\* | **1.16**\* | 0.94\* | 0.82\* | 0.69\* | 0.65\* |
| 4096 | 0.99 | 0.99 | 1.00 | 0.83\* | 0.68\* | 0.60\* | 0.53\* | 0.47\* |
| 8192 | 1.04 | 1.01 | 1.08 | 0.87\* | 0.74\* | 0.65\* | 0.57\* | 0.50\* |
| 17408 | 1.03 | 0.99 | **1.23**\* | **1.16**\* | 1.01 | 0.88\* | 0.77\* | 0.67\* |

**q6_K** (PR: MMVQ up to ne11 = 3, master: up to 8)

| ne01 \ ne11 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|
| 512 | 1.00 | 1.00 | 0.99 | **2.82**\* | **2.51**\* | **2.21**\* | **2.01**\* | **1.81**\* |
| 1024 | 0.96 | 0.96 | 0.95\* | **1.76**\* | **1.55**\* | **1.44**\* | **1.25**\* | **1.14**\* |
| 2048 | 0.97 | 1.06 | 0.99 | **1.23**\* | 1.07\* | 0.95 | 0.85\* | 0.76\* |
| 4096 | 0.99 | 0.99 | 1.08 | 1.09\* | 0.95 | 0.85\* | 0.76\* | 0.70\* |
| 8192 | 0.98 | 0.98 | 1.00 | **1.15**\* | 0.96 | 0.84\* | 0.76\* | 0.71\* |
| 17408 | 1.01 | 1.00 | 1.01 | 1.08\* | 1.02 | 0.93\* | 0.83\* | 0.77\* |

## Reading

1. **Narrow matrices lose badly.** At ne01 ≤ 1024 every early switch to MMQ is slower, by 1.06× to
   5.7×. Every type is affected. The loss is largest right after the switch point and shrinks toward
   ne11 = 8.
2. **Q2_K and Q3_K lose at every width.** For these two types the PR's thresholds are too low on
   gfx1151 even at ne01 = 17408 (Q3_K ne11 = 4: 1.84×, Q2_K ne11 = 5: 1.50×). The only gain is
   Q2_K at ne11 = 8 on the widest matrices (0.94–0.97).
3. **Q4_K and Q6_K gain on wide matrices, Q8_0 only on medium ones.** For Q4_K and Q6_K, MMQ is faster
   from ne01 ≈ 2048 and ne11 ≈ 5–7 on. Q8_0 gains only at ne01 = 2048–4096 and ne11 ≥ 7 (down to
   0.75) and stays neutral at 8192 and 17408.
   Q4_K reaches 0.47 (2.1× faster) at ne01 = 4096, ne11 = 8. This matches the +20 % at npl 8 on the
   dense 27B model. Its FFN matrices are 17408 wide.
4. **The rule is width, not type alone.** A plausible mechanism is occupancy. MMQ tiles the output
   rows, so a narrow matrix yields only a few workgroups for 40 CUs, while MMVQ spreads the same rows
   more finely. This mechanism was not measured. Empirically, on this chip within ne11 ≤ 8, MMQ only pays off
   when ne01 ≳ 2048, and only for Q4_K, Q6_K and (partly) Q8_0.
5. **This also explains the end-to-end −13 % at npl 4.** The dense 27B model's K/V projections are
   narrow (ne01 = 1024), and they pay the switch in every layer.

A table keyed on type alone cannot capture this. A condition on ne01, for example keeping MMVQ for
ne01 < 2048 and for Q2_K/Q3_K entirely, would keep the gains and avoid the losses on this chip. It
was not implemented or measured here.

## Side check: does Vulkan have the same problem?

No. Vulkan (`ggml-vulkan.cpp`) uses the matrix-vector kernel for every ne11 ≤ 8, regardless of type
or width. The opposite question is whether switching earlier to the matrix kernel would help
speculative decoding: MTP verifies 4–5 tokens per step. On the real Qwen3.8-27B shapes (our serving
build, `test-backend-ops`, 3 runs) the time at ne11 = 5 relative to ne11 = 1 is:

| matrix | share of weight bytes | t(5) / t(1) |
|---|---|---|
| ffn_gate / ffn_up (Q4_K, 5120 → 17408) | largest | 1.03 |
| ffn_down (Q6_K, 17408 → 5120) | large | 1.06 |
| attn_qkv (Q6_K, 5120 → 10240) | medium | 1.17 |

These three matrices hold about 80 % of the weights, and they already run at 218–257 GB/s at ne11 = 1,
which is the bandwidth limit. Smaller matrices show ratios up to 2.2×, but they fit in the 32 MB MALL
during the benchmark (up to 734 GB/s), which a real decode does not do. Even a perfect kernel would
save at most about 5 % of matmul time, realistically 0–3 % end-to-end, so we did not pursue it. Data:
[`vulkan-counterpart/analyse.txt`](messungen/2026-09-25/hip-mmvq-28613/vulkan-counterpart/analyse.txt).
