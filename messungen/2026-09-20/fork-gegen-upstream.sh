#!/bin/bash
# Lohnt der Fork noch? Vergleich Fork (600 Commits hinter Upstream, Basis 17.08.) gegen
# aktuelles Upstream, beide aus dem Quelltext mit System-RADV gebaut, gleiche Flags.
# Dazu die beiden HIP-Pruefungen: #28211 (falsche Logits bei Prompt > n_ubatch) und
# ROCBLAS_USE_HIPBLASLT als dokumentiert groesster Prefill-Hebel.
set -uo pipefail
FO=$HOME/src/strix-llama
UP=$HOME/src/llamacpp-upstream
L=$HOME/bench/v4-gegen-qwen/fork-gegen-upstream.log
FAM=$HOME/models/Qwen3.6-35B-A3B-MTP/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf
KLEIN=$HOME/models/aufgaben/Qwen3-1.7B-Q8_0.gguf
W=$HOME/bench/llamacpp-tuning/wiki.test.raw
: > "$L"

echo "== Baue Vulkan aus beiden Baeumen" | tee -a "$L"
for d in "$FO" "$UP"; do
  cmake -S "$d" -B "$d/build-vulkan" -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_VULKAN=ON \
        -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF > /dev/null 2>&1
  cmake --build "$d/build-vulkan" --target llama-bench llama-perplexity -j "$(nproc)" > /dev/null 2>&1 \
    && echo "  $(basename "$d"): gebaut" | tee -a "$L" || echo "  $(basename "$d"): BAU FEHLGESCHLAGEN" | tee -a "$L"
done

echo | tee -a "$L"
echo "== Familienmodell Qwen3.6-35B-A3B UD-IQ4_XS, Vulkan, -d 0 und 8192" | tee -a "$L"
for n in Fork:$FO Upstream:$UP; do
  echo "-- ${n%%:*}" | tee -a "$L"
  "${n#*:}/build-vulkan/bin/llama-bench" -m "$FAM" -fa 1 -p 512 -n 128 -d 0,8192 -r 2 2>/dev/null \
    | grep -E 'pp512|tg128' | tee -a "$L"
done

echo | tee -a "$L"
echo "== Korrektheit (Perplexitaet, 10 Bloecke)" | tee -a "$L"
for n in Fork:$FO Upstream:$UP; do
  echo -n "-- ${n%%:*}: " | tee -a "$L"
  "${n#*:}/build-vulkan/bin/llama-perplexity" -m "$FAM" -f "$W" -c 2048 --chunks 10 -ngl 999 -fa 1 2>&1 \
    | grep -oE 'Final estimate.*' | tee -a "$L"
done

echo | tee -a "$L"
echo "== HIP-Pruefung 1: Issue #28211 (falsche Logits bei Prompt > n_ubatch)" | tee -a "$L"
for ub in 512 2048; do
  echo -n "-- Fork-HIP, -ub $ub, 5 Bloecke: " | tee -a "$L"
  "$FO/build-hip/bin/llama-perplexity" -m "$KLEIN" -f "$W" -c 2048 --chunks 5 -ngl 999 -fa 1 -ub $ub 2>&1 \
    | grep -oE 'Final estimate.*' | tee -a "$L"
done
echo -n "-- Upstream-HIP, -ub 512, 5 Bloecke: " | tee -a "$L"
"$UP/build-hip/bin/llama-perplexity" -m "$KLEIN" -f "$W" -c 2048 --chunks 5 -ngl 999 -fa 1 -ub 512 2>&1 \
  | grep -oE 'Final estimate.*' | tee -a "$L"

echo | tee -a "$L"
echo "== HIP-Pruefung 2: ROCBLAS_USE_HIPBLASLT (groesster dokumentierter Prefill-Hebel)" | tee -a "$L"
for v in 0 1; do
  echo "-- ROCBLAS_USE_HIPBLASLT=$v" | tee -a "$L"
  ROCBLAS_USE_HIPBLASLT=$v "$UP/build-hip/bin/llama-bench" -m "$KLEIN" -fa 1 -p 512 -n 128 -r 2 2>/dev/null \
    | grep -E 'pp512|tg128' | tee -a "$L"
done
echo "FERTIG" | tee -a "$L"
