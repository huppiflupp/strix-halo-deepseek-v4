#!/bin/bash
# Welcher Fork-Kniff traegt den Prefill-Vorsprung? Jeder Schalter einzeln aus,
# Rest an (Ablation). Grundlage: Fork 1572 t/s gegen Upstream 1293 t/s (pp512,
# Qwen3.6-35B-A3B IQ4_XS). Wenn ein einzelner Schalter den Unterschied traegt,
# lohnt es, genau diesen Commit auf aktuelles Upstream zu heben.
set -uo pipefail
FO=$HOME/src/strix-llama
UP=$HOME/src/llamacpp-upstream
M=$HOME/models/Qwen3.6-35B-A3B-MTP/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf
L=$HOME/bench/v4-gegen-qwen/ablation.log
: > "$L"
messe() {  # $1 Etikett, Rest: Umgebungsvariablen
    local etikett="$1"; shift
    local wert
    wert=$(env "$@" "$FO/build-vulkan/bin/llama-bench" -m "$M" -fa 1 -p 512 -n 128 -r 2 2>/dev/null \
           | grep -E 'pp512' | grep -oE '[0-9]+\.[0-9]+ ±' | head -1 | tr -d ' ±')
    local tg
    tg=$(env "$@" "$FO/build-vulkan/bin/llama-bench" -m "$M" -fa 1 -p 0 -n 128 -r 2 2>/dev/null \
           | grep -E 'tg128' | grep -oE '[0-9]+\.[0-9]+ ±' | head -1 | tr -d ' ±')
    printf "%-34s pp512 %8s   tg128 %7s\n" "$etikett" "${wert:-?}" "${tg:-?}" | tee -a "$L"
}
echo "== Fork, Schalter einzeln abgeschaltet (Rest an)" | tee -a "$L"
messe "alle an (Fork-Vorgabe)"
for s in GGML_VK_MMID_ROWLISTS GGML_VK_MMID_SMALLN GGML_VK_MMID_BM64 GGML_VK_MMID_WAVE32 \
         GGML_VK_MMID_F16B GGML_VK_MMID_M128 GGML_VK_FA_WAVE32 GGML_VK_FA_KV_CONTIG; do
    messe "ohne $s" "$s=0"
done
messe "alle MMID aus" GGML_VK_MMID_ROWLISTS=0 GGML_VK_MMID_SMALLN=0 GGML_VK_MMID_BM64=0 \
      GGML_VK_MMID_WAVE32=0 GGML_VK_MMID_F16B=0 GGML_VK_MMID_M128=0
messe "alles aus" GGML_VK_MMID_ROWLISTS=0 GGML_VK_MMID_SMALLN=0 GGML_VK_MMID_BM64=0 \
      GGML_VK_MMID_WAVE32=0 GGML_VK_MMID_F16B=0 GGML_VK_MMID_M128=0 GGML_VK_FA_WAVE32=0 GGML_VK_FA_KV_CONTIG=0
echo "-- zum Vergleich Upstream:" | tee -a "$L"
"$UP/build-vulkan/bin/llama-bench" -m "$M" -fa 1 -p 512 -n 128 -r 2 2>/dev/null | grep -E 'pp512|tg128' | tee -a "$L"
echo "FERTIG" | tee -a "$L"
