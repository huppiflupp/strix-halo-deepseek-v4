#!/bin/bash
# Der entscheidende Fall: DeepSeek-V4-Flash. Dort brachte der Fork im August +62 % Prefill
# und +53 % Decode gegenueber Mainline b10488 (128,98 / 11,96 -> 208,40 / 18,33), waehrend
# er bei Qwen3-30B den Decode gar nicht veraenderte. Wenn Upstream hier aufgeschlossen hat,
# ist der Fork entbehrlich.
set -uo pipefail
FO=$HOME/src/strix-llama
UP=$HOME/src/llamacpp-upstream
M=$HOME/models/DeepSeek-V4-Flash-0731/UD-IQ3_XXS/DeepSeek-V4-Flash-0731-UD-IQ3_XXS-00001-of-00004.gguf
W=$HOME/bench/llamacpp-tuning/wiki.test.raw
L=$HOME/bench/v4-gegen-qwen/fork-gegen-upstream-v4.log
: > "$L"
for n in Fork:$FO Upstream:$UP; do
  echo "-- ${n%%:*}, Vulkan, DeepSeek-V4-Flash IQ3_XXS" | tee -a "$L"
  LLAMA_MOE_F16=0 "${n#*:}/build-vulkan/bin/llama-bench" -m "$M" -fa 1 -p 512 -n 128 -d 0,8192 -r 2 2>/dev/null \
    | grep -E 'pp512|tg128' | tee -a "$L"
  echo -n "   Perplexitaet (10 Bloecke): " | tee -a "$L"
  LLAMA_MOE_F16=0 "${n#*:}/build-vulkan/bin/llama-perplexity" -m "$M" -f "$W" -c 2048 --chunks 10 -ngl 999 -fa 1 2>&1 \
    | grep -oE 'Final estimate.*' | tee -a "$L"
done
echo "FERTIG" | tee -a "$L"
