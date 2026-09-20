#!/bin/bash
# Vor dem Familieneinsatz: rechnet der Vulkan-Quelltext-Bau gpt-oss-120b richtig?
# Pruefung ueber zwei unabhaengige Backends -- stimmen Vulkan (Fork) und HIP (Upstream)
# ueberein, ist ein stiller Rechenfehler auf einer Seite praktisch ausgeschlossen.
M=$HOME/models/gpt-oss-120b/gpt-oss-120b-MXFP4.gguf
W=$HOME/bench/llamacpp-tuning/wiki.test.raw
L=$HOME/bench/v4-gegen-qwen/gptoss-ppl.log
: > "$L"
echo -n "Vulkan, Fork-Quelltext-Bau, -ub 2048: " | tee -a "$L"
$HOME/src/strix-llama/build-vulkan/bin/llama-perplexity -m "$M" -f "$W" -c 2048 -ub 2048 --chunks 10 -ngl 999 -fa 1 2>&1 | grep -oE 'Final estimate.*' | tee -a "$L"
sleep 10
echo -n "HIP, Upstream-Bau:                    " | tee -a "$L"
$HOME/src/llamacpp-upstream/build-hip/bin/llama-perplexity -m "$M" -f "$W" -c 2048 -ub 2048 --chunks 10 -ngl 999 -fa 1 2>&1 | grep -oE 'Final estimate.*' | tee -a "$L"
echo FERTIG >> "$L"
