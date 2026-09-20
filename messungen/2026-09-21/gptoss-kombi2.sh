#!/bin/bash
# Gegenstueck zu gptoss-kombi.sh: derselbe q8_0-Fall auf dem Fork, damit der Vergleich fair ist.
set -uo pipefail
M=$HOME/models/gpt-oss-120b/gpt-oss-120b-MXFP4.gguf
W=$HOME/bench/llamacpp-tuning/wiki.test.raw
F=$HOME/src/strix-llama/build-vulkan/bin
L=$HOME/bench/v4-gegen-qwen/gptoss-kombi2.log
: > "$L"
"$F/llama-bench" -m "$M" -fa 1 -b 4096 -ub 4096 -ctk q8_0 -ctv q8_0 -p 2048,16384 -n 0 -r 2 2>/dev/null | grep -E '^\| *gpt-oss' | tee -a "$L"
for kv in f16 q8_0; do echo -n "PPL Fork KV $kv: " | tee -a "$L"
"$F/llama-perplexity" -m "$M" -f "$W" -c 2048 -ub 2048 --chunks 10 -ngl 999 -fa on -ctk $kv -ctv $kv 2>&1 | grep -oE 'Final estimate.*' | tee -a "$L"; done
echo FERTIG | tee -a "$L"
