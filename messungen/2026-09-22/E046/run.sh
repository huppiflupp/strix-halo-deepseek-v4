#!/bin/bash
# E046 (= E038b completed): q8_0 KV for gpt-oss-120b judged against the model's own perturbation floor.
# Raw wikitext, 8192-token windows, 6 chunks. Base = production config (f16 KV, wave32 + VT).
# Floor = only the summation order changes (wave32 off, measured 2026-09-21: KLD 0.054, 84.6 %) - re-measured here in the same run.
cd "$(dirname "$0")"
S="llama-gptoss.socket llama-coder.socket llama-qwen.socket llama-qwen36.socket llama-qwen36-familie.socket"
systemctl --user stop $S llama-gptoss llama-coder llama-qwen llama-qwen36 llama-qwen36-familie
T=/tmp/claude-1000/-home-seeas/427f6ada-002c-402d-9339-95dcfca48278/scratchpad
trap 'systemctl --user start $S; rm -f $T/e046.kld' EXIT
. ../../harness/guard.sh
B=$HOME/llama-serve/gptoss/build-vulkan/bin; O=$HOME/models/gpt-oss-120b/gpt-oss-120b-MXFP4.gguf
A="-m $O -f $HOME/bench/llamacpp-tuning/wiki.test.raw -c 8192 -ub 2048 --chunks 6 -ngl 999 -fa on"
F='Mean PPL\(Q\) |Mean +KLD|99.9%   KLD|Same top p'
export GGML_VK_LAB_FA_VT=1
run() { wait_for_memory; sleep 25; echo "== $1"; shift; env "$@" --kl-divergence-base $T/e046.kld --kl-divergence 2>&1 | grep -E "$F" | sed 's/^ */   /'; }
{
wait_for_memory; sleep 25; GGML_VK_LAB_FA_WAVE32=1 $B/llama-perplexity $A --kl-divergence-base $T/e046.kld 2>&1 | grep -oE 'Final estimate.*'
run "FLOOR: f16 KV, only the summation order differs (wave32 off)" GGML_VK_LAB_FA_WAVE32=0 $B/llama-perplexity $A
run "K=q8_0 V=q8_0" GGML_VK_LAB_FA_WAVE32=1 $B/llama-perplexity $A -ctk q8_0 -ctv q8_0
run "K=f16  V=q8_0" GGML_VK_LAB_FA_WAVE32=1 $B/llama-perplexity $A -ctk f16 -ctv q8_0
echo "BO_VA lines: $(journalctl -k -b --no-pager | grep -c BO_VA)"; echo done
} > result.txt 2>&1
