#!/bin/bash
# E049b: correctness of wave32 + VT on the second model (Qwen3-30B-A3B, head 128) against the ROCm backend. 3 model loads.
cd "$(dirname "$0")"
S="llama-gptoss.socket llama-coder.socket llama-qwen.socket llama-qwen36.socket llama-qwen36-familie.socket"
systemctl --user stop $S llama-gptoss llama-coder llama-qwen llama-qwen36 llama-qwen36-familie
T=/tmp/claude-1000/-home-seeas/427f6ada-002c-402d-9339-95dcfca48278/scratchpad
trap 'systemctl --user start $S; rm -f $T/e049.kld' EXIT
. ../../harness/guard.sh
B=$HOME/llama-serve/gptoss/build-vulkan/bin; H=$HOME/src/llamacpp-upstream/build-hip/bin
A="-m $HOME/models/Qwen3-30B-A3B-2507/Qwen3-30B-A3B-Instruct-2507-IQ4_XS.gguf -f $HOME/bench/llamacpp-tuning/wiki.test.raw -c 8192 -ub 2048 --chunks 6 -ngl 999 -fa on"
{
wait_for_memory; sleep 20; echo "== ROCm reference"; $H/llama-perplexity $A --kl-divergence-base $T/e049.kld 2>&1 | grep -oE 'Final estimate.*'; sleep 60
for v in "base|X=1" "both|GGML_VK_LAB_FA_WAVE32=1 GGML_VK_LAB_FA_VT=1"; do wait_for_memory; sleep 25; echo "== Vulkan ${v%%|*} vs ROCm"
  env ${v#*|} $B/llama-perplexity $A --kl-divergence-base $T/e049.kld --kl-divergence 2>&1 | grep -E 'Mean PPL\(Q\) |Mean +KLD|99.9%   KLD|Same top p' | sed 's/^ */   /'; done
echo "BO_VA lines: $(journalctl -k -b --no-pager | grep -c BO_VA)"; echo done
} > kld-result.txt 2>&1
