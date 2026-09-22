#!/bin/bash
# E044 step 0: perplexity + KLD of the new frozen build ~/llama-serve/hybrid on Qwen3.6, switch off / on, against the ROCm backend.
cd "$(dirname "$0")"
S="llama-gptoss.socket llama-coder.socket llama-qwen.socket llama-qwen36.socket llama-qwen36-familie.socket"
systemctl --user stop $S llama-gptoss llama-coder llama-qwen llama-qwen36 llama-qwen36-familie
T=/tmp/claude-1000/-home-seeas/427f6ada-002c-402d-9339-95dcfca48278/scratchpad
trap 'systemctl --user start $S; rm -f $T/e044.kld' EXIT
. ../../harness/guard.sh
B=$HOME/llama-serve/hybrid/build-vulkan/bin; H=$HOME/src/llamacpp-upstream/build-hip/bin
Q=$HOME/models/Qwen3.6-35B-A3B-MTP/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf
A="-m $Q -f $HOME/bench/llamacpp-tuning/wiki.test.raw -c 2048 -ub 2048 --chunks 20 -ngl 999 -fa on"
export GGML_VK_LAB_FA_WAVE32=1 GGML_VK_LAB_FA_VT=1
{
wait_for_memory; echo "== ROCm reference"; $H/llama-perplexity $A --kl-divergence-base $T/e044.kld 2>&1 | grep -oE 'Final estimate.*'; sleep 60
for v in X=1 GGML_VK_LAB_CONCAT_T=1; do wait_for_memory; sleep 20; echo "== new build, $v, vs ROCm"
  env $v $B/llama-perplexity $A --kl-divergence-base $T/e044.kld --kl-divergence 2>&1 | grep -E 'Mean PPL\(Q\) |Mean +KLD|Maximum KLD|Same top p' | sed 's/^ */   /'; done
echo "BO_VA lines: $(journalctl -k -b --no-pager | grep -c BO_VA)"; echo done
} > ppl-result.txt 2>&1
