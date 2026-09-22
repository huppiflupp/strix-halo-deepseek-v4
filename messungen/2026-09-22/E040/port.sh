#!/bin/bash
# E040 part 2: fork's tiled concat ported to the PR tree (GGML_VK_LAB_CONCAT_T=1). Correctness first, then speed. Qwen3.6 + coder model.
cd "$(dirname "$0")"
S="llama-gptoss.socket llama-coder.socket llama-qwen.socket llama-qwen36.socket llama-qwen36-familie.socket"
systemctl --user stop $S llama-gptoss llama-coder llama-qwen llama-qwen36 llama-qwen36-familie
trap 'systemctl --user start $S' EXIT
. ../../harness/guard.sh
B=$HOME/llama-work/pr27952-27703/build-vulkan/bin; W=$HOME/bench/llamacpp-tuning/wiki.test.raw
Q=$HOME/models/Qwen3.6-35B-A3B-MTP/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf; K=$HOME/models/Qwen3.8-27B/Qwen3.8-27B-Q4_K_M.gguf
export GGML_VK_LAB_FA_WAVE32=1 GGML_VK_LAB_FA_VT=1
T=/tmp/claude-1000/-home-seeas/427f6ada-002c-402d-9339-95dcfca48278/scratchpad
fmt() { grep -E '^\| *qwen|rror' | awk -F'|' '{print "  " $(NF-2) "|" $(NF-1) " tok/s"}'; }
{
echo "== op tests CONCAT with the switch"; wait_for_memory
GGML_VK_LAB_CONCAT_T=1 $B/test-backend-ops test -o CONCAT -b Vulkan0 2>&1 | tail -3
echo "== Qwen3.6 KLD: switch on vs. switch off (same build)"; wait_for_memory; sleep 20
$B/llama-perplexity -m $Q -f $W -c 2048 -ub 2048 --chunks 8 -ngl 999 -fa on --kl-divergence-base $T/e040.kld 2>&1 | grep -oE 'Final estimate.*'
wait_for_memory; sleep 20
GGML_VK_LAB_CONCAT_T=1 $B/llama-perplexity -m $Q -f $W -c 2048 -ub 2048 --chunks 8 -ngl 999 -fa on --kl-divergence-base $T/e040.kld --kl-divergence 2>&1 | grep -E 'Mean PPL\(Q\) |Mean +KLD|Maximum KLD|Same top p' | sed 's/^ */   /'
rm -f $T/e040.kld
for v in X=1 GGML_VK_LAB_CONCAT_T=1; do wait_for_memory; sleep 20; echo "== Qwen3.6 speed, $v"
  env $v $B/llama-bench -m $Q -ngl 999 -fa 1 -ub 2048 -p 512,2048 -n 32 -d 0,16384 -r 3 -o md 2>&1 | fmt; done
for v in X=1 GGML_VK_LAB_CONCAT_T=1; do wait_for_memory; sleep 20; echo "== Qwen3.8-27B (coder model, packed matmul, -ub 512), $v"
  env $v GGML_VK_LAB_MM_PACKED=2 $B/llama-bench -m $K -ngl 999 -fa 1 -ub 512 -p 512,2048 -n 0 -r 3 -o md 2>&1 | fmt; done
echo "BO_VA lines: $(journalctl -k -b --no-pager | grep -c BO_VA)"; echo done
} > port-result.txt 2>&1
