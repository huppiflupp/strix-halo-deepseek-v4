#!/bin/bash
# E045 (night 2026-09-22): column-major B loads of the packed matmul in the real coder model (E039's open in-model test).
# Correctness first (KLD against the same build without the switch), then speed. 4 model loads.
cd "$(dirname "$0")"
S="llama-gptoss.socket llama-coder.socket llama-qwen.socket llama-qwen36.socket llama-qwen36-familie.socket"
systemctl --user stop $S llama-gptoss llama-coder llama-qwen llama-qwen36 llama-qwen36-familie
T=/tmp/claude-1000/-home-seeas/427f6ada-002c-402d-9339-95dcfca48278/scratchpad
trap 'systemctl --user start $S; rm -f $T/e045.kld' EXIT
. ../../harness/guard.sh
B=$HOME/llama-work/lab/build-vulkan/bin; K=$HOME/models/Qwen3.8-27B/Qwen3.8-27B-Q4_K_M.gguf
A="-m $K -f $HOME/bench/llamacpp-tuning/wiki.test.raw -c 2048 -ub 2048 --chunks 8 -ngl 999 -fa on"
fmt() { grep -E '^\| *qwen|rror' | awk -F'|' '{print "  " $(NF-2) "|" $(NF-1) " tok/s"}'; }
{
wait_for_memory; sleep 20; GGML_VK_LAB_MM_PACKED=2 $B/llama-perplexity $A --kl-divergence-base $T/e045.kld 2>&1 | grep -oE 'Final estimate.*'
wait_for_memory; sleep 20; echo "== KLD, BCOL on vs off"
GGML_VK_LAB_MM_PACKED=2 GGML_VK_LAB_MM_BCOL=1 $B/llama-perplexity $A --kl-divergence-base $T/e045.kld --kl-divergence 2>&1 | grep -E 'Mean PPL\(Q\) |Mean +KLD|Maximum KLD|Same top p' | sed 's/^ */   /'
for v in X=1 GGML_VK_LAB_MM_BCOL=1; do wait_for_memory; sleep 20; echo "== speed $v"
  env GGML_VK_LAB_MM_PACKED=2 $v $B/llama-bench -m $K -ngl 999 -fa 1 -ub 512 -b 512 -p 512,2048 -n 0 -r 3 -o md 2>&1 | fmt; done
echo "BO_VA lines: $(journalctl -k -b --no-pager | grep -c BO_VA)"; echo done
} > result.txt 2>&1
