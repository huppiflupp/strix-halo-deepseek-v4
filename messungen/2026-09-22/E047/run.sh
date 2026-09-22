#!/bin/bash
# E047: attention phase split IN PRODUCTION CONFIGURATION (the review's X0; replaces E022's test-program split).
# gpt-oss-120b in the model: real causal mask, strided KV -> contiguous copy (#27703), wave32 + transposed V, 2048-token prompt on 16 k.
# LAB_SKIP is compile-time: build in place and run, one value after the other (copied bin dirs mixed two libggml copies - assert).
# Timing only - results are wrong when LAB_SKIP != 0.
cd "$(dirname "$0")"
P=$HOME/llama-work/pr27952-27703; SH=$P/ggml/src/ggml-vulkan/vulkan-shaders/flash_attn_cm1.comp
S="llama-gptoss.socket llama-coder.socket llama-qwen.socket llama-qwen36.socket llama-qwen36-familie.socket"
systemctl --user stop $S llama-gptoss llama-coder llama-qwen llama-qwen36 llama-qwen36-familie
restore() { sed -i -E "s/^#define LAB_SKIP [0-9]+/#define LAB_SKIP 0/" $SH; (cd $P && cmake --build build-vulkan -j 16 --target llama-bench llama-perplexity >/dev/null 2>&1); systemctl --user start $S; }
trap restore EXIT
. ../../harness/guard.sh
G=$HOME/models/gpt-oss-120b/gpt-oss-120b-MXFP4.gguf
export GGML_VK_LAB_FA_WAVE32=1 GGML_VK_LAB_FA_VT=1
{
for s in 0 3 44 16 63; do
  sed -i -E "s/^#define LAB_SKIP [0-9]+/#define LAB_SKIP $s/" $SH
  (cd $P && cmake --build build-vulkan -j 16 --target llama-bench 2>&1 | grep -E ' error' | head -3)
  wait_for_memory; sleep 25
  GGML_VK_PERF_LOGGER=1 $P/build-vulkan/bin/llama-bench -m $G -ngl 999 -fa 1 -ub 2048 -b 2048 -p 2048 -n 0 -d 16384 -r 1 -o json 2> perf-skip$s.log > bench-skip$s.json
  echo "skip=$s: $(grep -E '^FLASH_ATTN_EXT' perf-skip$s.log | head -3 | tr '\n' ' ') | $(grep -c GGML_ASSERT perf-skip$s.log) asserts"
done
echo "BO_VA lines: $(journalctl -k -b --no-pager | grep -c BO_VA)"; echo done
} > result.txt 2>&1
