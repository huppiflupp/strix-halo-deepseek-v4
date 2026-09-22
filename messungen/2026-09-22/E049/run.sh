#!/bin/bash
# E049: measurement series for the upstream comment on #27703 (wave32 + transposed V): 3 repetitions, depths 0 / 8k / 16k / 32k,
# two models with head size <= 128 (gpt-oss-120b hd 64, Qwen3-30B-A3B hd 128), four variants. 8 model loads, 30 s pause each.
cd "$(dirname "$0")"
S="llama-gptoss.socket llama-coder.socket llama-qwen.socket llama-qwen36.socket llama-qwen36-familie.socket"
systemctl --user stop $S llama-gptoss llama-coder llama-qwen llama-qwen36 llama-qwen36-familie
trap 'systemctl --user start $S' EXIT
. ../../harness/guard.sh
B=$HOME/llama-serve/gptoss/build-vulkan/bin
{
for m in "gpt-oss-120b|$HOME/models/gpt-oss-120b/gpt-oss-120b-MXFP4.gguf" "Qwen3-30B-A3B|$HOME/models/Qwen3-30B-A3B-2507/Qwen3-30B-A3B-Instruct-2507-IQ4_XS.gguf"; do
  name=${m%%|*}; M=${m#*|}
  for v in "base|X=1" "VT|GGML_VK_LAB_FA_VT=1" "wave32|GGML_VK_LAB_FA_WAVE32=1" "both|GGML_VK_LAB_FA_WAVE32=1 GGML_VK_LAB_FA_VT=1"; do
    wait_for_memory; sleep 30; echo "== $name ${v%%|*}"
    env ${v#*|} $B/llama-bench -m $M -ngl 999 -fa 1 -ub 2048 -b 2048 -p 2048 -n 32 -d 0,8192,16384,32768 -r 3 -o md 2>&1 | grep -E '^\| *(gpt|qwen)|rror' | awk -F'|' '{print "  " $(NF-2) "|" $(NF-1) " tok/s"}'
  done
done
echo "BO_VA lines: $(journalctl -k -b --no-pager | grep -c BO_VA)"; echo done
} > result.txt 2>&1
