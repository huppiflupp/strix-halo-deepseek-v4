#!/bin/bash
# E040 (short form): per-op GPU timing of ONE phase - Qwen3.6, 2048-token prompt - on the fork and on the new build. 2 model loads.
cd "$(dirname "$0")"
S="llama-gptoss.socket llama-coder.socket llama-qwen.socket llama-qwen36.socket llama-qwen36-familie.socket"
systemctl --user stop $S llama-gptoss llama-coder llama-qwen llama-qwen36 llama-qwen36-familie
trap 'systemctl --user start $S' EXIT
. ../../harness/guard.sh
M=$HOME/models/Qwen3.6-35B-A3B-MTP/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf
wait_for_memory; sleep 20
GGML_VK_PERF_LOGGER=1 $HOME/strix-fork/vulkan/llama-bench -m $M -ngl 999 -fa 1 -b 2048 -ub 2048 -p 2048 -n 0 -r 1 --no-warmup -o json 2> fork.log > fork.json
wait_for_memory; sleep 20
GGML_VK_PERF_LOGGER=1 GGML_VK_LAB_FA_WAVE32=1 GGML_VK_LAB_FA_VT=1 $HOME/llama-serve/gptoss/build-vulkan/bin/llama-bench -m $M -ngl 999 -fa 1 -b 2048 -ub 2048 -p 2048 -n 0 -r 1 --no-warmup -o json 2> new.log > new.json
echo "BO_VA lines: $(journalctl -k -b --no-pager | grep -c BO_VA)" > status.txt
