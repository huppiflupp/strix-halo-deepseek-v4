#!/bin/bash
# E040b: switch the fork's own levers off one by one (Qwen3.6, 2048-token prompt) to see which one carries its lead.
cd "$(dirname "$0")"
while systemctl --user is-active -q lab-e040; do sleep 20; done
S="llama-gptoss.socket llama-coder.socket llama-qwen.socket llama-qwen36.socket llama-qwen36-familie.socket"
systemctl --user stop $S llama-gptoss llama-coder llama-qwen llama-qwen36 llama-qwen36-familie
trap 'systemctl --user start $S' EXIT
. ../../harness/guard.sh
M=$HOME/models/Qwen3.6-35B-A3B-MTP/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf
{
for v in "" "LLAMA_MOE_F16=0"; do wait_for_memory; echo "== fork $v"
  env $v $HOME/strix-fork/vulkan/llama-bench -m $M -ngl 999 -fa 1 -ub 2048 -p 2048 -n 0 -r 3 -o md 2>&1 | grep -E '^\| *qwen' | awk -F'|' '{print "  " $(NF-2) "|" $(NF-1) " tok/s"}'; done
echo done
} > ab-result.txt 2>&1
