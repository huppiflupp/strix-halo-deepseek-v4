#!/bin/bash
# E040: why is Qwen3.6-35B-A3B 19-21 % slower at 2048-token prompts on the new build than on the fork (E034)? Per-op profile of both.
cd "$(dirname "$0")"
while systemctl --user is-active -q lab-e038b || systemctl --user is-active -q lab-e039m; do sleep 20; done
S="llama-gptoss.socket llama-coder.socket llama-qwen.socket llama-qwen36.socket llama-qwen36-familie.socket"
systemctl --user stop $S llama-gptoss llama-coder llama-qwen llama-qwen36 llama-qwen36-familie
trap 'systemctl --user start $S' EXIT
M=$HOME/models/Qwen3.6-35B-A3B-MTP/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf
{
../../harness/profile-ops.sh $HOME/strix-fork/vulkan $M qwen36.fork 2048
GGML_VK_LAB_FA_WAVE32=1 GGML_VK_LAB_FA_VT=1 ../../harness/profile-ops.sh $HOME/llama-serve/gptoss/build-vulkan/bin $M qwen36.new 2048
echo done
} > result.txt 2>&1
