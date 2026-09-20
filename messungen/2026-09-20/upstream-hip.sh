#!/bin/bash
# Gegenprobe mit Upstream-llama.cpp: rechnet HIP auf gfx1151 auch dort falsch?
# Befund 2026-09-20 mit dem Fork (Nathanw1014, 50c271f8e): Qwen3-1.7B Q8_0 ergibt unter HIP
# Perplexitaeten zwischen 24.000 und 290.000, dieselbe Datei auf CPU und Vulkan 16,5.
# Nicht Flash Attention (auch mit -fa 0), nicht die Matmul-Wahl (FORCE_MMQ und FORCE_CUBLAS
# gleichermassen kaputt), Werte schwanken zwischen Laeufen.
set -euo pipefail
Z=$HOME/src/llamacpp-upstream
[ -d "$Z" ] || git clone --depth 1 https://github.com/ggml-org/llama.cpp "$Z"
cd "$Z"
git log -1 --format='Upstream-Stand: %h %ad %s' --date=short
cmake -S . -B build-hip -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DGGML_HIP=ON -DAMDGPU_TARGETS=gfx1151 \
  -DCMAKE_HIP_COMPILER=/usr/lib64/rocm/llvm/bin/clang++ \
  -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF > /dev/null
cmake --build build-hip --target llama-perplexity llama-bench -j "$(nproc)" 2>&1 | tail -3
M=$HOME/models/aufgaben/Qwen3-1.7B-Q8_0.gguf
W=$HOME/bench/llamacpp-tuning/wiki.test.raw
echo "== Upstream HIP, GPU:"
./build-hip/bin/llama-perplexity -m "$M" -f "$W" -c 2048 --chunks 5 -fa 1 -ngl 999 2>&1 | grep -oE 'Final estimate.*'
echo "== Upstream HIP, CPU (Gegenprobe):"
./build-hip/bin/llama-perplexity -m "$M" -f "$W" -c 2048 --chunks 5 -fa 1 -ngl 0 --device none -t 16 2>&1 | grep -oE 'Final estimate.*'
