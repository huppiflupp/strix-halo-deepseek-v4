#!/bin/bash
# llama.cpp (Fork) mit HIP bauen, entscheidend: GGML_HIP_NO_VMM=ON.
# Anlass: Lucebox baut ihren dflash_server mit genau diesem Schalter, und unser
# August-Befund war, dass HIP das 97-GiB-Modell nicht laden kann, weil es nicht an
# GTT kommt. Der VMM-Allokator ist der uebliche Grund dafuer.
# ROCm hier: 7.1.1 (Fedora-Pakete), hipcc 7.1.52802, Ziel gfx1151.
set -euo pipefail
cd ~/src/strix-llama
cmake -S . -B build-hip -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_HIP=ON \
  -DGGML_HIP_NO_VMM=ON \
  -DAMDGPU_TARGETS=gfx1151 \
  -DCMAKE_HIP_COMPILER=/usr/lib64/rocm/llvm/bin/clang++ \
  -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF
cmake --build build-hip --target llama-server llama-bench llama-perplexity -j "$(nproc)"
echo "FERTIG"
ls -la build-hip/bin/llama-{server,bench,perplexity} 2>/dev/null
