#!/bin/bash
# Lucebox dflash_server bauen, Flags wie im Blogbeitrag, angepasst:
#   - CMAKE_HIP_COMPILER auf den Fedora-Pfad (/opt/rocm gibt es hier nicht)
#   - ROCm ist 7.1.1 statt ihrer 7.2.4; ob ihre gfx1151-Kernel damit uebersetzen, ist offen
set -uo pipefail
cd ~/src/lucebox
git log -1 --format='Lucebox-Stand: %h %ad %s' --date=short
cmake -S server -B server/build-hip -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_HIP_COMPILER=/usr/lib64/rocm/llvm/bin/clang++ \
  -DDFLASH27B_GPU_BACKEND=hip \
  -DDFLASH27B_HIP_ARCHITECTURES=gfx1151 \
  -DDFLASH27B_HIP_SM80_EQUIV=ON \
  -DCMAKE_HIP_FLAGS=-DDFLASH_WAVE_SIZE=32 \
  -DGGML_HIP_MMQ_MFMA=ON \
  -DGGML_HIP_NO_VMM=ON \
  -DGGML_HIP_GRAPHS=OFF 2>&1 | tail -15
echo "== Bauen"
cmake --build server/build-hip --target dflash_server -j "$(nproc)" 2>&1 | tail -20
ls -la server/build-hip/dflash_server 2>/dev/null && echo "BAU OK" || echo "BAU FEHLGESCHLAGEN"
