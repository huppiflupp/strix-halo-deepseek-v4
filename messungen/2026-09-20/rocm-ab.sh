#!/bin/bash
# ROCm 7.1.1 (Fedora) gegen 7.2.1 (Ollama-Buendel), gleicher HIP-Bau von llama.cpp.
#
# Getauscht wird nur die HIP-Laufzeit (libamdhip64, libhsa-runtime64, libamd_comgr) per
# LD_PRELOAD. Das System-rocBLAS bleibt, weil das Buendel keine gfx1151-Kernel mitbringt
# (nur gfx1030/1100/1101) -- ein vollstaendiger Tausch wuerde also nicht die ROCm-Version
# messen, sondern das Fehlen der passenden Kernel.
#
# Gemessen wird ein kleines Modell, damit der Lauf Minuten statt Stunden dauert; fuer die
# Frage "aendert die ROCm-Version etwas an llama.cpp" reicht das, und der 97-GiB-Lauf
# waere bei knappem Speicher ohnehin stoeranfaellig.
set -uo pipefail
B=$HOME/src/strix-llama/build-hip/bin
M=${1:-$HOME/models/aufgaben/Qwen3-1.7B-Q8_0.gguf}
BUENDEL=/usr/local/lib/ollama/rocm_v7_2
P="$BUENDEL/libamdhip64.so.7:$BUENDEL/libhsa-runtime64.so.1:$BUENDEL/libamd_comgr.so.3"

echo "== Modell: $(basename "$M")"
echo "-- ROCm 7.1.1 (System)"
"$B/llama-bench" -m "$M" -fa 1 -p 512 -n 128 -r 3 2>/dev/null | grep -E 'pp512|tg128'
echo "-- ROCm 7.2.1 (Ollama-Buendel, nur HIP-Laufzeit)"
LD_PRELOAD="$P" "$B/llama-bench" -m "$M" -fa 1 -p 512 -n 128 -r 3 2>/dev/null | grep -E 'pp512|tg128'
echo
echo "== Korrektheit (Perplexitaet, 10 Bloecke) -- auf gfx1151 rechnete HIP bis September still falsch"
echo "-- ROCm 7.1.1"
"$B/llama-perplexity" -m "$M" -f $HOME/bench/llamacpp-tuning/wiki.test.raw -c 2048 --chunks 10 -ngl 999 -fa 1 2>&1 | grep -oE 'Final estimate.*'
echo "-- ROCm 7.2.1"
LD_PRELOAD="$P" "$B/llama-perplexity" -m "$M" -f $HOME/bench/llamacpp-tuning/wiki.test.raw -c 2048 --chunks 10 -ngl 999 -fa 1 2>&1 | grep -oE 'Final estimate.*'
echo "-- Vulkan zum Vergleich (gleiche Bloecke)"
/home/seeas/src/strix-llama/build-vulkan/bin/llama-perplexity -m "$M" -f $HOME/bench/llamacpp-tuning/wiki.test.raw -c 2048 --chunks 10 -ngl 999 -fa 1 2>&1 | grep -oE 'Final estimate.*'
