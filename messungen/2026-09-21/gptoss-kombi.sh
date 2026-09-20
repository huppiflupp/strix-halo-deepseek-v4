#!/bin/bash
# Die drei wirksamen Hebel zusammen auf dem PR-Bau: #27952 + -ub 4096 + KV q8_0.
# Einzeln gemessen: PR +31,6 % (pp2048) gegen Master, -ub 4096 +10 %, KV q8_0 +7,8 % / +15 %
# Textausgabe bei 32k / 65k Tiefe. Addieren sie sich, und was kostet q8_0 beim Prompt?
set -uo pipefail
gtt_pfad() { for u in /sys/class/drm/card*/device/uevent; do grep -qx 'DRIVER=amdgpu' "$u" 2>/dev/null || continue
             echo "$(dirname "$u")/mem_info_gtt_used"; return 0; done; echo /dev/null; }
warte() { for _ in $(seq 240); do b=$(( $(cat "$(gtt_pfad)" 2>/dev/null || echo 0) / 1024**3 ))
          f=$(free -g | awk '/^Mem:/ {print $7}'); [ "$b" -lt 10 ] && [ "$f" -gt 100 ] && return 0; sleep 5; done; exit 1; }
M=$HOME/models/gpt-oss-120b/gpt-oss-120b-MXFP4.gguf
W=$HOME/bench/llamacpp-tuning/wiki.test.raw
PR=$HOME/llama-work/pr27952/build-vulkan/bin
L=$HOME/bench/v4-gegen-qwen/gptoss-kombi.log
: > "$L"
zeilen() { grep -E '^\| *gpt-oss|^\| *model' | sed 's/gpt-oss 120B MXFP4 MoE *//'; }
for kv in f16 q8_0; do
    echo "== PR-Bau, KV $kv: Prompt bei ub 2048/4096, Ausgabe bei Tiefe 0/32768" | tee -a "$L"; warte
    "$PR/llama-bench" -m "$M" -fa 1 -b 4096 -ub 2048,4096 -ctk $kv -ctv $kv -p 2048,16384 -n 0 -r 2 2>/dev/null | zeilen | tee -a "$L"
    warte
    "$PR/llama-bench" -m "$M" -fa 1 -b 4096 -ub 4096 -ctk $kv -ctv $kv -p 0 -n 64 -d 0,32768 -r 1 2>/dev/null | zeilen | tee -a "$L"
done
echo -n "== Perplexitaet PR-Bau mit KV q8_0 (f16 ergab 454,04): " | tee -a "$L"; warte
"$PR/llama-perplexity" -m "$M" -f "$W" -c 2048 -ub 2048 --chunks 10 -ngl 999 -fa on -ctk q8_0 -ctv q8_0 2>&1 | grep -oE 'Final estimate.*' | tee -a "$L"
ls -la "$PR/llama-server" >/dev/null 2>&1 && echo "llama-server im PR-Bau vorhanden" | tee -a "$L"
echo "FERTIG" | tee -a "$L"
