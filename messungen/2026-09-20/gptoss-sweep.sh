#!/bin/bash
# gpt-oss-120b (MXFP4, 63 GB): welche der Hebel, die am 20.09.2026 bei DeepSeek und Qwen
# etwas brachten, wirken auch hier? Nur Tempo, keine Qualitaetsmessung (Nutzerwunsch) --
# eine Zahl aus diesem Lauf sagt also nichts darueber, ob der Pfad richtig rechnet.
# Jeder llama-bench-Aufruf faehrt mehrere Werte in einem Prozess, das Modell laedt je
# Schritt nur einmal.
set -uo pipefail
gtt_pfad() {
    for u in /sys/class/drm/card*/device/uevent; do
        grep -qx 'DRIVER=amdgpu' "$u" 2>/dev/null || continue
        echo "$(dirname "$u")/mem_info_gtt_used"; return 0
    done; echo /dev/null
}
warte() {
    for _ in $(seq 240); do
        b=$(( $(cat "$(gtt_pfad)" 2>/dev/null || echo 0) / 1024**3 ))
        f=$(free -g | awk '/^Mem:/ {print $7}')
        [ "$b" -lt 10 ] && [ "$f" -gt 100 ] && return 0; sleep 5
    done; echo "Speicher wurde nicht frei" >&2; exit 1
}
while pgrep -f 'dl-teil.py.*gpt-oss' >/dev/null; do sleep 30; done
M=$HOME/models/gpt-oss-120b/gpt-oss-120b-MXFP4.gguf
E=$HOME/models/gpt-oss-120b/eagle3-gpt-oss-120b-Q8_0.gguf
VQ=$HOME/src/strix-llama/build-vulkan/bin          # Fork, Quelltext-Bau, System-RADV
VP=$HOME/strix-fork/vulkan                          # Fork, Payload, gebuendelter RADV
HU=$HOME/src/llamacpp-upstream/build-hip/bin        # Upstream, HIP (rechnet korrekt, anders als Fork-HIP)
R72=/usr/local/lib/ollama/rocm_v7_2
PRE="$R72/libamdhip64.so.7:$R72/libhsa-runtime64.so.1:$R72/libamd_comgr.so.3"
L=$HOME/bench/v4-gegen-qwen/gptoss-sweep.log
: > "$L"
[ -f "$M" ] || { echo "Modell fehlt" | tee -a "$L"; exit 1; }
zeilen() { grep -E '^\| *gpt-oss|^\| *model' | sed 's/gpt-oss 120B MXFP4 MoE *//' ; }
bank() { local titel="$1"; shift; echo; echo "== $titel" | tee -a "$L"; warte
         "$@" 2>/dev/null | zeilen | tee -a "$L"; }

echo "== 0. Ladezeit kalt (Vulkan, Quelltext-Bau)" | tee -a "$L"; warte
t0=$(date +%s.%N); "$VQ/llama-bench" -m "$M" -fa 1 -p 1 -n 0 -r 1 >/dev/null 2>&1; t1=$(date +%s.%N)
LC_ALL=C printf "Ladezeit: %.1f s\n" "$(echo "$t1 - $t0" | bc)" | tee -a "$L"

bank "1. Flash Attention aus/an (Vulkan)" \
     "$VQ/llama-bench" -m "$M" -fa 0,1 -b 2048 -ub 512 -p 512 -n 128 -r 2
bank "2. Mikrobatch bei 2048 Token Vorlauf (Vulkan)" \
     "$VQ/llama-bench" -m "$M" -fa 1 -b 2048 -ub 256,512,1024,2048 -p 2048 -n 0 -r 2
bank "3. Batch 512 gegen 2048 (Vulkan, 2048 Token Vorlauf)" \
     "$VQ/llama-bench" -m "$M" -fa 1 -b 512,2048 -ub 512 -p 2048 -n 0 -r 2
bank "4. Kontexttiefe 0/8192, KV f16 gegen q8_0 (Vulkan)" \
     "$VQ/llama-bench" -m "$M" -fa 1 -b 2048 -ub 512 -ctk f16,q8_0 -ctv f16,q8_0 -p 512 -n 128 -d 0,8192 -r 2
for s in GGML_VK_MMID_ROWLISTS GGML_VK_MMID_SMALLN GGML_VK_MMID_BM64; do
    bank "5. Fork-Schalter $s=0 (Rest an)" \
         env $s=0 "$VQ/llama-bench" -m "$M" -fa 1 -b 2048 -ub 512 -p 512 -n 128 -r 2
done
bank "6. Payload mit gebuendeltem RADV statt System-RADV" \
     "$VP/llama-bench" -m "$M" -fa 1 -b 2048 -ub 512 -p 512 -n 128 -r 2
bank "7. HIP, ROCm 7.1.1 (Upstream-Bau)" \
     "$HU/llama-bench" -m "$M" -fa 1 -b 2048 -ub 512,2048 -p 512,2048 -n 128 -r 2
bank "8. HIP, ROCm 7.2.1 (Laufzeit aus dem Ollama-Buendel, rocBLAS bleibt 7.1.1)" \
     env LD_PRELOAD="$PRE" "$HU/llama-bench" -m "$M" -fa 1 -b 2048 -ub 512,2048 -p 512,2048 -n 128 -r 2

echo | tee -a "$L"; echo "== 9. Chatbetrieb mit EAGLE3-Entwurfsmodell, Laenge 0 bis 4 (Vulkan)" | tee -a "$L"
for N in 0 1 2 3 4; do
    warte
    ARGS=(-m "$M" -ngl 999 -fa on -c 8192 -b 2048 -ub 512 --jinja --host 127.0.0.1 --port 8098 -np 1)
    [ "$N" -gt 0 ] && ARGS+=(-md "$E" -ngld 999 --spec-type draft-eagle3 --spec-draft-n-max $N)
    "$VQ/llama-server" "${ARGS[@]}" > /tmp/gptoss-n$N.log 2>&1 &
    PID=$!
    for _ in $(seq 300); do curl -sf -m 2 http://127.0.0.1:8098/health >/dev/null && break
        kill -0 $PID 2>/dev/null || break; sleep 2; done
    if kill -0 $PID 2>/dev/null; then
    python3 - "$N" "$L" <<'PY'
import json, sys, urllib.request
n, logdatei = sys.argv[1], sys.argv[2]
V = ("Der folgende Text dient als Kontext.\n\n" + "Ein Waermetauscher im Gegenstrom fuehrt zwei "
     "Stoffstroeme aneinander vorbei, sodass die Temperaturdifferenz gleich bleibt. " * 60)
P = [V + "\n\nErklaere in etwa 200 Woertern, wie ein Waermetauscher funktioniert.",
     V + "\n\nSchreibe eine kurze Funktion in Python, die Zahlen nach Quersumme sortiert.",
     V + "\n\nFasse Vor- und Nachteile von Fernwaerme gegenueber einer Waermepumpe zusammen."]
w = []
for p in P:
    d = json.dumps({"prompt": p, "n_predict": 256, "temperature": 0, "cache_prompt": False}).encode()
    t = json.load(urllib.request.urlopen(urllib.request.Request(
        "http://127.0.0.1:8098/completion", d, {"Content-Type": "application/json"}), timeout=900))["timings"]
    w.append((t["prompt_per_second"], t["predicted_per_second"]))
pp = sorted(x[0] for x in w)[1]; tg = sorted(x[1] for x in w)[1]
z = f"Entwurfslaenge {n}:  Prompt {pp:7.1f} Token/s   Textausgabe {tg:6.2f} Token/s"
print(z, flush=True); open(logdatei, "a").write(z + "\n")
PY
    else echo "Entwurfslaenge $N: Server startete nicht -- $(tail -1 /tmp/gptoss-n$N.log | cut -c1-100)" | tee -a "$L"; fi
    kill $PID 2>/dev/null; wait $PID 2>/dev/null
    grep -oE 'draft acceptance = [0-9.]+[^,]*, mean len = *[0-9.]+' /tmp/gptoss-n$N.log | tail -1 | tee -a "$L"
    sleep 5
done
echo "FERTIG" | tee -a "$L"
