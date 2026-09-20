#!/bin/bash
# Bei IQ2 nimmt das Hauptmodell 98 % der Entwuerfe an, n=3 schlaegt n=2 (28,91 gegen 27,46
# Token/s). Also weiter nach oben, bis es kippt.
set -uo pipefail
warte_auf_speicher() {   # erst starten, wenn die GPU wieder leer ist
    for _ in $(seq 120); do
        belegt=$(( $(cat /sys/class/drm/card0/device/mem_info_gtt_used) / 1024**3 ))
        frei=$(free -g | awk '/^Mem:/ {print $7}')
        [ "$belegt" -lt 10 ] && [ "$frei" -gt 100 ] && return 0
        sleep 5
    done
    echo "WARNUNG: GPU-Speicher wurde nicht frei (GTT ${belegt} GiB, frei ${frei} GB)" >&2
    return 1
}

while systemctl --user is-active --quiet v4-endstand.service; do sleep 30; done
FO=$HOME/src/strix-llama
M2=$HOME/models/DeepSeek-V4-Flash-0731-IQ2/UD-IQ2_XXS/DeepSeek-V4-Flash-0731-UD-IQ2_XXS-00001-of-00003.gguf
E=$HOME/models/DeepSeek-V4-Flash-0731/dspark-DeepSeek-V4-Flash-0731-Q8_0.gguf
L=$HOME/bench/v4-gegen-qwen/v4-n45.log
: > "$L"
for n in 4 5 6; do
    warte_auf_speicher; "$FO/build-vulkan/bin/llama-server" -m "$M2" -md "$E" -ngld 999 \
      --spec-type draft-dspark --spec-draft-n-max $n \
      -ngl 999 -fa on -c 4096 -b 2048 -ub 512 --jinja --host 127.0.0.1 --port 8092 \
      --alias v4 -np 1 > /tmp/v4-n$n.log 2>&1 &
    pid=$!
    for _ in $(seq 400); do curl -sf -m 2 http://127.0.0.1:8092/health >/dev/null && break
        kill -0 $pid 2>/dev/null || break; sleep 2; done
    python3 - "$n" "$L" <<'PY'
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
        "http://127.0.0.1:8092/completion", d, {"Content-Type": "application/json"}), timeout=1800))["timings"]
    w.append((t["prompt_per_second"], t["predicted_per_second"]))
pp = sorted(x[0] for x in w)[1]; tg = sorted(x[1] for x in w)[1]
z = f"IQ2 + DSpark n={n}   Prompt {pp:7.1f} Token/s   Textausgabe {tg:6.2f} Token/s"
print(z, flush=True); open(logdatei, "a").write(z + "\n")
PY
    kill $pid 2>/dev/null; wait $pid 2>/dev/null
    grep -oE 'draft acceptance = [0-9.]+[^,]*, mean len = *[0-9.]+' /tmp/v4-n$n.log | tail -1 | tee -a "$L"
    sleep 5
done
echo "FERTIG" | tee -a "$L"
