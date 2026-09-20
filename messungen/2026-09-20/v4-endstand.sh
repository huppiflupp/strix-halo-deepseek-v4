#!/bin/bash
# Abschliessende V4-Messung mit allem, was belegt hilft:
#  - Quelltext-Build des Forks mit System-RADV (das Payload bringt einen eigenen, aelteren
#    RADV mit; beim Benchmark waren das 233,7 gegen 209,2 Token/s zugunsten des Systems)
#  - DSpark-Entwurfsmodell, Laenge 2 und 3
#  - beide Quantisierungen
# Wartet, bis die Lucebox-Dateien geladen sind, sonst reicht der Speicher nicht.
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

while pgrep -f 'dl-teil.py' >/dev/null; do sleep 60; done
FO=$HOME/src/strix-llama
cmake --build "$FO/build-vulkan" --target llama-server -j "$(nproc)" >/dev/null 2>&1
D3=$HOME/models/DeepSeek-V4-Flash-0731
M3=$D3/UD-IQ3_XXS/DeepSeek-V4-Flash-0731-UD-IQ3_XXS-00001-of-00004.gguf
M2=$HOME/models/DeepSeek-V4-Flash-0731-IQ2/UD-IQ2_XXS/DeepSeek-V4-Flash-0731-UD-IQ2_XXS-00001-of-00003.gguf
E=$D3/dspark-DeepSeek-V4-Flash-0731-Q8_0.gguf
L=$HOME/bench/v4-gegen-qwen/v4-endstand.log
: > "$L"
messe() {   # $1 Etikett, $2 Server-Binaerdatei, $3 Modell, Rest: Zusatzargumente
    local etikett="$1" bin="$2" modell="$3"; shift 3
    local log=/tmp/v4end-$etikett.log
    "$bin" -m "$modell" -ngl 999 -fa on -c 4096 -b 2048 -ub 512 --jinja \
        --host 127.0.0.1 --port 8092 --alias v4 -np 1 "$@" > "$log" 2>&1 &
    local pid=$!
    for _ in $(seq 400); do curl -sf -m 2 http://127.0.0.1:8092/health >/dev/null && break
        kill -0 $pid 2>/dev/null || { echo "$etikett: Server beendet" | tee -a "$L"; return; }; sleep 2; done
    python3 - "$etikett" "$L" <<'PY'
import json, sys, urllib.request
etikett, logdatei = sys.argv[1], sys.argv[2]
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
zeile = f"{etikett:34s} Prompt {pp:7.1f} Token/s   Textausgabe {tg:6.2f} Token/s"
print(zeile, flush=True)
open(logdatei, "a").write(zeile + "\n")
PY
    kill $pid 2>/dev/null; wait $pid 2>/dev/null
    grep -oE 'draft acceptance = [0-9.]+[^,]*, mean len = *[0-9.]+' "$log" | tail -1 | tee -a "$L"
    sleep 5
}
DS2="--spec-type draft-dspark --spec-draft-n-max 2"
DS3="--spec-type draft-dspark --spec-draft-n-max 3"
echo "== Quelltext-Build (System-RADV)" | tee -a "$L"
messe "IQ3 + DSpark n=2 (Quelltext)"  warte_auf_speicher; "$FO/build-vulkan/bin/llama-server" "$M3" -md "$E" -ngld 999 $DS2
messe "IQ2 + DSpark n=2 (Quelltext)"  warte_auf_speicher; "$FO/build-vulkan/bin/llama-server" "$M2" -md "$E" -ngld 999 $DS2
messe "IQ2 + DSpark n=3 (Quelltext)"  warte_auf_speicher; "$FO/build-vulkan/bin/llama-server" "$M2" -md "$E" -ngld 999 $DS3
echo "== Payload (gebuendelter RADV), zur Gegenprobe" | tee -a "$L"
messe "IQ2 + DSpark n=2 (Payload)"    "$HOME/strix-fork/vulkan/llama-server" "$M2" -md "$E" -ngld 999 $DS2
echo "FERTIG" | tee -a "$L"
