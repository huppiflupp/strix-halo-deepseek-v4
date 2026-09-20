#!/bin/bash
# Reproduktion der Lucebox-Zahlen (25,31 Token/s ohne Entwurf, 32,0 mit DSpark) mit
# ihrem eigenen Server, ihrem Modell und ihrer Kommandozeile aus dem Blogbeitrag.
# Einziger Unterschied zu ihrer Anleitung: ROCm 7.1.1 statt 7.2.4, und der Bau brauchte
# -DCMAKE_POSITION_INDEPENDENT_CODE=ON (Fedora baut PIE, ihr CMake nicht).
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

while systemctl --user is-active --quiet v4-n45.service; do sleep 30; done
S=$HOME/src/lucebox/server/build-hip/dflash_server
MODEL=$HOME/models/lucebox/DeepSeek-V4-Flash-ROCMFP2-STRIX.gguf
DRAFT=$HOME/models/lucebox/DeepSeek-V4-Flash-DSpark-draft-Q4RMFP4-denseF16.gguf
L=$HOME/bench/v4-gegen-qwen/lucebox-lauf.log
: > "$L"
[ -f "$MODEL" ] || { echo "Modell fehlt" | tee -a "$L"; exit 1; }

messe() {   # $1 Etikett, Rest: Umgebung/Argumente ueber globale Variablen
    local etikett="$1"; shift
    local log=/tmp/lucebox-$etikett.log
    warte_auf_speicher
    ( "$@" > "$log" 2>&1 ) &
    local pid=$!
    for _ in $(seq 600); do curl -sf -m 2 http://127.0.0.1:8000/v1/models >/dev/null && break
        kill -0 $pid 2>/dev/null || { echo "$etikett: Server beendet sich"; tail -5 "$log"; return; }
        sleep 2; done
    python3 - "$etikett" "$L" <<'PY'
import json, sys, time, urllib.request
etikett, logdatei = sys.argv[1], sys.argv[2]
V = ("Der folgende Text dient als Kontext.\n\n" + "Ein Waermetauscher im Gegenstrom fuehrt zwei "
     "Stoffstroeme aneinander vorbei, sodass die Temperaturdifferenz gleich bleibt. " * 60)
P = [V + "\n\nErklaere in etwa 200 Woertern, wie ein Waermetauscher funktioniert.",
     V + "\n\nSchreibe eine kurze Funktion in Python, die Zahlen nach Quersumme sortiert.",
     V + "\n\nFasse Vor- und Nachteile von Fernwaerme gegenueber einer Waermepumpe zusammen."]
raten = []
for p in P:
    d = json.dumps({"model": "ds4", "messages": [{"role": "user", "content": p}],
                    "max_tokens": 256, "temperature": 0, "stream": False}).encode()
    t0 = time.time()
    r = json.load(urllib.request.urlopen(urllib.request.Request(
        "http://127.0.0.1:8000/v1/chat/completions", d, {"Content-Type": "application/json"}), timeout=1800))
    dt = time.time() - t0
    u = r.get("usage", {})
    n = u.get("completion_tokens") or 0
    raten.append(n / dt if dt else 0)
    print(f"      {n} Token in {dt:.1f} s", flush=True)
m = sorted(raten)[1]
z = f"{etikett:38s} Textausgabe {m:6.2f} Token/s (client-seitig gemessen)"
print(z, flush=True); open(logdatei, "a").write(z + "\n")
PY
    kill $pid 2>/dev/null; wait $pid 2>/dev/null
    grep -iE 'tok/s|tokens/s|decode|prefill' "$log" | tail -3 | tee -a "$L"
    sleep 8
}

echo "== Ohne Entwurfsmodell (Lucebox nennt 25,31 Token/s)" | tee -a "$L"
messe ohne-dspark env LUCE_MMVQ_MAX_NCOLS=4 "$S" "$MODEL" \
    --target-device hip:0 --host 127.0.0.1 --port 8000 \
    --max-ctx 8192 --default-max-tokens 2048 --chunk 2048 --ds4-prefill sparse \
    --ds4-fused-decode --ds4-expert-top-k 4 \
    --prefix-cache-slots 0 --prefill-cache-slots 0 --disk-prefix-cache off

echo "== Mit DSpark, q=4 (Lucebox nennt 32,0 Token/s)" | tee -a "$L"
messe mit-dspark env DFLASH_DS4_SPEC=1 DFLASH_DS4_FUSED_VERIFY=1 DFLASH_DS4_SPEC_Q=4 \
    DFLASH_DS4_TIMING=1 DFLASH_DS4_DRAFT="$DRAFT" LUCE_MMVQ_MAX_NCOLS=4 "$S" "$MODEL" \
    --target-device hip:0 --host 127.0.0.1 --port 8000 \
    --max-ctx 8192 --default-max-tokens 2048 --chunk 2048 --ds4-prefill sparse \
    --ds4-fused-decode --ds4-expert-top-k 4 \
    --prefix-cache-slots 0 --prefill-cache-slots 0 --disk-prefix-cache off

echo "== Gegenprobe: sechs Experten statt vier" | tee -a "$L"
messe sechs-experten env DFLASH_DS4_SPEC=1 DFLASH_DS4_FUSED_VERIFY=1 DFLASH_DS4_SPEC_Q=4 \
    DFLASH_DS4_DRAFT="$DRAFT" LUCE_MMVQ_MAX_NCOLS=4 "$S" "$MODEL" \
    --target-device hip:0 --host 127.0.0.1 --port 8000 \
    --max-ctx 8192 --default-max-tokens 2048 --chunk 2048 --ds4-prefill sparse \
    --ds4-fused-decode --ds4-expert-top-k 6 \
    --prefix-cache-slots 0 --prefill-cache-slots 0 --disk-prefix-cache off
echo "FERTIG" | tee -a "$L"
