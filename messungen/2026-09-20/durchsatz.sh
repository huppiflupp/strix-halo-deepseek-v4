#!/bin/bash
# Durchsatz von DeepSeek-V4-Flash (IQ3_XXS, 97 GiB) auf Vulkan:
#   - mit und ohne DSpark-Entwurf, Entwurfslaenge 1/2/3
#   - Ladeart: mmap (Vorgabe) gegen mlock, wegen Page-Cache-Druck bei nur ~13 GiB Rest
#   - Flash Attention ausdruecklich an
# GTT steht per Kernel-Befehlszeile auf 120 GiB, das Modell liegt also komplett
# GPU-adressierbar; VRAM-Split gibt es auf dieser Maschine nicht.
# Die anderen llama-Dienste werden gestoppt und am Ende wieder gestartet.
set -uo pipefail
V=$HOME/bench/v4-gegen-qwen
D=$HOME/models/DeepSeek-V4-Flash-0731
M=$D/UD-IQ3_XXS/DeepSeek-V4-Flash-0731-UD-IQ3_XXS-00001-of-00004.gguf
E=$D/dspark-DeepSeek-V4-Flash-0731-Q8_0.gguf
F=$HOME/strix-fork/vulkan
PORT=8092
ERG=$V/durchsatz.json
[ -f "$M" ] || { echo "Modell fehlt: $M"; exit 1; }
[ -f "$E" ] || { echo "Entwurfsmodell fehlt: $E"; exit 1; }

[ -f "$ERG" ] || echo "[]" > "$ERG"
gtt() { echo $(( $(cat /sys/class/drm/card0/device/mem_info_gtt_used) / 1024**2 )); }

dienste() {   # $1 = stop|start
    # Die Sockets muessen mit weg: Open WebUI fragt 8093 periodisch ab und wuerde den
    # Familiendienst mitten in der Messung wieder wecken (Socket-Aktivierung).
    local einheiten="llama-qwen36-familie.socket llama-qwen36-familie-proxy.service
                     llama-qwen36-familie.service llama-aufgaben.service
                     llama-coder.socket llama-qwen.socket llama-qwen36.socket"
    if [ "$1" = "stop" ]; then
        for d in $einheiten; do systemctl --user stop "$d" 2>/dev/null; done
    else
        for d in $einheiten; do systemctl --user start "$d" 2>/dev/null; done
    fi
}

messe() {   # $1 Etikett, Rest: zusaetzliche Argumente
    local etikett="$1"; shift
    if python3 -c "import json,sys; sys.exit(0 if any(e['etikett']=='$etikett' for e in json.load(open('$ERG'))) else 1)"; then
        echo "== $etikett (schon gemessen, uebersprungen)"; return
    fi
    local log="$V/server-$etikett.log"
    echo "== $etikett" | tee -a "$V/durchsatz.log"
    local t0=$(date +%s)
    "$F/llama-server" -m "$M" -ngl 999 -fa on -c 16384 -b 2048 -ub 512 --jinja \
        --host 127.0.0.1 --port $PORT --alias v4 -np 1 "$@" > "$log" 2>&1 &
    local pid=$!
    for _ in $(seq 600); do curl -sf -m 2 http://127.0.0.1:$PORT/health >/dev/null && break; sleep 2; done
    local ladezeit=$(( $(date +%s) - t0 ))
    local belegt=$(gtt)
    python3 - "$etikett" "$ladezeit" "$belegt" "$ERG" <<'PY'
import json, sys, time, urllib.request
etikett, ladezeit, belegt, ergdatei = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
# Langer Vorspann: Die Prompt-Rate an einem 20-Token-Prompt zu messen sagt nichts,
# da dominiert der Ueberhang. Rund 2000 Token sind naeher am Betrieb.
VORSPANN = ("Der folgende Text dient als Kontext fuer die anschliessende Frage.\n\n"
            + "Ein Waermetauscher im Gegenstrom fuehrt zwei Stoffstroeme in entgegengesetzter Richtung "
              "aneinander vorbei, sodass die Temperaturdifferenz ueber die gesamte Laenge annaehernd "
              "gleich bleibt und der Waermeuebergang gleichmaessig erfolgt. " * 60)
P = [VORSPANN + "\n\nErklaere in etwa 200 Woertern, wie ein Waermetauscher im Gegenstrom funktioniert.",
     VORSPANN + "\n\nSchreibe eine kurze Funktion in Python, die Zahlen nach Quersumme sortiert.",
     VORSPANN + "\n\nFasse Vor- und Nachteile von Fernwaerme gegenueber einer Waermepumpe zusammen."]
werte = []
for p in P:
    d = json.dumps({"prompt": p, "n_predict": 256, "temperature": 0, "cache_prompt": False}).encode()
    r = urllib.request.Request(f"http://127.0.0.1:8092/completion", d, {"Content-Type": "application/json"})
    t = json.load(urllib.request.urlopen(r, timeout=1800))["timings"]
    werte.append((t["prompt_per_second"], t["predicted_per_second"]))
    print(f"      Vorspann {t.get('prompt_n')} Token in {t.get('prompt_ms', 0)/1000:.1f} s", flush=True)
pp = sorted(x[0] for x in werte)[1]; tg = sorted(x[1] for x in werte)[1]
print(f"   Laden {ladezeit} s, GTT {belegt/1024:.1f} GiB, Prompt {pp:.1f} t/s, Generierung {tg:.2f} t/s")
alt = json.load(open(ergdatei))
alt.append({"etikett": etikett, "ladezeit_s": ladezeit, "gtt_gib": round(belegt/1024, 1),
            "prompt_ts": round(pp, 1), "gen_ts": round(tg, 2)})
json.dump(alt, open(ergdatei, "w"), indent=1)
PY
    kill $pid 2>/dev/null; wait $pid 2>/dev/null
    grep -iE 'accept|n_draft|draft acceptance' "$log" | tail -3
    sleep 5
}

dienste stop
sleep 5
echo "GTT vor den Messungen: $(( $(gtt) / 1024 )) GiB" | tee "$V/durchsatz.log"

messe ohne-entwurf-mmap
messe ohne-entwurf-mlock          --load-mode mlock
messe dspark-n1                   -md "$E" -ngld 999 --spec-type draft-dspark --spec-draft-n-max 1
messe dspark-n2                   -md "$E" -ngld 999 --spec-type draft-dspark --spec-draft-n-max 2
messe dspark-n3                   -md "$E" -ngld 999 --spec-type draft-dspark --spec-draft-n-max 3
messe dspark-n1-mlock             -md "$E" -ngld 999 --spec-type draft-dspark --spec-draft-n-max 1 --load-mode mlock

echo; echo "== llama-bench mit Kontexttiefe, MMID an gegen aus =="
# Die MoE-Optimierungen des Forks sind einzeln abschaltbar (alle opt-out ueber Umgebungsvariablen).
# Nur die MMID_*-Schalter werden umgelegt; die FA-Schalter bleiben in beiden Laeufen an,
# sonst waere der Vergleich nicht mehr auf MoE-Matmul beschraenkt.
tiefenlauf() {   # $1 Etikett, $2 = 1 (an) oder 0 (aus)
    local etikett="$1" an="$2"
    echo "-- MMID $etikett" | tee -a "$V/durchsatz.log"
    env GGML_VK_MMID_ROWLISTS=$an GGML_VK_MMID_SMALLN=$an GGML_VK_MMID_BM64=$an \
        GGML_VK_MMID_WAVE32=$an GGML_VK_MMID_F16B=$an GGML_VK_MMID_M128=$an \
        "$F/llama-bench" -m "$M" -fa 1 -p 512 -n 128 -d 0,8192,32768 -r 2 -o json \
        > "$V/bench-mmid-$etikett.json" 2> "$V/bench-mmid-$etikett.log"
    python3 - "$etikett" "$V/bench-mmid-$etikett.json" <<'PY2'
import json, sys
etikett, datei = sys.argv[1], sys.argv[2]
for e in json.load(open(datei)):
    art = f"pp{e['n_prompt']}" if e["n_prompt"] else f"tg{e['n_gen']}"
    print(f"   {etikett:4s} {art:8s} @ Tiefe {e['n_depth']:6d}: {e['avg_ts']:8.2f} t/s")
PY2
}
tiefenlauf an 1
tiefenlauf aus 0

echo; echo "== Was das fuer einen Agentenstart bedeutet =="
python3 - "$V" <<'PY3'
import json, sys, os
V = sys.argv[1]
for etikett in ("an", "aus"):
    p = f"{V}/bench-mmid-{etikett}.json"
    if not os.path.exists(p): continue
    d = json.load(open(p))
    pp = {e["n_depth"]: e["avg_ts"] for e in d if e["n_prompt"]}
    for tiefe, rate in sorted(pp.items()):
        print(f"   MMID {etikett:3s}: Prompt bei Tiefe {tiefe:6d} = {rate:7.1f} t/s "
              f"-> 15000 Token Startprompt in {15000/rate/60:5.1f} min")
PY3

if [ "${NEUSTART:-0}" = "1" ]; then dienste start; else
    echo "Die llama-Dienste bleiben gestoppt (NEUSTART=1 setzen, um sie wieder zu starten)."
fi
echo; echo "Ergebnisse: $ERG"
python3 -c "
import json
for e in json.load(open('$ERG')):
    print(f\"  {e['etikett']:22s} Laden {e['ladezeit_s']:4d} s  GTT {e['gtt_gib']:6.1f} GiB  \"
          f\"Prompt {e['prompt_ts']:7.1f} t/s  Generierung {e['gen_ts']:6.2f} t/s\")
"
