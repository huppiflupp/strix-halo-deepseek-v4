#!/bin/bash
# Nachstellung des fremden "Lucebox"-Profils (25,3 t/s, mit DSpark 32) auf unserer Maschine:
# dort 4 statt 6 aktive Experten und staerkere Quantisierung. Beide Abkuerzungen werden
# einzeln und zusammen gemessen -- Durchsatz UND Perplexitaet, damit sichtbar wird, was
# die Geschwindigkeit kostet. Unser Modell nutzt ab Werk 6 von 256 Experten je Token.
set -uo pipefail
V=$HOME/bench/v4-gegen-qwen
D3=$HOME/models/DeepSeek-V4-Flash-0731
D2=$HOME/models/DeepSeek-V4-Flash-0731-IQ2
M3=$D3/UD-IQ3_XXS/DeepSeek-V4-Flash-0731-UD-IQ3_XXS-00001-of-00004.gguf
M2=$D2/UD-IQ2_XXS/DeepSeek-V4-Flash-0731-UD-IQ2_XXS-00001-of-00003.gguf
E=$D3/dspark-DeepSeek-V4-Flash-0731-Q8_0.gguf
F=$HOME/strix-fork/vulkan
PPL=/home/seeas/src/strix-llama/build-vulkan/bin/llama-perplexity
W=$HOME/bench/llamacpp-tuning/wiki.test.raw
ERG=$V/durchsatz.json
PORT=8092
[ -f "$ERG" ] || echo "[]" > "$ERG"
gtt() { echo $(( $(cat /sys/class/drm/card0/device/mem_info_gtt_used) / 1024**2 )); }

messe() {   # $1 Etikett, $2 Modellpfad, Rest: zusaetzliche Argumente
    local etikett="$1" modell="$2"; shift 2
    if python3 -c "import json,sys; sys.exit(0 if any(e['etikett']=='$etikett' for e in json.load(open('$ERG'))) else 1)"; then
        echo "== $etikett (schon gemessen)"; return
    fi
    echo "== $etikett"
    local log="$V/server-$etikett.log" t0=$(date +%s)
    "$F/llama-server" -m "$modell" -ngl 999 -fa on -c 16384 -b 2048 -ub 512 --jinja \
        --host 127.0.0.1 --port $PORT --alias v4 -np 1 "$@" > "$log" 2>&1 &
    local pid=$!
    for _ in $(seq 600); do
        curl -sf -m 2 http://127.0.0.1:$PORT/health >/dev/null && break
        kill -0 $pid 2>/dev/null || { echo "   Server beendet sich -- uebersprungen:"; tail -2 "$log"; return; }
        sleep 2
    done
    python3 - "$etikett" "$(( $(date +%s) - t0 ))" "$(gtt)" "$ERG" <<'PY'
import json, sys, urllib.request
etikett, ladezeit, belegt, ergdatei = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
VORSPANN = ("Der folgende Text dient als Kontext.\n\n"
            + "Ein Waermetauscher im Gegenstrom fuehrt zwei Stoffstroeme in entgegengesetzter Richtung "
              "aneinander vorbei, sodass die Temperaturdifferenz ueber die Laenge gleich bleibt. " * 60)
P = [VORSPANN + "\n\nErklaere in etwa 200 Woertern, wie ein Waermetauscher im Gegenstrom funktioniert.",
     VORSPANN + "\n\nSchreibe eine kurze Funktion in Python, die Zahlen nach Quersumme sortiert.",
     VORSPANN + "\n\nFasse Vor- und Nachteile von Fernwaerme gegenueber einer Waermepumpe zusammen."]
werte = []
for p in P:
    d = json.dumps({"prompt": p, "n_predict": 256, "temperature": 0, "cache_prompt": False}).encode()
    r = urllib.request.Request("http://127.0.0.1:8092/completion", d, {"Content-Type": "application/json"})
    t = json.load(urllib.request.urlopen(r, timeout=1800))["timings"]
    werte.append((t["prompt_per_second"], t["predicted_per_second"]))
pp = sorted(x[0] for x in werte)[1]; tg = sorted(x[1] for x in werte)[1]
print(f"   Laden {ladezeit} s, GTT {belegt/1024:.1f} GiB, Prompt {pp:.1f} t/s, Generierung {tg:.2f} t/s")
alt = json.load(open(ergdatei))
alt.append({"etikett": etikett, "ladezeit_s": ladezeit, "gtt_gib": round(belegt/1024, 1),
            "prompt_ts": round(pp, 1), "gen_ts": round(tg, 2)})
json.dump(alt, open(ergdatei, "w"), indent=1)
PY
    kill $pid 2>/dev/null; wait $pid 2>/dev/null
    grep -oE 'draft acceptance = [0-9.]+ \([^)]*\), mean len = *[0-9.]+' "$log" | tail -1
    sleep 5
}

# Reduziertes Expert-Routing (Lucebox: top-k 4 statt 6) laesst sich NICHT per Metadaten
# umschalten: DeepSeek-V4 hat die Routing-Tabelle als Tensor 'blk.N.ffn_gate_tid2eid.weight'
# der Form 6 x 129280 im Modell. --override-kv aendert nur die Metadaten, dann scheitert die
# Formpruefung beim Laden. Dafuer braeuchte es ein neu gebautes GGUF.
DS="-md $E -ngld 999 --spec-type draft-dspark --spec-draft-n-max 2"

echo "### Durchsatz"
messe iq2-experten6            "$M2"
messe iq2-experten6-dspark-n2  "$M2" $DS

echo; echo "### Perplexitaet (wikitext, 20 Bloecke a 2048 Token)"
ppl() {   # $1 Etikett, $2 Modell, Rest: Zusatzargumente
    local etikett="$1" modell="$2"; shift 2
    local wert
    wert=$("$PPL" -m "$modell" -f "$W" -c 2048 --chunks 20 -ngl 999 -fa 1 "$@" 2>&1 \
           | grep -oE 'Final estimate: PPL = [0-9.]+ \+/- [0-9.]+' | tail -1)
    echo "   ${etikett}: ${wert:-fehlgeschlagen}" | tee -a "$V/perplexitaet.log"
}
: > "$V/perplexitaet.log"
ppl "IQ3_XXS (97 GiB)" "$M3"
ppl "IQ2_XXS (85 GiB)" "$M2"

echo; echo "### Gesamtuebersicht"
python3 -c "
import json
for e in json.load(open('$ERG')):
    print(f\"  {e['etikett']:28s} GTT {e['gtt_gib']:6.1f} GiB  Prompt {e['prompt_ts']:7.1f} t/s  Generierung {e['gen_ts']:6.2f} t/s\")"
cat "$V/perplexitaet.log"
