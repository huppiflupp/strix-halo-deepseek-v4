#!/bin/bash
# gpt-oss-120b: die vier Hebel, die ohne Eingriff des Nutzers messbar sind.
#  1. PR #27952 (int8-Matrixkerne fuer RDNA3, deckt MXFP4 ab) gegen Master vom selben Tag
#     und gegen unseren Fork -- mit Perplexitaetsprobe, weil ein neuer Kernelpfad rechnet
#  2. -ub 4096 gegen 2048 bei langen Prompts
#  3. KV-Cache q8_0 gegen f16 bei grosser Tiefe (32k, 65k)
#  4. reasoning_effort low/medium/high: Zeit, Tokenzahl, Richtigkeit
set -uo pipefail
gtt_pfad() { for u in /sys/class/drm/card*/device/uevent; do grep -qx 'DRIVER=amdgpu' "$u" 2>/dev/null || continue
             echo "$(dirname "$u")/mem_info_gtt_used"; return 0; done; echo /dev/null; }
warte() { for _ in $(seq 240); do b=$(( $(cat "$(gtt_pfad)" 2>/dev/null || echo 0) / 1024**3 ))
          f=$(free -g | awk '/^Mem:/ {print $7}'); [ "$b" -lt 10 ] && [ "$f" -gt 100 ] && return 0; sleep 5; done
          echo "Speicher wurde nicht frei" >&2; exit 1; }
M=$HOME/models/gpt-oss-120b/gpt-oss-120b-MXFP4.gguf
W=$HOME/bench/llamacpp-tuning/wiki.test.raw
FORK=$HOME/src/strix-llama/build-vulkan/bin
MASTER=$HOME/llama-work/master/build-vulkan/bin
PR=$HOME/llama-work/pr27952/build-vulkan/bin
L=$HOME/bench/v4-gegen-qwen/gptoss-hebel.log
: > "$L"
zeilen() { grep -E '^\| *gpt-oss|^\| *model' | sed 's/gpt-oss 120B MXFP4 MoE *//'; }

echo "== 1. PR #27952 gegen Master (beide Stand 18.09.) und Fork" | tee -a "$L"
for v in "Master ec9281505:$MASTER" "PR27952 df9bcc16a:$PR" "Fork 50c271f8e:$FORK"; do
    echo "-- ${v%%:*}" | tee -a "$L"; warte
    "${v#*:}/llama-bench" -m "$M" -fa 1 -b 2048 -ub 2048 -p 512,2048 -n 128 -r 2 2>/dev/null | zeilen | tee -a "$L"
done
echo -n "-- Perplexitaet PR27952 (Vergleichsband 436,9 / 455,5): " | tee -a "$L"; warte
"$PR/llama-perplexity" -m "$M" -f "$W" -c 2048 -ub 2048 --chunks 10 -ngl 999 -fa on 2>&1 | grep -oE 'Final estimate.*' | tee -a "$L"

echo | tee -a "$L"; echo "== 2. Mikrobatch 4096 gegen 2048 (Fork)" | tee -a "$L"; warte
"$FORK/llama-bench" -m "$M" -fa 1 -b 4096 -ub 2048,4096 -p 4096,16384 -n 0 -r 2 2>/dev/null | zeilen | tee -a "$L"

echo | tee -a "$L"; echo "== 3. KV-Cache bei grosser Tiefe (Fork, -ub 2048)" | tee -a "$L"
for kv in f16 q8_0; do
    echo "-- KV $kv" | tee -a "$L"; warte
    "$FORK/llama-bench" -m "$M" -fa 1 -b 2048 -ub 2048 -ctk $kv -ctv $kv -p 0 -n 64 -d 32768,65536 -r 1 2>/dev/null | zeilen | tee -a "$L"
done

echo | tee -a "$L"; echo "== 4. reasoning_effort (Fork-Server, Betriebseinstellung)" | tee -a "$L"; warte
"$FORK/llama-server" -m "$M" -ngl 999 -fa on -ub 2048 -c 32768 -np 1 --jinja --predict 16384 \
    --host 127.0.0.1 --port 8098 > /tmp/gptoss-effort.log 2>&1 &
PID=$!
for _ in $(seq 300); do curl -sf -m 2 http://127.0.0.1:8098/health >/dev/null && break; kill -0 $PID 2>/dev/null || break; sleep 2; done
python3 - "$L" <<'PY'
import json, sys, time, urllib.request
logdatei = sys.argv[1]
FRAGEN = [("Was ist 17 mal 23? Nur die Zahl.", "391"),
          ("Ein Zug faehrt 240 km in 1,5 Stunden. Wie schnell ist er in km/h? Nur die Zahl.", "160"),
          ("Schreibe das Wort 'Waermepumpe' rueckwaerts. Nur das Ergebnis.", "epmupemreaw"),
          ("Wie viele Primzahlen gibt es zwischen 50 und 100? Nur die Zahl.", "10"),
          ("Anna ist aelter als Ben, Ben ist aelter als Clara, Clara ist aelter als David. Wer ist am zweitjuengsten? Nur der Name.", "clara"),
          ("Ein Schlaeger und ein Ball kosten zusammen 1,10 Euro. Der Schlaeger kostet 1 Euro mehr als der Ball. Was kostet der Ball in Cent? Nur die Zahl.", "5")]
for stufe in ("low", "medium", "high"):
    zeit = tok = ok = 0
    for f, soll in FRAGEN:
        d = json.dumps({"messages": [{"role": "user", "content": f}], "temperature": 0, "max_tokens": 8000,
                        "chat_template_kwargs": {"reasoning_effort": stufe}}).encode()
        t0 = time.time()
        r = json.load(urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:8098/v1/chat/completions", d,
                {"Content-Type": "application/json"}), timeout=1200))
        zeit += time.time() - t0; tok += r["usage"]["completion_tokens"]
        a = (r["choices"][0]["message"].get("content") or "").strip().lower()
        ok += soll in a.replace(".", "").replace(",", "")
    z = f"reasoning_effort={stufe:6s}  richtig {ok}/{len(FRAGEN)}   Token gesamt {tok:5d}   Zeit gesamt {zeit:6.1f} s   je Frage {zeit/len(FRAGEN):5.1f} s"
    print(z, flush=True); open(logdatei, "a").write(z + "\n")
PY
kill $PID 2>/dev/null; wait $PID 2>/dev/null
echo "FERTIG" | tee -a "$L"
