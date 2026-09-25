#!/bin/bash
# Statistisch abgesicherter Sweep #28613: 6 Runden ABBA, Waechter gegen Fremdlast, Monitor.
D=~/bench/llamacpp-tuning/ergebnisse/sweep28613-width/stat
cd $D
ME=$$
monitor(){ while true; do
  printf '%s load=%s gpu=%s | %s\n' "$(date +%T)" "$(cut -d' ' -f1 /proc/loadavg)" \
    "$(cat /sys/class/drm/card*/device/gpu_busy_percent 2>/dev/null | head -1)" \
    "$(ps -eo pcpu,comm --sort=-pcpu --no-headers | head -4 | awk '{printf "%s:%s ",$2,$1}')"
  sleep 5; done; }
monitor > monitor.log & MON=$!
trap "kill $MON" EXIT
waechter(){  # wartet, bis System ruhig ist; protokolliert Wartezeit
  local t0=$(date +%s)
  while true; do
    l=$(cut -d' ' -f1 /proc/loadavg)
    fremd=$(ps -eo pcpu,comm --sort=-pcpu --no-headers | awk '$1>30 && $2!="test-backend-o"' | head -1)
    if awk "BEGIN{exit !($l<1.5)}" && [ -z "$fremd" ]; then break; fi
    sleep 10
  done
  echo "$(date +%T) ruhig nach $(( $(date +%s)-t0 )) s (load $l)" >> waechter.log
}
for r in 1 2 3 4 5 6; do
  if (( r % 2 )); then order="master pr28613"; else order="pr28613 master"; fi
  for w in $order; do
    waechter
    echo "$(date +%T) start r$r $w" >> waechter.log
    ~/llama-work/$w/build-hip/bin/test-backend-ops perf -o MUL_MAT -b ROCm0 --test-file ../width-sweep.txt 2>&1 \
      | grep -oE "name=[a-zA-Z0-9_]+|[0-9.]+ us/run" | paste - - | sed "s/^name=//; s/ us\/run//" > $w-r$r.tsv
    echo "$(date +%T) ende  r$r $w ($(wc -l < $w-r$r.tsv) Formen)" >> waechter.log
  done
done
echo fertig
