#!/bin/bash
# Killt llama-Prozesse, sobald MemAvailable unter die Schwelle faellt —
# bevor der globale OOM-Killer die ganze Maschine in einen Reboot reisst.
#
# Hintergrund: Beim Laden eines 97 GiB-Modells auf Strix Halo reissen
# Page-Cache und Staging-Puffer den System-RAM leer. Der Kernel-OOM-Killer
# trifft dann irgendeinen Prozess — im Zweifel einen, den systemd braucht.
#
# Aufruf:  ./memguard.sh &   (vor dem llama-Start)

THRESH_MB=4000

while true; do
  AVAIL=$(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo)
  PID=$(pgrep -f "llama-(cli|server|bench)" | head -1)
  [ -z "$PID" ] && { sleep 2; continue; }
  if [ "$AVAIL" -lt "$THRESH_MB" ]; then
    echo "$(date +%H:%M:%S) GUARD: MemAvailable=${AVAIL}M < ${THRESH_MB}M -> SIGKILL an PID $PID" | tee -a ~/memguard.log
    kill -9 "$PID"
    sleep 5
  fi
  sleep 1
done
