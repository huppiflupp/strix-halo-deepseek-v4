#!/bin/bash
# Aktuelle CPU fremder Prozesse (top, 2 s Fenster), ohne test-backend-ops, claude, Kernel-Threads.
while pgrep -f "sweep28613-width/stat/run.sh" >/dev/null; do
  top -b -n 2 -d 2 -w 200 | awk -v t="$(date +%T)" '
    /^top -/ {blk++} blk==2 && $1 ~ /^[0-9]+$/ {
      c=$9; n=$12; gsub(",",".",c);
      if (n !~ /^(test-backend|claude|kworker|top|ksoftirq|rcu_|migration|kswapd|irq\/)/ && c+0 > 0) {sum+=c; if (c+0>5) big=big n ":" c " "}
    }
    END {printf "%s fremd=%.0f %s\n", t, sum, big}'
done
