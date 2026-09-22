#!/bin/bash
# E044 step 2: the exact family command line on a test port - fork (today's service) and new build with the concat kernel.
# Text + image request (MTP, vision module, thinking budget), then 48 first-token probes each, compared.
cd "$(dirname "$0")"
while systemctl --user is-active -q lab-e044a; do sleep 15; done
S="llama-gptoss.socket llama-coder.socket llama-qwen.socket llama-qwen36.socket llama-qwen36-familie.socket"
systemctl --user stop $S llama-gptoss llama-coder llama-qwen llama-qwen36 llama-qwen36-familie
trap 'kill $PID 2>/dev/null; systemctl --user start $S' EXIT
. ../../harness/guard.sh
D=$HOME/models/Qwen3.6-35B-A3B-MTP
one() { # name, env, server binary
  wait_for_memory; sleep 20; echo "##### $1"
  env $2 $3 -m $D/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf --mmproj $D/mmproj-F16.gguf -ngl 999 -c 131072 -np 2 -fa 1 -ub 2048 --jinja --spec-type draft-mtp \
     --reasoning-budget 1000 --reasoning-budget-message " Genug nachgedacht, ich antworte jetzt." --host 127.0.0.1 --port 18198 > srv-$1.log 2>&1 & PID=$!
  for i in $(seq 120); do curl -sf http://127.0.0.1:18198/health >/dev/null && break; sleep 2; done
  python3 chat.py 18198; python3 ../E042/probe.py 18198 probe-$1.json; echo "  GTT with model loaded: $(gtt_gib) GiB"
  kill $PID; wait $PID 2>/dev/null
}
{
one fork "X=1" $HOME/strix-fork/vulkan/llama-server
one new  "GGML_VK_LAB_FA_WAVE32=1 GGML_VK_LAB_FA_VT=1 GGML_VK_LAB_CONCAT_T=1" $HOME/llama-serve/hybrid/build-vulkan/bin/llama-server
python3 - <<'PY'
import json, math
A, B = json.load(open("probe-fork.json")), json.load(open("probe-new.json"))
nan = sum(any(v is None for v in x.values()) for x in B); same = 0; kl = []
for x, y in zip(A, B):
    if any(v is None for v in list(x.values()) + list(y.values())): continue
    same += max(x, key=x.get) == max(y, key=y.get); ks = [k for k in x if k in y]; kl.append(sum(math.exp(x[k]) * (x[k] - y[k]) for k in ks))
print(f"##### probes: new build NaN {nan}/48 · same top token as the fork {same}/48 · mean KLD over shared top-20 {sum(kl)/len(kl):.5f} · max {max(kl):.4f}")
PY
echo "BO_VA lines: $(journalctl -k -b --no-pager | grep -c BO_VA)"; echo done
} > server-result.txt 2>&1
