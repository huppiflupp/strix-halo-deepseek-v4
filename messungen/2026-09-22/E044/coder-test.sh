#!/bin/bash
# E044 step 4: the exact coder command line on a test port - today's coder build vs. the new build with the concat kernel.
cd "$(dirname "$0")"
S="llama-gptoss.socket llama-coder.socket llama-qwen.socket llama-qwen36.socket llama-qwen36-familie.socket"
systemctl --user stop $S llama-gptoss llama-coder llama-qwen llama-qwen36 llama-qwen36-familie
T=/tmp/claude-1000/-home-seeas/427f6ada-002c-402d-9339-95dcfca48278/scratchpad/slots-e044; mkdir -p $T
trap 'kill $PID 2>/dev/null; systemctl --user start $S; rm -rf $T' EXIT
. ../../harness/guard.sh
M=$HOME/models/Qwen3.8-27B/Qwen3.8-27B-Q4_K_M.gguf
one() { wait_for_memory; sleep 20; echo "##### $1"
  env GGML_VK_LAB_MM_PACKED=2 $2 $3 -m $M -ngl 999 -c 131072 -np 1 -fa 1 --jinja --spec-type draft-mtp --spec-draft-n-max 4 --slot-save-path $T --host 127.0.0.1 --port 18198 > srv-coder-$1.log 2>&1 & PID=$!
  for i in $(seq 150); do curl -sf http://127.0.0.1:18198/health >/dev/null && break; sleep 2; done
  python3 - <<'PY'
import json, urllib.request
req = urllib.request.Request("http://127.0.0.1:18198/v1/chat/completions", json.dumps({"messages": [{"role": "user", "content": "Write a Python function that returns the n-th Fibonacci number iteratively. Code only."}], "max_tokens": 300, "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}}).encode(), {"Content-Type": "application/json"})
d = json.load(urllib.request.urlopen(req, timeout=600)); t = d.get("timings", {})
print("  answer:", d["choices"][0]["message"]["content"].strip().replace("\n", " ⏎ ")[:240])
print("  generation %.1f tok/s, draft accepted %s of %s" % (t.get("predicted_per_second", 0), t.get("draft_n_accepted"), t.get("draft_n")))
PY
  python3 ../E042/probe.py 18198 probe-coder-$1.json; echo "  GTT with model loaded: $(gtt_gib) GiB"; kill $PID; wait $PID 2>/dev/null; }
{
one old "X=1" $HOME/llama-serve/coder/build-vulkan/bin/llama-server
one new "GGML_VK_LAB_CONCAT_T=1" $HOME/llama-serve/hybrid/build-vulkan/bin/llama-server
python3 - <<'PY'
import json, math
A, B = json.load(open("probe-coder-old.json")), json.load(open("probe-coder-new.json"))
nan = sum(any(v is None for v in x.values()) for x in B); same = 0; kl = []
for x, y in zip(A, B):
    same += max(x, key=x.get) == max(y, key=y.get); ks = [k for k in x if k in y]; kl.append(sum(math.exp(x[k]) * (x[k] - y[k]) for k in ks))
print(f"##### probes: new build NaN {nan}/48 · same top token as today's coder build {same}/48 · mean KLD over shared top-20 {sum(kl)/len(kl):.6f} · max {max(kl):.5f}")
PY
echo "BO_VA lines: $(journalctl -k -b --no-pager | grep -c BO_VA)"; echo done
} > coder-result.txt 2>&1
