#!/bin/bash
# E048: deploy test for the coder: new frozen build ~/llama-serve/hybrid2 (= hybrid + column-major B loads).
# 1) perplexity/KLD with and without BCOL on the new build; 2) exact coder command line on a test port, today's service build
# (hybrid, CONCAT_T) vs. hybrid2 (CONCAT_T + BCOL): code answer, MTP, 48 first-token probes; 3) llama-bench both. 6 model loads.
cd "$(dirname "$0")"
S="llama-gptoss.socket llama-coder.socket llama-qwen.socket llama-qwen36.socket llama-qwen36-familie.socket"
systemctl --user stop $S llama-gptoss llama-coder llama-qwen llama-qwen36 llama-qwen36-familie
T=/tmp/claude-1000/-home-seeas/427f6ada-002c-402d-9339-95dcfca48278/scratchpad; mkdir -p $T/slots-e048
trap 'kill $PID 2>/dev/null; systemctl --user start $S; rm -rf $T/e048.kld $T/slots-e048' EXIT
. ../../harness/guard.sh
OLD=$HOME/llama-serve/hybrid/build-vulkan/bin; NEW=$HOME/llama-serve/hybrid2/build-vulkan/bin
K=$HOME/models/Qwen3.8-27B/Qwen3.8-27B-Q4_K_M.gguf
export GGML_VK_LAB_MM_PACKED=2 GGML_VK_LAB_CONCAT_T=1
A="-m $K -f $HOME/bench/llamacpp-tuning/wiki.test.raw -c 2048 -ub 2048 --chunks 8 -ngl 999 -fa on"
fmt() { grep -E '^\| *qwen|rror' | awk -F'|' '{print "  " $(NF-2) "|" $(NF-1) " tok/s"}'; }
one() { wait_for_memory; sleep 20; echo "##### server $1"
  env $2 $3/llama-server -m $K -ngl 999 -c 131072 -np 1 -fa 1 --jinja --spec-type draft-mtp --spec-draft-n-max 4 --slot-save-path $T/slots-e048 --host 127.0.0.1 --port 18198 > srv-$1.log 2>&1 & PID=$!
  for i in $(seq 150); do curl -sf http://127.0.0.1:18198/health >/dev/null && break; sleep 2; done
  python3 - <<'PY'
import json, urllib.request
req = urllib.request.Request("http://127.0.0.1:18198/v1/chat/completions", json.dumps({"messages": [{"role": "user", "content": "Write a Python function that returns the n-th Fibonacci number iteratively. Code only."}], "max_tokens": 300, "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}}).encode(), {"Content-Type": "application/json"})
d = json.load(urllib.request.urlopen(req, timeout=600)); t = d.get("timings", {}); c = d["choices"][0]["message"]["content"]
open("answer-" + __import__("os").environ.get("TAG", "x") + ".txt", "w").write(c)
print("  answer chars:", len(c), "| generation %.1f tok/s, draft accepted %s of %s" % (t.get("predicted_per_second", 0), t.get("draft_n_accepted"), t.get("draft_n")))
PY
  python3 ../E042/probe.py 18198 probe-$1.json; echo "  GTT with model loaded: $(gtt_gib) GiB"; kill $PID; wait $PID 2>/dev/null; }
{
wait_for_memory; sleep 20; $NEW/llama-perplexity $A --kl-divergence-base $T/e048.kld 2>&1 | grep -oE 'Final estimate.*'
wait_for_memory; sleep 20; echo "== new build, BCOL on vs off"
GGML_VK_LAB_MM_BCOL=1 $NEW/llama-perplexity $A --kl-divergence-base $T/e048.kld --kl-divergence 2>&1 | grep -E 'Mean PPL\(Q\) |Mean +KLD|Maximum KLD|Same top p' | sed 's/^ */   /'
TAG=old one old "X=1" $OLD
TAG=new one new "GGML_VK_LAB_MM_BCOL=1" $NEW
cmp -s answer-old.txt answer-new.txt && echo "code answers identical" || echo "code answers DIFFER"
python3 - <<'PY'
import json, math
A, B = json.load(open("probe-old.json")), json.load(open("probe-new.json"))
nan = sum(any(v is None for v in x.values()) for x in B); same = sum(max(x, key=x.get) == max(y, key=y.get) for x, y in zip(A, B))
kl = [sum(math.exp(x[k]) * (x[k] - y[k]) for k in x if k in y) for x, y in zip(A, B)]
print(f"##### probes: new NaN {nan}/48 · same top token {same}/48 · mean KLD {sum(kl)/len(kl):.6f} · max {max(kl):.5f}")
PY
for v in "OLD" "NEW"; do wait_for_memory; sleep 20; echo "== llama-bench $v, -ub 512"
  if [ $v = OLD ]; then $OLD/llama-bench -m $K -ngl 999 -fa 1 -ub 512 -b 512 -p 512,2048 -n 0 -r 3 -o md 2>&1 | fmt
  else GGML_VK_LAB_MM_BCOL=1 $NEW/llama-bench -m $K -ngl 999 -fa 1 -ub 512 -b 512 -p 512,2048 -n 0 -r 3 -o md 2>&1 | fmt; fi; done
echo "BO_VA lines: $(journalctl -k -b --no-pager | grep -c BO_VA)"; echo done
} > result.txt 2>&1
