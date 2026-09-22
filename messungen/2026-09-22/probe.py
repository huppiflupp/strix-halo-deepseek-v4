#!/usr/bin/env python3
"""First-token distributions for N prompts from a llama-server (/completion, n_probs). usage: probe.py port out.json"""
import sys, json, urllib.request
port, out = sys.argv[1], sys.argv[2]
t = open(__import__('os').environ.get('WIKI', 'wiki.test.raw')).read()
res = []
for i in range(48):
    n = (300, 1200, 2500, 6000)[i % 4]                       # ~75 .. 1500 tokens: below and above the expert-batch thresholds
    p = t[20000 + i * 9000: 20000 + i * 9000 + n]
    req = urllib.request.Request(f"http://127.0.0.1:{port}/completion", json.dumps({"prompt": p, "n_predict": 1, "temperature": 0, "n_probs": 20, "cache_prompt": False}).encode(), {"Content-Type": "application/json"})
    d = json.load(urllib.request.urlopen(req, timeout=300))
    c = d["completion_probabilities"][0]
    top = c.get("top_logprobs") or c.get("top_probs") or c.get("probs")
    res.append({(x.get("token") or x.get("tok_str")): (x["logprob"] if "logprob" in x else __import__("math").log(max(x["prob"], 1e-30))) for x in top})
json.dump(res, open(out, "w")); print("ok", len(res))
