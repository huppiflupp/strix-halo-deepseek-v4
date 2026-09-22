#!/usr/bin/env python3
import sys, json, math
def cmp(a, b):
    A, B = json.load(open(a)), json.load(open(b)); same = 0; kl = []; dp = []
    for x, y in zip(A, B):
        tx = max(x, key=x.get); ty = max(y, key=y.get); same += tx == ty
        ks = [k for k in x if k in y]; kl.append(sum(math.exp(x[k]) * (x[k] - y[k]) for k in ks)); dp.append(abs(math.exp(x[tx]) - math.exp(y.get(tx, -30))))
    return f"same top token {same}/{len(A)} · mean KLD over shared top-20 {sum(kl)/len(kl):.5f} · max {max(kl):.4f} · mean |Δp(top)| {sum(dp)/len(dp):.4f}"
for a, b in (("fork-default.json", "fork-default-2.json"), ("fork-default.json", "fork-w64.json"), ("new-build.json", "fork-default.json"), ("new-build.json", "fork-w64.json")):
    print(f"{a[:-5]:>14s} vs {b[:-5]:<16s}: {cmp(a, b)}")
