#!/usr/bin/env python3
"""Auswertung des ABBA-Sweeps #28613: gepaarte Verhaeltnisse PR/master je Runde,
Median, 95-%-Bootstrap-KI, Wilcoxon-Vorzeichen-Rang (exakt, zweiseitig).
Markiert Runden mit Fremdlast (monitor2: fremde CPU > 50 %, sonst load > 7)."""
import glob, math, random, re, sys, itertools, datetime as dt
from pathlib import Path

D = Path(__file__).parent
TYPES = ['q4_0', 'q8_0', 'q2_K', 'q3_K', 'q4_K', 'q6_K']
OS = [512, 1024, 2048, 4096, 8192, 17408]
PR_THRESH = {'q4_0': 6, 'q8_0': 4, 'q2_K': 4, 'q3_K': 3, 'q4_K': 2, 'q6_K': 3}  # PR: MMVQ bis ne11 <= x

def load(f):
    return {l.split('\t')[0]: float(l.split('\t')[1]) for l in open(f) if '\t' in l}

rounds = sorted({int(re.search(r'-r(\d+)', f).group(1)) for f in glob.glob(str(D / 'master-r*.tsv'))})
rounds = [r for r in rounds if (D / f'pr28613-r{r}.tsv').exists()]
M = {r: load(D / f'master-r{r}.tsv') for r in rounds}
P = {r: load(D / f'pr28613-r{r}.tsv') for r in rounds}

# Fremdlast je Block aus waechter.log + monitor.log
def parse_t(s): return dt.datetime.strptime(s, '%H:%M:%S')
blocks = {}
if (D / 'waechter.log').exists():
    starts = {}
    for l in open(D / 'waechter.log'):
        m = re.match(r'(\S+) start r(\d+) (\S+)', l)
        if m: starts[(int(m[2]), m[3])] = parse_t(m[1])
        m = re.match(r'(\S+) ende  r(\d+) (\S+)', l)
        if m and (int(m[2]), m[3]) in starts: blocks[(int(m[2]), m[3])] = (starts[(int(m[2]), m[3])], parse_t(m[1]))
mon = []
if (D / 'monitor.log').exists():
    for l in open(D / 'monitor.log'):
        m = re.match(r'(\S+) load=(\S+)', l)
        if m: mon.append((parse_t(m[1]), float(m[2])))
mon2 = []
if (D / 'monitor2.log').exists():
    for l in open(D / 'monitor2.log'):
        m = re.match(r'(\S+) fremd=(\S+)', l)
        if m: mon2.append((parse_t(m[1]), float(m[2])))
# Fremdlast: monitor2 (aktuelle CPU fremder Prozesse) > 50 % eines Kerns; fuer Zeitraeume vor
# monitor2 ersatzweise load > 7 (test-backend-ops allein erzeugt bis ~5.4).
gestoert = set()
for (r, w), (a, b) in sorted(blocks.items()):
    f2 = [x for t, x in mon2 if a <= t <= b]
    f1 = [ld for t, ld in mon if a <= t <= b]
    if f2 and (not mon2 or mon2[0][0] <= a):
        peak, lim, what = max(f2), 50.0, 'fremd-CPU %'
    else:
        peak, lim, what = max(f1 or [0]), 7.0, "load"
    if peak > lim: gestoert.add(r)
    print(f"Block r{r} {w:8s} {a:%H:%M:%S}-{b:%H:%M:%S}  max {what} {peak:.1f}{'  <-- FREMDLAST' if peak > lim else ''}")
use = [r for r in rounds if r not in gestoert] if '--alle' not in sys.argv else rounds
print(f"\nRunden gesamt {rounds}, ausgewertet {use} (gestoert: {sorted(gestoert)})\n")

def wilcoxon_p(d):
    d = [x for x in d if x != 0]
    n = len(d)
    if n == 0: return 1.0
    ranks = sorted(range(n), key=lambda i: abs(d[i]))
    rk = [0] * n
    for pos, i in enumerate(ranks): rk[i] = pos + 1
    w = sum(rk[i] for i in range(n) if d[i] > 0)
    tot = n * (n + 1) // 2
    cnt = 0; ext = 0
    for signs in itertools.product([0, 1], repeat=n):
        s = sum(r for r, g in zip(range(1, n + 1), signs) if g)
        cnt += 1
        if abs(s - tot / 2) >= abs(w - tot / 2) - 1e-9: ext += 1
    return ext / cnt

def boot_ci(x, B=4000, seed=1):
    rnd = random.Random(seed); n = len(x)
    med = sorted(sorted(rnd.choice(x) for _ in range(n))[n // 2] if n % 2 else
                 (lambda s: (s[n//2-1] + s[n//2]) / 2)(sorted(rnd.choice(x) for _ in range(n))) for _ in range(B))
    return med[int(0.025 * B)], med[int(0.975 * B)]

def median(x):
    s = sorted(x); n = len(s)
    return s[n // 2] if n % 2 else (s[n//2-1] + s[n//2]) / 2

res = {}
for t in TYPES:
    for o in OS:
        for n in range(1, 9):
            k = f'{t}_o{o}_n{n}'
            lr = [math.log(P[r][k] / M[r][k]) for r in use]
            lo, hi = boot_ci(lr)
            res[k] = (math.exp(median(lr)), math.exp(lo), math.exp(hi), wilcoxon_p(lr))

# Nullkontrolle ne11=1 (beide Builds nutzen MMVQ)
null = [res[f'{t}_o{o}_n1'] for t in TYPES for o in OS]
fp = sum(1 for m, lo, hi, p in null if (lo > 1 or hi < 1))
print(f"Nullkontrolle ne11=1: {fp}/{len(null)} Formen mit KI ohne 1 "
      f"(Median |Effekt| {100*median([abs(m-1) for m,*_ in null]):.1f} %)\n")

def cell(k):
    m, lo, hi, p = res[k]
    sig = (lo > 1.0 or hi < 1.0) and p <= 0.05 and abs(m - 1) >= 0.05
    return f"{m:5.2f}{'*' if sig else ' '}"

print("Zellen: Median-Zeitverhaeltnis PR/master (>1 = PR langsamer); * = 95-%-KI ohne 1, Wilcoxon p<=0.05 und |Effekt|>=5 %")
print(f"(n = {len(use)} gepaarte Runden; kleinstes moegliches Wilcoxon-p = {2/2**len(use):.3f})")
for t in TYPES:
    print(f"\n{t}  (PR: MMVQ bis ne11 <= {PR_THRESH[t]}, master bis 8)")
    print("ne01    " + " ".join(f"{n:>6}" for n in range(1, 9)))
    for o in OS:
        print(f"{o:<7} " + " ".join(cell(f'{t}_o{o}_n{n}') for n in range(1, 9)))

with open(D / 'ergebnis.tsv', 'w') as f:
    f.write("form\tmedian_ratio\tci_lo\tci_hi\twilcoxon_p\n")
    for k, (m, lo, hi, p) in res.items():
        f.write(f"{k}\t{m:.4f}\t{lo:.4f}\t{hi:.4f}\t{p:.4f}\n")
