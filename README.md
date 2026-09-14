# DeepSeek-V4-Flash auf AMD Strix Halo (Ryzen AI MAX+ 395)

Messprotokoll zum lokalen Betrieb von **DeepSeek-V4-Flash-0731** (284 B MoE, 13 B aktiv)
auf einem einzelnen AMD Ryzen AI MAX+ 395 mit 128 GB Unified Memory.

Stand: 2026-08-18

## TL;DR

* Das Modell **läuft** auf einer Einzelmaschine: 98,5 GiB belegt.
* **24–28 t/s Decode** mit dem Fork `Nathanw1014/strix-halo-llamacpp` + DSpark-Draft —
  gegenüber 11,94 t/s in Mainline-llama.cpp. Der Softwarestack macht den Unterschied,
  nicht die Konfiguration.
* **Spekulatives Decoding mit dem mitgelieferten `dspark`-Draft-Modell macht es langsamer**
  (7,7 statt 11,9 t/s) — die Faustregel für dichte Modelle greift bei MoE nicht.
* Die kursierenden „32 t/s auf Strix Halo" stammen **nicht** aus llama.cpp, sondern aus einem
  proprietären Server mit reduziertem Expert-Routing. Mit llama.cpp sind sie nicht reproduzierbar.
* Prefill: **127 t/s** (pp512). Das beworbene 1-M-Kontextfenster bleibt dennoch theoretisch —
  es wären ~2,2 Stunden reiner Prefill —, aber 32k/64k sind gut machbar.
* **Das HIP-Backend rechnete bis Anfang September still falsch** (Perplexity 727 statt 7,0 auf
  gfx1151) — bei identischem Durchsatz, also fuer jeden Benchmark unsichtbar. Auf aktuellem
  llama.cpp behoben. Siehe [HIP-Korrektheit](#hip-korrektheit-das-backend-rechnete-still-falsch).
* Das **HIP/ROCm-Backend kann das Modell nicht laden**, obwohl es 112 GiB meldet: es kommt
  nicht an GTT. Mit einem kleineren Modell laeuft HIP einwandfrei — es ist die Groesse,
  nicht das Backend. Vulkan/RADV nutzt VRAM+GTT als einen Pool und ist damit die praktische Wahl.
* **V4-Flash ist nicht bandbreitenlimitiert:** ein 27-B-Dense-Modell decodiert gleich schnell,
  obwohl es pro Token 3x mehr Bytes liest. Der MoE-Overhead frisst den theoretischen Vorteil.

## Hardware / Software

| | |
|---|---|
| APU | AMD Ryzen AI MAX+ 395, Radeon 8060S (gfx1151, RDNA 3.5) |
| Speicher | 128 GB unified |
| OS | Nobara 44, Kernel 7.1.4 |
| ROCm | 7.1 (HIP 7.1.52802) |
| llama.cpp | Commit b94041a, Vulkan- und HIP-Backend |
| Modell | `unsloth/DeepSeek-V4-Flash-0731-GGUF`, Quant `UD-IQ3_XXS` (97,6 GiB, 4 Splits) |
| Draft | `dspark-DeepSeek-V4-Flash-0731-BF16.gguf` (10,5 GiB) |

## Quant-Auswahl

Das Modell ist **quantization-aware trainiert** und liegt nativ in MXFP4 vor (die routed
Experts = 96 % der Gewichte). Weiter herunterzuquantisieren bedeutet also *doppelte*
Quantisierung — der Qualitätsabfall unter 4 bit ist steiler als bei üblichen bf16→Q4-Quants.

| Quant | Größe (GiB) | bei 128 GB unified |
|---|---|---|
| UD-IQ1_S | 76,9 | passt, aber unnötig verlustbehaftet |
| UD-Q2_K_XL | 90,2 | passt komfortabel |
| **UD-IQ3_XXS** | **97,1** | **gewählt** — größter Quant mit Platz für KV-Cache |
| UD-IQ3_S | 108,1 | zu knapp |
| UD-Q3_K_XL | 119,4 | nur mit Layer-Offload |
| UD-IQ4_XS | 127,3 | passt nicht |

## Speicherkonfiguration — der eigentliche Stolperstein

Bei Strix Halo ist ein **fest reservierter VRAM-Block die falsche Konfiguration**. Er ist
eine harte Wand, aus der die GPU nicht heraus kann, und nimmt dem System gleichzeitig RAM weg.
Richtig ist: UMA im BIOS klein halten und der GPU über **GTT** dynamischen Zugriff geben.

```bash
# GTT-Limit auf 110 GiB (Formel: Bytes / 4096 Pages)
sudo grubby --update-kernel=ALL \
  --args="ttm.pages_limit=28835840 ttm.page_pool_size=28835840"
```

`amdgpu.gttsize` gilt als deprecated. Nach dem Umstellen zeigt `mem_info_vram_total` nur noch
~1 GiB — das ist bei APUs normal, der nutzbare Speicher steckt in GTT.

### Gemessene BIOS-Varianten

| UMA-Einstellung | VRAM | System-RAM | GTT | Ergebnis |
|---|---|---|---|---|
| 96 GB | 96 GiB | 30 GiB | 15 GiB | **OOM-Absturz beim Laden** |
| Auto | 64 GiB | 62 GiB | 110 GiB | läuft |
| 512M (empfohlen) | ~1 GiB | ~125 GiB | 110 GiB | ehrliche Vulkan-Werte, ein Pool |

**Warnung zu `Auto`:** Vulkan meldet dann 174 GiB (64 VRAM + 110 GTT) — **mehr als physisch
verbaut**. llama.cpp glaubt diesen Wert und kann ins OOM laufen.

## Der OOM-Absturz

Mit UMA auf 96 GB blieben nur 30 GiB System-RAM. Der Ladevorgang riss die Maschine per
`global_oom` in einen Reboot:

```
kernel: oom-kill:constraint=CONSTRAINT_NONE,...,global_oom,task=(sd-pam)
kernel: Out of memory: Killed process 9535 (flm-real)
systemd-shutdown[1]: Syncing filesystems and block devices.
```

Bemerkenswert: **keine einzige amdgpu-Fehlermeldung**. Es war reiner RAM-Mangel durch
Page-Cache und Staging-Puffer beim Upload der 97,6 GiB — nicht die GPU.

**Lehre:** Vor dem Laden `MemAvailable` prüfen und bei < 10 GiB gar nicht erst starten.
Ein Watchdog (`memguard.sh` in diesem Repo) killt den llama-Prozess, bevor der globale
OOM-Killer die Maschine trifft.

## Messergebnisse

Konfiguration: `-ngl 999 -c 8192`, Vulkan/RADV-Backend, interaktive Läufe.

### Reproduzierbar (llama-bench, `-ngl 999 -p 512 -n 128 -r 2`)

| Backend | pp512 (Prefill) | tg128 (Decode) |
|---|---|---|
| **Vulkan / RADV** | **127,39 ± 2,67 t/s** | **11,94 ± 0,01 t/s** |
| HIP / ROCm | — | — (Modell laedt nicht, s.u.) |

### Interaktive Laeufe (llama-cli, `-ngl 999 -c 8192`)

| Konfiguration | Prefill | Decode | Belegung |
|---|---|---|---|
| ohne Draft-Modell | 25,8 t/s | **11,9 t/s** | 98,5 GiB (64 VRAM + 34,5 GTT) |
| mit `dspark`-Draft, `--spec-draft-n-max 5` | 21,5 t/s | **7,7 t/s** | 109,7 GiB (64 VRAM + 45,7 GTT) |

Die Prefill-Werte der interaktiven Laeufe sind **nicht** aussagekraeftig: bei sehr kurzen
Prompts dominiert der Overhead. Mit ordentlichem Batching (pp512) liegt der Prefill beim
Fuenffachen. Der Decode-Wert dagegen deckt sich exakt (11,9 vs. 11,94).

### Warum spekulatives Decoding hier bremst

Ein Draft-Modell lohnt sich nur, wenn es **pro Token deutlich billiger** ist als das
Hauptmodell. Auf bandbreitenlimitierter Hardware zählt dafür, wie viele Bytes pro Token
gelesen werden:

* **Hauptmodell:** 284 B total, aber nur 13 B aktiv bei ~3 bpw → **~5 GiB/Token**
* **dspark-Draft:** 10,5 GiB BF16, dense → **~10,5 GiB/Token**

Das Draft-Modell ist also rund **doppelt so teuer** wie das MoE, das es beschleunigen soll.
Selbst bei perfekter Trefferquote bliebe kein Gewinn. Die aus dichten Modellen bekannte
Faustregel „Draft-Modell = größter Hebel" kehrt sich bei MoE mit kleinem aktivem Anteil um.

## Einordnung der kursierenden Benchmarks

Die vielzitierten **32 t/s decode** auf dem Ryzen AI MAX+ 395 stammen von Lucebox und wurden
**nicht mit llama.cpp** gemessen, sondern mit einem proprietären `dflash_server`, eigenem
Mixed-Precision-Format (~2,88 bpw) und reduziertem Expert-Routing (`--ds4-expert-top-k 4`
statt 6). Mit llama.cpp sind diese Zahlen nicht reproduzierbar.

## Download

`HF_HUB_ENABLE_HF_TRANSFER` ist in huggingface-hub 1.x **wirkungslos** geworden. Der schnelle
Pfad heißt jetzt Xet:

```bash
export HF_XET_HIGH_PERFORMANCE=1
hf download unsloth/DeepSeek-V4-Flash-0731-GGUF \
  --include "UD-IQ3_XXS/*" --local-dir ./DeepSeek-V4-Flash-0731
```

Über Gbit erreicht: **78 MB/s**, 108 GiB in 25 Minuten.

Achtung: Xet legt Chunks zusätzlich in `~/.cache/huggingface` ab, der Platzbedarf ist während
des Downloads also grob doppelt. Beim Aufräumen **nur** das `.cache`-Residuum *im local-dir*
löschen — `~/.cache/huggingface/hub` enthält echte Modelle anderer Anwendungen.

## HIP/ROCm-Backend

Gebaut mit `rocblas-devel` und `hipblas-devel`:

```bash
cmake -B build-hip -DGGML_HIP=ON -DAMDGPU_TARGETS=gfx1151 -DGPU_TARGETS=gfx1151 \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=hipcc -DCMAKE_CXX_COMPILER=hipcc
```

Der Build laeuft durch, das Laden scheitert jedoch:

```
Device 0: AMD Radeon 8060S Graphics, gfx1151, VRAM: 112640 MiB
llama_bench: error: failed to load model
```

Auch mit `GGML_CUDA_ENABLE_UNIFIED_MEMORY=1` bleibt der Fehler identisch.

HIP meldet also 112 GiB (VRAM + GTT), kann aber nur den echten VRAM-Block von 64 GiB
allozieren — und darin haben die 97 GiB des Modells keinen Platz. Vulkan/RADV behandelt
VRAM und GTT dagegen als einen Pool und laedt problemlos.

### Gegenprobe mit einem kleineren Modell

Mit einem Modell, das in den echten VRAM passt (Qwen3.8, 27 B dense, Q4_K_M, 15,65 GiB),
laeuft HIP einwandfrei:

| Backend | pp512 | tg128 |
|---|---|---|
| HIP / ROCm | **338,13 ± 2,83 t/s** | 11,37 ± 0,01 t/s |
| Vulkan / RADV | 330,94 ± 1,13 t/s | **12,08 ± 0,01 t/s** |

Der Fehlschlag lag also **nicht am HIP-Backend**, sondern allein an der Modellgroesse.
Der Backend-Unterschied selbst ist klein und geht in beide Richtungen: HIP beim Prefill
+2 %, Vulkan beim Decode +6 %.

**Fazit: Vulkan/RADV ist auf dieser APU die praktische Wahl** — nicht wegen der Rohleistung,
sondern weil nur es VRAM und GTT als einen Pool nutzt und damit Modelle jenseits des
VRAM-Blocks ueberhaupt erst ermoeglicht.

## HIP-Korrektheit: das Backend rechnete still falsch

Nachtrag vom **2026-09-14**. Der Abschnitt oben kommt zu dem Schluss, Vulkan/RADV sei die
praktische Wahl, weil nur es VRAM und GTT als einen Pool nutzt. Es gibt dafuer einen zweiten,
davon unabhaengigen Grund: **der HIP-Pfad hat auf gfx1151 monatelang still falsch gerechnet.**

Gemessen mit `llama-perplexity` auf `Qwen3.6-35B-A3B-UD-IQ4_XS`, `-c 2048 --chunks 6`
(Chunk-Laenge groesser als das `n_ubatch`-Default von 512):

| llama.cpp | Backend | Perplexity |
|---|---|---|
| `9731ad3` (2026-08-18) | HIP | **727,00 ± 61,66** |
| `bfdc321` (2026-09-14) | HIP | 7,0098 ± 0,2464 |
| `bfdc321` (2026-09-14) | Vulkan | 7,0168 ± 0,2471 |

Faktor 104. Der aeltere Build laedt, laeuft, gibt Tokens aus und meldet keinen Fehler — er
produziert nur Unsinn. Auf dem aktuellen Master ist das behoben.

### Warum kein Benchmark das findet

| llama.cpp | Backend | pp4096 | tg200 |
|---|---|---|---|
| `9731ad3` | HIP | 965,3 ± 5,3 | 49,95 ± 0,12 |
| `bfdc321` | HIP | 966,9 ± 3,7 | 50,36 ± 0,07 |

**Der kaputte Build ist exakt so schnell wie der korrekte.** Der Fehler kostet keine Leistung,
er kostet Richtigkeit — und `llama-bench` misst nur Leistung. Wer hier ausschliesslich tok/s
vergleicht, sieht nichts.

Zugehoerige Issues, alle fuer gfx1151 zwischen Ende August und Anfang September eingereicht:
[#28211](https://github.com/ggml-org/llama.cpp/issues/28211) (offen, falsche Logits bei Prompts
ueber `n_ubatch`), [#28113](https://github.com/ggml-org/llama.cpp/issues/28113) (MoE-Modelle
geben nur noch Satzzeichen aus), [#28537](https://github.com/ggml-org/llama.cpp/issues/28537)
(Batch-Decoding korrumpiert Logits).

### Was ein Monat Software bringt

| llama.cpp | Backend | pp4096 | tg200 |
|---|---|---|---|
| `9731ad3` | Vulkan | 982,8 ± 2,2 | 62,23 ± 0,03 |
| `bfdc321` | Vulkan | **1037,4 ± 3,1** | **63,07 ± 0,05** |

Vulkan gewinnt durch das Update 5,6 % Prefill und war durchgehend korrekt. HIP gewinnt keine
Leistung, sondern Korrektheit. Und Vulkan bleibt auch danach **25 % vor HIP** beim Decode
(63,07 gegen 50,36) — die Empfehlung oben gilt unveraendert, jetzt aus zwei Gruenden.

### Konsequenz fuer die Zahlen in diesem Repo: nachgeprueft

Die HIP-Werte der Gegenprobe oben (338,13 pp512 / 11,37 tg128) **waren betroffen**. Mit
demselben Modell nachgemessen (Qwen3.8 27B dense Q4_K_M, 15,65 GiB):

| Build | Backend | Perplexity | pp512 | tg128 |
|---|---|---|---|---|
| August | HIP | **663,18 ± 56,5** | 318,20 ± 0,13 | 11,32 ± 0,01 |
| master `bfdc321` | HIP | 5,5706 ± 0,178 | 315,77 ± 2,10 | 11,86 ± 0,03 |
| August | Vulkan | — | 272,52 ± 2,02 | 12,27 ± 0,00 |
| master `bfdc321` | Vulkan | 5,5667 ± 0,179 | **317,25 ± 3,15** | **12,29 ± 0,01** |

Der Fehler trifft also **auch dichte Modelle**, nicht nur MoE — Faktor 119 hier, Faktor 104 beim
Qwen3.6-MoE. Die beiden korrekten Backends stimmen auf 0,07 % ueberein.

**Damit faellt die Aussage „HIP beim Prefill +2 %":** der HIP-Vorsprung war ein Artefakt des
fehlerhaften Backends. Korrekt gemessen ist der Prefill ein **Gleichstand** (315,8 gegen 317,3,
ueberlappende Fehlerbalken) und Vulkan fuehrt beim Decode mit **+3,6 %**. Die Empfehlung
zugunsten von Vulkan bleibt also — sie war nur aus dem falschen Grund knapp.

Nicht betroffen sind die Vulkan-Werte und alle Aussagen zu GTT, Quant-Auswahl, MoE-Overhead und
spekulativem Decoding: die wurden auf dem Vulkan-Pfad gemessen.

**Praxisregel fuer diese Hardware:** vor jeder GPU-Messung zuerst `llama-perplexity` gegen ein
zweites Backend laufen lassen. Ein stiller Korrektheitsfehler ist auf gfx1151 wahrscheinlicher
als ein Leistungsproblem, und er ist in Durchsatzzahlen unsichtbar.

## SMT: 16 Threads schlagen 32

Nachtrag vom **2026-09-14**. Strix Halo hat 16 Kerne und 32 Threads. Bei CPU-seitiger Inferenz
ist **die Kernzahl die richtige Thread-Zahl**, nicht die Thread-Zahl — und der Unterschied ist
kein Rundungsfehler.

Gemessen mit Qwen3.6-35B-A3B auf ruhiger Maschine, zwei unabhaengige Engines:

| Engine | 16 Threads | 32 Threads | Gewinn durch 16 |
|---|---|---|---|
| colibri (CPU-Pfad, int4-gs64) | 28,68 tok/s | 25,58 tok/s | **+12,1 %** |
| llama.cpp (`-ngl 0`, UD-IQ4_XS) | **24,42 ± 0,03** | 17,76 ± 3,06 | **+37,5 %** |

Beide zeigen denselben Abfall, also ist es eine Eigenschaft der Hardware und keine Eigenheit
einer Engine. Aufschlussreich ist auch die Streuung: llama.cpps 32-Thread-Lauf ist nicht nur
langsamer, sondern mit ±3,06 gegen ±0,03 auch deutlich unruhiger. Zwei SMT-Geschwister
konkurrieren um die Ladeeinheiten desselben Kerns; bei speichernaher Arbeit bringt der zweite
Thread keine zusaetzliche Arbeit durch, kostet aber Cache und Planbarkeit.

Voller Sweep mit colibri:

| Threads | 8 | 12 | **16** | 20 | 24 | 32 |
|---|---|---|---|---|---|---|
| tok/s | 20,16 | 27,63 | **28,68** | 26,37 | 27,71 | 25,58 |

Die Punkte bei 20 und 24 streuen um etwa ±1, der Verlauf dazwischen ist also kein sauberer Bogen.
Das Maximum bei 16 und der Abfall bei 32 sind aber belastbar.

```bash
# llama.cpp
llama-bench -m modell.gguf -t 16 ...
llama-server -m modell.gguf -t 16 ...

# colibri: nichts setzen. Der eingebaute OMP-Self-Tune findet 16 von allein.
# COLI_NO_OMP_TUNE=1 schaltet ihn ab und kostet ~10 % — die Empfehlung dafuer
# steht in docs/vulkan.md und gilt nur fuer den Vulkan-Pfad, nicht fuer CPU-Laeufe.
```

Der letzte Punkt war ein eigener Stolperstein: Die colibri-Vulkan-Dokumentation empfiehlt
`COLI_NO_OMP_TUNE=1`, weil spinnende Worker dort den asynchronen I/O-Pool aushungern. Auf einem
reinen CPU-Lauf gilt das nicht — dort schaltet man damit nur die Automatik ab, die ohnehin das
Richtige tut.

## MoE-Overhead: der interessanteste Befund

| Modell | Groesse | Prefill | Decode |
|---|---|---|---|
| Qwen3.8, 27 B **dense**, Q4_K_M | 15,65 GiB | 331 t/s | **12,08 t/s** |
| DeepSeek-V4-Flash, 284 B **MoE**, IQ3_XXS | 97,05 GiB | 127 t/s | **11,94 t/s** |

Ein 284-B-Modell decodiert praktisch gleich schnell wie ein 27-B-Modell — das ist der
MoE-Vorteil in Reinform. Es ist zugleich dessen Grenze: **rein nach Bandbreite gerechnet
muesste V4-Flash dreimal schneller sein.**

* V4-Flash: 13 B aktiv bei 3,06 bpw → **~4,6 GiB/Token**
* Qwen dense: alle 27,3 B aktiv bei Q4_K_M → **~15,65 GiB/Token**

Dass der Vorsprung ausbleibt — und der Prefill sogar 2,6× langsamer ist — zeigt: **V4-Flash
ist auf dieser Hardware nicht bandbreitenlimitiert.** Der MoE-Overhead (Expert-Routing,
schlechte Speicherlokalitaet, viele kleine Matrizen statt grosser GEMMs) frisst den
theoretischen Vorteil auf.

Das erklaert nachtraeglich auch, **warum spekulatives Decoding nichts brachte**: wenn nicht
die Bandbreite der Engpass ist, hilft es nicht, weniger Bytes zu lesen.

## Offene Punkte

* ~~UMA auf `512M` gegenpruefen~~ — erledigt, GTT ist nicht langsamer als VRAM (siehe BENCHMARKS.md)
* RADV-Fork `Nathanw1014/strix-halo-llamacpp` testen — belegt ~18,5 t/s plain, 21–27 mit Draft
* ~~llama.cpp aktualisieren (b94041a → aktuell)~~ — gemessen am 2026-09-14, siehe HIP-Korrektheit
* ~~HIP-Werte der Gegenprobe auf Korrektheit nachpruefen~~ — erledigt, sie waren betroffen (PPL 663 statt 5,57); korrigierte Tabelle im Abschnitt HIP-Korrektheit
* ~~Thread-Zahl gegenpruefen~~ — erledigt, 16 statt 32 bringt 12-38 %, siehe SMT-Abschnitt
* Prefill-Verhalten bei groesseren Kontexten (32k, 64k) vermessen
* Groesseren Quant `UD-Q3_K_XL` (119 GiB) testen — passt seit der UMA-Umstellung
