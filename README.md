# DeepSeek-V4-Flash auf AMD Strix Halo (Ryzen AI MAX+ 395)

Messprotokoll zum lokalen Betrieb von **DeepSeek-V4-Flash-0731** (284 B MoE, 13 B aktiv)
auf einem einzelnen AMD Ryzen AI MAX+ 395 mit 128 GB Unified Memory.

Stand: 2026-08-18, **Nachtrag 2026-09-20** (siehe
[Nachtrag](#nachtrag-2026-09-20-was-ein-monat-fork-entwicklung-geaendert-hat) — drei Aussagen
unten sind damit ueberholt)

## TL;DR

* Das Modell **läuft** auf einer Einzelmaschine: 98,5 GiB belegt.
* **24–28 t/s Decode** mit dem Fork `Nathanw1014/strix-halo-llamacpp` + DSpark-Draft —
  gegenüber 11,94 t/s in Mainline-llama.cpp. Der Softwarestack macht den Unterschied,
  nicht die Konfiguration.
* ~~**Spekulatives Decoding mit dem mitgelieferten `dspark`-Draft-Modell macht es langsamer**
  (7,7 statt 11,9 t/s)~~ — **ueberholt am 2026-09-20**: mit dem eigenen Spekulationstyp
  `--spec-type draft-dspark` bringt es **+24 %** (22,8 statt 18,4 t/s), und zwar bei
  Entwurfslaenge 2, nicht 1. Siehe Nachtrag.
* Die kursierenden „32 t/s auf Strix Halo" stammen **nicht** aus llama.cpp, sondern aus einem
  proprietären Server mit reduziertem Expert-Routing. Mit llama.cpp sind sie nicht reproduzierbar.
* ~~Prefill: **127 t/s** (pp512).~~ **Ueberholt**: heute **207 bis 234 Token/s** bei leerem
  Kontext und **175 Token/s** noch bei 32k Kontext. (Die urspruengliche Aufteilung "+24 %
  davon durch die MoE-Kernel" ist zurueckgezogen, siehe Nachtrag Abschnitt 2.) Das beworbene 1-M-Kontextfenster bleibt dennoch theoretisch —
  es wären ~2,2 Stunden reiner Prefill —, aber 32k/64k sind gut machbar.
* **Das HIP-Backend rechnete bis Anfang September still falsch** (Perplexity 727 statt 7,0 auf
  gfx1151) — bei identischem Durchsatz, also fuer jeden Benchmark unsichtbar. Auf aktuellem
  llama.cpp behoben. Siehe [HIP-Korrektheit](#hip-korrektheit-das-backend-rechnete-still-falsch).
* Das **HIP/ROCm-Backend kann das Modell nicht laden**, obwohl es 112 GiB meldet: es kommt
  nicht an GTT. Mit einem kleineren Modell laeuft HIP einwandfrei — es ist die Groesse,
  nicht das Backend. Vulkan/RADV nutzt VRAM+GTT als einen Pool und ist damit die praktische Wahl.
* **V4-Flash ist nicht bandbreitenlimitiert:** ein 27-B-Dense-Modell decodiert gleich schnell,
  obwohl es pro Token 3x mehr Bytes liest. Der MoE-Overhead frisst den theoretischen Vorteil.

* **Nebenmessung gpt-oss-120b** auf derselben Maschine (20.09.2026): `-ub 2048` bringt +33,5 %
  Prompt-Verarbeitung, die Textausgabe liegt bei 85 % der Speicherbandbreite, EAGLE3 kostet
  15 bis 46 % — siehe [GPT-OSS-120B.md](GPT-OSS-120B.md).

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

> **Ueberholt (2026-09-20).** Mit `--spec-type draft-dspark` bringt das Entwurfsmodell
> +24 % statt zu bremsen, und die Byte-Rechnung unten war zu grob. Siehe
> [Nachtrag, Abschnitt 1](#1-spekulatives-decoding-hilft-jetzt--mit-laenge-2-nicht-1).

Ein Draft-Modell lohnt sich nur, wenn es **pro Token deutlich billiger** ist als das
Hauptmodell. Auf bandbreitenlimitierter Hardware zählt dafür, wie viele Bytes pro Token
gelesen werden:

* **Hauptmodell:** 284 B total, aber nur 13 B aktiv bei ~3 bpw → **~5 GiB/Token**
* **dspark-Draft:** 10,5 GiB BF16, dense → **~10,5 GiB/Token**

Das Draft-Modell ist also rund **doppelt so teuer** wie das MoE, das es beschleunigen soll.
Selbst bei perfekter Trefferquote bliebe kein Gewinn. Die aus dichten Modellen bekannte
Faustregel „Draft-Modell = größter Hebel" kehrt sich bei MoE mit kleinem aktivem Anteil um.


## Nachtrag 2026-09-20: was ein Monat Fork-Entwicklung geaendert hat

Alle Zahlen dieses Abschnitts: derselbe Rechner, Nobara 44 / Kernel 7.2, Fork-Build
`strix-fork/vulkan` (ggml 0.20.1), Modell `UD-IQ3_XXS`, Entwurfsmodell
`dspark-DeepSeek-V4-Flash-0731-Q8_0.gguf` (10,9 GB), `-fa on -b 2048 -ub 512`.
Die anderen Dienste der Maschine waren gestoppt; GTT-Grenze 120 GiB
(`ttm.pages_limit=31457280`), belegt 103,8 GiB ohne und 114,5 GiB mit Entwurfsmodell.

### 1. Spekulatives Decoding hilft jetzt — mit Laenge 2, nicht 1

Der Fork kennt inzwischen `--spec-type draft-dspark` als eigenen Typ; im August lief DSpark
nur ueber den generischen Entwurfspfad. Gemessen an drei festen Prompts (Median, je 256 Token):

| Lauf | Generierung | Annahmequote | mittlere Entwurfslaenge |
|---|---|---|---|
| ohne Entwurf | 18,40 t/s | — | — |
| DSpark n=1 | 20,03 t/s | 0,68 | 1,68 |
| **DSpark n=2** | **22,77 t/s** | 0,61 | 2,23 |
| DSpark n=3 | 20,53 t/s | 0,30 | 1,91 |
| DSpark n=1 + `mlock` | 20,11 t/s | 0,68 | 1,68 |

Die Erwartung, dass bei einer MTP-Tiefe von 1 auch n=1 optimal sei, trifft nicht zu: n=2
liegt 14 % darueber. Die Annahmequote sinkt zwar mit der Laenge, aber bis n=2 ueberwiegt
der Gewinn aus zwei Token je Schritt. Bei n=3 kippt es — dort ist die Stichprobe allerdings
klein (66 Entwurfstoken), das sollte man nicht ueberbewerten.

**Damit ist die Begruendung aus dem August-Abschnitt "Warum spekulatives Decoding hier bremst"
widerlegt.** Die dortige Rechnung setzte die volle Dateigroesse des Entwurfsmodells
(10,5 GiB) als Verkehr je Token an und kam so auf "doppelt so teuer wie das MoE". Das trifft
nicht zu: Der groesste Teil der Datei sind Embedding- und Ausgabematrix, von denen je Token
nur ein Bruchteil gelesen wird. Was davon auf den eigenen Spekulationstyp und was auf die
korrigierte Buchhaltung entfaellt, ist damit noch nicht getrennt.

### 2. ZURUECKGEZOGEN: "MoE-Kernel verbessern nur den Prefill"

**Diese Messung war ungueltig und ist am selben Tag zurueckgezogen worden.** Sie verglich
`GGML_VK_MMID_*=1` gegen `GGML_VK_MMID_*=0` und fand +24 % Prompt-Verarbeitung bei
unveraenderter Textausgabe. Die Gegenprobe zeigte hinterher: **mit allen MMID-Schaltern auf 0
rechnet das Modell `nan`** (Perplexitaet nicht bestimmbar, "Unexpected negative standard
deviation of log(prob)"). Verglichen wurde also korrektes Rechnen gegen kaputtes Rechnen --
die Zahlen messen nichts.

Lehre daraus, die hier schon zweimal teuer war: **Auch der Lauf mit abgeschalteter
Optimierung braucht eine Korrektheitsprobe.** Es genuegt nicht, den optimierten Fall zu
pruefen.

Was stattdessen gilt, sauber gemessen:

**a) Der Fork gegen aktuelles Upstream** (beide korrekt, Perplexitaeten deckungsgleich) --
siehe [Abschnitt 9](#9-braucht-man-den-fork-noch-upstream-gegen-fork-gemessen).

**b) Die einzelnen Schalter, jeder mit Korrektheitsprobe.** Qwen3.6-35B-A3B UD-IQ4_XS,
`llama-bench -fa 1 -p 512 -n 128 -r 2`, jeweils ein Schalter auf 0, Rest auf Vorgabe:

| abgeschaltet | Prompt-Verarbeitung (Token/s) | gegen Vorgabe | Perplexitaet (3 Bloecke) |
|---|---|---|---|
| nichts (Fork-Vorgabe) | 1569,1 | — | 6,5915 |
| `GGML_VK_MMID_ROWLISTS` | 1439,7 | −8,2 % | 6,5915 |
| `GGML_VK_MMID_SMALLN` | 1449,9 | −7,6 % | 6,5915 |
| `GGML_VK_MMID_BM64` | 1484,6 | −5,4 % | 6,5915 |
| `GGML_VK_FA_WAVE32` | 1535,5 | −2,1 % | 6,5915 |
| `GGML_VK_FA_KV_CONTIG` | 1542,5 | −1,7 % | 6,5915 |
| `GGML_VK_MMID_M128` | 1560,7 | −0,5 % | 6,5915 |
| Upstream `a894dae` (ohne all das) | 1280,0 | −18,4 % | 5,7159 (10 Bloecke) |

Die Textausgabe blieb in allen Faellen bei 62,2 bis 62,5 Token/s -- die Kniffe wirken
tatsaechlich nur auf die Prompt-Verarbeitung, aber der Beleg dafuer sind diese Zeilen,
nicht die zurueckgezogene Messung.

**Zwei Schalter darf man nicht abschalten:**

* `GGML_VK_MMID_WAVE32=0` ergibt scheinbar **2216 Token/s** statt 1569 (+41 %) -- und
  `nan` in der Perplexitaet. Der Schalter traegt die Korrektheit des MoE-Pfads mit.
* `GGML_VK_MMID_F16B=0` stuerzt beim Start ab (`ggml_abort`).

### 3. Kontexttiefe: milder Abfall

Von Tiefe 0 auf 32768 verliert der Prefill 15 % und die Generierung 9 %. Fuer den
Agentenbetrieb ist der Prefill die entscheidende Zahl:

| Stand | pp512 | 15k Startprompt |
|---|---|---|
| August 2026 (127 t/s, `-b 512`, ohne MMID) | 127 t/s | 2,0 min |
| heute ohne MMID | 166 t/s | 1,5 min |
| **heute mit MMID** | **206 t/s** | **1,2 min** |

Der Sprung von 127 auf 166 t/s stammt nicht aus den MoE-Kerneln, sondern aus der groesseren
Batchgroesse und dem neueren Build; diese beiden sind hier nicht getrennt gemessen.

### 4. `mlock` gegen `mmap`: kein Unterschied

Die Vermutung, dass bei nur ~13 GiB freiem RAM der Seitencache-Druck bremst, bestaetigt sich
nicht:

| Ladeart | Ladezeit | GTT | Generierung |
|---|---|---|---|
| `--load-mode auto` (mmap) | 25 s | 103,8 GiB | 18,40 t/s |
| `--load-mode mlock` | 56 s | 103,8 GiB | 18,46 t/s |

0,3 % Unterschied bei mehr als doppelter Ladezeit. Plausibler Grund: Die Gewichte liegen
ohnehin im GTT, also in GPU-adressiertem Speicher, nicht im Seitencache — der schrumpfte
waehrend der Messung auf 2 GiB, ohne die Rate zu beruehren.

### 5. Reduziertes Expert-Routing laesst sich mit llama.cpp nicht zuschalten

Der Versuch, das Lucebox-Profil (top-k 4 statt 6) per Metadaten nachzustellen, scheitert
am Modellbau — und das praezisiert die Einordnung weiter unten:

```
$ llama-server -m ...UD-IQ3_XXS...gguf --override-kv deepseek4.expert_used_count=int:4
error loading model: check_tensor_dims: tensor 'blk.0.ffn_gate_tid2eid.weight'
has wrong shape; expected 4, 129280, got 6, 129280
```

DeepSeek-V4 traegt die Zahl der aktiven Experten nicht nur als Metadatum, sondern als
Routing-Tabelle im Tensor `blk.N.ffn_gate_tid2eid.weight` (Form 6 x 129280). `--override-kv`
aendert nur das Metadatum, danach schlaegt die Formpruefung fehl. Ein Betrieb mit vier
Experten braucht also ein neu gebautes GGUF, keinen Startschalter — die kursierenden Zahlen
bleiben mit Standard-llama.cpp unerreichbar, und zwar aus diesem Grund.

Die GGUF-Metadaten des verwendeten Quants zur Einordnung:

| Schluessel | Wert |
|---|---|
| `deepseek4.expert_count` | 256 |
| `deepseek4.expert_used_count` | 6 |
| `deepseek4.expert_shared_count` | 1 |
| `deepseek4.block_count` | 43 |
| `deepseek4.context_length` | 1 048 576 |
| `deepseek4.rope.scaling.original_context_length` | 65 536 |

### 6. IQ2_XXS gegen IQ3_XXS: schneller nur im Entwurfsbetrieb, und teuer erkauft

| | UD-IQ3_XXS (104 GB) | UD-IQ2_XXS (91 GB) |
|---|---|---|
| GTT belegt | 103,8 GiB | 91,4 GiB |
| Generierung ohne Entwurf | 18,40 t/s | 18,91 t/s (+2,8 %) |
| Generierung mit DSpark n=2 | 22,77 t/s | **26,25 t/s** (+15 %) |
| Annahmequote | 0,61 | 0,70 |
| pp512 @ Tiefe 0 | 206,5 t/s | 225,7 t/s |
| **Perplexitaet** (wikitext, 20 Bloecke a 2048) | **4,573 ± 0,080** | **5,213 ± 0,094** |

Bemerkenswert ist die Aufteilung: Im nackten Decode bringt die kleinere Quantisierung fast
nichts (+2,8 %), im spekulativen Betrieb dagegen deutlich (+15 %) — weil die Annahmequote
des Entwurfsmodells steigt (0,70 statt 0,61). Das Entwurfsmodell trifft offenbar leichter,
wenn das Hauptmodell selbst unschaerfer ist.

Bezahlt wird das mit **14 % hoeherer Perplexitaet**. Das ist viel: Das Modell ist
quantization-aware in MXFP4 trainiert, weiteres Herunterquantisieren ist doppelte
Quantisierung (siehe Abschnitt Quant-Auswahl). Fuer 3,5 t/s mehr wuerde ich IQ2 nicht nehmen.

### 7. Die Lucebox-Zahlen im direkten Vergleich

Der [Blogbeitrag von Lucebox](https://www.lucebox.com/blog/deepseek-v4-strix-halo) nennt
inzwischen den vollstaendigen Aufbau, damit ist der Vergleich sauber moeglich:

| | Lucebox | dieses Repo (2026-09-20) |
|---|---|---|
| Server | `dflash_server` (proprietaer), HIP, ROCm 7.2.4 | llama.cpp-Fork, Vulkan/RADV |
| Modell | `DeepSeek-V4-Flash-ROCMFP2-STRIX.gguf`, 102,3 GB, ~2,88 bpw | `UD-IQ3_XXS`, 104 GB, 3,06 bpw |
| Experten je Token | **4** (`--ds4-expert-top-k 4`) | 6 (Modellvorgabe) |
| Prefill | ~250 t/s, `--ds4-prefill sparse` | 206,5 t/s, dicht |
| Decode ohne Entwurf | 25,31 t/s | 18,40 t/s |
| Decode mit DSpark | **32,0 t/s** (`DFLASH_DS4_SPEC_Q=4`, fused verify) | 22,77 t/s (n=2) |
| Kontext der Messung | 8192 | 16384 (Tiefen 0/8192/32768 getrennt) |
| **relativer Gewinn durch Spekulation** | **+26 %** | **+24 %** |

Die letzte Zeile ist der eigentliche Befund: **Der Nutzen des Entwurfsmodells ist bei beiden
praktisch gleich.** Die Differenz entsteht nicht beim spekulativen Decoding, sondern in der
Grundmaschine — Expertenzahl, Quantisierung, Backend, Prefill-Pfad.

Zwei Einordnungen dazu:

* **KORREKTUR (nachgeprueft am Quelltext): llama.cpp rechnet ebenfalls duenn.** Die
  urspruengliche Fassung dieses Punktes behauptete, llama.cpp kenne den Indexer-Pfad nicht
  und rechne dicht. Das ist falsch. Sowohl Upstream als auch der Fork implementieren die
  DeepSeek Sparse Attention: `build_attn_inp_k_dsa`, "fused lightning indexer",
  `kq_mask_top_k`, eigener Indexer-Schluesselcache fuer die MSA-Schichten
  (`is_indexer_full`). Die GGUF traegt die zugehoerigen Metadaten
  (`deepseek4.attention.indexer.head_count` = 64, `.key_length` = 128, `.top_k` = 512) und
  60 Indexer-Tensoren je Teildatei. Unsere 233,7 Token/s sind also bereits duenne
  Aufmerksamkeit, und der Abstand zu Luceboxs 250 Token/s betraegt **7 %**, nicht 18 % --
  bei gleichem Verfahren.
* **Die vier Experten sind kein Gratis-Hebel, und Lucebox sagt das selbst:** "This changes
  model execution and trades some quality margin for speed", mit der Empfehlung, vor dem
  Produktiveinsatz gegen sechs Experten zu vergleichen. Eine Perplexitaetsangabe zu den vier
  Experten nennt der Beitrag nicht.

Mit `UD-IQ2_XXS` und DSpark kommen wir auf 26,25 t/s und damit auf 82 % ihres Decodewertes —
bei sechs statt vier Experten und ohne sparse Prefill.


### 7b. Was Lucebox technisch anders macht: eigene Wave32-Kernel

Ihr Bau schaltet mit `-DDFLASH27B_HIP_SM80_EQUIV=ON` einen eigenen rocWMMA-Prefill-Kernel
ein. Das klingt zunaechst widerspruechlich, weil rocWMMA-Flash-Attention in llama.cpp auf
gfx1151 als Bremse galt ([#24437](https://github.com/ggml-org/llama.cpp/issues/24437):
bis −41 % Prompt-Verarbeitung). Der Widerspruch loest sich am Quelltext:

* **rocWMMA ist eine Bibliothek, keine Optimierung.** Sie stellt die Matrixbefehle bereit;
  entscheidend ist, welcher Kernel damit geschrieben wurde.
* **llama.cpps rocWMMA-Pfad stammte aus der CDNA-Welt** (Wave64, andere Kachelgroessen) und
  wurde auf RDNA3.5 mitbenutzt -- daher die Regression. **In aktuellem llama.cpp existiert
  er nicht mehr**: weder Upstream `a894dae` noch der Fork enthalten eine Erwaehnung von
  rocWMMA, die Option `GGML_HIP_ROCWMMA_FATTN` gibt es dort nicht. Sie steht nur noch im
  mitgelieferten llama.cpp von Lucebox (ggml 0.9.11), dort mit Vorgabe `OFF`.
* **Luceboxs Kernel ist fuer RDNA geschrieben.** `server/src/flashprefill_kernels.hip.cu`,
  781 Zeilen, bricht den Bau ab, wenn die Wellenbreite nicht 32 ist: *"A wave64 target would
  need a different WMMA instruction and fragment layout, not a warp-width tweak, so we fail
  the build loudly rather than emit silently-wrong results."*

Der Dateikopf dokumentiert die Portierung aus ihrer CUDA-Fassung: `nvcuda::wmma` →
`rocwmma`, `cp.async` → direkte `uint4`-Ladebefehle, `__shfl_xor_sync` → `__shfl_xor`, und
vor allem die geaenderte Akkumulator-Anordnung (NVIDIA: zwei Zeilen je Lane; AMD RDNA3
Wave32: eine Zeile je Lane), wofuer Maskierung, Softmax und Rescale neu geschrieben wurden.

**Portierbarkeit nach llama.cpp**, falls jemand es versuchen will:

| | |
|---|---|
| Umfang | 781 Zeilen, eine Datei, fuenf Kernel (Mittelwertvektor, Blockbewertung ×2, Blockauswahl, duenne Flash-Attention, KV-Transposition) |
| Abhaengigkeiten | nur `hip_runtime.h`, `hip_bfloat16.h`, `rocwmma.hpp` -- kein projekteigener Header |
| Schnittstelle | vier `extern "C"`-Einstiegspunkte mit rohen Zeigern |
| Lizenz | Apache-2.0 (llama.cpp ist MIT) |

Dagegen sprechen drei Dinge: llama.cpp **hat** den duennen Pfad bereits (man wuerde ersetzen,
nicht ergaenzen); es ist ein **HIP**-Kernel, waehrend Vulkan hier das schnellere Backend ist;
und die Lizenzen passen nicht ohne Weiteres zusammen. Der praktischere Weg ist, ihren Server
direkt zu benutzen.

### 8. HIP nachgemessen -- eigener Befund: [HIP-BEFUND.md](HIP-BEFUND.md)

Ihr Bau-Aufruf enthaelt `-DGGML_HIP_NO_VMM=ON`, was zunaechst nach der fehlenden Zutat
aussah. Es ist keine: Der Schalter ist in llama.cpp die Vorgabe. Gemessen wurde trotzdem,
und das Ergebnis fuellt ein eigenes Dokument:

* HIP **laedt** das 97-GiB-Modell (103 GiB GTT) -- der August-Befund ist ueberholt.
* Vulkan ist bei gleicher Graphvariante **35 % / 37 %** schneller (pp512 / tg128).
* Die **ROCm-Version aendert nichts** (7.1.1 gegen 7.2.1: gleich schnell).
* HIP rechnet **modellabhaengig falsch**: DeepSeek-V4 korrekt (PPL 4,5637 gegen Vulkan
  4,5736), Qwen3-1.7B Q8_0 unbrauchbar (PPL 24 000 bis 290 000 statt 16,5, zwischen Laeufen
  schwankend).
* Der Fork stuerzt unter HIP ohne `LLAMA_MOE_F16=0` ab (f16-Expertenaktivierungen fallen ins
  CPU-Backend, das f32 erwartet).

Einzelheiten, Eingrenzung und Rohdaten: **[HIP-BEFUND.md](HIP-BEFUND.md)**.

### 9. Braucht man den Fork noch? Upstream gegen Fork, gemessen

Der Fork (`Nathanw1014/llama.cpp`, Commit `50c271f8e`) hat seine Upstream-Basis am
**2026-08-17** und liegt inzwischen **600 Commits zurueck**. Der Grund fuer seinen Einsatz
war der August-Befund weiter oben (+62 % Prompt-Verarbeitung, +53 % Textausgabe gegen
Mainline b10488). Beide Baeume heute aus dem Quelltext gebaut, Vulkan, System-RADV,
gleiche Flags, `LLAMA_MOE_F16=0` auf beiden Seiten:

**DeepSeek-V4-Flash UD-IQ3_XXS (97 GiB)**

| Messgroesse | Fork | Upstream `a894dae` | Unterschied |
|---|---|---|---|
| Prompt-Verarbeitung, leerer Kontext (Token/s) | 233,74 ± 0,49 | 206,87 ± 0,43 | +13 % Fork |
| **Prompt-Verarbeitung bei 8192 Token Vorlauf (Token/s)** | **209,64 ± 0,11** | **140,77 ± 0,50** | **+49 % Fork** |
| Textausgabe, leerer Kontext (Token/s) | 18,21 ± 0,00 | 18,58 ± 0,01 | +2 % Upstream |
| Textausgabe bei 8192 Token Vorlauf (Token/s) | 17,60 ± 0,02 | 17,18 ± 0,01 | +2 % Fork |
| Perplexitaet, 10 Bloecke (dimensionslos, kleiner = besser) | 4,0709 ± 0,0977 | 4,1050 ± 0,0989 | gleich |

**Qwen3.6-35B-A3B UD-IQ4_XS (17 GiB), zum Vergleich ein Modell mit wenigen Experten**

| Messgroesse | Fork | Upstream | Unterschied |
|---|---|---|---|
| Prompt-Verarbeitung, leerer Kontext (Token/s) | 1572,08 ± 13,30 | 1293,47 ± 0,35 | +22 % Fork |
| Prompt-Verarbeitung bei 8192 Token Vorlauf (Token/s) | 1349,73 ± 13,24 | 1101,85 ± 0,43 | +23 % Fork |
| Textausgabe, leerer Kontext (Token/s) | 62,28 ± 0,00 | 63,09 ± 0,06 | +1 % Upstream |
| Textausgabe bei 8192 Token Vorlauf (Token/s) | 58,32 ± 0,02 | 58,72 ± 0,02 | +1 % Upstream |
| Perplexitaet, 10 Bloecke | 5,7170 ± 0,1397 | 5,7159 ± 0,1396 | gleich |

Das Bild hat sich seit August **verschoben**:

* **Bei der Textausgabe hat Upstream aufgeschlossen.** Der August-Vorsprung des Forks
  (+53 %) ist weg; heute liegen beide innerhalb von 2 % beieinander, mal so, mal so.
* **Beim Prompt-Vorlauf bleibt der Fork vorn, und der Abstand waechst mit der Kontexttiefe.**
  Upstream verliert von leerem Kontext auf 8192 Token 32 % (206,9 -> 140,8 Token/s), der
  Fork nur 10 % (233,7 -> 209,6). Das passt zu seinen Flash-Attention-Aenderungen
  (KV einmal dequantisieren, strided f16-KV kontiguieren, 32er-Subgruppen festpinnen).
* **Beide rechnen korrekt** -- die Perplexitaeten sind auf beiden Modellen deckungsgleich.
* Die Spekulationstypen, die unsere Dienste brauchen (`draft-mtp`, `draft-dspark`), kennt
  **Upstream inzwischen ebenfalls**. Dafuer wird der Fork nicht mehr gebraucht.

Fuer den Alltag heisst das: Der Fork lohnt sich weiterhin, aber nur noch wegen des
Prompt-Vorlaufs bei gefuelltem Kontext -- also genau fuer Agenten- und Langkontextbetrieb.
Wer kurze Chats fuehrt, verliert mit Upstream nichts und gewinnt 600 Commits an Fehlerfixes.

### 10. Der HIP-Fehler ist ein bekannter Upstream-Fehler, im Fork noch enthalten

Das modellabhaengige Falschrechnen aus [HIP-BEFUND.md](HIP-BEFUND.md) ist vollstaendig
aufgeklaert: Es ist
[Issue #28211](https://github.com/ggml-org/llama.cpp/issues/28211) ("wrong logits, triggered
by prompts longer than n_ubatch"), und es trifft nur Prompts, die groesser als die
Mikrobatchgroesse sind.

| Qwen3-1.7B Q8_0, HIP, 5 Bloecke | Perplexitaet (kleiner = besser) |
|---|---|
| Fork, `-ub 512` (Vorgabe) | 112 536 |
| Fork, `-ub 2048` | **16,486** |
| Upstream `a894dae`, `-ub 512` | **16,511** |

Perplexitaetslaeufe mit `-c 2048` gegen die Vorgabe `-ub 512` ueberschreiten die Grenze in
jedem Block -- deshalb der Ausschlag. Mit `-ub 2048` rechnet auch der Fork richtig, und auf
aktuellem Upstream tritt der Fehler gar nicht mehr auf. Der Fork sitzt auf dem Stand vom
17.08. und damit vor dem Fix.

Ebenfalls gemessen und ohne Wirkung: `ROCBLAS_USE_HIPBLASLT=1`, in der Literatur der
groesste Prompt-Vorlauf-Hebel fuer gfx1151 (5472,89 ± 135,48 gegen 5533,75 ± 208,06 Token/s
auf Qwen3-1.7B). Die nativen gfx1151-Kernel dafuer kamen erst mit ROCm 7.2; wir laufen auf
7.1.1.

### 11. Bestwert auf dieser Maschine (Stand 2026-09-20)

Quelltext-Build des Forks mit System-RADV (nicht das Payload mit gebuendeltem RADV),
DSpark-Entwurfsmodell, drei feste Prompts, Median:

| Konfiguration | Prompt-Verarbeitung (Token/s) | Textausgabe (Token/s) | Annahmequote |
|---|---|---|---|
| IQ3_XXS + DSpark n=2 | 201,5 | 23,01 | 0,83 |
| IQ2_XXS + DSpark n=2 | 253,5 | 27,46 | **0,98** |
| **IQ2_XXS + DSpark n=3** | **261,7** | **28,91** | 0,81 |
| IQ2_XXS + DSpark n=4 | 251,6 | 26,05 | 0,64 |
| IQ2_XXS + DSpark n=5 | 252,7 | 25,06 | 0,65 |
| IQ2_XXS + DSpark n=6 | 249,6 | 25,01 | 0,65 |

n=3 ist das Optimum. Ab n=4 faellt die Annahmequote von 0,81 auf 0,64, und die zusaetzliche
Pruefarbeit kostet mehr als die laengeren Entwuerfe einbringen. n=5 und n=6 liefern exakt
dieselben Werte (195 von 299 angenommen, mittlere Laenge 4,25) -- das Modul gibt nicht mehr
her, was zum protokollierten `n_extract=3` passt.

Zum Vergleich der August-Stand: 11,96 Token/s Textausgabe in Mainline, 18,33 mit dem Fork.
Wir sind also bei rund dem **2,4-fachen** des August-Werts und bei **90 %** der von Lucebox
genannten 32,0 Token/s -- mit sechs statt vier Experten. Beim Prompt-Vorlauf liegen die
261,7 Token/s **ueber** deren 250.

Bei IQ2 nimmt das Hauptmodell 98 % der Entwurfstoken an, deshalb lohnt dort die laengere
Entwurfstiefe; bei IQ3 (Annahmequote 0,83) nicht. Das Entwurfsmodul arbeitet laut
Serverprotokoll intern mit `block_size=5, n_extract=3`, ist also kein freilaufendes
Entwurfsmodell, sondern eine Maskierungsvorhersage fester Breite -- was erklaert, warum
n=3 gut passt.

### 12. Warnung: IQ2 mit Entwurfsmodell steht an der Speichergrenze

Beim Versuch, n=4 zu messen, blieb der Treiber stehen:

```
[drm:amdgpu_gem_va_update_vm] *ERROR* Couldn't update BO_VA (-12)
INFO: task llama-server blocked for more than 245 seconds.
```

`-12` ist ENOMEM. 85 GiB Modell + 10,6 GiB Entwurfsmodell + Kontextspeicher bei 108 von
124 GB belegtem Arbeitsspeicher reichten nicht mehr fuer die Adressraum-Aktualisierung.
Der Prozess landete im nicht unterbrechbaren Kernelzustand und ueberlebte auch `SIGKILL`;
`systemctl stop` blockierte daraufhin ebenfalls.

**Auf dieser APU gibt es keinen Ausweg ausser einem Neustart:** `reset_method = -1`, und
der debugfs-Pfad `amdgpu_gpu_recover` verweigert den Schreibzugriff selbst als root, weil
die GPU zugleich die Anzeige treibt.

Konsequenzen fuer weitere Messungen:

* Kontext klein halten (`-c 4096` statt 16384), wenn nur kurze Antworten gemessen werden.
* Vor jedem Serverstart warten, bis GTT unter 10 GiB und mehr als 100 GB Arbeitsspeicher
  frei sind -- der Seitencache eines vorangegangenen Ladevorgangs reicht, um es zu kippen.
* Fuer Dauerbetrieb: **IQ3 ohne Entwurfsmodell oder IQ2 allein** sind die sicheren
  Konfigurationen. IQ2 mit Entwurfsmodell ist ein Messaufbau, kein Betriebszustand.

### 13. Die Lucebox-Zahlen nachgebaut -- und was wirklich dahintersteckt

Ihr Server ist offen (Apache-2.0, [Luce-Org/lucebox](https://github.com/Luce-Org/lucebox)),
Modell und Entwurfsmodell liegen ungesperrt auf Hugging Face. Damit ist die Reproduktion
moeglich. Aufbau: ihr `dflash_server` (Commit `e0048e0`), ihr
`DeepSeek-V4-Flash-ROCMFP2-STRIX.gguf` (102,3 GB, ~2,88 bpw), ihr DSpark-Entwurf
(11,3 GB), ihre Kommandozeile aus dem Blogbeitrag. **Einziger Unterschied: ROCm 7.1.1
statt ihrer 7.2.4.**

Beim Bau fehlt ihrer Anleitung eine Zutat fuer Fedora: `-DCMAKE_POSITION_INDEPENDENT_CODE=ON`
(sonst scheitert das Binden mit `relocation R_X86_64_32 ... recompile with -fPIC`), und die
rocWMMA-Suche greift nur nach `/opt/rocm/include` (`NO_DEFAULT_PATH`), weshalb
`-DDFLASH27B_ROCWMMA_INCLUDE_DIR=/usr/include` noetig ist.

Gemessen mit denselben drei Prompts wie alle Werte hier (rund 2000 Token Vorlauf, 256 Token
Ausgabe); die Zahlen stammen aus **ihrem** Serverprotokoll, das Vorlauf und Ausgabe getrennt
ausweist:

| Konfiguration | Prompt-Vorlauf (Token/s) | Textausgabe (Token/s), Median | Annahmequote |
|---|---|---|---|
| ohne Entwurfsmodell, 4 Experten | ~411 | 22,8 | — |
| mit DSpark q=4, 4 Experten | ~403 | **28,2** (Spitze 31,3) | 0,68-0,76 |
| mit DSpark q=4, **6 Experten** | ~367 | 26,5 | 0,80 |

**Ihre Angaben sind im Wesentlichen bestaetigt.** Die beworbenen "up to 32,0 tok/s" wurden
mit 31,3 fast erreicht. Der Basiswert ohne Entwurf liegt mit 22,8 gegen ihre 25,31 rund
10 % darunter -- die ROCm-Version ist die einzige verbliebene Abweichung, und laut
Fremdliteratur kamen die nativen gfx1151-Kernel erst mit 7.2.

**Was die Expertenreduktion wirklich bringt** (im Blogbeitrag ohne Zahl): von sechs auf vier
Experten **+6,4 % Textausgabe** (26,5 -> 28,2) und **+11 % Prompt-Vorlauf** (367 -> 411),
bei gleichzeitig sinkender Annahmequote des Entwurfsmodells (0,80 -> 0,69). Sechs Prozent
Tempo fuer ein Drittel weniger Experten ist ein schlechter Tausch -- ihre eigene Warnung
("trades some quality margin for speed") ist damit quantifiziert.

**Der eigentliche Vergleich:**

| | llama.cpp-Fork, IQ2_XXS + DSpark n=3 | Lucebox, ihr Modell, 4 Experten |
|---|---|---|
| Textausgabe (Token/s) | **28,91** | 28,2 (Spitze 31,3) |
| Prompt-Vorlauf (Token/s) | 261,7 | **~411** |
| Experten je Token | **6** | 4 |
| Quantisierung | 2,06 bpw (IQ2_XXS) | ~2,88 bpw (ROCmFPX) |

Beim Antworttempo ist llama.cpp gleichauf -- mit sechs statt vier Experten. **Ihr Vorsprung
liegt im Prompt-Vorlauf: 57 %.** Das ist der Teil, fuer den sie eigene Wave32-Kernel
geschrieben haben (siehe Abschnitt 7b), und fuer Agentenbetrieb mit grossem Startprompt der
wichtigere. Wer lange Kontexte verarbeitet, gewinnt mit ihrem Server; wer chattet, nicht.

### Messvorschrift

Die Skripte liegen nicht in diesem Repo, sondern entstanden ad hoc; die Kernbefehle:

```bash
# Durchsatz je Konfiguration (llama-server + drei feste Prompts, Median)
llama-server -m .../UD-IQ3_XXS/...-00001-of-00004.gguf -ngl 999 -fa on -c 16384 \
  -b 2048 -ub 512 --jinja -np 1 \
  -md .../dspark-DeepSeek-V4-Flash-0731-Q8_0.gguf -ngld 999 \
  --spec-type draft-dspark --spec-draft-n-max 2

# MMID-Isolierung (derselbe Build, nur Schalter)
GGML_VK_MMID_ROWLISTS=0 GGML_VK_MMID_SMALLN=0 GGML_VK_MMID_BM64=0 \
GGML_VK_MMID_WAVE32=0 GGML_VK_MMID_F16B=0 GGML_VK_MMID_M128=0 \
llama-bench -m ... -fa 1 -p 512 -n 128 -d 0,8192,32768 -r 2
```

Die Annahmequote steht im Serverprotokoll: `draft acceptance = 0.61404 (140 accepted /
228 generated), mean len = 2.23`.

## Einordnung der kursierenden Benchmarks

> **Ergaenzt (2026-09-20).** Warum das reduzierte Expert-Routing mit llama.cpp nicht
> nachstellbar ist, steht jetzt belegt im [Nachtrag, Abschnitt 5](#5-reduziertes-expert-routing-laesst-sich-mit-llamacpp-nicht-zuschalten).

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
* ~~Prefill-Verhalten bei groesseren Kontexten (32k, 64k) vermessen~~ — erledigt 2026-09-20, siehe Nachtrag
* ~~Reduziertes Expert-Routing (`--override-kv deepseek4.expert_used_count=int:4`) pruefen~~ —
  geht nicht, siehe Nachtrag Abschnitt 5 (Routing-Tabelle liegt als Tensor vor)
* `UD-IQ2_XXS` (85 GiB) gegen `UD-IQ3_XXS` messen, Durchsatz und Perplexitaet — laeuft
* Groesseren Quant `UD-Q3_K_XL` (119 GiB) testen — passt seit der UMA-Umstellung
