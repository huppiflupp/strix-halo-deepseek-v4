# DeepSeek-V4-Flash auf AMD Strix Halo (Ryzen AI MAX+ 395)

Messprotokoll zum lokalen Betrieb von **DeepSeek-V4-Flash-0731** (284 B MoE, 13 B aktiv)
auf einem einzelnen AMD Ryzen AI MAX+ 395 mit 128 GB Unified Memory.

Stand: 2026-08-18

## TL;DR

* Das Modell **läuft** auf einer Einzelmaschine: 98,5 GiB belegt, ~12 t/s Decode.
* **Spekulatives Decoding mit dem mitgelieferten `dspark`-Draft-Modell macht es langsamer**
  (7,7 statt 11,9 t/s) — die Faustregel für dichte Modelle greift bei MoE nicht.
* Die kursierenden „32 t/s auf Strix Halo" stammen **nicht** aus llama.cpp, sondern aus einem
  proprietären Server mit reduziertem Expert-Routing. Mit llama.cpp sind sie nicht reproduzierbar.
* Der Prefill ist mit ~25 t/s der eigentliche Engpass — das beworbene 1-M-Kontextfenster
  ist damit praktisch unbenutzbar.

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

| Konfiguration | Prefill | Decode | Belegung |
|---|---|---|---|
| ohne Draft-Modell | 25,8 t/s | **11,9 t/s** | 98,5 GiB (64 VRAM + 34,5 GTT) |
| mit `dspark`-Draft, `--spec-draft-n-max 5` | 21,5 t/s | **7,7 t/s** | 109,7 GiB (64 VRAM + 45,7 GTT) |

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

## Offene Punkte

* HIP/ROCm-Backend gegen Vulkan messen (Build benötigt `rocblas-devel`, `hipblas-devel`)
* Reproduzierbare Zahlen via `llama-bench` statt interaktiver Läufe
* UMA auf `512M` gegenprüfen — ob der einheitliche Pool die Decode-Rate verbessert
* Prefill-Verhalten bei größeren Kontexten (32k, 64k) vermessen
