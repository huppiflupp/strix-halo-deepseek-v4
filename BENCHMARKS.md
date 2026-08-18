# Backend-Vergleich und Bandbreitenanalyse

Alle Werte: llama.cpp Commit 9731ad3, `llama-bench -ngl 999 -p 512 -n 128 -r 2`,
AMD Ryzen AI MAX+ 395 (gfx1151), 128 GB unified, BIOS-UMA `Auto` (64 GiB VRAM + 110 GiB GTT).
Modelle aus dem lokalen Ollama-Bestand (`/var/lib/ollama/.ollama/models/blobs/`).

## HIP/ROCm vs. Vulkan/RADV

| Modell | GB | HIP pp512 | VK pp512 | Δ | HIP tg128 | VK tg128 | Δ |
|---|---:|---:|---:|---:|---:|---:|---:|
| gemma4:12b (Q4_K_M) | 7,37 | **816,0** | 741,0 | +10,1 % | 23,75 | **25,58** | +7,7 % |
| Hermes-4-14B (Q4_K_M) | 9,00 | **696,6** | 694,5 | +0,3 % | 21,51 | **23,45** | +9,0 % |
| muse-glimmer 30B (Q4_K_M) | 16,74 | **367,0** | 356,6 | +2,9 % | 11,38 | **12,40** | +9,0 % |
| qwen3.8 / qwen35 27B (Q4_K_M) | 16,80 | 320,6 | **329,4** | −2,7 % | 11,19 | **12,09** | +8,0 % |

**Zwei klare Muster:**

1. **Vulkan gewinnt beim Decode — immer, und bemerkenswert konsistent** (+7,7 / +9,0 / +9,0 / +8,0 %).
   Bei tg128 ist die Streuung winzig (σ < 0,09), der Vorsprung ist also real und kein Rauschen.
2. **Beim Prefill liegt HIP meist vorn, aber uneinheitlich** (+10,1 bis −2,7 %). Hier ist auch
   die Streuung deutlich groesser (σ bis 29).

Fuer den Praxisbetrieb ist Vulkan damit die bessere Wahl — zusaetzlich zu dem Umstand, dass
nur Vulkan VRAM und GTT als einen Pool nutzt und damit Modelle jenseits des VRAM-Blocks laedt.

## Effektive Speicherbandbreite

Bei autoregressivem Decoding mit Batch 1 muss pro Token einmal ueber alle aktiven Gewichte
gelesen werden. `Modellgroesse x tg128` ergibt daher die effektiv erreichte Bandbreite:

| Modell | GB gelesen/Token | tg128 | effektive Bandbreite |
|---|---:|---:|---:|
| gemma4:12b | 7,37 | 25,58 | 188 GB/s |
| Hermes-4-14B | 9,00 | 23,45 | **211 GB/s** |
| muse-glimmer 30B | 16,74 | 12,40 | 208 GB/s |
| qwen3.8 27B | 16,80 | 12,09 | 203 GB/s |
| | | **Mittel** | **203 GB/s** |

Die dichten Modelle liegen eng beieinander (188–211 GB/s) — sie sind sauber
**bandbreitenlimitiert**, wie erwartet. Bei 256 GB/s theoretischer Bandbreite entspricht das
einer Effizienz von 74–82 %, ein normaler Wert.

### DeepSeek-V4-Flash faellt drastisch heraus

| | |
|---|---|
| aktive Parameter | 13 B bei 3,0625 bpw → **4,98 GB/Token** |
| gemessen | 11,94 t/s |
| effektive Bandbreite | **59 GB/s** |
| Anteil am Mittel dichter Modelle | **29 %** |
| bei gleicher Effizienz moeglich | **40,7 t/s** |

Das MoE erreicht also nur **etwas mehr als ein Viertel** der Bandbreiteneffizienz, die dichte
Modelle auf derselben Hardware erzielen. Der theoretische Vorteil von 13 B aktiven Parametern
wird fast vollstaendig vom Overhead aufgefressen: Expert-Routing, schlechte Speicherlokalitaet
beim Sammeln der selektierten Experten, und viele kleine GEMMs statt weniger grosser.

**Das ordnet auch die kursierenden 32 t/s neu ein.** Sie liegen unterhalb der 40,7 t/s, die bei
sauberer Bandbreitenausnutzung moeglich waeren — die Zahl ist physikalisch also **nicht
unplausibel**. Der Flaschenhals liegt damit nicht in der Hardware, sondern in der
MoE-Implementierung von llama.cpp.

## Nicht ladbare Modelle

7 der 11 Ollama-Modelle liessen sich mit diesem llama.cpp-Build nicht laden:

```
error loading model hyperparameters:
key qwen35.rope.dimension_sections has wrong array length; expected 4, got 3
```

Betroffen: `qwen3.5:9b`, `qwen3.6`, `qwen3-vl:8b-instruct`, `qwen3-vl:30b`, `gemma4:e2b`,
`gemma4:31b`, `gemma4:26b-a4b-it-bf16`.

Ollama pflegt eigene Patches, und die GGUF-Metadaten dieser Modelle weichen von dem ab, was
mainline-llama.cpp erwartet. Kein Hardware- oder Backend-Problem — es fehlt eine passende
llama.cpp-Version. Aergerlich ist der Ausfall von `gemma4:26b-a4b-it-bf16`, dem einzigen
weiteren MoE im Bestand (4 B aktiv), das einen zweiten Datenpunkt zum MoE-Overhead geliefert haette.

---

# Nachbau: Qwen3-30B-A3B-Instruct-2507 (strix-halo-guide)

Der [strix-halo-guide](https://github.com/hogeheer499-commits/strix-halo-guide) berichtet
fuer dieses MoE **100,04 t/s tg128 / 1416,03 t/s pp512** (Build b9467, Vulkan/RADV, Mesa 26.1.1).
Nachbau mit `unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF`, IQ4_XS (15,25 GiB, 30,53 B / 3 B aktiv):

| Variante | pp512 | tg128 |
|---|---:|---:|
| A) Guide-Flags (`-fa 1 -mmp 0 -b 2048 -ub 512 -t 16 --poll 50`), Vulkan | 1378,19 ± 8,55 | **85,53 ± 0,53** |
| B) ohne Zusatzflags, Vulkan | 1339,84 ± 51,04 | 84,33 ± 0,31 |
| C) Guide-Flags, HIP/ROCm | 1295,68 ± 24,60 | 72,71 ± 0,35 |
| Guide-Referenz | 1416,03 | 100,04 |

**Drei Befunde:**

1. **Der Prefill ist reproduziert** (1378 vs. 1416, −2,7 %).
2. **Die Guide-Flags bringen fast nichts** (+1,4 % tg, +2,9 % pp). Flash Attention und groessere
   Batches sind auf dieser Hardware nicht der Hebel, als der sie oft gehandelt werden.
3. **Vulkan schlaegt HIP bei diesem MoE um +17,6 %** — mehr als doppelt so deutlich wie bei den
   dichten Modellen (+8 %).

## Die 15 % Decode-Differenz erklaeren sich durch die Quant-Groesse

Der Guide nennt sein Quant "IQ4_XS-**3.63bpw**", llama.cpp meldet fuer unsere Datei
"IQ4_XS - **4.25 bpw**" (15,25 GiB fuer 30,53 B). Skaliert man linear ueber die pro Token
gelesenen Bytes:

```
85,53 t/s x (4,25 / 3,63) = 100,14 t/s     Guide: 100,04 t/s     Abweichung: +0,1 %
```

Die Uebereinstimmung ist frappierend genau. **Einschraenkung:** bei den gaengigen Anbietern
existiert kein 3,63-bpw-IQ4_XS (bartowski liegt bei 4,31 bpw), die bpw-Angabe des Guides
koennte also anders gezaehlt sein (z. B. nur ueber quantisierte Tensoren, ohne Embedding- und
Output-Layer). Ein direkter Gegentest mit einem echten 3,63-bpw-Quant steht aus.

# Korrektur: MoE ist auf Strix Halo NICHT generell langsam

Der Qwen-Nachbau widerlegt die urspruengliche Schlussfolgerung, MoE-Overhead fresse auf dieser
Hardware generell den Bandbreitenvorteil auf:

| | GB/Token | t/s | effektive Bandbreite | Anteil |
|---|---:|---:|---:|---:|
| dichte Modelle (Mittel aus 4) | — | — | **203 GB/s** | 100 % |
| Qwen3-30B-A3B (3 B aktiv) | 1,59 | 85,5 | **136 GB/s** | **67 %** |
| DeepSeek-V4-Flash (13 B aktiv) | 4,98 | 11,9 | **59 GB/s** | **29 %** |

MoE kostet also durchaus Effizienz — aber Qwen3-30B-A3B holt zwei Drittel der
Dense-Bandbreite heraus und liefert damit 85 t/s. **V4-Flash mit 29 % ist der Ausreisser,
nicht die Regel.**

## Was bei V4-Flash NICHT hilft

| Massnahme | Ergebnis |
|---|---|
| spekulatives Decoding (dspark-Draft) | 11,9 → **7,7 t/s** (schlechter) |
| Flash Attention + `-b 2048 -ub 512 -t 16 --poll 50` | 11,94 → **11,90 t/s** (unveraendert) |
| HIP statt Vulkan | laedt nicht |

Der Engpass liegt also weder in der Attention noch im Batching. Zwei Kandidaten bleiben:

1. **GTT-Anteil:** 34,5 der 98,5 GiB (35 %) liegen ausserhalb des VRAM-Blocks. Qwen3-30B passt
   mit 15,25 GiB vollstaendig ins VRAM. Falls GTT-Zugriffe spuerbar langsamer sind, erklaert
   das einen grossen Teil der Luecke. **Testbar** durch BIOS-UMA auf `512M` — dann liegt alles
   einheitlich in GTT.
2. **`deepseek4`-Kernel:** Die Architektur ist neu in llama.cpp, waehrend `qwen3moe` seit
   langem optimiert ist. Ein Update auf einen neueren Build koennte messbar etwas bringen.

---

# Speicherkonfiguration: BIOS-UMA 512M (alles in GTT)

Nach Umstellung von UMA `Auto` (64 GiB fest zugeteiltes VRAM) auf `512M`:

```
RAM sichtbar:  124 GiB   (vorher 62)
VRAM total:    512 MiB   (vorher 64 GiB)
GTT total:     110 GiB
Vulkan meldet: 113.152 MiB — ehrlich (vorher 174 GiB, mehr als physisch verbaut)
```

| Modell | VRAM+GTT gemischt | alles in GTT |
|---|---:|---:|
| V4-Flash pp512 / tg128 | 127,39 / 11,94 | 128,37 / **12,02** |
| Qwen3-30B pp512 / tg128 | 1339,84 / 84,33 | 1316,42 / **84,34** |

**GTT ist nicht langsamer als VRAM.** Bei beiden Modellen identische Werte innerhalb der
Streuung. Rueckblickend logisch: es ist physisch derselbe Speicher, die Unterscheidung ist
Buchhaltung des Treibers. Die Hypothese, der GTT-Anteil erklaere die schlechte
V4-Flash-Effizienz, ist damit **widerlegt**.

Die Umstellung bleibt trotzdem richtig: 124 statt 62 GiB nutzbar, ehrliche Vulkan-Werte, und
der groessere Quant `UD-Q3_K_XL` (119 GiB) passt jetzt ueberhaupt erst.

# Headless-Betrieb

Umstellung auf `multi-user.target`, `plasmalogin.service` deaktiviert. Vorher liefen ~851 MB
GUI-Prozesse (plasma-login-gr, kwin_wayland, plasma-keyboard, …), ohne dass jemand eingeloggt war.

| Modell | mit GUI | headless |
|---|---:|---:|
| V4-Flash pp512 / tg128 | 128,37 / 12,02 | 128,98 / **11,96** |
| Qwen3-30B pp512 / tg128 | 1316,42 / 84,34 | 1318,01 / **84,85** |
| Qwen + Guide-Flags tg128 | 85,53 | 83,24 |

**Kein messbarer Effekt.** Die Vermutung, `kwin_wayland` halte GPU-Kontexte und verfaelsche die
Messungen, bestaetigt sich nicht — die bisherigen Zahlen waren bereits sauber. Headless spart
851 MB und zwei Services, ist aber **keine Performance-Massnahme**.

# Einordnung durch externe Quellen

## Gegenmessung mit identischem Quant

[slb350/strix-benchmarks](https://github.com/slb350/strix-benchmarks) hat exakt dasselbe Modell
und Quant auf derselben Hardware gemessen:

| | Build | pp512 | tg128 |
|---|---|---:|---:|
| slb350 (UD-IQ3_XXS, RADV) | b9518 | 114,7 | **12,4** |
| diese Messung (UD-IQ3_XXS, RADV) | b94041a | **127,39** | 11,94 |

Prefill 11 % darueber, Decode 4 % darunter. Die Konfiguration ist also in Ordnung; die
11,94 t/s sind kein Konfigurationsfehler, sondern der Mainline-Normalzustand.

## Ein RADV-Fork erreicht das Doppelte

Auf identischer Hardware, gleichem Quant (UD-IQ3_XXS, KV q8_0, 131k ctx), in **einer einzigen
Session** gegeneinander gemessen
([r/LocalLLaMA 1vlmh0b](https://www.reddit.com/r/LocalLLaMA/comments/1vlmh0b/deepseek_v4_flash_0731_at_27_ts_decode_on_strix/)):

| Stack | pp2048 | tg plain | tg mit DSpark-Draft |
|---|---:|---:|---:|
| Mainline llama.cpp + ROCm 7.14 | 191,28 | — | 13,35 (**DSpark-Gewinn: 0 %**) |
| `Nathanw1014/strix-halo-llamacpp` v0.6.1, RADV, `-fa on -b/-ub 2048` | **284,98** | **18,55** | **20,96 – 27,13** |

Ergaenzend mit demselben Fork, UD-IQ3_XXS mit Q6-Attention, 64k Kontext: 20,48 plain →
**Ø 28,5 t/s** mit `--spec-draft-n-max 3`
([r/LocalLLaMA 1vrm27o](https://www.reddit.com/r/LocalLLaMA/comments/1vrm27o/deepseek_v4_flash_0731_on_strix_halo_draft_model/)).

*(Diese Reddit-Werte stammen aus der Recherche und wurden hier nicht selbst nachgemessen.)*

## Korrektur: warum spekulatives Decoding scheiterte

Die urspruengliche Erklaerung in diesem Repo lautete, der DSpark-Draft sei mit 10,5 GiB BF16
pro Token teurer als das MoE-Hauptmodell (~10,5 vs. ~4,98 GB/Token) und koenne deshalb nicht
beschleunigen. **Die Rechnung stimmt, ist aber nicht die Ursache.**

In Mainline-llama.cpp bringt der DSpark-Draft auf dieser Hardware nachweislich **null Gewinn**
— unabhaengig von seiner Groesse. Im RADV-Fork bringt derselbe Draft **+50 %**. Es lag an der
Implementierung des spekulativen Decodings, nicht am Draft-Modell.

Damit relativiert sich auch der Lucebox-Wert endgueltig: **27–28 t/s sind mit offenem Werkzeug
und ohne Experten-Beschneidung erreichbar.** Die beworbenen 32 t/s kaufen die letzten ~15 %
mit `--ds4-expert-top-k 4`.

## Der Softwarestack ist der groesste Einzelfaktor

Identische Hardware, gpt-oss-120b MXFP4, RADV, pp512: **255,17 t/s (b6119) → 719,91 t/s
(b9187)** — Faktor 2,8 in neun Monaten
([kyuz0/amd-strix-halo-toolboxes](https://github.com/kyuz0/amd-strix-halo-toolboxes)).

Weitere Werte von dort (RADV): Qwen3-235B-A22B UD-Q3_K_XL **158,81 / 17,16**,
GLM-4.5-Air UD-Q4_K_XL **281,21 / 25,02**, gpt-oss-120b **719,91 / 56,61**.

## Weitere belegte Hinweise

* **rocWMMA meiden** auf gfx1151 — Langkontext-Killer: tg32@32k 18,07 **mit** vs. 35,39 **ohne**.
* **`amd_iommu=off`** bringt 5–12 % gegenueber `iommu=pt`.
* **GGUF-Herkunft ist messbar:** gpt-oss-120b ggml-org-MXFP4 vs. unsloth-„F16" bei gleicher
  Groesse → 41,52 vs. 29,72 t/s, weil ein F16-Embedding-Layer auf der CPU landet.
* **Kein Strix-Halo-Messwert existiert fuer DeepSeek V3/R1 (671B)** — passt auch bei Q2
  (~200 GB) nicht in 128 GB. Was als „R1 auf Strix Halo" kursiert, sind dichte Llama-70B-Distills.

# Naechste Schritte, nach erwartetem Ertrag

1. **Fork `Nathanw1014/strix-halo-llamacpp` v0.6.1 testen** — belegt 18,55 t/s plain und
   21–27 t/s mit DSpark, gegen unsere 11,94. Groesster Hebel.
2. **llama.cpp aktualisieren** (b94041a → aktuell) — Faktor bis 2,8 ueber Versionen belegt,
   behebt vermutlich auch die 7 nicht ladbaren Ollama-Modelle.
3. **`amd_iommu=off`** in die Kernel-Cmdline.
4. **`-ub 2048` fuer lange Prompts** sweepen.
5. Groesseren Quant `UD-Q3_K_XL` (119 GiB) testen — passt seit der UMA-Umstellung.
