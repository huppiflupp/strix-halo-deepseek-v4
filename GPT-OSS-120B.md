# gpt-oss-120b auf Strix Halo: welche Einstellungen wirken

Nebenmessung zum DeepSeek-Protokoll dieses Repos, Messtag 2026-09-20. Rohdaten und Skripte:
[`messungen/2026-09-20/gptoss-sweep.log`](messungen/2026-09-20/gptoss-sweep.log),
[`gptoss-sweep.sh`](messungen/2026-09-20/gptoss-sweep.sh),
[`gptoss-ppl.log`](messungen/2026-09-20/gptoss-ppl.log).

| | |
|---|---|
| Maschine | AMD Ryzen AI MAX+ 395, Radeon 8060S (gfx1151), 128 GB LPDDR5X-8000 unified |
| System | Nobara 44, Kernel 7.2.6, Mesa RADV (System), GTT-Grenze 120 GiB |
| Modell | `ggml-org/gpt-oss-120b-GGUF`, `gpt-oss-120b-MXFP4.gguf`, 63,4 GB, 116,8 Mrd. Parameter, 128 Experten, 4 aktiv |
| llama.cpp | Fork `Nathanw1014/llama.cpp` @ `50c271f8e` (Upstream-Basis 2026-08-17), aus dem Quelltext gebaut; für HIP Upstream `a894dae` |
| ROCm | 7.1.1 (Fedora-Pakete); zusätzlich Laufzeit 7.2.1 aus dem Ollama-Bündel |
| Leistungsmodus | vermutlich „Balanced" 85 W — nicht gesondert geprüft |

Alle Werte aus `llama-bench` (`-r 2`), sofern nicht anders angegeben. Bezugspunkt ist Vulkan
mit `-fa 1 -b 2048 -ub 512`: **834,9 Token/s Prompt-Verarbeitung (pp512), 53,5 Token/s
Textausgabe (tg128)**. Ladezeit aus dem Seitencache 5 s; ein echter Kaltstart von der Platte
wurde nicht gemessen.

## Ergebnis in einem Satz

Der größte Hebel ist der Mikrobatch (`-ub 2048`: +33,5 % Prompt-Verarbeitung), die
Textausgabe liegt bei 85 % der Speicherbandbreite und ist mit Einstellungen nicht mehr
nennenswert zu steigern, und das mitgelieferte EAGLE3-Entwurfsmodell macht es langsamer.

## Was wirkt

| Hebel | Prompt-Verarbeitung | Textausgabe |
|---|---|---|
| **Mikrobatch `-ub 2048` statt 512** (pp2048) | **+33,5 %** (838,2 → 1119,4 Token/s) | unberührt |
| Mikrobatch 1024 statt 512 | +17,8 % (987,6 Token/s) | unberührt |
| Mikrobatch 256 statt 512 | −21,2 % (660,3 Token/s) | unberührt |
| Flash Attention an statt aus | +14,4 % (729,5 → 834,9 Token/s) | +1,9 % |
| Fork-Schalter `GGML_VK_MMID_SMALLN` aus | −12,4 % | 0 |
| Fork-Schalter `GGML_VK_MMID_ROWLISTS` aus | −6,5 % | 0 |
| Fork-Schalter `GGML_VK_MMID_BM64` aus | −4,6 % | 0 |

## Was nichts bringt

| Hebel | Prompt-Verarbeitung | Textausgabe |
|---|---|---|
| Batch `-b 2048` statt 512 bei festem Mikrobatch | −0,7 % (Rauschen) | — |
| KV-Cache q8_0 statt f16, bei 8192 Token Tiefe | −2,4 % | +2,1 % |
| Fork-Payload mit gebündeltem RADV statt Quelltext-Bau mit System-RADV | −4,2 % | +2,3 % |
| ROCm-Laufzeit 7.2.1 statt 7.1.1 (HIP) | ±0,5 % | ±1 % |

Es zählt also der **Mikro**batch, nicht der Batch. Zur ROCm-Zeile gehört eine Einschränkung:
Getauscht wurden per `LD_PRELOAD` nur `libamdhip64`, `libhsa-runtime64` und `libamd_comgr`;
rocBLAS blieb 7.1.1, weil das Bündel keine gfx1151-Kernel mitbringt. Ein vollständig
installiertes ROCm 7.2 ist damit nicht gemessen.

## HIP gegen Vulkan: geteilt

| | Vulkan (Fork) | HIP (Upstream, ROCm 7.1.1) | |
|---|---|---|---|
| pp512, `-ub 512` | 834,9 Token/s | **939,7 Token/s** | HIP +12,5 % |
| pp2048, `-ub 2048` | 1119,4 Token/s | **1230,2 Token/s** | HIP +9,9 % |
| tg128 | **53,5 Token/s** | 47,6 Token/s | Vulkan +12,4 % |

HIP gewinnt die Prompt-Verarbeitung, Vulkan die Textausgabe. Das ist das Muster, das auch
andere Strix-Halo-Messungen zeigen — und das Gegenteil dessen, was dieses Repo für
DeepSeek-V4 misst, wo Vulkan beides gewinnt ([HIP-BEFUND.md](HIP-BEFUND.md)). Für den Chat
bleibt Vulkan die richtige Wahl; für reine Langprompt-Lasten ist HIP eine Option.

## Spekulatives Dekodieren: EAGLE3 kostet Tempo

Das Repository des Modells liefert ein EAGLE3-Entwurfsmodell mit (0,8 GB als Q8_0).
Gemessen im Serverbetrieb (`--spec-type draft-eagle3`), drei feste Prompts mit rund 2000
Token Vorlauf und 256 Token Ausgabe, Median:

| Entwurfslänge | Textausgabe | Annahmequote | mittlere angenommene Länge |
|---|---|---|---|
| ohne | **51,7 Token/s** | — | — |
| 1 | 43,7 Token/s (−15,5 %) | 0,48 | 1,48 |
| 2 | 38,9 Token/s (−24,7 %) | 0,31 | 1,62 |
| 3 | 32,4 Token/s (−37,3 %) | 0,22 | 1,66 |
| 4 | 27,7 Token/s (−46,4 %) | 0,17 | 1,66 |

Dazu sinkt die Prompt-Verarbeitung mit geladenem Entwurfsmodell um rund 9 %.

**Einschränkung:** Gemessen wurde mit deutschem Text über `/completion` ohne Chat-Vorlage.
gpt-oss ist auf sein Harmony-Format trainiert; im echten Chatbetrieb und mit englischem Text
kann die Annahmequote höher liegen. Belegt ist der Verlust nur für diesen Messfall. Dass bei
einem MoE-Modell jede zusätzlich geprüfte Position weitere Experten aktiviert und damit den
Gewinn auffrisst, ist als Erklärung plausibel, hier aber nicht isoliert gemessen.

## Kontexttiefe

| Tiefe | Prompt-Verarbeitung | Textausgabe |
|---|---|---|
| 0 | 834,2 Token/s | 53,6 Token/s |
| 8192 | 747,7 Token/s (−10,4 %) | 49,1 Token/s (−8,5 %) |
| ~87 000 (eine echte Anfrage über den Server) | 622 Token/s | 29,1 Token/s |

Die letzte Zeile stammt aus dem laufenden Dienst: 87 045 Token Vorlauf in 140 s.

## Warum die Textausgabe am Anschlag ist

Aus dem Kopf der GGUF-Datei (Größe je Tensor aus den Datenoffsets):

| Anteil | gesamt | je Token gelesen |
|---|---|---|
| Experten (MXFP4), 4 von 128 aktiv | 61,07 GB | 1,91 GB |
| Attention-Matrizen | 1,02 GB | 1,02 GB |
| Ausgabematrix | 0,62 GB | 0,62 GB |
| Router und Rest | 0,05 GB | 0,05 GB |
| **Summe** | | **3,59 GB** |

Bei den auf dieser Maschine gemessenen 226 GB/s Speicherbandbreite ergibt das eine
Obergrenze von **62,9 Token/s**. Die gemessenen 53,5 Token/s sind **85 %** davon. Mehr
Textausgabe gibt es nur mit weniger Bytes je Token — und die Experten lassen sich mit den
üblichen GGUF-Formaten nicht kleiner machen: Alle verbreiteten Quantisierungen dieses Modells
sind praktisch gleich groß, weil die Experten MXFP4 bleiben.

## Korrektheit

Der Parameterdurchlauf selbst lief **ohne** Korrektheitsprobe. Nachgeholt wurde sie für die
Betriebseinstellung (`-ub 2048`, Flash Attention an), und zwar über zwei unabhängige
Implementierungen:

| Backend | Perplexität (wikitext, 10 Blöcke à 2048) |
|---|---|
| Vulkan, Fork `50c271f8e` (ggml 0.20.1) | 436,9 ± 13,9 |
| HIP, Upstream `a894dae` (ggml 0.24.0) | 455,5 ± 14,4 |

Beide stimmen innerhalb einer Standardabweichung überein. Der absolut hohe Wert ist kein
Rechenfehler, sondern eine Eigenschaft des Modells: gpt-oss ist stark auf sein Chatformat
nachtrainiert und sagt rohen Wikipedia-Text schlecht vorher. **Wikitext-Perplexität taugt bei
diesem Modell nur zum Vergleich zweier Backends, nicht als Gütemaß** — und bei einem
Grundwert von rund 440 ist sie unempfindlich gegen kleine Fehler. Zusätzlich drei
Wissensfragen über den echten Dienstweg, alle richtig. Die übrigen Zeilen der Tabellen oben
(Fork-Schalter einzeln aus, KV q8_0, HIP mit getauschter Laufzeit) sind nicht einzeln auf
Korrektheit geprüft; wie schnell das schiefgeht, zeigt der zurückgezogene Abschnitt im
[README](README.md#2-zurueckgezogen-moe-kernel-verbessern-nur-den-prefill).

## Einordnung gegen Fremdwerte

Vergleichbar ist nur pp512 mit Standard-Mikrobatch: **834,9 Token/s** hier gegen die in
[BENCHMARKS.md](BENCHMARKS.md) zitierte Fremdmessung mit 719,9 Token/s (Textausgabe dort
56,6 gegen 53,5 Token/s hier). Die 1119 bzw. 1230 Token/s gelten für 2048 Token Vorlauf mit
`-ub 2048` und sind mit pp512-Werten anderer nicht vergleichbar — wer dieselbe Einstellung
setzt, dürfte ähnlich zulegen.

## Empfohlene Einstellung

```bash
llama-server -m gpt-oss-120b-MXFP4.gguf -ngl 999 -fa on -ub 2048 --jinja \
  -c 131072 -np 2 --kv-unified --predict 16384
```

* `-ub 2048` und `-fa on`: die beiden Hebel mit messbarer Wirkung.
* KV-Cache in f16, **kein** Entwurfsmodell.
* `--kv-unified`: ein gemeinsamer KV-Speicher für beide Slots — eine einzelne Anfrage darf
  die vollen 131 072 Token nutzen. Mit starrer Aufteilung scheiterte eine Anfrage mit
  86 820 Token an der Slot-Grenze von 32 768. Mehrkosten rund 2,5 GiB.
* `--predict 16384`: ohne Obergrenze lief ein einzelner Durchgang über 20 000 Token und
  hätte erst beim vollen Kontext aufgehört.
* Speicher mit geladenem Modell: rund 67 GiB. Zusammen mit einem zweiten großen Modell ist
  Vorsicht geboten — auf dieser APU bleibt der amdgpu-Treiber bei etwa 108 GB belegtem
  Speicher mit ENOMEM hängen, und einen GPU-Reset gibt es nicht
  ([README, Abschnitt 12](README.md#12-warnung-iq2-mit-entwurfsmodell-steht-an-der-speichergrenze)).

## Nicht gemessen

120-W-Leistungsmodus, `-ub 4096`, KV q8_0 bei sehr großer Tiefe (rechnerisch der einzige
Einstellungshebel, der die Textausgabe bei langen Kontexten noch heben könnte),
`reasoning_effort` (ändert nicht die Rate, aber die Zahl der Denk-Token und damit die
Wartezeit), n-gram-Spekulation, sowie der offene Upstream-PR
[#27952](https://github.com/ggml-org/llama.cpp/pull/27952) (int8-Matrixkerne für RDNA3, deckt
laut Beschreibung MXFP4 ab).
