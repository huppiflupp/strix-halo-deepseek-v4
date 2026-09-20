# HIP auf gfx1151: laedt, ist langsamer, und rechnet je nach Modell falsch

Messtag 2026-09-20. Rohdaten und Skripte in [`messungen/2026-09-20/`](messungen/2026-09-20/).

| | |
|---|---|
| Maschine | Ryzen AI MAX+ 395, Radeon 8060S (gfx1151), 128 GB unified, Nobara 44, Kernel 7.2 |
| GTT | 120 GiB (`ttm.pages_limit=31457280`), kein VRAM-Split |
| ROCm | 7.1.1 (Fedora-Pakete), hipcc 7.1.52802; zusaetzlich 7.2.1 aus dem Ollama-Buendel |
| llama.cpp | Fork `Nathanw1014/llama.cpp`, Commit `50c271f8e`, derselbe Baum fuer Vulkan und HIP |

## 1. Was vom August-Stand ueberholt ist

Der August-Befund lautete: "Das HIP/ROCm-Backend kann das Modell nicht laden, obwohl es
112 GiB meldet: es kommt nicht an GTT." Das stimmt so nicht mehr.

* HIP **laedt** DeepSeek-V4-Flash (UD-IQ3_XXS, 97 GiB) und belegt 103 GiB GTT.
* Es meldet `VMM: no` und sieht die vollen 122880 MiB.
* `-DGGML_HIP_NO_VMM=ON`, das Lucebox in seiner Bauanleitung auffuehrt, ist in llama.cpp
  ohnehin die **Vorgabe** (`option(GGML_HIP_NO_VMM ... ON)`). Es war also nie der fehlende
  Schalter, auch wenn das naheliegend schien.

## 2. Der Fork stuerzt unter HIP ab -- Umgehung: `LLAMA_MOE_F16=0`

Erster Versuch mit dem 97-GiB-Modell:

```
ggml/src/ggml-cpu/ggml-cpu.c:1594: GGML_ASSERT(src1->type == GGML_TYPE_F32) failed
```

Ein Knoten der f16-Expertenaktivierungen faellt unter HIP ins CPU-Backend zurueck, das f32
erwartet. Der Quelltext sagt selbst, wofuer die Kette gebaut ist: *"Every consumer reads f16
natively on the Vulkan backend"* (`src/llama-graph.cpp`, Kommentar vor `llm_graph_moe_f16`).
Mit `LLAMA_MOE_F16=0` laeuft es durch.

Was dabei wie ein Haenger aussah -- 103 GiB belegt, 0 % GPU-Last, vier geforkte Prozesse in
`anon_pipe_read` --, war `ggml_print_backtrace`, das per gdb den Stack zieht.

**Nebenbefund:** Die f16-Kette bringt bei diesem Modell nichts. Vulkan mit 209,24 t/s (pp512)
gegen 206,53 t/s ohne sie, Generierung 18,66 gegen 18,70 -- Rauschen.

## 3. Vulkan ist deutlich schneller, bei gleicher Graphvariante

Beide Seiten mit `LLAMA_MOE_F16=0`, `-fa 1`, `llama-bench -p 512 -n 128`:

| Modell | Backend | pp512 | tg128 |
|---|---|---|---|
| DeepSeek-V4-Flash UD-IQ3_XXS (97 GiB) | Vulkan | **209,24 t/s** | **18,66 t/s** |
| DeepSeek-V4-Flash UD-IQ3_XXS (97 GiB) | HIP | 155,43 t/s | 13,58 t/s |
| Qwen3-1.7B Q8_0 | Vulkan | **6536 t/s** | **107,3 t/s** |
| Qwen3-1.7B Q8_0 | HIP | 5007 t/s | 97,4 t/s |

Vulkan liegt beim grossen Modell 35 % (Prefill) und 37 % (Generierung) vorn.

## 4. Die ROCm-Version erklaert das nicht

Derselbe HIP-Bau, einmal gegen die Systemlaufzeit 7.1.1 und einmal gegen 7.2.1 aus dem
Ollama-Buendel (`LD_PRELOAD` von `libamdhip64`, `libhsa-runtime64`, `libamd_comgr`; das
System-rocBLAS bleibt, weil das Buendel keine gfx1151-Kernel mitbringt):

| | pp512 | tg128 |
|---|---|---|
| ROCm 7.1.1 | 5407,91 ± 40,50 | 96,74 ± 0,53 |
| ROCm 7.2.1 | 5340,24 ± 51,07 | 98,06 ± 0,12 |

Kein Unterschied. Eine 7.2-Installation wuerde die Luecke zu Vulkan also nicht schliessen.

## 5. Der ernste Teil: HIP rechnet modellabhaengig falsch

Perplexitaet auf wikitext, `-c 2048`. Dieselbe Modelldatei, derselbe Binaerbau:

| Modell | Pfad | Perplexitaet |
|---|---|---|
| DeepSeek-V4-Flash UD-IQ3_XXS | Vulkan | 4,5736 ± 0,0804 |
| DeepSeek-V4-Flash UD-IQ3_XXS | **HIP** | **4,5637 ± 0,0771** (richtig) |
| Qwen3-1.7B Q8_0 | Vulkan | 16,5200 |
| Qwen3-1.7B Q8_0 | HIP, CPU-Pfad (`--device none`) | 16,5399 |
| Qwen3-1.7B Q8_0 | **HIP, GPU** | **137509** |
| Qwen3-1.7B Q8_0 | **HIP, GPU, `-fa 0`** | **289344** |

Ausgeschlossen wurde:

* **Flash Attention** -- mit `-fa 0` ist es schlimmer, nicht besser.
* **Die Matmul-Wahl** -- `GGML_CUDA_FORCE_MMQ=1` ergibt 24713, `GGML_CUDA_FORCE_CUBLAS=1`
  ergibt 56575, Vorgabe 57972. Alle drei unbrauchbar.
* **Das Modell** -- dieselbe Datei rechnet auf CPU (16,54) und unter Vulkan (16,52) korrekt.
* **Die ROCm-Version** -- 7.1.1 ergibt 110288, 7.2.1 ergibt 97362.

Die Werte **schwanken zwischen Laeufen** (24 000 bis 290 000). Das deutet auf undefinierten
Speicher oder eine Wettlaufsituation, nicht auf einen systematisch falschen Kernel.

Damit gilt fuer diese Hardware weiterhin, was schon im Abschnitt
[HIP-Korrektheit](README.md#hip-korrektheit-das-backend-rechnete-still-falsch) steht: **Ein
Durchsatzwert unter HIP ist ohne Perplexitaetsprobe wertlos.** Dass DeepSeek-V4 korrekt
rechnet und Qwen3-1.7B nicht, macht es schlimmer, nicht besser -- ein einzelner geglueckter
Test sagt nichts ueber das naechste Modell.

## 6. Offen

* Gegenprobe mit Upstream-llama.cpp (gleicher HIP-Bau, gleicher Test): Fork-Fehler oder
  llama.cpp-Fehler auf gfx1151? -- laeuft
* Falls Upstream ebenfalls betroffen: Fehlerbericht an ggml-org/llama.cpp mit dieser
  Eingrenzung.
* Falls nur der Fork betroffen: Bericht an Nathanw1014 samt `LLAMA_MOE_F16=0`-Umgehung.
