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
