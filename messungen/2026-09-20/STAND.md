# Stand 2026-09-20, 19:30 — Wiederaufnahme nach dem Neustart

Die Sitzung lief auf ai395 und endete mit dem Neustart, der noetig wurde, weil der
amdgpu-Treiber bei einer Messung mit ENOMEM stehenblieb (siehe Repo, Abschnitt 12).

## Sofort nach dem Neustart pruefen

```bash
free -g                                    # sollte ~110 GB frei zeigen
systemctl --user list-units 'llama-*'      # Familiendienste kommen von selbst hoch
curl -sf http://127.0.0.1:8093/health      # Familienchat erreichbar?
```

Falls der Familienchat ueber den Netznamen nicht geht:
`systemctl --user start llama-qwen36-familie-proxy.service`

## Was offen ist, in dieser Reihenfolge

Alle Skripte liegen in `~/bench/v4-gegen-qwen/` und sind gegen den Speicherfehler
abgesichert (warten auf freien GTT, `-c 4096` statt 16384).

1. **Entwurfslaengen n=4/5/6 bei IQ2** — `v4-n45.sh`
   Offen, weil genau hier der Treiber haengenblieb. n=3 ergab 28,91 Token/s,
   n=2 ergab 27,46. Ob n=4 noch besser ist, ist unbekannt.
   **Vorsicht:** IQ2 (85 GiB) + Entwurfsmodell (10,6 GiB) ist die Kombination, die
   gekippt ist. Vor dem Start pruefen, dass wirklich >100 GB frei sind.

2. **Lucebox-Reproduktion** — `lucebox-lauf.sh`
   Server ist gebaut (`~/src/lucebox/server/build-hip/dflash_server`), Modell und
   Entwurfsmodell sind geladen (`~/models/lucebox/`, 114 GB).
   Misst drei Faelle: ohne Entwurf (ihre Angabe 25,31 Token/s), mit DSpark q=4 (32,0),
   und zusaetzlich sechs statt vier Experten als Gegenprobe, die ihr Blogbeitrag nicht hat.

3. **Kognitiver Vergleich** — `v4-kognitiv.sh`, rund 4,5 Stunden, laeuft ueber Nacht.
   Qwen3.6-35B-A3B Bezugswert steht: MMLU-Pro 81,3 %, GSM8K 97,5 %, HumanEval 95,1 %.
   Fehlt: dieselben drei Suiten mit DeepSeek-V4 IQ3_XXS + DSpark n=2 auf Port 8092.

Starten jeweils als eigene systemd-Einheit, nicht als Hintergrundprozess der
Claude-Sitzung — sonst raeumt die Speicherautomatik sie beim naechsten Engpass ab:

```bash
systemd-run --user --unit=v4-n45 --property=RuntimeMaxSec=5400 \
  /bin/bash ~/bench/v4-gegen-qwen/v4-n45.sh
```

## Was fertig und gesichert ist

Alles im Repo `huppiflupp/strix-halo-deepseek-v4` (Rohdaten in `messungen/2026-09-20/`):

* Bestwert **28,91 Token/s** (IQ2 + DSpark n=3), Prompt-Vorlauf 261,7 Token/s
* Entwurfslaengen n=1/2/3 bei IQ3, Ladeart mmap gegen mlock (kein Unterschied)
* Kontexttiefen 0/8192/32768
* Fork gegen aktuelles Upstream: Textausgabe gleichauf, Prompt-Vorlauf +49 % fuer den
  Fork bei 8192 Token Vorlauf
* HIP: laedt, rechnet bei DeepSeek richtig, bei Qwen3-1.7B falsch (Issue #28211,
  im Fork noch enthalten, in Upstream behoben), 35 % langsamer als Vulkan
* ROCm 7.1.1 gegen 7.2.1: kein Unterschied
* Ablation der Fork-Schalter mit Korrektheitsproben
* Zurueckgezogen: die erste MMID-Messung (verglich korrektes gegen kaputtes Rechnen)

## Zwei Fallen, die Zeit gekostet haben

1. **Abgeschaltete Optimierungen brauchen ebenfalls eine Korrektheitsprobe.** Mit allen
   MMID-Schaltern auf 0 rechnet das Modell `nan` — die daraus gewonnene Zahl war wertlos.
2. **`GGML_VK_MMID_WAVE32=0` sieht wie +41 % aus und liefert `nan`.** Ebenso stuerzt
   `GGML_VK_MMID_F16B=0` ab. Beide Schalter tragen die Korrektheit mit.

## Aufraeumen, wenn die Messungen durch sind

Drei Fassungen desselben Modells belegen rund 300 GB:

| Pfad | Groesse | danach |
|---|---|---|
| `~/models/DeepSeek-V4-Flash-0731` (IQ3_XXS + DSpark) | 108 GB | behalten — beste Perplexitaet (4,57) |
| `~/models/DeepSeek-V4-Flash-0731-IQ2` (IQ2_XXS) | 85 GB | loeschen, sobald n=4/5/6 gemessen ist (PPL 5,21) |
| `~/models/lucebox` (ROCMFP2 + DSpark-Entwurf) | 114 GB | loeschen, sobald die Reproduktion durch ist |
