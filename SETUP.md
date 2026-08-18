# Produktivsetup: DeepSeek-V4-Flash als lokaler Agent-Backend

Anleitung fuer den Dauerbetrieb auf AMD Strix Halo (128 GB unified), so wie es hier laeuft.
Ergebnis: **22–24 t/s** mit Tool-Calling, automatischer Start beim Booten.

Die Messwerte und ihre Herleitung stehen in [BENCHMARKS.md](BENCHMARKS.md).

## Voraussetzungen

### 1. BIOS: UMA auf Minimum

Nicht auf `Auto` oder gar 96 GB stellen. Ein fest reservierter VRAM-Block ist eine harte
Wand, aus der die GPU nicht herauskann, und nimmt dem System gleichzeitig RAM weg.
Auf `512M` bzw. das Minimum stellen — der Speicher kommt dann komplett ueber GTT.

### 2. GTT-Limit auf 120 GiB

```bash
sudo grubby --update-kernel=ALL \
  --args="ttm.pages_limit=31457280 ttm.page_pool_size=31457280"
```

**Das ist keine Feinjustierung, sondern Voraussetzung.** Modell (97 GiB) und DSpark-Draft
(10,5 GiB) belegen zusammen 111,3 GiB. Mit dem naheliegenden Limit von 110 GiB bleiben nur
1,3 GiB Puffer — und der reicht nicht, sobald ein Agent mit langem System-Prompt anfragt:

```
vk::Queue::submit: ErrorDeviceLost
kernel: amdgpu: Not enough memory for command submission!
```

Waehrend einer Agent-Anfrage steigt die GTT-Belegung um genau ~1,3 GiB fuer die
Command-Submission. Mit 120 GiB bleiben 11,5 GiB Puffer und der Betrieb ist stabil.
Kleinere Batch-Groessen (`-b 512 -ub 512`) helfen als Abhilfe **nicht**.

Die GPU erholt sich nach einem Device-Lost folgenlos, der Server bleibt aber unbrauchbar.

### 3. Der Fork, nicht Mainline

```bash
mkdir -p ~/strix-fork && cd ~/strix-fork
curl -sLO https://github.com/Nathanw1014/strix-halo-llamacpp/releases/download/v0.6.4/strix-halo-llamacpp-vulkan-portable.tar.gz
tar xzf strix-halo-llamacpp-vulkan-portable.tar.gz
```

Portables Payload mit gebuendeltem RADV-Treiber, kein Systemeingriff. Mainline-llama.cpp
kommt auf 11,9 t/s, der Fork auf 18,9 — und nur im Fork bringt das Draft-Modell etwas.

## Serverstart

`~/serve-v4.sh`:

```bash
#!/bin/bash
F=$HOME/strix-fork/vulkan
D=$HOME/models/DeepSeek-V4-Flash-0731
ARGS=(
  -m $D/UD-IQ3_XXS/DeepSeek-V4-Flash-0731-UD-IQ3_XXS-00001-of-00004.gguf
  -ngl 999 -c 65536 -b 512 -ub 512 --jinja --host 0.0.0.0 --port 8080
  --alias deepseek-v4-flash -np 1
)
if [ "$1" = "draft" ]; then
  ARGS+=( -md $D/dspark/dspark-DeepSeek-V4-Flash-0731-BF16.gguf -ngld 999 --spec-draft-n-max 3 )
fi
exec "$F/llama-server" "${ARGS[@]}"
```

`--jinja` ist fuer Tool-Calling zwingend: V4-Flash nutzt intern ein XML-Format
(`<|tool_calls|><|invoke name=…|>`), das llama-server erst damit nach OpenAI-JSON uebersetzt.

Der KV-Cache kostet dank DeepSeeks MLA nur ~0,7 GiB fuer die vollen 64k — bei diesem Modell
ist die Kontextlaenge also kein Speicherproblem.

## Autostart

### llama-server als User-Service

`~/.config/systemd/user/llama-v4.service` — kein sudo noetig, sofern Lingering aktiv ist
(`loginctl enable-linger $USER`):

```ini
[Unit]
Description=llama-server: DeepSeek-V4-Flash-0731 (Fork + DSpark-Draft)
After=network-online.target
Wants=network-online.target

[Service]
Type=exec
ExecStart=%h/serve-v4.sh draft
TimeoutStartSec=600
TimeoutStopSec=120
Restart=on-failure
RestartSec=30
OOMPolicy=kill
OOMScoreAdjust=200

[Install]
WantedBy=default.target
```

`TimeoutStartSec=600` ist noetig — das Laden von 110 GiB dauert laenger als systemds
Default von 90 s. `OOMPolicy=kill` sorgt dafuer, dass bei Speicherknappheit dieser Dienst
stirbt und nicht ein systemd-Prozess: der globale OOM-Killer hat die Maschine hier schon
einmal in einen Reboot gerissen.

```bash
systemctl --user daemon-reload
systemctl --user enable --now llama-v4.service
journalctl --user -u llama-v4 -f
```

**Stolperfalle:** Ein von Hand gestarteter Server belegt Port 8080 und laesst den Dienst in
`activating` haengen. Erst den Handstart beenden.

### GPU-Takt fixieren

`power_dpm_force_performance_level` faellt bei jedem Reboot auf `auto` zurueck.
`/etc/systemd/system/gpu-perf.service`:

```ini
[Unit]
Description=Radeon 8060S auf hohen Takt fixieren (LLM-Betrieb)
After=multi-user.target

[Service]
Type=oneshot
RemainAfterExit=yes
ExecStart=/bin/sh -c "echo high > /sys/class/drm/card1/device/power_dpm_force_performance_level"
ExecStop=/bin/sh -c "echo auto > /sys/class/drm/card1/device/power_dpm_force_performance_level"

[Install]
WantedBy=multi-user.target
```

`RemainAfterExit=yes` nicht vergessen, sonst zeigt `systemctl status` dauerhaft
`inactive (dead)`, obwohl die Einstellung wirkt.

Der CPU-Governor laeuft ueber `tuned` und ist im Gegensatz dazu persistent:

```bash
sudo systemctl enable --now tuned
sudo tuned-adm profile accelerator-performance
```

Ehrlicherweise: Beide Taktmassnahmen zusammen bringen beim Decode ~3 %, im Alltag geht das
in der normalen Schwankung unter. Der Fork bringt +58 %.

## Anbindung an Hermes Agent

`~/.hermes/config.yaml`:

```yaml
model:
  api_key: dummy
  base_url: http://127.0.0.1:8080/v1
  default: deepseek-v4-flash
  provider: custom
  context_length: 65536
  max_tokens: 8192
```

`context_length` muss mindestens 64.000 betragen — darunter startet Hermes nicht
(`MINIMUM_CONTEXT_LENGTH`). Der Wert muss zum `-c` des Servers passen, sonst bricht die
Kompression zum falschen Zeitpunkt los.

Verifiziert: Hermes ruft eigenstaendig das Terminal-Tool auf und liefert korrekte Ergebnisse.


# Drei Modelle, umschaltbar

Fuer Agent-Betrieb ist V4-Flash mit 22–24 t/s zaeh. **Qwen3-30B-A3B-Instruct-2507**
(30,5 B MoE, 3 B aktiv, IQ4_XS, 15,25 GiB) liefert auf derselben Maschine **83,5 t/s** und
beherrscht Tool-Calling ebenfalls — im Standard-Qwen-Format (`<tool_call>` mit JSON), das
llama.cpp besser unterstuetzt als V4-Flashs XML.

| | DeepSeek-V4-Flash | Qwen3-30B-A3B | LFM2.5-8B-A1B |
|---|---:|---:|---:|
| Parameter | 284 B (13 B aktiv) | 30,5 B (3 B aktiv) | 8,5 B (1 B aktiv) |
| Quant / Groesse | IQ3_XXS, 97 GiB | IQ4_XS, 15,25 GiB | Q4_K_M, 4,95 GiB |
| Belegung | 111,3 GiB (mit Draft) | 28,1 GiB | ~10 GiB |
| llama-bench tg128 | 18,85 | 87,57 | **155,02** |
| llama-bench pp512 | 207,5 | 1550 | **3957,9** |
| Decode ueber API | 22–24 t/s | 83,5 t/s | **144–150 t/s** |
| Kontext | 64k | 128k | 128k |
| Agent-Turn (Tool + Antwort) | Minuten | 39 s | **8 s** |
| Tool-Calling | XML, via `--jinja` | Qwen-JSON | JSON |

Beide Dienste teilen sich Port 8080 und den GPU-Speicher, koennen also nicht gleichzeitig
laufen. `Conflicts=` in beiden Units laesst systemd automatisch umschalten:

```bash
systemctl --user start llama-qwen    # stoppt llama-v4 automatisch
systemctl --user start llama-v4      # und umgekehrt
```

Danach in `~/.hermes/config.yaml` `default` und `context_length` anpassen
(`qwen3-30b-a3b` / 131072 bzw. `deepseek-v4-flash` / 65536).

```bash
systemctl --user start llama-lfm     # 150 t/s, Agent-Turn in 8 s
systemctl --user start llama-qwen    # 83 t/s
systemctl --user start llama-v4      # 23 t/s, dafuer 284 B
```

**Empfehlung nach Aufgabe:**

* **Qwen3-30B-A3B ist die richtige Wahl fuer Agent-Betrieb.** 83 t/s, und es loest Aufgaben
  zuverlaessig.
* **LFM2.5-8B-A1B nur fuer triviale Einzelaufrufe.** Die 150 t/s sind verlockend, aber 1 B
  aktive Parameter reichen nicht fuer mehrstufiges Problemloesen — siehe unten.
* **DeepSeek-V4-Flash** fuer einzelne schwere Aufgaben, wo Modellqualitaet ueber Durchsatz geht.

# Was aus dem strix-halo-guide NICHT noetig war

Der [strix-halo-guide](https://github.com/hogeheer499-commits/strix-halo-guide) empfiehlt
einige Punkte, die hier entweder schon erfuellt oder nicht uebertragbar sind:

| Empfehlung | Status hier |
|---|---|
| `amdgpu.gttsize=131072` | nicht gesetzt — `ttm.pages_limit` ist der modernere Weg und genuegt |
| AMDVLK entfernen | war nie installiert, nur RADV vorhanden |
| Mesa aus kisak-mesa PPA | Ubuntu-spezifisch; Fedora hat mit 26.1.5 eine neuere Version |
| `OLLAMA_VULKAN` / `OLLAMA_IGPU_ENABLE` | irrelevant — hier laeuft llama-server, nicht Ollama |
| `tuned accelerator-performance` | gesetzt |
| UMA auf 512 MB | gesetzt |
| IOMMU aktiviert lassen | Default, unveraendert (`amd_iommu=off` haette die NPU deaktiviert) |

Das `setup.sh` des Guides selbst ist **nicht lauffaehig auf Fedora/Nobara** — es nutzt `apt`,
`update-grub`, `update-initramfs` und `add-apt-repository`.


# Warnung: Geschwindigkeit ersetzt keine Faehigkeit

LFM2.5-8B-A1B ist mit 150 t/s knapp doppelt so schnell wie Qwen3-30B-A3B und besteht einfache
Tool-Tests fehlerfrei (5/5 korrekte Aufrufe). Bei mehrstufigen Aufgaben faellt es jedoch aus —
und zwar **stillschweigend**.

Testfall: *"What is the current time in Peking? Use the terminal tool."*
(Soll: 06:43 lokaler Zeit entsprechend; die Maschine steht auf CEST/UTC+2.)

| Modell | Antwort | |
|---|---|---|
| Qwen3-30B-A3B | `2026-08-19 06:43:40 CST` | korrekt, mit Zeitzonenkuerzel |
| LFM2.5-8B-A1B | `2026-08-18T22:43:23` | **falsch** — UTC statt CST, dazu falsches Datum |

Der Ablauf im Agent-Log zeigt das Muster:

```
python3 -c "import pytz; ..."        -> exit 1   (pytz nicht installiert)
date ... + 2 commands                -> exit 1   (naechster Versuch scheitert)
date +"%Y-%m-%d %H:%M:%S"            -> OK       (aber das ist die LOKALE Zeit)
```

Nach zwei Fehlschlaegen gibt das Modell den Zeitzonen-Teil auf, holt die lokale Zeit und
praesentiert sie als Ergebnis. Der naheliegende Weg `TZ=Asia/Shanghai date` kommt ihm nicht in
den Sinn. Entscheidend: **es meldet keinen Fehler**, sondern antwortet mit voller Ueberzeugung
falsch.

**Konsequenz fuer die Modellwahl:** Ein Agent muss erkennen koennen, dass sein erster Versuch
gescheitert ist, und einen anderen Weg waehlen. Diese Faehigkeit skaliert mit den aktiven
Parametern, nicht mit dem Durchsatz. Bei Benchmark-Tabellen wie diesem Repo lohnt daher die
Erinnerung: t/s sagt nichts darueber, ob die Antwort stimmt.
