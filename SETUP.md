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
