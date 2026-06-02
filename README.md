# BIoTan

**Zero-config, peer-relative anomaly detection for fleets of homogeneous IoT assets.**

BIoTan looks at a fleet of similar devices — solar inverters, refrigeration units,
pumps, drives, sensors — and tells you which ones are drifting away from their peers,
*without you configuring thresholds or manually grouping anything*.

It does this by comparing each device not against an absolute limit, but against
what its peers are doing at the same moment. Shared conditions — weather, load,
seasonality — affect every peer equally and cancel out, leaving only the deviations
that actually matter.

This repository is the **free, open core**: a batch backtesting engine. You give it
historical sensor data as CSV; it gives you back cohorts, per-device deviation
timelines, and flagged assets with reasons.

## Why peer-relative?

Most monitoring asks *"is this value above a threshold?"* — which means someone has
to set, tune, and maintain that threshold for every device, and it fires false alarms
whenever a shared condition (a cloudy day, a heavy-load shift) moves the whole fleet
at once.

BIoTan asks a different question: *"is this device behaving differently from its
peers right now?"* That requires no per-device setup, automatically ignores
fleet-wide common-mode changes, and surfaces the genuinely odd unit — the inverter
that's 12% below its neighbors after sun-angle correction, the drive accumulating bad
sectors faster than its cohort, the engine drifting from the healthy baseline as it
nears failure.

## What it does

- **Auto-clustering** — discovers behavioral cohorts from the data. No manual tagging.
- **Common-mode removal** — compares each device to its cohort peers at each timestamp,
  using robust statistics (median / MAD) so a few failing peers don't poison the baseline.
- **Multi-signal detection** — different faults look different, so it tracks several
  orthogonal signals: persistent offset, gradual change/drift, instability, and rigidity
  (a sensor that varies far *less* than its peers — i.e. stuck).
- **Effect-size gating** — flags require both statistical and *practical* significance,
  so near-zero noise doesn't trigger alerts.
- **Backtest timeline** — if you provide known failure/replacement dates, it shows when
  a device first started diverging — i.e. *how many days earlier you could have known.*

## What it does *not* do (and why)

Being honest about the boundary is the point.

- **It is not a guaranteed failure predictor.** Some failures happen with no prior
  signal in the data; no peer-relative method can catch those. BIoTan is a
  **risk-prioritization and degradation-tracking tool** — it tells you what to look at
  first, not that everything unflagged is safe.
- **It does not catch contextual anomalies well** — cases where a value is within normal
  range but wrong *for its context/timing*. Those need richer time-series models or
  user labels.
- **Backtest lead-times are an optimistic upper bound.** Because backtesting tunes to
  data that already happened, real-time results may differ.
- **This core is batch-only.** Real-time ingestion, MQTT/stream/database connectors,
  fleet operations, alert delivery (Slack/PagerDuty/email), and multi-node management
  are **not** part of this repository.

## Validation

The core method has been tested across seven independent datasets spanning very
different sensor physics — synthetic fleets, real air-quality and climate data, real
hard-drive SMART telemetry with true failure labels, NASA turbofan degradation, and
real satellite telemetry. The same pattern held throughout: where a location/condition
leaves a strong enough signature, peer-relative deviation tracks real problems; where
the signal is weak or the anomaly is contextual, the simple method reaches its limit.
The validation scripts are in [`/validation`](./validation).

## Quick start

```bash
pip install -r requirements.txt
python -m biotan backtest --input your_data.csv --out report.html
```

Your CSV needs at least three columns: `device_id`, `timestamp`, `value`.
Optional: `metric`, `group`, `unit`. Everything runs locally — **no data ever
leaves your machine, and there is no telemetry.**

## License

BIoTan-core is source-available under the
[PolyForm Noncommercial License 1.0.0](./LICENSE).
Free for evaluation, research, and noncommercial use.
Commercial or production use, and offering it as a hosted service,
require a separate commercial license — contact contact@frontli.ne.kr.

Copyright (c) 2026 Victor Minbeom Joo d/b/a Front-Line

---
*BIoTan is open core. Connectors, real-time operation, and fleet management are
available separately.*
