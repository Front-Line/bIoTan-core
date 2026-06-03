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

A reproducible validation against real public data is included in
[`/validation`](./validation) — run `python validation/run_cmapss.py` to download
the **NASA C-MAPSS FD001** turbofan fleet (100 engines, run to failure with true
failure points) and reproduce the numbers there. On that data, common-mode removal
lifts peer-z above 2σ before failure for 99/100 engines, and the conservative
zero-config gate confirms the fastest-degrading engines with a median lead of ~11
cycles. It also honestly surfaces a limit: the behavioral-profile clustering assumes
daily-cyclic data, so a non-cyclic run-to-failure fleet is best analysed as a single
cohort (the script shows both).

## Quick start

```bash
pip install -r requirements.txt
python -m biotan backtest --input your_data.csv --out report.html
```

Your CSV needs at least three columns: `device_id`, `timestamp`, `value`.
Optional: `metric`, `group`, `unit`. Everything runs locally — **no data ever
leaves your machine, and there is no telemetry.**

Try it on synthetic data with known cohorts and injected faults:

```bash
python scripts/make_synthetic.py --out demo.csv --faults 2 --validate
python -m biotan backtest --input demo.csv --labels demo.faults.csv --out report.html
```

### Commands

The engine runs as one batch pipeline, but each stage is also runnable on its own
(all read a CSV; nothing is configured by hand):

| command | what it does |
|---------|--------------|
| `python -m biotan summarize --input data.csv` | parse + normalize; print fleet summary and inferred cadence |
| `python -m biotan cluster --input data.csv` | discover behavioral cohorts (auto, zero-config) |
| `python -m biotan peerz --input data.csv` | peer-relative deviation (peer-z) timelines, common mode removed |
| `python -m biotan signals --input data.csv` | the four detection signals per device |
| `python -m biotan flag --input data.csv` | apply the effect-size gate; list flagged devices + reasons |
| `python -m biotan backtest --input data.csv --out report.html [--labels failures.csv]` | full pipeline → self-contained HTML report (+ lead time if labels given) |

A labels CSV (for lead time) needs `device_id` and `fault_start` (optional
`metric`). The HTML report is a single self-contained file — inline SVG charts, no
external assets, no network calls.

## License

BIoTan-core is source-available under the
[PolyForm Noncommercial License 1.0.0](./LICENSE.md).
Free for evaluation, research, and noncommercial use.
Commercial or production use, and offering it as a hosted service,
require a separate commercial license — contact contact@frontli.ne.kr.

Copyright (c) 2026 Victor Minbeom Joo d/b/a Front-Line.

---
*BIoTan is open core. Connectors, real-time operation, and fleet management are
available separately.*
