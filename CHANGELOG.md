# Changelog

All notable changes to this project are documented here.
This project adheres to [Semantic Versioning](https://semver.org/).

## [0.1.0] — 2026-06-03

Initial public release of the BIoTan open core.

### Scope
- Batch backtesting engine only: CSV in → console summary + a self-contained HTML
  report out. No real-time ingestion, connectors, streaming, alerting, web server,
  database, network calls, or telemetry. Runs 100% locally.

### Added
- Zero-config, peer-relative anomaly-detection pipeline:
  1. **Input normalization** — arbitrary CSV → canonical long format
     (`device_id, timestamp, value, metric[, group, unit]`); automatic sampling-cadence
     inference and resampling onto a shared time grid.
  2. **Automatic cohort discovery** — behavioral-profile clustering (KMeans with a
     silhouette-selected `k`, and HDBSCAN; the better-scoring result is chosen).
  3. **Common-mode removal** — robust peer-z,
     `(value − peer_median) / (1.4826 · peer_MAD)`, per cohort and timestamp.
  4. **Multi-signal detection** — persistent bias, change/drift, instability, and
     rigidity signals per device.
  5. **Effect-size gating** — a flag requires both statistical significance and a
     practical effect size, guarding against MAD≈0 z-score blow-ups.
  6. **Backtest lead-time** — reconstructs when a device first diverged and, given
     known failure labels, reports the (optimistic, hindsight) lead time; renders a
     self-contained HTML report with inline-SVG charts and a plain-language verdict.
- **CLI** (`biotan` / `python -m biotan`) with six subcommands:
  `summarize`, `cluster`, `peerz`, `signals`, `flag`, `backtest`.
- **Library API** — `biotan.backtest(data, labels=None)` returns a `Result` with
  `.flagged`, `.summary`, and `.to_html(...)` (same pipeline as the CLI). Exposes
  `biotan.__version__`.
- **Synthetic fleet generator** (`scripts/make_synthetic.py`) with injectable faults.
- **Reproducible real-data validation** against the NASA C-MAPSS FD001 run-to-failure
  fleet (`validation/run_cmapss.py`).
- **Packaging** — installable as `biotan` (`pyproject.toml`, console entry point,
  PEP 639 SPDX license metadata).

### Dependencies
- Python ≥ 3.9; pandas, numpy, scikit-learn, scipy.

### License
- Source-available under PolyForm Noncommercial 1.0.0 (noncommercial use only).

[0.1.0]: https://github.com/Front-Line/bIoTan-core/releases/tag/v0.1.0
