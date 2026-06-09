# Changelog

All notable changes to this project are documented here.
This project adheres to [Semantic Versioning](https://semver.org/).

## [0.2.0] — 2026-06-09

Adds **cohort event detection** and a **single-cohort override**. Backward
compatible — the existing backtest pipeline, CLI subcommands, public API, detection
statistics, and the C-MAPSS validation numbers are all unchanged; every addition is
opt-in.

### Added
- **Cohort event detection** — a new `events` subcommand (`python -m biotan events`)
  and API (`biotan.detect_events`, `EventResult`). Given a *fixed* cohort (a column
  you provide, not auto-clustering), it finds WHEN members diverge from their common
  baseline and WHO diverged.
  - **Mode A (default, peer-relative):** robust per-timestamp consensus (median/MAD,
    not a member's own history); works in derivative space; two-sided CUSUM for
    onset; signed, direction-consistent affected subset; an offset-invariant
    level-space effect-size gate (`--min-effect`); automatic daily-aggregation of
    diurnal data; cohort-wide (>50%) co-divergence treated as common-mode, not an
    event. Includes a self-contained inline-SVG event report.
  - **Mode B (experimental, opt-in — `--mode reference`):** reference-trajectory
    deviation for fleets that degrade in lockstep (where Mode A is silent by
    design). Explicitly **not** peer-relative — it uses a learned normal model.
    Builds/scores a portable, versioned reference profile (`--export-reference`,
    `--reference`; API `build_reference`, `score_against_reference`,
    `save_reference`, `load_reference`). Multi-sensor combined early-deviation
    *risk ranking* (a single sensor carries no signal); not an RUL predictor.
- **`--single-cohort`** option on the `cluster`, `peerz`, `signals`, `flag`, and
  `backtest` subcommands (and `force_single_cohort=...` in the API): force every
  device in a metric into one cohort, bypassing auto-clustering, for homogeneous or
  non-cyclic fleets that would otherwise be over-segmented.

## [0.1.1] — 2026-06-05

Metadata-only patch release. No code or behavior changes.

### Changed
- `Homepage` project URL now points to the demo site (`https://biotan.frontli.ne.kr`)
  instead of the GitHub repository (`Repository` still points to GitHub).
- Declared support for Python 3.13 (`Programming Language :: Python :: 3.13`
  classifier); `requires-python` is unchanged at `>=3.9`.
- README install snippet no longer carries the pre-publication "once published to
  PyPI" note (the package is now on PyPI).

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

[0.2.0]: https://github.com/Front-Line/bIoTan-core/releases/tag/v0.2.0
[0.1.1]: https://github.com/Front-Line/bIoTan-core/releases/tag/v0.1.1
[0.1.0]: https://github.com/Front-Line/bIoTan-core/releases/tag/v0.1.0
