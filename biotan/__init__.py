# BIoTan-core — zero-config, peer-relative anomaly backtesting (free open core).
# Copyright (c) 2026 Victor Minbeom Joo d/b/a Front-Line (https://frontli.ne.kr).
# Source-available under the PolyForm Noncommercial License 1.0.0 — noncommercial use only.
# Commercial or production use requires a separate license. See LICENSE.md (authoritative).
"""BIoTan-core: a batch backtesting engine for peer-relative IoT anomaly detection.

This package is the free, open core. It is strictly batch (CSV in -> report out):
no real-time ingestion, no connectors, no network I/O, no telemetry. Everything
runs locally.

Stage status (built incrementally — see the project roadmap):
  [x] 1. Input normalization   -> :mod:`biotan.normalize`
  [x] 2. Auto-clustering       -> :mod:`biotan.cluster`
  [ ] 3. Common-mode removal (peer-z)
  [ ] 4. Multi-signal detection
  [ ] 5. Effect-size gating
  [ ] 6. Backtest lead-time + HTML report
"""

from biotan.cluster import ClusterResult, cluster_fleet
from biotan.normalize import load, normalize_frame

__all__ = [
    "load",
    "normalize_frame",
    "cluster_fleet",
    "ClusterResult",
]

__version__ = "0.1.0"
