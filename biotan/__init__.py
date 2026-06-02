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
  [x] 3. Common-mode removal (peer-z) -> :mod:`biotan.peerz`
  [x] 4. Multi-signal detection      -> :mod:`biotan.detect`
  [ ] 5. Effect-size gating
  [ ] 6. Backtest lead-time + HTML report
"""

from biotan.cluster import ClusterResult, cluster_fleet
from biotan.detect import SignalScores, compute_signals, run_signals
from biotan.normalize import load, normalize_frame
from biotan.peerz import PeerZResult, compute_peer_z, run_peer_z

__all__ = [
    "load",
    "normalize_frame",
    "cluster_fleet",
    "ClusterResult",
    "compute_peer_z",
    "run_peer_z",
    "PeerZResult",
    "compute_signals",
    "run_signals",
    "SignalScores",
]

__version__ = "0.1.0"
