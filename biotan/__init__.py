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
  [x] 5. Effect-size gating          -> :mod:`biotan.gate`
  [x] 6. Backtest lead-time + HTML report -> :mod:`biotan.backtest`, :mod:`biotan.report`
"""

from biotan.api import Result, backtest
from biotan.backtest import BacktestResult, reconstruct_timelines, run_backtest
from biotan.cluster import ClusterResult, cluster_fleet
from biotan.detect import SignalScores, compute_signals, run_signals
from biotan.gate import FlagResult, apply_gate, gate_timeline, run_gate
from biotan.normalize import load, normalize_frame
from biotan.peerz import PeerZResult, compute_peer_z, run_peer_z
from biotan.report import build_report, write_report

__all__ = [
    # clean top-level library API
    "backtest",
    "Result",
    # building blocks
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
    "apply_gate",
    "gate_timeline",
    "run_gate",
    "FlagResult",
    "reconstruct_timelines",
    "run_backtest",
    "BacktestResult",
    "build_report",
    "write_report",
]

__version__ = "0.1.1"
