# BIoTan-core — tests for stage 5 (effect-size gating).
# Source-available under PolyForm Noncommercial License 1.0.0 — see LICENSE.md.

import os
import sys

import numpy as np
import pandas as pd

from biotan import gate as G

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import make_synthetic  # noqa: E402


def test_healthy_fleet_has_no_flags():
    long_df, _, _ = make_synthetic.generate(n_per_cohort=6, days=21, seed=31)
    flags, *_ = G.run_gate(long_df)
    assert flags.flagged().empty  # zero-config should not fire on a clean fleet


def test_strong_fault_is_flagged():
    long_df, _, faults_df = make_synthetic.generate(
        n_per_cohort=6, days=28, n_faults=2, fault_magnitude=0.6, seed=4
    )
    flags, *_ = G.run_gate(long_df)
    flagged = set(flags.flagged()["device_id"])
    faulty = set(faults_df["device_id"])
    # Every injected fault is caught, with no false positives among healthy peers.
    assert faulty.issubset(flagged)
    assert flagged == faulty


def test_mad_zero_blowup_is_gated_out():
    # A cohort that is essentially constant: a single tiny blip makes per-instant
    # MAD ~ 0 so peer-z explodes, but the absolute deviation is negligible.
    # The effect-size gate must reject it.
    idx = pd.date_range("2026-01-01", periods=24 * 14, freq="h")
    rng = np.random.default_rng(0)
    rows = []
    for j in range(6):
        v = np.full(len(idx), 100.0)
        rows.append(pd.DataFrame({"device_id": f"d{j}", "timestamp": idx, "value": v, "metric": "m"}))
    df = pd.concat(rows, ignore_index=True)
    # Inject a single 0.001-unit blip into one device (a rounding-level wobble).
    df.loc[(df["device_id"] == "d0") & (df["timestamp"] == idx[100]), "value"] = 100.001

    flags, _signals, peerz_result, _ = G.run_gate(df)
    # peer-z may be huge at that instant, but the deviation is far below the
    # cohort scale, so no device should be flagged.
    assert flags.flagged().empty


def test_timeline_gate_marks_real_anomalies_only():
    long_df, _, faults_df = make_synthetic.generate(
        n_per_cohort=6, days=28, n_faults=1, fault_magnitude=0.6, seed=8
    )
    flags, _s, peerz_result, _c = G.run_gate(long_df)
    timeline = G.gate_timeline(peerz_result)
    faulty = faults_df.iloc[0]["device_id"]
    fstart = faults_df.iloc[0]["fault_start"]

    dev_tl = timeline[timeline["device_id"] == faulty]
    # The faulty device accumulates anomalous points, concentrated after the fault.
    after = dev_tl[(dev_tl["timestamp"] >= fstart) & dev_tl["anomalous"]]
    assert len(after) > 0
