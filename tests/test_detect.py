# BIoTan-core — tests for stage 4 (multi-signal detection).
# Source-available under PolyForm Noncommercial License 1.0.0 — see LICENSE.md.

import os
import sys

import numpy as np
import pandas as pd

from biotan import detect as D

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import make_synthetic  # noqa: E402


def test_signal_columns_present():
    long_df, _, _ = make_synthetic.generate(n_per_cohort=6, days=21, seed=21)
    signals, _, _ = D.run_signals(long_df)
    for col in D.SCORE_COLUMNS:
        assert col in signals.table.columns


def test_drift_fault_shows_in_change_signal():
    long_df, _, faults_df = make_synthetic.generate(
        n_per_cohort=6, days=28, n_faults=1, fault_magnitude=0.5, seed=4
    )
    signals, _, _ = D.run_signals(long_df)
    sig = signals.table
    faulty = faults_df.iloc[0]["device_id"]

    fault_change = abs(sig.loc[sig["device_id"] == faulty, "change"].iloc[0])
    healthy_change = sig.loc[sig["device_id"] != faulty, "change"].abs().median()
    # A gradual drift must move the change signal far beyond the healthy background.
    assert fault_change > 3 * max(healthy_change, 1e-6)


def _peerz_table(devices: dict[str, np.ndarray], idx, cohorts: dict[str, int], peer_z=0.0):
    """Build a minimal peer-z long table for testing compute_signals directly."""
    parts = []
    for dev, values in devices.items():
        parts.append(
            pd.DataFrame(
                {
                    "metric": "m",
                    "device_id": dev,
                    "timestamp": idx,
                    "value": values,
                    "cohort": cohorts[dev],
                    "peer_z": peer_z if np.isscalar(peer_z) else peer_z[dev],
                }
            )
        )
    return pd.concat(parts, ignore_index=True)


def test_rigidity_flags_stuck_device():
    # Rigidity is a within-cohort comparison, so test it with all devices in one
    # cohort (clustering would otherwise isolate an always-stuck device).
    rng = np.random.default_rng(0)
    idx = pd.date_range("2026-01-01", periods=24 * 14, freq="h")
    h = idx.hour.to_numpy()
    devices = {
        f"live-{j}": 50 + 20 * np.sin(2 * np.pi * h / 24) + rng.normal(0, 1, len(idx))
        for j in range(6)
    }
    devices["stuck-0"] = 50 + rng.normal(0, 0.01, len(idx))  # frozen sensor
    cohorts = {d: 0 for d in devices}

    signals = D.compute_signals(_peerz_table(devices, idx, cohorts))
    sig = signals.table.set_index("device_id")
    assert sig.loc["stuck-0", "rigidity"] > 3.0
    assert sig.loc["stuck-0", "rigidity"] > sig.drop("stuck-0")["rigidity"].max()


def test_undersized_cohort_signals_are_nan():
    # A device alone in its cohort (size 1 < MIN_PEERS) and with no valid peer-z
    # must get NaN signals — no fabricated scores without peers.
    idx = pd.date_range("2026-01-01", periods=24 * 10, freq="h")
    h = idx.hour.to_numpy()
    devices = {f"norm-{j}": 50 + 20 * np.sin(2 * np.pi * h / 24) for j in range(4)}
    devices["lonely"] = np.full(len(idx), 500.0)
    cohorts = {**{f"norm-{j}": 0 for j in range(4)}, "lonely": 1}
    peer_z = {**{f"norm-{j}": 0.0 for j in range(4)}, "lonely": np.nan}

    signals = D.compute_signals(_peerz_table(devices, idx, cohorts, peer_z=peer_z))
    r = signals.table.set_index("device_id").loc["lonely"]
    assert pd.isna(r["persistent"]) and pd.isna(r["change"]) and pd.isna(r["instability"])
    assert pd.isna(r["rigidity"])  # cohort of size 1 -> no rigidity comparison
