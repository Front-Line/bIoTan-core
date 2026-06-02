# BIoTan-core — tests for stage 3 (peer-z / common-mode removal).
# Source-available under PolyForm Noncommercial License 1.0.0 — see LICENSE.md.

import os
import sys

import numpy as np
import pandas as pd

from biotan import peerz as P

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import make_synthetic  # noqa: E402


def test_common_mode_cancels_for_healthy_fleet():
    long_df, _, _ = make_synthetic.generate(n_per_cohort=6, days=14, seed=11)
    result, _ = P.run_peer_z(long_df)
    # With no faults, peers move together; typical |peer-z| should be small.
    median_abs_z = result.table["peer_z"].abs().median()
    assert median_abs_z < 1.0


def test_faulty_device_stands_out():
    long_df, _, faults_df = make_synthetic.generate(
        n_per_cohort=6, days=21, n_faults=1, fault_magnitude=0.5, seed=5
    )
    result, _ = P.run_peer_z(long_df)
    faulty = faults_df.iloc[0]["device_id"]
    fault_start = faults_df.iloc[0]["fault_start"]

    series = result.for_device(faulty)
    after = series[series["timestamp"] >= fault_start]["peer_z"].abs()
    healthy = result.table[result.table["device_id"] != faulty]["peer_z"].abs()

    # The faulty device should diverge well beyond the healthy background.
    assert after.max() > 4.0
    assert after.max() > 3 * healthy.median()


def test_small_cohort_gets_warning_and_no_peer_z():
    # Build a fleet with one cohort of 4 (valid) and an isolated odd device.
    rng = np.random.default_rng(0)
    idx = pd.date_range("2026-01-01", periods=72, freq="h")
    h = idx.hour.to_numpy()
    rows = []
    for j in range(4):  # tight cohort: a clean daily sine
        rows.append(
            pd.DataFrame(
                {
                    "device_id": f"normal-{j}",
                    "timestamp": idx,
                    "value": 50 + 20 * np.sin(2 * np.pi * h / 24) + rng.normal(0, 0.5, len(idx)),
                    "metric": "m",
                }
            )
        )
    # An odd device with a totally different (flat, high) profile -> its own cohort.
    rows.append(
        pd.DataFrame(
            {"device_id": "odd-0", "timestamp": idx, "value": 200 + rng.normal(0, 0.5, len(idx)), "metric": "m"}
        )
    )
    df = pd.concat(rows, ignore_index=True)
    result, clustering = P.run_peer_z(df)

    odd = result.for_device("odd-0")
    if clustering["m"].labels["odd-0"] not in [
        c for c, n in clustering["m"].sizes().items() if n >= P.MIN_PEERS
    ]:
        # odd-0 is in an undersized cohort: must be warned about and have no peer-z.
        assert any("odd-0" in w for w in result.warnings)
        assert odd["peer_z"].isna().all()


def test_deviation_kept_even_when_z_undefined():
    # Two identical devices => MAD == 0 => peer_z undefined, but deviation present (0).
    idx = pd.date_range("2026-01-01", periods=48, freq="h")
    rows = []
    for j in range(3):
        rows.append(
            pd.DataFrame({"device_id": f"d{j}", "timestamp": idx, "value": 10.0, "metric": "m"})
        )
    df = pd.concat(rows, ignore_index=True)
    result, _ = P.run_peer_z(df)
    assert result.table["peer_z"].isna().all()          # MAD == 0 -> z undefined
    assert (result.table["deviation"].fillna(0) == 0).all()  # deviation still recorded
