# BIoTan-core — tests for stage 2 (auto-clustering).
# Source-available under PolyForm Noncommercial License 1.0.0 — see LICENSE.md.

import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

from biotan import cluster as C

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import make_synthetic  # noqa: E402


def test_recovers_known_cohorts():
    long_df, truth_df = make_synthetic.generate(n_per_cohort=6, days=14, seed=1)
    results = C.cluster_fleet(long_df)
    res = results["power_kw"]
    truth_map = dict(zip(truth_df["device_id"], truth_df["cohort"]))
    gold = [truth_map[d] for d in res.device_order]
    pred = [res.labels[d] for d in res.device_order]
    ari = adjusted_rand_score(gold, pred)
    assert ari > 0.9  # cohorts are well separated; recovery should be near-perfect


def test_per_metric_independent():
    df_a, _ = make_synthetic.generate(n_per_cohort=5, days=10, metrics=("power_kw",), seed=2)
    df_b, _ = make_synthetic.generate(n_per_cohort=5, days=10, metrics=("temp_c",), seed=3)
    df = pd.concat([df_a, df_b], ignore_index=True)
    results = C.cluster_fleet(df)
    assert set(results.keys()) == {"power_kw", "temp_c"}


def test_tiny_fleet_is_single_cohort():
    rng = np.random.default_rng(0)
    rows = []
    idx = pd.date_range("2026-01-01", periods=48, freq="h")
    for dev in ("x", "y"):  # only 2 devices -> below MIN_DEVICES_TO_CLUSTER
        rows.append(
            pd.DataFrame(
                {"device_id": dev, "timestamp": idx, "value": rng.standard_normal(48), "metric": "m"}
            )
        )
    res = C.cluster_fleet(pd.concat(rows, ignore_index=True))["m"]
    assert res.method == "single"
    assert res.n_cohorts == 1
