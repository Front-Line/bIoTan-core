# BIoTan-core — tests for the opt-in single-cohort override.
# Source-available under PolyForm Noncommercial License 1.0.0 — see LICENSE.md.

import numpy as np
import pandas as pd

import biotan
from biotan import cluster as C
from biotan import gate as G


def _noncyclic_fleet(levels, drift_device=None, days=90, seed=0):
    """A non-daily fleet (no hour-of-day structure) — the case zero-config
    clustering over-segments. Optionally drift one device downward."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2026-01-01", periods=days, freq="D")
    parts = []
    for i, lvl in enumerate(levels):
        dev = f"d{i}"
        val = lvl + rng.normal(0, max(lvl * 0.01, 0.5), days)
        if dev == drift_device:
            ramp = np.zeros(days)
            ramp[days // 2:] = np.linspace(0, 0.5, days - days // 2)
            val = val * (1 - ramp)
        parts.append(pd.DataFrame({"device_id": dev, "timestamp": idx,
                                   "value": val, "metric": "m"}))
    return pd.concat(parts, ignore_index=True)


def test_single_cohort_helper_returns_one_cohort():
    df = biotan.normalize_frame(_noncyclic_fleet([10, 20, 30, 40, 50]))
    cl = C.single_cohort(df)["m"]
    assert cl.n_cohorts == 1
    assert set(cl.labels.values()) == {0}
    assert cl.method == "single"
    assert sorted(cl.labels) == ["d0", "d1", "d2", "d3", "d4"]


def test_force_single_cohort_makes_a_device_evaluable():
    # d0 is a far-off outlier, so auto-clustering isolates it into a tiny cohort
    # (no peers) while the other five form their own; the override reunites them.
    df = biotan.normalize_frame(
        _noncyclic_fleet([10, 5000, 5050, 5100, 5150, 5200], drift_device="d0")
    )
    # Auto path over-segments this non-cyclic fleet into >1 cohort...
    flags_auto, _s, _pz, cl_auto = G.run_gate(df)
    assert sum(c.n_cohorts for c in cl_auto.values()) > 1
    auto_row = flags_auto.table[flags_auto.table["device_id"] == "d0"].iloc[0]
    assert not auto_row["evaluable"]  # no peer baseline under auto-clustering

    # ...and the override collapses it to one cohort, restoring a peer baseline.
    flags_one, _s2, _pz2, cl_one = G.run_gate(df, force_single_cohort=True)
    assert cl_one["m"].n_cohorts == 1
    assert set(cl_one["m"].labels.values()) == {0}
    one_row = flags_one.table[flags_one.table["device_id"] == "d0"].iloc[0]
    assert one_row["evaluable"]


def test_default_path_unchanged_when_not_requested():
    # Same call without the flag must equal the explicit force_single_cohort=False.
    df = biotan.normalize_frame(_noncyclic_fleet([10, 20, 30, 40, 50, 60]))
    a, *_ = G.run_gate(df)
    b, *_ = G.run_gate(df, force_single_cohort=False)
    pd.testing.assert_frame_equal(a.table, b.table)
