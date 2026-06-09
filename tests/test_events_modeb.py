# BIoTan-core — tests for cohort event detection, Mode B (reference-trajectory).
# Source-available under PolyForm Noncommercial License 1.0.0 — see LICENSE.md.

import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from biotan import events as E
from biotan import normalize as N

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "validation"))
import make_event_synthetic as M  # noqa: E402


def _split_ref_target(lifetimes):
    """Reference = the longer-lived (healthier) half; targets = the rest."""
    order = sorted(lifetimes, key=lambda d: -lifetimes[d])
    half = len(order) // 2
    return set(order[:half]), set(order[half:])


def test_mode_b_multi_sensor_beats_single():
    df, lifetimes = M.degradation_fleet()
    df = N.normalize_frame(df)
    ref_u, tgt_u = _split_ref_target(lifetimes)

    ref = E.build_reference(df[df["device_id"].isin(ref_u)])
    scores = E.score_against_reference(df[df["device_id"].isin(tgt_u)], ref)
    scores["life"] = scores["device_id"].map(lifetimes)

    rho_multi = spearmanr(scores["combined_score"], scores["life"]).statistic
    sensor_cols = [c for c in scores.columns if c.startswith("score__")]
    single_rhos = [abs(spearmanr(scores[c], scores["life"], nan_policy="omit").statistic)
                   for c in sensor_cols]
    mean_single = float(np.nanmean(single_rhos))

    # The validated finding: a single sensor barely correlates with lifetime, but the
    # multi-sensor combined score does — reproduce that gap qualitatively.
    assert abs(rho_multi) > 1.8 * mean_single
    assert abs(rho_multi) > 0.4
    assert spearmanr(scores["combined_score"], scores["life"]).pvalue < 0.05


def test_mode_b_reference_roundtrips_identically(tmp_path):
    df, lifetimes = M.degradation_fleet()
    df = N.normalize_frame(df)
    ref_u, tgt_u = _split_ref_target(lifetimes)
    tgt = df[df["device_id"].isin(tgt_u)]

    ref = E.build_reference(df[df["device_id"].isin(ref_u)])
    scores_mem = E.score_against_reference(tgt, ref)

    path = tmp_path / "ref.json"
    E.save_reference(ref, str(path))
    ref_loaded = E.load_reference(str(path))
    scores_io = E.score_against_reference(tgt, ref_loaded)

    pd.testing.assert_frame_equal(scores_mem, scores_io)


def test_mode_b_reference_is_a_portable_versioned_artifact():
    df, _ = M.degradation_fleet(n_units=20)
    df = N.normalize_frame(df)
    ref = E.build_reference(df, metadata={"device_kind": "turbofan", "notes": "demo"})

    assert ref["schema_version"] == E.REFERENCE_SCHEMA_VERSION
    assert ref["kind"] == "biotan.reference-trajectory"
    assert ref["metadata"]["device_kind"] == "turbofan"
    assert ref["metadata"]["n_reference_units"] == 20
    assert "median_reference_life" in ref["metadata"]
    one = next(iter(ref["sensors"].values()))
    assert {"positions", "median", "mad", "scale"} <= set(one)
    assert len(one["positions"]) == len(one["median"]) == len(one["mad"])
