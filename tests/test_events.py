# BIoTan-core — tests for cohort event detection, Mode A (biotan.events).
# Source-available under PolyForm Noncommercial License 1.0.0 — see LICENSE.md.

import os
import sys

from biotan import events as E
from biotan import normalize as N

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "validation"))
import make_event_synthetic as M  # noqa: E402


def test_mode_a_freezer_recovers_interval_and_signed_subset():
    df, truth = M.freezer()
    res = E.detect_events(N.normalize_frame(df), cohort_col="group")

    assert len(res.events) == 1
    e = res.events[0]
    # exactly the near-door subset, warming (+), and nobody else
    assert sorted(a.device_id for a in e.affected) == truth["affected"]
    assert e.directions == "+"
    # the event interval overlaps the true door-open window
    assert e.t_start <= truth["t_end"] and e.t_end >= truth["t_start"]


def test_mode_a_solar_diurnal_aggregates_and_recovers_faulty():
    df, truth = M.solar()
    res = E.detect_events(N.normalize_frame(df), cohort_col="group")

    # diurnal data must be period-aggregated before differencing
    assert res.aggregated["power_kw"] == "daily"
    assert len(res.events) >= 1
    affected = set()
    for e in res.events:
        affected |= {a.device_id for a in e.affected}
    # exactly the injected faulty inverters — no noisy-but-healthy members pulled in
    assert affected == set(truth["affected"])
    assert all(e.directions == "-" for e in res.events)  # degraded = lower output


def test_mode_a_lockstep_stays_silent():
    # Everyone degrades together -> no intra-cohort divergence -> no events.
    # This is the NASA C-MAPSS boundary: the correct answer is silence, not noise.
    for seed in range(6):
        df, _ = M.lockstep(seed=seed)
        res = E.detect_events(N.normalize_frame(df), cohort_col="group")
        assert len(res.events) == 0


def test_mode_a_default_cohort_is_all_when_no_column():
    # Without a cohort column every device in a metric is one cohort (not clustered).
    df, truth = M.freezer()
    res = E.detect_events(N.normalize_frame(df))  # no cohort_col
    assert len(res.events) == 1
    assert res.events[0].cohort == "all"
