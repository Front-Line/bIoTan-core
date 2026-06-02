# BIoTan-core — tests for stage 6 (backtest lead-time + HTML report).
# Source-available under PolyForm Noncommercial License 1.0.0 — see LICENSE.md.

import os
import sys

import pandas as pd

from biotan import backtest as B
from biotan import report as R

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import make_synthetic  # noqa: E402


def test_lead_time_is_positive_for_labeled_fault():
    long_df, _, faults_df = make_synthetic.generate(
        n_per_cohort=6, days=28, n_faults=1, fault_magnitude=0.6, seed=4
    )
    result, *_ = B.run_backtest(long_df, labels=faults_df)
    faulty = faults_df.iloc[0]["device_id"]
    row = result.table[result.table["device_id"] == faulty].iloc[0]

    assert row["flagged"]
    assert pd.notna(row["divergence_start"])
    # Divergence must be found before the labeled failure -> positive lead time.
    assert row["lead_time_days"] > 0
    assert row["divergence_start"] < faults_df.iloc[0]["fault_start"]


def test_healthy_devices_have_no_anomaly():
    long_df, _, _ = make_synthetic.generate(n_per_cohort=6, days=21, seed=31)
    result, *_ = B.run_backtest(long_df)
    statuses = set(result.table["status"])
    assert result.table["flagged"].sum() == 0
    assert statuses <= {"no anomaly", "anomaly onset (no failure label)", "not evaluable (no peer baseline)"}


def test_no_clear_precursor_is_reported_honestly():
    # A label on a device that has no real divergence -> honest "no clear precursor".
    long_df, _, _ = make_synthetic.generate(n_per_cohort=6, days=21, seed=2)
    some_device = long_df["device_id"].unique()[0]
    labels = pd.DataFrame({"device_id": [some_device], "fault_start": [pd.Timestamp("2026-01-15")]})
    result, *_ = B.run_backtest(long_df, labels=labels)
    row = result.table[result.table["device_id"] == some_device].iloc[0]
    assert row["status"] in {"no clear precursor", "detected only at/after failure"}
    assert pd.isna(row["lead_time_days"])


def test_html_report_is_self_contained(tmp_path):
    long_df, _, faults_df = make_synthetic.generate(
        n_per_cohort=6, days=28, n_faults=1, fault_magnitude=0.6, seed=4
    )
    out = tmp_path / "report.html"
    R.write_report(str(out), long_df, labels=faults_df)
    text = out.read_text(encoding="utf-8")

    assert text.startswith("<!doctype html>")
    assert "<svg" in text                         # charts are inline SVG
    assert "http://" not in text.replace("http://www.w3.org/2000/svg", "")  # no external refs
    assert "optimistic upper bound" in text       # honesty disclaimer present
