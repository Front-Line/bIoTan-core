# BIoTan-core — tests for the top-level library API (biotan.backtest).
# Source-available under PolyForm Noncommercial License 1.0.0 — see LICENSE.md.

import os
import sys

import pandas as pd

import biotan

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import make_synthetic  # noqa: E402


def test_three_line_api(tmp_path):
    long_df, _, faults = make_synthetic.generate(
        n_per_cohort=6, days=28, n_faults=2, fault_magnitude=0.6, seed=4
    )
    csv = tmp_path / "demo.csv"
    long_df.to_csv(csv, index=False)
    lab = tmp_path / "faults.csv"
    faults.to_csv(lab, index=False)

    result = biotan.backtest(str(csv), labels=str(lab))

    # documented: .flagged is a DataFrame of flagged devices with reasons
    assert isinstance(result.flagged, pd.DataFrame)
    assert {"device_id", "reasons"}.issubset(result.flagged.columns)
    assert len(result.flagged) >= 1
    # lead time present because labels were given
    assert "lead_time_days" in result.flagged.columns

    # documented: .summary is a dict of counts
    s = result.summary
    assert s["devices"] == long_df["device_id"].nunique()
    assert s["records"] == len(long_df)
    assert s["flagged"] >= 1
    assert isinstance(s["metrics"], int) and s["metrics"] >= 1

    # documented: .to_html writes a self-contained report
    out = tmp_path / "report.html"
    written = result.to_html(str(out))
    assert os.path.exists(written)
    assert out.read_text(encoding="utf-8").startswith("<!doctype html>")

    assert isinstance(biotan.__version__, str) and biotan.__version__


def test_accepts_dataframe_and_optional_labels():
    long_df, _, _ = make_synthetic.generate(n_per_cohort=6, days=21, seed=31)
    result = biotan.backtest(long_df)  # DataFrame in, labels omitted
    assert result.summary["devices"] > 0
    assert isinstance(result.flagged, pd.DataFrame)


def test_results_match_the_cli_pipeline(tmp_path):
    # The library must use the same code path as run_backtest (identical numbers).
    long_df, _, faults = make_synthetic.generate(
        n_per_cohort=6, days=28, n_faults=1, fault_magnitude=0.6, seed=8
    )
    api_result = biotan.backtest(long_df, labels=faults)
    df = biotan.normalize_frame(long_df)
    bt, _flags, _pz, _cl = biotan.run_backtest(df, labels=faults)
    pd.testing.assert_frame_equal(
        api_result.table.reset_index(drop=True), bt.table.reset_index(drop=True)
    )
