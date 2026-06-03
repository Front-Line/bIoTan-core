# BIoTan-core — zero-config, peer-relative anomaly backtesting (free open core).
# Copyright (c) 2026 Victor Minbeom Joo d/b/a Front-Line (https://frontli.ne.kr).
# Source-available under the PolyForm Noncommercial License 1.0.0 — noncommercial use only.
# Commercial or production use requires a separate license. See LICENSE.md (authoritative).
"""The clean, stable library entry point.

A three-line experience over the exact pipeline the CLI runs — no new statistics,
no reimplementation::

    import biotan
    result = biotan.backtest("data.csv", labels="failures.csv")  # labels optional
    result.flagged            # DataFrame: flagged devices + plain reasons + lead time
    result.summary            # dict: records / devices / metrics / flagged counts
    result.to_html("report.html")
"""

from __future__ import annotations

import pandas as pd

from biotan import normalize as _normalize
# Import the submodule's names directly (not `from biotan import backtest`), because
# the top-level `backtest()` function below shadows the `biotan.backtest` attribute.
from biotan.backtest import BacktestResult, run_backtest


class Result:
    """Outcome of :func:`backtest`, wrapping the existing pipeline objects.

    Thin accessors over what stages 1–6 already produced — same numbers as the CLI.
    """

    def __init__(self, df, summary, clustering, flags, backtest_result: BacktestResult):
        self._df = df
        self._summary = summary
        self._clustering = clustering
        self._flags = flags
        self._bt = backtest_result

    # --- raw access (for advanced use) -----------------------------------
    @property
    def table(self) -> pd.DataFrame:
        """Per-device backtest table (divergence, lead time, status)."""
        return self._bt.table

    @property
    def timeline(self) -> pd.DataFrame:
        """Per-point peer-z timeline with the anomaly flag."""
        return self._bt.timeline

    # --- the documented surface ------------------------------------------
    @property
    def flagged(self) -> pd.DataFrame:
        """Flagged devices with their plain reasons, divergence, and lead time."""
        t = self._bt.table
        cols = [c for c in ("device_id", "metric", "reasons", "divergence_start",
                            "fault_start", "lead_time_days", "status") if c in t.columns]
        return (t[t["flagged"]][cols]
                .sort_values("lead_time_days", ascending=False, na_position="last")
                .reset_index(drop=True))

    @property
    def summary(self) -> dict:
        """Headline counts: records, devices, metrics, and flagged devices."""
        ft = self._flags.table
        return {
            "records": int(len(self._df)),
            "devices": int(self._summary.n_devices),
            "metrics": int(len(self._summary.metrics)),
            "flagged": int(ft[ft["flagged"]]["device_id"].nunique()),
            "flagged_devices": sorted(ft[ft["flagged"]]["device_id"].unique().tolist()),
        }

    def to_html(self, path: str, title: str = "BIoTan backtest report") -> str:
        """Write the self-contained HTML report (same renderer as the CLI)."""
        from biotan.report import build_report  # lazy: avoids import cycle at load
        html = build_report(self._summary, self._clustering, self._flags, self._bt, title=title)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(html)
        return path

    def __repr__(self) -> str:
        s = self.summary
        return (f"<biotan.Result {s['flagged']} flagged / {s['devices']} devices, "
                f"{s['metrics']} metric(s), {s['records']} records>")


def backtest(data, labels=None, *, force_single_cohort: bool = False,
             title: str = "BIoTan backtest report") -> Result:
    """Run the full peer-relative backtest and return a :class:`Result`.

    Parameters
    ----------
    data
        Path to a CSV, or an already-loaded ``pandas.DataFrame`` in the input schema
        (``device_id, timestamp, value`` [, ``metric``, ``group``, ``unit``]).
    labels
        Optional known failures: path to a CSV or a ``DataFrame`` with at least
        ``device_id`` and ``fault_start``. When given, the result includes lead time.
    force_single_cohort
        If True, bypass auto-clustering and treat every device in a metric as one
        cohort. Use for homogeneous / non-cyclic fleets the zero-config clustering
        would over-segment. Defaults to False (auto-clustering unchanged).
    title
        Title used by :meth:`Result.to_html`.
    """
    df = _normalize.load(data) if isinstance(data, str) else _normalize.normalize_frame(data)
    lab = pd.read_csv(labels) if isinstance(labels, str) else labels
    bt_result, flags, _peerz, clustering = run_backtest(
        df, labels=lab, force_single_cohort=force_single_cohort)
    summary = _normalize.summarize(df)
    return Result(df, summary, clustering, flags, bt_result)
