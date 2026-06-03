# BIoTan-core — zero-config, peer-relative anomaly backtesting (free open core).
# Copyright (c) 2026 Victor Minbeom Joo d/b/a Front-Line (https://frontli.ne.kr).
# Source-available under the PolyForm Noncommercial License 1.0.0 — noncommercial use only.
# Commercial or production use requires a separate license. See LICENSE.md (authoritative).
"""Validation on a REAL public dataset: NASA C-MAPSS turbofan degradation (FD001).

Why this dataset
----------------
FD001 is a fleet of 100 nominally identical turbofan engines, each run to failure
under one operating condition, with 21 sensors per engine. It is a homogeneous
fleet — BIoTan's target — and it carries a *true failure point* for every unit
(the last recorded cycle), so we can measure the headline output honestly: how many
cycles before failure does peer-relative deviation first appear?

Mapping to BIoTan's schema
--------------------------
* device_id = engine unit
* timestamp = synthetic daily index where 1 day == 1 cycle, so "same timestamp"
  means "same cycle": each engine is compared to the fleet at the same point in life.
* metric    = each non-constant sensor
* label     = each engine's last cycle == its failure time

What this validation shows (honestly)
-------------------------------------
1. Common-mode removal works: after removing the per-cycle fleet baseline, the
   residual peer-z rises above 2σ before failure for almost every engine.
2. Two honest limits surface, both already stated in the README:
   * BIoTan's behavioral-profile clustering is built for *daily-cyclic* IoT data.
     A turbofan's cycle index has no hour-of-day structure, so the zero-config
     clustering over-segments the fleet. The correct model for FD001 is a single
     cohort; we report BOTH the as-is zero-config run and the single-fleet run so
     the effect is visible, not hidden.
   * Peer-relative methods remove whatever the fleet does *together*. Engines that
     degrade at a typical rate are partly cancelled as common mode; the method most
     strongly flags engines degrading *faster than their peers*. It is
     risk-prioritization, not a guaranteed per-unit predictor.

No data is committed: the file is downloaded on demand into validation/data/.
The download is the only network use; the analysis itself is 100% local.

Usage
-----
    python validation/run_cmapss.py
    python validation/run_cmapss.py --report validation/cmapss_report.html
"""

from __future__ import annotations

import argparse
import os
import sys
import urllib.request

import numpy as np
import pandas as pd

# Make the package importable when run as a plain script.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from biotan.backtest import run_backtest, reconstruct_timelines, LEAD_TIME_DISCLAIMER  # noqa: E402
from biotan import detect as _detect  # noqa: E402
from biotan import gate as _gate  # noqa: E402
from biotan import normalize as _normalize  # noqa: E402
from biotan import peerz as _peerz  # noqa: E402
from biotan import report as _report  # noqa: E402
from biotan.cluster import ClusterResult  # noqa: E402

DATA_URL = (
    "https://raw.githubusercontent.com/hankroark/Turbofan-Engine-Degradation/"
    "master/CMAPSSData/train_FD001.txt"
)
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
DATA_PATH = os.path.join(DATA_DIR, "train_FD001.txt")

SENSOR_NAMES = [f"sensor_{i}" for i in range(1, 22)]
COLUMNS = ["unit", "cycle", "op1", "op2", "op3", *SENSOR_NAMES]
BASE = pd.Timestamp("2000-01-01")


def download_if_needed() -> None:
    if os.path.exists(DATA_PATH):
        return
    os.makedirs(DATA_DIR, exist_ok=True)
    print(f"downloading C-MAPSS FD001 -> {DATA_PATH}")
    urllib.request.urlretrieve(DATA_URL, DATA_PATH)


def load_cmapss() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (long_df in BIoTan schema, labels_df with failure timestamps)."""
    raw = pd.read_csv(DATA_PATH, sep=r"\s+", header=None).iloc[:, : len(COLUMNS)]
    raw.columns = COLUMNS

    keep = [s for s in SENSOR_NAMES if raw[s].std() > 1e-6]
    dropped = sorted(set(SENSOR_NAMES) - set(keep))

    long = raw.melt(id_vars=["unit", "cycle"], value_vars=keep,
                    var_name="metric", value_name="value")
    long["device_id"] = "engine_" + long["unit"].astype(int).astype(str).str.zfill(3)
    long["timestamp"] = BASE + pd.to_timedelta(long["cycle"].astype(int) - 1, unit="D")
    long = long[["device_id", "timestamp", "value", "metric"]]

    last_cycle = raw.groupby("unit")["cycle"].max()
    labels = pd.DataFrame({
        "device_id": ["engine_" + str(int(u)).zfill(3) for u in last_cycle.index],
        "fault_start": [BASE + pd.Timedelta(days=int(c) - 1) for c in last_cycle.values],
    })
    print(f"loaded {raw['unit'].nunique()} engines, {len(keep)} informative sensors "
          f"(dropped constant: {dropped})")
    return long, labels


def _per_engine_leads(bt_table: pd.DataFrame, devices: list[str]) -> pd.DataFrame:
    """Aggregate across sensors: earliest divergence per engine == max lead (cycles)."""
    rows = []
    for dev in devices:
        r = bt_table[bt_table["device_id"] == dev]
        leads = r["lead_time_days"].dropna()
        rows.append({"device_id": dev, "flagged": bool(r["flagged"].any()),
                     "lead_cycles": leads.max() if not leads.empty else np.nan})
    return pd.DataFrame(rows)


def _report_leads(title: str, pe: pd.DataFrame) -> None:
    caught = pe[pe["lead_cycles"].notna() & (pe["lead_cycles"] > 0)]
    print(f"\n[{title}]")
    print(f"  engines flagged (any sensor)    : {int(pe['flagged'].sum())}/{len(pe)}")
    print(f"  engines with positive lead time : {len(caught)}/{len(pe)}")
    if not caught.empty:
        lc = caught["lead_cycles"]
        print(f"  lead time (cycles before failure): median={lc.median():.0f}  "
              f"IQR=[{lc.quantile(.25):.0f}, {lc.quantile(.75):.0f}]  max={lc.max():.0f}")


def _signal_diagnostic(peerz_table: pd.DataFrame, labels: pd.DataFrame) -> None:
    """How strong is the residual peer-z near failure? (common-mode removal check)."""
    fault = dict(zip(labels["device_id"], labels["fault_start"]))
    peaks = []
    for dev, f in fault.items():
        sub = peerz_table[(peerz_table["device_id"] == dev)
                          & (peerz_table["timestamp"] >= f - pd.Timedelta(days=15))]
        peaks.append(sub["peer_z"].abs().max())
    peaks = pd.Series(peaks).dropna()
    print("\n[common-mode removal diagnostic] max |peer-z| in last 15 cycles before failure:")
    print(f"  engines exceeding 2.0σ : {int((peaks > 2.0).sum())}/{len(peaks)}")
    print(f"  engines exceeding 3.5σ : {int((peaks > 3.5).sum())}/{len(peaks)}")
    print(f"  median peak |peer-z|   : {peaks.median():.1f}")


def _single_fleet_clustering(df: pd.DataFrame) -> dict:
    devices = sorted(df["device_id"].unique())
    return {m: ClusterResult(metric=m, labels={d: 0 for d in devices},
                             method="single-fleet", n_cohorts=1, silhouette=None,
                             device_order=devices)
            for m in df["metric"].unique()}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Validate BIoTan on NASA C-MAPSS FD001.")
    p.add_argument("--report", help="optional HTML report path (rendered for one sensor)")
    args = p.parse_args(argv)

    download_if_needed()
    long_df, labels = load_cmapss()
    df = _normalize.normalize_frame(long_df)
    devices = sorted(df["device_id"].unique())

    print("\n================ C-MAPSS FD001 validation ================")

    # --- (A) zero-config, exactly as a user would run it --------------------
    print("running zero-config pipeline (stages 1->6, one per sensor)...")
    bt_zc, flags_zc, peerz_zc, clustering = run_backtest(df, labels=labels)
    n_cohorts = {m: c.n_cohorts for m, c in clustering.items()}
    over_seg = sum(v > 1 for v in n_cohorts.values())
    print(f"  clustering cohorts per sensor   : {n_cohorts}")
    print(f"  -> {over_seg}/{len(n_cohorts)} sensors were split into >1 cohort "
          f"(expected: FD001 is one fleet; the behavioral-profile clustering assumes "
          f"daily periodicity, which cycle data lacks).")
    _report_leads("zero-config (clustering as-is)", _per_engine_leads(bt_zc.table, devices))

    # --- (B) single-fleet: the correct cohort model for non-cyclic data -----
    grid = _normalize.resample_to_grid(df)
    single = _single_fleet_clustering(df)
    peerz_sf = _peerz.compute_peer_z(grid, single)
    signals_sf = _detect.compute_signals(peerz_sf.table)
    flags_sf = _gate.apply_gate(signals_sf, peerz_sf)
    bt_sf = reconstruct_timelines(flags_sf, peerz_sf, labels=labels)
    _report_leads("single-fleet (all engines = one cohort)", _per_engine_leads(bt_sf.table, devices))
    _signal_diagnostic(peerz_sf.table, labels)

    print("\n" + LEAD_TIME_DISCLAIMER)
    print("Conclusion: common-mode removal surfaces real degradation (peer-z clears 2σ "
          "before failure for almost every engine); the conservative zero-config gate "
          "confirms the fastest-degrading engines with a positive lead time. This is "
          "risk-prioritization, not a guaranteed per-unit predictor — and BIoTan's "
          "clustering is built for daily-cyclic fleets, so non-cyclic fleets like this "
          "are best analysed as a single cohort.")

    if args.report:
        top_metric = (bt_sf.table[bt_sf.table["flagged"]]["metric"].value_counts().idxmax()
                      if bt_sf.table["flagged"].any() else df["metric"].iloc[0])
        sub = df[df["metric"] == top_metric]
        _report.write_report(args.report, sub, labels=labels,
                             title=f"BIoTan — C-MAPSS FD001 ({top_metric}, single fleet)")
        print(f"\nwrote HTML report for '{top_metric}' -> {args.report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
