# BIoTan-core — zero-config, peer-relative anomaly backtesting (free open core).
# Copyright (c) 2026 Victor Minbeom Joo d/b/a Front-Line (https://frontli.ne.kr).
# Source-available under the PolyForm Noncommercial License 1.0.0 — noncommercial use only.
# Commercial or production use requires a separate license. See LICENSE.md (authoritative).
"""Generate a synthetic homogeneous-fleet dataset to exercise stages 1 & 2.

The fleet is built from a few *known* behavioral cohorts with distinct
hour-of-day shapes and levels, plus a fleet-wide common-mode signal (e.g. weather)
that affects everyone — exactly the structure peer-relative detection is designed
for. The ground-truth cohort of each device is written alongside the data so the
clustering can be scored.

Usage
-----
    python scripts/make_synthetic.py --out synthetic.csv
    python scripts/make_synthetic.py --out synthetic.csv --validate

``--validate`` runs the actual BIoTan clustering on the generated file and reports
the Adjusted Rand Index against ground truth (1.0 == perfect recovery).
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

# Cohort "personalities": (label, level, daily-amplitude, shape). Shapes are
# functions of hour-of-day in [0, 24).
_TWO_PI = 2 * np.pi


def _cohort_shapes() -> dict[str, dict]:
    return {
        # Daytime peaker (solar-inverter-like): peaks near noon.
        "daytime": {
            "level": 50.0,
            "amp": 30.0,
            "fn": lambda h: np.clip(np.sin((h - 6) / 12 * np.pi), 0, None),
        },
        # Evening peaker (residential-load-like): peaks ~19:00.
        "evening": {
            "level": 40.0,
            "amp": 25.0,
            "fn": lambda h: np.exp(-((h - 19) ** 2) / (2 * 3.0**2)),
        },
        # Baseload (industrial): roughly flat with a mild twice-daily ripple.
        "baseload": {
            "level": 70.0,
            "amp": 8.0,
            "fn": lambda h: 0.5 + 0.5 * np.cos(_TWO_PI * h / 12),
        },
    }


def generate(
    n_per_cohort: int = 6,
    days: int = 21,
    freq_minutes: int = 60,
    metrics: tuple[str, ...] = ("power_kw",),
    common_mode: float = 0.25,
    noise: float = 0.05,
    n_faults: int = 0,
    drift_onset_frac: float = 0.5,
    failure_frac: float = 0.85,
    fault_magnitude: float = 0.4,
    seed: int = 7,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Generate a synthetic fleet.

    Returns ``(long_df, truth_df, faults_df)``. ``truth_df`` has columns
    ``device_id, metric, cohort``; ``faults_df`` has ``device_id, metric,
    drift_onset, fault_start`` (empty if ``n_faults == 0``).

    A faulty device begins a gradual downward drift at ``drift_onset_frac`` of the
    history and is "failed/replaced" later, at ``failure_frac`` — that later date is
    the *label* (``fault_start``) handed to the backtest. The gap between the drift
    onset (which the engine should detect) and the failure label is exactly the
    lead time the backtest reconstructs.
    """
    rng = np.random.default_rng(seed)
    shapes = _cohort_shapes()

    start = pd.Timestamp("2026-01-01 00:00:00")
    periods = int(days * 24 * 60 / freq_minutes)
    index = pd.date_range(start, periods=periods, freq=f"{freq_minutes}min")
    hours = (index.hour + index.minute / 60.0).to_numpy(dtype=float)
    onset_idx = int(periods * drift_onset_frac)
    failure_idx = min(periods - 1, int(periods * failure_frac))

    records = []
    truth = []
    faults = []
    for metric in metrics:
        # Fleet-wide common-mode signal: a slow weather-like multiplier shared by all.
        cm = 1.0 + common_mode * np.sin(_TWO_PI * np.arange(periods) / (periods / days * 3))
        cm += common_mode * 0.5 * rng.standard_normal(periods).cumsum() / np.sqrt(periods)

        # Choose which devices fault for this metric.
        all_devices = [
            f"{cohort}-{metric}-{j:02d}"
            for cohort in shapes
            for j in range(n_per_cohort)
        ]
        faulty = set(rng.choice(all_devices, size=min(n_faults, len(all_devices)), replace=False)) if n_faults else set()

        for cohort, spec in shapes.items():
            base = spec["level"] + spec["amp"] * spec["fn"](hours)
            for j in range(n_per_cohort):
                device_id = f"{cohort}-{metric}-{j:02d}"
                # Per-device gentle offset/gain so members aren't identical.
                gain = 1.0 + 0.05 * rng.standard_normal()
                offset = 2.0 * rng.standard_normal()
                signal = base * gain * cm + offset
                signal = signal + noise * np.abs(base).mean() * rng.standard_normal(periods)

                if device_id in faulty:
                    ramp = np.zeros(periods)
                    ramp[onset_idx:] = np.linspace(0, fault_magnitude, periods - onset_idx)
                    signal = signal * (1.0 - ramp)
                    faults.append(
                        {
                            "device_id": device_id,
                            "metric": metric,
                            "drift_onset": index[onset_idx],
                            "fault_start": index[failure_idx],  # the failure/replacement label
                        }
                    )

                records.append(
                    pd.DataFrame(
                        {
                            "device_id": device_id,
                            "timestamp": index,
                            "value": signal,
                            "metric": metric,
                        }
                    )
                )
                truth.append({"device_id": device_id, "metric": metric, "cohort": cohort})

    long_df = pd.concat(records, ignore_index=True)
    truth_df = pd.DataFrame(truth)
    faults_df = pd.DataFrame(faults, columns=["device_id", "metric", "drift_onset", "fault_start"])
    return long_df, truth_df, faults_df


def _validate(csv_path: str, truth_df: pd.DataFrame, faults_df: pd.DataFrame) -> None:
    """Run BIoTan stages 1-3 and report clustering recovery + peer-z on faults."""
    # Make the package importable when run as a plain script.
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from sklearn.metrics import adjusted_rand_score

    from biotan import backtest as _backtest
    from biotan import normalize as _normalize

    df = _normalize.load(csv_path)
    bt, flags, peerz_result, clustering = _backtest.run_backtest(df, labels=faults_df)
    from biotan import detect as _detect
    signals = _detect.compute_signals(peerz_result.table)

    print("\n--- validation: clustering vs. ground truth ---")
    for metric, res in clustering.items():
        tdf = truth_df[truth_df["metric"] == metric]
        truth_map = dict(zip(tdf["device_id"], tdf["cohort"]))
        devices = res.device_order
        pred = [res.labels[d] for d in devices]
        gold = [truth_map[d] for d in devices]
        ari = adjusted_rand_score(gold, pred)
        tag = "  -> perfect cohort recovery." if ari >= 0.99 else ""
        print(
            f"  {metric:<12} method={res.method:<8} "
            f"cohorts={res.n_cohorts} (truth={tdf['cohort'].nunique()}) ARI={ari:.3f}{tag}"
        )

    print("\n--- validation: peer-z (common-mode removal) ---")
    t = peerz_result.table
    healthy_devices = set(t["device_id"].unique()) - set(faults_df["device_id"])
    healthy = t[t["device_id"].isin(healthy_devices)]
    print(
        f"  healthy devices median |peer-z| = "
        f"{healthy['peer_z'].abs().median():.3f}  (common mode should make this small)"
    )
    if not faults_df.empty:
        for _, row in faults_df.iterrows():
            series = t[(t["device_id"] == row["device_id"]) & (t["metric"] == row["metric"])]
            tail = series[series["timestamp"] >= row["drift_onset"]]["peer_z"].abs()
            peak = tail.max() if not tail.empty else float("nan")
            print(
                f"  FAULT {row['device_id']:<22} peak |peer-z| after onset = {peak:.2f} "
                f"(drift onset {row['drift_onset']})"
            )

    print("\n--- validation: multi-signal scores (stage 4) ---")
    sig = signals.table
    faulty_ids = set(faults_df["device_id"])
    healthy_change = sig[~sig["device_id"].isin(faulty_ids)]["change"].abs().median()
    print(f"  healthy devices median |change| = {healthy_change:.3f}")
    for dev in sorted(faulty_ids):
        r = sig[sig["device_id"] == dev].iloc[0]
        print(
            f"  FAULT {dev:<22} persistent={r['persistent']:+.2f} change={r['change']:+.2f} "
            f"instability={r['instability']:.2f} rigidity={r['rigidity']:+.2f}"
        )

    print("\n--- validation: effect-size gate / flags (stage 5) ---")
    ft = flags.table
    flagged_ids = set(ft[ft["flagged"]]["device_id"])
    caught = faulty_ids & flagged_ids
    false_pos = flagged_ids - faulty_ids
    if faulty_ids:
        print(f"  faults caught     : {len(caught)}/{len(faulty_ids)}")
    print(f"  false positives   : {len(false_pos)} {sorted(false_pos) if false_pos else ''}")
    for dev in sorted(flagged_ids):
        reasons = ft[ft["device_id"] == dev]["reasons"].iloc[0]
        tag = "FAULT " if dev in faulty_ids else "      "
        print(f"  {tag}FLAG {dev:<22} reasons: {reasons}")

    if not faults_df.empty:
        print("\n--- validation: backtest lead time (stage 6) ---")
        print(f"  ({_backtest.LEAD_TIME_DISCLAIMER})")
        onset_map = dict(zip(faults_df["device_id"], faults_df["drift_onset"]))
        for _, row in bt.table[bt.table["device_id"].isin(faulty_ids)].iterrows():
            true_onset = onset_map.get(row["device_id"])
            print(
                f"  FAULT {row['device_id']:<22} status: {row['status']}"
                + (f"  (lead {row['lead_time_days']:.1f} d; true onset {true_onset})"
                   if pd.notna(row["lead_time_days"]) else "")
            )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Generate a synthetic IoT fleet dataset.")
    p.add_argument("--out", default="synthetic.csv", help="output CSV path")
    p.add_argument("--per-cohort", type=int, default=6, help="devices per cohort")
    p.add_argument("--days", type=int, default=21, help="length of history in days")
    p.add_argument("--freq-minutes", type=int, default=60, help="sampling cadence in minutes")
    p.add_argument("--metrics", default="power_kw", help="comma-separated metric names")
    p.add_argument("--faults", type=int, default=0, help="number of devices to fault (per metric)")
    p.add_argument("--fault-magnitude", type=float, default=0.4, help="end drift as fraction of level")
    p.add_argument("--seed", type=int, default=7, help="random seed")
    p.add_argument("--validate", action="store_true", help="run the full pipeline and report metrics")
    args = p.parse_args(argv)

    metrics = tuple(m.strip() for m in args.metrics.split(",") if m.strip())
    long_df, truth_df, faults_df = generate(
        n_per_cohort=args.per_cohort,
        days=args.days,
        freq_minutes=args.freq_minutes,
        metrics=metrics,
        n_faults=args.faults,
        fault_magnitude=args.fault_magnitude,
        seed=args.seed,
    )

    long_df.to_csv(args.out, index=False)
    truth_path = os.path.splitext(args.out)[0] + ".truth.csv"
    truth_df.to_csv(truth_path, index=False)

    n_dev = long_df["device_id"].nunique()
    print(f"Wrote {len(long_df):,} rows, {n_dev} devices across {len(metrics)} metric(s) -> {args.out}")
    print(f"Wrote ground-truth cohorts -> {truth_path}")
    if not faults_df.empty:
        faults_path = os.path.splitext(args.out)[0] + ".faults.csv"
        faults_df.to_csv(faults_path, index=False)
        print(f"Wrote injected faults -> {faults_path}")

    if args.validate:
        _validate(args.out, truth_df, faults_df)
    return 0


if __name__ == "__main__":
    sys.exit(main())
