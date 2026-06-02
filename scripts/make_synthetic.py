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
    seed: int = 7,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (long_df, truth_df). ``truth_df`` has columns device_id, metric, cohort."""
    rng = np.random.default_rng(seed)
    shapes = _cohort_shapes()

    start = pd.Timestamp("2026-01-01 00:00:00")
    periods = int(days * 24 * 60 / freq_minutes)
    index = pd.date_range(start, periods=periods, freq=f"{freq_minutes}min")
    hours = (index.hour + index.minute / 60.0).to_numpy(dtype=float)

    records = []
    truth = []
    for metric in metrics:
        # Fleet-wide common-mode signal: a slow weather-like multiplier shared by all.
        cm = 1.0 + common_mode * np.sin(_TWO_PI * np.arange(periods) / (periods / days * 3))
        cm += common_mode * 0.5 * rng.standard_normal(periods).cumsum() / np.sqrt(periods)

        for cohort, spec in shapes.items():
            base = spec["level"] + spec["amp"] * spec["fn"](hours)
            for j in range(n_per_cohort):
                device_id = f"{cohort}-{metric}-{j:02d}"
                # Per-device gentle offset/gain so members aren't identical.
                gain = 1.0 + 0.05 * rng.standard_normal()
                offset = 2.0 * rng.standard_normal()
                signal = base * gain * cm + offset
                signal = signal + noise * np.abs(base).mean() * rng.standard_normal(periods)
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
    return long_df, truth_df


def _validate(csv_path: str, truth_df: pd.DataFrame) -> None:
    """Run BIoTan clustering on the file and report ARI vs. ground truth per metric."""
    # Make the package importable when run as a plain script.
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from sklearn.metrics import adjusted_rand_score

    from biotan import cluster as _cluster
    from biotan import normalize as _normalize

    df = _normalize.load(csv_path)
    results = _cluster.cluster_fleet(df)

    print("\n--- validation (clustering vs. ground truth) ---")
    for metric, res in results.items():
        tdf = truth_df[truth_df["metric"] == metric]
        truth_map = dict(zip(tdf["device_id"], tdf["cohort"]))
        devices = res.device_order
        pred = [res.labels[d] for d in devices]
        gold = [truth_map[d] for d in devices]
        ari = adjusted_rand_score(gold, pred)
        print(
            f"  {metric:<12} method={res.method:<8} "
            f"cohorts={res.n_cohorts} (truth={tdf['cohort'].nunique()}) "
            f"silhouette={res.silhouette if res.silhouette is None else round(res.silhouette, 3)} "
            f"ARI={ari:.3f}"
        )
        if ari >= 0.99:
            print("               -> perfect cohort recovery.")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Generate a synthetic IoT fleet dataset.")
    p.add_argument("--out", default="synthetic.csv", help="output CSV path")
    p.add_argument("--per-cohort", type=int, default=6, help="devices per cohort")
    p.add_argument("--days", type=int, default=21, help="length of history in days")
    p.add_argument("--freq-minutes", type=int, default=60, help="sampling cadence in minutes")
    p.add_argument("--metrics", default="power_kw", help="comma-separated metric names")
    p.add_argument("--seed", type=int, default=7, help="random seed")
    p.add_argument("--validate", action="store_true", help="run clustering and report ARI")
    args = p.parse_args(argv)

    metrics = tuple(m.strip() for m in args.metrics.split(",") if m.strip())
    long_df, truth_df = generate(
        n_per_cohort=args.per_cohort,
        days=args.days,
        freq_minutes=args.freq_minutes,
        metrics=metrics,
        seed=args.seed,
    )

    long_df.to_csv(args.out, index=False)
    truth_path = os.path.splitext(args.out)[0] + ".truth.csv"
    truth_df.to_csv(truth_path, index=False)

    n_dev = long_df["device_id"].nunique()
    print(f"Wrote {len(long_df):,} rows, {n_dev} devices across {len(metrics)} metric(s) -> {args.out}")
    print(f"Wrote ground-truth cohorts -> {truth_path}")

    if args.validate:
        _validate(args.out, truth_df)
    return 0


if __name__ == "__main__":
    sys.exit(main())
