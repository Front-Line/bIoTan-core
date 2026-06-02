# BIoTan-core — zero-config, peer-relative anomaly backtesting (free open core).
# Copyright (c) 2026 Victor Minbeom Joo d/b/a Front-Line (https://frontli.ne.kr).
# Source-available under the PolyForm Noncommercial License 1.0.0 — noncommercial use only.
# Commercial or production use requires a separate license. See LICENSE.md (authoritative).
"""Command-line entry point: ``python -m biotan``.

Subcommands implemented at this stage:
  * ``summarize`` — parse a CSV and print a normalized-fleet summary.
  * ``cluster``   — discover cohorts and print/save them.

``backtest`` (the full peer-z + detection + HTML report pipeline) is intentionally
not wired up yet; it is built in later stages.
"""

from __future__ import annotations

import argparse
import sys

import pandas as pd

from biotan import cluster as _cluster
from biotan import normalize as _normalize


def _cmd_summarize(args: argparse.Namespace) -> int:
    df = _normalize.load(args.input)
    s = _normalize.summarize(df)
    print(f"Input          : {args.input}")
    print(f"Rows (clean)   : {s.n_rows:,}")
    print(f"Devices        : {s.n_devices}")
    print(f"Metrics        : {', '.join(s.metrics)}")
    print(f"Time span      : {s.span_start}  ->  {s.span_end}")
    if s.periods:
        print("Inferred period:")
        for m, p in s.periods.items():
            print(f"    {m:<20} {p}")
    return 0


def _cmd_cluster(args: argparse.Namespace) -> int:
    df = _normalize.load(args.input)
    results = _cluster.cluster_fleet(df, resample=not args.no_resample)

    rows = []
    for metric, res in results.items():
        print(f"\n=== metric: {metric} ===")
        print(f"  devices    : {len(res.device_order)}")
        print(f"  method     : {res.method}")
        print(f"  cohorts    : {res.n_cohorts}")
        if res.silhouette is not None:
            print(f"  silhouette : {res.silhouette:.3f}")
        if res.k_candidates:
            ks = ", ".join(f"k={k}:{v:.2f}" for k, v in sorted(res.k_candidates.items()))
            print(f"  kmeans scan: {ks}")
        print(f"  sizes      : {res.sizes()}")
        if res.notes:
            print(f"  note       : {res.notes}")
        for dev, c in sorted(res.labels.items()):
            rows.append({"device_id": dev, "metric": metric, "cohort": c})

    if args.out:
        pd.DataFrame(rows).to_csv(args.out, index=False)
        print(f"\nWrote cohort assignments -> {args.out}")
    return 0


def _cmd_backtest(args: argparse.Namespace) -> int:
    print(
        "backtest is not implemented yet at this stage.\n"
        "Implemented so far: `summarize` (stage 1) and `cluster` (stage 2).\n"
        "Common-mode removal, multi-signal detection, gating, and the HTML report "
        "come in later stages."
    )
    return 1


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m biotan",
        description="BIoTan-core: zero-config, peer-relative IoT anomaly backtesting (batch only).",
    )
    sub = p.add_subparsers(dest="command", required=True)

    ps = sub.add_parser("summarize", help="parse a CSV and print a normalized-fleet summary")
    ps.add_argument("--input", required=True, help="input CSV path")
    ps.set_defaults(func=_cmd_summarize)

    pc = sub.add_parser("cluster", help="discover behavioral cohorts (auto, zero-config)")
    pc.add_argument("--input", required=True, help="input CSV path")
    pc.add_argument("--out", help="optional CSV path for device->cohort assignments")
    pc.add_argument(
        "--no-resample",
        action="store_true",
        help="skip resampling to a regular grid before clustering",
    )
    pc.set_defaults(func=_cmd_cluster)

    pb = sub.add_parser("backtest", help="(later stage) full peer-z pipeline + HTML report")
    pb.add_argument("--input", help="input CSV path")
    pb.add_argument("--out", help="output HTML report path")
    pb.set_defaults(func=_cmd_backtest)

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
