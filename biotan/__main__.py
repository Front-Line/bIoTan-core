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
from biotan import detect as _detect
from biotan import events as _events
from biotan import gate as _gate
from biotan import normalize as _normalize
from biotan import peerz as _peerz
from biotan import report as _report

_SINGLE_COHORT_HELP = (
    "treat all devices in each metric as one cohort (bypass auto-clustering); "
    "use for homogeneous / non-cyclic fleets that get over-segmented"
)


def _maybe_hint_single_cohort(warnings, enabled: bool) -> None:
    """If auto-clustering left undersized cohorts, suggest --single-cohort (stderr)."""
    if warnings and not enabled:
        print("hint: some cohorts were too small for peer comparison; re-run with "
              "--single-cohort to treat the fleet as one cohort.", file=sys.stderr)


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
    results = (
        _cluster.single_cohort(df)
        if args.single_cohort
        else _cluster.cluster_fleet(df, resample=not args.no_resample)
    )

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


def _cmd_peerz(args: argparse.Namespace) -> int:
    df = _normalize.load(args.input)
    result, clustering = _peerz.run_peer_z(df, force_single_cohort=args.single_cohort)

    for w in result.warnings:
        print(f"  warning: {w}")
    _maybe_hint_single_cohort(result.warnings, args.single_cohort)

    for metric, t in result.table.groupby("metric"):
        defined = t["peer_z"].notna()
        print(f"\n=== metric: {metric} ===")
        print(f"  cohorts        : {result.cohort_sizes.get(str(metric), {})}")
        print(f"  peer-z defined : {int(defined.sum()):,} / {len(t):,} rows")
        if defined.any():
            # Rank devices by their strongest sustained deviation (95th pct |z|).
            absz = t.assign(absz=t["peer_z"].abs())
            rank = (
                absz.dropna(subset=["absz"])
                .groupby("device_id")["absz"]
                .quantile(0.95)
                .sort_values(ascending=False)
            )
            print("  top |peer-z| (95th pct) by device:")
            for dev, v in rank.head(5).items():
                print(f"      {dev:<24} {v:.2f}")

    if args.out:
        result.table.to_csv(args.out, index=False)
        print(f"\nWrote peer-z table -> {args.out}")
    return 0


def _cmd_signals(args: argparse.Namespace) -> int:
    df = _normalize.load(args.input)
    signals, peerz_result, _ = _detect.run_signals(df, force_single_cohort=args.single_cohort)

    for w in peerz_result.warnings:
        print(f"  warning: {w}")
    _maybe_hint_single_cohort(peerz_result.warnings, args.single_cohort)

    for metric, t in signals.table.groupby("metric"):
        print(f"\n=== metric: {metric} ===")
        # Rank by the strongest single signal (each on its own scale, abs value).
        ranked = t.assign(
            _strength=t[_detect.SIGNAL_COLUMNS].abs().max(axis=1)
        ).sort_values("_strength", ascending=False)
        header = f"  {'device':<24}{'persist':>9}{'change':>9}{'instab':>9}{'rigid':>9}"
        print(header)
        for _, r in ranked.head(10).iterrows():
            def _f(x):
                return f"{x:>9.2f}" if pd.notna(x) else f"{'n/a':>9}"
            print(
                f"  {r['device_id']:<24}"
                f"{_f(r['persistent'])}{_f(r['change'])}{_f(r['instability'])}{_f(r['rigidity'])}"
            )

    if args.out:
        signals.table.to_csv(args.out, index=False)
        print(f"\nWrote signal scores -> {args.out}")
    return 0


def _cmd_flag(args: argparse.Namespace) -> int:
    df = _normalize.load(args.input)
    flags, _signals, peerz_result, _clustering = _gate.run_gate(
        df, force_single_cohort=args.single_cohort)

    for w in peerz_result.warnings:
        print(f"  warning: {w}")
    _maybe_hint_single_cohort(peerz_result.warnings, args.single_cohort)

    th = flags.thresholds
    print(
        f"\ngate: statistical (persistent>={th['PERSISTENT_Z']}, change>={th['CHANGE_Z']}, "
        f"instability>={th['INSTABILITY_Z']}, rigidity>={th['RIGIDITY_Z']} robust-sigma) "
        f"AND practical (effect >= {th['EFFECT_K']} x cohort scale)"
    )

    for metric, t in flags.table.groupby("metric"):
        n_eval = int(t["evaluable"].sum())
        flagged = t[t["flagged"]]
        print(f"\n=== metric: {metric} ===")
        print(f"  evaluable devices : {n_eval} / {len(t)}")
        print(f"  flagged           : {len(flagged)}")
        for _, r in flagged.iterrows():
            print(f"      {r['device_id']:<24} reasons: {r['reasons']}")
        if len(flagged) == 0 and n_eval > 0:
            print("      (no device passed both gates — no clear peer-relative anomaly)")

    if args.out:
        flags.table.to_csv(args.out, index=False)
        print(f"\nWrote flag decisions -> {args.out}")
    return 0


def _cmd_events(args: argparse.Namespace) -> int:
    df = _normalize.load(args.input)

    # --- build & export a reference trajectory (Mode B artifact) ---
    if args.export_reference:
        ref = _events.build_reference(df)
        _events.save_reference(ref, args.export_reference)
        n_sensors = len(ref["sensors"])
        print(f"Wrote reference trajectory -> {args.export_reference} "
              f"({ref['metadata']['n_reference_units']} units, {n_sensors} sensors, "
              f"schema {ref['schema_version']})")
        return 0

    # --- Mode B: reference-trajectory deviation (EXPERIMENTAL, opt-in) ---
    if args.mode == "reference":
        if not args.reference:
            print("error: --mode reference requires --reference <profile.json>", file=sys.stderr)
            return 2
        ref = _events.load_reference(args.reference)
        scores = _events.score_against_reference(df, ref, early_window=args.early_window)
        print("Mode B (experimental, model-based — NOT peer-relative): early-life "
              "deviation vs reference trajectory.")
        print("This is a weak early RISK RANKING, not an RUL predictor; combine sensors.")
        print(f"\n  {'device':<24}{'combined_score':>15}{'n_sensors':>11}")
        for _, r in scores.head(15).iterrows():
            print(f"  {r['device_id']:<24}{r['combined_score']:>15.3f}{int(r['n_sensors']):>11}")
        if args.out:
            scores.to_csv(args.out, index=False)
            print(f"\nWrote scores -> {args.out}")
        return 0

    # --- Mode A (default): peer-relative cohort events ---
    result = _events.detect_events(df, cohort_col=args.cohort_col, k=args.k, h=args.h,
                                   min_effect=args.min_effect, period_aggregate=args.period_agg)
    et = result.events_table()
    print(f"Cohort events (Mode A): {len(result.events)}")
    for _, e in et.iterrows():
        print(f"  [{e['t_start']:%Y-%m-%d %H:%M} .. {e['t_end']:%Y-%m-%d %H:%M}] "
              f"{e['metric']}/{e['cohort']} dir={e['direction']} "
              f"({e['n_affected']}/{e['n_members']}): {e['affected']}")
    for nt in result.notes[:8]:
        print(f"  note: {nt}")
    if args.out:
        et.to_csv(args.out, index=False)
        print(f"Wrote events -> {args.out}")
    if args.report:
        html = _report.build_event_report(df, result, cohort_col=args.cohort_col,
                                           title=args.title or "BIoTan cohort events")
        with open(args.report, "w", encoding="utf-8") as fh:
            fh.write(html)
        print(f"Wrote event report -> {args.report}")
    return 0


def _cmd_backtest(args: argparse.Namespace) -> int:
    df = _normalize.load(args.input)
    labels = pd.read_csv(args.labels) if args.labels else None

    result = _report.write_report(args.out, df, labels=labels, title=args.title,
                                  force_single_cohort=args.single_cohort)
    bt = result.table

    print(f"Wrote self-contained HTML report -> {args.out}")
    n_flagged = int(bt["flagged"].sum())
    print(f"Flagged devices: {n_flagged}")
    leads = bt["lead_time_days"].dropna()
    if not leads.empty:
        print(f"Lead times (days, optimistic upper bound): "
              f"min={leads.min():.1f} median={leads.median():.1f} max={leads.max():.1f}")
    for _, r in bt[bt["flagged"]].iterrows():
        lead = r["lead_time_days"]
        lead_txt = f" — lead {lead:.1f} d" if pd.notna(lead) else ""
        print(f"  {r['device_id']:<24} {r['status']}{lead_txt}  [{r['reasons']}]")
    return 0


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
    pc.add_argument("--single-cohort", action="store_true", help=_SINGLE_COHORT_HELP)
    pc.set_defaults(func=_cmd_cluster)

    pz = sub.add_parser("peerz", help="compute peer-relative deviation (peer-z) timelines")
    pz.add_argument("--input", required=True, help="input CSV path")
    pz.add_argument("--out", help="optional CSV path for the peer-z long table")
    pz.add_argument("--single-cohort", action="store_true", help=_SINGLE_COHORT_HELP)
    pz.set_defaults(func=_cmd_peerz)

    psig = sub.add_parser("signals", help="compute multi-signal detection scores per device")
    psig.add_argument("--input", required=True, help="input CSV path")
    psig.add_argument("--out", help="optional CSV path for the signal-score table")
    psig.add_argument("--single-cohort", action="store_true", help=_SINGLE_COHORT_HELP)
    psig.set_defaults(func=_cmd_signals)

    pf = sub.add_parser("flag", help="apply the effect-size gate and list flagged devices")
    pf.add_argument("--input", required=True, help="input CSV path")
    pf.add_argument("--out", help="optional CSV path for the flag-decision table")
    pf.add_argument("--single-cohort", action="store_true", help=_SINGLE_COHORT_HELP)
    pf.set_defaults(func=_cmd_flag)

    pe = sub.add_parser("events",
                        help="cohort event detection — when a cohort diverges internally, and who (Mode A)")
    pe.add_argument("--input", required=True, help="input CSV path")
    pe.add_argument("--cohort-col", help="column holding the fixed cohort id (e.g. group)")
    pe.add_argument("--out", help="optional CSV of detected events")
    pe.add_argument("--report", help="optional self-contained HTML event report")
    pe.add_argument("--title", help="report title")
    pe.add_argument("--k", type=float, default=_events.CUSUM_SLACK_K, help="CUSUM slack (robust-sigma)")
    pe.add_argument("--h", type=float, default=_events.CUSUM_THRESHOLD_H, help="CUSUM threshold")
    pe.add_argument("--min-effect", type=float, default=_events.EVENT_EFFECT_K, dest="min_effect",
                    help="effect-size gate (robust-sigma); raise for the conservative "
                         "'lockstep is silent' view on uniformly-degrading fleets")
    pe.add_argument("--period-agg", default="auto", choices=["auto", "daily", "none"],
                    dest="period_agg", help="period aggregation for diurnal data")
    pe.add_argument("--mode", default="peer", choices=["peer", "reference"],
                    help="peer = Mode A (default, peer-relative); reference = Mode B "
                         "(experimental, model-based, NOT peer-relative)")
    pe.add_argument("--reference", help="Mode B: reference-trajectory JSON to score against")
    pe.add_argument("--export-reference", help="build a reference-trajectory JSON from this data")
    pe.add_argument("--early-window", type=int, default=None,
                    help="Mode B: life-positions used for the early-deviation score")
    pe.set_defaults(func=_cmd_events)

    pb = sub.add_parser("backtest", help="full pipeline -> self-contained HTML report")
    pb.add_argument("--input", required=True, help="input CSV path")
    pb.add_argument("--out", required=True, help="output HTML report path")
    pb.add_argument("--labels", help="optional CSV of known failures (device_id, fault_start[, metric])")
    pb.add_argument("--title", default="BIoTan backtest report", help="report title")
    pb.add_argument("--single-cohort", action="store_true", help=_SINGLE_COHORT_HELP)
    pb.set_defaults(func=_cmd_backtest)

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
