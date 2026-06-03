# BIoTan-core — zero-config, peer-relative anomaly backtesting (free open core).
# Copyright (c) 2026 Victor Minbeom Joo d/b/a Front-Line (https://frontli.ne.kr).
# Source-available under the PolyForm Noncommercial License 1.0.0 — noncommercial use only.
# Commercial or production use requires a separate license. See LICENSE.md (authoritative).
"""Stage 5 — effect-size gating.

Robust-z alone is not allowed to decide a flag. When the per-instant MAD is ~0,
peer-z explodes even though the actual deviation is negligible, so a flag must pass
*two* gates:

1. **statistical significance** — the signal's robust magnitude (in peer-z units)
   is large; and
2. **practical effect size** — the deviation, measured in the data's own units, is
   at least :data:`EFFECT_K` times a *stable* cohort reference scale.

The reference scale is set with no configuration: it is the cohort's typical robust
spread over the whole period (median of the per-instant ``peer_mad``), floored at a
small fraction of the cohort's level so an all-flat cohort can't make the effect
gate vacuous. Because that scale is stable, the MAD≈0 instants — where peer-z blows
up — cannot pass the effect gate on noise alone.

A device is flagged if *any one* of its four signals passes both gates; every
passing signal is reported as a reason. Devices without a valid peer baseline
(undersized cohort) are marked not-evaluable rather than flagged or cleared — we do
not pretend to judge a device that has no peers.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from biotan import detect as _detect
from biotan import peerz as _peerz

# --- statistical-significance thresholds (robust peer-z units) ----------------
#: |median(peer_z)| for a persistent bias. Set higher than the drift threshold on
#: purpose: a steady offset between devices is very often a benign characteristic
#: (calibration, siting, model variant), so we demand a larger, clearer gap before
#: calling it. Drift away from one's own past is far more diagnostic of a problem.
PERSISTENT_Z = 3.0
#: |Δ median(peer_z)| between first/last window for a drift.
CHANGE_Z = 2.0
#: IQR(peer_z) for instability.
INSTABILITY_Z = 3.0
#: rigidity score (robust σ below peers' log-variability) for "stuck".
RIGIDITY_Z = 3.0
#: |peer_z| for a single anomalous *point* on the timeline (used by the backtest).
POINT_Z = 3.5

# --- practical effect-size settings -------------------------------------------
#: A deviation must reach this multiple of the cohort reference scale to matter.
EFFECT_K = 1.0
#: Floor for the reference scale, as a fraction of the cohort's absolute level.
LEVEL_FLOOR_FRAC = 0.01


@dataclass
class FlagResult:
    """Per-device flag decisions for a whole fleet."""

    table: pd.DataFrame                 # one row per (metric, device_id)
    thresholds: dict = field(default_factory=dict)

    def flagged(self) -> pd.DataFrame:
        return self.table[self.table["flagged"]].reset_index(drop=True)


def _reference_scale(metric_cohort_df: pd.DataFrame) -> float:
    """Stable robust scale for a cohort, in data units (floored by level)."""
    mad = metric_cohort_df["peer_mad"]
    typical = float(np.nanmedian(mad[mad > 0])) if (mad > 0).any() else np.nan
    level = float(np.nanmedian(metric_cohort_df["value"].abs()))
    floor = LEVEL_FLOOR_FRAC * level if np.isfinite(level) else 0.0
    if not np.isfinite(typical):
        typical = 0.0
    return max(typical, floor)


def _iqr(a: np.ndarray) -> float:
    a = a[np.isfinite(a)]
    if a.size < 2:
        return np.nan
    q25, q75 = np.nanpercentile(a, [25, 75])
    return float(q75 - q25)


def apply_gate(
    signals: _detect.SignalScores,
    peerz_result: _peerz.PeerZResult,
) -> FlagResult:
    """Apply the dual (statistical AND practical) gate to the stage-4 signals."""
    peerz_table = peerz_result.table

    # Reference scale + "do peers actually move?" per (metric, cohort).
    ref_scale: dict[tuple, float] = {}
    cohort_temporal_var: dict[tuple, float] = {}
    for (metric, cohort), g in peerz_table.groupby(["metric", "cohort"], dropna=False):
        if pd.isna(cohort):
            continue
        ref_scale[(metric, int(cohort))] = _reference_scale(g)
    sig = signals.table
    for (metric, cohort), g in sig.groupby(["metric", "cohort"], dropna=False):
        if pd.isna(cohort):
            continue
        cohort_temporal_var[(metric, int(cohort))] = float(np.nanmedian(g["temporal_var"]))

    # Per-device data-unit effect sizes from the deviation timeline.
    rows = []
    for (metric, device), g in peerz_table.groupby(["metric", "device_id"], sort=False):
        g = g.sort_values("timestamp")
        dev = g["deviation"]
        ts = g["timestamp"].reset_index(drop=True)
        persistent_eff = abs(float(np.nanmedian(dev))) if dev.notna().any() else np.nan
        instability_eff = _iqr(dev.to_numpy(dtype=float))
        change_eff = _detect._change_windows(ts, dev.reset_index(drop=True))
        change_eff = abs(change_eff) if pd.notna(change_eff) else np.nan
        rows.append(
            {
                "metric": metric,
                "device_id": device,
                "persistent_eff": persistent_eff,
                "change_eff": change_eff,
                "instability_eff": instability_eff,
            }
        )
    effects = pd.DataFrame(rows)

    out = sig.merge(effects, on=["metric", "device_id"], how="left")

    records = []
    for _, r in out.iterrows():
        key = (r["metric"], int(r["cohort"])) if pd.notna(r["cohort"]) else None
        rscale = ref_scale.get(key, np.nan)
        eff_floor = EFFECT_K * rscale if np.isfinite(rscale) else np.inf
        peers_move = cohort_temporal_var.get(key, 0.0) >= LEVEL_FLOOR_FRAC * abs(
            np.nanmedian(out[out["metric"] == r["metric"]]["temporal_var"])
        )

        evaluable = bool(r["n_obs"] > 0 and key is not None and np.isfinite(rscale) and rscale > 0)

        persistent_flag = evaluable and (
            abs(r["persistent"]) >= PERSISTENT_Z and r["persistent_eff"] >= eff_floor
        )
        change_flag = evaluable and (
            pd.notna(r["change"]) and abs(r["change"]) >= CHANGE_Z and r["change_eff"] >= eff_floor
        )
        instability_flag = evaluable and (
            r["instability"] >= INSTABILITY_Z and r["instability_eff"] >= eff_floor
        )
        rigidity_flag = evaluable and (
            pd.notna(r["rigidity"]) and r["rigidity"] >= RIGIDITY_Z and peers_move
        )

        reasons = [
            name
            for name, ok in [
                ("persistent", persistent_flag),
                ("change", change_flag),
                ("instability", instability_flag),
                ("rigidity", rigidity_flag),
            ]
            if ok
        ]
        records.append(
            {
                "metric": r["metric"],
                "device_id": r["device_id"],
                "cohort": r["cohort"],
                "evaluable": evaluable,
                "flagged": bool(reasons),
                "reasons": ",".join(reasons),
                "ref_scale": rscale,
                "persistent": r["persistent"],
                "persistent_flag": persistent_flag,
                "change": r["change"],
                "change_flag": change_flag,
                "instability": r["instability"],
                "instability_flag": instability_flag,
                "rigidity": r["rigidity"],
                "rigidity_flag": rigidity_flag,
            }
        )

    table = pd.DataFrame(records).sort_values(
        ["metric", "flagged", "device_id"], ascending=[True, False, True]
    ).reset_index(drop=True)

    thresholds = {
        "PERSISTENT_Z": PERSISTENT_Z,
        "CHANGE_Z": CHANGE_Z,
        "INSTABILITY_Z": INSTABILITY_Z,
        "RIGIDITY_Z": RIGIDITY_Z,
        "EFFECT_K": EFFECT_K,
        "LEVEL_FLOOR_FRAC": LEVEL_FLOOR_FRAC,
    }
    return FlagResult(table=table, thresholds=thresholds)


def gate_timeline(peerz_result: _peerz.PeerZResult) -> pd.DataFrame:
    """Mark each peer-z point as anomalous under the dual point gate.

    Returns the peer-z table with an added boolean ``anomalous`` column:
    ``|peer_z| >= POINT_Z`` AND ``|deviation| >= EFFECT_K * cohort_ref_scale``.
    This per-point gate is what the backtest (stage 6) walks to find the first
    sustained divergence.
    """
    t = peerz_result.table.copy()
    ref = {}
    for (metric, cohort), g in t.groupby(["metric", "cohort"], dropna=False):
        if pd.isna(cohort):
            continue
        ref[(metric, int(cohort))] = _reference_scale(g)

    def _floor(row):
        key = (row["metric"], int(row["cohort"])) if pd.notna(row["cohort"]) else None
        rscale = ref.get(key, np.nan)
        return EFFECT_K * rscale if np.isfinite(rscale) else np.inf

    eff_floor = t.apply(_floor, axis=1)
    t["anomalous"] = (
        (t["peer_z"].abs() >= POINT_Z) & (t["deviation"].abs() >= eff_floor)
    ).fillna(False)
    return t


def run_gate(df: pd.DataFrame, min_peers: int = _peerz.MIN_PEERS,
             force_single_cohort: bool = False):
    """End-to-end stages 1->5 from a normalized frame.

    Returns ``(FlagResult, SignalScores, PeerZResult, clustering)``.
    """
    signals, peerz_result, clustering = _detect.run_signals(
        df, min_peers=min_peers, force_single_cohort=force_single_cohort)
    flags = apply_gate(signals, peerz_result)
    return flags, signals, peerz_result, clustering
