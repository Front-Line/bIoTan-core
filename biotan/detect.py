# BIoTan-core — zero-config, peer-relative anomaly backtesting (free open core).
# Copyright (c) 2026 Victor Minbeom Joo d/b/a Front-Line (https://frontli.ne.kr).
# Source-available under the PolyForm Noncommercial License 1.0.0 — noncommercial use only.
# Commercial or production use requires a separate license. See LICENSE.md (authoritative).
"""Stage 4 — multi-signal detection.

A single number can't describe every way a device goes wrong, so each device gets
several *orthogonal* signals derived from its peer-z timeline (and, for rigidity,
its own raw variability relative to peers). This stage only *computes the scores*;
deciding what counts as a flag is the next stage's effect-size gate.

Signals (per device, per metric)
--------------------------------
* ``persistent``  — robust central bias: ``median(peer_z)``. Signed; a device that
  sits consistently above/below its peers.
* ``change``      — drift: ``median(last-week peer_z) - median(first-week peer_z)``.
  Signed; uses 7-day windows, falling back to first/last third for short histories.
* ``instability`` — spread: the inter-quartile range ``P75 - P25`` of ``peer_z``.
  Non-negative; a device that swings around its peers more than it should.
* ``rigidity``    — "stuck": the device's own step-to-step variability (MAD of the
  first difference) compared, in robust z-units, to its cohort peers. Positive ==
  *less* variable than peers (flatlined / frozen sensor). Computed on the raw
  signal, not peer-z, so it captures a device that no longer moves with the fleet.

All signals are NaN when there is no valid peer baseline (undersized cohort) — we
do not invent a score where there are no peers to compare against.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.stats import median_abs_deviation

from biotan import peerz as _peerz

#: 7-day windows for the change signal.
CHANGE_WINDOW = pd.Timedelta(days=7)
#: Below this total span, the 7-day windows would overlap, so fall back to thirds.
MIN_SPAN_FOR_WINDOWS = pd.Timedelta(days=21)
#: Consistency constant (MAD -> ~std), matching peerz.ROBUST_SCALE.
ROBUST_SCALE = _peerz.ROBUST_SCALE

SIGNAL_COLUMNS = ["persistent", "change", "instability", "rigidity"]
SCORE_COLUMNS = [
    "metric",
    "device_id",
    "cohort",
    "n_obs",
    *SIGNAL_COLUMNS,
    "temporal_var",  # the device's own first-difference MAD (rigidity input)
]


@dataclass
class SignalScores:
    """Per-device multi-signal scores for a whole fleet."""

    table: pd.DataFrame                          # one row per (metric, device_id)
    notes: list[str] = field(default_factory=list)

    def for_metric(self, metric: str) -> pd.DataFrame:
        return self.table[self.table["metric"] == metric].reset_index(drop=True)


def _change_windows(ts: pd.Series, z: pd.Series) -> float:
    """median(last window) - median(first window) of peer-z."""
    if z.notna().sum() < 2:
        return np.nan
    order = ts.argsort()
    ts = ts.iloc[order].reset_index(drop=True)
    z = z.iloc[order].reset_index(drop=True)
    span = ts.iloc[-1] - ts.iloc[0]
    if span >= MIN_SPAN_FOR_WINDOWS:
        first = z[ts <= ts.iloc[0] + CHANGE_WINDOW]
        last = z[ts >= ts.iloc[-1] - CHANGE_WINDOW]
    else:
        n = len(z)
        third = max(1, n // 3)
        first = z.iloc[:third]
        last = z.iloc[-third:]
    first_med = np.nanmedian(first) if first.notna().any() else np.nan
    last_med = np.nanmedian(last) if last.notna().any() else np.nan
    return last_med - first_med


def _temporal_variability(values: pd.Series) -> float:
    """Robust step-to-step variability: MAD of the first difference (NaN-aware)."""
    diffs = values.diff().dropna().to_numpy()
    if diffs.size < 2:
        return np.nan
    return float(median_abs_deviation(diffs, scale="normal"))


def compute_signals(
    peerz_table: pd.DataFrame,
    min_peers: int = _peerz.MIN_PEERS,
) -> SignalScores:
    """Compute the four detection signals from a peer-z long table."""
    rows = []
    notes: list[str] = []

    for (metric, device), g in peerz_table.groupby(["metric", "device_id"], sort=False):
        g = g.sort_values("timestamp")
        z = g["peer_z"]
        cohort = g["cohort"].iloc[0]
        cohort = int(cohort) if pd.notna(cohort) else None
        n_obs = int(z.notna().sum())

        if n_obs == 0:
            # No valid peer baseline (undersized cohort, etc.) — honest NaNs.
            persistent = change = instability = np.nan
        else:
            persistent = float(np.nanmedian(z))
            q25, q75 = np.nanpercentile(z.dropna(), [25, 75])
            instability = float(q75 - q25)
            change = _change_windows(g["timestamp"].reset_index(drop=True), z.reset_index(drop=True))

        rows.append(
            {
                "metric": metric,
                "device_id": device,
                "cohort": cohort,
                "n_obs": n_obs,
                "persistent": persistent,
                "change": change,
                "instability": instability,
                "rigidity": np.nan,  # filled in the cohort pass below
                "temporal_var": _temporal_variability(g["value"]),
            }
        )

    scores = pd.DataFrame(rows, columns=SCORE_COLUMNS)

    # --- rigidity: compare each device's variability to its cohort, robustly ----
    for (metric, cohort), idx in scores.groupby(["metric", "cohort"], dropna=False).groups.items():
        block = scores.loc[idx]
        var = block["temporal_var"].to_numpy(dtype=float)
        valid = np.isfinite(var) & (var > 0)
        if cohort is None or valid.sum() < min_peers:
            continue  # too few peers to judge rigidity; leave NaN
        log_var = np.log(np.where(valid, var, np.nan))
        center = np.nanmedian(log_var)
        scale = ROBUST_SCALE * median_abs_deviation(log_var[np.isfinite(log_var)], scale=1.0)
        if scale == 0:
            continue  # cohort variabilities identical; no rigidity signal
        # Positive rigidity == less variable than peers (stuck).
        scores.loc[idx, "rigidity"] = -(log_var - center) / scale

    return SignalScores(table=scores, notes=notes)


def run_signals(df: pd.DataFrame, min_peers: int = _peerz.MIN_PEERS):
    """End-to-end stages 1->4 from a normalized frame.

    Returns ``(SignalScores, PeerZResult, clustering)``.
    """
    peerz_result, clustering = _peerz.run_peer_z(df, min_peers=min_peers)
    signals = compute_signals(peerz_result.table, min_peers=min_peers)
    return signals, peerz_result, clustering
