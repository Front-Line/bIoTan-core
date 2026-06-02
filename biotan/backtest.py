# BIoTan-core — zero-config, peer-relative anomaly backtesting (free open core).
# Copyright (c) 2026 Victor Minbeom Joo d/b/a Front-Line (https://frontli.ne.kr).
# Source-available under the PolyForm Noncommercial License 1.0.0 — noncommercial use only.
# Commercial or production use requires a separate license. See LICENSE.md (authoritative).
"""Stage 6 — backtest: failure-timeline reconstruction and lead time.

This is the headline output. Walking each device's gated peer-z timeline, we find
when it *first started* drifting away from its peers in a sustained way — the
"divergence start". If the user supplies known failure / replacement dates, we turn
that into the question that actually matters:

    "How many days earlier would this tool have alerted?"

Honesty (built in, not optional)
--------------------------------
* The reported lead time is an **optimistic upper bound**. Backtesting looks at
  data that already happened with thresholds chosen with hindsight; in real time
  the threshold must be fixed in advance, so live lead time can be shorter. Every
  report says so.
* When a labeled failure has no sustained precursor in the data, we say
  "no clear precursor" — this is a health signal in itself (or this failure mode
  needs richer signals / labels). We never backfill an empty result with a guess.
* Without labels we only state when the anomaly *started*, not that a failure
  followed.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from biotan import gate as _gate

#: A divergence is "sustained" once at least this fraction of points in the
#: trailing window are anomalous.
PERSISTENCE_FRAC = 0.5
#: Trailing window length used to judge persistence (in time).
PERSISTENCE_WINDOW = pd.Timedelta(days=1)
#: Minimum number of points in the persistence window (covers coarse cadences).
MIN_WINDOW_POINTS = 3

LEAD_TIME_DISCLAIMER = (
    "Lead time is an optimistic upper bound: it is computed by hindsight on data "
    "that already happened. In live operation the alert threshold must be fixed in "
    "advance, so real-world lead time may be shorter."
)


@dataclass
class BacktestResult:
    """Per-device divergence / lead-time outcomes for a whole fleet."""

    table: pd.DataFrame                 # one row per (metric, device_id)
    timeline: pd.DataFrame             # peer-z table with the per-point `anomalous` flag
    disclaimer: str = LEAD_TIME_DISCLAIMER
    notes: list[str] = field(default_factory=list)


def _window_points(ts: pd.Series) -> int:
    """How many samples cover :data:`PERSISTENCE_WINDOW`, given the cadence."""
    if len(ts) < 2:
        return MIN_WINDOW_POINTS
    step = ts.sort_values().diff().dropna().median()
    if step <= pd.Timedelta(0):
        return MIN_WINDOW_POINTS
    return max(MIN_WINDOW_POINTS, int(round(PERSISTENCE_WINDOW / step)))


def find_divergence_start(dev_timeline: pd.DataFrame) -> pd.Timestamp | None:
    """First timestamp of the earliest sustained run of anomalous points."""
    d = dev_timeline.sort_values("timestamp").reset_index(drop=True)
    anomalous = d["anomalous"].fillna(False).to_numpy()
    if not anomalous.any():
        return None
    window = _window_points(d["timestamp"])
    rate = pd.Series(anomalous, dtype=float).rolling(window, min_periods=window).mean()
    qualifying = np.where(rate.to_numpy() >= PERSISTENCE_FRAC)[0]
    if qualifying.size == 0:
        return None
    i = int(qualifying[0])
    win_start = max(0, i - window + 1)
    # Onset = first anomalous point inside the window that first qualified.
    local = anomalous[win_start : i + 1]
    offset = int(np.argmax(local)) if local.any() else 0
    return d["timestamp"].iloc[win_start + offset]


def _normalize_labels(labels: pd.DataFrame | None) -> pd.DataFrame:
    """Coerce a labels frame to columns device_id, metric (optional), fault_start."""
    if labels is None or len(labels) == 0:
        return pd.DataFrame(columns=["device_id", "metric", "fault_start"])
    lab = labels.copy()
    lab.columns = [str(c).strip().lower() for c in lab.columns]
    rename = {"device": "device_id", "fault": "fault_start", "failure": "fault_start",
              "fault_time": "fault_start", "failure_date": "fault_start", "date": "fault_start"}
    lab = lab.rename(columns=rename)
    if "device_id" not in lab or "fault_start" not in lab:
        raise ValueError("labels need at least 'device_id' and 'fault_start' columns.")
    lab["device_id"] = lab["device_id"].astype(str).str.strip()
    lab["fault_start"] = pd.to_datetime(lab["fault_start"], errors="coerce")
    if "metric" not in lab:
        lab["metric"] = None
    return lab[["device_id", "metric", "fault_start"]]


def reconstruct_timelines(
    flags: _gate.FlagResult,
    peerz_result: _gate.FlagResult,  # PeerZResult, kept loose to avoid import cycle
    labels: pd.DataFrame | None = None,
) -> BacktestResult:
    """Reconstruct divergence timelines and (with labels) compute lead times."""
    timeline = _gate.gate_timeline(peerz_result)
    lab = _normalize_labels(labels)
    flag_by = flags.table.set_index(["metric", "device_id"])

    rows = []
    for (metric, device), g in timeline.groupby(["metric", "device_id"], sort=False):
        key = (metric, device)
        frow = flag_by.loc[key] if key in flag_by.index else None
        evaluable = bool(frow["evaluable"]) if frow is not None else False
        flagged = bool(frow["flagged"]) if frow is not None else False
        reasons = frow["reasons"] if frow is not None else ""

        divergence = find_divergence_start(g) if evaluable else None

        # Match a label (device + metric if the label specifies one).
        match = lab[lab["device_id"] == device]
        if "metric" in match and match["metric"].notna().any():
            match = match[(match["metric"].isna()) | (match["metric"] == metric)]
        fault_start = match["fault_start"].iloc[0] if len(match) and pd.notna(match["fault_start"].iloc[0]) else None

        lead_days = np.nan
        if not evaluable:
            status = "not evaluable (no peer baseline)"
        elif fault_start is not None:
            if divergence is not None and divergence < fault_start:
                lead_days = (fault_start - divergence) / pd.Timedelta(days=1)
                status = f"precursor {lead_days:.1f} d before failure"
            elif divergence is not None:
                status = "detected only at/after failure"
            else:
                status = "no clear precursor"
        else:
            if divergence is not None:
                status = "anomaly onset (no failure label)"
            elif flagged:
                status = "flagged; no distinct onset point"
            else:
                status = "no anomaly"

        rows.append(
            {
                "metric": metric,
                "device_id": device,
                "evaluable": evaluable,
                "flagged": flagged,
                "reasons": reasons,
                "divergence_start": divergence,
                "fault_start": fault_start,
                "lead_time_days": lead_days,
                "status": status,
            }
        )

    table = pd.DataFrame(rows).sort_values(
        ["metric", "flagged", "lead_time_days"], ascending=[True, False, False]
    ).reset_index(drop=True)
    return BacktestResult(table=table, timeline=timeline)


def run_backtest(df: pd.DataFrame, labels: pd.DataFrame | None = None, min_peers: int | None = None):
    """End-to-end stages 1->6 from a normalized frame.

    Returns ``(BacktestResult, FlagResult, PeerZResult, clustering)``.
    """
    kwargs = {} if min_peers is None else {"min_peers": min_peers}
    flags, _signals, peerz_result, clustering = _gate.run_gate(df, **kwargs)
    result = reconstruct_timelines(flags, peerz_result, labels=labels)
    return result, flags, peerz_result, clustering
