# BIoTan-core — zero-config, peer-relative anomaly backtesting (free open core).
# Copyright (c) 2026 Victor Minbeom Joo d/b/a Front-Line (https://frontli.ne.kr).
# Source-available under the PolyForm Noncommercial License 1.0.0 — noncommercial use only.
# Commercial or production use requires a separate license. See LICENSE.md (authoritative).
"""Stage 3 — common-mode removal (peer-z).

The core idea: judge each device not against an absolute limit, but against what
its cohort peers are doing *at the same instant*. Shared conditions (weather, load,
seasonality) move every peer together and cancel out; what remains is the part that
is genuinely specific to the device.

For each metric, cohort, and timestamp we compute a robust baseline over the cohort
members present at that instant:

    peer_median = median(values)
    peer_mad    = 1.4826 * median(|value - peer_median|)        # robust ~= std
    peer_z      = (value - peer_median) / peer_mad

Robust (median / MAD) statistics are used on purpose: a handful of already-failing
peers must not poison the baseline. The baseline includes the device itself
(whole-cohort baseline), which is safe because the median is barely moved by a
single drifting member in a cohort of several.

Honesty rules
-------------
* A cohort needs at least :data:`MIN_PEERS` members to be a valid peer group.
  Smaller cohorts are reported as a warning and their devices get NO peer-z
  (independent cohort) rather than a fabricated one.
* ``peer_z`` is left undefined (NaN) wherever the baseline is degenerate
  (``peer_mad == 0``) or too few peers are present at that instant. The raw
  ``deviation`` (value - peer_median) is always kept, because the next stage's
  effect-size gate needs it precisely for the MAD≈0 case where z blows up.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from biotan import cluster as _cluster
from biotan import normalize as _normalize

#: Consistency constant so that, for normal data, MAD-based scale ~= standard deviation.
ROBUST_SCALE = 1.4826

#: A cohort needs at least this many members (and this many present at an instant)
#: for its peer baseline to be considered meaningful.
MIN_PEERS = 3

#: Columns of the peer-z long table.
PEERZ_COLUMNS = [
    "metric",
    "device_id",
    "timestamp",
    "value",
    "cohort",
    "peer_median",
    "peer_mad",
    "deviation",
    "peer_z",
    "n_peers",
]


@dataclass
class PeerZResult:
    """Peer-z output for a whole fleet (all metrics)."""

    table: pd.DataFrame                 # long format, PEERZ_COLUMNS
    cohort_sizes: dict[str, dict[int, int]] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def for_device(self, device_id: str, metric: str | None = None) -> pd.DataFrame:
        """Return the peer-z time series for one device (optionally one metric)."""
        t = self.table[self.table["device_id"] == device_id]
        if metric is not None:
            t = t[t["metric"] == metric]
        return t.sort_values(["metric", "timestamp"]).reset_index(drop=True)


def _peer_z_for_metric(
    grid_metric: pd.DataFrame,
    labels: dict[str, int],
    min_peers: int,
) -> tuple[pd.DataFrame, dict[int, int], list[str]]:
    """Compute peer-z for a single metric on its regular grid."""
    df = grid_metric.copy()
    df["cohort"] = df["device_id"].map(labels).astype("Int64")

    cohort_sizes = {int(c): int(n) for c, n in df.groupby("cohort")["device_id"].nunique().items()}

    # Cohorts too small to form a peer group: flag and exclude from baseline math.
    small = {c for c, n in cohort_sizes.items() if n < min_peers}
    warnings: list[str] = []
    if small:
        members = (
            df[df["cohort"].isin(small)]
            .groupby("cohort")["device_id"]
            .apply(lambda s: sorted(s.unique()))
        )
        for c in sorted(small):
            warnings.append(
                f"[{df['metric'].iloc[0]}] cohort {c} has {cohort_sizes[c]} device(s) "
                f"(<{min_peers}); treated as independent — no peer-z for {list(members[c])}."
            )

    grp = df.groupby(["cohort", "timestamp"], sort=False)["value"]
    df["peer_median"] = grp.transform("median")
    df["n_peers"] = grp.transform("count").astype("Int64")
    df["deviation"] = df["value"] - df["peer_median"]
    abs_dev = df["deviation"].abs()
    raw_mad = abs_dev.groupby([df["cohort"], df["timestamp"]]).transform("median")
    df["peer_mad"] = ROBUST_SCALE * raw_mad

    # peer-z is defined only with enough present peers and a non-degenerate spread.
    valid = (
        (~df["cohort"].isin(small))
        & (df["n_peers"].astype("float") >= min_peers)
        & (df["peer_mad"] > 0)
        & df["value"].notna()
    )
    df["peer_z"] = np.where(valid, df["deviation"] / df["peer_mad"], np.nan)

    df = df[PEERZ_COLUMNS].sort_values(["device_id", "timestamp"]).reset_index(drop=True)
    return df, cohort_sizes, warnings


def compute_peer_z(
    grid_df: pd.DataFrame,
    clustering: dict[str, _cluster.ClusterResult],
    min_peers: int = MIN_PEERS,
) -> PeerZResult:
    """Compute peer-z for every metric, given a regular grid and a clustering.

    ``grid_df`` must already be on a shared per-metric time grid (see
    :func:`biotan.normalize.resample_to_grid`).
    """
    parts = []
    cohort_sizes: dict[str, dict[int, int]] = {}
    warnings: list[str] = []
    for metric, gm in grid_df.groupby("metric"):
        res = clustering.get(str(metric))
        labels = res.labels if res is not None else {d: 0 for d in gm["device_id"].unique()}
        part, sizes, warns = _peer_z_for_metric(gm, labels, min_peers)
        parts.append(part)
        cohort_sizes[str(metric)] = sizes
        warnings.extend(warns)

    table = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=PEERZ_COLUMNS)
    return PeerZResult(table=table, cohort_sizes=cohort_sizes, warnings=warnings)


def run_peer_z(df: pd.DataFrame, min_peers: int = MIN_PEERS) -> tuple[PeerZResult, dict]:
    """End-to-end stages 1->3 from a normalized frame.

    Resamples to a regular grid once, clusters on that grid, then computes peer-z.
    Returns ``(PeerZResult, clustering)``.
    """
    grid = _normalize.resample_to_grid(df)
    clustering = _cluster.cluster_fleet(grid, resample=False)
    result = compute_peer_z(grid, clustering, min_peers=min_peers)
    return result, clustering
