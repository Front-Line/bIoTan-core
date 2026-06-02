# BIoTan-core — zero-config, peer-relative anomaly backtesting (free open core).
# Copyright (c) 2026 Victor Minbeom Joo d/b/a Front-Line (https://frontli.ne.kr).
# Source-available under the PolyForm Noncommercial License 1.0.0 — noncommercial use only.
# Commercial or production use requires a separate license. See LICENSE.md (authoritative).
"""Stage 2 — automatic cohort discovery (clustering).

Devices are grouped into behavioral cohorts so that, in the next stage, each
device can be compared against the *right* peers. There is nothing to configure:
no manual grouping, no chosen ``k``.

Behavioral profile (per device, per metric)
-------------------------------------------
Each device is summarized by:
  * a 24-dim **hour-of-day shape** — the robust (median) daily pattern, centered
    and scaled by the device's own level/spread so it captures *shape*, not size;
  * its absolute **level** (median value);
  * its **variability** (robust MAD).

The shape block and the level/variability block are balanced so neither dominates
purely by having more dimensions.

Algorithm selection
--------------------
Both KMeans (with ``k`` chosen automatically by silhouette score) and HDBSCAN are
run; whichever yields the higher silhouette wins. HDBSCAN "noise" devices are
folded into their nearest cohort so every device always has peers. If no split is
meaningfully better than treating the fleet as one cohort (or the fleet is tiny),
a single cohort is returned — a genuinely homogeneous fleet is a valid answer, not
a failure.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.stats import median_abs_deviation
from sklearn.cluster import HDBSCAN, KMeans
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from biotan import normalize as _normalize

#: Below this silhouette, a proposed split is not considered worthwhile and the
#: whole metric is treated as a single cohort (homogeneous fleet).
SINGLE_COHORT_SILHOUETTE_FLOOR = 0.10

#: Fleets smaller than this are never split — too few peers to cluster meaningfully.
MIN_DEVICES_TO_CLUSTER = 4

#: Upper bound on candidate cluster counts for KMeans.
MAX_K = 10

_HOURS = 24


@dataclass
class ClusterResult:
    """Clustering outcome for a single metric."""

    metric: str
    labels: dict[str, int]              # device_id -> cohort id (0-based, contiguous)
    method: str                         # "kmeans" | "hdbscan" | "single"
    n_cohorts: int
    silhouette: float | None            # None when single cohort
    k_candidates: dict[int, float] = field(default_factory=dict)  # KMeans k -> silhouette
    device_order: list[str] = field(default_factory=list)         # rows of the feature matrix
    notes: str = ""

    def sizes(self) -> dict[int, int]:
        """Number of devices in each cohort."""
        out: dict[int, int] = {}
        for c in self.labels.values():
            out[c] = out.get(c, 0) + 1
        return dict(sorted(out.items()))


def _device_profile(values: np.ndarray, hours: np.ndarray) -> np.ndarray:
    """Build a 26-dim profile: 24 hour-of-day shape + [level, variability]."""
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.full(_HOURS + 2, np.nan)

    level = float(np.median(finite))
    mad = float(median_abs_deviation(finite, scale="normal"))
    scale = mad if mad > 0 else 1.0

    # Robust median per hour-of-day, reindexed to 0..23.
    prof = (
        pd.Series(values, index=hours)
        .groupby(level=0)
        .median()
        .reindex(range(_HOURS))
    )
    shape = (prof.to_numpy(dtype=float) - level) / scale
    return np.concatenate([shape, [level, mad]])


def _build_feature_matrix(metric_df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """Return (X, device_order) for one metric's normalized rows."""
    devices = sorted(metric_df["device_id"].unique().tolist())
    rows = []
    for dev in devices:
        d = metric_df[metric_df["device_id"] == dev]
        ts = pd.DatetimeIndex(d["timestamp"])
        rows.append(_device_profile(d["value"].to_numpy(dtype=float), ts.hour.to_numpy()))
    X = np.vstack(rows)

    # Impute any all-missing hours (column-wise mean) so clustering is well-defined.
    col_mean = np.nanmean(X, axis=0)
    col_mean = np.where(np.isfinite(col_mean), col_mean, 0.0)
    inds = np.where(~np.isfinite(X))
    X[inds] = np.take(col_mean, inds[1])

    X = StandardScaler().fit_transform(X)

    # Balance the two feature blocks so 24 shape dims don't swamp the 2 scalars:
    # rescale each block to unit total variance.
    X[:, :_HOURS] /= np.sqrt(_HOURS)
    X[:, _HOURS:] /= np.sqrt(X.shape[1] - _HOURS)
    return X, devices


def _relabel_contiguous(labels: np.ndarray) -> np.ndarray:
    """Map arbitrary integer labels to contiguous 0..n-1 by descending cohort size."""
    uniq, counts = np.unique(labels, return_counts=True)
    order = uniq[np.argsort(-counts)]
    remap = {old: new for new, old in enumerate(order)}
    return np.array([remap[l] for l in labels])


def _assign_noise_to_nearest(X: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Fold HDBSCAN noise points (-1) into their nearest non-noise cohort centroid."""
    labels = labels.copy()
    noise = labels == -1
    if not noise.any():
        return labels
    real = np.unique(labels[labels != -1])
    centroids = {c: X[labels == c].mean(axis=0) for c in real}
    for i in np.where(noise)[0]:
        dists = {c: np.linalg.norm(X[i] - ctr) for c, ctr in centroids.items()}
        labels[i] = min(dists, key=dists.get)
    return labels


def _cluster_matrix(X: np.ndarray) -> tuple[np.ndarray, str, float | None, dict[int, float]]:
    """Run KMeans + HDBSCAN on X and pick the better. Returns labels, method, sil, k_sils."""
    n = X.shape[0]
    if n < MIN_DEVICES_TO_CLUSTER:
        return np.zeros(n, dtype=int), "single", None, {}

    # --- KMeans across candidate k, scored by silhouette --------------------
    k_sils: dict[int, float] = {}
    best_km = None  # (sil, labels)
    k_upper = min(MAX_K, n - 1)
    for k in range(2, k_upper + 1):
        # Near-constant feature rows (e.g. a flat fleet) can collapse to fewer
        # distinct points than k; that is handled below, so silence the warning.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            labels = KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(X)
        if len(np.unique(labels)) < 2:
            continue
        sil = float(silhouette_score(X, labels))
        k_sils[k] = sil
        if best_km is None or sil > best_km[0]:
            best_km = (sil, labels)

    # --- HDBSCAN (auto cluster count) ---------------------------------------
    best_hdb = None  # (sil, labels)
    min_cluster_size = max(2, n // 10)
    hdb_labels = HDBSCAN(min_cluster_size=min_cluster_size, copy=True).fit_predict(X)
    if len(np.unique(hdb_labels[hdb_labels != -1])) >= 2:
        assigned = _assign_noise_to_nearest(X, hdb_labels)
        if len(np.unique(assigned)) >= 2:
            sil = float(silhouette_score(X, assigned))
            best_hdb = (sil, assigned)

    # --- choose the winner --------------------------------------------------
    candidates = []
    if best_km is not None:
        candidates.append(("kmeans", best_km[0], best_km[1]))
    if best_hdb is not None:
        candidates.append(("hdbscan", best_hdb[0], best_hdb[1]))

    if not candidates:
        return np.zeros(n, dtype=int), "single", None, k_sils

    method, sil, labels = max(candidates, key=lambda c: c[1])
    if sil < SINGLE_COHORT_SILHOUETTE_FLOOR:
        return np.zeros(n, dtype=int), "single", None, k_sils

    return _relabel_contiguous(labels), method, sil, k_sils


def cluster_metric(metric_df: pd.DataFrame, metric: str) -> ClusterResult:
    """Discover cohorts for a single metric's normalized rows."""
    X, devices = _build_feature_matrix(metric_df)
    labels, method, sil, k_sils = _cluster_matrix(X)
    label_map = {dev: int(lbl) for dev, lbl in zip(devices, labels)}
    n_cohorts = len(set(labels.tolist()))

    notes = ""
    if method == "single":
        if len(devices) < MIN_DEVICES_TO_CLUSTER:
            notes = f"fleet too small to cluster (<{MIN_DEVICES_TO_CLUSTER} devices); single cohort."
        else:
            notes = "no split beat the homogeneity floor; treating fleet as one cohort."

    return ClusterResult(
        metric=metric,
        labels=label_map,
        method=method,
        n_cohorts=n_cohorts,
        silhouette=sil,
        k_candidates=k_sils,
        device_order=devices,
        notes=notes,
    )


def cluster_fleet(
    df: pd.DataFrame,
    resample: bool = True,
) -> dict[str, ClusterResult]:
    """Cluster every metric independently. ``df`` is a normalized long-format frame.

    With ``resample=True`` (default) devices are first placed on a shared regular
    time grid per metric, matching how peer comparison will later align them.
    Returns ``{metric: ClusterResult}``.
    """
    work = _normalize.resample_to_grid(df) if resample else df
    results: dict[str, ClusterResult] = {}
    for metric, mdf in work.groupby("metric"):
        results[metric] = cluster_metric(mdf, str(metric))
    return results
