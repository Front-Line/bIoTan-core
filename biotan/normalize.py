# BIoTan-core — zero-config, peer-relative anomaly backtesting (free open core).
# Copyright (c) 2026 Victor Minbeom Joo d/b/a Front-Line (https://frontli.ne.kr).
# Source-available under the PolyForm Noncommercial License 1.0.0 — noncommercial use only.
# Commercial or production use requires a separate license. See LICENSE.md (authoritative).
"""Stage 1 — input normalization.

Every input, whatever its original shape, is reduced to a single tidy long-format
table with the canonical columns:

    device_id | timestamp | value | metric | group | unit

``metric``/``group``/``unit`` are optional in the input. If ``metric`` is absent,
all rows are assigned a single default metric so the rest of the pipeline can treat
"one metric" and "many metrics" uniformly.

Because peer comparison (a later stage) requires aligning devices at the *same
instant*, this module also infers the dominant sampling period per metric and can
resample every device onto a shared regular time grid. All of this is automatic —
there is nothing for the user to configure.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

#: Canonical columns the rest of the pipeline relies on.
REQUIRED_COLUMNS = ("device_id", "timestamp", "value")
OPTIONAL_COLUMNS = ("metric", "group", "unit")
CANONICAL_COLUMNS = REQUIRED_COLUMNS + OPTIONAL_COLUMNS

#: Used when the input has no ``metric`` column.
DEFAULT_METRIC = "value"


class NormalizationError(ValueError):
    """Raised when the input cannot be coerced into canonical long format."""


def _canonicalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Map incoming column names to canonical ones, case/whitespace-insensitively.

    A few common aliases are accepted so typical exports work without editing.
    """
    aliases = {
        "device": "device_id",
        "device_id": "device_id",
        "deviceid": "device_id",
        "id": "device_id",
        "asset": "device_id",
        "asset_id": "device_id",
        "serial": "device_id",
        "time": "timestamp",
        "timestamp": "timestamp",
        "datetime": "timestamp",
        "date": "timestamp",
        "ts": "timestamp",
        "value": "value",
        "val": "value",
        "reading": "value",
        "measurement": "value",
        "metric": "metric",
        "signal": "metric",
        "sensor": "metric",
        "channel": "metric",
        "group": "group",
        "cohort": "group",
        "unit": "unit",
        "units": "unit",
    }
    rename = {}
    for col in df.columns:
        key = str(col).strip().lower().replace(" ", "_")
        if key in aliases:
            rename[col] = aliases[key]
    out = df.rename(columns=rename)
    # If duplicate canonical names result (e.g. both "value" and "val"), keep first.
    out = out.loc[:, ~out.columns.duplicated()]
    return out


def normalize_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce an arbitrary dataframe into canonical tidy long format.

    Returns a new dataframe with columns ``device_id, timestamp, value, metric``
    (always present) plus ``group``/``unit`` when available. Rows with unparseable
    timestamps or non-numeric values are dropped.
    """
    df = _canonicalize_columns(df)

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise NormalizationError(
            f"input is missing required column(s): {', '.join(missing)}. "
            f"Required columns are {REQUIRED_COLUMNS} (optional: {OPTIONAL_COLUMNS})."
        )

    out = df.copy()

    # --- types -----------------------------------------------------------
    out["device_id"] = out["device_id"].astype(str).str.strip()
    out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce", utc=False)
    out["value"] = pd.to_numeric(out["value"], errors="coerce")

    if "metric" in out.columns:
        out["metric"] = out["metric"].astype(str).str.strip().replace("", DEFAULT_METRIC)
        out["metric"] = out["metric"].fillna(DEFAULT_METRIC)
    else:
        out["metric"] = DEFAULT_METRIC

    keep = list(CANONICAL_COLUMNS)
    if "group" in out.columns:
        out["group"] = out["group"].astype(str).str.strip()
    if "unit" in out.columns:
        out["unit"] = out["unit"].astype(str).str.strip()
    keep = [c for c in keep if c in out.columns]
    out = out[keep]

    # --- drop unusable rows ---------------------------------------------
    before = len(out)
    out = out.dropna(subset=["device_id", "timestamp", "value"])
    out = out[out["device_id"] != ""]
    dropped = before - len(out)
    if len(out) == 0:
        raise NormalizationError(
            "no valid rows after parsing timestamps and values — check the input format."
        )

    # Collapse exact duplicate (device, metric, timestamp) keys by robust median.
    grp_keys = ["metric", "device_id", "timestamp"]
    if out.duplicated(subset=grp_keys).any():
        agg = {"value": "median"}
        for extra in ("group", "unit"):
            if extra in out.columns:
                agg[extra] = "first"
        out = out.groupby(grp_keys, as_index=False).agg(agg)

    out = out.sort_values(["metric", "device_id", "timestamp"]).reset_index(drop=True)
    out.attrs["rows_dropped"] = int(dropped)
    return out


def load(path: str) -> pd.DataFrame:
    """Read a CSV from ``path`` and return it in canonical tidy long format."""
    raw = pd.read_csv(path)
    return normalize_frame(raw)


def infer_period(timestamps: pd.Series) -> pd.Timedelta | None:
    """Estimate the dominant sampling period from a series of timestamps.

    Uses the median gap between consecutive (sorted, de-duplicated) samples, which
    is robust to occasional gaps and bursts. Returns ``None`` if it cannot be
    determined (fewer than two distinct timestamps).
    """
    ts = pd.to_datetime(pd.Series(timestamps).dropna().unique())
    if len(ts) < 2:
        return None
    ts = pd.Series(ts).sort_values()
    diffs = ts.diff().dropna()
    diffs = diffs[diffs > pd.Timedelta(0)]
    if diffs.empty:
        return None
    return diffs.median()


def _round_period(td: pd.Timedelta) -> pd.Timedelta:
    """Round an inferred period to a tidy, human-sensible cadence."""
    secs = td.total_seconds()
    # Candidate cadences from 1s up to 1 day; snap to the nearest in log space.
    candidates = [
        1, 5, 10, 15, 30,
        60, 5 * 60, 10 * 60, 15 * 60, 30 * 60,
        3600, 2 * 3600, 3 * 3600, 6 * 3600, 12 * 3600,
        86400,
    ]
    best = min(candidates, key=lambda c: abs(np.log(secs) - np.log(c)))
    return pd.Timedelta(seconds=best)


def infer_metric_periods(df: pd.DataFrame) -> dict[str, pd.Timedelta]:
    """Infer a rounded sampling period per metric (median across that metric's devices)."""
    periods: dict[str, pd.Timedelta] = {}
    for metric, mdf in df.groupby("metric"):
        per_device = []
        for _, ddf in mdf.groupby("device_id"):
            p = infer_period(ddf["timestamp"])
            if p is not None:
                per_device.append(p)
        if per_device:
            median_td = pd.Series(per_device).median()
            periods[metric] = _round_period(median_td)
    return periods


def resample_to_grid(
    df: pd.DataFrame,
    freq: pd.Timedelta | None = None,
) -> pd.DataFrame:
    """Resample every device onto a regular, shared time grid, per metric.

    For each metric a common grid (``min..max`` of that metric at the inferred
    cadence) is built and every device is reindexed onto it; multiple raw samples
    in one bin are collapsed by median. Bins with no data become NaN — that is the
    honest representation of a gap, and downstream stages handle NaN explicitly.

    The cadence is auto-inferred per metric unless ``freq`` is given.
    """
    inferred = infer_metric_periods(df) if freq is None else {}
    out_parts = []
    for metric, mdf in df.groupby("metric"):
        rule = freq if freq is not None else inferred.get(metric)
        if rule is None:
            # Single-sample metric or undeterminable cadence: pass through as-is.
            out_parts.append(mdf.copy())
            continue
        grid = pd.date_range(mdf["timestamp"].min(), mdf["timestamp"].max(), freq=rule)
        extras = {c: mdf.groupby("device_id")[c].first() for c in ("group", "unit") if c in mdf.columns}
        for device, ddf in mdf.groupby("device_id"):
            s = (
                ddf.set_index("timestamp")["value"]
                .resample(rule)
                .median()
                .reindex(grid)
            )
            part = pd.DataFrame(
                {"device_id": device, "timestamp": grid, "value": s.values, "metric": metric}
            )
            for c, lookup in extras.items():
                part[c] = lookup.get(device, np.nan)
            out_parts.append(part)

    out = pd.concat(out_parts, ignore_index=True)
    out = out.sort_values(["metric", "device_id", "timestamp"]).reset_index(drop=True)
    out.attrs["periods"] = {m: str(p) for m, p in inferred.items()} if freq is None else {}
    return out


@dataclass
class FleetSummary:
    """Quick descriptive summary of a normalized fleet (for console output)."""

    n_rows: int
    n_devices: int
    metrics: list[str]
    span_start: pd.Timestamp
    span_end: pd.Timestamp
    periods: dict[str, str]


def summarize(df: pd.DataFrame) -> FleetSummary:
    """Compute a small human-readable summary of a normalized frame."""
    return FleetSummary(
        n_rows=len(df),
        n_devices=df["device_id"].nunique(),
        metrics=sorted(df["metric"].unique().tolist()),
        span_start=df["timestamp"].min(),
        span_end=df["timestamp"].max(),
        periods={m: str(p) for m, p in infer_metric_periods(df).items()},
    )
