# BIoTan-core — tests for stage 1 (input normalization).
# Source-available under PolyForm Noncommercial License 1.0.0 — see LICENSE.md.

import numpy as np
import pandas as pd
import pytest

from biotan import normalize as N


def test_missing_required_column_raises():
    df = pd.DataFrame({"device_id": ["a"], "timestamp": ["2026-01-01"]})  # no value
    with pytest.raises(N.NormalizationError):
        N.normalize_frame(df)


def test_aliases_and_default_metric():
    df = pd.DataFrame(
        {
            "Device": ["a", "a"],
            "Time": ["2026-01-01 00:00", "2026-01-01 01:00"],
            "Reading": [1.0, 2.0],
        }
    )
    out = N.normalize_frame(df)
    assert set(["device_id", "timestamp", "value", "metric"]).issubset(out.columns)
    assert (out["metric"] == N.DEFAULT_METRIC).all()
    assert pd.api.types.is_datetime64_any_dtype(out["timestamp"])


def test_drops_unparseable_rows():
    df = pd.DataFrame(
        {
            "device_id": ["a", "a", "a"],
            "timestamp": ["2026-01-01", "not-a-date", "2026-01-02"],
            "value": [1.0, 2.0, "oops"],
        }
    )
    out = N.normalize_frame(df)
    assert len(out) == 1  # only the first row survives


def test_duplicate_keys_collapsed_by_median():
    df = pd.DataFrame(
        {
            "device_id": ["a", "a", "a"],
            "timestamp": ["2026-01-01 00:00"] * 3,
            "value": [1.0, 3.0, 5.0],
            "metric": ["m"] * 3,
        }
    )
    out = N.normalize_frame(df)
    assert len(out) == 1
    assert out["value"].iloc[0] == 3.0


def test_infer_period_hourly():
    ts = pd.date_range("2026-01-01", periods=10, freq="h")
    assert N.infer_period(pd.Series(ts)) == pd.Timedelta(hours=1)


def test_resample_aligns_devices_on_common_grid():
    # Two devices, slightly irregular, hourly cadence.
    a = pd.DataFrame(
        {
            "device_id": "a",
            "timestamp": pd.date_range("2026-01-01 00:00", periods=5, freq="h"),
            "value": np.arange(5.0),
            "metric": "m",
        }
    )
    b = pd.DataFrame(
        {
            "device_id": "b",
            "timestamp": pd.date_range("2026-01-01 00:30", periods=5, freq="h"),
            "value": np.arange(5.0),
            "metric": "m",
        }
    )
    out = N.resample_to_grid(pd.concat([a, b], ignore_index=True))
    # Both devices share the same set of timestamps after gridding.
    grids = out.groupby("device_id")["timestamp"].apply(lambda s: tuple(s))
    assert grids["a"] == grids["b"]
