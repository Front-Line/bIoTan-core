# BIoTan-core — zero-config, peer-relative anomaly backtesting (free open core).
# Copyright (c) 2026 Victor Minbeom Joo d/b/a Front-Line (https://frontli.ne.kr).
# Source-available under the PolyForm Noncommercial License 1.0.0 — noncommercial use only.
# Commercial or production use requires a separate license. See LICENSE.md (authoritative).
"""Reproducible synthetic datasets for cohort *event* detection (biotan.events).

Each generator returns ``(long_df, truth)`` where ``long_df`` is in BIoTan's input
schema (``device_id, timestamp, value, metric, group``) and ``truth`` documents the
ground-truth event(s) so tests and demos can score detection.

Three scenarios, matching the three Mode-A behaviors:

* ``freezer`` — a steady-state cohort; a door-open interval warms a *subset* of
  near-door sensors. Mode A should recover the interval and the signed subset.
* ``solar`` — a diurnal inverter fleet; a few inverters drift down. Mode A must
  daily-aggregate (the diurnal cycle would otherwise dominate the derivatives) and
  recover the faulty inverters, excluding noisy-but-healthy ones.
* ``lockstep`` — every member degrades together from the start (the NASA C-MAPSS
  shape). There is no intra-cohort divergence, so Mode A should find ~no events —
  the correct, honest answer.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

_TWO_PI = 2 * np.pi


def freezer(n_sensors: int = 12, n_affected: int = 3, days: int = 5,
            freq_minutes: int = 15, level: float = -18.0, warm_peak: float = 5.0,
            event_day: int = 3, event_hour: int = 14, event_hours: float = 2.0,
            noise: float = 0.15, seed: int = 0):
    """Steady-state freezer zone; a door-open interval warms the near-door subset."""
    rng = np.random.default_rng(seed)
    periods = int(days * 24 * 60 / freq_minutes)
    idx = pd.date_range("2026-02-01", periods=periods, freq=f"{freq_minutes}min")

    start = pd.Timestamp("2026-02-01") + pd.Timedelta(days=event_day, hours=event_hour)
    end = start + pd.Timedelta(hours=event_hours)
    decay_end = end + pd.Timedelta(hours=event_hours)  # warming relaxes back after the door shuts

    affected = [f"frz-{i:02d}" for i in range(n_affected)]
    rows = []
    for i in range(n_sensors):
        dev = f"frz-{i:02d}"
        val = level + rng.normal(0, noise, periods)
        if dev in affected:
            bump = np.zeros(periods)
            rise = (idx >= start) & (idx <= end)
            fall = (idx > end) & (idx <= decay_end)
            # linear warm-up during the door-open window, then linear relaxation
            if rise.any():
                bump[rise] = np.linspace(0, warm_peak, rise.sum())
            if fall.any():
                bump[fall] = np.linspace(warm_peak, 0, fall.sum())
            val = val + bump
        rows.append(pd.DataFrame({"device_id": dev, "timestamp": idx, "value": val,
                                  "metric": "temp_c", "group": "zone_A"}))
    long_df = pd.concat(rows, ignore_index=True)
    truth = {"affected": sorted(affected), "direction": "+",
             "t_start": start, "t_end": decay_end}
    return long_df, truth


def solar(n_inverters: int = 30, n_faulty: int = 3, days: int = 40,
          freq_minutes: int = 60, drift_onset_frac: float = 0.45,
          fault_magnitude: float = 0.45, noise: float = 0.04, seed: int = 1):
    """Diurnal inverter fleet; a few inverters drift down (degraded output)."""
    rng = np.random.default_rng(seed)
    periods = int(days * 24 * 60 / freq_minutes)
    idx = pd.date_range("2026-03-01", periods=periods, freq=f"{freq_minutes}min")
    hours = (idx.hour + idx.minute / 60.0).to_numpy(float)
    daylight = np.clip(np.sin((hours - 6) / 12 * np.pi), 0, None)  # 0 at night, peak ~noon
    # shared weather: a slow per-day multiplier common to the whole fleet
    day_index = (idx.normalize() - idx.normalize().min()).days
    weather = 0.7 + 0.3 * rng.random(day_index.max() + 1)
    weather_t = weather[day_index]

    onset = idx[int(periods * drift_onset_frac)]
    faulty = [f"inv-{i:02d}" for i in rng.choice(n_inverters, size=n_faulty, replace=False)]
    rows = []
    for i in range(n_inverters):
        dev = f"inv-{i:02d}"
        gain = 1.0 + 0.03 * rng.normal()
        val = 50.0 * daylight * weather_t * gain
        val = val + noise * 50.0 * rng.normal(0, 1, periods) * (daylight > 0)
        if dev in faulty:
            r = np.zeros(periods)
            mask = idx >= onset
            r[mask] = np.linspace(0, fault_magnitude, mask.sum())
            val = val * (1 - r)
        val = np.clip(val, 0, None)
        rows.append(pd.DataFrame({"device_id": dev, "timestamp": idx, "value": val,
                                  "metric": "power_kw", "group": "string_1"}))
    long_df = pd.concat(rows, ignore_index=True)
    truth = {"affected": sorted(faulty), "direction": "-", "onset": onset}
    return long_df, truth


def lockstep(n_units: int = 20, cycles: int = 120, slope: float = 0.5,
             noise: float = 0.5, seed: int = 2):
    """Every unit degrades together from cycle 1 (the C-MAPSS shape) — no subset."""
    rng = np.random.default_rng(seed)
    base = pd.Timestamp("2026-01-01")
    idx = base + pd.to_timedelta(np.arange(cycles), unit="D")  # 1 day == 1 cycle
    rows = []
    for i in range(n_units):
        dev = f"unit-{i:02d}"
        level = 100.0 + rng.normal(0, 2.0)            # benign per-unit offset
        val = level - slope * np.arange(cycles) + rng.normal(0, noise, cycles)
        rows.append(pd.DataFrame({"device_id": dev, "timestamp": idx, "value": val,
                                  "metric": "sensor", "group": "fleet"}))
    long_df = pd.concat(rows, ignore_index=True)
    truth = {"affected": [], "direction": None}  # expect ~no events
    return long_df, truth


if __name__ == "__main__":
    for name, gen in [("freezer", freezer), ("solar", solar), ("lockstep", lockstep)]:
        df, truth = gen()
        print(f"{name:9s} rows={len(df):>7,} devices={df['device_id'].nunique():>3} "
              f"truth_affected={truth['affected']}")
