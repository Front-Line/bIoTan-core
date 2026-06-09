# BIoTan-core — zero-config, peer-relative anomaly backtesting (free open core).
# Copyright (c) 2026 Victor Minbeom Joo d/b/a Front-Line (https://frontli.ne.kr).
# Source-available under the PolyForm Noncommercial License 1.0.0 — noncommercial use only.
# Commercial or production use requires a separate license. See LICENSE.md (authoritative).
"""Cohort event detection — Mode A (peer-relative).

Given a **fixed** cohort (a column you provide, e.g. ``group`` — *not*
auto-clustering), find WHEN members start to diverge from their common baseline,
then WHO diverged. The guiding example: a freezer zone where opening a door warms
the near-door sensors first.

Algorithm (Mode A)
------------------
1. **Consensus, not history.** The baseline is the per-timestamp robust consensus
   of the cohort (median + MAD across members), so we detect members *spreading
   apart*, not a member changing vs its own past.
2. **Derivative space.** We work on first differences, so constant level offsets
   between members don't matter — only how they *move* together.
3. **Robust residual.** Each member's residual = its derivative − the cohort
   consensus derivative, standardized by the per-timestamp cohort MAD → robust z.
4. **CUSUM onset.** A two-sided CUSUM (slack ``k``, threshold ``h``) accumulates
   each member's signed divergence; sustained small drift fires, transient noise
   does not. The accumulation start estimates the event **onset**.
5. **Events.** Overlapping member firings merge into cohort event intervals.
6. **Subset (who).** Within an interval the affected members are those that fired
   in a *consistent signed direction* — direction-based, not magnitude-based.

Periodic data
-------------
On diurnal data (e.g. solar) differencing raw samples produces garbage events at
sunrise/sunset. When the cohort consensus shows a strong intra-day cycle, Mode A
first **daily-aggregates** (collapsing the period) before differencing. Steady-state
cohorts (e.g. a freezer) keep their native resolution so intra-day events survive.

Honest limits (by design, not bugs)
-----------------------------------
* If **>50% of a cohort diverges together**, the consensus is itself corrupted and
  nothing is flagged — that is a global common-mode shift, which BIoTan treats as
  baseline, not an event. Such events are dropped.
* On a fleet where **everyone degrades in lockstep** (e.g. NASA C-MAPSS, all engines
  failing from cycle 1), there is no intra-cohort divergence, so Mode A finds **~no
  events**. That is the correct answer; lowering ``h`` to force events just captures
  cohort-wide noise (empty subsets) and is meaningless. (Mode B exists for that case.)
* Reports WHO and WHEN, not WHAT KIND (classification is out of scope).
* Backtested onsets are hindsight estimates — an optimistic upper bound, consistent
  with the lead-time disclaimer elsewhere in BIoTan.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from biotan import normalize as _normalize
from biotan.peerz import ROBUST_SCALE

# --- tunable defaults (zero-config; documented) -------------------------------
#: CUSUM slack — divergence below this many robust-sigma per step is ignored.
CUSUM_SLACK_K = 0.5
#: CUSUM decision threshold — accumulated divergence that fires an alarm.
CUSUM_THRESHOLD_H = 5.0
#: An event's affected member must shift its level-vs-consensus by at least this
#: many robust-sigma (effect-size gate; offset-invariant) — filters noise firings.
EVENT_EFFECT_K = 4.0
#: A cohort needs at least this many members for a meaningful consensus.
MIN_MEMBERS = 4
#: Above this affected fraction, an "event" is a cohort-wide common-mode shift.
MAX_AFFECTED_FRAC = 0.5
#: Strength of the intra-day cycle (var of hour-of-day means / total var) above
#: which the data is treated as diurnal and daily-aggregated before differencing.
DIURNAL_STRENGTH = 0.20


@dataclass
class AffectedMember:
    device_id: str
    direction: str          # "+" (rose vs cohort) or "-" (fell vs cohort)
    peak_z: float           # strongest robust-z divergence within the event
    onset: pd.Timestamp


@dataclass
class CohortEvent:
    metric: str
    cohort: str
    t_start: pd.Timestamp
    t_end: pd.Timestamp
    n_members: int
    affected: list[AffectedMember] = field(default_factory=list)

    @property
    def n_affected(self) -> int:
        return len(self.affected)

    @property
    def directions(self) -> str:
        return "".join(sorted({a.direction for a in self.affected}))


@dataclass
class EventResult:
    """All cohort events found across metrics/cohorts (Mode A)."""

    events: list[CohortEvent] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    aggregated: dict = field(default_factory=dict)   # (metric) -> "daily" | "native"

    def events_table(self) -> pd.DataFrame:
        rows = []
        for i, e in enumerate(self.events):
            rows.append({
                "event_id": i, "metric": e.metric, "cohort": e.cohort,
                "t_start": e.t_start, "t_end": e.t_end,
                "n_members": e.n_members, "n_affected": e.n_affected,
                "direction": e.directions,
                "affected": ", ".join(a.device_id for a in e.affected),
            })
        return pd.DataFrame(rows, columns=["event_id", "metric", "cohort", "t_start",
                                           "t_end", "n_members", "n_affected",
                                           "direction", "affected"])

    def affected_table(self) -> pd.DataFrame:
        rows = []
        for i, e in enumerate(self.events):
            for a in e.affected:
                rows.append({"event_id": i, "metric": e.metric, "cohort": e.cohort,
                             "device_id": a.device_id, "direction": a.direction,
                             "peak_z": a.peak_z, "onset": a.onset})
        return pd.DataFrame(rows, columns=["event_id", "metric", "cohort", "device_id",
                                           "direction", "peak_z", "onset"])


# ---------------------------------------------------------------------------
# periodicity / aggregation
# ---------------------------------------------------------------------------

def _diurnal_strength(consensus: pd.Series) -> float:
    """Fraction of consensus variance explained by hour-of-day (0..1)."""
    if consensus.notna().sum() < 4:
        return 0.0
    hours = pd.DatetimeIndex(consensus.index).hour
    hour_means = consensus.groupby(hours).mean()
    total = float(np.nanvar(consensus.to_numpy(dtype=float)))
    if total <= 0:
        return 0.0
    return float(np.nanvar(hour_means.to_numpy(dtype=float)) / total)


def _maybe_daily_aggregate(metric_df: pd.DataFrame, period_aggregate: str):
    """Daily-aggregate when the data is diurnal; otherwise keep native resolution.

    Returns ``(aggregated_long_df, mode)`` where mode is "daily" or "native".
    """
    cadence = _normalize.infer_period(metric_df["timestamp"])
    sub_daily = cadence is not None and cadence < pd.Timedelta(days=1)

    if period_aggregate == "none":
        return metric_df, "native"
    if period_aggregate == "daily":
        do_agg = True
    else:  # "auto"
        if not sub_daily:
            do_agg = False
        else:
            consensus = (metric_df.groupby("timestamp")["value"].median())
            do_agg = _diurnal_strength(consensus) >= DIURNAL_STRENGTH

    if not do_agg:
        return metric_df, "native"

    agg = metric_df.copy()
    agg["timestamp"] = agg["timestamp"].dt.floor("D")
    agg = agg.groupby(["device_id", "timestamp"], as_index=False)["value"].mean()
    return agg, "daily"


# ---------------------------------------------------------------------------
# consensus derivative z + CUSUM
# ---------------------------------------------------------------------------

def _consensus_derivative_z(member_long: pd.DataFrame):
    """Robust-z of each member's derivative vs cohort consensus, plus level info.

    Returns ``(z, dev, scale, times, members)``:
      * ``z[t, m]``   — standardized divergence of member ``m`` from the cohort's
        consensus *movement* (derivative space) at time ``t``.
      * ``dev[t, m]`` — member ``m``'s *level* deviation from the cohort consensus.
      * ``scale``     — robust, offset-invariant scale of those level deviations
        (the typical relative fluctuation), used for the event effect-size gate.
    """
    wide = (member_long.pivot_table(index="timestamp", columns="device_id",
                                    values="value", aggfunc="mean").sort_index())
    members = list(wide.columns)

    deriv = wide.diff()                                   # derivative space
    med = deriv.median(axis=1)
    mad = ROBUST_SCALE * deriv.sub(med, axis=0).abs().median(axis=1)
    z = deriv.sub(med, axis=0).div(mad.where(mad > 0), axis=0)
    z = z.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # level deviation from the per-timestamp cohort consensus
    dev = wide.sub(wide.median(axis=1), axis=0)
    # offset-invariant scale: strip each member's constant offset, then pool
    detrended = dev.sub(dev.median(axis=0), axis=1).to_numpy(dtype=float)
    detrended = detrended[np.isfinite(detrended)]
    scale = ROBUST_SCALE * float(np.median(np.abs(detrended))) if detrended.size else 0.0
    scale = max(scale, 1e-9)
    return z, dev, scale, wide.index, members


def _divergence_runs(z: np.ndarray, k: float, h: float):
    """Two-sided CUSUM. Returns runs ``(onset_i, fire_i, end_i, direction)``."""
    runs = []
    for sign in (+1, -1):
        s = 0.0
        start = None
        fired = False
        fire_i = None
        for i in range(len(z)):
            zi = z[i] if np.isfinite(z[i]) else 0.0
            inc = (zi - k) if sign > 0 else (-zi - k)
            new_s = max(0.0, s + inc)
            if s == 0.0 and new_s > 0.0:
                start, fired, fire_i = i, False, None
            s = new_s
            if s > 0.0 and not fired and s >= h:
                fired, fire_i = True, i
            if s == 0.0 and start is not None:
                if fired:
                    runs.append((start, fire_i, i - 1, sign))
                start = None
        if start is not None and fired:
            runs.append((start, fire_i, len(z) - 1, sign))
    return runs


# ---------------------------------------------------------------------------
# per-cohort detection
# ---------------------------------------------------------------------------

def _level_effect(dev_col: np.ndarray, onset_i: int, lo: int, hi: int):
    """Signed change in a member's level-vs-consensus during [lo, hi].

    Baseline = robust level deviation *before* the run; peak = the deviation in the
    window furthest from that baseline. Returns ``(signed_effect, peak_index)``.
    Offset-invariant: a constant per-member offset cancels.
    """
    pre = dev_col[:onset_i]
    pre = pre[np.isfinite(pre)]
    baseline = float(np.median(pre)) if pre.size else 0.0
    seg = dev_col[lo:hi + 1]
    rel = seg - baseline
    if not np.isfinite(rel).any():
        return 0.0, lo
    j = int(np.nanargmax(np.abs(rel)))
    return float(rel[j]), lo + j


def _detect_cohort(metric: str, cohort: str, member_long: pd.DataFrame,
                   k: float, h: float, min_effect: float
                   ) -> tuple[list[CohortEvent], list[str]]:
    notes: list[str] = []
    n_members = member_long["device_id"].nunique()
    if n_members < MIN_MEMBERS:
        notes.append(f"[{metric}/{cohort}] only {n_members} members "
                     f"(<{MIN_MEMBERS}); too few for a consensus — skipped.")
        return [], notes

    z, dev, scale, times, members = _consensus_derivative_z(member_long)
    times = pd.DatetimeIndex(times)
    pos = {t: i for i, t in enumerate(times)}

    # per-member CUSUM divergence runs (either direction)
    member_runs = []   # (onset_t, end_t, device, onset_i, end_i)
    for m in members:
        zcol = z[m].to_numpy(dtype=float)
        for onset_i, fire_i, end_i, _sign in _divergence_runs(zcol, k, h):
            member_runs.append((times[onset_i], times[end_i], m, onset_i, end_i))
    if not member_runs:
        return [], notes

    # merge time-overlapping runs into candidate cohort events
    member_runs.sort(key=lambda r: r[0])
    clusters = [[member_runs[0]]]
    for run in member_runs[1:]:
        if run[0] <= max(r[1] for r in clusters[-1]):
            clusters[-1].append(run)
        else:
            clusters.append([run])

    events: list[CohortEvent] = []
    for cl in clusters:
        ev_lo = min(r[3] for r in cl)
        ev_hi = max(r[4] for r in cl)
        # candidate members that fired anywhere in this interval
        cand = {}
        for onset_t, end_t, m, onset_i, end_i in cl:
            cand.setdefault(m, onset_i)
            cand[m] = min(cand[m], onset_i)

        # *** effect-size gate + signed direction in LEVEL space ***
        affected_recs = []
        for m, onset_i in cand.items():
            effect, peak_i = _level_effect(dev[m].to_numpy(dtype=float),
                                           onset_i, ev_lo, ev_hi)
            if abs(effect) < min_effect * scale:
                continue   # noise-driven firing — no real level shift
            affected_recs.append({"dev": m, "dir": "+" if effect > 0 else "-",
                                  "peak": abs(effect) / scale, "onset": times[onset_i]})
        if not affected_recs:
            continue

        # *** signed subset: keep the dominant-direction divergers (not magnitude) ***
        dir_count: dict[str, int] = {}
        dir_strength: dict[str, float] = {}
        for r in affected_recs:
            dir_count[r["dir"]] = dir_count.get(r["dir"], 0) + 1
            dir_strength[r["dir"]] = dir_strength.get(r["dir"], 0.0) + r["peak"]
        dom = max(dir_count, key=lambda x: (dir_count[x], dir_strength[x]))
        chosen = [r for r in affected_recs if r["dir"] == dom]

        affected = sorted(
            (AffectedMember(r["dev"], r["dir"], r["peak"], r["onset"]) for r in chosen),
            key=lambda a: (-a.peak_z, a.device_id),
        )
        t_start = min(r["onset"] for r in chosen)
        t_end = times[ev_hi]

        # honest limit: >50% co-divergence == corrupted consensus / common mode
        if len(affected) > MAX_AFFECTED_FRAC * n_members:
            notes.append(f"[{metric}/{cohort}] {t_start:%Y-%m-%d %H:%M}: "
                         f"{len(affected)}/{n_members} members diverged together "
                         f"(>50%) — treated as common-mode shift, not an event.")
            continue
        events.append(CohortEvent(metric=metric, cohort=str(cohort), t_start=t_start,
                                  t_end=t_end, n_members=int(n_members),
                                  affected=affected))
    return events, notes


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------

def detect_events(
    df: pd.DataFrame,
    cohort_col: str | None = None,
    k: float = CUSUM_SLACK_K,
    h: float = CUSUM_THRESHOLD_H,
    min_effect: float = EVENT_EFFECT_K,
    period_aggregate: str = "auto",
) -> EventResult:
    """Mode A: find peer-relative cohort events (when members diverge, and who).

    Parameters
    ----------
    df
        Normalized long frame (``device_id, timestamp, value, metric`` [, cohort col]).
    cohort_col
        Column holding the fixed cohort id (e.g. ``"group"``). If ``None`` or absent,
        every device in a metric is treated as one cohort. Auto-clustering is never
        used here.
    k, h
        CUSUM slack and threshold (robust-sigma units).
    min_effect
        Effect-size gate: an affected member must shift its level-vs-consensus by at
        least this many robust-sigma. Lower (the default, 4) is sensitive and will
        surface individual *faster-degrading* members on a near-lockstep fleet
        (e.g. C-MAPSS); raise it (e.g. 8+) for the conservative "lockstep is silent"
        view that flags only clear minority breakouts.
    period_aggregate
        ``"auto"`` (default; daily-aggregate only when diurnal), ``"daily"``, or
        ``"none"``.
    """
    result = EventResult()
    use_col = cohort_col if (cohort_col and cohort_col in df.columns) else None

    for metric, mdf in df.groupby("metric"):
        mdf_agg, mode = _maybe_daily_aggregate(mdf, period_aggregate)
        result.aggregated[str(metric)] = mode
        # the cohort column may be lost by aggregation; re-attach from the original
        if use_col is not None:
            cohort_of = mdf.drop_duplicates("device_id").set_index("device_id")[use_col]
            mdf_agg = mdf_agg.copy()
            mdf_agg["__cohort__"] = mdf_agg["device_id"].map(cohort_of)
            groups = mdf_agg.groupby("__cohort__")
        else:
            mdf_agg = mdf_agg.copy()
            mdf_agg["__cohort__"] = "all"
            groups = mdf_agg.groupby("__cohort__")

        for cohort, cdf in groups:
            events, notes = _detect_cohort(str(metric), str(cohort), cdf, k, h, min_effect)
            result.events.extend(events)
            result.notes.extend(notes)

    result.events.sort(key=lambda e: (e.metric, e.t_start))
    return result


# ===========================================================================
# Mode B (OPTIONAL, EXPERIMENTAL) — reference-trajectory deviation
# ===========================================================================
# *** Mode B is NOT peer-relative. *** It compares each member to a NORMAL
# trajectory learned from reference data (other runs / other cohorts / this fleet's
# own history), indexed by life-position. That is a different philosophy from
# BIoTan's core "no normal model, compare to peers", traded for coverage of fleets
# where everyone degrades in lockstep (so Mode A is silent). It is opt-in and never
# the default.
#
# Validated, and reported honestly:
#   * SINGLE-sensor early deviation does NOT predict lifetime (Spearman rho~0.06, n.s.).
#   * MULTI-sensor combined early deviation DOES carry a moderate, real signal
#     (rho~0.46, p<0.01). So Mode B MUST combine sensors — a single one is useless.
#   * This is a WEAK-but-real early *risk-ranking* signal, NOT an RUL predictor.

#: Reference-artifact schema version (bump on incompatible format changes).
REFERENCE_SCHEMA_VERSION = "1.0"
#: A life-position needs at least this many reference units to be recorded.
REF_MIN_UNITS = 3
#: Fraction of a unit's life used as the "early window" for risk scoring.
EARLY_FRACTION = 0.3


def _life_positions(metric_df: pd.DataFrame) -> pd.DataFrame:
    """Add an integer life-position (1..L) per device, ordered by timestamp."""
    out = metric_df.sort_values(["device_id", "timestamp"]).copy()
    out["pos"] = out.groupby("device_id").cumcount() + 1
    return out


def build_reference(df: pd.DataFrame, metadata: dict | None = None,
                    min_units: int = REF_MIN_UNITS) -> dict:
    """Build a portable reference trajectory (per sensor, life-position-indexed).

    The result is a plain, JSON-serializable dict — a self-contained, versioned,
    SHAREABLE artifact (see :func:`save_reference`). It records, for each sensor and
    each life-position, the robust median and MAD across the reference units.

    ``metadata`` may carry human context (device kind, environment, units, source,
    notes); it is stored verbatim so a profile is self-describing.
    """
    sensors: dict[str, dict] = {}
    for metric, mdf in df.groupby("metric"):
        m = _life_positions(mdf)
        recs = []
        for pos, grp in m.groupby("pos"):
            vals = grp["value"].dropna()
            if vals.count() < min_units:
                continue
            med = float(vals.median())
            mad = float(ROBUST_SCALE * (vals - med).abs().median())
            recs.append((int(pos), med, mad))
        recs.sort()
        if recs:
            allv = m["value"].dropna()
            # per-sensor GLOBAL robust scale: used to standardize early-window
            # deviations. (Per-position MAD is ~0 early and amplifies noise; the
            # global scale is what makes the multi-sensor signal appear.)
            scale = float(ROBUST_SCALE * (allv - allv.median()).abs().median())
            sensors[str(metric)] = {
                "positions": [r[0] for r in recs],
                "median": [r[1] for r in recs],
                "mad": [r[2] for r in recs],
                "scale": max(scale, 1e-9),
            }

    lengths = df.groupby(["device_id", "metric"]).size().groupby("device_id").max()
    meta = {"device_kind": None, "environment": None, "units": {},
            "source": None, "notes": None}
    if metadata:
        meta.update(metadata)
    meta["n_reference_units"] = int(df["device_id"].nunique())
    meta["median_reference_life"] = int(lengths.median()) if len(lengths) else 0
    meta["created"] = datetime.now(timezone.utc).isoformat()
    from biotan import __version__ as _v  # lazy: biotan is fully initialized at call time
    return {
        "schema_version": REFERENCE_SCHEMA_VERSION,
        "kind": "biotan.reference-trajectory",
        "biotan_version": _v,
        "life_position": "ordinal_sample_index",  # 1 == first sample, e.g. cycle 1
        "metadata": meta,
        "sensors": sensors,
    }


def save_reference(reference: dict, path: str) -> str:
    """Write a reference trajectory to a portable JSON file."""
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(reference, fh, indent=2)
    return path


def load_reference(path: str) -> dict:
    """Load a reference trajectory JSON file (validates the schema version)."""
    with open(path, encoding="utf-8") as fh:
        ref = json.load(fh)
    sv = ref.get("schema_version")
    if sv != REFERENCE_SCHEMA_VERSION:
        raise ValueError(f"unsupported reference schema_version {sv!r} "
                         f"(expected {REFERENCE_SCHEMA_VERSION!r})")
    return ref


def score_against_reference(df: pd.DataFrame, reference: dict,
                            early_window: int | None = None) -> pd.DataFrame:
    """Score each unit's *early-life* deviation from the reference trajectory.

    For each sensor, the early-window mean of ``|value − reference_median[pos]| /
    reference_scale`` (a per-sensor global robust scale) is computed; the
    ``combined_score`` averages those across sensors. The result is ranked by
    ``combined_score``; per-sensor scores are returned as ``score__<sensor>``.

    *** Combine sensors. *** A single sensor's early deviation carries no signal;
    averaging across sensors recovers a weak-but-real correlation with lifetime. This
    is an early *risk ranking*, NOT an RUL prediction.

    ``early_window`` is the number of life-positions (e.g. cycles) used; if ``None``
    it defaults to ~10% of the reference's median life.
    """
    sensors = reference.get("sensors", {})
    if early_window is None:
        med_life = reference.get("metadata", {}).get("median_reference_life", 0) or 0
        early_window = max(5, round(0.1 * med_life)) if med_life else 20

    rows = []
    for dev, ddf in df.groupby("device_id"):
        per_sensor: dict[str, float] = {}
        life = 0
        for metric, mdf in ddf.groupby("metric"):
            key = str(metric)
            if key not in sensors:
                continue
            lut_med = dict(zip(sensors[key]["positions"], sensors[key]["median"]))
            scale = sensors[key].get("scale") or 1e-9
            vals = mdf.sort_values("timestamp")["value"].to_numpy(dtype=float)
            life = max(life, len(vals))
            zs = []
            for i in range(min(early_window, len(vals))):
                pos = i + 1
                if pos in lut_med and np.isfinite(vals[i]):
                    zs.append(abs((vals[i] - lut_med[pos]) / scale))
            if zs:
                per_sensor[key] = float(np.mean(zs))
        if per_sensor:
            rec = {"device_id": dev,
                   "combined_score": float(np.mean(list(per_sensor.values()))),
                   "n_sensors": len(per_sensor), "life_length": int(life),
                   "early_window": int(early_window)}
            for kk, vv in per_sensor.items():
                rec[f"score__{kk}"] = vv
            rows.append(rec)
    cols = ["device_id", "combined_score", "n_sensors", "life_length", "early_window"]
    out = pd.DataFrame(rows)
    if out.empty:
        return pd.DataFrame(columns=cols)
    return out.sort_values("combined_score", ascending=False).reset_index(drop=True)
