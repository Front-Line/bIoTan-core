# BIoTan-core — zero-config, peer-relative anomaly backtesting (free open core).
# Copyright (c) 2026 Victor Minbeom Joo d/b/a Front-Line (https://frontli.ne.kr).
# Source-available under the PolyForm Noncommercial License 1.0.0 — noncommercial use only.
# Commercial or production use requires a separate license. See LICENSE.md (authoritative).
"""Stage 6 — self-contained HTML report.

Renders a single standalone HTML file: dataset summary, discovered cohorts,
per-cohort peer-z overviews, detailed peer-z timelines for each flagged device
(with divergence-start and, if labeled, failure markers), and the backtest
lead-time table. Charts are drawn as inline SVG so there are **no chart-library
dependencies, no external assets, and no network calls** — the file is fully local
and self-contained, in keeping with the privacy guarantees of the project.
"""

from __future__ import annotations

import html
from datetime import datetime

import numpy as np
import pandas as pd

from biotan import backtest as _backtest
from biotan import gate as _gate
from biotan import normalize as _normalize

# ---------------------------------------------------------------------------
# tiny inline-SVG charting (no dependencies)
# ---------------------------------------------------------------------------

_PAD_L, _PAD_R, _PAD_T, _PAD_B = 48, 14, 12, 22


def _xmap(ts, t0, t1, width):
    span = (t1 - t0) / pd.Timedelta(seconds=1)
    if span <= 0:
        return _PAD_L
    frac = (ts - t0) / pd.Timedelta(seconds=1) / span
    return _PAD_L + frac * (width - _PAD_L - _PAD_R)


def _ymap(v, ycap, height):
    frac = (v + ycap) / (2 * ycap)
    frac = min(max(frac, 0.0), 1.0)
    return height - _PAD_B - frac * (height - _PAD_T - _PAD_B)


def _polyline(ts, vals, t0, t1, ycap, width, height, color, w=1.5, opacity=1.0):
    pts = []
    for t, v in zip(ts, vals):
        if pd.isna(v):
            continue
        pts.append(f"{_xmap(t, t0, t1, width):.1f},{_ymap(float(v), ycap, height):.1f}")
    if not pts:
        return ""
    return (
        f'<polyline fill="none" stroke="{color}" stroke-width="{w}" '
        f'opacity="{opacity}" points="{" ".join(pts)}"/>'
    )


def _vline(ts, t0, t1, width, height, color, label):
    x = _xmap(ts, t0, t1, width)
    return (
        f'<line x1="{x:.1f}" y1="{_PAD_T}" x2="{x:.1f}" y2="{height - _PAD_B}" '
        f'stroke="{color}" stroke-width="1.5" stroke-dasharray="4,3"/>'
        f'<text x="{x + 3:.1f}" y="{_PAD_T + 10}" font-size="10" fill="{color}">{html.escape(label)}</text>'
    )


def _chart_frame(width, height, ycap, threshold, t0, t1):
    """Axes, zero line, ±threshold band, and a few tick labels."""
    parts = [f'<rect x="0" y="0" width="{width}" height="{height}" fill="#ffffff"/>']
    # threshold band
    yb_hi = _ymap(threshold, ycap, height)
    yb_lo = _ymap(-threshold, ycap, height)
    parts.append(
        f'<rect x="{_PAD_L}" y="{yb_hi:.1f}" width="{width - _PAD_L - _PAD_R:.1f}" '
        f'height="{yb_lo - yb_hi:.1f}" fill="#eef3f8"/>'
    )
    # zero line
    y0 = _ymap(0, ycap, height)
    parts.append(
        f'<line x1="{_PAD_L}" y1="{y0:.1f}" x2="{width - _PAD_R}" y2="{y0:.1f}" '
        f'stroke="#b8c2cc" stroke-width="1"/>'
    )
    # y ticks
    for val in (ycap, 0, -ycap):
        y = _ymap(val, ycap, height)
        parts.append(
            f'<text x="{_PAD_L - 6}" y="{y + 3:.1f}" font-size="10" fill="#667" '
            f'text-anchor="end">{val:.0f}</text>'
        )
    # x ticks (start / end dates)
    for ts, anchor, x in ((t0, "start", _PAD_L), (t1, "end", width - _PAD_R)):
        parts.append(
            f'<text x="{x:.1f}" y="{height - 6}" font-size="10" fill="#667" '
            f'text-anchor="{"start" if anchor == "start" else "end"}">'
            f'{ts.strftime("%Y-%m-%d")}</text>'
        )
    return "".join(parts)


def _ycap_for(values, threshold):
    finite = np.abs(values[np.isfinite(values)])
    hi = np.nanpercentile(finite, 99) if finite.size else threshold
    return float(min(max(threshold * 1.3, hi), 15.0))


def peerz_chart_svg(dev_timeline, t0, t1, divergence_start=None, fault_start=None,
                    width=760, height=180):
    """SVG of one device's peer-z timeline with markers."""
    d = dev_timeline.sort_values("timestamp")
    threshold = _gate.POINT_Z
    ycap = _ycap_for(d["peer_z"].to_numpy(dtype=float), threshold)
    parts = [f'<svg viewBox="0 0 {width} {height}" width="100%" '
             f'preserveAspectRatio="xMidYMid meet" xmlns="http://www.w3.org/2000/svg">']
    parts.append(_chart_frame(width, height, ycap, threshold, t0, t1))
    parts.append(_polyline(d["timestamp"], d["peer_z"], t0, t1, ycap, width, height, "#c0392b", 1.4))
    if divergence_start is not None:
        parts.append(_vline(divergence_start, t0, t1, width, height, "#e67e22", "divergence"))
    if fault_start is not None:
        parts.append(_vline(fault_start, t0, t1, width, height, "#8e44ad", "failure"))
    parts.append("</svg>")
    return "".join(parts)


def cohort_overview_svg(metric_timeline, flagged_ids, t0, t1, width=760, height=220):
    """SVG of every device's peer-z in a metric, flagged ones highlighted."""
    threshold = _gate.POINT_Z
    ycap = _ycap_for(metric_timeline["peer_z"].to_numpy(dtype=float), threshold)
    parts = [f'<svg viewBox="0 0 {width} {height}" width="100%" '
             f'preserveAspectRatio="xMidYMid meet" xmlns="http://www.w3.org/2000/svg">']
    parts.append(_chart_frame(width, height, ycap, threshold, t0, t1))
    # healthy peers first (faint), flagged on top (bold red)
    for device, g in metric_timeline.groupby("device_id"):
        g = g.sort_values("timestamp")
        if device in flagged_ids:
            continue
        parts.append(_polyline(g["timestamp"], g["peer_z"], t0, t1, ycap, width, height,
                               "#9fb3c8", 0.8, opacity=0.5))
    for device, g in metric_timeline.groupby("device_id"):
        if device not in flagged_ids:
            continue
        g = g.sort_values("timestamp")
        parts.append(_polyline(g["timestamp"], g["peer_z"], t0, t1, ycap, width, height,
                               "#c0392b", 1.6))
    parts.append("</svg>")
    return "".join(parts)


# ---------------------------------------------------------------------------
# HTML assembly
# ---------------------------------------------------------------------------

_CSS = """
:root { color-scheme: light; }
* { box-sizing: border-box; }
body { font-family: -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif;
       margin: 0; color: #1f2d3d; background: #f4f6f8; }
.wrap { max-width: 920px; margin: 0 auto; padding: 28px 20px 60px; }
h1 { font-size: 22px; margin: 0 0 4px; }
h2 { font-size: 17px; margin: 30px 0 10px; border-bottom: 2px solid #e2e8f0; padding-bottom: 4px; }
h3 { font-size: 14px; margin: 18px 0 6px; }
.sub { color: #647688; font-size: 13px; margin: 0 0 18px; }
.card { background: #fff; border: 1px solid #e2e8f0; border-radius: 8px; padding: 14px 16px; margin: 12px 0; }
.banner { background: #fff8e6; border: 1px solid #f0d488; border-radius: 8px; padding: 12px 16px; font-size: 13px; }
.banner b { color: #9a6b00; }
table { border-collapse: collapse; width: 100%; font-size: 13px; }
th, td { text-align: left; padding: 6px 8px; border-bottom: 1px solid #eef2f6; }
th { color: #647688; font-weight: 600; }
.flag { color: #c0392b; font-weight: 600; }
.ok { color: #27870a; }
.muted { color: #8a99a8; }
.pill { display: inline-block; background: #fdecea; color: #c0392b; border-radius: 10px;
        padding: 1px 8px; font-size: 12px; margin-right: 4px; }
.kvs { font-size: 13px; color: #3c4b5a; }
.kvs span { margin-right: 16px; }
footer { color: #8a99a8; font-size: 12px; margin-top: 36px; text-align: center; }
""".strip()


def _esc(x) -> str:
    return html.escape(str(x))


def _fmt(x, nd=2):
    if x is None:
        return "—"
    if pd.api.types.is_scalar(x) and pd.isna(x):  # NaN, NaT, None
        return "—"
    if isinstance(x, (int, np.integer)):
        return str(int(x))
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    if isinstance(x, (pd.Timestamp, datetime)):
        return pd.Timestamp(x).strftime("%Y-%m-%d %H:%M")
    return _esc(x)


def build_report(
    summary: _normalize.FleetSummary,
    clustering: dict,
    flags: _gate.FlagResult,
    result: _backtest.BacktestResult,
    title: str = "BIoTan backtest report",
) -> str:
    timeline = result.timeline
    bt = result.table
    th = flags.thresholds
    has_labels = bt["fault_start"].notna().any()

    out = [
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>",
        "<meta name='viewport' content='width=device-width, initial-scale=1'>",
        f"<title>{_esc(title)}</title><style>{_CSS}</style></head><body><div class='wrap'>",
        f"<h1>{_esc(title)}</h1>",
        f"<p class='sub'>Generated {datetime.now().strftime('%Y-%m-%d %H:%M')} · "
        f"{summary.n_devices} devices · {len(summary.metrics)} metric(s) · "
        f"{_fmt(summary.span_start)} → {_fmt(summary.span_end)}</p>",
        "<div class='banner'><b>Read me first.</b> "
        f"{_esc(_backtest.LEAD_TIME_DISCLAIMER)} "
        "This is a peer-relative, batch backtest: it prioritizes what to inspect — "
        "it is not a guaranteed failure predictor. Devices without enough peers are "
        "marked <i>not evaluable</i>, and failures with no sustained precursor are "
        "reported honestly as <i>no clear precursor</i>.</div>",
    ]

    flagged_all = set(flags.table[flags.table["flagged"]]["device_id"])

    for metric in summary.metrics:
        mt = timeline[timeline["metric"] == metric]
        if mt.empty:
            continue
        t0, t1 = mt["timestamp"].min(), mt["timestamp"].max()
        cres = clustering.get(metric)
        mflags = flags.table[flags.table["metric"] == metric]
        flagged_ids = set(mflags[mflags["flagged"]]["device_id"])

        out.append(f"<h2>Metric: {_esc(metric)}</h2>")

        # cohorts
        if cres is not None:
            sizes = cres.sizes()
            cells = " ".join(
                f"<span><b>cohort {c}</b>: {n}</span>" for c, n in sizes.items()
            )
            method = cres.method + (f", silhouette {cres.silhouette:.2f}" if cres.silhouette else "")
            out.append(
                f"<div class='card'><div class='kvs'>Cohorts ({method}): {cells}</div></div>"
            )

        # cohort overview chart
        out.append(
            "<div class='card'><h3>Cohort overview — peer-z (common mode removed)</h3>"
            "<p class='sub'>Healthy peers cluster near zero; flagged devices in red diverge.</p>"
            + cohort_overview_svg(mt, flagged_ids, t0, t1)
            + "</div>"
        )

        # flagged device detail
        if flagged_ids:
            out.append("<h3>Flagged devices</h3>")
        for device in sorted(flagged_ids):
            brow = bt[(bt["metric"] == metric) & (bt["device_id"] == device)].iloc[0]
            frow = mflags[mflags["device_id"] == device].iloc[0]
            dev_tl = mt[mt["device_id"] == device]
            reasons = "".join(f"<span class='pill'>{_esc(r)}</span>" for r in frow["reasons"].split(",") if r)
            lead = brow["lead_time_days"]
            lead_txt = f"{lead:.1f} days before failure" if pd.notna(lead) else "—"
            out.append(
                "<div class='card'>"
                f"<h3>{_esc(device)} {reasons}</h3>"
                "<div class='kvs'>"
                f"<span>status: <b>{_esc(brow['status'])}</b></span>"
                f"<span>divergence: {_fmt(brow['divergence_start'])}</span>"
                + (f"<span>failure: {_fmt(brow['fault_start'])}</span>" if pd.notna(brow["fault_start"]) else "")
                + (f"<span>lead time: <b>{lead_txt}</b></span>" if pd.notna(lead) else "")
                + "</div>"
                + peerz_chart_svg(dev_tl, t0, t1,
                                  divergence_start=brow["divergence_start"] if pd.notna(brow["divergence_start"]) else None,
                                  fault_start=brow["fault_start"] if pd.notna(brow["fault_start"]) else None)
                + "</div>"
            )

    # backtest table
    out.append("<h2>Backtest summary</h2>")
    out.append("<div class='card'><table><thead><tr>"
               "<th>metric</th><th>device</th><th>flagged</th><th>reasons</th>"
               "<th>divergence start</th>"
               + ("<th>failure</th><th>lead (days)</th>" if has_labels else "")
               + "<th>status</th></tr></thead><tbody>")
    show = bt[bt["flagged"] | bt["fault_start"].notna()] if (bt["flagged"].any() or has_labels) else bt
    for _, r in show.iterrows():
        flag_cell = "<span class='flag'>yes</span>" if r["flagged"] else "<span class='muted'>no</span>"
        out.append(
            "<tr>"
            f"<td>{_esc(r['metric'])}</td><td>{_esc(r['device_id'])}</td>"
            f"<td>{flag_cell}</td><td>{_esc(r['reasons'])}</td>"
            f"<td>{_fmt(r['divergence_start'])}</td>"
            + (f"<td>{_fmt(r['fault_start'])}</td><td>{_fmt(r['lead_time_days'], 1)}</td>" if has_labels else "")
            + f"<td>{_esc(r['status'])}</td></tr>"
        )
    out.append("</tbody></table></div>")

    out.append(
        "<footer>BIoTan-core · peer-relative batch backtest · 100% local, no telemetry, "
        "no data leaves your machine.<br>Source-available under PolyForm Noncommercial 1.0.0 "
        f"— gate: persistent≥{th['PERSISTENT_Z']}, change≥{th['CHANGE_Z']}, "
        f"instability≥{th['INSTABILITY_Z']}, rigidity≥{th['RIGIDITY_Z']}, "
        f"effect≥{th['EFFECT_K']}× cohort scale.</footer>"
    )
    out.append("</div></body></html>")
    return "".join(out)


def write_report(
    path: str,
    df: pd.DataFrame,
    labels: pd.DataFrame | None = None,
    title: str = "BIoTan backtest report",
) -> _backtest.BacktestResult:
    """Run stages 1->6 and write a self-contained HTML report to ``path``."""
    summary = _normalize.summarize(df)
    result, flags, _peerz_result, clustering = _backtest.run_backtest(df, labels=labels)
    htmldoc = build_report(summary, clustering, flags, result, title=title)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(htmldoc)
    return result
