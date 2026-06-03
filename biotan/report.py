# BIoTan-core — zero-config, peer-relative anomaly backtesting (free open core).
# Copyright (c) 2026 Victor Minbeom Joo d/b/a Front-Line (https://frontli.ne.kr).
# Source-available under the PolyForm Noncommercial License 1.0.0 — noncommercial use only.
# Commercial or production use requires a separate license. See LICENSE.md (authoritative).
"""Stage 6 — self-contained HTML report.

Renders a single standalone HTML file: a plain-language verdict ("look at these
first"), the peer-relative "breaking away from its peers" charts, the lead-time
payoff when failures are labeled, and a calm honesty note. Charts are drawn as
inline SVG so there are **no chart-library dependencies, no external assets, and no
network calls** — the file is fully local and self-contained.

This module is presentation only. It does not compute or alter any statistic — it
reads what stages 1–6 already produced and renders it.
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
# theme
# ---------------------------------------------------------------------------

# Dark, high-contrast palette tuned to look good as a screenshot/hero image.
_BG = "#0e151f"
_PANEL = "#172230"
_PANEL_2 = "#1d2a3a"
_INK = "#e8eef6"
_MUTED = "#8aa0b8"
_GRID = "#2a3b50"
_RED = "#ff6b6b"        # flagged device
_RED_SOFT = "#ff6b6b"
_GREEN = "#3ddc97"      # all-clear
_AMBER = "#ffb454"      # divergence marker
_PURPLE = "#b794f6"     # failure marker
_BAND = "#5b7fb0"       # cohort band / peers


def _esc(x) -> str:
    return html.escape(str(x))


def _fmt(x, nd=2):
    if x is None:
        return "—"
    if pd.api.types.is_scalar(x) and pd.isna(x):
        return "—"
    if isinstance(x, (int, np.integer)):
        return str(int(x))
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    if isinstance(x, (pd.Timestamp, datetime)):
        return pd.Timestamp(x).strftime("%Y-%m-%d %H:%M")
    return _esc(x)


# ---------------------------------------------------------------------------
# SVG coordinate helpers (region-based: every chart draws into a plot rect)
# ---------------------------------------------------------------------------

def _secs(t, t0):
    return (pd.Timestamp(t) - pd.Timestamp(t0)) / pd.Timedelta(seconds=1)


def _xt(t, t0, t1, rx, rw):
    span = _secs(t1, t0)
    frac = 0.0 if span <= 0 else _secs(t, t0) / span
    return rx + min(max(frac, 0.0), 1.0) * rw


def _yv(v, vmin, vmax, ry, rh):
    span = vmax - vmin
    frac = 0.5 if span <= 0 else (v - vmin) / span
    return ry + rh - min(max(frac, 0.0), 1.0) * rh


def _line(ts, vals, t0, t1, vmin, vmax, region, color, w=2.0, opacity=1.0):
    rx, ry, rw, rh = region
    pts = []
    for t, v in zip(ts, vals):
        if pd.isna(v):
            continue
        pts.append(f"{_xt(t, t0, t1, rx, rw):.1f},{_yv(float(v), vmin, vmax, ry, rh):.1f}")
    if not pts:
        return ""
    return (f'<polyline fill="none" stroke="{color}" stroke-width="{w}" '
            f'stroke-linejoin="round" stroke-linecap="round" opacity="{opacity}" '
            f'points="{" ".join(pts)}"/>')


def _vmarker(t, t0, t1, region, color, label):
    rx, ry, rw, rh = region
    x = _xt(t, t0, t1, rx, rw)
    return (f'<line x1="{x:.1f}" y1="{ry}" x2="{x:.1f}" y2="{ry + rh}" stroke="{color}" '
            f'stroke-width="1.6" stroke-dasharray="5,4"/>'
            f'<text x="{x + 4:.1f}" y="{ry + 12}" font-size="11" fill="{color}" '
            f'font-weight="600">{_esc(label)}</text>')


# ---------------------------------------------------------------------------
# the differentiator: "breaking away from its peers" (value space)
# ---------------------------------------------------------------------------

def _cohort_band(metric_tl: pd.DataFrame, cohort) -> pd.DataFrame:
    """Per-timestamp cohort value band (p25 / median / p75) — the 'normal pack'."""
    g = metric_tl[metric_tl["cohort"] == cohort]
    gb = g.groupby("timestamp")["value"]
    band = pd.DataFrame({"p25": gb.quantile(0.25), "p50": gb.median(),
                         "p75": gb.quantile(0.75)}).reset_index()
    return band.sort_values("timestamp")


def _breakout_elements(region, band, dev_ts, dev_vals, t0, t1, div, fail):
    """SVG elements: cohort IQR band + median, the device line breaking out, markers."""
    rx, ry, rw, rh = region
    allv = np.concatenate([band["p25"].to_numpy(float), band["p75"].to_numpy(float),
                           np.asarray(dev_vals, float)])
    allv = allv[np.isfinite(allv)]
    vmin, vmax = float(allv.min()), float(allv.max())
    pad = (vmax - vmin) * 0.08 or 1.0
    vmin, vmax = vmin - pad, vmax + pad

    parts = [f'<rect x="{rx}" y="{ry}" width="{rw}" height="{rh}" fill="{_PANEL_2}" rx="6"/>']
    # cohort IQR band as a filled polygon (p75 forward, p25 back)
    top = [f"{_xt(t, t0, t1, rx, rw):.1f},{_yv(v, vmin, vmax, ry, rh):.1f}"
           for t, v in zip(band["timestamp"], band["p75"]) if pd.notna(v)]
    bot = [f"{_xt(t, t0, t1, rx, rw):.1f},{_yv(v, vmin, vmax, ry, rh):.1f}"
           for t, v in zip(band["timestamp"][::-1], band["p25"][::-1]) if pd.notna(v)]
    if top and bot:
        parts.append(f'<polygon points="{" ".join(top + bot)}" fill="{_BAND}" '
                     f'opacity="0.22"/>')
    # cohort median
    parts.append(_line(band["timestamp"], band["p50"], t0, t1, vmin, vmax, region,
                       _BAND, w=1.6, opacity=0.9))
    # the device, breaking out
    parts.append(_line(dev_ts, dev_vals, t0, t1, vmin, vmax, region, _RED, w=2.6))
    # markers
    if div is not None:
        parts.append(_vmarker(div, t0, t1, region, _AMBER, "divergence"))
    if fail is not None:
        parts.append(_vmarker(fail, t0, t1, region, _PURPLE, "failure"))
    # date ticks
    parts.append(f'<text x="{rx}" y="{ry + rh + 15}" font-size="11" fill="{_MUTED}">'
                 f'{pd.Timestamp(t0).strftime("%Y-%m-%d")}</text>')
    parts.append(f'<text x="{rx + rw}" y="{ry + rh + 15}" font-size="11" fill="{_MUTED}" '
                 f'text-anchor="end">{pd.Timestamp(t1).strftime("%Y-%m-%d")}</text>')
    return "".join(parts)


def breakout_chart_svg(metric_tl, device, cohort, t0, t1, div=None, fail=None,
                       width=860, height=240):
    """Standalone SVG: a device breaking out of its cohort's value band."""
    band = _cohort_band(metric_tl, cohort)
    dev = metric_tl[metric_tl["device_id"] == device].sort_values("timestamp")
    region = (54, 16, width - 70, height - 44)
    inner = _breakout_elements(region, band, dev["timestamp"], dev["value"].to_numpy(float),
                               t0, t1, div, fail)
    return (f'<svg viewBox="0 0 {width} {height}" width="100%" '
            f'preserveAspectRatio="xMidYMid meet" xmlns="http://www.w3.org/2000/svg">'
            f'<rect width="{width}" height="{height}" fill="{_PANEL}"/>{inner}</svg>')


# ---------------------------------------------------------------------------
# plain-language verdict helpers (derived from existing numbers only)
# ---------------------------------------------------------------------------

def _pct_vs_cohort(metric_tl, device, cohort, since) -> float | None:
    """Median %% gap between the device and its cohort over the window [since, end]."""
    band = _cohort_band(metric_tl, cohort).set_index("timestamp")["p50"]
    dev = metric_tl[metric_tl["device_id"] == device].set_index("timestamp")["value"]
    if since is not None:
        band, dev = band[band.index >= since], dev[dev.index >= since]
    common = band.index.intersection(dev.index)
    if len(common) == 0:
        return None
    coh = float(np.nanmedian(band.loc[common]))
    dv = float(np.nanmedian(dev.loc[common]))
    if not np.isfinite(coh) or abs(coh) < 1e-9:
        return None
    return (dv - coh) / abs(coh) * 100.0


def _day_index(ts, t0) -> int:
    return int((pd.Timestamp(ts) - pd.Timestamp(t0)) / pd.Timedelta(days=1)) + 1


def _plain_reason(metric_tl, device, cohort, reasons, div, t0) -> str:
    """One human sentence, e.g. 'running 12% below its cohort since day 40'."""
    rs = [r for r in reasons.split(",") if r]
    pct = _pct_vs_cohort(metric_tl, device, cohort, div)
    direction = "below" if (pct is not None and pct < 0) else "above"
    mag = f"{abs(pct):.0f}%" if pct is not None else "noticeably"
    when_start = f"starting day {_day_index(div, t0)}" if div is not None else "from the start"
    when_since = f"since day {_day_index(div, t0)}" if div is not None else "across the record"

    primary = ("change" if "change" in rs else "persistent" if "persistent" in rs
               else "rigidity" if "rigidity" in rs else "instability" if "instability" in rs
               else (rs[0] if rs else ""))
    if primary == "change":
        return f"drifted to {mag} {direction} its cohort, {when_start}"
    if primary == "persistent":
        return f"running consistently {mag} {direction} its cohort"
    if primary == "rigidity":
        return f"flatlined while its cohort kept varying, {when_start}"
    if primary == "instability":
        return f"swinging erratically against its cohort, {when_since}"
    return f"{mag} {direction} its cohort {when_since}"


def _severity(metric_tl, device) -> float:
    s = metric_tl[metric_tl["device_id"] == device]["peer_z"].abs()
    return float(s.max()) if s.notna().any() else 0.0


def _lead_payoff(bt: pd.DataFrame) -> str | None:
    leads = bt[bt["flagged"]]["lead_time_days"].dropna()
    leads = leads[leads > 0]
    if leads.empty:
        return None
    med = leads.median()
    unit = "day" if abs(med - 1) < 1e-9 else "days"
    return f"{med:.0f} {unit}" if med >= 10 else f"{med:.1f} {unit}"


# ---------------------------------------------------------------------------
# hero (verdict + payoff + the #1 device breakout) — one self-contained SVG
# ---------------------------------------------------------------------------

def _hero_svg(n_flag, n_total, payoff, headline_device, reason_line, metric_tl,
              cohort, t0, t1, div, fail, width=920, height=470):
    parts = [f'<svg viewBox="0 0 {width} {height}" width="100%" '
             f'preserveAspectRatio="xMidYMid meet" xmlns="http://www.w3.org/2000/svg" '
             f'font-family="-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif">']
    parts.append(f'<rect width="{width}" height="{height}" fill="{_PANEL}" rx="14"/>')

    if n_flag == 0:
        parts.append(f'<text x="40" y="74" font-size="34" font-weight="800" fill="{_GREEN}">'
                     f'No devices need attention</text>')
        parts.append(f'<text x="40" y="110" font-size="16" fill="{_MUTED}">'
                     f'All {n_total} devices are tracking their peers. '
                     f'Nothing is breaking away from its cohort.</text>')
        parts.append("</svg>")
        return "".join(parts)

    # verdict
    parts.append(f'<text x="40" y="54" font-size="14" fill="{_MUTED}" '
                 f'letter-spacing="2">VERDICT</text>')
    parts.append(f'<text x="40" y="92" font-size="30" font-weight="800" fill="{_INK}">'
                 f'<tspan fill="{_RED}">{n_flag}</tspan> of {n_total} devices need a look</text>')
    # lead-time payoff (full-width line, no overlap)
    if payoff:
        parts.append(f'<text x="40" y="124" font-size="17" font-weight="700" fill="{_AMBER}">'
                     f'Early warning — flagged a median of {_esc(payoff)} before failure</text>')

    # headline device label + plain reason
    yname = 158 if payoff else 132
    parts.append(f'<text x="40" y="{yname}" font-size="18" font-weight="700" fill="{_INK}">'
                 f'{_esc(headline_device)}</text>')
    parts.append(f'<text x="40" y="{yname + 23}" font-size="15" fill="{_RED}">'
                 f'{_esc(reason_line)}</text>')

    # the breakout chart, drawn in a region inside the hero
    region = (54, 212, width - 94, height - 252)
    band = _cohort_band(metric_tl, cohort)
    dev = metric_tl[metric_tl["device_id"] == headline_device].sort_values("timestamp")
    parts.append(_breakout_elements(region, band, dev["timestamp"],
                                    dev["value"].to_numpy(float), t0, t1, div, fail))
    # legend
    ly = height - 8
    parts.append(f'<rect x="54" y="{ly - 9}" width="22" height="3" fill="{_RED}"/>'
                 f'<text x="82" y="{ly - 5}" font-size="11" fill="{_MUTED}">this device</text>'
                 f'<rect x="170" y="{ly - 12}" width="22" height="10" fill="{_BAND}" opacity="0.3"/>'
                 f'<text x="198" y="{ly - 5}" font-size="11" fill="{_MUTED}">cohort range (25–75%)</text>')
    parts.append("</svg>")
    return "".join(parts)


# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------

_CSS = f"""
:root {{ color-scheme: dark; }}
* {{ box-sizing: border-box; }}
body {{ font-family: -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
       margin: 0; background: {_BG}; color: {_INK};
       -webkit-font-smoothing: antialiased; }}
.wrap {{ max-width: 980px; margin: 0 auto; padding: 32px 22px 64px; }}
.top {{ display: flex; align-items: baseline; justify-content: space-between; gap: 16px;
       flex-wrap: wrap; margin-bottom: 18px; }}
h1 {{ font-size: 20px; margin: 0; letter-spacing: 0.2px; }}
.meta {{ color: {_MUTED}; font-size: 13px; }}
h2 {{ font-size: 14px; letter-spacing: 1.4px; text-transform: uppercase; color: {_MUTED};
     margin: 38px 0 12px; }}
.panel {{ background: {_PANEL}; border: 1px solid {_GRID}; border-radius: 14px;
         padding: 18px 20px; margin: 14px 0; }}
.hero {{ margin: 4px 0 8px; }}
.list {{ list-style: none; padding: 0; margin: 0; }}
.row {{ display: flex; align-items: flex-start; gap: 14px; padding: 14px 4px;
       border-bottom: 1px solid {_GRID}; }}
.row:last-child {{ border-bottom: none; }}
.rank {{ flex: 0 0 30px; height: 30px; border-radius: 8px; background: {_PANEL_2};
        color: {_RED}; font-weight: 800; display: flex; align-items: center;
        justify-content: center; font-size: 15px; }}
.dev {{ font-weight: 700; font-size: 15px; }}
.why {{ color: {_INK}; font-size: 14px; margin-top: 2px; }}
.sub {{ color: {_MUTED}; font-size: 12.5px; margin-top: 3px; }}
.pill {{ display: inline-block; background: {_PANEL_2}; color: {_AMBER};
        border: 1px solid {_GRID}; border-radius: 999px; padding: 1px 9px;
        font-size: 11.5px; margin-right: 5px; }}
.lead {{ color: {_AMBER}; font-weight: 700; }}
table {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
th, td {{ text-align: left; padding: 7px 9px; border-bottom: 1px solid {_GRID}; }}
th {{ color: {_MUTED}; font-weight: 600; }}
.flag {{ color: {_RED}; font-weight: 700; }}
.muted {{ color: {_MUTED}; }}
footer {{ color: {_MUTED}; font-size: 12px; margin-top: 40px; line-height: 1.55;
         border-top: 1px solid {_GRID}; padding-top: 16px; }}
footer b {{ color: {_INK}; }}
""".strip()


# ---------------------------------------------------------------------------
# report assembly
# ---------------------------------------------------------------------------

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
    flagged_rows = flags.table[flags.table["flagged"]]
    n_flag = int(flagged_rows["device_id"].nunique())
    n_total = int(flags.table["device_id"].nunique())

    # rank flagged (device, metric) pairs by severity, build plain-language rows
    ranked = []
    for _, fr in flagged_rows.iterrows():
        metric, device = fr["metric"], fr["device_id"]
        mt = timeline[timeline["metric"] == metric]
        brow = bt[(bt["metric"] == metric) & (bt["device_id"] == device)].iloc[0]
        t0, t1 = mt["timestamp"].min(), mt["timestamp"].max()
        div = brow["divergence_start"] if pd.notna(brow["divergence_start"]) else None
        reason = _plain_reason(mt, device, fr["cohort"], fr["reasons"], div, t0)
        ranked.append({
            "metric": metric, "device": device, "cohort": fr["cohort"],
            "reasons": fr["reasons"], "reason_line": reason, "severity": _severity(mt, device),
            "brow": brow, "t0": t0, "t1": t1, "div": div,
            "fail": brow["fault_start"] if pd.notna(brow["fault_start"]) else None,
        })
    ranked.sort(key=lambda r: r["severity"], reverse=True)
    payoff = _lead_payoff(bt) if has_labels else None

    out = [
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>",
        "<meta name='viewport' content='width=device-width, initial-scale=1'>",
        f"<title>{_esc(title)}</title><style>{_CSS}</style></head><body><div class='wrap'>",
        "<div class='top'>",
        f"<h1>{_esc(title)}</h1>",
        f"<div class='meta'>{summary.n_devices} devices · {len(summary.metrics)} metric(s) · "
        f"{_fmt(summary.span_start)} → {_fmt(summary.span_end)}</div>",
        "</div>",
    ]

    # ---- HERO: lead with the answer ----
    if ranked:
        h = ranked[0]
        hmt = timeline[timeline["metric"] == h["metric"]]
        hero = _hero_svg(n_flag, n_total, payoff, h["device"], h["reason_line"],
                         hmt, h["cohort"], h["t0"], h["t1"], h["div"], h["fail"])
    else:
        hero = _hero_svg(0, n_total, None, None, None, None, None, None, None, None, None)
    out.append(f"<div class='hero'>{hero}</div>")

    # ---- LOOK AT THESE FIRST ----
    if ranked:
        out.append("<h2>Look at these first</h2><div class='panel'><ol class='list'>")
        for i, r in enumerate(ranked, 1):
            brow = r["brow"]
            lead = brow["lead_time_days"]
            metric_note = f" · {_esc(r['metric'])}" if len(summary.metrics) > 1 else ""
            lead_note = (f"<span class='lead'>~{lead:.0f} days before failure</span> · "
                         if pd.notna(lead) and lead > 0 else "")
            pills = "".join(f"<span class='pill'>{_esc(x)}</span>"
                            for x in r["reasons"].split(",") if x)
            out.append(
                "<li class='row'>"
                f"<div class='rank'>{i}</div>"
                "<div>"
                f"<div class='dev'>{_esc(r['device'])}{metric_note}</div>"
                f"<div class='why'>{_esc(r['reason_line'])}</div>"
                f"<div class='sub'>{lead_note}{pills}</div>"
                "</div></li>"
            )
        out.append("</ol></div>")

    # ---- per-device breakout charts ----
    for r in ranked:
        mt = timeline[timeline["metric"] == r["metric"]]
        brow = r["brow"]
        chart = breakout_chart_svg(mt, r["device"], r["cohort"], r["t0"], r["t1"],
                                   div=r["div"], fail=r["fail"])
        bits = [f"<span class='muted'>status:</span> {_esc(brow['status'])}"]
        if r["div"] is not None:
            bits.append(f"<span class='muted'>divergence:</span> {_fmt(r['div'])}")
        if pd.notna(brow["fault_start"]):
            bits.append(f"<span class='muted'>failure:</span> {_fmt(brow['fault_start'])}")
        out.append(
            "<div class='panel'>"
            f"<div class='dev'>{_esc(r['device'])} — {_esc(r['reason_line'])}</div>"
            f"<div class='sub' style='margin:4px 0 10px'>{' · '.join(bits)}</div>"
            f"{chart}</div>"
        )

    # ---- backtest table (secondary, full detail) ----
    out.append("<h2>All devices</h2>")
    out.append("<div class='panel'><table><thead><tr>"
               "<th>device</th><th>metric</th><th>flagged</th><th>reasons</th>"
               "<th>divergence start</th>"
               + ("<th>failure</th><th>lead (days)</th>" if has_labels else "")
               + "<th>status</th></tr></thead><tbody>")
    show = bt[bt["flagged"] | bt["fault_start"].notna()] if (bt["flagged"].any() or has_labels) else bt
    show = show.sort_values(["flagged", "lead_time_days"], ascending=[False, False])
    for _, r in show.iterrows():
        flag_cell = "<span class='flag'>yes</span>" if r["flagged"] else "<span class='muted'>no</span>"
        out.append(
            "<tr>"
            f"<td>{_esc(r['device_id'])}</td><td>{_esc(r['metric'])}</td>"
            f"<td>{flag_cell}</td><td>{_esc(r['reasons'])}</td>"
            f"<td>{_fmt(r['divergence_start'])}</td>"
            + (f"<td>{_fmt(r['fault_start'])}</td><td>{_fmt(r['lead_time_days'], 1)}</td>" if has_labels else "")
            + f"<td>{_esc(r['status'])}</td></tr>"
        )
    out.append("</tbody></table></div>")

    # ---- calm honesty footer (kept visible, never buried) ----
    out.append(
        "<footer>"
        f"<b>What this does and doesn't tell you.</b> BIoTan compares each device to its "
        f"own peers — no thresholds, no manual grouping. It prioritizes what to inspect; it "
        f"is not a guaranteed failure predictor. {_esc(_backtest.LEAD_TIME_DISCLAIMER)} "
        f"Devices without enough peers are left out rather than guessed at, and failures "
        f"with no sustained precursor are reported as such.<br><br>"
        f"100% local · no data leaves your machine · no telemetry. "
        f"Source-available under PolyForm Noncommercial 1.0.0. "
        f"Gate: persistent≥{th['PERSISTENT_Z']}, change≥{th['CHANGE_Z']}, "
        f"instability≥{th['INSTABILITY_Z']}, rigidity≥{th['RIGIDITY_Z']}, "
        f"effect≥{th['EFFECT_K']}× cohort scale."
        "</footer>"
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
