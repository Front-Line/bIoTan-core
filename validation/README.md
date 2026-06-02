# Validation

Reproducible validation of the BIoTan-core method against **real, public** data.
No dataset is committed here — each script downloads its data on demand into
`validation/data/` (git-ignored). The download is the only network access; all
analysis is local.

## NASA C-MAPSS turbofan (FD001) — `run_cmapss.py`

A fleet of **100 nominally identical turbofan engines**, each run to failure under
one operating condition, 21 sensors per engine. Every engine has a **true failure
point** (its last cycle), which lets us measure the headline output — lead time —
honestly.

```bash
pip install -r requirements.txt
python validation/run_cmapss.py
# optional self-contained HTML report for the most-flagged sensor:
python validation/run_cmapss.py --report validation/cmapss_report.html
```

Mapping to BIoTan's schema: `device_id` = engine, `timestamp` = 1 day per cycle
(so "same timestamp" means "same cycle"), `metric` = each non-constant sensor,
label = each engine's last cycle.

### What it shows (and what it honestly does not)

Representative output (zero-config, no tuning):

| run | engines flagged | positive lead | median lead (cycles) |
|-----|----------------:|--------------:|---------------------:|
| zero-config (clustering as-is) | 8 / 100 | 13 / 100 | 4 |
| single-fleet (all engines = one cohort) | 13 / 100 | 11 / 100 | 11 (IQR 10–16) |

Common-mode-removal diagnostic — max `|peer-z|` in the last 15 cycles before failure:
**99/100 engines exceed 2σ**, 54/100 exceed 3.5σ (median peak ≈ 3.6).

Two honest findings, both consistent with the README's stated limits:

1. **Common-mode removal works.** After removing the per-cycle fleet baseline, the
   residual peer-z rises above 2σ before failure for almost every engine — the
   degradation really does separate from the shared operating signal.

2. **Two real limits surface.**
   - BIoTan's behavioral-profile clustering is built for **daily-cyclic** IoT data
     (solar, HVAC, air quality, climate). A turbofan's *cycle* index has no
     hour-of-day structure, so the zero-config clustering over-segments the fleet
     (14/15 sensors split into >1 cohort), which weakens peer comparison. The
     correct model for FD001 is a **single cohort**; the script reports both runs
     so the effect is visible, not hidden.
   - Peer-relative detection removes whatever the fleet does **together** and most
     strongly flags engines degrading *faster than their peers*. It is
     risk-prioritization, **not** a guaranteed per-unit predictor, and the reported
     lead time is an optimistic (hindsight) upper bound.

### Adding more datasets

The harness pattern is simple: map any source to the canonical long format
(`device_id, timestamp, value, metric`), build a labels frame
(`device_id, fault_start`) if failure dates are known, and call
`biotan.backtest.run_backtest(df, labels=...)`. Daily-cyclic homogeneous fleets
(e.g. multi-site air-quality or climate stations) are the on-design case for the
clustering stage.
