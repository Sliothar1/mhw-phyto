#!/usr/bin/env python3
"""End-to-end walking skeleton.

    python scripts/run_walking_skeleton.py                 # synthetic
    python scripts/run_walking_skeleton.py --source local --path data/region.nc

Runs: load -> MHW detection -> chlorophyll lag composites -> regime map ->
forecast with baselines -> figures. On synthetic data it also checks whether
the pipeline recovered the planted ground truth, which is the acceptance test.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mhwphyto import composite, data, forecast, mhw  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "outputs"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="synthetic")
    ap.add_argument("--path", default=None)
    ap.add_argument("--baseline", default="1998-01-01,2017-12-31")
    ap.add_argument("--horizon", type=int, default=28)
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args()

    OUT.mkdir(exist_ok=True)
    baseline = tuple(args.baseline.split(","))

    # ---------------------------------------------------------- 1. load
    print(f"[1/6] loading source={args.source}")
    ds = data.load(source=args.source, path=args.path)
    time = pd.DatetimeIndex(ds["time"].values)
    sst = ds["sst"].values
    chl = ds["chl"].values
    print(f"      cube {sst.shape} = (time, lat, lon), "
          f"{time[0].date()} to {time[-1].date()}")

    # ------------------------------------------------- 2. MHW detection
    print("[2/6] detecting marine heatwaves (Hobday 2016)")
    events, clim, thresh = mhw.detect_grid(sst, time, baseline=baseline)
    ncell = sst.shape[1] * sst.shape[2]
    nyear = (time[-1] - time[0]).days / 365.25
    print(f"      {len(events)} events across {ncell} cells "
          f"({len(events)/ncell/nyear:.2f} events/cell/yr)")
    print(f"      mean duration {events.duration.mean():.1f} d, "
          f"mean max intensity {events.intensity_max.mean():.2f} degC")
    print("      category mix: "
          + ", ".join(f"{k}={v}" for k, v in
                      events.category_name.value_counts().sort_index().items()))
    events.to_csv(OUT / "mhw_events.csv", index=False)

    # ------------------------------------------ 3. chlorophyll response
    print("[3/6] compositing chlorophyll anomaly on event onsets")
    chl_anom = composite.log_anomaly(chl, time, baseline=baseline)
    resp = composite.grid_response_map(chl_anom, events, min_events=5)
    resp.to_csv(OUT / "response_map.csv", index=False)
    print(resp.groupby("direction").size().to_string())

    # ------------------------------------------- 4. ground-truth check
    verdict = {}
    if "regime" in ds.variables and args.source == "synthetic":
        print("[4/6] ACCEPTANCE TEST against planted ground truth")
        truth = ds["regime"].values
        expected = {"stratified": "down", "light_limited": "up", "insensitive": "none"}
        hits = tot = 0
        for _, r in resp.iterrows():
            t = str(truth[int(r.lat_index), int(r.lon_index)])
            if r.direction == "none" and r.regime == "insufficient_events":
                continue
            tot += 1
            hits += int(r.direction == expected[t])
            verdict.setdefault(t, []).append(r.direction)
        for t, got in verdict.items():
            got_s = pd.Series(got).value_counts().to_dict()
            print(f"      planted {t:14s} -> recovered {got_s} "
                  f"(want {expected[t]!r})")
        acc = hits / tot if tot else float("nan")
        print(f"      regime sign accuracy: {acc:.0%} over {tot} cells "
              f"{'PASS' if acc >= 0.75 else 'REVIEW'}")
        verdict = {"accuracy": acc, "n_cells": tot}
    else:
        print("[4/6] no ground truth available (real data) -- skipping")

    # ---------------------------------------------------- 5. forecast
    print(f"[5/6] forecasting chl anomaly at +{args.horizon} d, single pilot cell")
    j = i = 0
    sst_anom = sst[:, j, i] - clim[:, j, i]
    flag = mhw.mhw_day_flag(sst[:, j, i], time, clim[:, j, i], thresh[:, j, i])
    feat = forecast.build_features(
        time, sst_anom, chl_anom[:, j, i], flag,
        mld=ds["mld"].values[:, j, i] if "mld" in ds else None,
        par=ds["par"].values[:, j, i] if "par" in ds else None,
        horizon=args.horizon,
    )
    folds, models = forecast.blocked_cv(feat, n_folds=4)
    cols = ["fold", "rmse_model", "rmse_persistence", "rmse_climatology",
            "skill_vs_best_baseline", "brier_model", "brier_climatology"]
    print(folds[cols].to_string(index=False, float_format=lambda v: f"{v:8.4f}"))
    mean_skill = folds.skill_vs_best_baseline.mean()
    print(f"      mean skill vs best baseline: {mean_skill:+.3f} "
          f"({'beats' if mean_skill > 0 else 'LOSES TO'} baseline)")
    folds.to_csv(OUT / "forecast_folds.csv", index=False)

    d = feat.dropna(subset=["target"])
    feats = [c for c in d.columns if c not in forecast.FEATURE_BLOCKLIST]
    last_year = d[d.index.year == d.index.year.max()]
    imp = forecast.permutation_importance(d, models[-1], feats, last_year)
    print("      top predictors: "
          + ", ".join(imp.head(5).feature.tolist()))
    imp.to_csv(OUT / "feature_importance.csv", index=False)

    # ----------------------------------------------------- 6. figures
    if not args.no_figures:
        print("[6/6] figures")
        make_figures(time, sst, clim, thresh, chl_anom, events, resp, folds)
    else:
        print("[6/6] skipped")

    summary = dict(
        source=args.source, n_events=int(len(events)),
        mean_duration_days=float(events.duration.mean()),
        regime_counts=resp.direction.value_counts().to_dict(),
        mean_forecast_skill=float(mean_skill),
        ground_truth_check=verdict,
    )
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"\nwrote {len(list(OUT.iterdir()))} artefacts to {OUT}")
    return 0


def make_figures(time, sst, clim, thresh, chl_anom, events, resp, folds):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 1, figsize=(11, 11))

    # (a) MHW detection on the pilot cell, last 5 years
    ax = axes[0]
    sel = time >= time[-1] - pd.Timedelta(days=5 * 365)
    t, s, c, th = time[sel], sst[sel, 0, 0], clim[sel, 0, 0], thresh[sel, 0, 0]
    ax.plot(t, s, lw=0.7, color="0.25", label="SST")
    ax.plot(t, c, lw=1.1, color="tab:blue", label="climatology")
    ax.plot(t, th, lw=1.0, color="tab:orange", ls="--", label="90th pctile")
    ax.fill_between(t, th, s, where=s > th, color="tab:red", alpha=0.6,
                    interpolate=True, label="MHW")
    ax.set_ylabel("SST (degC)")
    ax.set_title("(a) Marine heatwave detection, pilot cell")
    ax.legend(ncol=4, fontsize=8, loc="lower left")

    # (b) lag composite
    ax = axes[1]
    onsets = events[(events.lat_index == 0) & (events.lon_index == 0)].start_index
    comp = composite.lag_composite(chl_anom[:, 0, 0], onsets)
    ax.axhline(0, color="0.6", lw=0.8)
    ax.axvline(0, color="0.6", lw=0.8, ls=":")
    ax.plot(comp.index, comp["mean"], color="tab:green", lw=1.8)
    ax.fill_between(comp.index, comp["lo"], comp["hi"], color="tab:green", alpha=0.25)
    ax.set_xlabel("days relative to MHW onset")
    ax.set_ylabel("log10 chl anomaly")
    ax.set_title(f"(b) Chlorophyll response composite, n={len(onsets)} events "
                 "(shading = 95% bootstrap CI over events)")

    # (c) forecast skill vs baselines
    ax = axes[2]
    x = np.arange(len(folds))
    w = 0.27
    ax.bar(x - w, folds.rmse_climatology, w, label="climatology", color="0.75")
    ax.bar(x, folds.rmse_persistence, w, label="persistence", color="0.5")
    ax.bar(x + w, folds.rmse_model, w, label="gradient boosting", color="tab:purple")
    ax.set_xticks(x, [str(pd.Timestamp(d).year) for d in folds.test_start])
    ax.set_ylabel("RMSE (log10 chl)")
    ax.set_title("(c) Forecast skill vs baselines, expanding-window CV")
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(OUT / "walking_skeleton.png", dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
