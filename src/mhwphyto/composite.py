"""Event-relative composites: what does chlorophyll do before, during and
after a marine heatwave, and does the answer differ by location?

The central object is the lag composite -- chlorophyll anomaly as a function
of days relative to MHW onset, averaged over all events at a location, with a
bootstrap confidence interval over events. From that we derive:

  response_during  mean anomaly over lags [0, 30)
  response_post    mean anomaly over lags [30, 60)
  recovery_days    first lag after onset where the CI re-crosses zero
  regime label     down / up / none, from sign and significance

Note on units: chlorophyll is log-normally distributed, so all anomalies are
computed in log10 space. A response of -0.15 means roughly a 30% reduction.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .mhw import day_of_year_index, _circular_running_mean


def log_anomaly(
    chl: np.ndarray,
    time: pd.DatetimeIndex,
    baseline: tuple[str, str] | None = None,
    smooth_window: int = 31,
    floor: float = 1e-4,
) -> np.ndarray:
    """log10 chlorophyll anomaly relative to a smoothed day-of-year climatology."""
    chl = np.asarray(chl, dtype=float)
    time = pd.DatetimeIndex(time)
    lg = np.log10(np.clip(chl, floor, None))
    doy = day_of_year_index(time)

    if baseline is None:
        mask = np.ones(len(time), dtype=bool)
    else:
        lo, hi = pd.Timestamp(baseline[0]), pd.Timestamp(baseline[1])
        mask = (time >= lo) & (time <= hi)

    spatial = lg.shape[1:]
    clim_doy = np.full((366,) + spatial, np.nan)
    doy_b, lg_b = doy[mask], lg[mask]
    for d in range(1, 367):
        dist = np.abs(doy_b - d)
        dist = np.minimum(dist, 366 - dist)
        sel = dist <= 5
        if sel.any():
            clim_doy[d - 1] = np.nanmean(lg_b[sel], axis=0)
    clim_doy = _circular_running_mean(clim_doy, smooth_window)
    return lg - clim_doy[doy - 1]


def lag_composite(
    anom: np.ndarray,
    onset_indices,
    lag_min: int = -30,
    lag_max: int = 90,
    n_boot: int = 500,
    ci: float = 0.95,
    seed: int = 0,
) -> pd.DataFrame:
    """Composite anomaly by lag relative to event onset.

    Returns a DataFrame indexed by lag with columns mean, lo, hi, n.
    Bootstrap resamples EVENTS (not days), which is the right unit -- days
    within an event are strongly autocorrelated and would fake significance.
    """
    anom = np.asarray(anom, dtype=float)
    lags = np.arange(lag_min, lag_max + 1)
    onset_indices = np.asarray(list(onset_indices), dtype=int)

    if onset_indices.size == 0:
        return pd.DataFrame({"mean": np.nan, "lo": np.nan, "hi": np.nan, "n": 0},
                            index=pd.Index(lags, name="lag"))

    n = len(anom)
    mat = np.full((len(onset_indices), len(lags)), np.nan)
    for r, t0 in enumerate(onset_indices):
        idx = t0 + lags
        ok = (idx >= 0) & (idx < n)
        mat[r, ok] = anom[idx[ok]]

    mean = np.nanmean(mat, axis=0)
    counts = np.sum(np.isfinite(mat), axis=0)

    rng = np.random.default_rng(seed)
    boot = np.empty((n_boot, len(lags)))
    for b in range(n_boot):
        pick = rng.integers(0, mat.shape[0], mat.shape[0])
        with np.errstate(invalid="ignore"):
            boot[b] = np.nanmean(mat[pick], axis=0)
    alpha = (1 - ci) / 2
    lo = np.nanpercentile(boot, 100 * alpha, axis=0)
    hi = np.nanpercentile(boot, 100 * (1 - alpha), axis=0)

    return pd.DataFrame({"mean": mean, "lo": lo, "hi": hi, "n": counts},
                        index=pd.Index(lags, name="lag"))


def summarise_composite(comp: pd.DataFrame) -> dict:
    """Collapse a lag composite into scalar response metrics."""
    def window_mean(a, b):
        sel = comp.loc[(comp.index >= a) & (comp.index < b), "mean"]
        return float(sel.mean()) if len(sel) else np.nan

    during = window_mean(0, 30)
    post = window_mean(30, 60)
    pre = window_mean(-30, 0)

    # significance: does the CI over lags 0..59 exclude zero consistently?
    sel = comp.loc[(comp.index >= 0) & (comp.index < 60)]
    sig_neg = float((sel["hi"] < 0).mean())
    sig_pos = float((sel["lo"] > 0).mean())

    if sig_neg >= 0.30:
        regime, direction = "stratified_like", "down"
    elif sig_pos >= 0.30:
        regime, direction = "light_limited_like", "up"
    else:
        regime, direction = "insensitive_like", "none"

    # peak response and its lag, within 0..90
    win = comp.loc[comp.index >= 0, "mean"]
    peak_lag = int(win.abs().idxmax()) if win.notna().any() else -1
    peak_val = float(win.loc[peak_lag]) if peak_lag >= 0 else np.nan

    # recovery: first lag > peak_lag where CI re-includes zero and stays 14 days
    recovery = np.nan
    after = comp.loc[comp.index > peak_lag]
    inside = (after["lo"] <= 0) & (after["hi"] >= 0)
    run = 0
    for lag, ok in inside.items():
        run = run + 1 if ok else 0
        if run >= 14:
            recovery = float(lag - 13)
            break

    return dict(
        response_pre=pre,
        response_during=during,
        response_post=post,
        peak_response=peak_val,
        peak_lag=peak_lag,
        frac_significant_negative=sig_neg,
        frac_significant_positive=sig_pos,
        recovery_days=recovery,
        regime=regime,
        direction=direction,
    )


def grid_response_map(
    chl_anom: np.ndarray,
    events: pd.DataFrame,
    min_events: int = 5,
    **composite_kwargs,
) -> pd.DataFrame:
    """Per-cell composite summary for a (time, lat, lon) anomaly cube.

    `events` must carry lat_index / lon_index / start_index columns
    (as produced by mhw.detect_grid).
    """
    rows = []
    grouped = events.groupby(["lat_index", "lon_index"])["start_index"]
    nlat, nlon = chl_anom.shape[1], chl_anom.shape[2]
    for j in range(nlat):
        for i in range(nlon):
            try:
                onsets = grouped.get_group((j, i)).to_numpy()
            except KeyError:
                onsets = np.array([], dtype=int)
            if len(onsets) < min_events:
                rows.append(dict(lat_index=j, lon_index=i, n_events=len(onsets),
                                 regime="insufficient_events", direction="none"))
                continue
            comp = lag_composite(chl_anom[:, j, i], onsets, **composite_kwargs)
            s = summarise_composite(comp)
            s.update(lat_index=j, lon_index=i, n_events=len(onsets))
            rows.append(s)
    return pd.DataFrame(rows)
