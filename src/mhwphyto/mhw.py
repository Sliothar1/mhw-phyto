"""Marine heatwave detection following Hobday et al. (2016) with the
categorisation scheme of Hobday et al. (2018).

Definition implemented here
---------------------------
A marine heatwave (MHW) is a period of >= `min_duration` consecutive days
where SST exceeds a seasonally varying 90th-percentile threshold computed
from a fixed climatological baseline. Successive events separated by a gap
of <= `max_gap` days are joined into one event.

Everything is plain numpy so it runs without dask and is easy to unit test.
Reference values to check against: Hobday et al. 2016, Prog. Oceanogr. 141;
Hobday et al. 2018, Oceanography 31(2). VERIFY these citations yourself --
they were written from memory, not fetched.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Iterable

import numpy as np
import pandas as pd

DEFAULT_WINDOW_HALF_WIDTH = 5   # -> 11-day window around each day-of-year
DEFAULT_SMOOTH_WINDOW = 31      # running mean applied across day-of-year
DEFAULT_PCTILE = 90.0
DEFAULT_MIN_DURATION = 5        # days
DEFAULT_MAX_GAP = 2             # days


# ---------------------------------------------------------------- climatology

def day_of_year_index(time: pd.DatetimeIndex) -> np.ndarray:
    """Day-of-year in 1..366, with non-leap years mapped onto the 366-day grid
    so that a fixed DOY always refers to the same calendar position."""
    time = pd.DatetimeIndex(time)
    doy = time.dayofyear.to_numpy().astype(int)
    leap = np.asarray(time.is_leap_year)
    # in non-leap years shift everything after Feb 28 up by one
    doy = np.where((~leap) & (doy > 59), doy + 1, doy)
    return doy


def _circular_running_mean(x: np.ndarray, window: int) -> np.ndarray:
    """Running mean along axis 0 with wrap-around (day-of-year is circular)."""
    if window <= 1:
        return x
    half = window // 2
    n = x.shape[0]
    idx = (np.arange(n)[:, None] + np.arange(-half, half + 1)[None, :]) % n
    return np.nanmean(x[idx], axis=1)


def climatology(
    sst: np.ndarray,
    time: pd.DatetimeIndex,
    baseline: tuple[str, str] | None = None,
    pctile: float = DEFAULT_PCTILE,
    window_half_width: int = DEFAULT_WINDOW_HALF_WIDTH,
    smooth_window: int = DEFAULT_SMOOTH_WINDOW,
) -> tuple[np.ndarray, np.ndarray]:
    """Seasonal mean and percentile threshold.

    Parameters
    ----------
    sst : array, shape (time,) or (time, ...) -- extra dims are spatial.
    baseline : (start, end) date strings defining the climatological period.
               Best practice is a fixed 30-year baseline, e.g. 1982-2011.

    Returns
    -------
    (clim, thresh), each shaped like `sst` -- i.e. broadcast back onto the
    full time axis so anomalies are a simple subtraction.
    """
    time = pd.DatetimeIndex(time)
    sst = np.asarray(sst, dtype=float)
    doy = day_of_year_index(time)

    if baseline is None:
        mask = np.ones(len(time), dtype=bool)
    else:
        lo, hi = pd.Timestamp(baseline[0]), pd.Timestamp(baseline[1])
        mask = (time >= lo) & (time <= hi)
        if mask.sum() < 365:
            raise ValueError(
                f"baseline {baseline} covers only {mask.sum()} days; "
                "need at least a year (30 years recommended)"
            )

    spatial = sst.shape[1:]
    clim_doy = np.full((366,) + spatial, np.nan)
    thr_doy = np.full((366,) + spatial, np.nan)

    doy_b = doy[mask]
    sst_b = sst[mask]

    for d in range(1, 367):
        # circular distance on a 366-day year
        dist = np.abs(doy_b - d)
        dist = np.minimum(dist, 366 - dist)
        sel = dist <= window_half_width
        if not sel.any():
            continue
        window = sst_b[sel]
        clim_doy[d - 1] = np.nanmean(window, axis=0)
        thr_doy[d - 1] = np.nanpercentile(window, pctile, axis=0)

    clim_doy = _circular_running_mean(clim_doy, smooth_window)
    thr_doy = _circular_running_mean(thr_doy, smooth_window)

    return clim_doy[doy - 1], thr_doy[doy - 1]


# ------------------------------------------------------------------- events

@dataclass
class MHWEvent:
    start_index: int
    end_index: int
    start_date: pd.Timestamp
    end_date: pd.Timestamp
    peak_date: pd.Timestamp
    duration: int                 # days
    intensity_max: float          # degC above climatology
    intensity_mean: float         # degC
    intensity_cumulative: float   # degC-days
    category: int                 # 1 Moderate .. 4 Extreme
    category_name: str

    def as_dict(self) -> dict:
        return asdict(self)


CATEGORY_NAMES = {1: "Moderate", 2: "Strong", 3: "Severe", 4: "Extreme"}


def _runs(flag: np.ndarray) -> list[tuple[int, int]]:
    """Inclusive (start, end) index pairs of True runs."""
    if flag.size == 0:
        return []
    f = flag.astype(np.int8)
    d = np.diff(np.concatenate(([0], f, [0])))
    starts = np.flatnonzero(d == 1)
    ends = np.flatnonzero(d == -1) - 1
    return list(zip(starts.tolist(), ends.tolist()))


def detect_events(
    sst: np.ndarray,
    time: pd.DatetimeIndex,
    clim: np.ndarray,
    thresh: np.ndarray,
    min_duration: int = DEFAULT_MIN_DURATION,
    max_gap: int = DEFAULT_MAX_GAP,
) -> list[MHWEvent]:
    """Detect MHW events in a single 1-D SST time series."""
    time = pd.DatetimeIndex(time)
    sst = np.asarray(sst, dtype=float)
    if sst.ndim != 1:
        raise ValueError("detect_events expects a 1-D series; use detect_grid")

    exceed = np.zeros(sst.shape, dtype=bool)
    valid = np.isfinite(sst) & np.isfinite(thresh)
    exceed[valid] = sst[valid] > thresh[valid]

    runs = [(s, e) for s, e in _runs(exceed) if (e - s + 1) >= min_duration]

    # join runs separated by a short gap
    joined: list[list[int]] = []
    for s, e in runs:
        if joined and (s - joined[-1][1] - 1) <= max_gap:
            joined[-1][1] = e
        else:
            joined.append([s, e])

    events: list[MHWEvent] = []
    for s, e in joined:
        sl = slice(s, e + 1)
        anom = sst[sl] - clim[sl]
        span = thresh[sl] - clim[sl]          # 1 "category unit"
        peak = int(np.nanargmax(anom))
        imax = float(np.nanmax(anom))
        with np.errstate(divide="ignore", invalid="ignore"):
            cat = int(np.clip(np.floor(np.nanmax(anom / span)), 1, 4))
        events.append(
            MHWEvent(
                start_index=s,
                end_index=e,
                start_date=time[s],
                end_date=time[e],
                peak_date=time[s + peak],
                duration=e - s + 1,
                intensity_max=imax,
                intensity_mean=float(np.nanmean(anom)),
                intensity_cumulative=float(np.nansum(anom)),
                category=cat,
                category_name=CATEGORY_NAMES[cat],
            )
        )
    return events


def events_to_frame(events: Iterable[MHWEvent]) -> pd.DataFrame:
    rows = [e.as_dict() for e in events]
    if not rows:
        return pd.DataFrame(columns=[f.name for f in MHWEvent.__dataclass_fields__.values()])
    return pd.DataFrame(rows)


def detect_grid(
    sst: np.ndarray,
    time: pd.DatetimeIndex,
    baseline: tuple[str, str] | None = None,
    **kwargs,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Run detection over a (time, lat, lon) cube.

    Returns
    -------
    events : DataFrame with lat_index / lon_index columns added
    clim, thresh : arrays shaped like `sst`
    """
    sst = np.asarray(sst, dtype=float)
    if sst.ndim != 3:
        raise ValueError("detect_grid expects (time, lat, lon)")
    clim, thresh = climatology(sst, time, baseline=baseline)

    frames = []
    nlat, nlon = sst.shape[1], sst.shape[2]
    for j in range(nlat):
        for i in range(nlon):
            ev = detect_events(sst[:, j, i], time, clim[:, j, i], thresh[:, j, i], **kwargs)
            if ev:
                f = events_to_frame(ev)
                f["lat_index"] = j
                f["lon_index"] = i
                frames.append(f)
    events = pd.concat(frames, ignore_index=True) if frames else events_to_frame([])
    return events, clim, thresh


def mhw_day_flag(sst: np.ndarray, time, clim: np.ndarray, thresh: np.ndarray,
                 **kwargs) -> np.ndarray:
    """Boolean 'is an MHW day' mask, honouring duration and gap rules.
    Works on (time,) or (time, lat, lon)."""
    sst = np.asarray(sst, dtype=float)
    if sst.ndim == 1:
        out = np.zeros(sst.shape, dtype=bool)
        for e in detect_events(sst, time, clim, thresh, **kwargs):
            out[e.start_index:e.end_index + 1] = True
        return out
    out = np.zeros(sst.shape, dtype=bool)
    for j in range(sst.shape[1]):
        for i in range(sst.shape[2]):
            out[:, j, i] = mhw_day_flag(
                sst[:, j, i], time, clim[:, j, i], thresh[:, j, i], **kwargs
            )
    return out
