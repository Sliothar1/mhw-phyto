"""Unit tests for the detection rules. Run: python -m pytest tests -q"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mhwphyto import composite, mhw  # noqa: E402


def _series(n=800):
    time = pd.date_range("2000-01-01", periods=n, freq="D")
    sst = np.full(n, 15.0)
    clim = np.full(n, 15.0)
    thresh = np.full(n, 16.0)
    return time, sst, clim, thresh


def test_min_duration_rule():
    """4 days above threshold is not an event; 5 days is."""
    time, sst, clim, thresh = _series()
    sst[100:104] = 17.0
    assert mhw.detect_events(sst, time, clim, thresh) == []

    sst[:] = 15.0
    sst[100:105] = 17.0
    ev = mhw.detect_events(sst, time, clim, thresh)
    assert len(ev) == 1 and ev[0].duration == 5


def test_gap_joining_rule():
    """Two 5-day events split by a 2-day gap join; a 3-day gap does not."""
    time, sst, clim, thresh = _series()
    sst[100:105] = 17.0
    sst[107:112] = 17.0          # gap of 2 days
    ev = mhw.detect_events(sst, time, clim, thresh)
    assert len(ev) == 1 and ev[0].duration == 12

    sst[:] = 15.0
    sst[100:105] = 17.0
    sst[108:113] = 17.0          # gap of 3 days
    assert len(mhw.detect_events(sst, time, clim, thresh)) == 2


def test_intensity_and_category():
    """Category = floor(max anomaly / (threshold - climatology))."""
    time, sst, clim, thresh = _series()
    sst[200:210] = 17.5          # anomaly 2.5, span 1.0 -> category 2
    ev = mhw.detect_events(sst, time, clim, thresh)[0]
    assert np.isclose(ev.intensity_max, 2.5)
    assert np.isclose(ev.intensity_cumulative, 25.0)
    assert ev.category == 2 and ev.category_name == "Strong"

    sst[:] = 15.0
    sst[200:210] = 19.5          # anomaly 4.5 -> category 4, clipped
    assert mhw.detect_events(sst, time, clim, thresh)[0].category == 4


def test_strict_exceedance():
    """Equal to the threshold is not an exceedance."""
    time, sst, clim, thresh = _series()
    sst[300:320] = 16.0
    assert mhw.detect_events(sst, time, clim, thresh) == []


def test_climatology_is_seasonal():
    """A pure sinusoid should give a climatology that tracks it and a
    threshold above it everywhere."""
    time = pd.date_range("1990-01-01", "2019-12-31", freq="D")
    doy = time.dayofyear.to_numpy()
    rng = np.random.default_rng(0)
    sst = 15 + 4 * np.sin(2 * np.pi * doy / 365.25) + rng.normal(0, 0.5, len(time))
    clim, thresh = mhw.climatology(sst, time, baseline=("1990-01-01", "2019-12-31"))
    assert np.nanmax(clim) - np.nanmin(clim) > 6.0
    assert np.all(thresh > clim)
    assert np.nanmean(sst > thresh) < 0.15   # ~10% by construction


def test_leap_year_doy_alignment():
    """1 March maps to the same DOY slot in leap and non-leap years."""
    t = pd.DatetimeIndex(["2003-03-01", "2004-03-01", "2003-02-28", "2004-02-28"])
    d = mhw.day_of_year_index(t)
    assert d[0] == d[1]
    assert d[2] == d[3]


def test_composite_recovers_planted_response():
    """Plant a negative lagged response and check the composite finds it."""
    n = 4000
    time = pd.date_range("2000-01-01", periods=n, freq="D")
    rng = np.random.default_rng(1)
    anom = rng.normal(0, 0.05, n)
    onsets = np.arange(200, n - 200, 300)
    for t0 in onsets:
        anom[t0 + 10:t0 + 50] -= 0.30
    comp = composite.lag_composite(anom, onsets, n_boot=200)
    s = composite.summarise_composite(comp)
    assert s["direction"] == "down"
    assert s["response_during"] < -0.1
    assert 5 <= s["peak_lag"] <= 55


def test_composite_no_false_positive_on_noise():
    n = 4000
    time = pd.date_range("2000-01-01", periods=n, freq="D")
    rng = np.random.default_rng(2)
    anom = rng.normal(0, 0.15, n)
    onsets = np.arange(200, n - 200, 300)
    s = composite.summarise_composite(
        composite.lag_composite(anom, onsets, n_boot=200))
    assert s["direction"] == "none"


def test_mhw_flag_matches_events():
    time, sst, clim, thresh = _series()
    sst[100:112] = 17.0
    flag = mhw.mhw_day_flag(sst, time, clim, thresh)
    assert flag.sum() == 12 and flag[100] and flag[111] and not flag[112]
