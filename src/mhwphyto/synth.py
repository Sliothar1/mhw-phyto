"""Synthetic ocean generator: SST and chlorophyll cubes with a KNOWN
marine-heatwave -> phytoplankton response, so the analysis pipeline can be
validated before any real data arrives.

Why bother
----------
The whole project rests on estimating a lagged, sign-ambiguous response of
chlorophyll to marine heatwaves. If the pipeline cannot recover a response
we planted ourselves, any number it produces from CMEMS is uninterpretable.
This module is the test oracle, and it is also a good slide: "here is our
method recovering a known answer".

Three regimes are planted:
  stratified    : nutrient starvation -> chlorophyll DOWN, long lag
  light_limited : warming relieves light/mixing limits -> chlorophyll UP, short lag
  insensitive   : no response (control, guards against false positives)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr

REGIMES = ("stratified", "light_limited", "insensitive")

# planted ground truth: (response amplitude in log10(chl) per degC, lag in days)
REGIME_TRUTH = {
    "stratified":    (-0.16, 21),
    "light_limited": (+0.11, 7),
    "insensitive":   (0.0, 0),
}


def _gamma_kernel(lag_peak: int, length: int = 120, shape: float = 3.0) -> np.ndarray:
    """Causal response kernel peaking near `lag_peak` days. Normalised to sum 1."""
    if lag_peak <= 0:
        k = np.zeros(length)
        k[0] = 1.0
        return k
    t = np.arange(length)
    scale = lag_peak / max(shape - 1.0, 1e-6)
    k = (t ** (shape - 1)) * np.exp(-t / scale)
    s = k.sum()
    return k / s if s > 0 else k


def _ar1(n: int, rho: float, sigma: float, rng: np.random.Generator) -> np.ndarray:
    e = rng.normal(0.0, sigma, size=n)
    x = np.zeros(n)
    for t in range(1, n):
        x[t] = rho * x[t - 1] + e[t]
    return x


def make_dataset(
    start: str = "1998-01-01",
    end: str = "2023-12-31",
    nlat: int = 6,
    nlon: int = 6,
    lat0: float = 40.0,
    lon0: float = -10.0,
    resolution: float = 0.25,
    warming_per_decade: float = 0.22,
    seed: int = 20261005,
) -> xr.Dataset:
    """Build a daily (time, lat, lon) cube of SST, chlorophyll and forcings."""
    rng = np.random.default_rng(seed)
    time = pd.date_range(start, end, freq="D")
    nt = len(time)
    doy = time.dayofyear.to_numpy()
    yearfrac = (time.year.to_numpy() - time.year.min()) + doy / 365.25

    lat = lat0 + np.arange(nlat) * resolution
    lon = lon0 + np.arange(nlon) * resolution

    # regime map: latitude bands, with an insensitive control stripe
    regime_id = np.zeros((nlat, nlon), dtype=int)
    for j in range(nlat):
        for i in range(nlon):
            if i == nlon - 1:
                regime_id[j, i] = 2                      # insensitive control column
            else:
                regime_id[j, i] = 0 if j < nlat // 2 else 1
    regime_name = np.array(REGIMES, dtype=object)[regime_id]

    sst = np.empty((nt, nlat, nlon))
    chl = np.empty((nt, nlat, nlon))
    mld = np.empty((nt, nlat, nlon))
    par = np.empty((nt, nlat, nlon))

    seasonal_sst_amp = 3.6
    for j in range(nlat):
        for i in range(nlon):
            base = 16.0 - 0.35 * (lat[j] - lat0)
            seas = seasonal_sst_amp * -np.cos(2 * np.pi * (doy - 15) / 365.25)
            trend = warming_per_decade * yearfrac / 10.0
            noise = _ar1(nt, rho=0.93, sigma=0.45, rng=rng)

            # inject discrete warm blobs so events are unambiguous
            blob = np.zeros(nt)
            n_blob = rng.poisson(0.9 * (nt / 365.25))
            for _ in range(int(n_blob)):
                t0 = rng.integers(0, nt - 1)
                dur = int(rng.integers(8, 95))
                amp = float(rng.gamma(2.4, 0.75))
                w = np.arange(dur)
                blob[t0:t0 + dur] += amp * np.sin(np.pi * w / dur)[: max(0, min(dur, nt - t0))]

            s = base + seas + trend + noise + blob
            sst[:, j, i] = s

            # forcings that a real model would also use as predictors
            mld[:, j, i] = np.clip(
                60 - 34 * -np.cos(2 * np.pi * (doy - 15) / 365.25)
                - 6.0 * (s - (base + seas)) + rng.normal(0, 4, nt),
                8, 220,
            )
            par[:, j, i] = np.clip(
                28 + 20 * -np.cos(2 * np.pi * (doy - 172) / 365.25) + rng.normal(0, 2.5, nt),
                1, None,
            )

    # chlorophyll: seasonal bloom in log10 space + lagged response to warm anomaly
    clim_sst = sst.mean(axis=0, keepdims=True)
    for j in range(nlat):
        for i in range(nlon):
            reg = regime_name[j, i]
            gain, lag = REGIME_TRUTH[reg]

            spring = np.exp(-0.5 * ((doy - 105) / 26.0) ** 2)
            autumn = 0.45 * np.exp(-0.5 * ((doy - 285) / 22.0) ** 2)
            log_chl = -0.75 + 0.80 * spring + 0.80 * autumn

            warm = np.maximum(sst[:, j, i] - clim_sst[0, j, i], 0.0)
            if gain != 0.0:
                k = _gamma_kernel(lag)
                resp = np.convolve(warm, k, mode="full")[:len(warm)]
                log_chl = log_chl + gain * resp

            log_chl = log_chl + _ar1(len(time), rho=0.80, sigma=0.10, rng=rng)
            chl[:, j, i] = 10 ** log_chl

    ds = xr.Dataset(
        data_vars=dict(
            sst=(("time", "lat", "lon"), sst, {"units": "degree_Celsius",
                                               "long_name": "sea surface temperature"}),
            chl=(("time", "lat", "lon"), chl, {"units": "mg m-3",
                                               "long_name": "chlorophyll-a concentration"}),
            mld=(("time", "lat", "lon"), mld, {"units": "m",
                                               "long_name": "mixed layer depth"}),
            par=(("time", "lat", "lon"), par, {"units": "E m-2 d-1",
                                               "long_name": "photosynthetically available radiation"}),
            regime=(("lat", "lon"), regime_name.astype(str),
                    {"long_name": "planted response regime (ground truth)"}),
        ),
        coords=dict(time=time, lat=lat, lon=lon),
        attrs=dict(
            title="Synthetic MHW / phytoplankton test cube",
            source="mhwphyto.synth.make_dataset",
            warning="SYNTHETIC DATA - not observations. For pipeline validation only.",
            ground_truth=str(REGIME_TRUTH),
        ),
    )
    return ds
