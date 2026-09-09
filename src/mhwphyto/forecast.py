"""Forecasting chlorophyll anomaly at a lead time, with baselines that are
allowed to win.

The single most common way a hackathon ML project loses credibility is
reporting an R-squared with nothing to compare it to. So this module computes
persistence and climatology baselines FIRST, and reports skill as the
fractional reduction in mean squared error relative to the better of the two.
A negative skill score is a legitimate and publishable result.

Validation is a blocked, expanding-window split by calendar year. Random
k-fold on a time series leaks the future into the past through
autocorrelation and will flatter any model by a wide margin.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

DEFAULT_HORIZON = 28  # days ahead


def build_features(
    time: pd.DatetimeIndex,
    sst_anom: np.ndarray,
    chl_anom: np.ndarray,
    mhw_flag: np.ndarray,
    mld: np.ndarray | None = None,
    par: np.ndarray | None = None,
    horizon: int = DEFAULT_HORIZON,
) -> pd.DataFrame:
    """Tabular design matrix for one location. One row per day.

    All predictors are strictly backward-looking, so nothing leaks. The target
    is the chlorophyll log-anomaly `horizon` days in the future.
    """
    time = pd.DatetimeIndex(time)
    df = pd.DataFrame(index=time)
    df["sst_anom"] = sst_anom
    df["chl_anom"] = chl_anom
    df["mhw_flag"] = np.asarray(mhw_flag, dtype=float)
    if mld is not None:
        df["mld"] = mld
    if par is not None:
        df["par"] = par

    for w in (7, 30, 90):
        df[f"sst_anom_mean{w}"] = df["sst_anom"].rolling(w, min_periods=w // 2).mean()
        df[f"chl_anom_mean{w}"] = df["chl_anom"].rolling(w, min_periods=w // 2).mean()

    # cumulative MHW intensity = degC-days accumulated in the recent past.
    # This is the variable that encodes "how much heat stress has built up",
    # which matters more for a lagged ecological response than today's SST.
    warm = df["sst_anom"].clip(lower=0) * df["mhw_flag"]
    for w in (30, 90, 180):
        df[f"mhw_cum{w}"] = warm.rolling(w, min_periods=1).sum()
    df["mhw_days90"] = df["mhw_flag"].rolling(90, min_periods=1).sum()

    if mld is not None:
        df["mld_anom30"] = df["mld"] - df["mld"].rolling(365, min_periods=90).mean()
    if par is not None:
        df["par_mean30"] = df["par"].rolling(30, min_periods=10).mean()

    doy = time.dayofyear.to_numpy()
    df["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)

    for lag in (7, 14, 30):
        df[f"chl_anom_lag{lag}"] = df["chl_anom"].shift(lag)

    df["target"] = df["chl_anom"].shift(-horizon)
    df["persistence"] = df["chl_anom"]      # baseline 1
    df["climatology"] = 0.0                 # baseline 2 (anomaly space)
    return df


@dataclass
class FoldResult:
    fold: int
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    n_test: int
    rmse_model: float
    rmse_persistence: float
    rmse_climatology: float
    mae_model: float
    skill_vs_best_baseline: float
    brier_model: float
    brier_climatology: float


FEATURE_BLOCKLIST = {"target", "persistence", "climatology"}


def _fit_predict(train: pd.DataFrame, test: pd.DataFrame, feats: list[str], seed: int):
    model = HistGradientBoostingRegressor(
        max_depth=4, learning_rate=0.06, max_iter=350,
        min_samples_leaf=40, l2_regularization=1.0, random_state=seed,
    )
    model.fit(train[feats], train["target"])
    return model, model.predict(test[feats])


def blocked_cv(
    df: pd.DataFrame,
    n_folds: int = 4,
    min_train_years: int = 8,
    failure_quantile: float = 0.20,
    seed: int = 0,
) -> tuple[pd.DataFrame, list]:
    """Expanding-window CV by calendar year. Returns (per-fold table, models)."""
    d = df.dropna(subset=["target"]).copy()
    feats = [c for c in d.columns if c not in FEATURE_BLOCKLIST]
    years = np.array(sorted(d.index.year.unique()))
    if len(years) < min_train_years + n_folds:
        raise ValueError(
            f"need >= {min_train_years + n_folds} years of data, got {len(years)}"
        )
    test_years = years[-n_folds:]

    results, models = [], []
    for k, ty in enumerate(test_years):
        train = d[d.index.year < ty]
        test = d[d.index.year == ty]
        if len(train) < 365 * min_train_years or test.empty:
            continue

        model, pred = _fit_predict(train, test, feats, seed)
        models.append(model)

        y = test["target"].to_numpy()
        rmse_m = float(np.sqrt(mean_squared_error(y, pred)))
        rmse_p = float(np.sqrt(mean_squared_error(y, test["persistence"])))
        rmse_c = float(np.sqrt(mean_squared_error(y, test["climatology"])))
        best_base = min(rmse_p, rmse_c)
        skill = 1.0 - (rmse_m ** 2) / (best_base ** 2) if best_base > 0 else np.nan

        # "bloom failure" event framing: is the anomaly in the lower tail?
        thr = float(train["target"].quantile(failure_quantile))
        y_bin = (y < thr).astype(int)
        resid = train["target"] - _fit_predict(train, train, feats, seed)[1]
        sigma = float(resid.std(ddof=1)) or 1.0
        from math import erf, sqrt
        p_model = np.array([0.5 * (1 + erf((thr - m) / (sigma * sqrt(2)))) for m in pred])
        base_rate = float(y_bin.mean()) if len(y_bin) else failure_quantile
        brier_m = float(np.mean((p_model - y_bin) ** 2))
        brier_c = float(np.mean((failure_quantile - y_bin) ** 2))

        results.append(FoldResult(
            fold=k, test_start=test.index[0], test_end=test.index[-1], n_test=len(test),
            rmse_model=rmse_m, rmse_persistence=rmse_p, rmse_climatology=rmse_c,
            mae_model=float(mean_absolute_error(y, pred)),
            skill_vs_best_baseline=skill,
            brier_model=brier_m, brier_climatology=brier_c,
        ).__dict__ | {"failure_base_rate": base_rate})

    return pd.DataFrame(results), models


def permutation_importance(df: pd.DataFrame, model, feats: list[str],
                           test: pd.DataFrame, n_repeat: int = 5, seed: int = 0):
    """Cheap permutation importance -- which predictors actually carry signal."""
    rng = np.random.default_rng(seed)
    y = test["target"].to_numpy()
    base = mean_squared_error(y, model.predict(test[feats]))
    rows = []
    for f in feats:
        losses = []
        for _ in range(n_repeat):
            t = test.copy()
            t[f] = rng.permutation(t[f].to_numpy())
            losses.append(mean_squared_error(y, model.predict(t[feats])))
        rows.append(dict(feature=f, delta_mse=float(np.mean(losses) - base)))
    return pd.DataFrame(rows).sort_values("delta_mse", ascending=False, ignore_index=True)
