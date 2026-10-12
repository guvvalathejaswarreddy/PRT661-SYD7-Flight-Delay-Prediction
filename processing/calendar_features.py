"""Leakage-safe rolling delay rates and training-window graph features (Assessment 4).

These are the same definitions as sections 16-17 of the analysis notebook, in a form the pipeline and the tests can import.

* rolling rate: flight-weighted delay rate over the SEVEN CALENDAR DAYS strictly before the flight date, [d-7, d-1].
  The Assessment 2 code averaged the daily rates of the previous seven *active* days; the difference is measured in the notebook.
* graph features (hub tier, out-degree centrality) are computed from the training years only.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

DATA_START = np.datetime64("2018-01-01")
DATA_END = np.datetime64("2023-12-31")
N_DAYS = int((DATA_END - DATA_START).astype(int)) + 1          # 2,191 calendar days


def day_index(dates: pd.Series) -> np.ndarray:
    """Calendar-day index (0 = 2018-01-01)."""
    t = (pd.to_datetime(dates).values.astype("datetime64[D]") - DATA_START).astype(np.int64)
    if t.min() < 0 or t.max() >= N_DAYS:
        raise ValueError("flight_date outside 2018-2023")
    return t


def daily_counts(codes: np.ndarray, t: np.ndarray, delayed: np.ndarray, n_keys: int):
    """Dense [key x day] arrays: flights N and delayed flights D."""
    flat = codes.astype(np.int64) * N_DAYS + t
    N = np.bincount(flat, minlength=n_keys * N_DAYS).reshape(n_keys, N_DAYS)
    D = np.bincount(flat[delayed == 1], minlength=n_keys * N_DAYS).reshape(n_keys, N_DAYS)
    return N, D


def rolling_rate(N: np.ndarray, D: np.ndarray, window: int = 7):
    """Flight-weighted delay rate over days [t-window, t-1]; NaN when the window holds no flights."""
    z = np.zeros((N.shape[0], 1), dtype=np.int64)
    CN, CD = np.concatenate([z, np.cumsum(N, axis=1)], axis=1), np.concatenate([z, np.cumsum(D, axis=1)], axis=1)
    t = np.arange(N.shape[1])
    lo = np.maximum(t - window, 0)
    n, d = CN[:, t] - CN[:, lo], CD[:, t] - CD[:, lo]
    return np.where(n > 0, d / np.maximum(n, 1), np.nan).astype(np.float32), n


def add_rolling_rates(df: pd.DataFrame, train_years, window: int = 7) -> pd.DataFrame:
    """Adds carrier_delay_rate_7d and origin_delay_rate_7d (NaN-free: first days use the training-window mean)."""
    t = day_index(df["flight_date"])
    y = df["arr_del15"].to_numpy()
    fill = float(df.loc[df["year"].isin(train_years), "arr_del15"].mean())
    out = df.copy()
    for key, col in (("carrier", "carrier_delay_rate_7d"), ("origin", "origin_delay_rate_7d")):
        cat = df[key].astype("category")
        N, D = daily_counts(cat.cat.codes.to_numpy(), t, y, len(cat.cat.categories))
        rate, _ = rolling_rate(N, D, window)
        v = rate[cat.cat.codes.to_numpy(), t]
        out[col] = np.where(np.isnan(v), fill, v).astype("float32")
    return out


def graph_features(df: pd.DataFrame, train_years):
    """Out-degree centrality (0-1) and hub tier per origin, from the training years only."""
    tr = df.loc[df["year"].isin(train_years), ["origin", "destination"]]
    degree = tr.groupby("origin", observed=True)["destination"].nunique()
    span = degree.max() - degree.min()
    degree_norm = (degree - degree.min()) / span if span else degree * 0.0
    volume = tr["origin"].value_counts()
    volume = volume[volume > 0]
    q90, q60 = volume.quantile(0.90), volume.quantile(0.60)
    tier = volume.apply(lambda v: "major" if v >= q90 else ("medium" if v >= q60 else "small"))
    return degree_norm, tier


def add_graph_features(df: pd.DataFrame, train_years) -> pd.DataFrame:
    degree, tier = graph_features(df, train_years)
    out = df.copy()
    out["origin_degree_centrality"] = out["origin"].map(degree).fillna(0.0).astype("float32")      # unseen airport -> 0
    out["origin_hub_tier"] = out["origin"].map(tier).fillna("small").astype("category")             # unseen airport -> small
    return out
