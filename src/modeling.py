"""Shared helpers for the ARIMA/LSTM forecasting notebooks (Tasks 2-3)."""

from __future__ import annotations

import numpy as np
import pandas as pd


def make_windows(series: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    """Turn a 1-D array into overlapping (X, y) supervised sequences.

    X[i] = series[i : i+window], y[i] = series[i+window]
    """
    X, y = [], []
    for i in range(len(series) - window):
        X.append(series[i : i + window])
        y.append(series[i + window])
    return np.array(X), np.array(y)


def monte_carlo_price_paths(
    last_price: float,
    daily_drift: float,
    daily_vol: float,
    horizon: int,
    n_sims: int = 2000,
    seed: int = 42,
) -> np.ndarray:
    """Simulate `n_sims` future price paths of length `horizon` via geometric
    Brownian motion calibrated to a model's implied daily drift/volatility.

    Used to derive Monte Carlo confidence bands around a point forecast (e.g. an
    LSTM's recursive forecast, which has no native analytic confidence interval).
    Returns an array of shape (n_sims, horizon).
    """
    rng = np.random.default_rng(seed)
    shocks = rng.normal(loc=daily_drift, scale=daily_vol, size=(n_sims, horizon))
    log_paths = np.cumsum(shocks, axis=1)
    paths = last_price * np.exp(log_paths)
    return paths


def summarize_paths(paths: np.ndarray, index: pd.DatetimeIndex, alpha: float = 0.05) -> pd.DataFrame:
    """Collapse simulated price paths into median + (alpha) confidence band."""
    lower_q, upper_q = 100 * (alpha / 2), 100 * (1 - alpha / 2)
    return pd.DataFrame(
        {
            "median": np.median(paths, axis=0),
            "lower": np.percentile(paths, lower_q, axis=0),
            "upper": np.percentile(paths, upper_q, axis=0),
        },
        index=index,
    )
