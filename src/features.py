"""Feature engineering and risk-metric helpers shared by the analysis notebooks."""

from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller

TRADING_DAYS_PER_YEAR = 252


def daily_returns(prices: pd.Series) -> pd.Series:
    """Simple daily percentage returns, first (undefined) observation dropped."""
    return prices.pct_change().dropna()


def rolling_stats(returns: pd.Series, window: int = 30) -> pd.DataFrame:
    """Rolling mean and standard deviation of returns over the given window."""
    return pd.DataFrame(
        {
            "rolling_mean": returns.rolling(window).mean(),
            "rolling_std": returns.rolling(window).std(),
        }
    )


def detect_outliers(returns: pd.Series, n_std: float = 3.0) -> pd.Series:
    """Return the subset of `returns` more than `n_std` standard deviations from the mean."""
    mean, std = returns.mean(), returns.std()
    mask = (returns - mean).abs() > n_std * std
    return returns[mask].sort_values()


def adf_test(series: pd.Series, name: str = "") -> dict:
    """Run the Augmented Dickey-Fuller test and return a summary dict."""
    series = series.dropna()
    result = adfuller(series, autolag="AIC")
    return {
        "series": name,
        "adf_statistic": result[0],
        "p_value": result[1],
        "n_lags": result[2],
        "n_obs": result[3],
        "critical_values": result[4],
        "is_stationary_at_5pct": result[1] < 0.05,
    }


def historical_var(returns: pd.Series, confidence: float = 0.95) -> float:
    """Historical (empirical) Value at Risk at the given confidence level.

    Returned as a positive number representing the daily loss threshold.
    """
    return -np.percentile(returns.dropna(), (1 - confidence) * 100)


def sharpe_ratio(
    returns: pd.Series, risk_free_rate: float = 0.0, periods_per_year: int = TRADING_DAYS_PER_YEAR
) -> float:
    """Annualized Sharpe ratio from a series of periodic (daily) returns."""
    excess = returns - risk_free_rate / periods_per_year
    if excess.std() == 0:
        return np.nan
    return (excess.mean() / excess.std()) * np.sqrt(periods_per_year)


def annualized_return(returns: pd.Series, periods_per_year: int = TRADING_DAYS_PER_YEAR) -> float:
    return returns.mean() * periods_per_year


def annualized_volatility(returns: pd.Series, periods_per_year: int = TRADING_DAYS_PER_YEAR) -> float:
    return returns.std() * np.sqrt(periods_per_year)
