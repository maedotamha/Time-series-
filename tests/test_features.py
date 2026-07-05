import numpy as np
import pandas as pd

from src.features import (
    annualized_return,
    annualized_volatility,
    daily_returns,
    detect_outliers,
    historical_var,
    rolling_stats,
    sharpe_ratio,
)


def test_daily_returns_basic():
    prices = pd.Series([100.0, 110.0, 99.0])
    returns = daily_returns(prices)
    assert len(returns) == 2
    assert np.isclose(returns.iloc[0], 0.10)
    assert np.isclose(returns.iloc[1], -0.10)


def test_rolling_stats_shape():
    returns = pd.Series(np.random.normal(0, 0.01, 100))
    stats = rolling_stats(returns, window=10)
    assert list(stats.columns) == ["rolling_mean", "rolling_std"]
    assert stats["rolling_mean"].isna().sum() == 9


def test_detect_outliers_flags_extreme_value():
    returns = pd.Series([0.001] * 50 + [0.5])
    outliers = detect_outliers(returns, n_std=3.0)
    assert 0.5 in outliers.values


def test_historical_var_positive_for_lossy_returns():
    returns = pd.Series(np.random.normal(-0.001, 0.02, 1000))
    var = historical_var(returns, confidence=0.95)
    assert var > 0


def test_sharpe_ratio_zero_for_zero_mean_returns():
    returns = pd.Series([0.01, -0.01] * 100)
    ratio = sharpe_ratio(returns, risk_free_rate=0.0)
    assert np.isclose(ratio, 0.0, atol=1e-8)


def test_annualized_return_and_volatility_scale_correctly():
    returns = pd.Series([0.001] * 252)
    assert np.isclose(annualized_return(returns), 0.252)
    zero_vol_returns = pd.Series([0.001] * 252)
    assert np.isclose(annualized_volatility(zero_vol_returns), 0.0, atol=1e-8)
