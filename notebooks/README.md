# Notebooks

Run in order; each later notebook depends on artifacts saved by earlier ones under
`data/processed/`.

- `1.0-data-extraction-and-eda.ipynb` — Task 1: fetch TSLA/BND/SPY data, clean it, run EDA, stationarity tests, and risk metrics.
- `2.0-arima-forecasting.ipynb` — Task 2: chronological train/test split and ARIMA modeling for TSLA.
- `2.1-lstm-forecasting.ipynb` — Task 2: LSTM model on the same split, compared against ARIMA. Saves the trained model/scaler to `data/processed/`.
- `3.0-future-forecast.ipynb` — Task 3: genuine 12-month recursive LSTM forecast with Monte Carlo confidence intervals, trend/opportunity/risk analysis.
- `4.0-portfolio-optimization.ipynb` — Task 4: Efficient Frontier, Max Sharpe / Min Volatility portfolios via PyPortfolioOpt.
- `5.0-backtesting.ipynb` — Task 5: backtest the recommended portfolio against a static 60/40 SPY/BND benchmark.

Notebooks 2.1 through 5.0 were generated from the scripts in `../scripts/_build_*.py`
(kept for reproducibility/auditability of the notebook structure) and then executed via
`jupyter nbconvert --execute` against the `timeseries-venv` kernel.
