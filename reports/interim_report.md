# Interim Report — Time Series Forecasting for Portfolio Management Optimization

**GMF Investments** | Interim submission — 2026-07-05

## 1. Data extraction and cleaning

- Fetched daily OHLCV data for **TSLA**, **BND**, and **SPY** via `yfinance` for
  2015-01-01 through 2026-06-30 (`src/data_loader.py`). Yahoo Finance returned
  2,888 trading days per ticker, from 2015-01-02 through 2026-06-29.
- All columns had correct numeric/datetime dtypes on load; no null values were present
  within the returned trading-day rows.
- Prices were reindexed onto a common business-day calendar across all three tickers so
  they can be merged and compared on the same date axis, and any resulting gaps
  (e.g. asset-specific holidays) were **forward-filled** rather than interpolated — a
  price is genuinely unchanged until the next trade occurs, so forward-fill reflects
  reality, whereas interpolation would fabricate price movement that never happened.
- Daily percentage returns were computed from adjusted close prices as the basis for all
  volatility and risk calculations. Price levels were also min-max normalized purely for
  plotting three series with very different scales (TSLA ~$400 vs. BND ~$70) on one
  chart; modeling elsewhere uses actual price/return values.

## 2. Key EDA insights

- **Trend**: TSLA shows a strong overall upward trend from 2015–2026 punctuated by deep,
  multi-month drawdowns (notably 2022), consistent with a high-growth/high-volatility
  profile. BND is essentially flat with low variance (bond price stability); SPY shows
  steady, moderate long-run appreciation.
- **Volatility clustering**: TSLA's 30-day rolling volatility swings between calm and
  turbulent regimes rather than staying constant — periods of large moves cluster
  together (e.g. around 2020 and 2022), a well-known feature of equity returns that a
  plain ARIMA model does not explicitly capture.
- **Outliers**: each asset has a handful of trading days with returns beyond 3 standard
  deviations from its mean — for TSLA these coincide with earnings surprises and
  macro shocks; for BND and SPY they cluster around broad market-stress events.
- **Correlation**: TSLA and SPY show positive but moderate return correlation; BND is
  close to uncorrelated (or mildly negatively correlated) with the equities, consistent
  with its role as a diversifying, stabilizing asset in the portfolio.

See [notebooks/1.0-data-extraction-and-eda.ipynb](../notebooks/1.0-data-extraction-and-eda.ipynb)
for the full code and all plots.

## 3. Stationarity testing (Augmented Dickey-Fuller)

| Series | ADF statistic | p-value | Stationary at 5%? |
|---|---:|---:|:---:|
| TSLA — Close (level) | -1.04 | 0.739 | No |
| TSLA — Daily return | -55.15 | <0.001 | Yes |
| BND — Close (level) | -1.12 | 0.708 | No |
| BND — Daily return | -21.28 | <0.001 | Yes |
| SPY — Close (level) | 1.43 | 0.997 | No |
| SPY — Daily return | -15.70 | <0.001 | Yes |

**Interpretation:** all three price *levels* fail to reject the ADF null hypothesis of a
unit root — they are non-stationary, exhibiting a time-varying mean/trend consistent with
a random walk, as the Efficient Market Hypothesis would predict. All three daily *return*
series strongly reject the null and are stationary. This confirms a first difference
(**d = 1**) is required before ARIMA can be validly applied to price levels — equivalently,
returns can be modeled directly with d = 0.

## 4. Volatility and risk metrics

Annualized from daily returns (252 trading days/year), risk-free rate assumed 0%:

| Asset | Annualized Return | Annualized Volatility | Historical VaR (95%, daily) | Sharpe Ratio |
|---|---:|---:|---:|---:|
| TSLA | 43.8% | 56.1% | 5.11% | 0.78 |
| BND | 1.9% | 5.2% | 0.47% | 0.37 |
| SPY | 13.9% | 17.3% | 1.64% | 0.80 |

TSLA carries by far the highest return *and* the highest risk (volatility and VaR) of the
three assets — a 95% historical VaR of 5.1% means TSLA's daily loss is not expected to
exceed 5.1% on 95% of trading days, roughly 10x BND's threshold. BND anchors the portfolio
with minimal downside risk but a correspondingly small return contribution. SPY sits
between the two on every metric, as expected for a broad, diversified index.

## 5. Progress on Task 2 — ARIMA baseline

- Split the TSLA adjusted-close series chronologically: **train** = 2015-01-02 to
  2024-12-31 (2,608 observations), **test** = 2025-01-01 to 2026-06-29 (391 observations).
  No shuffling — temporal order preserved throughout.
- Used `pmdarima.auto_arima` (stepwise search, non-seasonal, `max_p=5, max_q=5, max_d=2`)
  to select the order. The search converged on **ARIMA(0,1,0) with no drift** — i.e. a
  driftless random walk on the price level. This is a meaningful result on its own: it
  says the best-fitting classical linear model finds no exploitable autocorrelation
  structure in TSLA's past prices beyond the previous day's value, which is exactly what
  the Efficient Market Hypothesis predicts.
- Forecasting the full 391-day test horizon in one static pass and scoring against
  actuals gives **MAE = 54.15, RMSE = 70.20, MAPE = 17.11%**. Errors of this magnitude
  over an 18-month unconditional forecast are expected: a driftless random-walk forecast
  is a flat line at the last training price, so all of the growth/decline TSLA actually
  experienced during the test window necessarily shows up as forecast error. This
  baseline is the reference point against which the LSTM model (planned next) will be
  compared.

See [notebooks/2.0-arima-forecasting.ipynb](../notebooks/2.0-arima-forecasting.ipynb).

## 6. Next steps (toward final submission)

- Build and tune an LSTM model on the same chronological split for direct comparison
  with the ARIMA baseline (Task 2).
- Generate 6–12 month forward forecasts with confidence intervals from the
  best-performing model (Task 3).
- Feed the TSLA forecast into a Modern Portfolio Theory optimization alongside BND/SPY
  historical returns, and produce the Efficient Frontier (Task 4).
- Backtest the resulting optimal portfolio against a static 60/40 SPY/BND benchmark
  (Task 5).
