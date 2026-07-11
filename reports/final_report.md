# Final Report — Time Series Forecasting for Portfolio Management Optimization

**GMF Investments** | Investment Committee Memo | Final submission — 2026-07-07
Prepared by: Kerod, Mahbubah, Feven

---

## Executive Summary

This memo presents GMF Investments' time series forecasting and portfolio optimization
analysis for a three-asset portfolio: **TSLA** (high-growth equity), **BND** (investment-grade
bonds), and **SPY** (broad market index), using daily data from 2015-01-01 to 2026-06-30.
We (1) cleaned and explored the data, (2) built and compared ARIMA and LSTM forecasting
models for TSLA, (3) generated a 12-month forward forecast with confidence intervals,
(4) used Modern Portfolio Theory to derive an optimal three-asset portfolio, and
(5) backtested that portfolio against a passive 60/40 SPY/BND benchmark. Consistent with
the Efficient Market Hypothesis, we find limited exploitable signal in TSLA's historical
prices alone; our recommendation therefore treats the forecast as one input among several,
not a standalone trading signal.

---

## Task 1 — Data Preprocessing and Exploratory Analysis

### Data extraction and cleaning
Daily OHLCV data for TSLA, BND, and SPY were fetched via `yfinance` for 2015-01-01 through
2026-06-30 (`src/data_loader.py`), returning 2,888 trading days per ticker. All columns
loaded with correct numeric/datetime dtypes and no nulls were present in the returned
trading-day rows. Prices were reindexed onto a common business-day calendar and any
resulting gaps were **forward-filled** (not interpolated), since a price is genuinely
unchanged until the next trade occurs. Daily percentage returns were computed from
adjusted close prices as the basis for all volatility/risk calculations; price levels
were separately min-max normalized purely for cross-asset plotting.

### Key EDA insights
- **Trend**: TSLA shows a strong upward trend from 2015-2026 with deep multi-month
  drawdowns (notably 2022); BND is essentially flat and low-variance; SPY shows steady,
  moderate appreciation.
- **Volatility clustering**: TSLA's 30-day rolling volatility swings between calm and
  turbulent regimes (e.g. around 2020, 2022) rather than staying constant.
- **Outliers**: TSLA had 50 trading days with |return| > 3σ (most extreme: +22.7% on
  2025-04-09); BND had 35 (most extreme: -5.4%, 2020-03-12); SPY had 42 (most extreme:
  -10.9%, 2020-03-16). Extreme days cluster around the March 2020 COVID shock across all
  three assets, and around isolated earnings/macro events for TSLA specifically.
- **Correlation**: TSLA-SPY daily return correlation ≈ 0.49 (moderate positive); BND is
  nearly uncorrelated with both equities (≈0.06 with TSLA, ≈0.12 with SPY), confirming its
  role as a portfolio diversifier.

### Stationarity testing (Augmented Dickey-Fuller)

| Series | ADF statistic | p-value | Stationary at 5%? |
|---|---:|---:|:---:|
| TSLA — Close (level) | -1.04 | 0.739 | No |
| TSLA — Daily return | -55.15 | <0.001 | Yes |
| BND — Close (level) | -1.12 | 0.708 | No |
| BND — Daily return | -21.28 | <0.001 | Yes |
| SPY — Close (level) | 1.43 | 0.997 | No |
| SPY — Daily return | -15.70 | <0.001 | Yes |

All three price *levels* fail to reject the ADF null (non-stationary, random-walk-like),
consistent with the Efficient Market Hypothesis; all daily *return* series are stationary.
This confirms a first difference (**d = 1**) is required before ARIMA can be validly
applied to price levels.

### Risk metrics (annualized, 252 trading days/year, 0% risk-free rate)

| Asset | Annualized Return | Annualized Volatility | Historical VaR (95%, daily) | Sharpe Ratio |
|---|---:|---:|---:|---:|
| TSLA | 43.8% | 56.1% | 5.11% | 0.78 |
| BND | 1.9% | 5.2% | 0.47% | 0.37 |
| SPY | 13.9% | 17.3% | 1.64% | 0.80 |

TSLA carries by far the highest return *and* risk; BND anchors the portfolio with minimal
downside; SPY sits between the two on every metric, as expected for a diversified index.

See [notebooks/1.0-data-extraction-and-eda.ipynb](../notebooks/1.0-data-extraction-and-eda.ipynb).

---

## Task 2 — Forecasting Models: ARIMA vs. LSTM

### Methodology
The TSLA adjusted-close series was split **chronologically** (no shuffling): train =
2015-01-02 to 2024-12-31 (2,608 observations), test = 2025-01-01 to 2026-06-29
(391 observations).

- **ARIMA**: `pmdarima.auto_arima` (non-seasonal, stepwise, max_p=5/max_q=5/max_d=2)
  selected **ARIMA(0,1,0)** — a driftless random walk on price level — forecast
  generated statically over the full 391-day test horizon.
- **LSTM**: 60-day lookback window, two stacked LSTM layers (50 units each) with dropout,
  dense output layer, trained with early stopping on validation loss. Test-set predictions
  use the *true* preceding 60 days of price at each step (one-step-ahead evaluation),
  distinct from the genuinely recursive forecast used in Task 3.

### Model comparison

| Model | MAE | RMSE | MAPE (%) | Forecast mode |
|---|---:|---:|---:|---|
| ARIMA(0,1,0) | 54.15 | 70.20 | 17.11 | Static, unconditional, 391-day-ahead |
| LSTM (60-day window) | 17.22 | 21.34 | 4.60 | One-step-ahead, conditioned on true history |

**Discussion:** These are not a perfectly apples-to-apples comparison — the ARIMA number
reflects a genuinely unconditional 18-month-ahead static forecast, while the LSTM number
reflects one-step-ahead predictions re-conditioned on true history at every step (an
inherently easier task). The LSTM's lower error here is therefore expected and does not by
itself prove greater long-horizon predictive skill. What both models agree on is that TSLA
price *levels* are close to a random walk with little linear autocorrelation structure
(ARIMA's own order search found no AR/MA terms beyond differencing) — consistent with the
Efficient Market Hypothesis. We select the **LSTM** to carry forward into Task 3, since it
achieves lower error under the available comparable protocol and its learned
representation is a reasonable basis for recursive forecasting.

See [notebooks/2.0-arima-forecasting.ipynb](../notebooks/2.0-arima-forecasting.ipynb) and
[notebooks/2.1-lstm-forecasting.ipynb](../notebooks/2.1-lstm-forecasting.ipynb).

---

## Task 3 — Forecasting Future Market Trends

Using the LSTM, we generated a genuine 12-month (252 trading day) **recursive** forecast
(each day's prediction is fed back in as the next day's context, since no real future data
exists to condition on), with a 95% confidence band derived via Monte Carlo simulation
calibrated to TSLA's historical drift/volatility and re-centered (multiplicatively, so the
band can never cross zero) on the LSTM's own trajectory.

- **Last observed price** (2026-06-29): $411.84
- **12-month point forecast**: $277.90 (implied return: **-32.5%**)
- **95% CI at 12 months**: [$97.57, $846.63]

**Trend analysis — a fixed-point artifact, not a bearish thesis:** Inspecting the forecast
month by month reveals its real shape: price drops from $411.84 toward the high-$270s over
the first ~2 months, then **converges to a fixed point (~$277.90) and stays essentially
flat for the remaining ~10 months**. This is a well-documented failure mode of naive
recursive one-step LSTM forecasting, not a genuine 12-month bearish signal. Because each
step beyond the first conditions on an increasingly model-generated (not real) window, the
recursion behaves like an iterated function converging to an attractor — once consecutive
predictions stop changing, the model simply reproduces the same value indefinitely. This
is the same underlying limitation the ARIMA(0,1,0) baseline exhibits when its own forecast
collapses to a flat drift line, reached by a different mechanism. We treat the -32.5%
point estimate with real skepticism rather than at face value; it is driven almost
entirely by the early collapse, not a considered 12-month bearish view.

The 95% confidence interval widens sharply with the forecast horizon — from ~$57 wide at
day 1 to nearly $750 wide by month 12 ($97.57–$846.63) — which is itself a strong,
independent signal that the point forecast should not be read with confidence at long
horizons.

**Opportunities:** the forecast's early weeks (before the fixed-point collapse) are the
most behaviorally grounded and should be weighted far more heavily than the flat 3-12
month tail; a materially negative point forecast is itself a risk-management signal
against increasing TSLA exposure on momentum grounds.

**Risks:** TSLA's ~56% annualized historical volatility means realized outcomes can
deviate sharply from the point forecast in either direction; the width of the 12-month
band implies material upside is roughly as plausible as the modeled downside; recursive
forecasting compounds any systematic one-step bias over 252 steps; and the observed
fixed-point convergence is itself evidence the model has stopped incorporating new
information, not evidence that TSLA's price will genuinely be flat for 10 months.

**Critical assessment:** both the collapse-to-fixed-point behavior of the point forecast
and the rapidly widening confidence band are, together, a quantitative admission that this
model's long-horizon certainty is low. This is precisely why the forecast is used in
Task 4 as only one input to portfolio construction (and, as it turns out, an input the
optimizer chose to largely disregard), not as a standalone trading signal.

See [notebooks/3.0-future-forecast.ipynb](../notebooks/3.0-future-forecast.ipynb).

---

## Task 4 — Portfolio Optimization (Modern Portfolio Theory)

**Expected returns:** TSLA's expected return is derived from the Task 3 LSTM 12-month
forecast (**-32.5%** annualized); BND and SPY use their historical mean daily returns,
annualized (1.9% and 13.9% respectively, per Task 1). **Covariance matrix**: sample
covariance of historical daily returns for all three assets, annualized.

We swept the efficient frontier and identified two key portfolios:

| Portfolio | TSLA weight | BND weight | SPY weight | Expected Return | Volatility | Sharpe Ratio |
|---|---:|---:|---:|---:|---:|---:|
| **Max Sharpe (recommended)** | 0.0% | 54.7% | 45.3% | 7.35% | 8.66% | 0.85 |
| Min Volatility | 0.0% | 94.5% | 5.5% | 2.58% | 5.13% | 0.50 |

Both optimal portfolios allocate **0% to TSLA**. Given TSLA's negative forecasted return
(Task 3) combined with its by-far-highest volatility (~56% annualized, Task 1) and only
moderate diversification benefit (≈0.49 return correlation with SPY, Task 1), no efficient
portfolio wants TSLA exposure at this input — a direct, mechanical, and defensible
consequence of feeding the model's own forecast into the optimizer honestly, rather than
overriding it.

**Recommendation:** We recommend the **Maximum Sharpe Ratio (tangency) portfolio**
(54.7% BND / 45.3% SPY, Sharpe 0.85). It maximizes risk-adjusted return rather than raw
return or raw risk in isolation, which is the standard institutional default absent a
client-specific risk mandate, and aligns with GMF's stated goal of balancing risk
minimization against capitalizing on opportunity. The Minimum Volatility portfolio
(94.5% BND / 5.5% SPY, Sharpe 0.50) remains the appropriate alternative for a more
risk-averse client mandate, at the cost of materially lower expected return. We
explicitly flag that this recommendation is conditional on the Task 3 TSLA view — a
more optimistic TSLA forecast (e.g. from a model without the recursive fixed-point
artifact) would likely restore some TSLA allocation; this sensitivity is itself a useful
finding, illustrating how directly the forecast quality upstream drives the portfolio
decision downstream.

See [notebooks/4.0-portfolio-optimization.ipynb](../notebooks/4.0-portfolio-optimization.ipynb)
for the Efficient Frontier plot and covariance heatmap.

---

## Task 5 — Strategy Backtesting

We simulated the Max Sharpe portfolio (monthly rebalanced back to target weights) over the
last 12 months of the dataset (2025-07-01 to 2026-06-29, out-of-sample relative to model
training) against a static 60% SPY / 40% BND benchmark.

| Metric | Strategy (Max Sharpe) | Benchmark (60/40) |
|---|---:|---:|
| Total Return | 11.64% | 14.21% |
| Annualized Return | 11.31% | 13.80% |
| Annualized Volatility | 6.48% | 7.96% |
| Sharpe Ratio | **1.69** | 1.66 |
| Max Drawdown | **-4.91%** | -5.82% |

**Conclusion:** No unambiguous win either way. The 60/40 benchmark delivered a higher
total and annualized return over this specific window — it holds more SPY (60% vs. the
strategy's 45.3%), and this backtest period was a rising market in which SPY outperformed
BND. The Max Sharpe strategy edged out the benchmark on risk-adjusted terms — a slightly
higher Sharpe Ratio and a smaller maximum drawdown — because it held more BND (54.7% vs.
40%) and zero TSLA, both of which reduced volatility. In short: **the strategy was the
steadier ride; the benchmark was the better raw return, in this particular year.** This
is a direct, mechanical consequence of the Task 3 forecast: because the LSTM's 12-month
TSLA view was materially negative, Task 4's optimizer allocated 0% to TSLA and leaned into
BND for stability — a defensible, risk-averse allocation given the inputs, but one that
also gave up the upside SPY captured this year. A single 12-month backtest, with a narrow
Sharpe/drawdown edge in one direction, is far too small a margin over too short and
singular a window to call this proof of durable outperformance in either direction.

**Limitations:** the Task 4 covariance/return estimates were computed over the full
historical dataset, which overlaps with this backtest window (look-ahead bias); a single
12-month window is one draw from a highly uncertain distribution of market paths and
cannot establish statistical significance; no transaction costs, taxes, or slippage are
modeled; and the TSLA "view" driving the strategy allocation itself carries wide
uncertainty per Task 3's confidence interval. A production deployment would require
walk-forward validation across many non-overlapping historical windows.

See [notebooks/5.0-backtesting.ipynb](../notebooks/5.0-backtesting.ipynb).

---

## Overall Conclusion

This project demonstrates the full pipeline from raw price data to a backtested,
model-informed portfolio recommendation. The central, recurring finding across every task
is consistent with the Efficient Market Hypothesis: TSLA's price history contains limited
robust, exploitable structure, and both classical (ARIMA) and deep learning (LSTM) models
largely rediscover this fact rather than refuting it. Forecast uncertainty compounds
quickly beyond a few weeks, and any portfolio decision built on these forecasts should
treat them as one input among several — combined with historical risk data on BND/SPY and
sound portfolio construction (MPT) — rather than as precise, standalone predictions. GMF's
recommended allocation reflects a risk-adjusted (Max Sharpe) balance across all three
assets, validated (with caveats) by a one-year backtest against a passive benchmark.
