# Time Series Forecasting for Portfolio Management Optimization

GMF Investments case study: forecast Tesla (TSLA) price behavior, compare it against
Vanguard Total Bond Market ETF (BND) and the S&P 500 ETF (SPY), and use the results to
build and backtest an optimized portfolio using Modern Portfolio Theory.

## Assets

| Asset | Ticker | Role |
|---|---|---|
| Tesla | TSLA | High-growth / high-risk |
| Vanguard Total Bond Market ETF | BND | Stability, low risk |
| S&P 500 ETF | SPY | Diversified, moderate risk |

Data window: 2015-01-01 to 2026-06-30, sourced from Yahoo Finance via `yfinance`.

## Project structure

```
portfolio-optimization/
├── data/
│   ├── raw/            # per-ticker CSVs from yfinance (gitignored, regenerate via src/data_loader.py)
│   └── processed/      # cleaned/combined datasets used by the notebooks
├── notebooks/          # exploratory + modeling notebooks (see notebooks/README.md)
├── src/                # reusable data loading / feature engineering code
├── scripts/            # standalone CLI entry points
├── tests/              # unit tests
└── reports/            # written analysis (interim + final investment memo)
```

## Setup

```bash
python -m venv .venv
source .venv/Scripts/activate   # Windows Git Bash
pip install -r requirements.txt
python src/data_loader.py       # fetches and caches TSLA/BND/SPY data
jupyter notebook
```

## Status — complete

- **Task 1 — Data preprocessing & EDA**: [notebooks/1.0-data-extraction-and-eda.ipynb](notebooks/1.0-data-extraction-and-eda.ipynb)
- **Task 2 — ARIMA & LSTM forecasting models**: [notebooks/2.0-arima-forecasting.ipynb](notebooks/2.0-arima-forecasting.ipynb), [notebooks/2.1-lstm-forecasting.ipynb](notebooks/2.1-lstm-forecasting.ipynb)
- **Task 3 — 12-month future forecast with confidence intervals**: [notebooks/3.0-future-forecast.ipynb](notebooks/3.0-future-forecast.ipynb)
- **Task 4 — Portfolio optimization (Efficient Frontier)**: [notebooks/4.0-portfolio-optimization.ipynb](notebooks/4.0-portfolio-optimization.ipynb)
- **Task 5 — Strategy backtesting**: [notebooks/5.0-backtesting.ipynb](notebooks/5.0-backtesting.ipynb)
- **Final investment memo**: [reports/final_report.md](reports/final_report.md) / [reports/final_report.pdf](reports/final_report.pdf)
- Interim submission memo: [reports/interim_report.md](reports/interim_report.md)

### Headline results

- LSTM outperforms the ARIMA(0,1,0) random-walk baseline on one-step-ahead test accuracy
  (MAPE 4.60% vs. 17.11%), though the two are evaluated under different protocols (see
  Task 2 notebook for the important caveat on that comparison).
- The LSTM's genuine 12-month recursive forecast collapses to a fixed point after ~2
  months — a real, documented limitation of naive recursive LSTM forecasting, not a
  considered bearish thesis. See the Task 3 notebook for the full discussion.
- Given that forecast, the Max Sharpe portfolio allocates 0% TSLA / 54.7% BND / 45.3% SPY
  (Sharpe 0.85). Backtested over the last 12 months, it edged the 60/40 SPY/BND benchmark
  on Sharpe Ratio (1.69 vs 1.66) and max drawdown, but trailed on raw return.

## Key dates

- Interim submission: 2026-07-05
- Final submission: 2026-07-07
