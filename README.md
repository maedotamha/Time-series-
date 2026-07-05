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

## Status

- **Task 1 — Data preprocessing & EDA**: complete. See
  [notebooks/1.0-data-extraction-and-eda.ipynb](notebooks/1.0-data-extraction-and-eda.ipynb)
  and [reports/interim_report.md](reports/interim_report.md).
- **Task 2 — Forecasting models**: ARIMA baseline in progress. See
  [notebooks/2.0-arima-forecasting.ipynb](notebooks/2.0-arima-forecasting.ipynb).
- **Task 3-5** (future forecasting, portfolio optimization, backtesting): not yet started.

## Key dates

- Interim submission: 2026-07-05
- Final submission: 2026-07-07
