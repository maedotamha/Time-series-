"""Fetch and cache historical price data for TSLA, BND, and SPY via yfinance."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import yfinance as yf

TICKERS = ["TSLA", "BND", "SPY"]
START_DATE = "2015-01-01"
END_DATE = "2026-06-30"

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
PROCESSED_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"


def fetch_ticker(ticker: str, start: str = START_DATE, end: str = END_DATE) -> pd.DataFrame:
    """Download OHLCV data for a single ticker and flatten the column index."""
    df = yf.download(ticker, start=start, end=end, auto_adjust=False, progress=False)
    if df.empty:
        raise ValueError(f"No data returned for {ticker}")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.index.name = "Date"
    return df


def fetch_all(tickers: list[str] = TICKERS) -> dict[str, pd.DataFrame]:
    """Fetch raw data for all tickers and persist each to data/raw/<ticker>.csv."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    data = {}
    for ticker in tickers:
        df = fetch_ticker(ticker)
        df.to_csv(RAW_DIR / f"{ticker}.csv")
        data[ticker] = df
        print(f"{ticker}: {len(df)} rows, {df.index.min().date()} -> {df.index.max().date()}")
    return data


def build_combined_frame(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Combine per-ticker frames into one long-format DataFrame with an Asset column."""
    frames = []
    for ticker, df in data.items():
        tmp = df.copy()
        tmp["Asset"] = ticker
        frames.append(tmp)
    combined = pd.concat(frames).reset_index()
    combined = combined.sort_values(["Asset", "Date"]).reset_index(drop=True)
    return combined


def save_processed(combined: pd.DataFrame) -> Path:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PROCESSED_DIR / "combined_prices.csv"
    combined.to_csv(out_path, index=False)
    return out_path


if __name__ == "__main__":
    raw = fetch_all()
    combined = build_combined_frame(raw)
    path = save_processed(combined)
    print(f"Saved combined dataset to {path} ({len(combined)} rows)")
