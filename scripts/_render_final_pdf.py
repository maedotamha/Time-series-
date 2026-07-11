import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from reportlab.lib.units import inch
from reportlab.platypus import PageBreak, Paragraph, Spacer

from build_final_report_pdf import build, fig, styled_table, styles

ROOT = Path(__file__).resolve().parents[1]

stats = [
    ("4.60%", "LSTM test MAPE"),
    ("-32.5%", "TSLA 12mo forecast"),
    ("0.85", "Max Sharpe portfolio"),
    ("1.69 vs 1.66", "Strategy vs. benchmark Sharpe"),
]


def body(story):
    story.append(Paragraph("Executive Summary", styles["H1"]))
    story.append(Paragraph(
        "This memo presents GMF Investments' time series forecasting and portfolio "
        "optimization analysis for a three-asset portfolio: <b>TSLA</b> (high-growth "
        "equity), <b>BND</b> (investment-grade bonds), and <b>SPY</b> (broad market index), "
        "using daily data from 2015-01-01 to 2026-06-30. We cleaned and explored the data, "
        "built and compared ARIMA and LSTM forecasting models for TSLA, generated a "
        "12-month forward forecast with confidence intervals, used Modern Portfolio Theory "
        "to derive an optimal three-asset portfolio, and backtested that portfolio against "
        "a passive 60/40 SPY/BND benchmark. Consistent with the Efficient Market "
        "Hypothesis, we find limited exploitable signal in TSLA's historical prices alone; "
        "our recommendation therefore treats the forecast as one input among several, not "
        "a standalone trading signal.", styles["Body"]))

    story.append(Paragraph("Task 1 — Data Preprocessing and Exploratory Analysis", styles["H1"]))
    story.append(Paragraph(
        "Daily OHLCV data for TSLA, BND, and SPY were fetched via yfinance for 2015-01-01 "
        "through 2026-06-30, returning 2,888 trading days per ticker with no missing "
        "values. Prices were reindexed onto a common business-day calendar and any gaps "
        "were forward-filled (not interpolated), since a price is genuinely unchanged "
        "until the next trade occurs.", styles["Body"]))
    story += fig("eda_cell10_1.png", caption="Figure 1. Adjusted close price, actual and normalized.")
    story += fig("eda_cell12_2.png", caption="Figure 2. Daily percentage returns by asset.")
    story += fig("eda_cell14_3.png", caption="Figure 3. 30-day rolling volatility.")

    story.append(Paragraph("Stationarity testing (Augmented Dickey-Fuller)", styles["H2"]))
    story.append(styled_table(
        ["Series", "ADF statistic", "p-value", "Stationary at 5%?"],
        [
            ["TSLA — Close (level)", "-1.04", "0.739", "No"],
            ["TSLA — Daily return", "-55.15", "<0.001", "Yes"],
            ["BND — Close (level)", "-1.12", "0.708", "No"],
            ["BND — Daily return", "-21.28", "<0.001", "Yes"],
            ["SPY — Close (level)", "1.43", "0.997", "No"],
            ["SPY — Daily return", "-15.70", "<0.001", "Yes"],
        ],
    ))
    story.append(Spacer(1, 8))
    story.append(Paragraph(
        "All three price <i>levels</i> fail to reject the ADF null (non-stationary, "
        "random-walk-like), consistent with the Efficient Market Hypothesis; all daily "
        "<i>return</i> series are stationary, confirming a first difference (d=1) is "
        "required for ARIMA.", styles["Callout"]))

    story.append(Paragraph("Risk metrics (annualized, 0% risk-free rate)", styles["H2"]))
    story.append(styled_table(
        ["Asset", "Ann. Return", "Ann. Volatility", "VaR (95%, daily)", "Sharpe"],
        [
            ["TSLA", "43.8%", "56.1%", "5.11%", "0.78"],
            ["BND", "1.9%", "5.2%", "0.47%", "0.37"],
            ["SPY", "13.9%", "17.3%", "1.64%", "0.80"],
        ],
    ))

    story.append(PageBreak())
    story.append(Paragraph("Task 2 — Forecasting Models: ARIMA vs. LSTM", styles["H1"]))
    story.append(Paragraph(
        "The TSLA adjusted-close series was split chronologically (no shuffling): train = "
        "2015-01-02 to 2024-12-31 (2,608 observations), test = 2025-01-01 to 2026-06-29 "
        "(391 observations). ARIMA: pmdarima.auto_arima selected ARIMA(0,1,0) — a "
        "driftless random walk — forecast statically over the full test horizon. LSTM: "
        "60-day lookback window, two stacked LSTM(50) layers with dropout, trained with "
        "early stopping; test predictions use the true preceding 60 days at each step "
        "(one-step-ahead evaluation).", styles["Body"]))
    story.append(styled_table(
        ["Model", "MAE", "RMSE", "MAPE (%)", "Forecast mode"],
        [
            ["ARIMA(0,1,0)", "54.15", "70.20", "17.11", "Static, unconditional, 391-day"],
            ["LSTM (60-day window)", "17.22", "21.34", "4.60", "One-step-ahead, true history"],
        ],
    ))
    story.append(Spacer(1, 8))
    story += fig("lstm_forecast_vs_actual.png", caption="Figure 4. LSTM one-step-ahead forecast vs. actual, test period.")
    story.append(Paragraph(
        "These are not a perfectly apples-to-apples comparison — ARIMA forecasts the full "
        "18-month horizon unconditionally, while the LSTM is re-conditioned on true history "
        "at every step (an inherently easier task). Both models agree TSLA price levels are "
        "close to a random walk with little linear autocorrelation structure, consistent "
        "with the Efficient Market Hypothesis. We select the <b>LSTM</b> to carry forward "
        "into Task 3.", styles["Body"]))

    story.append(Paragraph("Task 3 — Forecasting Future Market Trends", styles["H1"]))
    story.append(Paragraph(
        "A genuine 12-month (252 trading day) <b>recursive</b> LSTM forecast was generated "
        "(each day's prediction feeds back in as the next day's context), with a 95% "
        "confidence band from Monte Carlo simulation calibrated to TSLA's historical "
        "drift/volatility, multiplicatively re-centered on the LSTM's own trajectory so the "
        "band can never cross zero.", styles["Body"]))
    story += fig("tsla_future_forecast.png", caption="Figure 5. 12-month recursive LSTM forecast with 95% Monte Carlo CI.")
    story.append(Paragraph(
        "<b>Last observed price</b> (2026-06-29): $411.84 &nbsp;&nbsp; "
        "<b>12-month point forecast</b>: $277.90 (implied return: -32.5%) &nbsp;&nbsp; "
        "<b>95% CI at 12 months</b>: [$97.57, $846.63]", styles["Body"]))
    story.append(Paragraph(
        "<b>A fixed-point artifact, not a bearish thesis.</b> The forecast drops from "
        "$411.84 toward the high-$270s over the first ~2 months, then converges to a fixed "
        "point (~$277.90) and stays essentially flat for the remaining ~10 months. This is "
        "a well-documented failure mode of naive recursive one-step LSTM forecasting: once "
        "consecutive model-generated predictions stop changing, the recursion has settled "
        "into an attractor and simply reproduces it. We treat the -32.5% point estimate "
        "with real skepticism rather than at face value — it reflects the early collapse, "
        "not a considered 12-month view. The 95% interval itself widens from ~$57 at day 1 "
        "to ~$749 by month 12, an independent signal that long-horizon certainty is low.",
        styles["Callout"]))

    story.append(PageBreak())
    story.append(Paragraph("Task 4 — Portfolio Optimization (Modern Portfolio Theory)", styles["H1"]))
    story.append(Paragraph(
        "TSLA's expected return uses the Task 3 LSTM forecast (-32.5% annualized); BND and "
        "SPY use historical mean annualized daily returns (1.9% and 13.9%). The covariance "
        "matrix uses sample covariance of historical daily returns for all three assets, "
        "annualized.", styles["Body"]))
    story += fig("efficient_frontier.png", caption="Figure 6. Efficient Frontier — TSLA / BND / SPY.")
    story.append(styled_table(
        ["Portfolio", "TSLA", "BND", "SPY", "Exp. Return", "Volatility", "Sharpe"],
        [
            ["Max Sharpe (recommended)", "0.0%", "54.7%", "45.3%", "7.35%", "8.66%", "0.85"],
            ["Min Volatility", "0.0%", "94.5%", "5.5%", "2.58%", "5.13%", "0.50"],
        ],
    ))
    story.append(Spacer(1, 8))
    story.append(Paragraph(
        "Both optimal portfolios allocate <b>0% to TSLA</b>: given its negative forecasted "
        "return combined with by-far-highest volatility (~56% annualized) and only moderate "
        "diversification benefit (~0.49 correlation with SPY), no efficient portfolio wants "
        "TSLA exposure at this input. We recommend the <b>Maximum Sharpe Ratio portfolio</b> "
        "(54.7% BND / 45.3% SPY) as the standard institutional default absent a "
        "client-specific risk mandate. This recommendation is explicitly conditional on the "
        "Task 3 TSLA view — a more optimistic forecast would likely restore some TSLA "
        "allocation, illustrating how directly forecast quality drives the downstream "
        "portfolio decision.", styles["Body"]))
    story += fig("covariance_heatmap.png", width=3.1 * inch, caption="Figure 7. Annualized covariance matrix.")

    story.append(Paragraph("Task 5 — Strategy Backtesting", styles["H1"]))
    story.append(Paragraph(
        "The Max Sharpe portfolio (monthly rebalanced) was simulated over the last 12 "
        "months of the dataset (2025-07-01 to 2026-06-29) against a static 60% SPY / 40% "
        "BND benchmark.", styles["Body"]))
    story += fig("backtest_cumulative_returns.png", caption="Figure 8. Cumulative returns — strategy vs. benchmark.")
    story.append(styled_table(
        ["Metric", "Strategy (Max Sharpe)", "Benchmark (60/40)"],
        [
            ["Total Return", "11.64%", "14.21%"],
            ["Annualized Return", "11.31%", "13.80%"],
            ["Annualized Volatility", "6.48%", "7.96%"],
            ["Sharpe Ratio", "1.69", "1.66"],
            ["Max Drawdown", "-4.91%", "-5.82%"],
        ],
    ))
    story.append(Spacer(1, 8))
    story.append(Paragraph(
        "No unambiguous win either way. The benchmark delivered higher raw return (more "
        "SPY exposure in a rising market); the strategy edged out the benchmark on "
        "risk-adjusted terms (higher Sharpe, smaller drawdown) by holding more BND and zero "
        "TSLA. In short: the strategy was the steadier ride, the benchmark was the better "
        "raw return, in this particular year — a direct consequence of Task 3's negative "
        "TSLA view driving Task 4's allocation. A single 12-month backtest with this narrow "
        "a margin is far too little evidence to call this proof of durable outperformance "
        "in either direction.", styles["Body"]))

    story.append(Paragraph("Overall Conclusion", styles["H1"]))
    story.append(Paragraph(
        "This project demonstrates the full pipeline from raw price data to a backtested, "
        "model-informed portfolio recommendation. The central, recurring finding across "
        "every task is consistent with the Efficient Market Hypothesis: TSLA's price "
        "history contains limited robust, exploitable structure, and both classical "
        "(ARIMA) and deep learning (LSTM) models largely rediscover this fact rather than "
        "refuting it — the LSTM's own recursive forecast collapsing to a fixed point is a "
        "vivid illustration. Forecast uncertainty compounds quickly beyond a few weeks, and "
        "any portfolio decision built on these forecasts should treat them as one input "
        "among several — combined with historical risk data on BND/SPY and sound portfolio "
        "construction (MPT) — rather than as precise, standalone predictions. GMF's "
        "recommended allocation (54.7% BND / 45.3% SPY, 0% TSLA) reflects a risk-adjusted "
        "balance validated, with meaningful caveats, by a one-year backtest against a "
        "passive benchmark.", styles["Body"]))


build(ROOT / "reports" / "final_report.pdf", stats, body)
