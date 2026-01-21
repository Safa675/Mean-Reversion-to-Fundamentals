import math
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent

DAILY_PATH = ROOT / "Last 2 Years" / "bist_2024_2025_daily_selected.xlsx"
OUTPUT_PATH = ROOT / "Last 2 Years" / "bist_2024q4_sortino_sharpe.xlsx"
DAILY_SHEET = "Daily_Prices"
ANNUAL_DAYS = 252  # trading days for annualization
RISK_FREE_ANNUAL = 0.30  # adjustable risk-free rate (annual, decimal)
START_INVEST_DATE = pd.Timestamp("2024-12-31")
END_INVEST_DATE = pd.Timestamp("2025-12-31")
INITIAL_PER_STOCK = 100.0  # TRY invested per stock


def annualized_sharpe(returns: pd.Series, rf_daily: float = 0.0) -> float:
    """Calculate annualized Sharpe ratio."""
    if returns.empty:
        return np.nan
    excess = returns - rf_daily
    mean_ret = excess.mean()
    std_ret = excess.std()
    if std_ret == 0 or pd.isna(std_ret):
        return np.nan
    return (mean_ret * ANNUAL_DAYS) / (std_ret * math.sqrt(ANNUAL_DAYS))


def annualized_sortino(returns: pd.Series, rf_daily: float = 0.0) -> float:
    """Calculate annualized Sortino ratio."""
    if returns.empty:
        return np.nan
    excess = returns - rf_daily
    mean_ret = excess.mean()
    downside = excess[excess < 0]
    if downside.empty:
        # No downside volatility; treat as very favorable
        return np.inf
    downside_std = math.sqrt((downside**2).mean())
    if downside_std == 0 or pd.isna(downside_std):
        return np.nan
    return (mean_ret * ANNUAL_DAYS) / (downside_std * math.sqrt(ANNUAL_DAYS))


def compute_metrics(daily_df: pd.DataFrame, rf_daily: float) -> pd.DataFrame:
    """Compute Sharpe and Sortino ratios per ticker, storing start/end info."""
    records = []
    for ticker, grp in daily_df.groupby("Ticker"):
        grp = grp.sort_values("Date")
        prices = grp["Price"]
        returns = prices.pct_change().dropna()

        sharpe = annualized_sharpe(returns, rf_daily=rf_daily)
        sortino = annualized_sortino(returns, rf_daily=rf_daily)

        start_price = prices.iloc[0]
        end_price = prices.iloc[-1]
        total_return = end_price / start_price - 1 if start_price > 0 else np.nan
        start_value = INITIAL_PER_STOCK
        end_value = start_value * (end_price / start_price) if start_price > 0 else np.nan

        records.append(
            {
                "Ticker": ticker,
                "Sector": grp["Sector"].iloc[0],
                "Overvaluation_pct": grp["Overvaluation_pct"].iloc[0],
                "Start_Date": grp["Date"].iloc[0],
                "End_Date": grp["Date"].iloc[-1],
                "Start_Price": start_price,
                "End_Price": end_price,
                "Start_Value": start_value,
                "End_Value": end_value,
                "Days": len(returns),
                "Total_Return": total_return,
                "Sharpe": sharpe,
                "Sortino": sortino,
            }
        )

    return pd.DataFrame(records)


def main():
    daily_df = pd.read_excel(DAILY_PATH, sheet_name=DAILY_SHEET)
    required_cols = {"Ticker", "Date", "HGDG_KAPANIS"}
    missing = required_cols - set(daily_df.columns)
    if missing:
        raise ValueError(f"Missing columns in daily price file: {missing}")

    daily_df["Date"] = pd.to_datetime(daily_df["Date"], errors="coerce")
    daily_df["Price"] = pd.to_numeric(daily_df["HGDG_KAPANIS"], errors="coerce")
    daily_df["Sector"] = daily_df.get("Sector", pd.Series(np.nan, index=daily_df.index))
    daily_df["Overvaluation_pct"] = daily_df.get("Overvaluation_pct", pd.Series(np.nan, index=daily_df.index))
    daily_df = daily_df.dropna(subset=["Ticker", "Date", "Price"])
    daily_df["Ticker"] = daily_df["Ticker"].astype(str).str.upper()

    # Restrict to investment window
    invest_df = daily_df[
        (daily_df["Date"] >= START_INVEST_DATE) & (daily_df["Date"] <= END_INVEST_DATE)
    ].copy()

    # Compute per-ticker metrics
    rf_daily = (1 + RISK_FREE_ANNUAL) ** (1 / ANNUAL_DAYS) - 1
    metrics_df = compute_metrics(invest_df, rf_daily=rf_daily)
    metrics_df = metrics_df.sort_values(["Sector", "Ticker"]).reset_index(drop=True)

    # Portfolio path: 100 TRY per ticker at START_INVEST_DATE (or next available date)
    portfolio_values = []
    tickers_used = []
    for ticker, grp in invest_df.groupby("Ticker"):
        grp = grp.sort_values("Date")
        # choose the first available date on/after START_INVEST_DATE
        start_row = grp[grp["Date"] >= START_INVEST_DATE].head(1)
        if start_row.empty:
            continue
        start_price = start_row["Price"].iloc[0]
        if start_price <= 0:
            continue
        shares = INITIAL_PER_STOCK / start_price
        grp = grp[grp["Date"] >= START_INVEST_DATE]
        if grp.empty:
            continue
        grp = grp[["Date", "Price"]].copy()
        grp["Value"] = grp["Price"] * shares
        grp["Ticker"] = ticker
        portfolio_values.append(grp)
        tickers_used.append(ticker)

    if not portfolio_values:
        raise ValueError("No tickers had valid start prices on the investment date.")

    portfolio_df = pd.concat(portfolio_values, ignore_index=True)
    # pivot to sum across tickers
    pivot = portfolio_df.pivot_table(index="Date", values="Value", aggfunc="sum").sort_index()
    portfolio_returns = pivot["Value"].pct_change().dropna()

    portfolio_sharpe = annualized_sharpe(portfolio_returns, rf_daily=rf_daily)
    portfolio_sortino = annualized_sortino(portfolio_returns, rf_daily=rf_daily)
    portfolio_start = pivot["Value"].iloc[0]
    portfolio_end = pivot["Value"].iloc[-1]
    portfolio_total_return = portfolio_end / portfolio_start - 1
    ticker_count = len(tickers_used)
    invested_total = ticker_count * INITIAL_PER_STOCK

    with pd.ExcelWriter(OUTPUT_PATH, engine="openpyxl") as writer:
        metrics_df.to_excel(writer, sheet_name="Metrics", index=False)
        pivot.to_excel(writer, sheet_name="Portfolio_Value")
        pd.DataFrame(
            [
                {
                    "Tickers": ", ".join(sorted(tickers_used)),
                    "Ticker_Count": ticker_count,
                    "Invested_per_Stock": INITIAL_PER_STOCK,
                    "Invested_Total": invested_total,
                    "Start_Value": portfolio_start,
                    "End_Value": portfolio_end,
                    "Total_Return": portfolio_total_return,
                    "Sharpe": portfolio_sharpe,
                    "Sortino": portfolio_sortino,
                    "RiskFree_Annual": RISK_FREE_ANNUAL,
                    "RiskFree_Daily": rf_daily,
                }
            ]
        ).to_excel(writer, sheet_name="Portfolio_Metrics", index=False)

    print(f"Saved metrics for {len(metrics_df)} tickers to {OUTPUT_PATH}")
    print(
        f"Portfolio Sharpe: {portfolio_sharpe:.4f}, Sortino: {portfolio_sortino:.4f}, "
        f"Total return: {portfolio_total_return:.2%}, "
        f"Start={portfolio_start:.2f}, End={portfolio_end:.2f}, "
        f"Invested per stock={INITIAL_PER_STOCK:.2f}"
    )


if __name__ == "__main__":
    main()
