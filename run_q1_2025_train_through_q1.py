"""
Train both models using data through Q1 2025, build Q1 2025 undervalued portfolios,
and compare their live performance vs BIST.

Models:
- Factor model (Lasso-Factor Mean Reversion 0.1): factors only
- All-fundamentals model (Lasso-ElasticNet All-Fundamentals Model): full feature set

Workflow:
1) Load panel data
2) Train factor model on data <= 2025-03-31
3) Train all-fundamentals model on data < 2025-04-01 (includes Q1 2025)
4) Select Q1 2025 undervalued portfolios (bottom 20%, min 10B TL, max 20 names)
5) Fetch daily prices from 2025-04-01 to today and compare vs BIST100
"""

from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import statsmodels.api as sm
import yfinance as yf

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from pooled_ols_residuals_bist import prepare_panel_data, TARGET_COL  # noqa: E402

ALL_FUND_DIR = ROOT / "Lasso-ElasticNet All-Fundamentals Model"
sys.path.insert(0, str(ALL_FUND_DIR))
from all_fundamentals_model import run_all_fundamentals_model  # noqa: E402

PICK_QUARTER = pd.Timestamp("2025-03-31")
TRAIN_END_FACTOR = "2025-03-31"
TRAIN_END_ALL = "2025-04-01"  # < end includes Q1 2025 data
EVAL_START = dt.date(2025, 4, 1)
MIN_MCAP = 10e9
MAX_STOCKS = 20
BENCHMARKS = {"BIST100": "XU100.IS"}


def ensure_ist_suffix(ticker: str) -> str:
    ticker = ticker.strip().upper()
    return ticker if ticker.endswith(".IS") else f"{ticker}.IS"


def select_undervalued(
    df: pd.DataFrame,
    residual_col: str,
    percentile: float = 20.0,
    min_mcap: float = MIN_MCAP,
    max_stocks: int = MAX_STOCKS,
) -> pd.DataFrame:
    """Rank by residual and pick the bottom percentile with liquidity filter."""
    df = df.copy()
    df = df.dropna(subset=[residual_col, TARGET_COL])
    if "MarketCap" not in df.columns:
        df["MarketCap"] = np.exp(df[TARGET_COL])
    df["Rank"] = df[residual_col].rank()
    df["Percentile"] = df[residual_col].rank(pct=True) * 100

    filtered = df[df["Percentile"] <= percentile].copy()
    filtered = filtered[filtered["MarketCap"] >= min_mcap]

    if filtered.empty:
        return filtered

    if len(filtered) > max_stocks:
        filtered = filtered.nsmallest(max_stocks, residual_col)

    return filtered.sort_values("Rank")


def compute_portfolio_returns(close_df: pd.DataFrame, tickers: List[str]) -> Optional[Dict]:
    """Equal-weight portfolio daily returns and cumulative performance."""
    available = [t for t in tickers if t in close_df.columns]
    if not available:
        return None

    close_sub = close_df[available].dropna(how="all")
    if close_sub.empty:
        return None

    rets = close_sub.pct_change().dropna(how="all")
    if rets.empty:
        return None

    port = rets.mean(axis=1)
    cum = (1 + port).cumprod() - 1
    return {"daily": port, "cum": cum, "tickers_used": available}


def fetch_closes(tickers: List[str], start: dt.date) -> pd.DataFrame:
    """Fetch daily close prices for tickers from start date."""
    data = yf.download(
        tickers=tickers,
        start=start,
        end=dt.date.today() + dt.timedelta(days=1),
        interval="1d",
        progress=False,
        auto_adjust=False,
        threads=True,
    )
    if isinstance(data.columns, pd.MultiIndex):
        close = data["Close"]
    else:
        close = data[["Close"]] if "Close" in data.columns else data.to_frame(name=tickers[0])
        if len(tickers) == 1:
            close.columns = [tickers[0]]
    close = close.dropna(how="all")
    return close


def save_picks(path: Path, tickers: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    weights = [1 / len(tickers)] * len(tickers) if tickers else []
    pd.DataFrame({"Ticker": tickers, "Weight": weights}).to_csv(path, index=False)


def main() -> None:
    print("=" * 80)
    print("TRAIN THROUGH Q1 2025 → BUILD Q1 2025 PORTFOLIOS → COMPARE VS BIST100")
    print("=" * 80)
    print(f"Train (factor):   <= {TRAIN_END_FACTOR}")
    print(f"Train (all-fund): <  {TRAIN_END_ALL}")
    print(f"Portfolio quarter: {PICK_QUARTER.date()}")
    print(f"Evaluation start:  {EVAL_START}")
    print("=" * 80)

    print("\n[1/4] Loading panel data...")
    df_panel, _, _ = prepare_panel_data()
    print(f"Loaded {len(df_panel):,} observations")
    print(f"Date range: {df_panel.index.get_level_values('QuarterDate').min()} to {df_panel.index.get_level_values('QuarterDate').max()}")

    # ------------------------------------------------------------------ #
    # Factor model (train through Q1 2025)
    # ------------------------------------------------------------------ #
    print("\n[2/4] Training factor model through Q1 2025...")
    df_train_factor = df_panel[df_panel.index.get_level_values("QuarterDate") <= TRAIN_END_FACTOR].copy()
    factors = [
        "Factor_Scale",
        "Factor_Leverage",
        "Factor_RD",
        "Factor_Profitability",
        "Factor_Growth",
        "Factor_CapitalEfficiency",
    ]
    available_factors = [f for f in factors if f in df_train_factor.columns]
    X_train = df_train_factor[available_factors].dropna()
    y_train = df_train_factor.loc[X_train.index, TARGET_COL]
    valid_idx = y_train.dropna().index
    X_train = X_train.loc[valid_idx]
    y_train = y_train.loc[valid_idx]

    if len(X_train) < 50:
        raise RuntimeError(f"Insufficient training sample for factor model: {len(X_train)}")

    model_factor = sm.OLS(y_train, sm.add_constant(X_train)).fit(cov_type="HC3")
    print(f"Factor model R²: {model_factor.rsquared:.3f} on {len(X_train):,} obs")

    df_q1 = df_panel[df_panel.index.get_level_values("QuarterDate") == PICK_QUARTER].copy()
    X_q1 = df_q1[available_factors].dropna()
    y_q1 = df_q1.loc[X_q1.index, TARGET_COL]
    valid_q1 = y_q1.dropna().index
    X_q1 = X_q1.loc[valid_q1]
    y_q1 = y_q1.loc[valid_q1]

    preds_q1 = model_factor.predict(sm.add_constant(X_q1))
    residuals_q1 = y_q1 - preds_q1

    df_q1_factor = df_q1.loc[valid_q1].copy()
    df_q1_factor["Residual_LogMarketCap"] = residuals_q1
    df_q1_factor["MarketCap"] = np.exp(y_q1)

    factor_port = select_undervalued(df_q1_factor, "Residual_LogMarketCap")
    factor_tickers = [idx[0] for idx in factor_port.index] if not factor_port.empty else []
    print(f"Factor portfolio size: {len(factor_tickers)} tickers")

    factor_dir = ROOT / "Lasso-Factor Mean Reversion 0.1" / "backtest_q1_2025_train_through_q1"
    save_picks(factor_dir / "picks.csv", factor_tickers)
    factor_port.to_csv(factor_dir / "portfolio_details.csv")

    # ------------------------------------------------------------------ #
    # All-fundamentals model (train through Q1 2025)
    # ------------------------------------------------------------------ #
    print("\n[3/4] Training all-fundamentals model through Q1 2025...")
    all_fund_result = run_all_fundamentals_model(
        method="lasso",
        train_end=TRAIN_END_ALL,
        val_end=None,
        max_features=20,
        min_coverage_pct=10.0,
        corr_threshold=0.95,
    )
    if all_fund_result is None:
        raise RuntimeError("All-fundamentals model failed.")

    df_all = all_fund_result["results_df"].reset_index()
    df_q1_all = df_all[df_all["QuarterDate"] == PICK_QUARTER].copy()
    df_q1_all["Residual_LogMarketCap"] = df_q1_all["Residual_LogMarketCap"]
    df_q1_all["MarketCap"] = df_q1_all.get("MarketCap", np.exp(df_q1_all[TARGET_COL]))

    all_port = select_undervalued(df_q1_all, "Residual_LogMarketCap")
    all_tickers = all_port["Ticker"].tolist() if not all_port.empty else []
    print(f"All-fundamentals portfolio size: {len(all_tickers)} tickers")

    all_dir = ALL_FUND_DIR / "backtest_q1_2025_train_through_q1"
    save_picks(all_dir / "picks.csv", all_tickers)
    all_port.to_csv(all_dir / "portfolio_details.csv", index=False)

    overlap = sorted(set(factor_tickers) & set(all_tickers))
    print(f"Portfolio overlap: {len(overlap)} tickers")
    if overlap:
        print("Overlap tickers:", ", ".join(overlap))

    # ------------------------------------------------------------------ #
    # Performance vs BIST100
    # ------------------------------------------------------------------ #
    print("\n[4/4] Evaluating performance vs BIST100...")
    tickers_with_suffix = {ensure_ist_suffix(t) for t in factor_tickers + all_tickers}
    bench_with_suffix = set(BENCHMARKS.values())
    all_symbols = sorted(tickers_with_suffix | bench_with_suffix)

    close_df = fetch_closes(all_symbols, start=EVAL_START)
    if close_df.empty:
        raise RuntimeError("No price data fetched for evaluation period.")

    perf_results = {}

    factor_perf = compute_portfolio_returns(close_df, [ensure_ist_suffix(t) for t in factor_tickers])
    if factor_perf:
        perf_results["Factor"] = factor_perf

    all_perf = compute_portfolio_returns(close_df, [ensure_ist_suffix(t) for t in all_tickers])
    if all_perf:
        perf_results["AllFund"] = all_perf

    bench_perf = {}
    for name, sym in BENCHMARKS.items():
        series = close_df[[sym]].dropna(how="all")
        if not series.empty:
            rets = series.pct_change().dropna(how="all")[sym]
            cum = (1 + rets).cumprod() - 1
            bench_perf[name] = {"daily": rets, "cum": cum}

    last_date = close_df.index.max().date()

    summary_rows = []
    for label, perf in perf_results.items():
        cum_pct = float(perf["cum"].iloc[-1] * 100)
        summary_rows.append({"Name": label, "Cumulative_%": cum_pct})
    for name, perf in bench_perf.items():
        cum_pct = float(perf["cum"].iloc[-1] * 100)
        summary_rows.append({"Name": name, "Cumulative_%": cum_pct})

    summary_df = pd.DataFrame(summary_rows)

    summary_path = ROOT / "backtest_q1_2025_train_through_q1_summary.md"
    summary_lines = []
    summary_lines.append("# Q1 2025 Portfolios (trained through Q1 2025)")
    summary_lines.append("")
    summary_lines.append(f"- Factor train ≤ {TRAIN_END_FACTOR}, All-fund train < {TRAIN_END_ALL}")
    summary_lines.append(f"- Portfolio quarter: {PICK_QUARTER.date()}")
    summary_lines.append(f"- Evaluation: {EVAL_START} → {last_date}")
    summary_lines.append(f"- Factor tickers: {len(factor_tickers)} | All-fund tickers: {len(all_tickers)} | Overlap: {len(overlap)}")
    if overlap:
        summary_lines.append(f"- Overlap tickers: {', '.join(overlap)}")
    summary_lines.append("")
    summary_lines.append("## Performance vs BIST100")
    summary_lines.append("")
    summary_lines.append("| Name | Cumulative Return % |")
    summary_lines.append("|---|---|")
    for _, row in summary_df.iterrows():
        summary_lines.append(f"| {row['Name']} | {row['Cumulative_%']:+.2f}% |")

    summary_path.write_text("\n".join(summary_lines), encoding="utf-8")
    print(f"[OK] Summary saved to {summary_path}")

    print("\nPerformance:")
    for line in summary_lines[6:]:
        print(line)


if __name__ == "__main__":
    main()
