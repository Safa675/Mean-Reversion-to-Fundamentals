"""
Build a daily Markdown dashboard of your manually picked undervalued stocks.

What it does
------------
1) Reads your picks from `picks.csv` in the same folder (Ticker, optional Weight).
2) Fetches latest + previous close prices from Yahoo Finance for:
   - Your tickers
   - Benchmarks: BIST30 (XU030.IS), BIST100 (XU100.IS)
3) Calculates daily returns and compares portfolio vs benchmarks.
4) Builds full daily performance table (from 2025-11-15 to today).
5) Writes `README.md` (next to this script) with:
   - Portfolio performance vs benchmarks (today)
   - Cumulative vs benchmarks since 2025-11-15
   - Daily return history (since 2025-11-15)
   - Holdings table with price/return

How to run locally
------------------
pip install -r requirements-dashboard.txt  # ensure yfinance, pandas
python build_dashboard_live.py
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import yfinance as yf

# Paths (all relative to this file's folder)
ROOT = Path(__file__).resolve().parent
PICKS_PATH = ROOT / "picks.csv"
OUTPUT_PATH = ROOT / "README.md"
# Reporting start date (Nov 15, 2025)
HISTORY_START = dt.date(2025, 11, 15)

# Benchmarks: name -> Yahoo Finance ticker
BENCHMARKS = {
    "BIST30": "XU030.IS",
    "BIST100": "XU100.IS",
}

# Default weights if none provided
DEFAULT_WEIGHT = 1.0


def _ensure_ist_suffix(ticker: str) -> str:
    ticker = ticker.strip().upper()
    return ticker if ticker.endswith(".IS") else f"{ticker}.IS"


def load_picks(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Picks file not found at {path}. Create it with columns: Ticker[,Weight]"
        )
    df = pd.read_csv(path)
    if "Ticker" not in df.columns:
        raise ValueError("Picks file must have a 'Ticker' column")
    df["Ticker"] = df["Ticker"].astype(str).apply(_ensure_ist_suffix)
    if "Weight" not in df.columns:
        df["Weight"] = DEFAULT_WEIGHT
    # Normalize weights to 1.0
    total_w = df["Weight"].sum()
    if total_w <= 0:
        df["Weight"] = 1 / len(df)
    else:
        df["Weight"] = df["Weight"] / total_w
    return df


def fetch_history(tickers: List[str], start: dt.date) -> pd.DataFrame:
    """Fetch daily closes from start date to today for given tickers."""
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
        if "Close" in data.columns:
            close = data[["Close"]]
            if len(tickers) == 1:
                close.columns = [tickers[0]]
        else:
            close = data.to_frame(name=tickers[0])
    close = close.dropna(how="all")
    return close


def build_portfolio_section(
    picks: pd.DataFrame, daily_returns: pd.DataFrame
) -> Tuple[float, float, pd.DataFrame, pd.Series]:
    if daily_returns.empty:
        picks = picks.copy()
        picks["DailyReturn_%"] = float("nan")
        return float("nan"), float("nan"), picks, daily_returns

    picks = picks.copy()
    # Latest daily return per holding
    latest_returns = daily_returns.iloc[-1]
    picks["DailyReturn_%"] = picks["Ticker"].map(latest_returns.to_dict())
    # Per-holding cumulative since start
    cum_map = {}
    for t in picks["Ticker"]:
        if t in daily_returns.columns:
            series = daily_returns[t].dropna()
            cum_map[t] = float((1 + series / 100).prod() - 1) if not series.empty else float("nan")
        else:
            cum_map[t] = float("nan")
    picks["Cumulative_%"] = picks["Ticker"].map(cum_map)
    # Portfolio daily and cumulative
    weight_vec = picks.set_index("Ticker")["Weight"]
    portfolio_daily = daily_returns[picks["Ticker"]].mul(weight_vec, axis=1).sum(axis=1)
    portfolio_latest = float(portfolio_daily.iloc[-1]) if not portfolio_daily.empty else float("nan")
    portfolio_cum = float((1 + portfolio_daily / 100).prod() - 1) if not portfolio_daily.empty else float("nan")
    return portfolio_latest, portfolio_cum, picks, portfolio_daily


def build_benchmark_section(
    bench_returns: Dict[str, float], bench_cum: Dict[str, float]
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows_daily = []
    rows_cum = []
    for name in bench_returns:
        rows_daily.append({"Benchmark": name, "DailyReturn_%": bench_returns[name]})
        rows_cum.append({"Benchmark": name, "Cumulative_%": bench_cum.get(name, float("nan"))})
    return pd.DataFrame(rows_daily), pd.DataFrame(rows_cum)


def render_markdown(
    asof: dt.date,
    start_date: dt.date,
    portfolio_ret: float,
    portfolio_cum: float,
    bench_df: pd.DataFrame,
    bench_cum_df: pd.DataFrame,
    picks_df: pd.DataFrame,
    history_df: pd.DataFrame,
) -> str:
    def fmt_pct(val: float) -> str:
        return "NA" if pd.isna(val) else f"{val:+.2f}%"

    lines = []
    lines.append(f"# Daily Undervalued Dashboard — {asof.isoformat()}")
    lines.append("")
    lines.append("## Performance vs Benchmarks (daily)")
    lines.append("")
    lines.append("| Benchmark | Daily Return % | Beat? |")
    lines.append("|---|---|---|")
    for _, row in bench_df.iterrows():
        ret_val = row["DailyReturn_%"]
        beat = "✅" if pd.notna(portfolio_ret) and pd.notna(ret_val) and portfolio_ret > ret_val else "❌"
        lines.append(
            f"| {row['Benchmark']} | {fmt_pct(ret_val)} | {beat} |"
        )
    lines.append("")
    lines.append(f"- Portfolio daily return: **{fmt_pct(portfolio_ret)}**")
    lines.append("")
    lines.append(f"## Cumulative since {start_date.isoformat()}")
    lines.append("")
    lines.append("| Benchmark | Cumulative Return % | Beat? |")
    lines.append("|---|---|---|")
    for _, row in bench_cum_df.iterrows():
        ret_val = row["Cumulative_%"]
        beat = "✅" if pd.notna(portfolio_cum) and pd.notna(ret_val) and portfolio_cum > ret_val else "❌"
        lines.append(
            f"| {row['Benchmark']} | {fmt_pct(ret_val)} | {beat} |"
        )
    lines.append(f"| Portfolio | {fmt_pct(portfolio_cum)} | — |")
    lines.append("")
    lines.append("## Holdings")
    lines.append("")
    lines.append("| Ticker | Weight | Daily Return % | Cumulative % |")
    lines.append("|---|---|---|---|")
    for _, row in picks_df.iterrows():
        ret_val = row["DailyReturn_%"]
        cum_val = row.get("Cumulative_%", float("nan"))
        lines.append(f"| {row['Ticker']} | {row['Weight']:.3f} | {fmt_pct(ret_val)} | {fmt_pct(cum_val)} |")
    lines.append("")
    lines.append(f"## Daily returns since {start_date.isoformat()}")
    lines.append("")
    lines.append("| Date | Portfolio % | BIST30 % | BIST100 % | Beat30 | Beat100 |")
    lines.append("|---|---|---|---|---|---|")
    for _, row in history_df.iterrows():
        beat30 = "✅" if pd.notna(row["Portfolio"]) and pd.notna(row["BIST30"]) and row["Portfolio"] > row["BIST30"] else "❌"
        beat100 = "✅" if pd.notna(row["Portfolio"]) and pd.notna(row["BIST100"]) and row["Portfolio"] > row["BIST100"] else "❌"
        lines.append(
            f"| {row['Date']} | {fmt_pct(row['Portfolio'])} | {fmt_pct(row['BIST30'])} | {fmt_pct(row['BIST100'])} | {beat30} | {beat100} |"
        )
    lines.append("")
    lines.append("> Data source: Yahoo Finance (close-to-close). Benchmarks: XU030, XU100.")
    return "\n".join(lines)


def main():
    asof = dt.date.today()
    picks = load_picks(PICKS_PATH)

    tickers = picks["Ticker"].tolist()
    bench_tickers = list(BENCHMARKS.values())

    start_date = HISTORY_START
    close_hist = fetch_history(tickers + bench_tickers, start=start_date)
    # Allow partial data (benchmarks may have gaps), drop rows only if all NaN
    daily_returns = close_hist.pct_change().dropna(how="all") * 100.0

    portfolio_ret, portfolio_cum, picks_with_ret, portfolio_daily = build_portfolio_section(
        picks, daily_returns
    )
    bench_returns = {}
    bench_cum = {}
    for idx, name in enumerate(BENCHMARKS.keys()):
        ticker = bench_tickers[idx]
        if ticker in daily_returns.columns and not daily_returns.empty:
            series = daily_returns[ticker].dropna()
            bench_returns[name] = float(series.iloc[-1]) if not series.empty else float("nan")
            bench_cum[name] = float((1 + series / 100).prod() - 1) if not series.empty else float("nan")
        else:
            bench_returns[name] = float("nan")
            bench_cum[name] = float("nan")
    bench_df, bench_cum_df = build_benchmark_section(bench_returns, bench_cum)

    history_df = pd.DataFrame({
        "Date": portfolio_daily.index.strftime("%Y-%m-%d") if not portfolio_daily.empty else [],
        "Portfolio": portfolio_daily.values if not portfolio_daily.empty else [],
        "BIST30": daily_returns[bench_tickers[0]].values if bench_tickers[0] in daily_returns.columns else [],
        "BIST100": daily_returns[bench_tickers[1]].values if bench_tickers[1] in daily_returns.columns else [],
    })

    markdown = render_markdown(
        asof,
        start_date,
        portfolio_ret,
        portfolio_cum,
        bench_df,
        bench_cum_df,
        picks_with_ret,
        history_df,
    )
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(markdown, encoding="utf-8")
    print(f"[OK] Dashboard written to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
