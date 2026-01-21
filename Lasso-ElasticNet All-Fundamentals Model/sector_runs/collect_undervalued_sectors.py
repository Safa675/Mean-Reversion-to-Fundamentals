"""
Collect most undervalued stocks (lowest residuals) in the latest quarter for each sector.

Reads sector results from ../outputs/*_lasso_fundamentals_results.csv, finds the latest
QuarterDate per file, picks the top N most negative residuals, and writes per-sector CSVs
into this folder.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "outputs"
DEST_DIR = Path(__file__).resolve().parent

# Sector code -> results filename stem
SECTOR_FILES = {
    "CAP": "cap_lasso_fundamentals_results.csv",
    "COM": "com_lasso_fundamentals_results.csv",
    "CON": "con_lasso_fundamentals_results.csv",
    "FIN": "fin_lasso_fundamentals_results.csv",
    "IND": "ind_lasso_fundamentals_results.csv",
    "INT": "int_lasso_fundamentals_results.csv",
    "RE": "re_lasso_fundamentals_results.csv",
}

TOP_N = 10  # number of most undervalued tickers to keep


def collect_for_sector(code: str, path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["QuarterDate"])
    df = df.dropna(subset=["Residual_LogMarketCap"])
    if df.empty:
        return pd.DataFrame()
    latest_q = df["QuarterDate"].max()
    df_q = df[df["QuarterDate"] == latest_q].copy()
    if df_q.empty:
        return pd.DataFrame()
    df_q = df_q.sort_values("Residual_LogMarketCap")
    df_q["Overvaluation_pct"] = df_q.get("Overvaluation_pct")
    df_q = df_q.head(TOP_N)
    return df_q[
        ["Ticker", "QuarterDate", "Residual_LogMarketCap", "Overvaluation_pct"]
    ]


def main() -> None:
    DEST_DIR.mkdir(exist_ok=True)
    summary_rows: List[dict] = []

    for code, fname in SECTOR_FILES.items():
        path = OUTPUT_DIR / fname
        if not path.exists():
            print(f"[WARN] Missing results for {code}: {path}")
            continue
        df_top = collect_for_sector(code, path)
        if df_top.empty:
            print(f"[WARN] No data for sector {code}")
            continue
        dest = DEST_DIR / f"undervalued_{code}.csv"
        df_top.to_csv(dest, index=False)
        print(f"[OK] Saved {len(df_top)} names for {code} to {dest}")
        summary_rows.append(
            {
                "Sector": code,
                "LatestQuarter": df_top["QuarterDate"].iloc[0].date(),
                "Names": ", ".join(df_top["Ticker"].tolist()),
            }
        )

    if summary_rows:
        summary = pd.DataFrame(summary_rows)
        summary_path = DEST_DIR / "undervalued_summary.csv"
        summary.to_csv(summary_path, index=False)
        print(f"[OK] Summary saved to {summary_path}")
    else:
        print("[WARN] No sector outputs found; nothing written.")


if __name__ == "__main__":
    sys.exit(main())
