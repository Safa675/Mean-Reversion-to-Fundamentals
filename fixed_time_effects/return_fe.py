import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Headless backend for environments without a display
import matplotlib.pyplot as plt
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
RESULTS_PATH = ROOT / "fixed_time_effects_outputs" / "bist_fixed_time_effects.xlsx"
MARKET_CAP_PATH = ROOT / "all_tickers_quarterly_2016_2026.csv"
SHEET_NAME = "Pooled_FE_All"

OUTPUT_RESULTS_TEMPLATE = "bist_mean_reversion_analysis_fe_{sector}.xlsx"
OUTPUT_SECTOR_EXTREMES = ROOT / "bist_sector_extremes_last_year_fe.xlsx"
OUTPUT_PLOT = ROOT / "bist_mean_reversion_by_overvaluation_fe.png"
RECENT_QUARTERS = 4
UNDERVALUED_TOP_N = 10
FOUR_QUARTER_SHEET = "4quarters"
SECTOR_TOP_N = 10
SECTOR_CODES = ["CAP", "COM", "CON", "FIN", "IND", "INT", "RE"]

SECTOR_DISPLAY_NAMES = {
    "CAP": "CAP - Asset-Heavy - Capital Intensive",
    "COM": "COM - Commodity - Energy - Mining",
    "CON": "CON - Consumer - Retail - Food",
    "FIN": "FIN - Financial Balance Sheet Businesses",
    "IND": "IND - Manufacturing & Industrials",
    "INT": "INT - Intangible - Knowledge Driven",
    "RE": "RE - Real Estate - Asset Revaluation (GYO, tourism property-heavy)",
}


def coerce_numeric(value):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.nan
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none"}:
        return np.nan
    try:
        return float(text)
    except Exception:
        pass
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        lower = line.lower()
        if lower.startswith("name:") or lower.startswith("ticker") or "dtype:" in lower:
            continue
        tokens = line.split()
        for token in reversed(tokens):
            try:
                return float(token)
            except Exception:
                continue
    matches = re.findall(r"[-+]?(?:\d+\.\d*|\d*\.\d+|\d+)(?:[eE][-+]?\d+)?", text)
    for match in matches:
        if "." in match or "e" in match.lower():
            try:
                return float(match)
            except Exception:
                continue
    for match in matches:
        try:
            return float(match)
        except Exception:
            continue
    return np.nan


def load_results_and_market_data(sheet_name=SHEET_NAME, sector_code=None):
    """Load FE+TE results and market cap data."""
    df_results = pd.read_excel(RESULTS_PATH, sheet_name=sheet_name)
    if "QuarterDate" not in df_results.columns or "Ticker" not in df_results.columns:
        raise ValueError("Results file missing required columns: QuarterDate or Ticker.")
    df_results["QuarterDate"] = pd.to_datetime(df_results["QuarterDate"], errors="coerce").dt.normalize()
    df_results["Ticker"] = df_results["Ticker"].astype(str).str.upper().str.replace(r"\.IS$", "", regex=True)

    if "SectorGroup" not in df_results.columns or df_results["SectorGroup"].dropna().empty:
        fundamentals_path = ROOT / "bist_quarterly_fundamentals.csv"
        if fundamentals_path.exists():
            df_fund = pd.read_csv(fundamentals_path, usecols=["Ticker", "SectorGroup"], dtype=str)
            df_fund["Ticker"] = df_fund["Ticker"].str.upper().str.replace(r"\.IS$", "", regex=True)
            sector_map = (
                df_fund.dropna(subset=["SectorGroup"])
                .groupby("Ticker")["SectorGroup"]
                .agg(lambda s: s.iloc[0])
            )
            df_results["SectorGroup"] = df_results["Ticker"].map(sector_map).fillna("")
        else:
            df_results["SectorGroup"] = ""
    else:
        df_results["SectorGroup"] = df_results["SectorGroup"].fillna("").astype(str).str.upper()

    if sector_code:
        df_results.loc[:, "SectorGroup"] = sector_code

    df_results["SectorName"] = df_results["SectorGroup"].map(SECTOR_DISPLAY_NAMES).fillna(df_results["SectorGroup"])

    df_mc = pd.read_csv(MARKET_CAP_PATH)
    if "QuarterEnd" in df_mc.columns:
        price_cols = [c for c in df_mc.columns if c.endswith("_Close_QuarterEnd")]
        if not price_cols:
            raise ValueError("No price columns found in all_tickers_quarterly_2016_2026.csv.")
        df_mc = df_mc[["QuarterEnd"] + price_cols].melt(
            id_vars=["QuarterEnd"],
            value_vars=price_cols,
            var_name="Ticker",
            value_name="Price",
        )
        df_mc["Ticker"] = df_mc["Ticker"].str.replace("_Close_QuarterEnd", "", regex=False)
        df_mc = df_mc.rename(columns={"QuarterEnd": "QuarterDate"})
    else:
        rename_map = {}
        if "ticker" in df_mc.columns:
            rename_map["ticker"] = "Ticker"
        if "quarter_end" in df_mc.columns:
            rename_map["quarter_end"] = "QuarterDate"
        if "market_cap" in df_mc.columns:
            rename_map["market_cap"] = "MarketCap"
        if "price" in df_mc.columns:
            rename_map["price"] = "Price"
        if "shares" in df_mc.columns:
            rename_map["shares"] = "Shares"
        df_mc = df_mc.rename(columns=rename_map)
        if "MarketCap" not in df_mc.columns:
            for alt in ("marketcap", "market_capitalization"):
                if alt in df_mc.columns:
                    df_mc = df_mc.rename(columns={alt: "MarketCap"})
                    break

    df_mc["QuarterDate"] = pd.to_datetime(df_mc["QuarterDate"], errors="coerce").dt.normalize()
    df_mc["Ticker"] = df_mc["Ticker"].astype(str).str.upper().str.replace(r"\.IS$", "", regex=True)
    for col in ("MarketCap", "Price", "Shares"):
        if col in df_mc.columns:
            df_mc[col] = df_mc[col].apply(coerce_numeric)

    value_col = None
    if "MarketCap" in df_mc.columns:
        value_col = "MarketCap"
        if "Price" in df_mc.columns and "Shares" in df_mc.columns:
            missing_cap = df_mc["MarketCap"].isna() & df_mc["Price"].notna() & df_mc["Shares"].notna()
            df_mc.loc[missing_cap, "MarketCap"] = df_mc.loc[missing_cap, "Price"] * df_mc.loc[missing_cap, "Shares"]
    elif "Price" in df_mc.columns:
        value_col = "Price"

    if value_col is None:
        raise ValueError("Neither MarketCap nor Price column found in all_tickers_quarterly_2016_2026.csv.")

    df_mc = df_mc.dropna(subset=["QuarterDate", value_col])
    df_mc = df_mc[df_mc[value_col] > 0]

    return df_results, df_mc, value_col


def compute_forward_returns(df_results, df_mc, horizons=(1, 2, 4), value_col="MarketCap"):
    df = df_results.merge(
        df_mc[["Ticker", "QuarterDate", value_col]],
        on=["Ticker", "QuarterDate"],
        how="left",
    )
    df = df.sort_values(["Ticker", "QuarterDate"])

    for h in horizons:
        forward_col = f"{value_col}_forward_{h}Q"
        df[forward_col] = df.groupby("Ticker")[value_col].shift(-h)
        base = df[value_col]
        fwd = df[forward_col]
        valid = (base > 0) & (fwd > 0)
        df[f"Return_{h}Q"] = np.where(valid, (fwd / base - 1) * 100, np.nan)
        df[f"LogReturn_{h}Q"] = np.where(valid, np.log(fwd / base) * 100, np.nan)

    return df


def print_recent_undervalued(df, n_quarters=RECENT_QUARTERS, top_n=UNDERVALUED_TOP_N):
    if "Overvaluation_pct" not in df.columns:
        print("Overvaluation_pct column missing; cannot list undervalued stocks.")
        return
    date_series = pd.to_datetime(df["QuarterDate"], errors="coerce").dropna()
    if date_series.empty:
        print("No valid QuarterDate values found.")
        return
    unique_dates = np.sort(date_series.unique())
    n_quarters = min(n_quarters, len(unique_dates))
    last_dates = unique_dates[-n_quarters:]

    print(f"\nMost undervalued stocks in the last {n_quarters} quarters:")
    for q in last_dates:
        df_q = df[(df["QuarterDate"] == q) & (df["Overvaluation_pct"] < 0)].dropna(subset=["Overvaluation_pct"])
        if df_q.empty:
            print(f"\nQuarter {pd.Timestamp(q).date()}: no data")
            continue
        cols = [
            c for c in [
                "Ticker",
                "QuarterDate",
                "Overvaluation_pct",
                "Overval_Quintile",
                "Overval_Decile",
                "Return_1Q",
                "Return_2Q",
                "Return_4Q",
            ]
            if c in df_q.columns
        ]
        report = df_q.nsmallest(top_n, "Overvaluation_pct")[cols]
        print(f"\nQuarter {pd.Timestamp(q).date()}:")
        print(report.to_string(index=False))


def print_recent_sector_extremes(df, n_quarters=RECENT_QUARTERS, top_n=SECTOR_TOP_N):
    """Print undervalued names by sector for recent quarters."""
    if "SectorGroup" not in df.columns or "Overvaluation_pct" not in df.columns:
        print("Sector information missing; skipping sector-level summaries.")
        return

    date_series = pd.to_datetime(df["QuarterDate"], errors="coerce").dropna()
    if date_series.empty:
        print("No valid QuarterDate values found.")
        return
    unique_dates = np.sort(date_series.unique())
    last_dates = unique_dates[-min(n_quarters, len(unique_dates)):]

    print(f"\nMost undervalued / overvalued by sector for last {len(last_dates)} quarters:")
    for q in last_dates:
        df_q = df[(df["QuarterDate"] == q) & (df["Overvaluation_pct"] < 0)].dropna(subset=["Overvaluation_pct"])
        if df_q.empty:
            continue
        print(f"\nQuarter {pd.Timestamp(q).date()}:")
        for sector, df_sec in df_q.groupby("SectorName"):
            under = df_sec.nsmallest(top_n, "Overvaluation_pct")
            if under.empty:
                continue
            print(f"  Sector {sector or 'UNKNOWN'}:")
            print("    Undervalued:")
            print(under[["Ticker", "Overvaluation_pct"]].to_string(index=False))


def create_overvaluation_groups(df, n_groups=5):
    """Create quintile and decile groupings for overvaluation."""
    df["Overval_Quintile"] = pd.qcut(
        df["Overvaluation_pct"],
        q=n_groups,
        labels=[f"Q{i}" for i in range(1, n_groups + 1)],
        duplicates="drop",
    )
    df["Overval_Decile"] = pd.qcut(
        df["Overvaluation_pct"],
        q=10,
        labels=[f"D{i}" for i in range(1, 11)],
        duplicates="drop",
    )
    return df


def analyze_mean_reversion(df, horizons=(1, 2, 4)):
    """Test if overvalued stocks underperform (mean reversion)."""
    results = []
    quintile_analysis = pd.DataFrame()
    decile_analysis = pd.DataFrame()

    for h in horizons:
        return_col = f"Return_{h}Q"
        log_return_col = f"LogReturn_{h}Q"

        quintile_analysis = df.groupby("Overval_Quintile", observed=False).agg({
            return_col: ["mean", "median", "std", "count"],
            log_return_col: ["mean", "median"],
            "Overvaluation_pct": "mean",
        }).round(2)
        quintile_analysis.columns = [
            f"Mean_Return_{h}Q", f"Median_Return_{h}Q",
            f"Std_Return_{h}Q", f"N_Obs_{h}Q",
            f"Mean_LogReturn_{h}Q", f"Median_LogReturn_{h}Q",
            f"Avg_Overval_pct",
        ]

        decile_analysis = df.groupby("Overval_Decile", observed=False).agg({
            return_col: ["mean", "median", "count"],
            "Overvaluation_pct": "mean",
        }).round(2)
        decile_analysis.columns = [
            f"Mean_Return_{h}Q", f"Median_Return_{h}Q",
            f"N_Obs_{h}Q", f"Avg_Overval_pct",
        ]

        valid_data = df[[return_col, "Overvaluation_pct"]].dropna()
        corr = p_value = np.nan
        if len(valid_data) > 0:
            corr, p_value = stats.spearmanr(
                valid_data["Overvaluation_pct"],
                valid_data[return_col],
            )
            results.append({
                "Horizon": f"{h}Q",
                "Correlation": corr,
                "P_value": p_value,
                "N_Obs": len(valid_data),
                "Significant": "Yes" if p_value < 0.05 else "No",
            })

        print(f"\n{'='*70}")
        print(f"MEAN REVERSION ANALYSIS - {h} Quarter Forward Returns")
        print(f"{'='*70}")
        print("\nBy Quintile (Q1=Most Undervalued, Q5=Most Overvalued):")
        print(quintile_analysis)

        print("\n\nBy Decile (D1=Most Undervalued, D10=Most Overvalued):")
        print(decile_analysis)

        if len(valid_data) > 0:
            print("\n\nCorrelation Test (Overvaluation vs Forward Returns):")
            print(f"Spearman Correlation: {corr:.4f}")
            print(f"P-value: {p_value:.4f}")
            print(
                "Interpretation: "
                + ("SIGNIFICANT mean reversion detected!" if p_value < 0.05 else "No significant relationship")
            )

    correlation_summary = pd.DataFrame(results)
    return quintile_analysis, decile_analysis, correlation_summary


def decile_returns_by_year(df, horizon=4):
    """Aggregate forward returns by decile for the given horizon grouped by year."""
    return_col = f"Return_{horizon}Q"
    if return_col not in df.columns or "Overval_Decile" not in df.columns:
        return pd.DataFrame()

    df_year = df.copy()
    df_year["Year"] = pd.to_datetime(df_year["QuarterDate"], errors="coerce").dt.year
    df_year = df_year.dropna(subset=["Year"])

    grouped = df_year.groupby(["Year", "Overval_Decile"], observed=False).agg({
        return_col: ["mean", "median", "count"],
        "Overvaluation_pct": "mean",
        "Ticker": lambda s: ", ".join(sorted(pd.Series(s).dropna().unique())),
    }).round(2)

    grouped.columns = [
        f"Mean_Return_{horizon}Q",
        f"Median_Return_{horizon}Q",
        f"N_Obs_{horizon}Q",
        "Avg_Overval_pct",
        "Tickers",
    ]

    return grouped.reset_index()


def get_recent_quarters(df, n_quarters=RECENT_QUARTERS):
    dates = pd.to_datetime(df["QuarterDate"], errors="coerce").dropna().unique()
    if len(dates) == 0:
        return []
    dates = sorted(dates)
    return dates[-min(n_quarters, len(dates)):]


def build_sector_extremes(df, last_quarters, top_n=SECTOR_TOP_N):
    """Build per-sector under/over-valued tables for specified quarters."""
    if "SectorGroup" not in df.columns or "Overvaluation_pct" not in df.columns:
        return pd.DataFrame(), pd.DataFrame()

    records_under = []
    records_over = []
    for sector_code, df_sector in df.groupby("SectorName"):
        for q in last_quarters:
            df_q = df_sector[
                (df_sector["QuarterDate"] == q) & (df_sector["Overvaluation_pct"] < 0)
            ].dropna(subset=["Overvaluation_pct"])
            if df_q.empty:
                continue
            under = df_q.nsmallest(top_n, "Overvaluation_pct")
            over = df_q.nlargest(top_n, "Overvaluation_pct")

            for _, row in under.iterrows():
                records_under.append({
                    "Sector": sector_code or "UNKNOWN",
                    "QuarterDate": q,
                    "Ticker": row["Ticker"],
                    "Overvaluation_pct": row["Overvaluation_pct"],
                    "Return_1Q": row.get("Return_1Q", np.nan),
                    "Return_2Q": row.get("Return_2Q", np.nan),
                    "Return_4Q": row.get("Return_4Q", np.nan),
                })
            for _, row in over.iterrows():
                records_over.append({
                    "Sector": sector_code or "UNKNOWN",
                    "QuarterDate": q,
                    "Ticker": row["Ticker"],
                    "Overvaluation_pct": row["Overvaluation_pct"],
                    "Return_1Q": row.get("Return_1Q", np.nan),
                    "Return_2Q": row.get("Return_2Q", np.nan),
                    "Return_4Q": row.get("Return_4Q", np.nan),
                })

    under_df = pd.DataFrame(records_under)
    over_df = pd.DataFrame(records_over)
    under_df = under_df.sort_values(["QuarterDate", "Sector", "Overvaluation_pct"]).reset_index(drop=True)
    over_df = over_df.sort_values(["QuarterDate", "Sector", "Overvaluation_pct"], ascending=[True, True, False]).reset_index(drop=True)
    return under_df, over_df


def plot_mean_reversion(df, horizons=(1, 2, 4)):
    """Create visualizations of mean reversion patterns."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(
        "Mean Reversion Analysis (FE+TE): Do Overvalued Stocks Underperform?",
        fontsize=16,
        fontweight="bold",
    )

    ax1 = axes[0, 0]
    for h in horizons:
        return_col = f"Return_{h}Q"
        quintile_returns = df.groupby("Overval_Quintile")[return_col].mean()
        ax1.plot(
            quintile_returns.index,
            quintile_returns.values,
            marker="o",
            linewidth=2,
            markersize=8,
            label=f"{h}Q ahead",
        )

    ax1.axhline(y=0, color="red", linestyle="--", alpha=0.7)
    ax1.set_xlabel("Overvaluation Quintile (Q1=Undervalued, Q5=Overvalued)", fontweight="bold")
    ax1.set_ylabel("Average Forward Return (%)", fontweight="bold")
    ax1.set_title("Forward Returns by Overvaluation Quintile")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2 = axes[0, 1]
    return_4q = f"Return_{horizons[-1]}Q"
    df_box = df[["Overval_Quintile", return_4q]].dropna()
    df_box.boxplot(column=return_4q, by="Overval_Quintile", ax=ax2)
    ax2.axhline(y=0, color="red", linestyle="--", alpha=0.7)
    ax2.set_xlabel("Overvaluation Quintile", fontweight="bold")
    ax2.set_ylabel("1-Year Forward Return (%)", fontweight="bold")
    ax2.set_title("Distribution of 1-Year Returns by Quintile")
    plt.sca(ax2)
    plt.xticks(rotation=0)

    ax3 = axes[1, 0]
    scatter_data = df[["Overvaluation_pct", return_4q]].dropna()
    if len(scatter_data) > 1000:
        scatter_data = scatter_data.sample(1000, random_state=42)
    ax3.scatter(scatter_data["Overvaluation_pct"], scatter_data[return_4q], alpha=0.4, s=30)

    z = np.polyfit(scatter_data["Overvaluation_pct"], scatter_data[return_4q], 1)
    p = np.poly1d(z)
    x_line = np.linspace(
        scatter_data["Overvaluation_pct"].min(),
        scatter_data["Overvaluation_pct"].max(),
        100,
    )
    ax3.plot(x_line, p(x_line), "r--", linewidth=2, label="Trend line")

    ax3.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax3.axvline(x=0, color="gray", linestyle="--", alpha=0.5)
    ax3.set_xlabel("Current Overvaluation (%)", fontweight="bold")
    ax3.set_ylabel("1-Year Forward Return (%)", fontweight="bold")
    ax3.set_title("Overvaluation vs Future Returns (Scatter)")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    ax4 = axes[1, 1]
    long_short_returns = []
    horizon_labels = []

    for h in horizons:
        return_col = f"Return_{h}Q"
        decile_returns = df.groupby("Overval_Decile")[return_col].mean()
        if "D1" in decile_returns.index and "D10" in decile_returns.index:
            long_short = decile_returns["D1"] - decile_returns["D10"]
            long_short_returns.append(long_short)
            horizon_labels.append(f"{h}Q")

    if long_short_returns:
        ax4.bar(horizon_labels, long_short_returns, color="steelblue", alpha=0.7)
        ax4.axhline(y=0, color="red", linestyle="--", linewidth=2)
        ax4.set_xlabel("Forward Horizon", fontweight="bold")
        ax4.set_ylabel("Long-Short Return (%)", fontweight="bold")
        ax4.set_title("Long-Short Portfolio Returns\n(Buy D1 Undervalued - Sell D10 Overvalued)")
        ax4.grid(True, alpha=0.3, axis="y")

        for i, v in enumerate(long_short_returns):
            ax4.text(
                i,
                v + (1 if v > 0 else -1),
                f"{v:.1f}%",
                ha="center",
                va="bottom" if v > 0 else "top",
                fontweight="bold",
            )

    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT, dpi=300, bbox_inches="tight")
    print(f"\nPlot saved to: {OUTPUT_PLOT}")
    return fig


def export_results(df, quintile_analysis, decile_analysis, correlation_summary, decile_by_year_4q,
                   sector_under=None, sector_over=None, output_path=None):
    """Export all results to Excel."""
    output_path = output_path or (ROOT / OUTPUT_RESULTS_TEMPLATE.format(sector="fe_all"))
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        df[[
            "Ticker", "QuarterDate", "Overvaluation_pct", "Overval_Quintile",
            "Overval_Decile", "Return_1Q", "Return_2Q", "Return_4Q",
        ]].to_excel(writer, sheet_name="Full_Data", index=False)

        quintile_analysis.to_excel(writer, sheet_name="Quintile_Analysis")
        decile_analysis.to_excel(writer, sheet_name="Decile_Analysis")
        correlation_summary.to_excel(writer, sheet_name="Correlation_Tests", index=False)

        if decile_by_year_4q is not None:
            decile_by_year_4q.to_excel(writer, sheet_name="Decile_By_Year_4Q", index=False)
        if sector_under is not None:
            sector_under.to_excel(writer, sheet_name="Sector_Undervalued", index=False)
        if sector_over is not None:
            sector_over.to_excel(writer, sheet_name="Sector_Overvalued", index=False)

    print(f"\nResults exported to: {output_path}")
    return output_path


def run_analysis_for_sheet(sheet_name, sector_label, sector_code=None):
    print("\n" + "=" * 70)
    print(f"Processing sector {sector_label} (sheet: {sheet_name})")
    print("=" * 70)

    df_results, df_mc, value_col = load_results_and_market_data(sheet_name=sheet_name, sector_code=sector_code)
    df_results = create_overvaluation_groups(df_results)
    df = compute_forward_returns(df_results, df_mc, horizons=(1, 2, 4), value_col=value_col)
    df = create_overvaluation_groups(df)

    print(f"Loaded {len(df)} observations for sector {sector_code or 'ALL'}")
    print("[Step] Computing forward returns (1Q, 2Q, 4Q)...")
    print("[Step] Creating overvaluation groups...")
    print("[Step] Recent undervalued/overvalued names:")

    print_recent_undervalued(df)
    print_recent_sector_extremes(df)

    print("[Step] Mean reversion analysis...")
    quintile_analysis, decile_analysis, correlation_summary = analyze_mean_reversion(df, horizons=(1, 2, 4))
    decile_by_year_4q = decile_returns_by_year(df, horizon=4)

    last_quarters = get_recent_quarters(df, n_quarters=RECENT_QUARTERS)
    sector_under, sector_over = build_sector_extremes(df, last_quarters, top_n=SECTOR_TOP_N)

    output_path = ROOT / OUTPUT_RESULTS_TEMPLATE.format(
        sector="all" if sector_code is None else sector_code.lower()
    )
    export_results(
        df,
        quintile_analysis,
        decile_analysis,
        correlation_summary,
        decile_by_year_4q,
        sector_under=sector_under,
        sector_over=sector_over,
        output_path=output_path,
    )
    return df


def main():
    print("\n" + "=" * 70)
    print(" MEAN REVERSION VALIDATION TEST (BIST FE+TE)")
    print("Testing if fundamental overvaluation predicts future underperformance")
    print("=" * 70)

    # Overall FE+TE model
    df_all = run_analysis_for_sheet(SHEET_NAME, sector_label="Pooled FE+TE", sector_code=None)

    # Sector-level FE+TE models
    all_sectors = []
    for idx, code in enumerate(SECTOR_CODES, 1):
        sector_label = SECTOR_DISPLAY_NAMES.get(code, code)
        sheet = f"{code}_FE_All"
        df_sector = run_analysis_for_sheet(sheet, sector_label=sector_label, sector_code=code)
        if df_sector is not None and not df_sector.empty:
            all_sectors.append(df_sector.assign(SectorCode=code))

    if all_sectors:
        combined = pd.concat(all_sectors, ignore_index=True)
        combined.to_excel(ROOT / "bist_mean_reversion_analysis_fe_all_sectors.xlsx", index=False)
        print("\n[OK] Combined sector mean reversion saved: bist_mean_reversion_analysis_fe_all_sectors.xlsx")

    # Visualizations for pooled model
    if df_all is not None and not df_all.empty:
        plot_mean_reversion(df_all, horizons=(1, 2, 4))

    print("\n[OK] FE+TE mean reversion analysis complete.")


if __name__ == "__main__":
    main()
