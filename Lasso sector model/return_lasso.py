import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

RESULTS_PATH = ROOT / "Lasso sector model" / "bist_lasso_sector_models.xlsx"
MARKET_CAP_PATH = ROOT / "all_tickers_quarterly_2016_2026.csv"

OUTPUT_RESULTS_TEMPLATE = ROOT / "Lasso sector model" / "bist_mean_reversion_lasso_{sector}.xlsx"
OUTPUT_COMBINED = ROOT / "Lasso sector model" / "bist_mean_reversion_lasso_all_sectors.xlsx"
OUTPUT_PLOT = ROOT / "Lasso sector model" / "bist_mean_reversion_by_overvaluation_lasso.png"

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


def coerce_numeric(val):
    try:
        return float(val)
    except Exception:
        return np.nan


def load_market_prices():
    df_mc = pd.read_csv(MARKET_CAP_PATH)
    if "QuarterEnd" in df_mc.columns:
        price_cols = [c for c in df_mc.columns if c.endswith("_Close_QuarterEnd")]
        df_mc = df_mc[["QuarterEnd"] + price_cols].melt(
            id_vars=["QuarterEnd"],
            value_vars=price_cols,
            var_name="Ticker",
            value_name="Price",
        )
        df_mc["Ticker"] = df_mc["Ticker"].str.replace("_Close_QuarterEnd", "", regex=False)
        df_mc = df_mc.rename(columns={"QuarterEnd": "QuarterDate"})
    else:
        rename_map = {"ticker": "Ticker", "quarter_end": "QuarterDate", "price": "Price"}
        df_mc = df_mc.rename(columns={k: v for k, v in rename_map.items() if k in df_mc.columns})
    df_mc["QuarterDate"] = pd.to_datetime(df_mc["QuarterDate"]).dt.normalize()
    df_mc["Ticker"] = df_mc["Ticker"].astype(str).str.upper().str.replace(r"\\.IS$", "", regex=True)
    if "Price" in df_mc.columns:
        df_mc["Price"] = df_mc["Price"].apply(coerce_numeric)
    return df_mc


def compute_forward_returns(df, df_mc, horizons=(1, 2, 4)):
    df = df.merge(df_mc[["Ticker", "QuarterDate", "Price"]], on=["Ticker", "QuarterDate"], how="left")
    df = df.sort_values(["Ticker", "QuarterDate"])
    for h in horizons:
        fwd = df.groupby("Ticker")["Price"].shift(-h)
        base = df["Price"]
        valid = (base > 0) & (fwd > 0)
        df[f"Return_{h}Q"] = np.where(valid, (fwd / base - 1) * 100, np.nan)
        df[f"LogReturn_{h}Q"] = np.where(valid, np.log(fwd / base) * 100, np.nan)
    return df


def create_overvaluation_groups(df):
    df = df.copy()
    df["Overval_Quintile"] = pd.qcut(df["Overvaluation_pct"], 5, labels=[f"Q{i}" for i in range(1, 6)], duplicates="drop")
    df["Overval_Decile"] = pd.qcut(df["Overvaluation_pct"], 10, labels=[f"D{i}" for i in range(1, 11)], duplicates="drop")
    return df


def mean_reversion_analysis(df, horizons=(1, 2, 4)):
    results = []
    for h in horizons:
        ret_col = f"Return_{h}Q"
        log_ret_col = f"LogReturn_{h}Q"
        quintile = df.groupby("Overval_Quintile", observed=False).agg({
            ret_col: ["mean", "median", "std", "count"],
            log_ret_col: ["mean", "median"],
            "Overvaluation_pct": "mean",
        }).round(2)
        quintile.columns = [
            f"Mean_Return_{h}Q", f"Median_Return_{h}Q", f"Std_Return_{h}Q", f"N_Obs_{h}Q",
            f"Mean_LogReturn_{h}Q", f"Median_LogReturn_{h}Q", "Avg_Overval_pct",
        ]
        decile = df.groupby("Overval_Decile", observed=False).agg({
            ret_col: ["mean", "median", "count"],
            "Overvaluation_pct": "mean",
        }).round(2)
        decile.columns = [
            f"Mean_Return_{h}Q", f"Median_Return_{h}Q", f"N_Obs_{h}Q", "Avg_Overval_pct",
        ]
        valid = df[[ret_col, "Overvaluation_pct"]].dropna()
        corr = pval = np.nan
        if not valid.empty:
            corr, pval = stats.spearmanr(valid["Overvaluation_pct"], valid[ret_col])
        results.append((h, quintile, decile, corr, pval))
    return results


def export_sector(df, sector_code, analyses):
    out_path = OUTPUT_RESULTS_TEMPLATE.with_name(OUTPUT_RESULTS_TEMPLATE.name.format(sector=sector_code.lower()))
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="Full_Data", index=False)
        for h, quint, dec, corr, pval in analyses:
            quint.to_excel(writer, sheet_name=f"Quintile_{h}Q")
            dec.to_excel(writer, sheet_name=f"Decile_{h}Q")
    print(f"[OK] Sector {sector_code} mean reversion saved: {out_path}")
    return out_path


def plot_decile(df, ret_col="Return_4Q"):
    plt.figure(figsize=(10, 6))
    dec = df.groupby("Overval_Decile", observed=False)[ret_col].mean()
    dec.plot(kind="bar", color="#A23B72", edgecolor="black")
    plt.axhline(0, color="red", linestyle="--", linewidth=1.5)
    plt.title("Mean Forward Returns by Overvaluation Decile (Lasso)")
    plt.ylabel("Avg Forward Return (%)")
    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[OK] Plot saved: {OUTPUT_PLOT}")


def load_results_sheet(sheet, excel=None):
    if excel is None:
        excel = pd.ExcelFile(RESULTS_PATH)
    if sheet not in excel.sheet_names:
        print(f"[WARN] Missing sheet {sheet}")
        return None
    df = pd.read_excel(excel, sheet_name=sheet)
    df["QuarterDate"] = pd.to_datetime(df["QuarterDate"]).dt.normalize()
    df["Ticker"] = df["Ticker"].astype(str).str.upper().str.replace(r"\\.IS$", "", regex=True)
    return df


def main():
    df_mc = load_market_prices()
    combined = []

    xl = pd.ExcelFile(RESULTS_PATH)

    # Pooled
    pooled = load_results_sheet("Pooled_Lasso_All", excel=xl)
    if pooled is not None:
        pooled = compute_forward_returns(pooled, df_mc)
        pooled = create_overvaluation_groups(pooled)
        analyses = mean_reversion_analysis(pooled)
        export_sector(pooled, "all", analyses)
        combined.append(pooled.assign(SectorGroup="ALL"))

    for code in SECTOR_CODES:
        sheet = f"{code}_Lasso_All"
        df_sec = load_results_sheet(sheet, excel=xl)
        if df_sec is None:
            continue
        df_sec = compute_forward_returns(df_sec, df_mc)
        df_sec = create_overvaluation_groups(df_sec)
        analyses = mean_reversion_analysis(df_sec)
        export_sector(df_sec, code, analyses)
        combined.append(df_sec.assign(SectorGroup=code))

    if combined:
        df_all = pd.concat(combined, ignore_index=True)
        analyses_all = mean_reversion_analysis(df_all)
        with pd.ExcelWriter(OUTPUT_COMBINED, engine="openpyxl") as writer:
            df_all.to_excel(writer, sheet_name="Full_Data", index=False)
            for h, quint, dec, corr, pval in analyses_all:
                quint.to_excel(writer, sheet_name=f"Quintile_{h}Q")
                dec.to_excel(writer, sheet_name=f"Decile_{h}Q")
                pd.DataFrame(
                    {"SpearmanIC": [corr], "p_value": [pval]}
                ).to_excel(writer, sheet_name=f"IC_{h}Q", index=False)
        print(f"[OK] Combined mean reversion saved: {OUTPUT_COMBINED}")
        plot_decile(df_all)


if __name__ == "__main__":
    main()
