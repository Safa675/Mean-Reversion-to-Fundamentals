import os
import re
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Headless backend for environments without a display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]

FINTABLES_DIR = ROOT / "fintables_data_new"
FUNDAMENTAL_DATA_DIR = ROOT / "fundamental_data"
MARKET_CAP_PATH = ROOT / "all_tickers_quarterly_2016_2026.csv"
FUNDAMENTALS_OUTPUT = ROOT / "bist_quarterly_fundamentals.csv"
REBUILD_FUNDAMENTALS = os.getenv("REBUILD_FUNDAMENTALS", "0").lower() not in {"0", "false", "no"}
SHARES_CACHE_PATH = ROOT / "bist_quarterly_shares_from_prices.csv"
REBUILD_SHARES = os.getenv("REBUILD_SHARES", "0").lower() not in {"0", "false", "no"}

OUTPUT_XLSX = ROOT / "bist_bubbleness_analysis_pooled_ols.xlsx"
OUTPUT_TS = ROOT / "bist_bubbleness_timeseries_pooled_ols.png"
OUTPUT_DIST = ROOT / "bist_bubbleness_distribution_pooled_ols.png"
VIF_OUTPUT = ROOT / "bist_vif_table_pooled_ols.csv"
CORR_OUTPUT = ROOT / "bist_correlation_vs_log_mcap_pooled_ols.csv"
QUARTER_FREQ_RESAMPLE = "QE-DEC"  # use new alias to avoid pandas deprecation spam
QUARTER_FREQ_PERIOD = "Q-DEC"     # Period still expects the legacy alias
TARGET_RAW_COL = "MarketCap"
TARGET_COL = "LogMarketCap"

MIN_FACTOR_COVERAGE = 0.1
MIN_FACTOR_OBS = 500
MAX_LASSO_FEATURES = 8  # cap to avoid overfitting/collinearity

# Manual features used in the pooled OLS specification
MANUAL_FEATURES = [
    "TotalEquity",
    "OperatingIncome",
    "FreeCashFlow",
    "NetDebt",
    "Goodwill",
    "Revenue",
    "RD_Expense",
    "FCF_Level",
]

GROWTH_COMPONENTS = ["Revenue_Growth_z", "EPS_Growth_z", "FCF_Growth_z"]
PROFIT_MARGIN_COMPONENTS = ["Gross_Margin_z", "Operating_Margin_z", "EBITDA_Margin_z", "Net_Margin_z"]
CAP_EFF_COMPONENTS = ["AssetTurnover_z", "FCF_to_Equity_z"]

# Human-friendly sector display names (ASCII to avoid Windows console encoding issues)
SECTOR_DISPLAY_NAMES = {
    "CAP": "CAP - Asset-Heavy - Capital Intensive",
    "COM": "COM - Commodity - Energy - Mining",
    "CON": "CON - Consumer - Retail - Food",
    "FIN": "FIN - Financial Balance Sheet Businesses",
    "IND": "IND - Manufacturing & Industrials",
    "INT": "INT - Intangible - Knowledge Driven",
    "RE": "RE - Real Estate - Asset Revaluation (GYO, tourism property-heavy)",
}


def factor_mean(df, cols):
    """Calculate mean of available columns, requiring at least 1 valid value"""
    cols = [c for c in cols if c in df.columns]
    if not cols:
        return np.nan
    # Require at least 1 non-nan value per row
    result = df[cols].mean(axis=1, skipna=True)
    # Set to NaN if all values were NaN
    valid_count = df[cols].notna().sum(axis=1)
    result[valid_count == 0] = np.nan
    return result


def drop_constant_columns(df, cols, tol=1e-12):
    keep = []
    dropped = []
    for c in cols:
        if c not in df.columns:
            continue
        series = df[c].dropna()
        if series.empty or series.var() < tol:
            dropped.append(c)
        else:
            keep.append(c)
    return keep, dropped


def safe_log1p_signed(series: pd.Series) -> pd.Series:
    """Stabilize scale by log1p on absolute values while preserving sign."""
    return np.sign(series) * np.log1p(np.abs(series))


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


def pick_date_column(df):
    exact_matches = {"date", "tarih"}
    contains_matches = {"date", "tarih"}
    lower_cols = {col: str(col).lower() for col in df.columns}

    for col, low in lower_cols.items():
        if low in exact_matches:
            return col

    for col, low in lower_cols.items():
        if any(token in low for token in contains_matches):
            return col

    raise ValueError(f"Could not find a date column. Columns: {list(df.columns)}")


def pick_shares_column(df):
    candidates = [
        "SERMAYE", "Sermaye", "sermaye",
        "Shares", "shares", "shares_outstanding", "outstanding_shares",
    ]
    for c in candidates:
        if c in df.columns:
            return c

    lower_cols = {col: str(col).lower() for col in df.columns}
    for col, low in lower_cols.items():
        if "sermay" in low or "share" in low:
            return col

    raise ValueError(f"Could not find a shares/outstanding column. Columns: {list(df.columns)}")


def load_quarterly_shares_from_prices(price_dir, cache_path=SHARES_CACHE_PATH, rebuild=REBUILD_SHARES):
    price_dir = Path(price_dir)
    cache_path = Path(cache_path) if cache_path is not None else None

    if cache_path and cache_path.exists() and not rebuild:
        try:
            df_cached = pd.read_csv(cache_path, parse_dates=["QuarterDate"])
            df_cached["QuarterDate"] = pd.to_datetime(df_cached["QuarterDate"]).dt.normalize()
            print(f"Using cached quarterly shares from {cache_path} ({len(df_cached)} rows)")
            return df_cached[["Ticker", "QuarterDate", "Shares"]]
        except Exception as exc:
            print(f"[WARN] Failed to read shares cache {cache_path}: {exc}. Rebuilding from price files...")

    frames = []
    if not price_dir.exists():
        print(f"[WARN] Price directory missing for shares extraction: {price_dir}")
        return pd.DataFrame(columns=["Ticker", "QuarterDate", "Shares"])

    price_files = sorted(price_dir.glob("*_daily_and_quarterly.xlsx"))
    total_files = len(price_files)
    if total_files == 0:
        print(f"[WARN] No price files found in {price_dir}")
        return pd.DataFrame(columns=["Ticker", "QuarterDate", "Shares"])

    def _process_price_file(path):
        ticker = path.stem.split("_")[0].upper()
        try:
            df_daily = pd.read_excel(
                path,
                sheet_name="daily",
                usecols=lambda c: any(tok in str(c).lower() for tok in ("tarih", "date", "sermay", "share")),
            )
        except Exception as exc:
            print(f"[WARN] {ticker}: failed to read daily sheet for shares ({exc})")
            return None

        try:
            date_col = pick_date_column(df_daily)
            shares_col = pick_shares_column(df_daily)
        except Exception as exc:
            print(f"[WARN] {ticker}: {exc}")
            return None

        df_daily[date_col] = pd.to_datetime(df_daily[date_col], dayfirst=True, errors="coerce")
        df_daily = df_daily.dropna(subset=[date_col]).sort_values(date_col)

        df_daily[shares_col] = pd.to_numeric(df_daily[shares_col], errors="coerce").ffill()
        if df_daily[shares_col].dropna().empty:
            print(f"[WARN] {ticker}: no usable shares data in {shares_col}")
            return None

        shares_q = (
            df_daily.set_index(date_col)[shares_col]
            .dropna()
            .resample(QUARTER_FREQ_RESAMPLE)
            .last()
            .reset_index()
            .rename(columns={shares_col: "Shares", date_col: "QuarterDate"})
        )
        shares_q["Ticker"] = ticker
        shares_q["QuarterDate"] = pd.to_datetime(shares_q["QuarterDate"]).dt.normalize()
        return shares_q[["Ticker", "QuarterDate", "Shares"]]

    max_workers = min(8, max(1, (os.cpu_count() or 2)))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_path = {executor.submit(_process_price_file, path): path for path in price_files}
        for idx, future in enumerate(as_completed(future_to_path), 1):
            path = future_to_path[future]
            try:
                shares_df = future.result()
            except Exception as exc:
                print(f"[WARN] {path.stem.upper()}: unexpected error while processing shares ({exc})")
                continue

            if shares_df is not None and not shares_df.empty:
                frames.append(shares_df)

            if idx % 50 == 0 or idx == total_files:
                print(f"  Processed {idx}/{total_files} price files for quarterly shares")

    if not frames:
        return pd.DataFrame(columns=["Ticker", "QuarterDate", "Shares"])

    result = pd.concat(frames, ignore_index=True)
    if cache_path:
        try:
            result.to_csv(cache_path, index=False)
            print(f"Saved quarterly shares cache to {cache_path}")
        except Exception as exc:
            print(f"[WARN] Failed to write shares cache {cache_path}: {exc}")
    return result


def build_bubble_outputs(model_name, y_log_mcap, predicted_log_mcap, base_df):
    if isinstance(predicted_log_mcap, pd.DataFrame):
        predicted_log_mcap = predicted_log_mcap.iloc[:, 0]

    predicted_log_mcap = pd.Series(predicted_log_mcap, index=base_df.index, name="Predicted_LogMCap")
    actual_log_mcap = pd.Series(y_log_mcap, index=base_df.index, name="Actual_LogMCap")

    actual_mcap = base_df.get(TARGET_RAW_COL)
    if actual_mcap is None:
        actual_mcap = np.exp(actual_log_mcap)
    predicted_mcap = np.exp(predicted_log_mcap)

    residuals_log = actual_log_mcap - predicted_log_mcap
    # Center residuals to allow over/undervaluation symmetry, then scale
    residuals_centered = residuals_log - np.nanmedian(residuals_log)
    overvaluation_pct = residuals_centered * 100

    df_results = base_df.copy()
    df_results.loc[:, "Predicted_LogMCap"] = predicted_log_mcap
    df_results.loc[:, "Actual_LogMCap"] = actual_log_mcap
    df_results.loc[:, "Residual_LogMCap"] = residuals_log
    df_results.loc[:, "Predicted_MarketCap"] = predicted_mcap
    df_results.loc[:, "Actual_MarketCap"] = actual_mcap
    df_results.loc[:, "Overvaluation_pct"] = overvaluation_pct
    df_results.loc[:, "Model"] = model_name
    if "SectorGroup" in base_df.columns:
        df_results.loc[:, "SectorGroup"] = base_df["SectorGroup"]
    return df_results


def analyze_bubbles(model_name, df_results, top_n=10, verbose=True):
    stats_block = {}
    if verbose:
        print("\n" + "=" * 60)
        print(f"{model_name} - BUBBLENESS ANALYSIS")
        print("=" * 60)

    if verbose:
        print("\nRESIDUAL STATISTICS (log market cap):")
        print(df_results["Residual_LogMCap"].describe())

        print("\nOVERVALUATION STATISTICS (%):")
        print(df_results["Overvaluation_pct"].describe())

    df_reset = df_results.reset_index()
    bubble_by_quarter = df_reset.groupby("QuarterDate").agg({
        "Overvaluation_pct": ["mean", "median", "std"],
        "Residual_LogMCap": ["mean", "median"],
    }).reset_index()
    bubble_by_quarter.columns = [
        "QuarterDate", "Mean_Overval_pct", "Median_Overval_pct",
        "Std_Overval_pct", "Mean_Residual_LogMCap", "Median_Residual_LogMCap",
    ]

    bubble_by_ticker = df_reset.groupby("Ticker").agg({
        "Overvaluation_pct": ["mean", "median", "std", "count"],
        "Residual_LogMCap": "mean",
    }).reset_index()
    bubble_by_ticker.columns = [
        "Ticker", "Mean_Overval_pct", "Median_Overval_pct",
        "Std_Overval_pct", "Obs_Count", "Mean_Residual_LogMCap",
    ]

    top_overvalued = df_reset.nlargest(top_n, "Overvaluation_pct")[
        ["Ticker", "QuarterDate", "Actual_MarketCap", "Predicted_MarketCap",
         "Residual_LogMCap", "Overvaluation_pct"]
    ]
    top_undervalued = df_reset.nsmallest(top_n, "Overvaluation_pct")[
        ["Ticker", "QuarterDate", "Actual_MarketCap", "Predicted_MarketCap",
         "Residual_LogMCap", "Overvaluation_pct"]
    ]

    if verbose:
        print("\n--- TOP 10 MOST OVERVALUED OBSERVATIONS ---")
        print(top_overvalued.to_string(index=False))

        print("\n--- TOP 10 MOST UNDERVALUED OBSERVATIONS ---")
        print(top_undervalued.to_string(index=False))

        print("\n--- BUBBLENESS OVER TIME (by Quarter) ---")
        print(bubble_by_quarter.tail(20).to_string(index=False))

        print("\n--- TOP 10 MOST CONSISTENTLY OVERVALUED COMPANIES ---")
        print(bubble_by_ticker.nlargest(10, "Mean_Overval_pct").to_string(index=False))

        print("\n--- TOP 10 MOST CONSISTENTLY UNDERVALUED COMPANIES ---")
        print(bubble_by_ticker.nsmallest(10, "Mean_Overval_pct").to_string(index=False))

    stats_block["Mean_Overval_pct"] = df_results["Overvaluation_pct"].mean()
    stats_block["Std_Overval_pct"] = df_results["Overvaluation_pct"].std()
    return {
        "bubble_by_quarter": bubble_by_quarter,
        "bubble_by_ticker": bubble_by_ticker,
        "top_overvalued": top_overvalued,
        "top_undervalued": top_undervalued,
        "stats": stats_block,
    }


def collect_model_metrics(model_name, result_obj):
    metrics = {"Model": model_name, "R2_overall": np.nan, "R2_within": np.nan, "R2_between": np.nan, "Adj_R2": np.nan, "N": np.nan}
    if hasattr(result_obj, "nobs"):
        metrics["N"] = result_obj.nobs
    r2_obj = getattr(result_obj, "rsquared", None)
    if r2_obj is None:
        return metrics
    if hasattr(r2_obj, "overall"):
        metrics["R2_overall"] = r2_obj.overall
        metrics["R2_within"] = r2_obj.within
        metrics["R2_between"] = r2_obj.between
    else:
        try:
            metrics["R2_overall"] = float(r2_obj)
        except Exception:
            metrics["R2_overall"] = np.nan
        metrics["Adj_R2"] = getattr(result_obj, "rsquared_adj", np.nan)
    return metrics


def compute_vif(df, vars):
    X = df[vars].dropna().astype(float)
    vif_data = pd.DataFrame()
    vif_data["Variable"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    return vif_data


def export_vif_and_corr(df_common, regressors, label, vif_path, corr_path, target_col=TARGET_COL):
    if df_common is None or df_common.empty or not regressors:
        print(f"[WARN] Skipping VIF for {label}: missing data or regressors")
        return None, None

    df_sub = df_common.dropna(subset=regressors + [target_col])
    if df_sub.empty or len(df_sub) <= len(regressors):
        print(f"[WARN] Skipping VIF for {label}: insufficient rows after dropna")
        return None, None

    print("\n" + "="*60)
    print(f"MULTICOLLINEARITY DIAGNOSTICS ({label})")
    print("="*60)

    vif_results = compute_vif(df_sub, regressors)
    print("\nVariance Inflation Factors (VIF):")
    print(vif_results.to_string(index=False))
    print("\nNote: VIF > 10 suggests high multicollinearity")

    corr_df = df_sub[regressors + [target_col]].dropna()
    corr_series = corr_df.corr()[target_col].drop(target_col).sort_values(ascending=False)
    print(f"\nCorrelations with {target_col}:")
    print(corr_series.to_string())

    vif_path = Path(vif_path)
    corr_path = Path(corr_path)
    vif_results.to_csv(vif_path, index=False)
    corr_series.to_csv(corr_path, header=["corr"])
    print(f"\n[OK] VIF saved to {vif_path}")
    print(f"[OK] Correlations saved to {corr_path}")
    return vif_path, corr_path


def print_factor_composition(factor_map, regressors):
    print("\nFACTOR COMPOSITION (sub-components used)")
    for fac in regressors:
        comps = factor_map.get(fac, [])
        if comps:
            print(f" - {fac}: {', '.join(comps)}")
        else:
            print(f" - {fac}: [no sub-components listed]")


def select_common_sample(df_p, factors, min_ratio=MIN_FACTOR_COVERAGE, min_obs=MIN_FACTOR_OBS, target_col=TARGET_COL):
    available = [f for f in factors if f in df_p.columns]
    total = len(df_p)
    counts = {f: df_p[f].notna().sum() for f in available}
    selected = [
        f for f in available
        if counts.get(f, 0) >= min_obs and (counts.get(f, 0) / total) >= min_ratio
    ]
    dropped = [f for f in available if f not in selected]
    if not selected:
        selected = available[:]
        dropped = []
    while selected:
        df_common = df_p.dropna(subset=[target_col] + selected).copy()
        if not df_common.empty:
            return selected, df_common, dropped, counts
        lowest = min(selected, key=lambda f: counts.get(f, 0))
        selected.remove(lowest)
        if lowest not in dropped:
            dropped.append(lowest)
    return [], df_p.head(0), dropped, counts


def optimize_factor_coverage(df_p, factors, min_factors=3, min_gain=500, target_col=TARGET_COL):
    """Greedily drop the lowest-coverage factor if it meaningfully boosts sample size."""
    current = list(factors)
    coverage = {f: df_p[f].notna().sum() for f in current}
    best_sample = df_p.dropna(subset=[target_col] + current).shape[0]
    improved = False

    while len(current) > min_factors:
        worst = min(current, key=lambda f: coverage.get(f, 0))
        candidate = [f for f in current if f != worst]
        candidate_sample = df_p.dropna(subset=[target_col] + candidate).shape[0]
        if candidate_sample > best_sample + min_gain:
            print(f"[WARN] Removing low-coverage factor {worst} increases sample {best_sample} -> {candidate_sample}")
            current = candidate
            best_sample = candidate_sample
            improved = True
            coverage = {f: df_p[f].notna().sum() for f in current}
        else:
            break

    return current, improved, best_sample


def add_yoy_growth(df, base_col, out_col, group_col="Ticker", period_col="QuarterPeriod", lag=4):
    """Add YoY growth calculation with proper handling of missing data"""
    if base_col not in df.columns:
        df[out_col] = np.nan
        return df
    
    # Sort to ensure proper ordering
    df = df.sort_values([group_col, period_col])
    
    # Create lagged column
    df[f"{out_col}_lag"] = df.groupby(group_col)[base_col].shift(lag)
    
    # Calculate growth
    df[out_col] = (df[base_col] / df[f"{out_col}_lag"]) - 1
    
    # Handle division by zero or negative denominators
    df.loc[df[f"{out_col}_lag"] <= 0, out_col] = np.nan
    df.loc[df[f"{out_col}_lag"].isna(), out_col] = np.nan
    
    # Drop temporary column
    df = df.drop(columns=[f"{out_col}_lag"])
    
    return df


def normalize_label(text):
    if text is None:
        return ""
    text = str(text).translate(
        str.maketrans(
            {
                "\u0131": "i",
                "\u0130": "i",
                "\u011f": "g",
                "\u011e": "g",
                "\u015f": "s",
                "\u015e": "s",
                "\u00f6": "o",
                "\u00d6": "o",
                "\u00fc": "u",
                "\u00dc": "u",
                "\u00e7": "c",
                "\u00c7": "c",
            }
        )
    )
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"[^a-zA-Z0-9]+", " ", text).lower()
    return " ".join(text.split())


def parse_quarter_label(label):
    match = re.search(r"(\d{4})\s*/\s*(\d{1,2})", str(label))
    if not match:
        return None
    year = int(match.group(1))
    month = int(match.group(2))
    if month not in (3, 6, 9, 12):
        return None
    day = 31 if month in (3, 12) else 30
    return pd.Timestamp(year, month, day)


def find_sheet(xls, candidates):
    normalized = {normalize_label(name): name for name in xls.sheet_names}
    for cand in candidates:
        if cand in normalized:
            return normalized[cand]
    for cand in candidates:
        for norm_name, orig_name in normalized.items():
            if cand in norm_name:
                return orig_name
    return None


def statement_long(file_path, sheet_name):
    if sheet_name is None:
        return pd.DataFrame(columns=["QuarterDate", "item_norm", "value"])
    df = pd.read_excel(file_path, sheet_name=sheet_name)
    if df.empty:
        return pd.DataFrame(columns=["QuarterDate", "item_norm", "value"])
    if "Kalem" not in df.columns:
        df = df.rename(columns={df.columns[0]: "Kalem"})
    df = df.dropna(subset=["Kalem"])
    q_cols = [c for c in df.columns if re.match(r"^\d{4}/\d{1,2}$", str(c).strip())]
    if not q_cols:
        return pd.DataFrame(columns=["QuarterDate", "item_norm", "value"])
    df_long = df.melt(id_vars=["Kalem"], value_vars=q_cols, var_name="Quarter", value_name="value")
    df_long["QuarterDate"] = df_long["Quarter"].apply(parse_quarter_label)
    df_long["value"] = pd.to_numeric(df_long["value"], errors="coerce")
    df_long["item_norm"] = df_long["Kalem"].apply(normalize_label)
    df_long = df_long.dropna(subset=["QuarterDate"])
    return df_long[["QuarterDate", "item_norm", "value"]]

def pivot_statement(df, prefix):
    """Pivot a long-form financial statement into wide columns with a prefix."""
    if df is None or df.empty:
        return pd.DataFrame()
    pivot = df.pivot_table(index="QuarterDate", columns="item_norm", values="value", aggfunc="first")
    pivot = pivot.reset_index()
    pivot.columns = ["QuarterDate"] + [f"{prefix}{str(c).replace(' ', '_')}" for c in pivot.columns if c != "QuarterDate"]
    return pivot


def series_from_candidates(df, candidates, sum_matches=False):
    if df is None or df.empty:
        return None
    if sum_matches:
        exact = df[df["item_norm"].isin(candidates)]
        if exact.empty:
            mask = False
            for cand in candidates:
                mask = mask | df["item_norm"].str.contains(cand, na=False)
            exact = df[mask]
        if exact.empty:
            return None
        return exact.groupby("QuarterDate")["value"].sum(min_count=1)

    for cand in candidates:
        exact = df[df["item_norm"] == cand]
        if not exact.empty:
            return exact.groupby("QuarterDate")["value"].first()
    for cand in candidates:
        contains = df[df["item_norm"].str.contains(cand, na=False)]
        if not contains.empty:
            return contains.groupby("QuarterDate")["value"].first()
    return None


def build_ticker_fundamentals(file_path, ticker):
    xls = pd.ExcelFile(file_path)
    balance_sheet = find_sheet(xls, ["bilanco"])
    income_sheet = find_sheet(xls, ["gelir tablosu ceyreklik", "gelir tablosu donemsel"])
    cash_sheet = find_sheet(xls, ["nakit akis ceyreklik", "nakit akis donemsel"])

    balance_df = statement_long(file_path, balance_sheet)
    income_df = statement_long(file_path, income_sheet)
    cash_df = statement_long(file_path, cash_sheet)

    # Wide pivots of every line item to keep full fundamental coverage
    balance_pivot = pivot_statement(balance_df, "BS_")
    income_pivot = pivot_statement(income_df, "IS_")
    cash_pivot = pivot_statement(cash_df, "CF_")
    raw_pivot = None
    for pivot_df in (balance_pivot, income_pivot, cash_pivot):
        if pivot_df.empty:
            continue
        if raw_pivot is None:
            raw_pivot = pivot_df
        else:
            raw_pivot = raw_pivot.merge(pivot_df, on="QuarterDate", how="outer")

    total_assets = series_from_candidates(
        balance_df, ["toplam varliklar", "toplam kaynaklar"]
    )
    total_equity = series_from_candidates(balance_df, ["toplam ozkaynaklar"])
    cash_eq = series_from_candidates(balance_df, ["nakit ve nakit benzerleri"])
    financial_debt = series_from_candidates(
        balance_df, ["finansal borclar"], sum_matches=True
    )
    goodwill = series_from_candidates(balance_df, ["serefiye"])

    revenue = series_from_candidates(
        income_df,
        ["satis gelirleri", "toplam hasilat", "net faiz geliri veya gideri", "faaliyet brut kari"],
    )
    cost_of_revenue = series_from_candidates(income_df, ["satislarin maliyeti"])
    gross_profit = series_from_candidates(
        income_df,
        ["brut kar zarar", "ticari faaliyetlerden brut kar zarar", "faaliyet brut kari"],
    )
    operating_income = series_from_candidates(
        income_df,
        [
            "faaliyet kari zarari",
            "esas faaliyet kari zarari",
            "net faaliyet kari zarari",
            "finansman geliri gideri oncesi faaliyet kari zarari",
        ],
    )
    net_income = series_from_candidates(
        income_df,
        ["donem kari zarari", "donem net kari veya zarari", "surdurulen faaliyetler donem net kari zarari"],
    )
    rd_expense = series_from_candidates(income_df, ["arastirma ve gelistirme giderleri"])
    amort_income = series_from_candidates(income_df, ["amortisman"])
    amort_cash = series_from_candidates(cash_df, ["amortisman ve itfa gideri ile ilgili duzeltmeler"])
    ebitda_direct = series_from_candidates(income_df, ["favok", "ebitda"])

    cfo = series_from_candidates(
        cash_df, ["isletme faaliyetlerinden nakit akislar", "faaliyetlerden elde edilen nakit akislar"]
    )
    capex = series_from_candidates(
        cash_df,
        [
            "maddi ve maddi olmayan duran varliklarin alimindan kaynaklanan nakit cikislari",
            "maddi duran varliklarin alimindan kaynaklanan nakit cikislari",
            "maddi olmayan duran varliklarin alimindan kaynaklanan nakit cikislari",
        ],
        sum_matches=True,
    )

    if gross_profit is None and revenue is not None and cost_of_revenue is not None:
        gross_profit = revenue - cost_of_revenue

    ebitda = ebitda_direct
    if ebitda is None and operating_income is not None:
        if amort_income is not None:
            ebitda = operating_income.add(amort_income, fill_value=0)
        elif amort_cash is not None:
            ebitda = operating_income.add(amort_cash, fill_value=0)

    free_cash_flow = None
    if cfo is not None:
        free_cash_flow = cfo.copy()
        if capex is not None:
            free_cash_flow = cfo.add(capex, fill_value=0)

    net_debt = None
    if financial_debt is not None and cash_eq is not None:
        net_debt = financial_debt.sub(cash_eq, fill_value=0)
    elif financial_debt is not None:
        net_debt = financial_debt
    elif cash_eq is not None:
        net_debt = cash_eq.mul(-1)

    series_map = {
        "TotalAssets": total_assets,
        "TotalEquity": total_equity,
        "Revenue": revenue,
        "CostOfRevenue": cost_of_revenue,
        "GrossProfit": gross_profit,
        "OperatingIncome": operating_income,
        "NetIncome": net_income,
        "RD_Expense": rd_expense,
        "FreeCashFlow": free_cash_flow,
        "FCF_Level": free_cash_flow,
        "NetDebt": net_debt,
        "Goodwill": goodwill,
        "EBITDA": ebitda,
    }

    series_map = {k: v for k, v in series_map.items() if v is not None}
    df_manual = pd.DataFrame(series_map) if series_map else pd.DataFrame()
    if not df_manual.empty:
        df_manual = df_manual.reset_index().rename(columns={"index": "QuarterDate"})

    # Combine manual selections with raw pivoted fundamentals
    combined = None
    for candidate in (df_manual, raw_pivot):
        if candidate is None or candidate.empty:
            continue
        if combined is None:
            combined = candidate
        else:
            combined = combined.merge(candidate, on="QuarterDate", how="outer")

    if combined is None or combined.empty:
        return pd.DataFrame()

    combined["Ticker"] = ticker
    combined["QuarterDate"] = pd.to_datetime(combined["QuarterDate"]).dt.normalize()
    combined["Quarter"] = combined["QuarterDate"].apply(
        lambda d: f"Q{((d.month - 1) // 3) + 1} {d.year}"
    )
    return combined


def load_fintables_fundamentals(data_dir):
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Missing fintables folder: {data_dir}")
    frames = []
    stock_dirs = sorted([p for p in data_dir.iterdir() if p.is_dir()])
    for stock_dir in stock_dirs:
        ticker = stock_dir.name.upper()
        file_path = stock_dir / f"{ticker}.xlsx"
        if not file_path.exists():
            candidates = list(stock_dir.glob("*.xlsx"))
            if not candidates:
                print(f"[WARN] {ticker}: missing xlsx file")
                continue
            file_path = candidates[0]
        try:
            df = build_ticker_fundamentals(file_path, ticker)
        except Exception as exc:
            print(f"[WARN] {ticker}: {exc}")
            continue
        if not df.empty:
            df["SectorGroup"] = "FINTABLES"
            df["_priority"] = 1  # lower priority than curated sector files
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def load_sector_fundamentals(sector_dir, sector_label, priority=0):
    frames = []
    for file_path in sorted(Path(sector_dir).glob("*.xlsx")):
        ticker = file_path.stem.upper()
        try:
            df = build_ticker_fundamentals(file_path, ticker)
        except Exception as exc:
            print(f"[WARN] {ticker} ({sector_label}): {exc}")
            continue
        if df.empty:
            continue
        df["SectorGroup"] = sector_label
        df["_priority"] = priority
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def combine_fundamentals_with_priority(frames):
    df_all = pd.concat(frames, ignore_index=True)
    df_all["QuarterDate"] = pd.to_datetime(df_all["QuarterDate"], errors="coerce").dt.normalize()
    df_all = df_all.dropna(subset=["QuarterDate", "Ticker"])
    df_all = df_all.sort_values(["Ticker", "QuarterDate", "_priority"])

    def _merge_group(group):
        result = {
            "Ticker": group.name[0],
            "QuarterDate": group.name[1],
            "SectorGroup": next((v for v in group["SectorGroup"] if pd.notna(v)), np.nan),
        }
        for col in group.columns:
            if col in {"Ticker", "QuarterDate", "_priority", "SectorGroup"}:
                continue
            series = group[col].dropna()
            result[col] = series.iloc[0] if not series.empty else np.nan
        return pd.Series(result)

    group_obj = df_all.groupby(["Ticker", "QuarterDate"], as_index=False)
    try:
        merged = group_obj.apply(_merge_group, include_groups=False).reset_index(drop=True)
    except TypeError:
        # pandas < 2.2 fallback
        merged = group_obj.apply(_merge_group).reset_index(drop=True)
    merged["Quarter"] = merged["QuarterDate"].apply(lambda d: f"Q{((d.month - 1) // 3) + 1} {d.year}")
    return merged


def load_all_fundamentals():
    frames = []
    if FUNDAMENTAL_DATA_DIR.exists():
        sector_dirs = sorted([p for p in FUNDAMENTAL_DATA_DIR.iterdir() if p.is_dir()])
        for sector_dir in sector_dirs:
            sector_name = sector_dir.name.replace("\u2014", "-")
            sector_code = re.split(r"[-]", sector_name)[0].strip().split()[0]
            df_sector = load_sector_fundamentals(sector_dir, sector_code, priority=0)
            if not df_sector.empty:
                frames.append(df_sector)
            else:
                print(f"[WARN] No data loaded for sector folder: {sector_dir}")
    else:
        print(f"[WARN] Sector folder missing: {FUNDAMENTAL_DATA_DIR}")

    if FINTABLES_DIR.exists():
        frames.append(load_fintables_fundamentals(FINTABLES_DIR))
    else:
        print(f"[WARN] Fintables folder missing: {FINTABLES_DIR}")

    frames = [f for f in frames if f is not None and not f.empty]
    if not frames:
        raise ValueError("No fundamental data loaded from either sector folders or fintables.")

    merged = combine_fundamentals_with_priority(frames)
    print(f"Loaded fundamentals across sources: {len(merged)} records, {merged['Ticker'].nunique()} tickers")
    return merged


def prepare_panel_data():
    if not MARKET_CAP_PATH.exists():
        raise FileNotFoundError(f"Missing market cap file: {MARKET_CAP_PATH}")

    df_edgar = None
    built_from_source = False
    use_cache = FUNDAMENTALS_OUTPUT.exists() and not REBUILD_FUNDAMENTALS
    if use_cache:
        print(f"Loading fundamentals from cached CSV: {FUNDAMENTALS_OUTPUT}")
        df_edgar = pd.read_csv(FUNDAMENTALS_OUTPUT)
        if df_edgar.empty:
            print("[WARN] Cached fundamentals empty, rebuilding from sources.")
            df_edgar = None
        elif "SectorGroup" not in df_edgar.columns:
            print("[WARN] Cached fundamentals missing SectorGroup; rebuilding from sector folders.")
            df_edgar = None

    if df_edgar is None:
        built_from_source = True
        print("Loading fundamentals data from sector folders and fintables...")
        df_edgar = load_all_fundamentals()
        if df_edgar.empty:
            raise ValueError("No fundamentals data loaded from provided sources.")
        print(f"Loaded {len(df_edgar)} fundamental records")
    else:
        print(f"Loaded {len(df_edgar)} fundamental records from cache")

    if "QuarterDate" not in df_edgar.columns or "Ticker" not in df_edgar.columns:
        raise ValueError("Fundamentals file missing required columns: QuarterDate or Ticker.")
    df_edgar["QuarterDate"] = pd.to_datetime(df_edgar["QuarterDate"], errors="coerce").dt.normalize()
    df_edgar["Ticker"] = df_edgar["Ticker"].astype(str).str.upper().str.replace(r"\.IS$", "", regex=True)
    if "SectorGroup" not in df_edgar.columns:
        df_edgar["SectorGroup"] = np.nan
    df_edgar["Year"] = df_edgar["QuarterDate"].dt.year
    if built_from_source:
        df_edgar.to_csv(FUNDAMENTALS_OUTPUT, index=False)
        print(f"Saved fundamentals to {FUNDAMENTALS_OUTPUT}")

    print("Loading market data (price/shares)...")
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
        if "Sermaye" in df_mc.columns:
            rename_map["Sermaye"] = "Shares"
        df_mc = df_mc.rename(columns=rename_map)

    if "Price" not in df_mc.columns:
        raise ValueError("Price column not found in all_tickers_quarterly_2016_2026.csv.")

    df_mc["QuarterDate"] = pd.to_datetime(df_mc["QuarterDate"], errors="coerce").dt.normalize()
    df_mc["Ticker"] = df_mc["Ticker"].astype(str).str.upper().str.replace(r"\.IS$", "", regex=True)
    df_mc["Price"] = df_mc["Price"].apply(coerce_numeric)
    if "MarketCap" in df_mc.columns:
        df_mc["MarketCap"] = df_mc["MarketCap"].apply(coerce_numeric)
    if "Shares" in df_mc.columns:
        df_mc["Shares"] = df_mc["Shares"].apply(coerce_numeric)

    shares_from_prices = load_quarterly_shares_from_prices(ROOT / "isyatirim_prices")
    if not shares_from_prices.empty:
        df_mc = df_mc.merge(
            shares_from_prices,
            on=["Ticker", "QuarterDate"],
            how="left",
            suffixes=("", "_from_prices"),
        )
        if "Shares_from_prices" in df_mc.columns:
            if "Shares" not in df_mc.columns:
                df_mc["Shares"] = df_mc["Shares_from_prices"]
            else:
                df_mc["Shares"] = df_mc["Shares"].fillna(df_mc["Shares_from_prices"])
            df_mc = df_mc.drop(columns=["Shares_from_prices"])

    if "Shares" not in df_mc.columns or df_mc["Shares"].dropna().empty:
        raise ValueError("Shares data is missing; could not compute market cap from price * shares.")

    df_mc["Shares"] = df_mc["Shares"].apply(coerce_numeric)

    df_mc = df_mc.dropna(subset=["QuarterDate", "Price", "Shares"])
    df_mc = df_mc[(df_mc["Price"] > 0) & (df_mc["Shares"] > 0)]

    df_mc["MarketCap_from_px_shares"] = df_mc["Price"] * df_mc["Shares"]
    if "MarketCap" not in df_mc.columns:
        df_mc["MarketCap"] = df_mc["MarketCap_from_px_shares"]
    else:
        df_mc["MarketCap"] = df_mc["MarketCap"].fillna(df_mc["MarketCap_from_px_shares"])
    df_mc = df_mc.drop(columns=["MarketCap_from_px_shares"])
    df_mc = df_mc[df_mc["MarketCap"] > 0]

    print(f"Loaded {len(df_mc)} price/share records")

    print("Merging fundamentals with market price...")
    merge_cols = ["Ticker", "QuarterDate", "Price", "Shares", "MarketCap"]
    df_edgar = df_edgar.merge(
        df_mc[merge_cols],
        on=["Ticker", "QuarterDate"],
        how="inner",
    )

    print(f"After merge: {len(df_edgar)} records")
    if df_edgar.empty:
        raise ValueError("No records after merging fundamentals with market price. Check ticker and date alignment.")

    # Ensure required columns exist
    if "FCF_Level" not in df_edgar.columns and "FreeCashFlow" in df_edgar.columns:
        df_edgar["FCF_Level"] = df_edgar["FreeCashFlow"]
    if "RD_Expense" not in df_edgar.columns:
        df_edgar["RD_Expense"] = 0  # Default to 0 instead of NaN

    df_edgar["QuarterPeriod"] = df_edgar["QuarterDate"].dt.to_period(QUARTER_FREQ_PERIOD)

    # Prepare growth base columns with better defaults
    df_edgar["Revenue_For_Growth"] = df_edgar.get("Revenue", pd.Series(np.nan, index=df_edgar.index))
    df_edgar["EPS_For_Growth"] = df_edgar.get("EPSDiluted", df_edgar.get("EPSBasic", pd.Series(np.nan, index=df_edgar.index)))
    df_edgar["FCF_For_Growth"] = df_edgar.get("FCF_Level", df_edgar.get("FreeCashFlow", pd.Series(np.nan, index=df_edgar.index)))
    df_edgar["Revenue_For_Margins"] = df_edgar["Revenue_For_Growth"]

    print("Calculating YoY growth metrics...")
    df_edgar = add_yoy_growth(df_edgar, "Revenue_For_Growth", "Revenue_Growth")
    df_edgar = add_yoy_growth(df_edgar, "EPS_For_Growth", "EPS_Growth")
    df_edgar = add_yoy_growth(df_edgar, "FCF_For_Growth", "FCF_Growth")
    
    # Report growth data availability
    print(f"Revenue_Growth: {df_edgar['Revenue_Growth'].notna().sum()} valid values")
    print(f"EPS_Growth: {df_edgar['EPS_Growth'].notna().sum()} valid values")
    print(f"FCF_Growth: {df_edgar['FCF_Growth'].notna().sum()} valid values")

    # Margin calculations with safer division
    print("Calculating margin metrics...")
    rev_base = df_edgar["Revenue_For_Margins"]
    rev_nonzero = rev_base.replace(0, np.nan)
    
    df_edgar["RD_to_Revenue"] = df_edgar.get("RD_Expense", 0) / rev_nonzero
    
    if "CostOfRevenue" in df_edgar.columns:
        df_edgar["Gross_Margin"] = (rev_base - df_edgar["CostOfRevenue"]) / rev_nonzero
    else:
        df_edgar["Gross_Margin"] = np.nan
        
    if "OperatingIncome" in df_edgar.columns:
        df_edgar["Operating_Margin"] = df_edgar["OperatingIncome"] / rev_nonzero
    else:
        df_edgar["Operating_Margin"] = np.nan
        
    if "EBITDA" in df_edgar.columns:
        df_edgar["EBITDA_Margin"] = df_edgar["EBITDA"] / rev_nonzero
    else:
        df_edgar["EBITDA_Margin"] = np.nan
        
    if "NetIncome" in df_edgar.columns:
        df_edgar["Net_Margin"] = df_edgar["NetIncome"] / rev_nonzero
    else:
        df_edgar["Net_Margin"] = np.nan
        
    if "TotalAssets" in df_edgar.columns:
        ta_nonzero = df_edgar["TotalAssets"].replace(0, np.nan)
        df_edgar["AssetTurnover"] = rev_base / ta_nonzero
    else:
        df_edgar["AssetTurnover"] = np.nan
        
    fcf_for_cap_eff = df_edgar.get("FreeCashFlow", df_edgar.get("FCF_Level", pd.Series(np.nan, index=df_edgar.index)))
    
    if "TotalEquity" in df_edgar.columns:
        eq_nonzero = df_edgar["TotalEquity"].replace(0, np.nan)
        df_edgar["FCF_to_Equity"] = fcf_for_cap_eff / eq_nonzero
    else:
        df_edgar["FCF_to_Equity"] = np.nan

    # Select all numeric fundamentals (including raw pivoted items) for z-scoring
    exclude_cols = {
        TARGET_COL,
        TARGET_RAW_COL,
        "Price",
        "Shares",
        "Quarter",
        "Year",
        "QuarterPeriod",
        "SectorGroup",
        "_priority",
        "Revenue_For_Growth",
        "EPS_For_Growth",
        "FCF_For_Growth",
        "Revenue_For_Margins",
    }
    exclude_prefixes = ("Predicted_", "Actual_")
    numeric_cols = [c for c in df_edgar.columns if df_edgar[c].dtype.kind in "biufc"]
    feature_cols = [
        c for c in numeric_cols
        if c not in exclude_cols and not any(c.startswith(pref) for pref in exclude_prefixes)
    ]
    print(f"\nSelected {len(feature_cols)} numeric features for z-scoring (full fundamentals):")
    for col in sorted(feature_cols)[:50]:
        valid_count = df_edgar[col].notna().sum()
        print(f"  {col}: {valid_count} valid values ({100*valid_count/len(df_edgar):.1f}%)")
    if len(feature_cols) > 50:
        print(f"  ... and {len(feature_cols)-50} more")

    # Clean and winsorize market cap, then log-transform target
    mcap_lower, mcap_upper = df_edgar["MarketCap"].quantile([0.01, 0.99])
    df_edgar["MarketCap"] = df_edgar["MarketCap"].clip(lower=mcap_lower, upper=mcap_upper)
    df_edgar[TARGET_COL] = np.log(df_edgar["MarketCap"])

    # Stabilize scale then winsorize features
    print("\nStabilizing (log1p-signed) and winsorizing features...")
    for col in feature_cols:
        df_edgar[col] = safe_log1p_signed(df_edgar[col])
        series = df_edgar[col].dropna()
        if series.empty or len(series) < 10:
            continue
        lower, upper = series.quantile([0.01, 0.99])
        df_edgar[col] = df_edgar[col].clip(lower=lower, upper=upper)

    df_edgar = df_edgar.replace([np.inf, -np.inf], np.nan)

    # Z-score normalization
    print("Calculating z-scores...")
    for col in feature_cols:
        mean_val = df_edgar[col].mean()
        std_val = df_edgar[col].std()
        if pd.isna(std_val) or np.isclose(std_val, 0):
            print(f"  Warning: {col} has zero/NA std, setting z-score to 0")
            df_edgar[f"{col}_z"] = 0.0
        else:
            df_edgar[f"{col}_z"] = (df_edgar[col] - mean_val) / std_val
            print(f"  {col}_z: mean={mean_val:.2f}, std={std_val:.2f}")

    # Build factors from z-scored components
    print("\nBuilding composite factors...")
    scale_components = [
        "Revenue_z", "OperatingIncome_z", "FreeCashFlow_z", "TotalEquity_z",
        "NetIncome_z", "EBITDA_z", "GrossProfit_z",
    ]
    scale_components = [c for c in scale_components if c in df_edgar.columns]
    
    df_edgar["Factor_Scale"] = factor_mean(df_edgar, scale_components)
    df_edgar["Factor_Leverage"] = df_edgar.get("NetDebt_z", np.nan)
    df_edgar["Factor_Intangibles"] = df_edgar.get("Goodwill_z", np.nan)
    df_edgar["Factor_RD"] = df_edgar.get("RD_to_Revenue_z", np.nan)

    # Sort and set index
    df_edgar = df_edgar.sort_values(["Ticker", "QuarterDate"])
    df_p = df_edgar.set_index(["Ticker", "QuarterDate"]).sort_index()

    # Calculate time-varying factors
    growth_comps = [c for c in GROWTH_COMPONENTS if c in df_p.columns]
    margin_comps = [c for c in PROFIT_MARGIN_COMPONENTS if c in df_p.columns]
    capeff_comps = [c for c in CAP_EFF_COMPONENTS if c in df_p.columns]
    
    df_p["Factor_Profitability"] = factor_mean(df_p, margin_comps)
    df_p["Factor_Growth"] = factor_mean(df_p, growth_comps)
    df_p["Factor_CapitalEfficiency"] = factor_mean(df_p, capeff_comps)

    # Report factor availability
    factor_set = [
        "Factor_Scale",
        "Factor_Leverage",
        "Factor_Intangibles",
        "Factor_RD",
        "Factor_Profitability",
        "Factor_Growth",
        "Factor_CapitalEfficiency",
    ]
    
    print("\nFactor availability:")
    for factor in factor_set:
        if factor in df_p.columns:
            valid_count = df_p[factor].notna().sum()
            print(f"  {factor}: {valid_count} valid values ({100*valid_count/len(df_p):.1f}%)")
        else:
            print(f"  {factor}: NOT CREATED")
    
    factor_component_map = {
        "Factor_Scale": scale_components,
        "Factor_Leverage": ["NetDebt_z"],
        "Factor_Intangibles": ["Goodwill_z"],
        "Factor_RD": ["RD_to_Revenue_z"],
        "Factor_Profitability": margin_comps,
        "Factor_Growth": growth_comps,
        "Factor_CapitalEfficiency": capeff_comps,
    }
    
    return df_p, factor_set, factor_component_map


def run_pooled_ols(df_p, regressors, model_key, model_name):
    """Run pooled OLS with better error handling"""
    print(f"\n{'='*60}")
    print(f"Preparing data for {model_name}")
    print(f"{'='*60}")
    
    # Check data availability
    print(f"Total observations: {len(df_p)}")
    print(f"Observations with {TARGET_COL}: {df_p[TARGET_COL].notna().sum()}")
    for reg in regressors:
        if reg in df_p.columns:
            print(f"Observations with {reg}: {df_p[reg].notna().sum()}")
        else:
            print(f"WARNING: {reg} not in dataframe!")
    
    # Drop rows with missing values
    df_sub = df_p.dropna(subset=[TARGET_COL] + regressors).copy()
    
    if df_sub.empty:
        print("\n[ERROR] No data available after dropping NA rows")
        print("This usually means:")
        print("1. Not enough historical data for growth calculations (need 4+ quarters)")
        print("2. Missing fundamental data fields")
        print("3. All values filtered out by winsorization")
        return None
    
    print(f"\nFinal sample size: {len(df_sub)} observations")
    print(f"Number of unique tickers: {df_sub.index.get_level_values('Ticker').nunique()}")
    print(f"Date range: {df_sub.index.get_level_values('QuarterDate').min()} to {df_sub.index.get_level_values('QuarterDate').max()}")

    y = df_sub[TARGET_COL]
    X = sm.add_constant(df_sub[regressors])
    
    print("\nRunning OLS regression...")
    res = sm.OLS(y, X, hasconst=True).fit(cov_type="HC3")

    print(f"\n--- {model_name} RESULTS ---")
    print(f"R^2: {res.rsquared:.4f}")
    print(f"Adj. R^2: {res.rsquared_adj:.4f}")
    print(f"N: {res.nobs:,}")
    print("\n" + str(res.summary()))

    pooled_df = build_bubble_outputs(model_key, y, res.predict(X), df_sub)
    pooled_analysis = analyze_bubbles(model_name, pooled_df, verbose=True)

    pooled_metrics = collect_model_metrics(model_name, res)
    pooled_metrics["Key"] = model_key
    pooled_metrics.update(pooled_analysis["stats"])

    return {
        "key": model_key,
        "name": model_name,
        "df_results": pooled_df,
        **pooled_analysis,
        "metrics": pooled_metrics,
    }


def build_lasso_feature_pool(df):
    """Pick a wide set of z-scored fundamentals and composite factors for Lasso search."""
    z_cols = [c for c in df.columns if c.endswith("_z") and c != TARGET_COL]
    factor_cols = [c for c in df.columns if c.startswith("Factor_")]
    pool = sorted(set(z_cols + factor_cols))
    return pool


def run_lasso_feature_selection(df, candidate_features, sector_label="ALL", min_features=3, max_features=MAX_LASSO_FEATURES):
    """Use LassoCV to pick a compact, sector-specific feature set."""
    if df.empty:
        print(f"[WARN] Lasso skipped for {sector_label}: empty dataframe")
        return []

    total = len(df)
    coverage = {f: df[f].notna().sum() for f in candidate_features if f in df.columns}
    usable = [
        f for f in candidate_features
        if f in df.columns
        and coverage.get(f, 0) >= MIN_FACTOR_OBS
        and (coverage.get(f, 0) / total) >= MIN_FACTOR_COVERAGE
        and df[f].dropna().var() > 1e-12
    ]

    if not usable:
        relaxed = [
            f for f in candidate_features
            if f in df.columns and df[f].dropna().var() > 1e-12
        ]
        if not relaxed:
            print(f"[WARN] Lasso usable features for {sector_label} too few (0); skipping selection")
            return []
        print(f"[WARN] Lasso coverage filters removed all features for {sector_label}; using relaxed variance-only set ({len(relaxed)} features)")
        usable = relaxed

    min_keep = min(min_features, len(usable))

    df_sub = df.dropna(subset=[TARGET_COL]).copy()
    if df_sub.empty:
        print(f"[WARN] Lasso skipped for {sector_label}: target column empty after dropna")
        return usable[:min_features]

    X = df_sub[usable]
    y = df_sub[TARGET_COL]

    # Median imputation to avoid losing rows due to patchy coverage
    X_imputed = X.apply(lambda col: col.fillna(col.median()))

    n_rows = len(X_imputed)
    if n_rows < 3:
        print(f"[WARN] Lasso skipped for {sector_label}: only {n_rows} usable rows")
        return usable[:min_keep]
    cv_folds = max(2, min(5, n_rows - 1))  # keep folds sensible for small samples

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed.values)

    print(f"\n[INFO] Running LassoCV for {sector_label}: {len(usable)} candidates, {n_rows} rows, cv={cv_folds}")
    try:
        lasso = LassoCV(cv=cv_folds, random_state=42, n_alphas=50, max_iter=5000)
        lasso.fit(X_scaled, y)
    except Exception as exc:
        print(f"[WARN] Lasso failed for {sector_label}: {exc}")
        return usable[:min_keep]

    coef_pairs = list(zip(usable, lasso.coef_))
    selected = [feat for feat, coef in coef_pairs if abs(coef) > 1e-6]

    if not selected:
        # Fallback: keep top-variance drivers even if coefficients shrink to zero
        coef_pairs.sort(key=lambda x: abs(x[1]), reverse=True)
        selected = [feat for feat, _ in coef_pairs[:min_keep]]

    # Cap maximum features to avoid overfitting/collinearity
    if len(selected) > max_features:
        coef_pairs = [p for p in coef_pairs if p[0] in selected]
        coef_pairs.sort(key=lambda x: abs(x[1]), reverse=True)
        selected = [feat for feat, _ in coef_pairs[:max_features]]

    print(f"[INFO] Lasso-selected features for {sector_label}: {', '.join(selected)}")
    return selected


def export_outputs(payloads, metrics_df=None):
    with pd.ExcelWriter(OUTPUT_XLSX, engine="openpyxl") as writer:
        if metrics_df is not None and not metrics_df.empty:
            metrics_df.to_excel(writer, sheet_name="Model_Comparison", index=False)

        for payload in payloads:
            df_to_export = payload["df_results"].reset_index()
            bubble_q = payload["bubble_by_quarter"]
            bubble_t = payload["bubble_by_ticker"]
            top_over = payload["top_overvalued"]
            top_under = payload["top_undervalued"]

            key = payload["key"]
            df_to_export[["Ticker", "QuarterDate", "Actual_MarketCap", "Predicted_MarketCap",
                          "Residual_LogMCap", "Overvaluation_pct"]].to_excel(
                writer, sheet_name=f"{key}_All"[:31], index=False
            )
            bubble_q.to_excel(writer, sheet_name=f"{key}_ByQuarter"[:31], index=False)
            bubble_t.to_excel(writer, sheet_name=f"{key}_ByCompany"[:31], index=False)
            top_over.to_excel(writer, sheet_name=f"{key}_MostOver"[:31], index=False)
            top_under.to_excel(writer, sheet_name=f"{key}_MostUnder"[:31], index=False)

    print(f"\n[OK] Results exported to: {OUTPUT_XLSX}")


def plot_outputs(payload, output_ts_path, output_dist_path):
    best_quarter = payload["bubble_by_quarter"]
    best_df_results = payload["df_results"].reset_index()

    plt.figure(figsize=(14, 6))
    plt.plot(best_quarter["QuarterDate"], best_quarter["Mean_Overval_pct"],
             marker="o", linewidth=2, color='#2E86AB')
    plt.axhline(y=0, color="red", linestyle="--", linewidth=2, label="Fair Value")
    plt.title(f"Average Market Overvaluation Over Time\n({payload['name']})", 
              fontsize=14, fontweight="bold")
    plt.xlabel("Quarter", fontsize=12)
    plt.ylabel("Average Overvaluation (%)", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_ts_path, dpi=300, bbox_inches='tight')
    print(f"[OK] Time series plot saved: {output_ts_path}")

    plt.figure(figsize=(12, 6))
    plt.hist(best_df_results["Overvaluation_pct"], bins=50, 
             edgecolor="black", alpha=0.7, color='#A23B72')
    plt.axvline(x=0, color="red", linestyle="--", linewidth=2, label="Fair Value")
    plt.title(f"Distribution of Market Overvaluation\n({payload['name']})", 
              fontsize=14, fontweight="bold")
    plt.xlabel("Overvaluation (%)", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dist_path, dpi=300, bbox_inches='tight')
    print(f"[OK] Distribution plot saved: {output_dist_path}")

    plt.close("all")


def run_model_pipeline(df_p, factor_set, factor_component_map, key, name, save_common_path=None, plot_paths=None):
    print("\n" + "="*70)
    print(f" RUNNING MODEL: {name}")
    print("="*70)

    selected_factors, df_common, dropped_factors, factor_counts = select_common_sample(
        df_p, factor_set, target_col=TARGET_COL
    )
    print(f"Common sample size: {df_common.shape[0]:,} observations")
    if dropped_factors:
        dropped_sorted = sorted(dropped_factors, key=lambda f: factor_counts.get(f, 0))
        dropped_msg = ", ".join(
            f"{f} ({factor_counts.get(f, 0)} valid)"
            for f in dropped_sorted
        )
        print(f"[WARN] Dropping sparse factors: {dropped_msg}")
    if df_common.empty or not selected_factors:
        print("[ERROR] No data after forming common sample.")
        return None, pd.DataFrame(), df_common, []

    optimized_factors, improved, best_sample = optimize_factor_coverage(
        df_p, selected_factors, target_col=TARGET_COL
    )
    if improved:
        factor_set = optimized_factors
        df_common = df_p.dropna(subset=[TARGET_COL] + factor_set).copy()
        print(f"Rebuilt common sample after optimization: {len(df_common):,} observations")
    else:
        factor_set = selected_factors

    if save_common_path is not None:
        df_common.reset_index().to_csv(save_common_path, index=False)
        print(f"[OK] Common sample saved to {save_common_path}")

    regressors, dropped_const = drop_constant_columns(df_common, factor_set)
    if dropped_const:
        print(f"\n[WARN] Dropped near-constant columns: {dropped_const}")
    
    if not regressors:
        print("\n[ERROR] No valid regressors available after dropping constant columns.")
        return None, pd.DataFrame(), df_common, []

    comp_map = dict(factor_component_map)
    for reg in regressors:
        comp_map.setdefault(reg, [reg])
    print_factor_composition(comp_map, regressors)

    payload = run_pooled_ols(
        df_common,
        regressors,
        model_key=key,
        model_name=f"{name} ({len(regressors)}-Factor Model)",
    )
    
    if payload is None:
        print("\n[ERROR] Model estimation failed.")
        return None, pd.DataFrame(), df_common, regressors

    if plot_paths:
        plot_outputs(payload, plot_paths[0], plot_paths[1])

    metrics_df = pd.DataFrame([payload["metrics"]])
    return payload, metrics_df, df_common, regressors


def run_sector_model(sector_code):
    """Entry point to run a single sector model with VIF export."""
    print("\n" + "="*70)
    print(f" RUNNING SECTOR MODEL: {sector_code}")
    print("="*70)
    
    df_p, factor_set, factor_component_map = prepare_panel_data()
    lasso_feature_pool = build_lasso_feature_pool(df_p)

    sector_series = df_p["SectorGroup"] if "SectorGroup" in df_p.columns else pd.Series("", index=df_p.index)
    df_sector = df_p[sector_series.fillna("") == sector_code]
    if df_sector.empty:
        print(f"[ERROR] Sector {sector_code}: no data after merge.")
        return None, None

    sector_label = SECTOR_DISPLAY_NAMES.get(sector_code, f"{sector_code} Sector")
    sector_features = run_lasso_feature_selection(
        df_sector,
        lasso_feature_pool,
        sector_label=sector_label,
        min_features=3,
    )
    if not sector_features:
        print(f"[WARN] No Lasso-selected features for {sector_label}; falling back to pooled factor set")
        sector_features = factor_set

    ts_path = ROOT / f"bist_bubbleness_timeseries_{sector_code.lower()}.png"
    dist_path = ROOT / f"bist_bubbleness_distribution_{sector_code.lower()}.png"

    payload, metrics_df, df_common, regressors = run_model_pipeline(
        df_sector,
        sector_features,
        factor_component_map,
        key=f"{sector_code}_OLS",
        name=f"{sector_label} Sector OLS",
        save_common_path=None,
        plot_paths=(ts_path, dist_path),
    )
    if payload is None:
        return None, None

    export_vif_and_corr(
        df_common,
        regressors,
        label=sector_label,
        vif_path=ROOT / f"bist_vif_table_{sector_code.lower()}.csv",
        corr_path=ROOT / f"bist_correlation_vs_log_mcap_{sector_code.lower()}.csv",
    )
    return payload, metrics_df


def main():
    print("\n" + "="*70)
    print(" BIST MARKET VALUATION ANALYSIS - Pooled OLS Model")
    print("="*70)
    
    try:
        df_p, factor_set, factor_component_map = prepare_panel_data()
    except Exception as e:
        print(f"\n[ERROR] Failed to prepare data: {e}")
        import traceback
        traceback.print_exc()
        return

    lasso_feature_pool = build_lasso_feature_pool(df_p)
    print(f"\nLasso candidate pool size: {len(lasso_feature_pool)} z-scored fundamentals/factors")

    payloads = []
    metrics_frames = []

    # Overall model (lasso-selected fundamentals)
    overall_features = run_lasso_feature_selection(
        df_p,
        lasso_feature_pool,
        sector_label="ALL",
        min_features=5,
    )
    if not overall_features:
        print("[WARN] Lasso returned no features for ALL; falling back to pooled factor set")
        overall_features = factor_set

    overall_payload, overall_metrics, overall_df_common, overall_regressors = run_model_pipeline(
        df_p,
        overall_features,
        factor_component_map,
        key="Pooled_OLS",
        name="Pooled OLS (Lasso-selected fundamentals)",
        save_common_path=ROOT / "bist_df_common_pooled_ols.csv",
        plot_paths=(OUTPUT_TS, OUTPUT_DIST),
    )
    if overall_payload is not None:
        payloads.append(overall_payload)
        metrics_frames.append(overall_metrics)

    # Sector-specific models based on curated folders
    sector_codes = ["CAP", "COM", "CON", "FIN", "IND", "INT", "RE"]
    sector_series = df_p["SectorGroup"] if "SectorGroup" in df_p.columns else pd.Series("", index=df_p.index)
    for code in sector_codes:
        df_sector = df_p[sector_series.fillna("") == code]
        if df_sector.empty:
            print(f"[WARN] Skipping sector {code}: no data after merge.")
            continue
        sector_label = SECTOR_DISPLAY_NAMES.get(code, f"{code} Sector")
        sector_features = run_lasso_feature_selection(
            df_sector,
            lasso_feature_pool,
            sector_label=sector_label,
            min_features=3,
        )
        if not sector_features:
            print(f"[WARN] No Lasso-selected features for {sector_label}; falling back to pooled factor set")
            sector_features = factor_set
        ts_path = ROOT / f"bist_bubbleness_timeseries_{code.lower()}.png"
        dist_path = ROOT / f"bist_bubbleness_distribution_{code.lower()}.png"
        payload, metrics_df, df_common, regressors = run_model_pipeline(
            df_sector,
            sector_features,
            factor_component_map,
            key=f"{code}_OLS",
            name=f"{sector_label} Sector OLS",
            save_common_path=None,
            plot_paths=(ts_path, dist_path),
        )
        if payload is not None:
            payloads.append(payload)
            metrics_frames.append(metrics_df)
            export_vif_and_corr(
                df_common,
                regressors,
                label=sector_label,
                vif_path=ROOT / f"bist_vif_table_{code.lower()}.csv",
                corr_path=ROOT / f"bist_correlation_vs_log_mcap_{code.lower()}.csv",
            )

    if not payloads:
        print("[ERROR] No models ran successfully. Check data coverage.")
        return

    metrics_df = pd.concat(metrics_frames, ignore_index=True) if metrics_frames else None
    export_outputs(payloads, metrics_df=metrics_df)

    # VIF analysis for overall model
    if overall_payload is not None and overall_df_common is not None and overall_regressors:
        export_vif_and_corr(
            overall_df_common,
            overall_regressors,
            label="Overall",
            vif_path=VIF_OUTPUT,
            corr_path=CORR_OUTPUT,
        )
    
    print("\n" + "="*70)
    print(" ANALYSIS COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()
