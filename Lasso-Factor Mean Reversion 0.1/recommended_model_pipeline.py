"""
RECOMMENDED MODEL PIPELINE: Lasso + Pooled OLS

Fixes:
1. Uses correct target column: "LogMarketCap" (not "LogMCap_z")
2. Integrates with existing prepare_panel_data()
3. Walk-forward validation with proper IC calculation
4. Outputs to separate folder: recommended_model_outputs/
5. Pre-filters correlated features before Lasso
6. Refits with OLS for unbiased coefficients

Usage:
    python recommended_model_pipeline.py --sector ALL
    python recommended_model_pipeline.py --sector FIN --method elasticnet
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path to import from main Bist folder
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LassoCV, ElasticNetCV
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

# Import your existing data preparation
try:
    from pooled_ols_residuals_bist import (
        prepare_panel_data,
        SECTOR_DISPLAY_NAMES,
        TARGET_COL,  # This is "LogMarketCap"
    )
    print(f"[OK] Successfully imported from pooled_ols_residuals_bist.py")
    print(f"[OK] Target column is: '{TARGET_COL}'")
except ImportError as e:
    print(f"[ERROR] Could not import from pooled_ols_residuals_bist.py: {e}")
    print("Make sure pooled_ols_residuals_bist.py is in the parent directory.")
    exit(1)

ROOT = Path(__file__).resolve().parent.parent  # Go to parent (main Bist folder)
OUTPUT_DIR = ROOT / "recommended_model_outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

DEFAULT_FUNDAMENTAL_LAG_QUARTERS = 1


def apply_fundamental_lag(
    df: pd.DataFrame,
    lag_quarters: int,
    exclude_cols: set[str] | None = None,
) -> pd.DataFrame:
    """
    Lag (shift) fundamentals so that QuarterDate t uses fundamentals from QuarterDate t-lag.

    This prevents using not-yet-published quarter fundamentals when forming signals.
    Example with lag_quarters=1:
      - Use 2019-Q2 fundamentals with 2019-Q3 price/market cap.

    Notes:
    - We shift by ticker over the panel index order.
    - Market-based columns (Price/Shares/MarketCap/LogMarketCap) should NOT be lagged.
    """
    if lag_quarters <= 0:
        return df

    exclude_cols = exclude_cols or set()
    cols_to_lag = [c for c in df.columns if c not in exclude_cols]
    if not cols_to_lag:
        return df

    print(f"\n{'='*60}")
    print("APPLYING FUNDAMENTAL LAG")
    print(f"{'='*60}")
    print(f"Lag quarters: {lag_quarters}")
    print("Signal at QuarterDate t uses fundamentals from t - lag.")

    df_lagged = df.copy()
    df_lagged[cols_to_lag] = df_lagged.groupby(level="Ticker")[cols_to_lag].shift(lag_quarters)
    return df_lagged


def compute_forward_returns(df, periods=[1, 2, 4]):
    """
    Compute forward returns for multiple horizons using PRICE if available,
    otherwise MarketCap (fallback to exp(LogMarketCap)).

    Args:
        df: Panel DataFrame with MultiIndex (Ticker, QuarterDate)
        periods: List of forward periods in quarters

    Returns:
        DataFrame with added columns: Return_1Q, Return_2Q, Return_4Q
    """
    df = df.copy()

    price_col = "Price" if "Price" in df.columns else None

    if price_col:
        base_series = df[price_col]
    else:
        if "MarketCap" in df.columns:
            base_series = df["MarketCap"]
        elif TARGET_COL in df.columns:
            base_series = np.exp(df[TARGET_COL])
        else:
            print("[ERROR] Cannot compute returns: no Price, MarketCap, or LogMarketCap column")
            return df

    for p in periods:
        df[f"Return_{p}Q"] = (
            base_series.groupby(level="Ticker")
            .pct_change(periods=p)
            .shift(-p)
        )

    return df


def prefilter_correlated_features(df, feature_pool, target_col, corr_threshold=0.85):
    """
    Remove highly correlated features to reduce multicollinearity.

    Strategy:
    1. For each pair of features with correlation > threshold
    2. Keep the one more correlated with target, drop the other

    Args:
        df: Panel DataFrame with features
        feature_pool: List of candidate features
        target_col: Target variable name
        corr_threshold: Correlation threshold (0.85 = 85%)

    Returns:
        List of filtered features
    """
    print(f"\n{'='*60}")
    print("PRE-FILTERING CORRELATED FEATURES")
    print(f"{'='*60}")

    # Compute correlation matrix on valid data only
    df_subset = df[[f for f in feature_pool if f in df.columns] + [target_col]].dropna()

    if df_subset.empty:
        print("[WARN] No valid data after dropna")
        return feature_pool

    available_features = [f for f in feature_pool if f in df_subset.columns]

    if len(available_features) < 2:
        print(f"[WARN] Only {len(available_features)} features available, skipping pre-filter")
        return available_features

    corr_matrix = df_subset[available_features].corr().abs()

    # Find pairs with high correlation
    upper_tri = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )

    # Correlation with target
    target_corr = df_subset[available_features].corrwith(df_subset[target_col]).abs()

    # Drop features
    to_drop = set()
    for col in upper_tri.columns:
        for row in upper_tri.index:
            if upper_tri.loc[row, col] > corr_threshold:
                # Keep the feature more correlated with target
                if target_corr[col] > target_corr[row]:
                    if row not in to_drop:  # Only print once
                        to_drop.add(row)
                        print(f"  Dropping {row:40s} (corr={upper_tri.loc[row, col]:.3f} with {col})")
                else:
                    if col not in to_drop:  # Only print once
                        to_drop.add(col)
                        print(f"  Dropping {col:40s} (corr={upper_tri.loc[row, col]:.3f} with {row})")

    filtered = [f for f in available_features if f not in to_drop]

    print(f"\nOriginal features: {len(available_features)}")
    print(f"Filtered features: {len(filtered)}")
    print(f"Removed:           {len(to_drop)}")

    return filtered


def fit_regularized_model(X_train, y_train, method="lasso", cv_folds=5):
    """
    Fit Lasso or ElasticNet with cross-validation.

    Args:
        X_train: Training features
        y_train: Training target
        method: "lasso" or "elasticnet"
        cv_folds: Number of CV folds

    Returns:
        Fitted model
    """
    print(f"\n{'='*60}")
    print(f"FITTING {method.upper()} WITH {cv_folds}-FOLD CV")
    print(f"{'='*60}")

    if method == "lasso":
        model = LassoCV(
            cv=cv_folds,
            max_iter=10000,
            random_state=42,
            n_jobs=-1,
            selection="random",  # Faster convergence
        )
    elif method == "elasticnet":
        model = ElasticNetCV(
            l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9],
            cv=cv_folds,
            max_iter=10000,
            random_state=42,
            n_jobs=-1,
            selection="random",
        )
    else:
        raise ValueError(f"Unknown method: {method}")

    model.fit(X_train, y_train)

    print(f"  Best alpha: {model.alpha_:.6f}")
    if hasattr(model, "l1_ratio_"):
        print(f"  Best l1_ratio: {model.l1_ratio_:.3f}")

    return model


def refit_with_pooled_ols(X, y, selected_features):
    """
    Refit using OLS on selected features for interpretable coefficients.

    Lasso/ElasticNet shrinks coefficients toward zero (biased).
    OLS on selected features gives unbiased estimates.

    Args:
        X: Feature matrix (full sample)
        y: Target vector
        selected_features: Features selected by Lasso/ElasticNet

    Returns:
        OLS results object
    """
    print(f"\n{'='*60}")
    print("REFITTING WITH POOLED OLS (UNBIASED COEFFICIENTS)")
    print(f"{'='*60}")

    X_selected = sm.add_constant(X[selected_features])
    model = sm.OLS(y, X_selected)
    results = model.fit(cov_type="HC3")  # Robust standard errors

    print(results.summary())

    return results


def evaluate_ic(residuals, returns, label=""):
    """Compute Information Coefficient (Spearman correlation)."""
    valid_idx = residuals.dropna().index.intersection(returns.dropna().index)

    if len(valid_idx) < 10:
        print(f"{label:20s} N < 10, skipped")
        return None

    res = residuals.loc[valid_idx]
    ret = returns.loc[valid_idx]

    ic, pval = spearmanr(res, ret)

    status = "✓" if (ic < 0 and pval < 0.05) else "✗"
    print(f"{status} {label:20s} IC = {ic:+.4f} (p={pval:.4f}, N={len(valid_idx):,})")

    return {"IC": ic, "p_value": pval, "N": len(valid_idx)}


def run_single_model(
    sector_code="ALL",
    method="lasso",
    train_end="2020-01-01",
    test_start="2023-01-01",
    max_features=15,
    corr_threshold=0.85,
    fundamental_lag_quarters: int = DEFAULT_FUNDAMENTAL_LAG_QUARTERS,
    export_outputs: bool = True,
):
    """
    Run the recommended single modeling approach.

    Args:
        sector_code: "ALL" or sector code (CAP, FIN, etc.)
        method: "lasso" or "elasticnet"
        train_end: End of training period
        test_start: Start of test period
        max_features: Maximum features to select
        corr_threshold: Pre-filter correlation threshold

    Returns:
        dict with results
    """
    print("\n" + "="*70)
    print(f"RECOMMENDED MODEL PIPELINE: {method.upper()} + POOLED OLS")
    print("="*70)

    # Load data using existing pipeline
    print("\nLoading panel data (using existing prepare_panel_data)...")
    df_panel, factor_set, factor_component_map = prepare_panel_data()

    print(f"Loaded {len(df_panel):,} observations")
    print(f"Date range: {df_panel.index.get_level_values('QuarterDate').min()} to {df_panel.index.get_level_values('QuarterDate').max()}")
    print(f"Unique tickers: {df_panel.index.get_level_values('Ticker').nunique()}")

    # Apply lag so that we never use same-quarter fundamentals with same-quarter price.
    # Keep market-based columns unshifted.
    df_panel = apply_fundamental_lag(
        df_panel,
        lag_quarters=fundamental_lag_quarters,
        exclude_cols={
            TARGET_COL,
            "MarketCap",
            "Price",
            "Shares",
            # Meta/time columns should stay aligned to the QuarterDate index
            "Year",
            "Quarter",
            "QuarterPeriod",
            # Meta columns (not true fundamentals)
            "SectorGroup",
            "_priority",
        },
    )

    # Filter by sector if needed
    if sector_code != "ALL":
        sector_series = df_panel["SectorGroup"] if "SectorGroup" in df_panel.columns else pd.Series("", index=df_panel.index)
        df_panel = df_panel[sector_series.fillna("") == sector_code]
        sector_name = SECTOR_DISPLAY_NAMES.get(sector_code, sector_code)
        print(f"\nFiltering to sector: {sector_name}")
        print(f"  Observations: {len(df_panel):,}")
    else:
        sector_name = "ALL Sectors"

    if df_panel.empty:
        print("[ERROR] No data available for this sector.")
        return None

    # Use factor_set from prepare_panel_data() instead of all z-scored columns
    # This contains only features with sufficient coverage
    target_col = TARGET_COL

    if target_col not in df_panel.columns:
        print(f"[ERROR] Target column '{target_col}' not found.")
        print(f"Available columns: {list(df_panel.columns[:20])}...")
        return None

    print(f"\nTarget column: {target_col}")
    print(f"Factor set from prepare_panel_data: {len(factor_set)} features")
    print(f"Features: {factor_set}")

    # Use the factor_set as our feature pool (these are pre-validated)
    feature_pool = factor_set.copy()

    # Check which features are actually in the data
    available_features = [f for f in feature_pool if f in df_panel.columns]

    if len(available_features) < 3:
        print(f"[ERROR] Only {len(available_features)} features available in data. Need at least 3.")
        print(f"  Requested: {feature_pool}")
        print(f"  Available: {available_features}")
        return None

    print(f"Features available in data: {len(available_features)}/{len(feature_pool)}")

    # CRITICAL FIX: Check coverage of each feature with target
    # Some factors like Factor_Intangibles only have 1% coverage
    print("\nChecking feature coverage with target:")
    target_valid = df_panel[target_col].notna()

    feature_coverage = {}
    for feat in available_features:
        feat_valid = df_panel[feat].notna()
        overlap = (target_valid & feat_valid).sum()
        coverage_pct = overlap / target_valid.sum() * 100 if target_valid.sum() > 0 else 0
        feature_coverage[feat] = {
            'valid': feat_valid.sum(),
            'overlap': overlap,
            'coverage_pct': coverage_pct
        }
        print(f"  {feat:30s} overlap: {overlap:,} ({coverage_pct:.1f}%)")

    # Remove features with very low coverage (< 10% overlap with target)
    MIN_COVERAGE_PCT = 10.0
    high_coverage_features = [
        feat for feat, stats in feature_coverage.items()
        if stats['coverage_pct'] >= MIN_COVERAGE_PCT
    ]

    low_coverage = [f for f in available_features if f not in high_coverage_features]
    if low_coverage:
        print(f"\n[WARN] Removing {len(low_coverage)} features with < {MIN_COVERAGE_PCT}% coverage:")
        for feat in low_coverage:
            print(f"  - {feat} ({feature_coverage[feat]['coverage_pct']:.1f}%)")

    if len(high_coverage_features) < 3:
        print(f"\n[ERROR] Only {len(high_coverage_features)} features with sufficient coverage. Need at least 3.")
        return None

    available_features = high_coverage_features
    print(f"\nUsing {len(available_features)} features with good coverage")

    # Pre-filter correlated features
    filtered_features = prefilter_correlated_features(
        df_panel, available_features, target_col, corr_threshold
    )

    if len(filtered_features) == 0:
        print(f"[ERROR] No features remain after pre-filtering.")
        print(f"[WARN] This may be due to missing data. Using all available features without pre-filtering.")
        filtered_features = available_features

    if len(filtered_features) < 3:
        print(f"[ERROR] Only {len(filtered_features)} features available. Need at least 3.")
        return None

    # Compute forward returns
    print("\nComputing forward returns (1Q, 2Q, 4Q)...")
    df_panel = compute_forward_returns(df_panel)

    # Split data
    dates = df_panel.index.get_level_values("QuarterDate")
    df_train = df_panel[dates < pd.Timestamp(train_end)]
    df_val = df_panel[(dates >= pd.Timestamp(train_end)) & (dates < pd.Timestamp(test_start))]
    df_test = df_panel[dates >= pd.Timestamp(test_start)]

    print(f"\n{'='*60}")
    print("DATA SPLIT")
    print(f"{'='*60}")
    print(f"Training:   {df_train.index.get_level_values('QuarterDate').min()} to {df_train.index.get_level_values('QuarterDate').max()} (N={len(df_train):,})")
    if not df_val.empty:
        print(f"Validation: {df_val.index.get_level_values('QuarterDate').min()} to {df_val.index.get_level_values('QuarterDate').max()} (N={len(df_val):,})")
    if not df_test.empty:
        print(f"Test:       {df_test.index.get_level_values('QuarterDate').min()} to {df_test.index.get_level_values('QuarterDate').max()} (N={len(df_test):,})")

    # Prepare training data
    df_train_clean = df_train.dropna(subset=["Return_1Q"])
    X_train = df_train_clean[filtered_features].dropna()
    y_train = df_train_clean.loc[X_train.index, target_col]

    print(f"\nTraining sample after dropna: {len(X_train):,}")

    if len(X_train) < 100:
        print("[ERROR] Training sample too small (< 100 observations)")
        return None

    # Fit regularized model
    reg_model = fit_regularized_model(X_train, y_train, method=method)

    # Extract selected features
    coefs = pd.Series(reg_model.coef_, index=filtered_features)
    selected_features_all = coefs[coefs != 0].sort_values(key=abs, ascending=False)

    if len(selected_features_all) == 0:
        print(f"[ERROR] {method} selected ZERO features. Try reducing regularization.")
        return None

    print(f"\n{len(selected_features_all)} features selected by {method}:")
    for feat, coef in selected_features_all.head(max_features).items():
        print(f"  {feat:40s} {coef:+.6f}")

    selected_features = selected_features_all.head(max_features).index.tolist()

    # Refit with OLS for unbiased coefficients
    ols_results = refit_with_pooled_ols(X_train, y_train, selected_features)

    # Evaluate on all periods
    print(f"\n{'='*60}")
    print("INFORMATION COEFFICIENT (IC) EVALUATION")
    print(f"{'='*60}")
    print("IC = Spearman(residuals, 1Q forward returns)")
    print("Negative IC is GOOD (undervalued → outperform)\n")

    results = {}

    for period_name, df_period in [("Training", df_train), ("Validation", df_val), ("Test", df_test)]:
        if df_period.empty:
            print(f"{period_name:20s} Empty, skipped")
            continue

        # Prepare data
        df_period_clean = df_period.dropna(subset=["Return_1Q"])
        X_period = df_period_clean[selected_features]
        y_period = df_period_clean[target_col]

        valid_idx = X_period.dropna().index.intersection(y_period.dropna().index)
        X_period = X_period.loc[valid_idx]
        y_period = y_period.loc[valid_idx]

        if len(X_period) < 10:
            print(f"{period_name:20s} N < 10, skipped")
            continue

        # Predict using OLS model
        X_with_const = sm.add_constant(X_period)
        y_pred = ols_results.predict(X_with_const)
        residuals = y_period - y_pred

        # Compute IC
        returns_1q = df_period_clean.loc[valid_idx, "Return_1Q"]
        ic_stats = evaluate_ic(residuals, returns_1q, label=period_name)

        results[f"{period_name.lower()}_ic"] = ic_stats
        results[f"{period_name.lower()}_residuals"] = residuals

    # Save results
    output_key = f"{sector_code}_{method}"
    results_df = df_panel.copy()
    results_df["Predicted_LogMarketCap"] = np.nan
    results_df["Residual_LogMarketCap"] = np.nan

    # Predict on full dataset
    X_full = df_panel[selected_features]
    y_full = df_panel[target_col]
    valid_idx = X_full.dropna().index.intersection(y_full.dropna().index)

    if len(valid_idx) > 0:
        X_with_const = sm.add_constant(X_full.loc[valid_idx])
        y_pred_full = ols_results.predict(X_with_const)

        results_df.loc[valid_idx, "Predicted_LogMarketCap"] = y_pred_full
        results_df.loc[valid_idx, "Residual_LogMarketCap"] = y_full.loc[valid_idx] - y_pred_full
        results_df["Overvaluation_pct"] = (np.exp(results_df["Residual_LogMarketCap"]) - 1) * 100

    if export_outputs:
        # Export results
        output_path = OUTPUT_DIR / f"{output_key}_results.csv"
        results_df.reset_index().to_csv(output_path, index=False)
        print(f"\n[OK] Results saved to {output_path}")

        # Save model summary
        summary_path = OUTPUT_DIR / f"{output_key}_model_summary.txt"
        with open(summary_path, "w") as f:
            f.write(f"Model: {method.upper()} + Pooled OLS\n")
            f.write(f"Sector: {sector_name}\n")
            f.write(f"Fundamental lag (quarters): {fundamental_lag_quarters}\n")
            f.write(f"Selected Features ({len(selected_features)}):\n")
            for feat in selected_features:
                f.write(f"  - {feat}\n")
            f.write("\n" + "="*70 + "\n")
            f.write(str(ols_results.summary()))
            f.write("\n\n" + "="*70 + "\n")
            f.write("INFORMATION COEFFICIENTS:\n")
            for key, val in results.items():
                if key.endswith("_ic") and val is not None:
                    f.write(f"  {key}: IC={val['IC']:+.4f}, p={val['p_value']:.4f}, N={val['N']}\n")

        print(f"[OK] Model summary saved to {summary_path}")

        # Save selected features list
        features_path = OUTPUT_DIR / f"{output_key}_selected_features.txt"
        with open(features_path, "w") as f:
            f.write(f"Selected Features for {sector_name} ({method.upper()}):\n")
            f.write(f"Fundamental lag (quarters): {fundamental_lag_quarters}\n")
            f.write(f"Total: {len(selected_features)}\n\n")
            for i, feat in enumerate(selected_features, 1):
                coef = ols_results.params.get(feat, np.nan)
                f.write(f"{i:2d}. {feat:40s} (coef={coef:+.6f})\n")

        print(f"[OK] Selected features saved to {features_path}")

        print(f"\n{'='*70}")
        print("MODEL PIPELINE COMPLETE")
        print(f"{'='*70}")
        print(f"Fundamental lag (quarters): {fundamental_lag_quarters}")
        print(f"Output directory: {OUTPUT_DIR}")
        print(f"Files created:")
        print(f"  1. {output_key}_results.csv          (all predictions & residuals)")
        print(f"  2. {output_key}_model_summary.txt    (OLS regression output)")
        print(f"  3. {output_key}_selected_features.txt (feature list with coefficients)")
    else:
        print(f"\n[OK] Export skipped (--no-export).")
        print(f"Fundamental lag (quarters): {fundamental_lag_quarters}")

    return {
        "sector": sector_code,
        "method": method,
        "selected_features": selected_features,
        "ols_results": ols_results,
        "results_df": results_df,
        "fundamental_lag_quarters": fundamental_lag_quarters,
        **results,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run recommended single model pipeline")
    parser.add_argument("--sector", default="ALL", help="Sector code (ALL, CAP, FIN, etc.)")
    parser.add_argument("--method", default="lasso", choices=["lasso", "elasticnet"], help="Regularization method")
    parser.add_argument("--max-features", type=int, default=15, help="Maximum features to select")
    parser.add_argument("--corr-threshold", type=float, default=0.85, help="Correlation threshold for pre-filtering")
    parser.add_argument(
        "--fundamental-lag",
        type=int,
        default=DEFAULT_FUNDAMENTAL_LAG_QUARTERS,
        help=f"Quarter lag between fundamentals and price (default: {DEFAULT_FUNDAMENTAL_LAG_QUARTERS})",
    )
    parser.add_argument(
        "--no-export",
        action="store_true",
        help="Run model but do not write outputs to recommended_model_outputs/",
    )

    args = parser.parse_args()

    run_single_model(
        sector_code=args.sector,
        method=args.method,
        max_features=args.max_features,
        corr_threshold=args.corr_threshold,
        fundamental_lag_quarters=args.fundamental_lag,
        export_outputs=not args.no_export,
    )
