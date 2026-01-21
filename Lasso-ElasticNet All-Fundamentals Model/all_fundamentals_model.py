"""
ALL-FUNDAMENTALS LASSO/ELASTICNET MODEL

This model uses ALL available fundamental variables (~500+) from financial statements
and applies Lasso/ElasticNet for automatic feature selection to avoid omitted variable bias.

Key Differences from Previous Model:
1. Uses ALL z-scored fundamentals (not just 6-7 pre-defined factors)
2. Applies Lasso/ElasticNet to select the most relevant variables automatically
3. No manual factor construction - let the data speak
4. Walk-forward validation with proper IC calculation
5. Outputs which specific fundamentals matter most

Usage:
    python all_fundamentals_model.py --method lasso --max-features 20
    python all_fundamentals_model.py --method elasticnet --max-features 30
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
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Import existing data preparation
from pooled_ols_residuals_bist import (
    prepare_panel_data,
    SECTOR_DISPLAY_NAMES,
    TARGET_COL,
)

ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


def get_all_z_scored_fundamentals(df):
    """
    Extract ALL z-scored fundamental variables from the dataset.

    These include:
    - All calculated ratios (margins, growth rates, etc.) with _z suffix
    - All raw fundamentals from financial statements with _z suffix
    - Balance Sheet items (BS_*_z)
    - Income Statement items (IS_*_z)
    - Cash Flow items (CF_*_z)

    Returns:
        List of column names for z-scored fundamentals
    """
    z_cols = [c for c in df.columns if c.endswith('_z')]

    # Exclude any that are actually targets or derived from targets
    exclude_patterns = ['LogMarketCap', 'MarketCap', 'Predicted', 'Residual', 'Overvaluation']
    z_cols = [c for c in z_cols if not any(pattern in c for pattern in exclude_patterns)]

    return sorted(z_cols)


def filter_by_coverage(df, features, target_col, min_coverage_pct=10.0, min_obs=500):
    """
    Filter features based on data coverage with the target variable.

    Args:
        df: DataFrame with features and target
        features: List of feature column names
        target_col: Target column name
        min_coverage_pct: Minimum percentage of overlap with target (default 10%)
        min_obs: Minimum number of observations (default 500)

    Returns:
        Tuple of (filtered_features, coverage_stats dict)
    """
    print(f"\n{'='*70}")
    print("FEATURE COVERAGE ANALYSIS")
    print(f"{'='*70}")

    target_valid = df[target_col].notna()
    total_target = target_valid.sum()

    coverage_stats = {}
    for feat in features:
        if feat not in df.columns:
            continue
        feat_valid = df[feat].notna()
        overlap = (target_valid & feat_valid).sum()
        coverage_pct = 100 * overlap / total_target if total_target > 0 else 0

        coverage_stats[feat] = {
            'valid': feat_valid.sum(),
            'overlap': overlap,
            'coverage_pct': coverage_pct
        }

    # Filter based on criteria
    filtered = [
        feat for feat, stats in coverage_stats.items()
        if stats['overlap'] >= min_obs and stats['coverage_pct'] >= min_coverage_pct
    ]

    removed = [feat for feat in features if feat not in filtered]

    print(f"Original features: {len(features)}")
    print(f"After coverage filter (>={min_coverage_pct}%, >={min_obs} obs): {len(filtered)}")
    print(f"Removed: {len(removed)}")

    if removed:
        print(f"\n  Sample of removed features (showing first 20):")
        for feat in removed[:20]:
            stats = coverage_stats.get(feat, {})
            print(f"    {feat:60s} {stats.get('overlap', 0):6,} ({stats.get('coverage_pct', 0):5.1f}%)")
        if len(removed) > 20:
            print(f"    ... and {len(removed)-20} more")

    # Show top features by coverage
    sorted_features = sorted(
        [(f, coverage_stats[f]) for f in filtered],
        key=lambda x: x[1]['overlap'],
        reverse=True
    )

    print(f"\n  Top 50 features by coverage:")
    for i, (feat, stats) in enumerate(sorted_features[:50], 1):
        print(f"    {i:3d}. {feat:60s} {stats['overlap']:6,} ({stats['coverage_pct']:5.1f}%)")

    return filtered, coverage_stats


def remove_highly_correlated(df, features, target_col, corr_threshold=0.95):
    """
    Remove highly correlated features to reduce multicollinearity.

    For each pair of features with correlation > threshold:
    - Keep the one more correlated with target
    - Drop the other

    Args:
        df: DataFrame with features
        features: List of feature names
        target_col: Target column name
        corr_threshold: Correlation threshold (default 0.95)

    Returns:
        List of filtered features
    """
    print(f"\n{'='*70}")
    print("REMOVING HIGHLY CORRELATED FEATURES")
    print(f"{'='*70}")

    available_features = [f for f in features if f in df.columns]

    if len(available_features) < 2:
        print(f"[WARN] Only {len(available_features)} features available")
        return available_features

    # Don't dropna on all features at once - compute correlations pairwise
    # to handle missing data more gracefully
    print(f"Computing pairwise correlations for {len(available_features)} features...")

    # Compute correlation with target first (handle missing data)
    target_corr = {}
    for feat in available_features:
        valid_mask = df[feat].notna() & df[target_col].notna()
        if valid_mask.sum() < 100:  # Need at least 100 obs
            target_corr[feat] = 0
            continue
        target_corr[feat] = df.loc[valid_mask, [feat, target_col]].corr().iloc[0, 1]

    target_corr = pd.Series(target_corr).abs()

    # Compute pairwise correlations between features
    print(f"Computing pairwise feature correlations...")
    corr_matrix = pd.DataFrame(np.nan, index=available_features, columns=available_features)

    for i, feat1 in enumerate(available_features):
        corr_matrix.loc[feat1, feat1] = 1.0
        for feat2 in available_features[i+1:]:
            valid_mask = df[feat1].notna() & df[feat2].notna()
            if valid_mask.sum() < 100:
                corr_matrix.loc[feat1, feat2] = 0
                corr_matrix.loc[feat2, feat1] = 0
                continue
            corr_val = df.loc[valid_mask, [feat1, feat2]].corr().iloc[0, 1]
            corr_matrix.loc[feat1, feat2] = abs(corr_val)
            corr_matrix.loc[feat2, feat1] = abs(corr_val)

    # Find pairs to drop
    to_drop = set()
    dropped_pairs = []

    for i, feat1 in enumerate(available_features):
        if feat1 in to_drop:
            continue
        for feat2 in available_features[i+1:]:
            if feat2 in to_drop:
                continue

            corr_val = corr_matrix.loc[feat1, feat2]
            if pd.isna(corr_val):
                continue

            if corr_val > corr_threshold:
                # Keep the feature more correlated with target
                if target_corr[feat1] > target_corr[feat2]:
                    to_drop.add(feat2)
                    dropped_pairs.append((feat2, feat1, corr_val))
                else:
                    to_drop.add(feat1)
                    dropped_pairs.append((feat1, feat2, corr_val))
                    break  # feat1 is dropped, move to next feat1

    filtered = [f for f in available_features if f not in to_drop]

    print(f"\nOriginal features: {len(available_features)}")
    print(f"After removing correlations >{corr_threshold}: {len(filtered)}")
    print(f"Removed: {len(to_drop)}")

    if dropped_pairs:
        print(f"\n  Sample of removed pairs (showing first 20):")
        for dropped, kept, corr in dropped_pairs[:20]:
            print(f"    Dropped: {dropped:50s} (corr={corr:.3f} with {kept[:40]})")
        if len(dropped_pairs) > 20:
            print(f"    ... and {len(dropped_pairs)-20} more pairs")

    return filtered


def fit_regularized_model(X_train, y_train, method="lasso", cv_folds=5):
    """
    Fit Lasso or ElasticNet with cross-validation for automatic lambda selection.

    Args:
        X_train: Training features (numpy array or DataFrame)
        y_train: Training target
        method: "lasso" or "elasticnet"
        cv_folds: Number of CV folds

    Returns:
        Fitted model
    """
    print(f"\n{'='*70}")
    print(f"FITTING {method.upper()} WITH {cv_folds}-FOLD CROSS-VALIDATION")
    print(f"{'='*70}")
    print(f"Training observations: {len(X_train):,}")
    print(f"Features: {X_train.shape[1]}")

    if method == "lasso":
        model = LassoCV(
            cv=cv_folds,
            max_iter=20000,
            random_state=42,
            n_jobs=-1,
            selection="random",
            n_alphas=100,  # More alpha values for better selection
        )
    elif method == "elasticnet":
        model = ElasticNetCV(
            l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99],  # More l1_ratio values
            cv=cv_folds,
            max_iter=20000,
            random_state=42,
            n_jobs=-1,
            selection="random",
            n_alphas=100,
        )
    else:
        raise ValueError(f"Unknown method: {method}")

    print(f"\nFitting model...")
    model.fit(X_train, y_train)

    print(f"\n  Best alpha: {model.alpha_:.6f}")
    if hasattr(model, "l1_ratio_"):
        print(f"  Best l1_ratio: {model.l1_ratio_:.3f}")

    # Count non-zero coefficients
    non_zero = (model.coef_ != 0).sum()
    print(f"  Non-zero coefficients: {non_zero} / {len(model.coef_)}")

    return model


def refit_with_ols(X, y, selected_features):
    """
    Refit using OLS on selected features for unbiased coefficients.

    Lasso/ElasticNet shrink coefficients toward zero (biased).
    OLS on selected features gives unbiased estimates.

    Args:
        X: Feature matrix (DataFrame)
        y: Target vector
        selected_features: Features selected by regularization

    Returns:
        OLS results object
    """
    print(f"\n{'='*70}")
    print("REFITTING WITH OLS ON SELECTED FEATURES")
    print(f"{'='*70}")

    X_selected = sm.add_constant(X[selected_features])
    model = sm.OLS(y, X_selected)
    results = model.fit(cov_type="HC3")  # Robust standard errors

    print(f"\nR-squared: {results.rsquared:.4f}")
    print(f"Adj. R-squared: {results.rsquared_adj:.4f}")
    print(f"N: {results.nobs:,}")
    print(f"\n{results.summary()}")

    return results


def compute_forward_returns(df, periods=[1, 2, 4]):
    """
    Compute forward returns for IC calculation.

    Args:
        df: Panel DataFrame with MultiIndex (Ticker, QuarterDate)
        periods: List of forward periods in quarters

    Returns:
        DataFrame with added columns: Return_1Q, Return_2Q, Return_4Q
    """
    df = df.copy()

    # Use MarketCap if available, otherwise exp(LogMarketCap)
    if "MarketCap" not in df.columns:
        if TARGET_COL in df.columns:
            df["MarketCap"] = np.exp(df[TARGET_COL])
        else:
            print("[ERROR] Cannot compute returns: no MarketCap or LogMarketCap column")
            return df

    for p in periods:
        # Forward return = (MarketCap_t+p / MarketCap_t) - 1
        df[f"Return_{p}Q"] = (
            df.groupby(level="Ticker")["MarketCap"]
            .pct_change(periods=p)
            .shift(-p)  # Shift backwards to align with current quarter
        )

    return df


def evaluate_ic(residuals, returns, label=""):
    """
    Compute Information Coefficient (Spearman correlation between residuals and returns).

    Negative IC is good: undervalued stocks (negative residuals) should outperform.

    Args:
        residuals: Model residuals (actual - predicted)
        returns: Forward returns
        label: Label for printing

    Returns:
        Dict with IC, p-value, N
    """
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


def run_all_fundamentals_model(
    method="lasso",
    train_end=None,  # If None, use all data for training
    val_end=None,    # If None, no validation split
    max_features=20,
    min_coverage_pct=10.0,
    corr_threshold=0.95,
    sector_code="ALL",
    min_obs=500,
):
    """
    Run the all-fundamentals model with Lasso/ElasticNet feature selection.

    Args:
        method: "lasso" or "elasticnet"
        train_end: End of training period (None = use all data)
        val_end: End of validation period (None = no validation split)
        max_features: Maximum features to select
        min_coverage_pct: Minimum coverage percentage
        corr_threshold: Correlation threshold for pre-filtering
        sector_code: Sector filter (ALL, CAP, COM, CON, FIN, IND, INT, RE)
        min_obs: Minimum overlapping observations with target for a feature

    Returns:
        dict with results
    """
    print("\n" + "="*70)
    print(f"ALL-FUNDAMENTALS MODEL: {method.upper()}")
    print("="*70)

    # Load data
    print("\nLoading panel data...")
    df_panel, _, _ = prepare_panel_data()

    print(f"Loaded {len(df_panel):,} observations")
    print(f"Date range: {df_panel.index.get_level_values('QuarterDate').min()} to {df_panel.index.get_level_values('QuarterDate').max()}")
    print(f"Unique tickers: {df_panel.index.get_level_values('Ticker').nunique()}")

    sector_label = SECTOR_DISPLAY_NAMES.get(sector_code, sector_code)
    if sector_code != "ALL":
        if "SectorGroup" not in df_panel.columns:
            print(f"[ERROR] SectorGroup column missing; cannot filter to {sector_code}")
            return None
        df_panel = df_panel[df_panel["SectorGroup"] == sector_code]
        print(f"\nFiltering to sector {sector_label} ({sector_code})")
        print(f"Observations after filter: {len(df_panel):,}")
        if df_panel.empty:
            print(f"[ERROR] No data available for sector {sector_code}")
            return None

    # Get ALL z-scored fundamentals
    all_fundamentals = get_all_z_scored_fundamentals(df_panel)
    print(f"\nTotal z-scored fundamentals available: {len(all_fundamentals)}")

    # Filter by coverage
    high_coverage_features, coverage_stats = filter_by_coverage(
        df_panel, all_fundamentals, TARGET_COL, min_coverage_pct=min_coverage_pct, min_obs=min_obs
    )

    if len(high_coverage_features) < 10:
        print(f"\n[ERROR] Only {len(high_coverage_features)} features with sufficient coverage")
        return None

    # Remove highly correlated features
    filtered_features = remove_highly_correlated(
        df_panel, high_coverage_features, TARGET_COL, corr_threshold=corr_threshold
    )

    if len(filtered_features) < 10:
        print(f"\n[ERROR] Only {len(filtered_features)} features after correlation filter")
        return None

    print(f"\nFinal feature pool: {len(filtered_features)} fundamentals")

    # Compute forward returns
    print("\nComputing forward returns (1Q, 2Q, 4Q)...")
    df_panel = compute_forward_returns(df_panel)

    # Split data
    dates = df_panel.index.get_level_values("QuarterDate")

    if train_end is None:
        # Use all data for training (production mode)
        df_train = df_panel
        df_val = pd.DataFrame()
        df_test = pd.DataFrame()
        print(f"\n{'='*70}")
        print("USING ALL DATA FOR TRAINING (Production Mode)")
        print(f"{'='*70}")
        print(f"Training:   {df_train.index.get_level_values('QuarterDate').min()} to {df_train.index.get_level_values('QuarterDate').max()} (N={len(df_train):,})")
        print("\n⚠️  No validation/test split - using all historical data")
        print("   This is appropriate for production deployment where you want")
        print("   to use all available data and will evaluate on future quarters.")
    else:
        # Split for backtesting
        df_train = df_panel[dates < pd.Timestamp(train_end)]
        if val_end is not None:
            df_val = df_panel[(dates >= pd.Timestamp(train_end)) & (dates < pd.Timestamp(val_end))]
            df_test = df_panel[dates >= pd.Timestamp(val_end)]
        else:
            df_val = pd.DataFrame()
            df_test = df_panel[dates >= pd.Timestamp(train_end)]

        print(f"\n{'='*70}")
        print("DATA SPLIT (Backtest Mode)")
        print(f"{'='*70}")
        print(f"Training:   {df_train.index.get_level_values('QuarterDate').min()} to {df_train.index.get_level_values('QuarterDate').max()} (N={len(df_train):,})")
        if not df_val.empty:
            print(f"Validation: {df_val.index.get_level_values('QuarterDate').min()} to {df_val.index.get_level_values('QuarterDate').max()} (N={len(df_val):,})")
        if not df_test.empty:
            print(f"Test:       {df_test.index.get_level_values('QuarterDate').min()} to {df_test.index.get_level_values('QuarterDate').max()} (N={len(df_test):,})")

    # Prepare training data with median imputation
    print(f"\nPreparing training data with median imputation...")

    # Get target valid indices
    y_train_full = df_train[TARGET_COL]
    valid_target_idx = y_train_full.dropna().index

    print(f"Training observations with valid target: {len(valid_target_idx):,}")

    if len(valid_target_idx) < 100:
        print("[ERROR] Training sample too small (< 100 observations with valid target)")
        return None

    # Get features for valid target indices
    X_train = df_train.loc[valid_target_idx, filtered_features]
    y_train = y_train_full.loc[valid_target_idx]

    # Check how much data we have per feature
    print(f"\nFeature coverage in training set:")
    feature_coverage = {}
    for feat in filtered_features:
        valid_count = X_train[feat].notna().sum()
        pct = 100 * valid_count / len(X_train)
        feature_coverage[feat] = valid_count
        if pct < 50:
            print(f"  [WARN] {feat:60s} only {pct:5.1f}% coverage")

    # Median imputation for missing values
    print(f"\nApplying median imputation for missing values...")
    X_train_imputed = X_train.copy()
    for col in filtered_features:
        if X_train_imputed[col].isna().any():
            median_val = X_train_imputed[col].median()
            n_missing = X_train_imputed[col].isna().sum()
            X_train_imputed[col] = X_train_imputed[col].fillna(median_val)
            if n_missing > 0:
                pct_missing = 100 * n_missing / len(X_train_imputed)
                if pct_missing > 10:
                    print(f"  Imputed {n_missing:,} ({pct_missing:.1f}%) missing values in {col}")

    print(f"\nTraining sample size: {len(X_train_imputed):,}")
    print(f"Number of features: {X_train_imputed.shape[1]}")

    # Fit regularized model
    reg_model = fit_regularized_model(X_train_imputed.values, y_train.values, method=method)

    # Extract selected features
    coefs = pd.Series(reg_model.coef_, index=filtered_features)
    selected_features_all = coefs[coefs != 0].sort_values(key=abs, ascending=False)

    if len(selected_features_all) == 0:
        print(f"[ERROR] {method} selected ZERO features. Try reducing regularization.")
        return None

    print(f"\n{method.upper()} selected {len(selected_features_all)} features:")
    print(f"\nTop {min(30, len(selected_features_all))} by absolute coefficient:")
    for i, (feat, coef) in enumerate(selected_features_all.head(30).items(), 1):
        # Show coverage stats
        cov = coverage_stats.get(feat, {})
        print(f"  {i:3d}. {feat:60s} coef={coef:+.6f}  (coverage: {cov.get('coverage_pct', 0):5.1f}%)")

    # Limit to max_features
    selected_features = selected_features_all.head(max_features).index.tolist()
    print(f"\nUsing top {len(selected_features)} features for OLS refit")

    # Refit with OLS on imputed data
    X_train_selected = X_train_imputed[selected_features]
    ols_results = refit_with_ols(X_train_selected, y_train, selected_features)

    # Evaluate on all periods
    print(f"\n{'='*70}")
    print("INFORMATION COEFFICIENT (IC) EVALUATION")
    print(f"{'='*70}")
    print("IC = Spearman(residuals, 1Q forward returns)")
    print("Negative IC is GOOD (undervalued → outperform)\n")

    results = {}

    # Store median values for imputation in other periods
    feature_medians = X_train_imputed[selected_features].median()

    for period_name, df_period in [("Training", df_train), ("Validation", df_val), ("Test", df_test)]:
        if df_period.empty:
            print(f"{period_name:20s} Empty, skipped")
            continue

        # Prepare data with median imputation
        y_period = df_period[TARGET_COL]
        returns_1q = df_period.get("Return_1Q")

        # Get valid target indices
        valid_target = y_period.notna()
        if returns_1q is not None:
            valid_target = valid_target & returns_1q.notna()

        if valid_target.sum() < 10:
            print(f"{period_name:20s} N < 10, skipped")
            continue

        valid_idx = df_period.index[valid_target]

        # Get features and impute missing values using training medians
        X_period = df_period.loc[valid_idx, selected_features].copy()
        for col in selected_features:
            X_period[col] = X_period[col].fillna(feature_medians[col])

        y_period = y_period.loc[valid_idx]

        # Predict using OLS model
        X_with_const = sm.add_constant(X_period)
        y_pred = ols_results.predict(X_with_const)
        residuals = y_period - y_pred

        # Compute IC
        if returns_1q is not None:
            returns_1q_period = returns_1q.loc[valid_idx]
            ic_stats = evaluate_ic(residuals, returns_1q_period, label=period_name)
            results[f"{period_name.lower()}_ic"] = ic_stats

        results[f"{period_name.lower()}_residuals"] = residuals

    # Predict on full dataset with median imputation
    results_df = df_panel.copy()
    results_df["Predicted_LogMarketCap"] = np.nan
    results_df["Residual_LogMarketCap"] = np.nan

    y_full = df_panel[TARGET_COL]
    valid_idx = y_full.dropna().index

    if len(valid_idx) > 0:
        # Get features and impute using training medians
        X_full = df_panel.loc[valid_idx, selected_features].copy()
        for col in selected_features:
            X_full[col] = X_full[col].fillna(feature_medians[col])

        X_with_const = sm.add_constant(X_full)
        y_pred_full = ols_results.predict(X_with_const)

        results_df.loc[valid_idx, "Predicted_LogMarketCap"] = y_pred_full
        results_df.loc[valid_idx, "Residual_LogMarketCap"] = y_full.loc[valid_idx] - y_pred_full
        results_df["Overvaluation_pct"] = (np.exp(results_df["Residual_LogMarketCap"]) - 1) * 100

    # Save results
    if sector_code == "ALL":
        output_key = f"all_fundamentals_{method}"
    else:
        output_key = f"{sector_code.lower()}_{method}_fundamentals"

    # Save full results
    output_path = OUTPUT_DIR / f"{output_key}_results.csv"
    results_df.reset_index().to_csv(output_path, index=False)
    print(f"\n[OK] Results saved to {output_path}")

    # Save model summary
    summary_path = OUTPUT_DIR / f"{output_key}_model_summary.txt"
    with open(summary_path, "w") as f:
        f.write(f"ALL-FUNDAMENTALS MODEL: {method.upper()} ({sector_label})\n")
        f.write(f"="*70 + "\n\n")
        f.write(f"Sector: {sector_code}\n\n")
        f.write(f"Selected Features ({len(selected_features)}):\n")
        f.write(f"-"*70 + "\n")
        for i, feat in enumerate(selected_features, 1):
            coef = ols_results.params.get(feat, np.nan)
            cov = coverage_stats.get(feat, {})
            f.write(f"{i:3d}. {feat:60s} coef={coef:+.6f}  cov={cov.get('coverage_pct', 0):5.1f}%\n")
        f.write("\n" + "="*70 + "\n")
        f.write("OLS REGRESSION RESULTS:\n")
        f.write("="*70 + "\n")
        f.write(str(ols_results.summary()))
        f.write("\n\n" + "="*70 + "\n")
        f.write("INFORMATION COEFFICIENTS:\n")
        f.write("="*70 + "\n")
        for key, val in results.items():
            if key.endswith("_ic") and val is not None:
                f.write(f"  {key}: IC={val['IC']:+.4f}, p={val['p_value']:.4f}, N={val['N']}\n")

    print(f"[OK] Model summary saved to {summary_path}")

    # Save selected features with details
    features_path = OUTPUT_DIR / f"{output_key}_selected_features.txt"
    with open(features_path, "w") as f:
        f.write(f"SELECTED FEATURES: {method.upper()} + OLS ({sector_label})\n")
        f.write(f"="*70 + "\n")
        f.write(f"Total: {len(selected_features)}\n")
        f.write(f"Method: {method}\n")
        f.write(f"Sector: {sector_code}\n")
        f.write(f"Alpha: {reg_model.alpha_:.6f}\n")
        if hasattr(reg_model, "l1_ratio_"):
            f.write(f"L1 Ratio: {reg_model.l1_ratio_:.3f}\n")
        f.write("\n" + "="*70 + "\n\n")

        f.write(f"{'Rank':>4}  {'Feature':60}  {'Lasso Coef':>12}  {'OLS Coef':>12}  {'Coverage':>10}\n")
        f.write("-"*105 + "\n")

        for i, feat in enumerate(selected_features, 1):
            lasso_coef = selected_features_all.get(feat, 0)
            ols_coef = ols_results.params.get(feat, np.nan)
            cov = coverage_stats.get(feat, {}).get('coverage_pct', 0)
            f.write(f"{i:4d}  {feat:60}  {lasso_coef:+12.6f}  {ols_coef:+12.6f}  {cov:9.1f}%\n")

    print(f"[OK] Selected features saved to {features_path}")

    print(f"\n{'='*70}")
    print(f"MODEL PIPELINE COMPLETE ({sector_label})")
    print(f"{'='*70}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Files created:")
    print(f"  1. {output_key}_results.csv          (all predictions & residuals)")
    print(f"  2. {output_key}_model_summary.txt    (OLS regression output)")
    print(f"  3. {output_key}_selected_features.txt (feature list with coefficients)")

    return {
        "method": method,
        "selected_features": selected_features,
        "ols_results": ols_results,
        "results_df": results_df,
        "coverage_stats": coverage_stats,
        "output_key": output_key,
        "sector_code": sector_code,
        "sector_label": sector_label,
        **results,
    }


def run_sector_batch(sector_codes, **kwargs):
    """Run the model for multiple sectors and write a compact summary."""
    summaries = []
    for code in sector_codes:
        print("\n" + "="*80)
        print(f"RUNNING SECTOR: {code}")
        print("="*80)
        res = run_all_fundamentals_model(sector_code=code, **kwargs)
        if res is None:
            print(f"[WARN] Skipping sector {code} (no result)")
            continue
        params = res["ols_results"].params if "ols_results" in res else {}
        top_feats = []
        for feat in res.get("selected_features", [])[:10]:
            coef = params.get(feat, np.nan)
            top_feats.append(f"{feat} ({coef:+.3f})")
        summaries.append({
            "sector": res.get("sector_code", code),
            "label": res.get("sector_label", code),
            "n_features": len(res.get("selected_features", [])),
            "top_features": top_feats,
            "output_key": res.get("output_key"),
        })

    if not summaries:
        print("[WARN] No sector runs completed.")
        return summaries

    lines = []
    lines.append("# Sector Models — Selected Fundamentals")
    lines.append("")
    lines.append("| Sector | #Features | Top Features (coef) | Outputs |")
    lines.append("|---|---|---|---|")
    for row in summaries:
        output_stub = f"{row['output_key']}_results.csv" if row.get("output_key") else "n/a"
        lines.append(
            f"| {row['label']} | {row['n_features']} | {', '.join(row['top_features'])} | {output_stub} |"
        )
    summary_path = OUTPUT_DIR / "sector_feature_summary.md"
    summary_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n[OK] Sector summary saved to {summary_path}")
    return summaries


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run all-fundamentals model with Lasso/ElasticNet")
    parser.add_argument("--method", default="lasso", choices=["lasso", "elasticnet"],
                       help="Regularization method")
    parser.add_argument("--max-features", type=int, default=20,
                       help="Maximum features to select (default: 20)")
    parser.add_argument("--min-coverage", type=float, default=10.0,
                       help="Minimum coverage percentage (default: 10.0)")
    parser.add_argument("--corr-threshold", type=float, default=0.95,
                       help="Correlation threshold for pre-filtering (default: 0.95)")
    parser.add_argument("--train-end", default=None,
                       help="End of training period (YYYY-MM-DD). None = use all data (default: None)")
    parser.add_argument("--val-end", default=None,
                       help="End of validation period (YYYY-MM-DD). None = no validation split (default: None)")
    parser.add_argument("--production", action="store_true",
                       help="Production mode: train on all data (equivalent to --train-end=None)")
    parser.add_argument("--sector", default="ALL",
                       help="Sector code to run (ALL, CAP, COM, CON, FIN, IND, INT, RE)")
    parser.add_argument("--sectors", default=None,
                       help="Comma-separated sector codes to run as a batch")
    parser.add_argument("--all-sectors", action="store_true",
                       help="Run the 7 standard sectors (CAP, COM, CON, FIN, IND, INT, RE)")
    parser.add_argument("--min-obs", type=int, default=500,
                       help="Minimum overlapping observations for a feature (default: 500)")

    args = parser.parse_args()

    # Production mode overrides train-end
    if args.production:
        train_end = None
        val_end = None
    else:
        train_end = args.train_end
        val_end = args.val_end

    # Determine sector list
    sector_list = []
    if args.all_sectors:
        sector_list = ["CAP", "COM", "CON", "FIN", "IND", "INT", "RE"]
    elif args.sectors:
        sector_list = [s.strip().upper() for s in args.sectors.split(",") if s.strip()]

    if sector_list:
        run_sector_batch(
            sector_list,
            method=args.method,
            max_features=args.max_features,
            min_coverage_pct=args.min_coverage,
            corr_threshold=args.corr_threshold,
            train_end=train_end,
            val_end=val_end,
            min_obs=args.min_obs,
        )
        print("\n[SUCCESS] Sector batch completed.")
    else:
        result = run_all_fundamentals_model(
            method=args.method,
            max_features=args.max_features,
            min_coverage_pct=args.min_coverage,
            corr_threshold=args.corr_threshold,
            train_end=train_end,
            val_end=val_end,
            sector_code=args.sector.upper(),
            min_obs=args.min_obs,
        )

        if result is None:
            print("\n[ERROR] Model failed. Check logs above.")
            sys.exit(1)

        print("\n[SUCCESS] Model completed successfully!")
