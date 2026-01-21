"""
Proper Feature Selection Pipeline with Walk-Forward Validation

This fixes the data leakage issue in IC-based feature selection by:
1. Splitting data into train/validation periods
2. Selecting features on training data only
3. Testing predictive power on out-of-sample validation data
4. Using expanding window to progressively train on more data
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LassoCV, ElasticNetCV
from scipy.stats import spearmanr
from pathlib import Path

def compute_forward_returns(df, periods=[1, 2, 4]):
    """
    Compute forward returns for multiple horizons.

    Args:
        df: DataFrame with MultiIndex (Ticker, QuarterDate) and MarketCap column
        periods: List of forward periods in quarters

    Returns:
        DataFrame with added columns: Return_1Q, Return_2Q, Return_4Q
    """
    df = df.copy()

    for p in periods:
        # Forward return = (MarketCap_t+p / MarketCap_t) - 1
        df[f"Return_{p}Q"] = (
            df.groupby(level="Ticker")["MarketCap"]
            .pct_change(periods=p)
            .shift(-p)  # Shift backwards to align with current quarter
        )

    return df


def select_features_walk_forward(
    df,
    feature_pool,
    target_col,
    train_start="2016-01-01",
    validation_start="2020-01-01",
    test_start="2023-01-01",
    method="lasso",
    cv_folds=5,
    max_features=15,
    min_ic=0.05,
):
    """
    Select features using walk-forward validation to avoid data leakage.

    Process:
    1. Train period (2016-2020): Fit Lasso/ElasticNet, select features
    2. Validation period (2020-2023): Test IC on out-of-sample data
    3. Test period (2023-2026): Final evaluation on holdout data

    Args:
        df: Panel data with MultiIndex (Ticker, QuarterDate)
        feature_pool: List of candidate features (z-scored fundamentals)
        target_col: Target variable (e.g., "LogMCap_z")
        train_start, validation_start, test_start: Date strings
        method: "lasso" or "elasticnet"
        cv_folds: Cross-validation folds for regularization parameter
        max_features: Maximum number of features to select
        min_ic: Minimum IC threshold for validation

    Returns:
        dict with:
            - selected_features: List of selected feature names
            - train_ic: IC on training data (in-sample)
            - validation_ic: IC on validation data (out-of-sample)
            - test_ic: IC on test data (holdout)
            - model: Fitted model object
    """

    # Split data into periods
    dates = df.index.get_level_values("QuarterDate")

    df_train = df[dates < pd.Timestamp(validation_start)]
    df_val = df[(dates >= pd.Timestamp(validation_start)) & (dates < pd.Timestamp(test_start))]
    df_test = df[dates >= pd.Timestamp(test_start)]

    print(f"\n{'='*60}")
    print("WALK-FORWARD FEATURE SELECTION")
    print(f"{'='*60}")
    print(f"Training period:   {df_train.index.get_level_values('QuarterDate').min()} to {df_train.index.get_level_values('QuarterDate').max()}")
    print(f"Validation period: {df_val.index.get_level_values('QuarterDate').min()} to {df_val.index.get_level_values('QuarterDate').max()}")
    print(f"Test period:       {df_test.index.get_level_values('QuarterDate').min()} to {df_test.index.get_level_values('QuarterDate').max()}")
    print(f"Train N: {len(df_train):,} | Val N: {len(df_val):,} | Test N: {len(df_test):,}")

    # Compute forward returns on all data
    df_train = compute_forward_returns(df_train)
    df_val = compute_forward_returns(df_val)
    df_test = compute_forward_returns(df_test)

    # Drop rows with missing forward returns (last periods)
    df_train = df_train.dropna(subset=["Return_1Q"])
    df_val = df_val.dropna(subset=["Return_1Q"])
    df_test = df_test.dropna(subset=["Return_1Q"])

    # Prepare training data
    X_train = df_train[feature_pool].dropna()
    y_train = df_train.loc[X_train.index, target_col]

    print(f"\nFeature pool size: {len(feature_pool)}")
    print(f"Training sample after dropna: {len(X_train):,}")

    # Fit regularized model
    if method == "lasso":
        model = LassoCV(
            cv=cv_folds,
            max_iter=5000,
            random_state=42,
            n_jobs=-1,
        )
    elif method == "elasticnet":
        model = ElasticNetCV(
            l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9],
            cv=cv_folds,
            max_iter=5000,
            random_state=42,
            n_jobs=-1,
        )
    else:
        raise ValueError(f"Unknown method: {method}")

    print(f"\nFitting {method.upper()} with {cv_folds}-fold CV...")
    model.fit(X_train, y_train)

    # Extract selected features (non-zero coefficients)
    coefs = pd.Series(model.coef_, index=feature_pool)
    selected_features = coefs[coefs != 0].sort_values(key=abs, ascending=False)

    print(f"\n{len(selected_features)} features selected by {method}:")
    for feat, coef in selected_features.head(max_features).items():
        print(f"  {feat:40s} {coef:+.4f}")

    # Limit to max_features
    selected_features = selected_features.head(max_features).index.tolist()

    # Predict residuals on each dataset
    def compute_residuals_and_ic(df_subset, features, model, target_col):
        X = df_subset[features]
        y = df_subset[target_col]

        # Get valid rows (no NaN in features or target)
        valid_idx = X.dropna().index.intersection(y.dropna().index)
        X = X.loc[valid_idx]
        y = y.loc[valid_idx]

        if len(X) == 0:
            return None, None, None

        # Predict
        y_pred = model.predict(X)
        residuals = y - y_pred

        # Get forward returns
        returns_1q = df_subset.loc[valid_idx, "Return_1Q"]

        # Remove NaN returns
        valid_returns_idx = returns_1q.dropna().index
        residuals = residuals.loc[valid_returns_idx]
        returns_1q = returns_1q.loc[valid_returns_idx]

        if len(residuals) < 10:
            return residuals, returns_1q, None

        # Compute IC (Spearman correlation: undervalued → positive returns)
        ic, pval = spearmanr(residuals, returns_1q)

        return residuals, returns_1q, {"IC": ic, "p_value": pval, "N": len(residuals)}

    # Evaluate on all periods
    train_residuals, train_returns, train_ic = compute_residuals_and_ic(
        df_train, selected_features, model, target_col
    )
    val_residuals, val_returns, val_ic = compute_residuals_and_ic(
        df_val, selected_features, model, target_col
    )
    test_residuals, test_returns, test_ic = compute_residuals_and_ic(
        df_test, selected_features, model, target_col
    )

    # Print results
    print(f"\n{'='*60}")
    print("INFORMATION COEFFICIENT (IC) RESULTS")
    print(f"{'='*60}")
    print("IC = Spearman correlation between residuals and 1Q forward returns")
    print("Negative IC is GOOD (undervalued → positive returns)\n")

    if train_ic:
        print(f"Training IC:   {train_ic['IC']:+.4f} (p={train_ic['p_value']:.4f}, N={train_ic['N']:,})")
    if val_ic:
        print(f"Validation IC: {val_ic['IC']:+.4f} (p={val_ic['p_value']:.4f}, N={val_ic['N']:,}) ← OUT-OF-SAMPLE")
    if test_ic:
        print(f"Test IC:       {test_ic['IC']:+.4f} (p={test_ic['p_value']:.4f}, N={test_ic['N']:,}) ← HOLDOUT")

    # Validation check
    if val_ic and abs(val_ic['IC']) < min_ic:
        print(f"\n⚠️  WARNING: Validation IC ({val_ic['IC']:+.4f}) below threshold ({min_ic})")
        print("    This suggests weak out-of-sample predictability.")
        print("    Consider: More features, different feature engineering, or regime-specific models.")
    elif val_ic and val_ic['p_value'] > 0.05:
        print(f"\n⚠️  WARNING: Validation IC not statistically significant (p={val_ic['p_value']:.4f})")
    elif val_ic:
        print(f"\n✓ Validation IC is significant and above threshold!")

    return {
        "selected_features": selected_features,
        "model": model,
        "train_ic": train_ic,
        "validation_ic": val_ic,
        "test_ic": test_ic,
        "train_residuals": train_residuals,
        "val_residuals": val_residuals,
        "test_residuals": test_residuals,
    }


def select_features_expanding_window(
    df,
    feature_pool,
    target_col,
    start_train_date="2016-01-01",
    start_test_date="2020-01-01",
    refit_frequency="4Q",  # Refit every 4 quarters (1 year)
    method="lasso",
    max_features=15,
):
    """
    Expanding window approach: progressively train on more data.

    Process:
    - Start training on 2016-2019 (4 years)
    - Test on 2020Q1, refit on 2016-2020Q1
    - Test on 2020Q2, refit on 2016-2020Q2
    - ... and so on

    This simulates real-world deployment where you retrain periodically
    with new data.

    Args:
        df: Panel data with MultiIndex (Ticker, QuarterDate)
        feature_pool: List of candidate features
        target_col: Target variable
        start_train_date: Start of training window
        start_test_date: First test quarter
        refit_frequency: How often to refit ("1Q", "4Q", etc.)
        method: "lasso" or "elasticnet"
        max_features: Max features to select

    Returns:
        DataFrame with columns: [QuarterDate, IC, N_obs, Selected_Features]
    """

    df = compute_forward_returns(df)

    dates = df.index.get_level_values("QuarterDate")
    test_quarters = sorted(dates[dates >= pd.Timestamp(start_test_date)].unique())

    # Refit frequency
    if refit_frequency == "1Q":
        refit_every = 1
    elif refit_frequency == "4Q":
        refit_every = 4
    else:
        refit_every = int(refit_frequency.replace("Q", ""))

    results = []
    current_model = None
    current_features = None

    print(f"\n{'='*70}")
    print("EXPANDING WINDOW BACKTEST")
    print(f"{'='*70}")
    print(f"Training start:   {start_train_date}")
    print(f"Testing start:    {start_test_date}")
    print(f"Refit frequency:  Every {refit_frequency}")
    print(f"Test quarters:    {len(test_quarters)}")
    print(f"{'='*70}\n")

    for i, test_quarter in enumerate(test_quarters):
        # Refit model every N quarters
        if i % refit_every == 0:
            print(f"\n[Refitting model for quarter {i+1}/{len(test_quarters)}]")

            # Training data: all data up to test quarter
            df_train = df[dates < test_quarter]
            df_train = df_train.dropna(subset=["Return_1Q"])

            X_train = df_train[feature_pool].dropna()
            y_train = df_train.loc[X_train.index, target_col]

            # Fit model
            if method == "lasso":
                model = LassoCV(cv=5, max_iter=5000, random_state=42, n_jobs=-1)
            else:
                model = ElasticNetCV(
                    l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9],
                    cv=5,
                    max_iter=5000,
                    random_state=42,
                    n_jobs=-1,
                )

            model.fit(X_train, y_train)

            # Select features
            coefs = pd.Series(model.coef_, index=feature_pool)
            selected = coefs[coefs != 0].sort_values(key=abs, ascending=False).head(max_features)

            current_model = model
            current_features = selected.index.tolist()

            print(f"  Training on {len(X_train):,} obs → {len(current_features)} features selected")

        # Test on current quarter
        df_test = df[dates == test_quarter]
        df_test = df_test.dropna(subset=["Return_1Q"])

        X_test = df_test[current_features]
        y_test = df_test[target_col]

        valid_idx = X_test.dropna().index.intersection(y_test.dropna().index)
        X_test = X_test.loc[valid_idx]
        y_test = y_test.loc[valid_idx]

        if len(X_test) < 5:
            print(f"  {test_quarter.date()}: Skipped (N={len(X_test)})")
            continue

        # Predict and compute IC
        y_pred = current_model.predict(X_test)
        residuals = y_test - y_pred
        returns_1q = df_test.loc[valid_idx, "Return_1Q"]

        valid_returns_idx = returns_1q.dropna().index
        residuals = residuals.loc[valid_returns_idx]
        returns_1q = returns_1q.loc[valid_returns_idx]

        if len(residuals) >= 5:
            ic, pval = spearmanr(residuals, returns_1q)
            print(f"  {test_quarter.date()}: IC = {ic:+.3f} (p={pval:.3f}, N={len(residuals):3d})")

            results.append({
                "QuarterDate": test_quarter,
                "IC": ic,
                "p_value": pval,
                "N_obs": len(residuals),
                "N_features": len(current_features),
                "Selected_Features": ", ".join(current_features[:5]) + "...",
            })

    results_df = pd.DataFrame(results)

    print(f"\n{'='*70}")
    print("SUMMARY STATISTICS")
    print(f"{'='*70}")
    print(f"Mean IC:       {results_df['IC'].mean():+.4f}")
    print(f"Median IC:     {results_df['IC'].median():+.4f}")
    print(f"Std IC:        {results_df['IC'].std():.4f}")
    print(f"% Negative IC: {(results_df['IC'] < 0).mean() * 100:.1f}% (expect >50% if signal works)")
    print(f"% Significant: {(results_df['p_value'] < 0.05).mean() * 100:.1f}%")

    return results_df


if __name__ == "__main__":
    # Example usage (you'll need to adapt to your data loading)

    print("=" * 70)
    print("PROPER FEATURE SELECTION DEMO")
    print("=" * 70)
    print("\nThis script demonstrates walk-forward feature selection.")
    print("To use it, you need to:")
    print("1. Load your panel data with MultiIndex (Ticker, QuarterDate)")
    print("2. Prepare z-scored features (your existing pipeline)")
    print("3. Call select_features_walk_forward() or select_features_expanding_window()")
    print("\nExample:")
    print("  from pooled_ols_residuals_bist import prepare_panel_data")
    print("  df_panel, factor_set, _ = prepare_panel_data()")
    print("  ")
    print("  feature_pool = [c for c in df_panel.columns if c.endswith('_z')]")
    print("  ")
    print("  results = select_features_walk_forward(")
    print("      df_panel,")
    print("      feature_pool,")
    print("      target_col='LogMCap_z',")
    print("      method='lasso',")
    print("  )")
    print("  ")
    print("  # Use results['selected_features'] for final model")
