"""
Train Multiple Versions of Factor Model with Different Time Windows

Creates 3 versions:
1. All data (2016-2025)
2. Last 5 years (2020-2025)
3. Last 3 years (2022-2025)

For each version:
- Trains model
- Generates Q3 2025 portfolio
- Creates dashboard from Nov 15, 2025
- Saves to separate output folders

Usage:
    python train_multiple_versions.py
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy.stats import spearmanr

# Add parent directory to import pooled_ols_residuals_bist
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    from pooled_ols_residuals_bist import prepare_panel_data, TARGET_COL
    print("[OK] Imported data preparation functions")
except ImportError as e:
    print(f"[ERROR] Could not import: {e}")
    print("Make sure pooled_ols_residuals_bist.py is in parent directory")
    exit(1)

ROOT = Path(__file__).resolve().parent


def train_factor_model(df_panel, train_start, train_end, version_name):
    """
    Train factor model on specified time window.

    Args:
        df_panel: Full panel data
        train_start: Start date for training (str)
        train_end: End date for training (str)
        version_name: Name for this version (str)

    Returns:
        model results, coefficients, etc.
    """
    print(f"\n{'='*80}")
    print(f"TRAINING: {version_name}")
    print(f"Period: {train_start} to {train_end}")
    print(f"{'='*80}")

    # Filter to training period
    df_train = df_panel[
        (df_panel.index.get_level_values('QuarterDate') >= train_start) &
        (df_panel.index.get_level_values('QuarterDate') <= train_end)
    ].copy()

    print(f"Training observations: {len(df_train)}")
    print(f"Date range: {df_train.index.get_level_values('QuarterDate').min()} to {df_train.index.get_level_values('QuarterDate').max()}")

    # Define factors (from FACTOR_COMPOSITION_GUIDE.md)
    factor_features = [
        'Factor_Scale',
        'Factor_Leverage',
        'Factor_RD',
        'Factor_Profitability',
        'Factor_Growth',
        'Factor_CapitalEfficiency'
    ]

    # Remove any missing factors
    available_factors = [f for f in factor_features if f in df_train.columns]
    print(f"Available factors: {available_factors}")

    # Prepare for regression
    X_train = df_train[available_factors].dropna()
    y_train = df_train.loc[X_train.index, TARGET_COL]

    # Drop rows with missing target
    valid_idx = y_train.dropna().index
    X_train = X_train.loc[valid_idx]
    y_train = y_train.loc[valid_idx]

    print(f"Training sample (after dropna): {len(X_train)}")

    if len(X_train) < 100:
        print(f"[ERROR] Training sample too small: {len(X_train)}")
        return None

    # Fit OLS model
    X_train_const = sm.add_constant(X_train)
    model = sm.OLS(y_train, X_train_const).fit(cov_type='HC3')

    print(f"\nModel R²: {model.rsquared:.3f}")
    print(f"Adj R²: {model.rsquared_adj:.3f}")
    print(f"\nCoefficients:")
    print(model.params)

    # Compute training IC (residual vs forward return)
    residuals_train = y_train - model.predict(X_train_const)

    # For IC, need forward returns
    # Simple approach: use next quarter's return
    df_train_with_resid = df_train.loc[valid_idx].copy()
    df_train_with_resid['Residual'] = residuals_train

    # Compute forward returns (next quarter)
    df_train_with_resid['MarketCap'] = np.exp(df_train_with_resid[TARGET_COL])
    df_train_with_resid['FwdReturn_1Q'] = (
        df_train_with_resid.groupby(level='Ticker')['MarketCap']
        .pct_change(periods=1)
        .shift(-1)
    )

    # IC = correlation(residual, forward return)
    ic_data = df_train_with_resid[['Residual', 'FwdReturn_1Q']].dropna()
    if len(ic_data) > 100:
        ic, ic_pval = spearmanr(ic_data['Residual'], ic_data['FwdReturn_1Q'])
        print(f"\nTraining IC (1Q forward): {ic:.4f} (p={ic_pval:.4f}, N={len(ic_data)})")
    else:
        ic = np.nan
        print("\n[WARN] Not enough data to compute IC")

    # Save model summary
    output_dir = ROOT / f"outputs_{version_name}"
    output_dir.mkdir(exist_ok=True)

    summary_path = output_dir / "model_summary.txt"
    with open(summary_path, 'w') as f:
        f.write(f"FACTOR MODEL: {version_name}\n")
        f.write(f"{'='*80}\n\n")
        f.write(f"Training Period: {train_start} to {train_end}\n")
        f.write(f"Training Observations: {len(X_train)}\n")
        f.write(f"R²: {model.rsquared:.3f}\n")
        f.write(f"Adj R²: {model.rsquared_adj:.3f}\n")
        if not np.isnan(ic):
            f.write(f"Training IC (1Q): {ic:.4f} (p={ic_pval:.4f})\n")
        f.write(f"\n{'='*80}\n")
        f.write("OLS REGRESSION RESULTS\n")
        f.write(f"{'='*80}\n")
        f.write(str(model.summary()))

    print(f"[OK] Saved model summary to {summary_path}")

    return {
        'model': model,
        'factors': available_factors,
        'ic': ic,
        'r2': model.rsquared,
        'output_dir': output_dir
    }


def predict_and_rank(df_panel, model_info, latest_quarter='2025-09-30'):
    """
    Predict on latest quarter and rank stocks.
    """
    model = model_info['model']
    factors = model_info['factors']
    output_dir = model_info['output_dir']

    print(f"\nPredicting for quarter: {latest_quarter}")

    # Filter to latest quarter
    df_latest = df_panel[
        df_panel.index.get_level_values('QuarterDate') == latest_quarter
    ].copy()

    print(f"Stocks in latest quarter: {len(df_latest)}")

    # Prepare features
    X_latest = df_latest[factors].dropna()
    y_latest = df_latest.loc[X_latest.index, TARGET_COL]

    valid_idx = y_latest.dropna().index
    X_latest = X_latest.loc[valid_idx]
    y_latest = y_latest.loc[valid_idx]

    print(f"Valid predictions: {len(X_latest)}")

    # Predict
    X_latest_const = sm.add_constant(X_latest)
    predictions = model.predict(X_latest_const)
    residuals = y_latest - predictions

    # Build results DataFrame
    results = df_latest.loc[valid_idx].copy()
    results['Predicted_LogMarketCap'] = predictions
    results['Residual_LogMarketCap'] = residuals
    results['MarketCap'] = np.exp(y_latest)

    # Rank by residual (lower = more undervalued)
    results['Undervaluation_Rank'] = residuals.rank()
    results['Undervaluation_Percentile'] = residuals.rank(pct=True) * 100

    # Sort by rank
    results = results.sort_values('Undervaluation_Rank')

    # Save results
    results_path = output_dir / f"predictions_{latest_quarter.replace('-', '')}.csv"
    results.to_csv(results_path)
    print(f"[OK] Saved predictions to {results_path}")

    # Select bottom 20% (most undervalued)
    threshold_pct = 20.0
    undervalued = results[results['Undervaluation_Percentile'] <= threshold_pct].copy()

    # Apply quality filters
    min_mcap = 10e9  # 10B TL
    undervalued = undervalued[undervalued['MarketCap'] >= min_mcap]

    # Limit to top 20 stocks
    max_stocks = 20
    if len(undervalued) > max_stocks:
        undervalued = undervalued.nsmallest(max_stocks, 'Residual_LogMarketCap')

    print(f"\nPortfolio: {len(undervalued)} stocks")
    print(f"Top 10:")
    for i, (idx, row) in enumerate(undervalued.head(10).iterrows(), 1):
        ticker = idx[0]  # First element of multiindex
        rank = int(row['Undervaluation_Rank'])
        pct = row['Undervaluation_Percentile']
        mcap = row['MarketCap'] / 1e9
        sector = row.get('SectorGroup', 'N/A')
        print(f"  {i:2d}. {ticker:<10} Rank={rank:<3} Percentile={pct:5.1f}%  MCap={mcap:6.2f}B  {sector}")

    # Save portfolio
    portfolio_output = undervalued[[
        'Undervaluation_Rank', 'Undervaluation_Percentile',
        'Residual_LogMarketCap', 'MarketCap', 'SectorGroup'
    ]].copy()

    portfolio_path = output_dir / f"portfolio_{latest_quarter.replace('-', '')}.csv"
    portfolio_output.to_csv(portfolio_path)
    print(f"[OK] Saved portfolio to {portfolio_path}")

    return portfolio_output


def main():
    """
    Main execution: train 3 versions and generate portfolios.
    """
    print("="*80)
    print("TRAINING MULTIPLE FACTOR MODEL VERSIONS")
    print("="*80)

    # Load data
    print("\nLoading panel data...")
    result = prepare_panel_data()

    # prepare_panel_data returns tuple - just take first element (df_panel)
    if isinstance(result, tuple):
        df_panel = result[0]
    else:
        df_panel = result

    if df_panel is None or df_panel.empty:
        print("[ERROR] Failed to load data")
        return

    print(f"Loaded {len(df_panel)} observations")
    print(f"Date range: {df_panel.index.get_level_values('QuarterDate').min()} to {df_panel.index.get_level_values('QuarterDate').max()}")

    # Define versions
    versions = [
        {
            'name': 'all_data',
            'train_start': '2016-01-01',
            'train_end': '2025-09-30',
            'description': 'All available data (2016-2025)'
        },
        {
            'name': 'last_5years',
            'train_start': '2020-01-01',
            'train_end': '2025-09-30',
            'description': 'Last 5 years (2020-2025)'
        },
        {
            'name': 'last_3years',
            'train_start': '2022-01-01',
            'train_end': '2025-09-30',
            'description': 'Last 3 years (2022-2025)'
        }
    ]

    results = {}

    # Train each version
    for version in versions:
        model_info = train_factor_model(
            df_panel,
            version['train_start'],
            version['train_end'],
            version['name']
        )

        if model_info is None:
            print(f"[ERROR] Failed to train {version['name']}")
            continue

        # Generate portfolio
        portfolio = predict_and_rank(df_panel, model_info, latest_quarter='2025-09-30')

        results[version['name']] = {
            'model_info': model_info,
            'portfolio': portfolio,
            'description': version['description']
        }

    # Print summary
    print(f"\n{'='*80}")
    print("SUMMARY OF ALL VERSIONS")
    print(f"{'='*80}\n")

    for version_name, result in results.items():
        info = result['model_info']
        portfolio = result['portfolio']
        desc = result['description']

        print(f"{version_name.upper()}:")
        print(f"  Description: {desc}")
        print(f"  R²: {info['r2']:.3f}")
        print(f"  IC: {info['ic']:.4f}")
        print(f"  Portfolio size: {len(portfolio)} stocks")
        print(f"  Output directory: {info['output_dir']}")
        print()

    print("[SUCCESS] All versions trained!")
    print("\nNext steps:")
    print("1. Review outputs in outputs_all_data/, outputs_last_5years/, outputs_last_3years/")
    print("2. Run dashboard builder for each version to compare performance")
    print("3. Use: python build_dashboards.py")


if __name__ == "__main__":
    main()
