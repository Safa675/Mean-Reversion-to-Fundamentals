"""
Full Year 2025 Backtest

Train both models on data through 2024 Q4 (2024-12-31)
Test on Q1 2025 portfolio (2025-03-31 picks)
Measure performance: March 2025 to January 2026 (~10 months)

This tests if models can predict 1-year forward performance.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "BIST"))

print("="*80)
print("FULL YEAR 2025 BACKTEST")
print("="*80)
print("\nTrain: 2016-2024")
print("Test: Q1 2025 portfolio (picked on 2025-03-31)")
print("Performance period: April 2025 - January 2026 (~10 months)")
print("="*80)

# Import data preparation
try:
    from pooled_ols_residuals_bist import prepare_panel_data, TARGET_COL
    print("\n[OK] Imported data preparation")
except ImportError as e:
    print(f"\n[ERROR] Could not import: {e}")
    exit(1)

# Load data
print("\nLoading panel data...")
result = prepare_panel_data()
if isinstance(result, tuple):
    df_panel = result[0]
else:
    df_panel = result

print(f"Loaded {len(df_panel)} observations")
print(f"Full date range: {df_panel.index.get_level_values('QuarterDate').min()} to {df_panel.index.get_level_values('QuarterDate').max()}")

# ============================================================================
# TRAIN FACTOR MODEL (2016-2024)
# ============================================================================

print(f"\n{'='*80}")
print("TRAINING FACTOR MODEL (2016-2024)")
print(f"{'='*80}")

train_end_factor = '2024-12-31'
df_train_factor = df_panel[
    df_panel.index.get_level_values('QuarterDate') <= train_end_factor
].copy()

print(f"Training data: {len(df_train_factor)} observations")
print(f"Date range: {df_train_factor.index.get_level_values('QuarterDate').min()} to {df_train_factor.index.get_level_values('QuarterDate').max()}")

# Factor features
factor_features = [
    'Factor_Scale',
    'Factor_Leverage',
    'Factor_RD',
    'Factor_Profitability',
    'Factor_Growth',
    'Factor_CapitalEfficiency'
]

available_factors = [f for f in factor_features if f in df_train_factor.columns]
print(f"Available factors: {available_factors}")

# Prepare for regression
X_train_factor = df_train_factor[available_factors].dropna()
y_train_factor = df_train_factor.loc[X_train_factor.index, TARGET_COL]
valid_idx = y_train_factor.dropna().index
X_train_factor = X_train_factor.loc[valid_idx]
y_train_factor = y_train_factor.loc[valid_idx]

print(f"Training sample: {len(X_train_factor)}")

# Fit model
X_train_factor_const = sm.add_constant(X_train_factor)
model_factor = sm.OLS(y_train_factor, X_train_factor_const).fit(cov_type='HC3')

print(f"\nFactor Model:")
print(f"  R²: {model_factor.rsquared:.3f}")
print(f"  Coefficients:")
for feature, coef in model_factor.params.items():
    print(f"    {feature}: {coef:.4f}")

# Predict on Q1 2025
print(f"\nPredicting on Q1 2025 (2025-03-31)...")
df_q1_2025 = df_panel[
    df_panel.index.get_level_values('QuarterDate') == '2025-03-31'
].copy()

X_q1_factor = df_q1_2025[available_factors].dropna()
y_q1_factor = df_q1_2025.loc[X_q1_factor.index, TARGET_COL]
valid_idx_q1 = y_q1_factor.dropna().index
X_q1_factor = X_q1_factor.loc[valid_idx_q1]
y_q1_factor = y_q1_factor.loc[valid_idx_q1]

print(f"Q1 2025 stocks: {len(X_q1_factor)}")

X_q1_factor_const = sm.add_constant(X_q1_factor)
predictions_factor = model_factor.predict(X_q1_factor_const)
residuals_factor = y_q1_factor - predictions_factor

# Rank and select
results_factor = df_q1_2025.loc[valid_idx_q1].copy()
results_factor['Residual'] = residuals_factor
results_factor['Rank'] = residuals_factor.rank()
results_factor['Percentile'] = residuals_factor.rank(pct=True) * 100

# Select bottom 20%
undervalued_factor = results_factor[results_factor['Percentile'] <= 20].copy()

# Apply filters
min_mcap = 10e9
undervalued_factor['MarketCap'] = np.exp(y_q1_factor.loc[undervalued_factor.index])
undervalued_factor = undervalued_factor[undervalued_factor['MarketCap'] >= min_mcap]

# Limit to 20 stocks
if len(undervalued_factor) > 20:
    undervalued_factor = undervalued_factor.nsmallest(20, 'Residual')

print(f"\nFactor Model Portfolio (Q1 2025): {len(undervalued_factor)} stocks")
tickers_factor = [idx[0] for idx in undervalued_factor.index]
print("Tickers:", tickers_factor[:10], "..." if len(tickers_factor) > 10 else "")

# ============================================================================
# TRAIN ALL-FUNDAMENTALS MODEL (2016-2024)
# ============================================================================

print(f"\n{'='*80}")
print("TRAINING ALL-FUNDAMENTALS MODEL (2016-2024)")
print(f"{'='*80}")

# For all-fundamentals, we need to import from the Lasso-ElasticNet folder
sys.path.insert(0, str(ROOT / "BIST" / "Lasso-ElasticNet All-Fundamentals Model"))

try:
    from all_fundamentals_model import run_all_fundamentals_model
    print("[OK] Imported all-fundamentals model")

    # Train model
    print("\nTraining all-fundamentals model...")
    print("(This may take a few minutes...)")

    model_all_fund = run_all_fundamentals_model(
        method='lasso',
        train_end='2024-12-31',
        val_end=None,  # No validation split
        max_features=20,
        min_coverage_pct=10.0,
        corr_threshold=0.95
    )

    if model_all_fund is None:
        print("[ERROR] All-fundamentals model training failed")
        exit(1)

    print(f"\nAll-Fundamentals Model:")
    print(f"  Selected features: {len(model_all_fund['selected_features'])}")
    print(f"  R²: {model_all_fund['model'].rsquared:.3f}")

    # Predict on Q1 2025
    print(f"\nPredicting on Q1 2025...")

    # Get Q1 2025 data
    df_q1_all = df_panel[
        df_panel.index.get_level_values('QuarterDate') == '2025-03-31'
    ].copy()

    selected_features = model_all_fund['selected_features']
    X_q1_all = df_q1_all[selected_features].dropna()
    y_q1_all = df_q1_all.loc[X_q1_all.index, TARGET_COL]
    valid_idx_q1_all = y_q1_all.dropna().index
    X_q1_all = X_q1_all.loc[valid_idx_q1_all]
    y_q1_all = y_q1_all.loc[valid_idx_q1_all]

    # Impute missing values
    training_medians = model_all_fund['training_medians']
    for col in selected_features:
        if col in X_q1_all.columns and X_q1_all[col].isna().any():
            median_val = training_medians.get(col, X_q1_all[col].median())
            X_q1_all[col] = X_q1_all[col].fillna(median_val)

    print(f"Q1 2025 stocks: {len(X_q1_all)}")

    X_q1_all_const = sm.add_constant(X_q1_all)
    predictions_all = model_all_fund['model'].predict(X_q1_all_const)
    residuals_all = y_q1_all - predictions_all

    # Rank and select
    results_all = df_q1_all.loc[valid_idx_q1_all].copy()
    results_all['Residual'] = residuals_all
    results_all['Rank'] = residuals_all.rank()
    results_all['Percentile'] = residuals_all.rank(pct=True) * 100

    # Select bottom 20%
    undervalued_all = results_all[results_all['Percentile'] <= 20].copy()

    # Apply filters
    undervalued_all['MarketCap'] = np.exp(y_q1_all.loc[undervalued_all.index])
    undervalued_all = undervalued_all[undervalued_all['MarketCap'] >= min_mcap]

    # Limit to 20 stocks
    if len(undervalued_all) > 20:
        undervalued_all = undervalued_all.nsmallest(20, 'Residual')

    print(f"\nAll-Fundamentals Portfolio (Q1 2025): {len(undervalued_all)} stocks")
    tickers_all = [idx[0] for idx in undervalued_all.index]
    print("Tickers:", tickers_all[:10], "..." if len(tickers_all) > 10 else "")

except Exception as e:
    print(f"[ERROR] All-fundamentals model failed: {e}")
    import traceback
    traceback.print_exc()
    tickers_all = []

# ============================================================================
# SAVE PORTFOLIOS FOR DASHBOARD
# ============================================================================

print(f"\n{'='*80}")
print("SAVING PORTFOLIOS")
print(f"{'='*80}")

# Factor model portfolio
factor_output_dir = ROOT / "BIST" / "Lasso-Factor Mean Reversion 0.1" / "backtest_2025_q1"
factor_output_dir.mkdir(exist_ok=True)

factor_picks = pd.DataFrame({
    'Ticker': tickers_factor,
    'Weight': [1/len(tickers_factor)] * len(tickers_factor)
})
factor_picks.to_csv(factor_output_dir / 'picks.csv', index=False)
print(f"[OK] Saved factor portfolio: {factor_output_dir / 'picks.csv'}")

# All-fundamentals portfolio
if tickers_all:
    all_fund_output_dir = ROOT / "BIST" / "Lasso-ElasticNet All-Fundamentals Model" / "backtest_2025_q1"
    all_fund_output_dir.mkdir(exist_ok=True)

    all_fund_picks = pd.DataFrame({
        'Ticker': tickers_all,
        'Weight': [1/len(tickers_all)] * len(tickers_all)
    })
    all_fund_picks.to_csv(all_fund_output_dir / 'picks.csv', index=False)
    print(f"[OK] Saved all-fundamentals portfolio: {all_fund_output_dir / 'picks.csv'}")

# ============================================================================
# SUMMARY
# ============================================================================

print(f"\n{'='*80}")
print("SUMMARY")
print(f"{'='*80}")

print(f"\nFactor Model:")
print(f"  Training period: 2016-2024")
print(f"  R²: {model_factor.rsquared:.3f}")
print(f"  Q1 2025 portfolio: {len(tickers_factor)} stocks")

if tickers_all:
    print(f"\nAll-Fundamentals Model:")
    print(f"  Training period: 2016-2024")
    print(f"  R²: {model_all_fund['model'].rsquared:.3f}")
    print(f"  Q1 2025 portfolio: {len(tickers_all)} stocks")

print(f"\nPortfolio overlap:")
overlap = set(tickers_factor) & set(tickers_all) if tickers_all else set()
print(f"  Common stocks: {len(overlap)}")
if overlap:
    print(f"  Tickers: {list(overlap)}")

print(f"\n{'='*80}")
print("NEXT STEPS")
print(f"{'='*80}")
print("\n1. Copy dashboard builder to backtest folders")
print("2. Run dashboards from April 2025 start date")
print("3. Compare 10-month performance vs BIST100")
print("\nRun:")
print("  python build_backtest_dashboards.py")
