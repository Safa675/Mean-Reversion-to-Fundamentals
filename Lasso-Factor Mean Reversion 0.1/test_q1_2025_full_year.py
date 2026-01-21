"""
Q1 2025 Full Year Backtest

Train: 2016-2024
Portfolio: Q1 2025 picks (2025-03-31)
Test period: April 2025 - January 2026 (~10 months)
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import statsmodels.api as sm

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))

from pooled_ols_residuals_bist import prepare_panel_data, TARGET_COL

print("="*80)
print("Q1 2025 FULL YEAR BACKTEST - FACTOR MODEL")
print("="*80)
print("\nTrain: 2016-2024")
print("Portfolio: Q1 2025 (picked on 2025-03-31)")
print("Test: April 2025 - January 2026 (~10 months)")
print("="*80)

# Load data
print("\nLoading data...")
result = prepare_panel_data()
df_panel = result[0] if isinstance(result, tuple) else result

print(f"Date range: {df_panel.index.get_level_values('QuarterDate').min()} to {df_panel.index.get_level_values('QuarterDate').max()}")

# Train on 2016-2024
train_end = '2024-12-31'
df_train = df_panel[df_panel.index.get_level_values('QuarterDate') <= train_end].copy()

print(f"\nTraining on: {df_train.index.get_level_values('QuarterDate').min()} to {df_train.index.get_level_values('QuarterDate').max()}")
print(f"Training observations: {len(df_train)}")

# Factors
factors = ['Factor_Scale', 'Factor_Leverage', 'Factor_RD', 'Factor_Profitability', 'Factor_Growth', 'Factor_CapitalEfficiency']
available_factors = [f for f in factors if f in df_train.columns]

X_train = df_train[available_factors].dropna()
y_train = df_train.loc[X_train.index, TARGET_COL]
valid_idx = y_train.dropna().index
X_train = X_train.loc[valid_idx]
y_train = y_train.loc[valid_idx]

print(f"Training sample: {len(X_train)}")

# Fit model
X_train_const = sm.add_constant(X_train)
model = sm.OLS(y_train, X_train_const).fit(cov_type='HC3')

print(f"\nModel R²: {model.rsquared:.3f}")
print(f"Coefficients:")
for name, coef in model.params.items():
    print(f"  {name}: {coef:.4f}")

# Predict on Q1 2025
print(f"\n{'='*80}")
print("PREDICTING ON Q1 2025 (2025-03-31)")
print(f"{'='*80}")

df_q1 = df_panel[df_panel.index.get_level_values('QuarterDate') == '2025-03-31'].copy()

X_q1 = df_q1[available_factors].dropna()
y_q1 = df_q1.loc[X_q1.index, TARGET_COL]
valid_idx_q1 = y_q1.dropna().index
X_q1 = X_q1.loc[valid_idx_q1]
y_q1 = y_q1.loc[valid_idx_q1]

print(f"Q1 2025 stocks: {len(X_q1)}")

X_q1_const = sm.add_constant(X_q1)
predictions = model.predict(X_q1_const)
residuals = y_q1 - predictions

# Rank
df_q1_results = df_q1.loc[valid_idx_q1].copy()
df_q1_results['Residual'] = residuals
df_q1_results['Rank'] = residuals.rank()
df_q1_results['Percentile'] = residuals.rank(pct=True) * 100
df_q1_results['MarketCap'] = np.exp(y_q1)

# Select bottom 20%
undervalued = df_q1_results[df_q1_results['Percentile'] <= 20].copy()

# Filter by market cap
min_mcap = 10e9
undervalued = undervalued[undervalued['MarketCap'] >= min_mcap]

# Limit to 20 stocks
if len(undervalued) > 20:
    undervalued = undervalued.nsmallest(20, 'Residual')

undervalued = undervalued.sort_values('Rank')

print(f"\nPortfolio: {len(undervalued)} stocks")
print(f"\n{'Rank':<6} {'Ticker':<10} {'Percentile':<12} {'MCap (B TL)':<15}")
print("-"*50)

for idx, row in undervalued.iterrows():
    ticker = idx[0]  # MultiIndex: (Ticker, QuarterDate)
    rank = int(row['Rank'])
    pct = row['Percentile']
    mcap = row['MarketCap'] / 1e9
    print(f"{rank:<6} {ticker:<10} {pct:10.1f}% {mcap:13.2f}")

# Extract tickers
tickers = [idx[0] for idx in undervalued.index]

# Save portfolio
output_dir = ROOT / 'backtest_q1_2025'
output_dir.mkdir(exist_ok=True)

picks_df = pd.DataFrame({
    'Ticker': tickers,
    'Weight': [1/len(tickers)] * len(tickers)
})

picks_path = output_dir / 'picks.csv'
picks_df.to_csv(picks_path, index=False)

print(f"\n[OK] Saved portfolio to: {picks_path}")

# Save model summary
summary_path = output_dir / 'model_summary.txt'
with open(summary_path, 'w') as f:
    f.write("Q1 2025 BACKTEST - FACTOR MODEL\n")
    f.write("="*80 + "\n\n")
    f.write(f"Training: 2016-2024\n")
    f.write(f"Portfolio: Q1 2025 (2025-03-31)\n")
    f.write(f"Test period: April 2025 - January 2026\n\n")
    f.write(f"Model R²: {model.rsquared:.3f}\n")
    f.write(f"Portfolio size: {len(tickers)} stocks\n\n")
    f.write("Portfolio:\n")
    for i, ticker in enumerate(tickers, 1):
        f.write(f"  {i}. {ticker}\n")
    f.write("\n" + "="*80 + "\n")
    f.write(str(model.summary()))

print(f"[OK] Saved model summary to: {summary_path}")

print(f"\n{'='*80}")
print("NEXT STEPS")
print(f"{'='*80}")
print(f"\n1. Dashboard is in: {output_dir}")
print("\n2. Copy dashboard script:")
print(f"   cp dashboard_bundle/build_dashboard_live.py {output_dir}/")
print("\n3. Modify start date to 2025-04-01 (Q1 earnings release)")
print("\n4. Run dashboard to see full year performance!")

print("\n[SUCCESS] Q1 2025 portfolio created!")
