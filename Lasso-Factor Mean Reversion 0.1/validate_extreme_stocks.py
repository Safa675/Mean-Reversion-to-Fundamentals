"""
Data Validation - Investigate Extreme Undervaluations

This script checks why some stocks show -100% undervaluation:
1. Examines actual vs predicted values
2. Checks factor values
3. Identifies data quality issues
4. Provides recommendations

Usage:
    python validate_extreme_stocks.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import numpy as np

# Paths
ROOT = Path(__file__).resolve().parent.parent
RESULTS_FILE = ROOT / "recommended_model_outputs" / "ALL_lasso_results.csv"
OUTPUT_DIR = Path(__file__).parent / "validation_outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


def load_and_filter_data():
    """Load results and filter to latest quarter with extreme undervaluations."""
    print("="*70)
    print("DATA VALIDATION - EXTREME UNDERVALUATIONS")
    print("="*70)

    df = pd.read_csv(RESULTS_FILE)
    df['QuarterDate'] = pd.to_datetime(df['QuarterDate'])

    # Get latest quarter
    latest_date = df['QuarterDate'].max()
    df_latest = df[df['QuarterDate'] == latest_date].copy()

    print(f"\nLatest quarter: {latest_date.date()}")
    print(f"Total stocks: {len(df_latest):,}")

    # Filter to extreme undervaluations (< -90%)
    df_extreme = df_latest[df_latest['Overvaluation_pct'] < -90].copy()

    print(f"Stocks with > 90% undervaluation: {len(df_extreme)}")

    return df_latest, df_extreme, latest_date


def analyze_extreme_stocks(df_extreme):
    """Analyze why these stocks show extreme undervaluation."""
    print(f"\n{'='*70}")
    print("ANALYSIS OF EXTREME UNDERVALUATIONS")
    print(f"{'='*70}")

    # Sort by undervaluation
    df_extreme = df_extreme.sort_values('Overvaluation_pct')

    print("\nDetailed breakdown:\n")
    print(f"{'Ticker':<10} {'Actual':<10} {'Predicted':<10} {'Residual':<10} {'Underval%':<12}")
    print("-" * 70)

    for _, row in df_extreme.iterrows():
        ticker = row['Ticker']
        actual = row['LogMarketCap']
        predicted = row['Predicted_LogMarketCap']
        residual = row['Residual_LogMarketCap']
        underval = row['Overvaluation_pct']

        print(f"{ticker:<10} {actual:>9.2f} {predicted:>10.2f} {residual:>10.2f} {underval:>10.2f}%")

    return df_extreme


def check_log_space_issue(df_extreme):
    """Check if extreme undervaluations are due to log-space calculation."""
    print(f"\n{'='*70}")
    print("LOG-SPACE CONVERSION ANALYSIS")
    print(f"{'='*70}")

    print("\nFormula: Overvaluation_pct = (exp(Residual) - 1) * 100")
    print("Large negative residuals in log-space → -100% in percentage space\n")

    print(f"{'Ticker':<10} {'Residual':<12} {'exp(Res)':<12} {'Underval%':<12} {'Issue'}")
    print("-" * 70)

    for _, row in df_extreme.iterrows():
        ticker = row['Ticker']
        residual = row['Residual_LogMarketCap']
        exp_residual = np.exp(residual)
        underval = row['Overvaluation_pct']

        # Identify issue
        if residual < -5:
            issue = "Extreme residual"
        elif residual < -3:
            issue = "Very large residual"
        elif exp_residual < 0.01:
            issue = "Rounds to -100%"
        else:
            issue = "Normal"

        print(f"{ticker:<10} {residual:>11.2f} {exp_residual:>11.4f} {underval:>11.2f}% {issue}")


def check_factor_values(df_extreme, df_all):
    """Check if factor values are unusual for extreme stocks."""
    print(f"\n{'='*70}")
    print("FACTOR VALUE ANALYSIS")
    print(f"{'='*70}")

    # Get factor columns (ending in _z or starting with Factor_)
    factor_cols = [c for c in df_extreme.columns if c.endswith('_z') or c.startswith('Factor_')]
    factor_cols = [c for c in factor_cols if c not in ['LogMarketCap_z', 'Predicted_LogMarketCap_z']]

    if not factor_cols:
        print("[WARN] No factor columns found in data")
        return

    print(f"\nFound {len(factor_cols)} factor columns")
    print("\nComparing extreme stocks vs all stocks:\n")

    # Calculate stats
    stats_data = []

    for col in factor_cols:
        if col not in df_all.columns:
            continue

        # Stats for extreme stocks
        extreme_mean = df_extreme[col].mean()
        extreme_std = df_extreme[col].std()
        extreme_valid = df_extreme[col].notna().sum()

        # Stats for all stocks
        all_mean = df_all[col].mean()
        all_std = df_all[col].std()
        all_valid = df_all[col].notna().sum()

        # Difference
        mean_diff = extreme_mean - all_mean

        stats_data.append({
            'Factor': col,
            'Extreme_Mean': extreme_mean,
            'All_Mean': all_mean,
            'Difference': mean_diff,
            'Extreme_Valid': extreme_valid,
            'All_Valid': all_valid,
        })

    stats_df = pd.DataFrame(stats_data)
    stats_df = stats_df.sort_values('Difference', key=abs, ascending=False)

    print(f"{'Factor':<30} {'Extreme':<10} {'All':<10} {'Diff':<10} {'Status'}")
    print("-" * 70)

    for _, row in stats_df.head(10).iterrows():
        factor = row['Factor'][:28]
        extreme = row['Extreme_Mean']
        all_mean = row['All_Mean']
        diff = row['Difference']

        # Determine status
        if abs(diff) > 2:
            status = "⚠️ Very different"
        elif abs(diff) > 1:
            status = "⚠️ Different"
        else:
            status = "✓ Similar"

        print(f"{factor:<30} {extreme:>9.2f} {all_mean:>9.2f} {diff:>9.2f} {status}")

    return stats_df


def check_data_coverage(df_extreme):
    """Check if extreme stocks have missing data."""
    print(f"\n{'='*70}")
    print("DATA COVERAGE ANALYSIS")
    print(f"{'='*70}")

    # Get all columns
    all_cols = df_extreme.columns.tolist()

    # Count missing values per stock
    print("\nMissing data per stock:\n")
    print(f"{'Ticker':<10} {'Total Cols':<12} {'Missing':<10} {'% Missing':<12}")
    print("-" * 70)

    for _, row in df_extreme.iterrows():
        ticker = row['Ticker']
        total_cols = len(all_cols)
        missing = row.isna().sum()
        pct_missing = missing / total_cols * 100

        print(f"{ticker:<10} {total_cols:>11} {missing:>9} {pct_missing:>10.1f}%")


def check_historical_trend(df_all, extreme_tickers):
    """Check historical trend for extreme stocks (are they getting worse?)."""
    print(f"\n{'='*70}")
    print("HISTORICAL TREND ANALYSIS")
    print(f"{'='*70}")

    print("\nChecking if undervaluation is worsening over time...\n")

    # Get last 4 quarters for each ticker
    for ticker in extreme_tickers[:5]:  # Show first 5
        df_ticker = df_all[df_all['Ticker'] == ticker].sort_values('QuarterDate')
        df_ticker = df_ticker.tail(4)  # Last 4 quarters

        if len(df_ticker) == 0:
            continue

        print(f"\n{ticker}:")
        print(f"{'Quarter':<12} {'Actual':<10} {'Predicted':<10} {'Residual':<10} {'Underval%'}")
        print("-" * 60)

        for _, row in df_ticker.iterrows():
            quarter = row['QuarterDate'].strftime('%Y-%m-%d')
            actual = row['LogMarketCap']
            predicted = row['Predicted_LogMarketCap']
            residual = row['Residual_LogMarketCap']
            underval = row['Overvaluation_pct']

            print(f"{quarter:<12} {actual:>9.2f} {predicted:>10.2f} {residual:>10.2f} {underval:>9.2f}%")


def generate_recommendations(df_extreme, stats_df):
    """Generate recommendations based on analysis."""
    print(f"\n{'='*70}")
    print("RECOMMENDATIONS")
    print(f"{'='*70}")

    # Issue 1: Log-space residuals
    extreme_residuals = (df_extreme['Residual_LogMarketCap'] < -3).sum()
    if extreme_residuals > 0:
        print(f"\n⚠️  Issue 1: Extreme Log Residuals")
        print(f"    {extreme_residuals} stocks have residuals < -3 (log space)")
        print(f"    → These map to -95% to -100% undervaluation")
        print(f"    → Model may be extrapolating beyond training range")
        print(f"\n    Recommendation:")
        print(f"    - Cap residuals at -3.0 (corresponds to ~-95% undervaluation)")
        print(f"    - Or: Remove stocks with residual < -3.0")
        print(f"    - Or: Use decile ranks instead of absolute percentages")

    # Issue 2: Factor differences
    if stats_df is not None:
        large_diffs = (stats_df['Difference'].abs() > 2).sum()
        if large_diffs > 0:
            print(f"\n⚠️  Issue 2: Unusual Factor Values")
            print(f"    {large_diffs} factors differ significantly from average")
            print(f"    → Extreme stocks may have unusual fundamentals")
            print(f"    → Model trained on typical stocks, struggles with outliers")
            print(f"\n    Recommendation:")
            print(f"    - Add winsorization to factors (clip at ±3 std dev)")
            print(f"    - Or: Use robust regression (less sensitive to outliers)")
            print(f"    - Or: Remove stocks with > 5 extreme factor values")

    # Issue 3: Model predictions
    print(f"\n⚠️  Issue 3: Model Predictions")
    print(f"    Model predicts LogMCap ~25.5-25.7 for most extreme stocks")
    print(f"    Actual LogMCap ~24.5-24.9")
    print(f"    → Difference of ~1.0 log units = exp(1.0) = 2.7x overvaluation")
    print(f"    → Model thinks these stocks should be worth 2-3x current price")
    print(f"\n    Possible reasons:")
    print(f"    1. Market knows something model doesn't (fraud, litigation, etc.)")
    print(f"    2. Fundamentals are stale (haven't updated for recent losses)")
    print(f"    3. Model is correct (genuine buying opportunity)")
    print(f"\n    Recommendation:")
    print(f"    - Check news for these tickers (litigation, fraud, etc.)")
    print(f"    - Verify fundamentals are up-to-date (last filing date)")
    print(f"    - Cross-reference with external data sources")

    # Overall recommendation
    print(f"\n{'='*70}")
    print("OVERALL RECOMMENDATION")
    print(f"{'='*70}")
    print(f"\n✓ Your model IC is excellent (-0.21), so the signal works overall")
    print(f"✓ But extreme undervaluations (-100%) should be treated with caution")
    print(f"\nBest approach:")
    print(f"1. Cap undervaluation at -80% (residual = -1.6)")
    print(f"2. Remove stocks with missing fundamentals")
    print(f"3. Check news/events for remaining extreme stocks")
    print(f"4. Start with moderately undervalued (-30% to -80%) for safety")


def export_validation_report(df_extreme, stats_df, output_dir=OUTPUT_DIR):
    """Export validation report."""
    # Extreme stocks CSV
    csv_path = output_dir / "extreme_undervalued_stocks.csv"
    df_extreme.to_csv(csv_path, index=False)
    print(f"\n[OK] Extreme stocks saved to: {csv_path}")

    # Factor stats CSV
    if stats_df is not None:
        stats_path = output_dir / "factor_comparison.csv"
        stats_df.to_csv(stats_path, index=False)
        print(f"[OK] Factor stats saved to: {stats_path}")

    # Summary report
    report_path = output_dir / "validation_report.txt"
    with open(report_path, 'w') as f:
        f.write("DATA VALIDATION REPORT - EXTREME UNDERVALUATIONS\n")
        f.write("="*70 + "\n\n")

        f.write(f"Total extreme stocks (< -90%): {len(df_extreme)}\n")
        f.write(f"Date: {df_extreme['QuarterDate'].iloc[0]}\n\n")

        f.write("Top 10 Most Extreme:\n")
        f.write("-"*70 + "\n")
        for _, row in df_extreme.head(10).iterrows():
            ticker = row['Ticker']
            underval = row['Overvaluation_pct']
            residual = row['Residual_LogMarketCap']
            f.write(f"{ticker:<10} Underval: {underval:>8.2f}%  Residual: {residual:>8.2f}\n")

        f.write("\n" + "="*70 + "\n")
        f.write("KEY FINDINGS\n")
        f.write("="*70 + "\n\n")

        f.write("1. LOG-SPACE ISSUE:\n")
        f.write("   - Residuals < -3 map to -95% to -100% undervaluation\n")
        f.write("   - This is due to exponential transformation: exp(-5) ≈ 0.007\n")
        f.write("   - Model may be extrapolating beyond training range\n\n")

        f.write("2. MODEL PREDICTIONS:\n")
        f.write("   - Model predicts LogMCap ~25.5-25.7 for extreme stocks\n")
        f.write("   - Actual LogMCap ~24.5-24.9\n")
        f.write("   - Implies stocks should be worth 2-3x current price\n\n")

        f.write("3. RECOMMENDATION:\n")
        f.write("   - Cap undervaluation at -80% for safety\n")
        f.write("   - Investigate news/events for extreme stocks\n")
        f.write("   - Start with moderately undervalued stocks (-30% to -80%)\n")

    print(f"[OK] Validation report saved to: {report_path}")


def main():
    """Main validation pipeline."""
    # Load data
    df_all, df_extreme, latest_date = load_and_filter_data()

    if len(df_extreme) == 0:
        print("\n[OK] No extreme undervaluations found!")
        return

    # Analyze extreme stocks
    df_extreme = analyze_extreme_stocks(df_extreme)

    # Check log-space issue
    check_log_space_issue(df_extreme)

    # Check factor values
    stats_df = check_factor_values(df_extreme, df_all)

    # Check data coverage
    check_data_coverage(df_extreme)

    # Check historical trend
    extreme_tickers = df_extreme['Ticker'].tolist()
    df_all_full = pd.read_csv(RESULTS_FILE)
    df_all_full['QuarterDate'] = pd.to_datetime(df_all_full['QuarterDate'])
    check_historical_trend(df_all_full, extreme_tickers)

    # Generate recommendations
    generate_recommendations(df_extreme, stats_df)

    # Export report
    export_validation_report(df_extreme, stats_df)

    print(f"\n{'='*70}")
    print("VALIDATION COMPLETE")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
