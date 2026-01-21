"""
MODEL VALIDATION SCRIPT

Quick health checks for the all-fundamentals model.
Run this after retraining to ensure everything looks good.

Usage:
    python validate_model.py
"""

import pandas as pd
import numpy as np
from pathlib import Path

def validate_model():
    """Run validation checks on model outputs"""

    print("\n" + "="*80)
    print("MODEL VALIDATION CHECKS")
    print("="*80)

    output_dir = Path(__file__).parent / "outputs"

    # Check 1: Output files exist
    print("\n[1/8] Checking output files exist...")

    required_files = [
        "all_fundamentals_lasso_results.csv",
        "all_fundamentals_lasso_model_summary.txt",
        "all_fundamentals_lasso_selected_features.txt"
    ]

    missing_files = []
    for f in required_files:
        if not (output_dir / f).exists():
            missing_files.append(f)

    if missing_files:
        print(f"  ❌ FAIL: Missing files: {missing_files}")
        print("  → Run: python all_fundamentals_model.py --production")
        return False
    else:
        print("  ✅ PASS: All output files exist")

    # Check 2: Load results
    print("\n[2/8] Loading model results...")

    results_path = output_dir / "all_fundamentals_lasso_results.csv"
    try:
        df = pd.read_csv(results_path)
        print(f"  ✅ PASS: Loaded {len(df)} predictions")
    except Exception as e:
        print(f"  ❌ FAIL: Could not load results: {e}")
        return False

    # Check 3: Required columns exist
    print("\n[3/8] Checking required columns...")

    required_cols = ['Ticker', 'QuarterDate', 'LogMarketCap',
                     'Predicted_LogMarketCap', 'Residual_LogMarketCap']
    missing_cols = [c for c in required_cols if c not in df.columns]

    if missing_cols:
        print(f"  ❌ FAIL: Missing columns: {missing_cols}")
        return False
    else:
        print("  ✅ PASS: All required columns present")

    # Check 4: Data coverage
    print("\n[4/8] Checking data coverage...")

    df['QuarterDate'] = pd.to_datetime(df['QuarterDate'])
    date_range = f"{df['QuarterDate'].min()} to {df['QuarterDate'].max()}"
    n_quarters = df['QuarterDate'].nunique()
    n_stocks = df['Ticker'].nunique()

    print(f"  Date range: {date_range}")
    print(f"  Number of quarters: {n_quarters}")
    print(f"  Number of unique stocks: {n_stocks}")

    if n_quarters < 20:
        print(f"  ⚠️  WARN: Only {n_quarters} quarters (expected 30-40)")
    else:
        print("  ✅ PASS: Sufficient historical data")

    # Check 5: Latest quarter
    print("\n[5/8] Checking latest quarter...")

    latest_quarter = df['QuarterDate'].max()
    df_latest = df[df['QuarterDate'] == latest_quarter]

    print(f"  Latest quarter: {latest_quarter.strftime('%Y-%m-%d')}")
    print(f"  Stocks in latest quarter: {len(df_latest)}")

    if len(df_latest) < 50:
        print(f"  ⚠️  WARN: Only {len(df_latest)} stocks (expected 100-200)")
    else:
        print("  ✅ PASS: Sufficient coverage in latest quarter")

    # Check 6: Residual distribution
    print("\n[6/8] Checking residual distribution...")

    residuals = df_latest['Residual_LogMarketCap'].dropna()

    if len(residuals) == 0:
        print("  ❌ FAIL: No valid residuals in latest quarter")
        return False

    print(f"  Mean residual: {residuals.mean():.4f} (should be ~0)")
    print(f"  Std residual: {residuals.std():.4f}")
    print(f"  Min residual: {residuals.min():.4f}")
    print(f"  Max residual: {residuals.max():.4f}")

    if abs(residuals.mean()) > 0.5:
        print(f"  ⚠️  WARN: Mean residual far from 0 ({residuals.mean():.4f})")
    else:
        print("  ✅ PASS: Residuals centered around 0")

    # Check 7: Portfolio file
    print("\n[7/8] Checking portfolio file...")

    portfolio_files = list(output_dir.glob("portfolio_*.csv"))

    if not portfolio_files:
        print("  ⚠️  WARN: No portfolio file found")
        print("  → Run: python build_portfolio_correct.py")
    else:
        latest_portfolio = sorted(portfolio_files)[-1]
        df_port = pd.read_csv(latest_portfolio)

        print(f"  Portfolio file: {latest_portfolio.name}")
        print(f"  Number of stocks: {len(df_port)}")

        if 'SectorGroup' in df_port.columns:
            sector_dist = df_port['SectorGroup'].value_counts()
            max_sector_pct = 100 * sector_dist.max() / len(df_port)

            print(f"\n  Sector distribution:")
            for sector, count in sector_dist.items():
                pct = 100 * count / len(df_port)
                print(f"    {sector:<10} {count:>2} stocks ({pct:5.1f}%)")

            if max_sector_pct > 50:
                print(f"  ⚠️  WARN: Sector concentration > 50% ({max_sector_pct:.1f}%)")
            else:
                print("  ✅ PASS: Good sector diversification")
        else:
            print("  ✅ PASS: Portfolio file exists")

    # Check 8: Model summary
    print("\n[8/8] Checking model statistics...")

    summary_path = output_dir / "all_fundamentals_lasso_model_summary.txt"

    with open(summary_path, 'r') as f:
        summary_text = f.read()

    # Extract IC
    if 'training_ic:' in summary_text:
        ic_line = [l for l in summary_text.split('\n') if 'training_ic:' in l][0]

        # Parse IC value (e.g., "training_ic: IC=-0.0684, p=0.0000, N=16188")
        import re
        ic_match = re.search(r'IC=([-\d.]+)', ic_line)

        if ic_match:
            ic = float(ic_match.group(1))
            print(f"  Training IC: {ic:.4f}")

            if ic > 0:
                print("  ❌ FAIL: IC is positive (should be negative for mean reversion)")
                return False
            elif ic > -0.03:
                print(f"  ⚠️  WARN: IC is weak ({ic:.4f}, expected < -0.05)")
            else:
                print("  ✅ PASS: IC is negative and reasonable")
        else:
            print("  ⚠️  WARN: Could not parse IC from summary")

    # Extract R²
    if 'R-squared:' in summary_text:
        r2_line = [l for l in summary_text.split('\n') if 'R-squared:' in l][0]
        r2_match = re.search(r'R-squared:\s+([\d.]+)', r2_line)

        if r2_match:
            r2 = float(r2_match.group(1))
            print(f"  R-squared: {r2:.3f}")

            if r2 < 0.20:
                print(f"  ⚠️  WARN: R² is low ({r2:.3f}, expected > 0.25)")
            elif r2 > 0.60:
                print(f"  ⚠️  WARN: R² is high ({r2:.3f}, risk of overfitting)")
            else:
                print("  ✅ PASS: R² is reasonable (0.25-0.50)")

    # Final summary
    print("\n" + "="*80)
    print("VALIDATION COMPLETE")
    print("="*80)
    print("\n✅ Model appears healthy!")
    print("\nNext steps:")
    print("  1. Review portfolio: cat outputs/portfolio_*.csv")
    print("  2. Check model summary: cat outputs/all_fundamentals_lasso_model_summary.txt")
    print("  3. Implement portfolio with proper risk management")
    print("\nRemember:")
    print("  - Use RELATIVE rankings, not absolute predictions")
    print("  - Expect 2-5% annual alpha, not 99% gains")
    print("  - Diversify across 10-20 stocks")
    print("  - Rebalance quarterly")

    return True

if __name__ == "__main__":
    try:
        success = validate_model()
        exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ VALIDATION FAILED: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
