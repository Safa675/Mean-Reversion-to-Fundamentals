"""
FIX EXTRAPOLATION ISSUE

The model is making extreme predictions (trillions of times higher than actual)
because 2025 fundamentals are outside the training range (2016-2019).

This script applies post-hoc fixes:
1. Clip predictions to reasonable range based on training data
2. Recompute residuals and overvaluation %
3. Save corrected results

Usage:
    python fix_extrapolation.py --input outputs/all_fundamentals_lasso_results.csv
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path

def fix_extrapolation(input_path, output_path=None, train_end='2020-01-01'):
    """
    Fix extrapolation issues by clipping predictions.

    Args:
        input_path: Path to model results CSV
        output_path: Path to save corrected results (default: same as input with _fixed suffix)
        train_end: End of training period to determine clipping bounds
    """
    print(f"\nLoading results from: {input_path}")
    df = pd.read_csv(input_path)
    df['QuarterDate'] = pd.to_datetime(df['QuarterDate'])

    # Identify training period
    df_train = df[df['QuarterDate'] < pd.Timestamp(train_end)]

    print(f"\nTraining period statistics:")
    print(f"  Observations: {len(df_train):,}")
    print(f"  LogMarketCap range: [{df_train['LogMarketCap'].min():.2f}, {df_train['LogMarketCap'].max():.2f}]")
    print(f"  LogMarketCap mean: {df_train['LogMarketCap'].mean():.2f}")
    print(f"  LogMarketCap std: {df_train['LogMarketCap'].std():.2f}")

    # Compute reasonable bounds (mean ± 3 std)
    mean_log = df_train['LogMarketCap'].mean()
    std_log = df_train['LogMarketCap'].std()

    lower_bound = mean_log - 3 * std_log
    upper_bound = mean_log + 3 * std_log

    print(f"\nClipping bounds (mean ± 3 std):")
    print(f"  Lower: {lower_bound:.2f}")
    print(f"  Upper: {upper_bound:.2f}")

    # Clip predictions
    df['Predicted_LogMarketCap_Original'] = df['Predicted_LogMarketCap']
    df['Predicted_LogMarketCap'] = df['Predicted_LogMarketCap'].clip(lower=lower_bound, upper=upper_bound)

    # Count clipped values
    clipped = (df['Predicted_LogMarketCap_Original'] != df['Predicted_LogMarketCap']).sum()
    print(f"\nClipped {clipped:,} predictions ({100*clipped/len(df):.1f}%)")

    # Recompute residuals and overvaluation
    df['Residual_LogMarketCap'] = df['LogMarketCap'] - df['Predicted_LogMarketCap']
    df['Overvaluation_pct'] = (np.exp(df['Residual_LogMarketCap']) - 1) * 100

    # Show corrected statistics for latest quarter
    latest_quarter = df['QuarterDate'].max()
    df_latest = df[df['QuarterDate'] == latest_quarter].copy()

    print(f"\n{'='*80}")
    print(f"CORRECTED RESULTS FOR {latest_quarter.strftime('%Y-%m-%d')}")
    print(f"{'='*80}")

    print(f"\nOvervaluation statistics:")
    print(f"  Mean: {df_latest['Overvaluation_pct'].mean():+.2f}%")
    print(f"  Median: {df_latest['Overvaluation_pct'].median():+.2f}%")
    print(f"  Min: {df_latest['Overvaluation_pct'].min():+.2f}%")
    print(f"  Max: {df_latest['Overvaluation_pct'].max():+.2f}%")

    # Top 10 undervalued (most negative overvaluation)
    top_undervalued = df_latest.nsmallest(10, 'Overvaluation_pct')[
        ['Ticker', 'Overvaluation_pct', 'Residual_LogMarketCap', 'SectorGroup']
    ]

    print(f"\nTop 10 Most Undervalued:")
    print(f"{'Rank':<6} {'Ticker':<10} {'Overval %':<12} {'Residual':<12} {'Sector':<10}")
    print("-"*60)
    for i, row in enumerate(top_undervalued.itertuples(), 1):
        print(f"{i:<6} {row.Ticker:<10} {row.Overvaluation_pct:+11.2f}% {row.Residual_LogMarketCap:+11.4f} {row.SectorGroup:<10}")

    # Top 10 overvalued (most positive overvaluation)
    top_overvalued = df_latest.nlargest(10, 'Overvaluation_pct')[
        ['Ticker', 'Overvaluation_pct', 'Residual_LogMarketCap', 'SectorGroup']
    ]

    print(f"\nTop 10 Most Overvalued:")
    print(f"{'Rank':<6} {'Ticker':<10} {'Overval %':<12} {'Residual':<12} {'Sector':<10}")
    print("-"*60)
    for i, row in enumerate(top_overvalued.itertuples(), 1):
        print(f"{i:<6} {row.Ticker:<10} {row.Overvaluation_pct:+11.2f}% {row.Residual_LogMarketCap:+11.4f} {row.SectorGroup:<10}")

    # Save corrected results
    if output_path is None:
        input_stem = Path(input_path).stem
        output_path = Path(input_path).parent / f"{input_stem}_fixed.csv"

    df.to_csv(output_path, index=False)
    print(f"\n[OK] Corrected results saved to: {output_path}")

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fix extrapolation issues in model results")
    parser.add_argument("--input", default="outputs/all_fundamentals_lasso_results.csv",
                       help="Path to model results CSV")
    parser.add_argument("--output", default=None,
                       help="Output path (default: input path with _fixed suffix)")
    parser.add_argument("--train-end", default="2020-01-01",
                       help="End of training period (default: 2020-01-01)")

    args = parser.parse_args()

    # Resolve paths relative to script directory
    script_dir = Path(__file__).parent
    input_path = args.input if Path(args.input).is_absolute() else script_dir / args.input

    if not input_path.exists():
        print(f"\n[ERROR] Input file not found: {input_path}")
        exit(1)

    result = fix_extrapolation(
        input_path=str(input_path),
        output_path=args.output,
        train_end=args.train_end
    )

    print("\n[SUCCESS] Extrapolation issues fixed!")
    print("\nNow you can use:")
    print(f"  python find_undervalued.py --input {Path(args.output or input_path).parent / (Path(input_path).stem + '_fixed.csv')}")
