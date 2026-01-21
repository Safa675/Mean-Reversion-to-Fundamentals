"""
FIND UNDERVALUED STOCKS

Uses the model output to identify the most undervalued stocks
in the latest quarter.

Usage:
    python find_undervalued.py --input outputs/all_fundamentals_lasso_results.csv --top-n 20
"""

import argparse
import pandas as pd
from pathlib import Path

def find_undervalued_stocks(input_path, top_n=20, output_dir=None):
    """
    Find the most undervalued stocks from model results.

    Args:
        input_path: Path to results CSV file
        top_n: Number of top undervalued stocks to return
        output_dir: Directory to save output (default: same as input)

    Returns:
        DataFrame with top undervalued stocks
    """
    print(f"\nLoading results from: {input_path}")
    df = pd.read_csv(input_path)

    # Convert date column
    df['QuarterDate'] = pd.to_datetime(df['QuarterDate'])

    # Get latest quarter
    latest_quarter = df['QuarterDate'].max()
    print(f"Latest quarter: {latest_quarter.strftime('%Y-%m-%d')}")

    # Filter to latest quarter
    df_latest = df[df['QuarterDate'] == latest_quarter].copy()
    print(f"Stocks in latest quarter: {len(df_latest)}")

    # Remove stocks with missing residuals
    df_latest = df_latest.dropna(subset=['Residual_LogMarketCap'])
    print(f"Stocks with valid predictions: {len(df_latest)}")

    if df_latest.empty:
        print("[ERROR] No valid predictions in latest quarter")
        return None

    # Sort by residual (most undervalued = most negative residual)
    df_latest = df_latest.sort_values('Residual_LogMarketCap')

    # Get top N
    top_undervalued = df_latest.head(top_n).copy()

    # Add rank
    top_undervalued['Rank'] = range(1, len(top_undervalued) + 1)

    # Select relevant columns
    output_cols = [
        'Rank', 'Ticker', 'Residual_LogMarketCap', 'Overvaluation_pct',
        'Predicted_LogMarketCap', 'LogMarketCap', 'SectorGroup'
    ]

    # Only use columns that exist
    output_cols = [c for c in output_cols if c in top_undervalued.columns]
    top_undervalued = top_undervalued[output_cols]

    # Print results
    print(f"\n{'='*80}")
    print(f"TOP {top_n} MOST UNDERVALUED STOCKS - {latest_quarter.strftime('%Y-%m-%d')}")
    print(f"{'='*80}\n")

    print(f"{'Rank':<6} {'Ticker':<10} {'Residual':<12} {'Underval %':<12} {'Sector':<10}")
    print("-"*60)

    for _, row in top_undervalued.iterrows():
        rank = int(row['Rank'])
        ticker = row['Ticker']
        residual = row['Residual_LogMarketCap']
        underval = row['Overvaluation_pct']
        sector = row.get('SectorGroup', 'N/A')

        print(f"{rank:<6} {ticker:<10} {residual:+11.4f} {underval:+11.2f}% {sector:<10}")

    # Statistics
    print(f"\n{'='*80}")
    print("STATISTICS")
    print(f"{'='*80}")
    print(f"Mean undervaluation: {top_undervalued['Overvaluation_pct'].mean():+.2f}%")
    print(f"Median undervaluation: {top_undervalued['Overvaluation_pct'].median():+.2f}%")
    print(f"Most undervalued: {top_undervalued['Overvaluation_pct'].min():+.2f}%")
    print(f"Least undervalued: {top_undervalued['Overvaluation_pct'].max():+.2f}%")

    # Sector distribution
    if 'SectorGroup' in top_undervalued.columns:
        print(f"\n{'='*80}")
        print("SECTOR DISTRIBUTION")
        print(f"{'='*80}")
        sector_counts = top_undervalued['SectorGroup'].value_counts()
        for sector, count in sector_counts.items():
            pct = 100 * count / len(top_undervalued)
            print(f"{sector:<10} {count:>3} stocks ({pct:5.1f}%)")

    # Save output
    if output_dir is None:
        output_dir = Path(input_path).parent

    output_path = Path(output_dir) / f"top_undervalued_{latest_quarter.strftime('%Y%m%d')}.txt"

    with open(output_path, 'w') as f:
        f.write(f"TOP {top_n} MOST UNDERVALUED STOCKS - {latest_quarter.strftime('%Y-%m-%d')}\n")
        f.write(f"{'='*80}\n\n")

        f.write(f"{'Rank':<6} {'Ticker':<10} {'Residual':<12} {'Underval %':<12} {'Sector':<10}\n")
        f.write("-"*60 + "\n")

        for _, row in top_undervalued.iterrows():
            rank = int(row['Rank'])
            ticker = row['Ticker']
            residual = row['Residual_LogMarketCap']
            underval = row['Overvaluation_pct']
            sector = row.get('SectorGroup', 'N/A')

            f.write(f"{rank:<6} {ticker:<10} {residual:+11.4f} {underval:+11.2f}% {sector:<10}\n")

        f.write(f"\n{'='*80}\n")
        f.write("STATISTICS\n")
        f.write(f"{'='*80}\n")
        f.write(f"Mean undervaluation: {top_undervalued['Overvaluation_pct'].mean():+.2f}%\n")
        f.write(f"Median undervaluation: {top_undervalued['Overvaluation_pct'].median():+.2f}%\n")
        f.write(f"Most undervalued: {top_undervalued['Overvaluation_pct'].min():+.2f}%\n")
        f.write(f"Least undervalued: {top_undervalued['Overvaluation_pct'].max():+.2f}%\n")

        if 'SectorGroup' in top_undervalued.columns:
            f.write(f"\n{'='*80}\n")
            f.write("SECTOR DISTRIBUTION\n")
            f.write(f"{'='*80}\n")
            for sector, count in sector_counts.items():
                pct = 100 * count / len(top_undervalued)
                f.write(f"{sector:<10} {count:>3} stocks ({pct:5.1f}%)\n")

    print(f"\n[OK] Results saved to: {output_path}")

    # Also save as CSV
    csv_path = Path(output_dir) / f"top_undervalued_{latest_quarter.strftime('%Y%m%d')}.csv"
    top_undervalued.to_csv(csv_path, index=False)
    print(f"[OK] CSV saved to: {csv_path}")

    return top_undervalued


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Find most undervalued stocks from model results")
    parser.add_argument("--input", default="outputs/all_fundamentals_lasso_results.csv",
                       help="Path to model results CSV")
    parser.add_argument("--top-n", type=int, default=20,
                       help="Number of top undervalued stocks to return")
    parser.add_argument("--output-dir", default=None,
                       help="Output directory (default: same as input)")

    args = parser.parse_args()

    # Resolve input path relative to script directory
    script_dir = Path(__file__).parent
    input_path = args.input

    # If input is relative path, make it relative to script directory
    if not Path(input_path).is_absolute():
        input_path = script_dir / input_path

    # Check if file exists
    if not Path(input_path).exists():
        print(f"\n[ERROR] Input file not found: {input_path}")
        print("\nDid you run the model first?")
        print("Run this command first:")
        print("  python all_fundamentals_model.py")
        print("\nThen try again:")
        print("  python find_undervalued.py")
        exit(1)

    result = find_undervalued_stocks(
        input_path=str(input_path),
        top_n=args.top_n,
        output_dir=args.output_dir
    )

    if result is None:
        print("\n[ERROR] Failed to find undervalued stocks")
        exit(1)

    print("\n[SUCCESS] Completed successfully!")
