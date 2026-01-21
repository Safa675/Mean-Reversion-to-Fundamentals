"""
Simple script to find most undervalued stocks from model results.
No extra dependencies needed - just reads CSV and prints results.

Usage:
    python find_undervalued.py
"""

import csv
from collections import defaultdict
from pathlib import Path

# Paths
ROOT = Path(__file__).resolve().parent.parent
RESULTS_FILE = ROOT / "recommended_model_outputs" / "ALL_lasso_results.csv"

def load_csv(filepath):
    """Load CSV file."""
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        return list(reader)

def parse_float(value):
    """Parse float safely."""
    try:
        return float(value) if value and value != 'nan' else None
    except:
        return None

def find_latest_quarter_undervalued(top_n=20):
    """Find most undervalued stocks in latest quarter."""
    print("="*70)
    print("FINDING MOST UNDERVALUED STOCKS")
    print("="*70)

    # Load data
    print(f"\nLoading: {RESULTS_FILE}")
    if not RESULTS_FILE.exists():
        print(f"[ERROR] File not found!")
        print(f"Please run recommended_model_pipeline.py first")
        return

    data = load_csv(RESULTS_FILE)
    print(f"Loaded {len(data):,} observations")

    # Find latest quarter
    dates = sorted(set(row['QuarterDate'] for row in data if row['QuarterDate']))
    latest_date = dates[-1]
    print(f"Latest quarter: {latest_date}")

    # Filter to latest quarter with valid signals
    latest_data = []
    for row in data:
        if row['QuarterDate'] == latest_date:
            residual = parse_float(row.get('Residual_LogMarketCap'))
            overval = parse_float(row.get('Overvaluation_pct'))

            if residual is not None and residual < 0:  # Undervalued only
                row['_residual'] = residual
                row['_overval'] = overval if overval is not None else residual * 100
                latest_data.append(row)

    print(f"Undervalued stocks in latest quarter: {len(latest_data)}")

    if len(latest_data) == 0:
        print("[ERROR] No undervalued stocks found!")
        return

    # Sort by residual (most negative = most undervalued)
    latest_data.sort(key=lambda x: x['_residual'])

    # Get top N
    top_stocks = latest_data[:top_n]

    # Count by sector
    sector_counts = defaultdict(int)
    for stock in top_stocks:
        sector = stock.get('SectorGroup', 'Unknown')
        sector_counts[sector] += 1

    # Print results
    print(f"\n{'='*70}")
    print(f"TOP {len(top_stocks)} MOST UNDERVALUED STOCKS ({latest_date})")
    print(f"{'='*70}\n")

    print(f"{'Rank':<6} {'Ticker':<10} {'Underval%':<12} {'Sector':<8} {'LogMCap':<10}")
    print("-" * 70)

    for i, stock in enumerate(top_stocks, 1):
        ticker = stock.get('Ticker', 'N/A')
        underval = stock['_overval']
        sector = stock.get('SectorGroup', 'N/A')
        log_mcap = parse_float(stock.get('LogMarketCap'))
        log_mcap_str = f"{log_mcap:.2f}" if log_mcap else "N/A"

        print(f"{i:<6} {ticker:<10} {underval:>10.2f}%  {sector:<8} {log_mcap_str:<10}")

    # Print sector distribution
    print(f"\n{'='*70}")
    print("SECTOR DISTRIBUTION")
    print(f"{'='*70}")
    for sector, count in sorted(sector_counts.items(), key=lambda x: -x[1]):
        pct = count / len(top_stocks) * 100
        print(f"  {sector:<10} {count:>3} stocks ({pct:>5.1f}%)")

    # Print statistics
    print(f"\n{'='*70}")
    print("UNDERVALUATION STATISTICS")
    print(f"{'='*70}")

    undervals = [s['_overval'] for s in top_stocks]
    mean_underval = sum(undervals) / len(undervals)
    min_underval = min(undervals)
    max_underval = max(undervals)

    print(f"  Mean undervaluation: {mean_underval:>8.2f}%")
    print(f"  Most undervalued:    {min_underval:>8.2f}%")
    print(f"  Least undervalued:   {max_underval:>8.2f}%")

    # Save to file
    output_file = Path(__file__).parent / "portfolio_outputs" / f"top_undervalued_{latest_date.replace('-', '')}.txt"
    output_file.parent.mkdir(exist_ok=True)

    with open(output_file, 'w') as f:
        f.write(f"TOP {len(top_stocks)} MOST UNDERVALUED STOCKS - {latest_date}\n")
        f.write("="*70 + "\n\n")

        f.write(f"{'Rank':<6} {'Ticker':<10} {'Underval%':<12} {'Sector':<8}\n")
        f.write("-" * 70 + "\n")

        for i, stock in enumerate(top_stocks, 1):
            ticker = stock.get('Ticker', 'N/A')
            underval = stock['_overval']
            sector = stock.get('SectorGroup', 'N/A')
            f.write(f"{i:<6} {ticker:<10} {underval:>10.2f}%  {sector:<8}\n")

        f.write("\n" + "="*70 + "\n")
        f.write("SECTOR DISTRIBUTION\n")
        f.write("="*70 + "\n")
        for sector, count in sorted(sector_counts.items(), key=lambda x: -x[1]):
            pct = count / len(top_stocks) * 100
            f.write(f"  {sector:<10} {count:>3} stocks ({pct:>5.1f}%)\n")

    print(f"\n[OK] Results saved to: {output_file}")

    return top_stocks

if __name__ == "__main__":
    find_latest_quarter_undervalued(top_n=20)
