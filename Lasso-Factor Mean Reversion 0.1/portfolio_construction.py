"""
Portfolio Construction from Lasso Mean Reversion Model

This script:
1. Loads the model results from recommended_model_outputs/
2. Identifies the most recent quarter
3. Finds the most undervalued stocks
4. Applies quality filters (positive equity, not distressed)
5. Diversifies across sectors
6. Outputs a portfolio of top stocks to invest in

Usage:
    python portfolio_construction.py
    python portfolio_construction.py --top-n 20 --max-sector-weight 0.30
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

# Configuration
ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "recommended_model_outputs"
OUTPUT_DIR = ROOT / "Lasso-Factor Mean Reversion 0.1" / "portfolio_outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

RESULTS_FILE = RESULTS_DIR / "ALL_lasso_results.csv"


def load_model_results():
    """Load the model results CSV."""
    if not RESULTS_FILE.exists():
        print(f"[ERROR] Results file not found: {RESULTS_FILE}")
        print("Please run recommended_model_pipeline.py first!")
        sys.exit(1)

    print(f"Loading results from: {RESULTS_FILE}")
    df = pd.read_csv(RESULTS_FILE)

    # Parse dates
    df['QuarterDate'] = pd.to_datetime(df['QuarterDate'])

    print(f"Loaded {len(df):,} observations")
    print(f"Date range: {df['QuarterDate'].min()} to {df['QuarterDate'].max()}")
    print(f"Unique tickers: {df['Ticker'].nunique()}")

    return df


def get_latest_quarter(df):
    """Get data for the most recent quarter."""
    latest_date = df['QuarterDate'].max()
    df_latest = df[df['QuarterDate'] == latest_date].copy()

    print(f"\n{'='*70}")
    print(f"LATEST QUARTER: {latest_date.date()}")
    print(f"{'='*70}")
    print(f"Stocks with data: {len(df_latest):,}")

    return df_latest, latest_date


def apply_quality_filters(df):
    """
    Apply quality filters to avoid value traps.

    Filters:
    1. Has valid mispricing signal (Residual_LogMarketCap not NaN)
    2. Has positive Total Equity (not bankrupt)
    3. Not overvalued (only select undervalued stocks)
    """
    print(f"\n{'='*70}")
    print("APPLYING QUALITY FILTERS")
    print(f"{'='*70}")

    initial_count = len(df)

    # Filter 1: Valid mispricing signal
    df = df[df['Residual_LogMarketCap'].notna()].copy()
    print(f"Filter 1 - Valid signal: {len(df):,} / {initial_count:,} stocks")

    # Filter 2: Only undervalued stocks (negative residual = undervalued)
    df = df[df['Residual_LogMarketCap'] < 0].copy()
    print(f"Filter 2 - Undervalued only: {len(df):,} stocks")

    # Filter 3: Positive Total Equity (if available)
    if 'TotalEquity' in df.columns:
        before = len(df)
        df = df[df['TotalEquity'] > 0].copy()
        print(f"Filter 3 - Positive equity: {len(df):,} / {before:,} stocks")

    # Filter 4: Has Overvaluation_pct
    if 'Overvaluation_pct' in df.columns:
        df = df[df['Overvaluation_pct'].notna()].copy()

    print(f"\nFinal candidates: {len(df):,} stocks")

    return df


def rank_stocks(df):
    """
    Rank stocks by mispricing signal.

    Lower residual (more negative) = more undervalued = higher rank
    """
    df = df.copy()

    # Rank by residual (ascending: most negative = rank 1)
    df['Rank'] = df['Residual_LogMarketCap'].rank(method='min')

    # Sort by rank
    df = df.sort_values('Rank').reset_index(drop=True)

    return df


def diversify_portfolio(df, top_n=20, max_sector_weight=0.30):
    """
    Build diversified portfolio with sector constraints.

    Args:
        df: Ranked stocks DataFrame
        top_n: Number of stocks to select
        max_sector_weight: Maximum weight per sector (0.30 = 30%)

    Returns:
        DataFrame with selected portfolio
    """
    print(f"\n{'='*70}")
    print(f"BUILDING DIVERSIFIED PORTFOLIO (Top {top_n})")
    print(f"{'='*70}")
    print(f"Max sector weight: {max_sector_weight*100:.0f}%")

    portfolio = []
    sector_counts = {}
    max_per_sector = int(top_n * max_sector_weight)

    print(f"Max stocks per sector: {max_per_sector}")

    for idx, row in df.iterrows():
        if len(portfolio) >= top_n:
            break

        sector = row.get('SectorGroup', 'Unknown')
        current_count = sector_counts.get(sector, 0)

        if current_count < max_per_sector:
            portfolio.append(row)
            sector_counts[sector] = current_count + 1

    portfolio_df = pd.DataFrame(portfolio)

    print(f"\nSelected {len(portfolio_df)} stocks")
    print("\nSector distribution:")
    if 'SectorGroup' in portfolio_df.columns:
        sector_dist = portfolio_df['SectorGroup'].value_counts()
        for sector, count in sector_dist.items():
            pct = count / len(portfolio_df) * 100
            print(f"  {sector}: {count} stocks ({pct:.1f}%)")

    return portfolio_df


def calculate_portfolio_weights(portfolio_df, method='equal'):
    """
    Calculate portfolio weights.

    Methods:
    - equal: Equal weight (1/N)
    - signal: Weight by absolute mispricing (more undervalued = higher weight)
    - inverse_volatility: Weight by inverse volatility (if available)
    """
    portfolio_df = portfolio_df.copy()

    if method == 'equal':
        portfolio_df['Weight'] = 1.0 / len(portfolio_df)

    elif method == 'signal':
        # Weight by absolute residual (more undervalued = higher weight)
        abs_residual = portfolio_df['Residual_LogMarketCap'].abs()
        portfolio_df['Weight'] = abs_residual / abs_residual.sum()

    else:
        raise ValueError(f"Unknown weighting method: {method}")

    return portfolio_df


def export_portfolio(portfolio_df, latest_date, output_dir=OUTPUT_DIR):
    """Export portfolio to CSV and Excel."""
    date_str = latest_date.strftime('%Y%m%d')

    # Select key columns
    export_cols = [
        'Ticker',
        'Rank',
        'Residual_LogMarketCap',
        'Overvaluation_pct',
        'Weight',
        'LogMarketCap',
        'Predicted_LogMarketCap',
    ]

    # Add sector if available
    if 'SectorGroup' in portfolio_df.columns:
        export_cols.append('SectorGroup')

    # Add forward returns if available
    for col in ['Return_1Q', 'Return_2Q', 'Return_4Q']:
        if col in portfolio_df.columns:
            export_cols.append(col)

    # Filter to available columns
    export_cols = [c for c in export_cols if c in portfolio_df.columns]

    df_export = portfolio_df[export_cols].copy()

    # CSV output
    csv_path = output_dir / f"portfolio_{date_str}.csv"
    df_export.to_csv(csv_path, index=False)
    print(f"\n[OK] Portfolio saved to: {csv_path}")

    # Excel output with formatting
    excel_path = output_dir / f"portfolio_{date_str}.xlsx"
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        df_export.to_excel(writer, sheet_name='Portfolio', index=False)

    print(f"[OK] Portfolio saved to: {excel_path}")

    return csv_path, excel_path


def print_portfolio_summary(portfolio_df):
    """Print summary statistics of the portfolio."""
    print(f"\n{'='*70}")
    print("PORTFOLIO SUMMARY")
    print(f"{'='*70}")

    print(f"\nNumber of stocks: {len(portfolio_df)}")

    if 'Overvaluation_pct' in portfolio_df.columns:
        print(f"\nUndervaluation statistics:")
        print(f"  Mean: {portfolio_df['Overvaluation_pct'].mean():.2f}%")
        print(f"  Median: {portfolio_df['Overvaluation_pct'].median():.2f}%")
        print(f"  Min (most undervalued): {portfolio_df['Overvaluation_pct'].min():.2f}%")
        print(f"  Max (least undervalued): {portfolio_df['Overvaluation_pct'].max():.2f}%")

    if 'Weight' in portfolio_df.columns:
        print(f"\nWeights:")
        print(f"  Min: {portfolio_df['Weight'].min():.4f} ({portfolio_df['Weight'].min()*100:.2f}%)")
        print(f"  Max: {portfolio_df['Weight'].max():.4f} ({portfolio_df['Weight'].max()*100:.2f}%)")
        print(f"  Sum: {portfolio_df['Weight'].sum():.4f}")

    print(f"\n{'='*70}")
    print("TOP 10 MOST UNDERVALUED STOCKS")
    print(f"{'='*70}")

    display_cols = ['Rank', 'Ticker', 'Overvaluation_pct', 'Weight']
    if 'SectorGroup' in portfolio_df.columns:
        display_cols.append('SectorGroup')

    display_cols = [c for c in display_cols if c in portfolio_df.columns]

    print(portfolio_df[display_cols].head(10).to_string(index=False))


def main(top_n=20, max_sector_weight=0.30, weighting='equal'):
    """Main portfolio construction pipeline."""
    print("="*70)
    print("PORTFOLIO CONSTRUCTION - LASSO MEAN REVERSION MODEL")
    print("="*70)

    # Load results
    df = load_model_results()

    # Get latest quarter
    df_latest, latest_date = get_latest_quarter(df)

    # Apply quality filters
    df_filtered = apply_quality_filters(df_latest)

    if len(df_filtered) < top_n:
        print(f"\n[WARN] Only {len(df_filtered)} stocks pass filters, less than requested {top_n}")
        print(f"[WARN] Will select all {len(df_filtered)} stocks")
        top_n = len(df_filtered)

    # Rank stocks
    df_ranked = rank_stocks(df_filtered)

    # Build diversified portfolio
    portfolio = diversify_portfolio(df_ranked, top_n=top_n, max_sector_weight=max_sector_weight)

    # Calculate weights
    portfolio = calculate_portfolio_weights(portfolio, method=weighting)

    # Print summary
    print_portfolio_summary(portfolio)

    # Export
    csv_path, excel_path = export_portfolio(portfolio, latest_date)

    print(f"\n{'='*70}")
    print("PORTFOLIO CONSTRUCTION COMPLETE")
    print(f"{'='*70}")
    print(f"Portfolio files saved to: {OUTPUT_DIR}")

    return portfolio


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Construct portfolio from mean reversion model")
    parser.add_argument("--top-n", type=int, default=20, help="Number of stocks to select (default: 20)")
    parser.add_argument("--max-sector-weight", type=float, default=0.30, help="Maximum weight per sector (default: 0.30)")
    parser.add_argument("--weighting", choices=['equal', 'signal'], default='equal', help="Weighting method (default: equal)")

    args = parser.parse_args()

    portfolio = main(
        top_n=args.top_n,
        max_sector_weight=args.max_sector_weight,
        weighting=args.weighting
    )
