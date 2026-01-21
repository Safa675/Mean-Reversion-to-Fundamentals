"""
BUILD PORTFOLIO THE CORRECT WAY

Use the model for RANKING (which works) not absolute predictions (which don't).

This script:
1. Ranks stocks by residual (within each quarter)
2. Selects bottom 20% (most undervalued relative to others)
3. Applies quality filters
4. Builds diversified portfolio

Usage:
    python build_portfolio_correct.py --input outputs/all_fundamentals_lasso_results.csv
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path

def build_portfolio(input_path, top_pct=20, min_mcap=1e9, max_stocks=20):
    """
    Build portfolio using RELATIVE rankings, not absolute predictions.

    Args:
        input_path: Path to model results CSV
        top_pct: Top percentage to select (20 = bottom 20% = most undervalued)
        min_mcap: Minimum market cap filter
        max_stocks: Maximum stocks in portfolio

    Returns:
        DataFrame with portfolio
    """
    print(f"\nLoading results from: {input_path}")
    df = pd.read_csv(input_path)
    df['QuarterDate'] = pd.to_datetime(df['QuarterDate'])

    # Get latest quarter
    latest_quarter = df['QuarterDate'].max()
    print(f"Latest quarter: {latest_quarter.strftime('%Y-%m-%d')}")

    df_latest = df[df['QuarterDate'] == latest_quarter].copy()
    print(f"Stocks in latest quarter: {len(df_latest)}")

    # Remove stocks with missing residuals
    df_latest = df_latest.dropna(subset=['Residual_LogMarketCap', 'MarketCap'])
    print(f"Stocks with valid predictions: {len(df_latest)}")

    if df_latest.empty:
        print("[ERROR] No valid predictions")
        return None

    # Rank by residual (lower = more undervalued)
    df_latest['Undervaluation_Rank'] = df_latest['Residual_LogMarketCap'].rank()
    df_latest['Undervaluation_Percentile'] = df_latest['Residual_LogMarketCap'].rank(pct=True) * 100

    print(f"\n{'='*80}")
    print("STEP 1: RANK BY RELATIVE UNDERVALUATION")
    print(f"{'='*80}")
    print(f"Using residuals to rank stocks (not absolute 'undervaluation %')")
    print(f"Lower residual = more undervalued RELATIVE to peers")

    # Select bottom percentile
    threshold_pct = top_pct
    undervalued = df_latest[df_latest['Undervaluation_Percentile'] <= threshold_pct].copy()

    print(f"\n  Selected bottom {threshold_pct}% by residual: {len(undervalued)} stocks")

    if undervalued.empty:
        print("[ERROR] No stocks passed initial filter")
        return None

    # Apply quality filters
    print(f"\n{'='*80}")
    print("STEP 2: APPLY QUALITY FILTERS")
    print(f"{'='*80}")

    initial_count = len(undervalued)

    # Filter 1: Minimum market cap (liquidity)
    undervalued = undervalued[undervalued['MarketCap'] >= min_mcap]
    print(f"  After min market cap filter (>= {min_mcap/1e9:.1f}B TL): {len(undervalued)} stocks")

    # Filter 2: Positive net income (if available)
    if 'NetIncome' in undervalued.columns:
        profitable = undervalued['NetIncome'] > 0
        if profitable.sum() > 0:
            undervalued = undervalued[profitable]
            print(f"  After profitability filter (NetIncome > 0): {len(undervalued)} stocks")

    # Filter 3: Reasonable leverage (if available)
    if 'NetDebt' in undervalued.columns and 'TotalEquity' in undervalued.columns:
        equity_positive = undervalued['TotalEquity'] > 0
        undervalued = undervalued[equity_positive]

        leverage = undervalued['NetDebt'] / undervalued['TotalEquity']
        reasonable_leverage = leverage < 3  # Debt/Equity < 3
        if reasonable_leverage.sum() > 0:
            undervalued = undervalued[reasonable_leverage]
            print(f"  After leverage filter (Debt/Equity < 3): {len(undervalued)} stocks")

    if undervalued.empty:
        print("\n[WARN] No stocks passed quality filters. Relaxing filters...")
        # Retry with just market cap filter
        undervalued = df_latest[df_latest['Undervaluation_Percentile'] <= threshold_pct]
        undervalued = undervalued[undervalued['MarketCap'] >= min_mcap/2]  # Relax to 500M

    # Limit to max_stocks
    if len(undervalued) > max_stocks:
        undervalued = undervalued.nsmallest(max_stocks, 'Residual_LogMarketCap')
        print(f"\n  Limited to top {max_stocks} stocks by residual")

    # Sort by rank
    undervalued = undervalued.sort_values('Undervaluation_Rank')

    print(f"\n{'='*80}")
    print(f"FINAL PORTFOLIO: {len(undervalued)} STOCKS")
    print(f"{'='*80}\n")

    print(f"{'Rank':<6} {'Ticker':<10} {'Percentile':<12} {'MCap (B TL)':<15} {'Sector':<10}")
    print("-"*60)

    for _, row in undervalued.iterrows():
        rank = int(row['Undervaluation_Rank'])
        pct = row['Undervaluation_Percentile']
        mcap = row['MarketCap'] / 1e9
        ticker = row['Ticker']
        sector = row.get('SectorGroup', 'N/A')

        print(f"{rank:<6} {ticker:<10} {pct:10.1f}% {mcap:13.2f} {sector:<10}")

    # Statistics
    print(f"\n{'='*80}")
    print("PORTFOLIO STATISTICS")
    print(f"{'='*80}")
    print(f"Number of stocks: {len(undervalued)}")
    print(f"Total market cap: {undervalued['MarketCap'].sum()/1e9:.2f}B TL")
    print(f"Average market cap: {undervalued['MarketCap'].mean()/1e9:.2f}B TL")
    print(f"Median market cap: {undervalued['MarketCap'].median()/1e9:.2f}B TL")

    # Sector distribution
    if 'SectorGroup' in undervalued.columns:
        print(f"\nSector distribution:")
        sector_counts = undervalued['SectorGroup'].value_counts()
        for sector, count in sector_counts.items():
            pct = 100 * count / len(undervalued)
            print(f"  {sector:<10} {count:>3} stocks ({pct:5.1f}%)")

    # Expected statistics
    print(f"\n{'='*80}")
    print("INTERPRETATION")
    print(f"{'='*80}")
    print("These stocks are in the BOTTOM percentile of valuations")
    print("(most undervalued RELATIVE to peers with similar fundamentals)")
    print("")
    print("Expected behavior:")
    print("  - May outperform the index by 2-5% annually (based on IC)")
    print("  - High volatility (value stocks can stay cheap for long)")
    print("  - Not guaranteed - IC shows tendency, not certainty")
    print("")
    print("Risk factors:")
    print("  - Value traps (cheap for a reason)")
    print("  - Illiquidity (hard to trade)")
    print("  - Regime shift (mean reversion may not work in all periods)")

    # Save portfolio
    output_dir = Path(input_path).parent
    output_path = output_dir / f"portfolio_{latest_quarter.strftime('%Y%m%d')}.csv"

    portfolio_output = undervalued[[
        'Ticker', 'Undervaluation_Rank', 'Undervaluation_Percentile',
        'Residual_LogMarketCap', 'MarketCap', 'SectorGroup'
    ]].copy()

    portfolio_output.to_csv(output_path, index=False)
    print(f"\n[OK] Portfolio saved to: {output_path}")

    return undervalued


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build portfolio using relative rankings")
    parser.add_argument("--input", default="outputs/all_fundamentals_lasso_results.csv",
                       help="Path to model results CSV")
    parser.add_argument("--top-pct", type=float, default=20.0,
                       help="Top percentage to select (default: 20)")
    parser.add_argument("--min-mcap", type=float, default=1e9,
                       help="Minimum market cap in TL (default: 1B)")
    parser.add_argument("--max-stocks", type=int, default=20,
                       help="Maximum stocks in portfolio (default: 20)")

    args = parser.parse_args()

    # Resolve paths relative to script directory
    script_dir = Path(__file__).parent
    input_path = args.input if Path(args.input).is_absolute() else script_dir / args.input

    if not input_path.exists():
        print(f"\n[ERROR] Input file not found: {input_path}")
        print("Run the model first: python all_fundamentals_model.py --production")
        exit(1)

    result = build_portfolio(
        input_path=str(input_path),
        top_pct=args.top_pct,
        min_mcap=args.min_mcap,
        max_stocks=args.max_stocks
    )

    if result is None:
        print("\n[ERROR] Failed to build portfolio")
        exit(1)

    print("\n[SUCCESS] Portfolio built successfully!")
    print("\nRemember: These are RELATIVE rankings, not absolute predictions.")
    print("Expect modest outperformance (2-5% annual alpha), not 99% gains!")
