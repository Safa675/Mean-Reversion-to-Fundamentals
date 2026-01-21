"""
ENSEMBLE PORTFOLIO BUILDER

Combines the old Factor-based model with the new All-Fundamentals model.
Stocks that appear in BOTH portfolios are more likely to be truly undervalued.

Usage:
    python ensemble_portfolios.py \
        --new-model outputs/portfolio_20250930.csv \
        --old-model "../Lasso-Factor Mean Reversion 0.1/outputs/undervalued_portfolio.csv"
"""

import argparse
import pandas as pd
from pathlib import Path

def load_portfolio(path, ticker_col='Ticker'):
    """Load portfolio CSV and extract tickers"""
    df = pd.read_csv(path)

    # Handle different column names
    if ticker_col not in df.columns:
        # Try common alternatives
        for alt_col in ['ticker', 'Symbol', 'symbol', 'Code', 'code']:
            if alt_col in df.columns:
                ticker_col = alt_col
                break

    if ticker_col not in df.columns:
        print(f"[ERROR] Could not find ticker column in {path}")
        print(f"Available columns: {df.columns.tolist()}")
        return None

    return df

def ensemble_portfolios(new_model_path, old_model_path):
    """
    Combine portfolios from new and old models.

    Returns:
        - Intersection: Stocks in BOTH portfolios (highest confidence)
        - Union: Stocks in EITHER portfolio (diversified)
        - New-only: Stocks only in new model
        - Old-only: Stocks only in old model
    """
    print("\n" + "="*80)
    print("ENSEMBLE PORTFOLIO BUILDER")
    print("="*80)

    # Load portfolios
    print(f"\nLoading new model portfolio: {new_model_path}")
    df_new = load_portfolio(new_model_path)
    if df_new is None:
        return None

    print(f"Loading old model portfolio: {old_model_path}")
    df_old = load_portfolio(old_model_path)
    if df_old is None:
        return None

    # Extract tickers
    tickers_new = set(df_new['Ticker'].unique())
    tickers_old = set(df_old.iloc[:, 0].unique())  # First column is usually ticker

    print(f"\n  New model: {len(tickers_new)} stocks")
    print(f"  Old model: {len(tickers_old)} stocks")

    # Compute sets
    intersection = tickers_new & tickers_old
    union = tickers_new | tickers_old
    new_only = tickers_new - tickers_old
    old_only = tickers_old - tickers_new

    print(f"\n" + "="*80)
    print("ENSEMBLE ANALYSIS")
    print("="*80)

    # Intersection (highest confidence)
    print(f"\n1. INTERSECTION ({len(intersection)} stocks)")
    print("   → Stocks in BOTH portfolios (highest confidence)")
    print("   → Recommended for core positions")
    print("-"*60)

    if intersection:
        for ticker in sorted(intersection):
            # Get details from new model
            new_row = df_new[df_new['Ticker'] == ticker].iloc[0]
            rank = int(new_row.get('Undervaluation_Rank', 0))
            pct = new_row.get('Undervaluation_Percentile', 0)
            mcap = new_row.get('MarketCap', 0) / 1e9
            sector = new_row.get('SectorGroup', 'N/A')

            print(f"  {ticker:<10} Rank={rank:<3} Percentile={pct:5.1f}%  MCap={mcap:6.2f}B  {sector}")
    else:
        print("  [WARN] No stocks found in both portfolios!")
        print("  This suggests the two models have very different views.")

    # New model only
    print(f"\n2. NEW MODEL ONLY ({len(new_only)} stocks)")
    print("   → Stocks the new all-fundamentals model likes")
    print("   → Consider for satellite positions")
    print("-"*60)

    if new_only:
        for ticker in sorted(list(new_only)[:10]):  # Top 10
            new_row = df_new[df_new['Ticker'] == ticker].iloc[0]
            rank = int(new_row.get('Undervaluation_Rank', 0))
            pct = new_row.get('Undervaluation_Percentile', 0)
            print(f"  {ticker:<10} Rank={rank:<3} Percentile={pct:5.1f}%")

    # Old model only
    print(f"\n3. OLD MODEL ONLY ({len(old_only)} stocks)")
    print("   → Stocks the old factor-based model likes")
    print("   → Consider for satellite positions")
    print("-"*60)

    if old_only:
        for ticker in sorted(list(old_only)[:10]):  # Top 10
            print(f"  {ticker:<10}")

    # Recommendations
    print(f"\n" + "="*80)
    print("RECOMMENDATIONS")
    print("="*80)

    if len(intersection) >= 5:
        print(f"\n✅ GOOD CONSENSUS: {len(intersection)} stocks in both portfolios")
        print(f"   → Allocate 60-70% to intersection (core)")
        print(f"   → Allocate 15-20% to new-only (satellite)")
        print(f"   → Allocate 15-20% to old-only (satellite)")
    elif len(intersection) >= 3:
        print(f"\n⚠️  MODERATE CONSENSUS: {len(intersection)} stocks in both portfolios")
        print(f"   → Allocate 40-50% to intersection (core)")
        print(f"   → Allocate 25-30% to new-only (satellite)")
        print(f"   → Allocate 25-30% to old-only (satellite)")
    else:
        print(f"\n❌ LOW CONSENSUS: Only {len(intersection)} stocks in both portfolios")
        print(f"   → Models have divergent views")
        print(f"   → Consider equal weight to both models")
        print(f"   → Or pick the model with stronger historical IC")

    print(f"\nHistorical IC comparison:")
    print(f"  Old factor model: IC ≈ -0.21 (backtest)")
    print(f"  New all-fundamentals model: IC = -0.0684 (training)")
    print(f"\n  → Old model has stronger historical IC")
    print(f"  → But new model has less omitted variable bias")
    print(f"  → Ensemble reduces model-specific risk")

    # Save results
    output_dir = Path(new_model_path).parent

    # Intersection portfolio
    if intersection:
        intersection_df = df_new[df_new['Ticker'].isin(intersection)].copy()
        intersection_path = output_dir / "ensemble_intersection.csv"
        intersection_df.to_csv(intersection_path, index=False)
        print(f"\n[OK] Intersection portfolio saved: {intersection_path}")

    # Union portfolio
    union_df = pd.concat([
        df_new,
        df_old.rename(columns={df_old.columns[0]: 'Ticker'})
    ]).drop_duplicates(subset=['Ticker'])

    union_path = output_dir / "ensemble_union.csv"
    union_df.to_csv(union_path, index=False)
    print(f"[OK] Union portfolio saved: {union_path}")

    return {
        'intersection': intersection,
        'union': union,
        'new_only': new_only,
        'old_only': old_only
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Combine new and old model portfolios")
    parser.add_argument("--new-model",
                       default="outputs/portfolio_20250930.csv",
                       help="Path to new all-fundamentals model portfolio")
    parser.add_argument("--old-model",
                       default="../Lasso-Factor Mean Reversion 0.1/outputs/undervalued_portfolio.csv",
                       help="Path to old factor-based model portfolio")

    args = parser.parse_args()

    # Resolve paths
    script_dir = Path(__file__).parent
    new_model_path = args.new_model if Path(args.new_model).is_absolute() else script_dir / args.new_model
    old_model_path = args.old_model if Path(args.old_model).is_absolute() else script_dir / args.old_model

    # Check files exist
    if not new_model_path.exists():
        print(f"\n[ERROR] New model portfolio not found: {new_model_path}")
        print("Run: python build_portfolio_correct.py")
        exit(1)

    if not old_model_path.exists():
        print(f"\n[ERROR] Old model portfolio not found: {old_model_path}")
        print("Run the old factor-based model first")
        exit(1)

    # Build ensemble
    result = ensemble_portfolios(str(new_model_path), str(old_model_path))

    if result is None:
        print("\n[ERROR] Failed to build ensemble")
        exit(1)

    print("\n[SUCCESS] Ensemble portfolios created!")
    print("\nNext steps:")
    print("  1. Review ensemble_intersection.csv (highest confidence)")
    print("  2. Review ensemble_union.csv (diversified)")
    print("  3. Decide on allocation: 60% core + 40% satellite")
    print("  4. Implement risk management (position sizing, stop losses)")
