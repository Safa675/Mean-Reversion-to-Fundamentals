"""
Quick test script to verify the recommended pipeline works.

This script:
1. Tests if imports work
2. Tests if prepare_panel_data() works
3. Shows available columns and target
4. Helps diagnose any issues before running full pipeline
"""

import sys
from pathlib import Path

# Add parent directory to path to import from main Bist folder
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

print("="*70)
print("TESTING RECOMMENDED MODEL PIPELINE")
print("="*70)

# Test imports
print("\n[1/4] Testing imports...")
try:
    import numpy as np
    import pandas as pd
    import statsmodels.api as sm
    from sklearn.linear_model import LassoCV
    print("  ✓ All packages available")
except ImportError as e:
    print(f"  ✗ Import failed: {e}")
    print("  → Install missing packages:")
    print("     pip install numpy pandas statsmodels scikit-learn scipy matplotlib")
    sys.exit(1)

# Test data loading
print("\n[2/4] Testing data preparation...")
try:
    from pooled_ols_residuals_bist import prepare_panel_data, TARGET_COL, SECTOR_DISPLAY_NAMES
    print("  ✓ Successfully imported from pooled_ols_residuals_bist.py")
    print(f"  ✓ Target column is: '{TARGET_COL}'")
except ImportError as e:
    print(f"  ✗ Import failed: {e}")
    print("  → Make sure pooled_ols_residuals_bist.py is in the parent directory")
    print(f"  → Expected at: {Path(__file__).resolve().parent.parent / 'pooled_ols_residuals_bist.py'}")
    sys.exit(1)

# Load data
print("\n[3/4] Loading panel data...")
try:
    df_panel, factor_set, factor_component_map = prepare_panel_data()
    print(f"  ✓ Loaded {len(df_panel):,} observations")
    print(f"  ✓ Date range: {df_panel.index.get_level_values('QuarterDate').min()} to {df_panel.index.get_level_values('QuarterDate').max()}")
    print(f"  ✓ Unique tickers: {df_panel.index.get_level_values('Ticker').nunique()}")
except Exception as e:
    print(f"  ✗ Data loading failed: {e}")
    sys.exit(1)

# Check target and features
print("\n[4/4] Checking target column and features...")
print(f"  Target column: {TARGET_COL}")

if TARGET_COL in df_panel.columns:
    print(f"  ✓ Target column '{TARGET_COL}' found in data")
    print(f"    - Valid values: {df_panel[TARGET_COL].notna().sum():,}")
    print(f"    - Mean: {df_panel[TARGET_COL].mean():.2f}")
    print(f"    - Std: {df_panel[TARGET_COL].std():.2f}")
else:
    print(f"  ✗ Target column '{TARGET_COL}' NOT found in data")
    print(f"  Available columns: {list(df_panel.columns[:20])}")
    sys.exit(1)

# Check for z-scored features
z_features = [c for c in df_panel.columns if c.endswith("_z")]
print(f"\n  Z-scored features: {len(z_features)} available")
if len(z_features) > 0:
    print(f"  ✓ Examples: {z_features[:5]}")
else:
    print(f"  ✗ No z-scored features found!")
    sys.exit(1)

# Check sectors
if "SectorGroup" in df_panel.columns:
    sectors = df_panel["SectorGroup"].value_counts()
    print(f"\n  Sectors available:")
    for sector, count in sectors.items():
        sector_name = SECTOR_DISPLAY_NAMES.get(sector, sector)
        print(f"    - {sector}: {count:,} obs ({sector_name})")
else:
    print(f"\n  ✗ SectorGroup column not found")

print("\n" + "="*70)
print("ALL TESTS PASSED ✓")
print("="*70)
print("\nYou can now run:")
print("  python recommended_model_pipeline.py --sector ALL --method lasso --max-features 12")
print("\nOr try a specific sector:")
print("  python recommended_model_pipeline.py --sector FIN --method lasso --max-features 10")
