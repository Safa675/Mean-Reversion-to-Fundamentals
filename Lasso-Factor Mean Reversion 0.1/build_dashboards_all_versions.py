"""
Build dashboards for all 3 factor model versions.

Creates separate dashboard folders for:
1. All data (2016-2025)
2. Last 5 years (2020-2025)
3. Last 3 years (2022-2025)

Each dashboard tracks performance from Nov 15, 2025.
"""

from pathlib import Path
import pandas as pd
import shutil

ROOT = Path(__file__).resolve().parent

versions = [
    {
        'name': 'all_data',
        'folder': 'dashboard_all_data',
        'portfolio_path': 'outputs_all_data/portfolio_20250930.csv',
        'description': 'All Data (2016-2025)'
    },
    {
        'name': 'last_5years',
        'folder': 'dashboard_last_5years',
        'portfolio_path': 'outputs_last_5years/portfolio_20250930.csv',
        'description': 'Last 5 Years (2020-2025)'
    },
    {
        'name': 'last_3years',
        'folder': 'dashboard_last_3years',
        'portfolio_path': 'outputs_last_3years/portfolio_20250930.csv',
        'description': 'Last 3 Years (2022-2025)'
    }
]

print("="*80)
print("BUILDING DASHBOARDS FOR ALL VERSIONS")
print("="*80)

for version in versions:
    print(f"\n{version['description']}...")

    # Create dashboard folder
    dashboard_dir = ROOT / version['folder']
    dashboard_dir.mkdir(exist_ok=True)

    # Load portfolio
    portfolio_path = ROOT / version['portfolio_path']
    if not portfolio_path.exists():
        print(f"  [ERROR] Portfolio not found: {portfolio_path}")
        continue

    portfolio_df = pd.read_csv(portfolio_path, index_col=0)

    # Get tickers from index (multi-index: Ticker, QuarterDate)
    tickers = portfolio_df.index.get_level_values(0).unique().tolist()

    print(f"  Portfolio: {len(tickers)} stocks")

    # Create picks.csv with equal weights
    picks_df = pd.DataFrame({
        'Ticker': tickers,
        'Weight': [1/len(tickers)] * len(tickers)
    })

    picks_path = dashboard_dir / 'picks.csv'
    picks_df.to_csv(picks_path, index=False)
    print(f"  [OK] Created {picks_path}")

    # Copy build_dashboard_live.py
    src_script = ROOT / 'dashboard_bundle' / 'build_dashboard_live.py'
    dst_script = dashboard_dir / 'build_dashboard_live.py'

    if src_script.exists():
        shutil.copy(src_script, dst_script)
        print(f"  [OK] Copied dashboard script")
    else:
        print(f"  [WARN] Dashboard script not found at {src_script}")

print("\n" + "="*80)
print("ALL DASHBOARDS CREATED")
print("="*80)

print("\nNow run dashboards:")
for version in versions:
    dashboard_dir = ROOT / version['folder']
    print(f"\ncd \"{dashboard_dir}\"")
    print(f"/home/safa/anaconda3/bin/python build_dashboard_live.py")

print("\nOr run all at once:")
print("python run_all_dashboards.py")
