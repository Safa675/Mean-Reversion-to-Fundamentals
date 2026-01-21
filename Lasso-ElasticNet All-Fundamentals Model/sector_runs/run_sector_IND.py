"""Run all-fundamentals model for IND sector only."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from all_fundamentals_model import run_all_fundamentals_model  # noqa: E402

SECTOR = "IND"


def main() -> None:
    parser = argparse.ArgumentParser(description=f"Run IND sector model ({SECTOR})")
    parser.add_argument("--method", default="lasso", choices=["lasso", "elasticnet"])
    parser.add_argument("--max-features", type=int, default=20)
    parser.add_argument("--min-coverage", type=float, default=10.0)
    parser.add_argument("--corr-threshold", type=float, default=0.95)
    parser.add_argument("--train-end", default=None)
    parser.add_argument("--val-end", default=None)
    parser.add_argument("--production", action="store_true")
    parser.add_argument("--min-obs", type=int, default=500)
    args = parser.parse_args()

    train_end = None if args.production else args.train_end
    val_end = None if args.production else args.val_end

    res = run_all_fundamentals_model(
        method=args.method,
        max_features=args.max_features,
        min_coverage_pct=args.min_coverage,
        corr_threshold=args.corr_threshold,
        train_end=train_end,
        val_end=val_end,
        sector_code=SECTOR,
        min_obs=args.min_obs,
    )
    if res is None:
        raise SystemExit("[ERROR] Sector run failed.")

    print(f"[SUCCESS] Sector {SECTOR} completed. Output key: {res.get('output_key')}")


if __name__ == "__main__":
    main()
