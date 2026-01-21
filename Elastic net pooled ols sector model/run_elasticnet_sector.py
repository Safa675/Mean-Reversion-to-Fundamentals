import sys
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from pooled_ols_residuals_bist import (
    TARGET_COL,
    prepare_panel_data,
    build_lasso_feature_pool,
    add_forward_returns,
    select_features_by_ic,
    run_pooled_ols,
    export_outputs,
)

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "Elastic net sector model"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_XLSX = OUTPUT_DIR / "bist_elasticnet_sector_models.xlsx"

SECTOR_CODES = ["CAP", "COM", "CON", "FIN", "IND", "INT", "RE"]


def run_sector(df_p, code):
    sector_series = df_p["SectorGroup"] if "SectorGroup" in df_p.columns else pd.Series("", index=df_p.index)
    df_sector = df_p[sector_series.fillna("") == code].copy()
    if df_sector.empty:
        print(f"[WARN] Skipping sector {code}: no data")
        return None

    df_for_ic = add_forward_returns(df_sector.reset_index(), horizons=(1, 2, 4))
    candidate_pool = build_lasso_feature_pool(df_for_ic)
    selected = select_features_by_ic(
        df_for_ic,
        candidate_pool,
        model_type="elasticnet",
        sector_label=code,
        min_features=5,
        max_features=20,
    )
    if not selected:
        print(f"[WARN] No IC-selected features for sector {code}; skipping model")
        return None

    df_common = df_sector.dropna(subset=[TARGET_COL] + selected).copy()
    df_common["SectorGroup"] = code

    payload = run_pooled_ols(
        df_common,
        selected,
        model_key=f"{code}_EN",
        model_name=f"{code} ElasticNet (IC)",
    )
    return payload


def main():
    df_p, _, _ = prepare_panel_data()
    # Overall pooled model
    df_all_for_ic = add_forward_returns(df_p.reset_index(), horizons=(1, 2, 4))
    candidate_pool = build_lasso_feature_pool(df_all_for_ic)
    selected_all = select_features_by_ic(
        df_all_for_ic,
        candidate_pool,
        model_type="elasticnet",
        sector_label="ALL",
        min_features=8,
        max_features=25,
    )
    payloads = []
    metrics = []

    if selected_all:
        df_common_all = df_p.dropna(subset=[TARGET_COL] + selected_all).copy()
        payload = run_pooled_ols(
            df_common_all,
            selected_all,
            model_key="Pooled_EN",
            model_name=f"Pooled ElasticNet (IC, {len(selected_all)} vars)",
        )
        if payload:
            payloads.append(payload)

    for code in SECTOR_CODES:
        payload = run_sector(df_p, code)
        if payload:
            payloads.append(payload)

    if not payloads:
        print("[ERROR] No ElasticNet IC models ran.")
        return

    # Export to dedicated workbook
    with pd.ExcelWriter(OUTPUT_XLSX, engine="openpyxl") as writer:
        for payload in payloads:
            df_to_export = payload["df_results"].reset_index()
            bubble_q = payload["bubble_by_quarter"]
            bubble_t = payload["bubble_by_ticker"]
            top_over = payload["top_overvalued"]
            top_under = payload["top_undervalued"]

            key = payload["key"]
            df_to_export[["Ticker", "QuarterDate", "Actual_MarketCap", "Predicted_MarketCap",
                          "Residual_LogMCap", "Overvaluation_pct"]].to_excel(
                writer, sheet_name=f"{key}_All"[:31], index=False
            )
            bubble_q.to_excel(writer, sheet_name=f"{key}_ByQuarter"[:31], index=False)
            bubble_t.to_excel(writer, sheet_name=f"{key}_ByCompany"[:31], index=False)
            top_over.to_excel(writer, sheet_name=f"{key}_MostOver"[:31], index=False)
            top_under.to_excel(writer, sheet_name=f"{key}_MostUnder"[:31], index=False)

    print(f"[OK] ElasticNet sector models saved to {OUTPUT_XLSX}")


if __name__ == "__main__":
    main()
