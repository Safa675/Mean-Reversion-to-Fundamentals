"""
Fixed + Time Effects panel model for BIST fundamentals.

Uses a standalone data-prep/Lasso flow (not the pooled OLS module) to load all
fundamentals, z-score them, run Lasso selection, then fit PanelOLS with entity
and time effects. Outputs bubble diagnostics, plots, and VIF/correlation tables
for overall and each sector.
"""

import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from linearmodels.panel import PanelOLS

# Force rebuilding fundamentals for FE runs to stay independent of pooled OLS cache
os.environ["REBUILD_FUNDAMENTALS"] = "0"

try:
    from fixed_time_effects.fe_utils import (
        TARGET_COL,
        prepare_panel_data,
        run_lasso_feature_selection,
        select_common_sample,
        optimize_factor_coverage,
        drop_constant_columns,
        build_bubble_outputs,
        analyze_bubbles,
        plot_outputs,
        export_vif_and_corr,
        SECTOR_DISPLAY_NAMES,
    )
except ImportError:
    from fe_utils import (
        TARGET_COL,
        prepare_panel_data,
        run_lasso_feature_selection,
        select_common_sample,
        optimize_factor_coverage,
        drop_constant_columns,
        build_bubble_outputs,
        analyze_bubbles,
        plot_outputs,
        export_vif_and_corr,
        SECTOR_DISPLAY_NAMES,
    )

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = REPO_ROOT / "fixed_time_effects_outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

OUTPUT_XLSX = OUTPUT_DIR / "bist_fixed_time_effects.xlsx"


def run_fe_model(df_p, regressors, model_key, model_name, plot_paths=None):
    print(f"\n{'='*60}")
    print(f"Preparing data for {model_name}")
    print(f"{'='*60}")

    df_sub = df_p.dropna(subset=[TARGET_COL] + regressors).copy()
    if df_sub.empty:
        print("[ERROR] No data available after dropping NA rows")
        return None

    if "SectorGroup" not in df_sub.columns and "SectorGroup" in df_p.columns:
        df_sub["SectorGroup"] = df_p["SectorGroup"]

    print(f"\nFinal sample size: {len(df_sub)} observations")
    print(f"Number of unique tickers: {df_sub.index.get_level_values('Ticker').nunique()}")
    print(f"Date range: {df_sub.index.get_level_values('QuarterDate').min()} to {df_sub.index.get_level_values('QuarterDate').max()}")

    y = df_sub[TARGET_COL]
    X = df_sub[regressors]

    model = PanelOLS(
        y,
        X,
        entity_effects=True,
        time_effects=True,
        check_rank=False,
        drop_absorbed=True,  # auto-drop variables fully absorbed by entity/time effects
    )
    print("\nRunning Fixed + Time Effects regression...")
    res = model.fit(cov_type="robust")

    print(f"\n--- {model_name} RESULTS ---")
    print(f"R^2 (overall): {res.rsquared:.4f}")
    print(f"N: {res.nobs:,}")
    print("\n" + str(res.summary))

    fitted = res.predict().fitted_values
    pooled_df = build_bubble_outputs(model_key, y, fitted, df_sub)
    pooled_analysis = analyze_bubbles(model_name, pooled_df, verbose=True)

    metrics = {
        "Key": model_key,
        "Model": model_name,
        "R2_overall": res.rsquared,
        "Adj_R2": np.nan,
        "N": res.nobs,
        **pooled_analysis["stats"],
    }

    payload = {
        "key": model_key,
        "name": model_name,
        "df_results": pooled_df,
        **pooled_analysis,
        "metrics": metrics,
    }

    if plot_paths:
        plot_outputs(payload, plot_paths[0], plot_paths[1])

    return payload


def run_fe_pipeline(df_p, factor_set, key, name, save_common_path=None, plot_paths=None):
    selected_factors, df_common, dropped_factors, factor_counts = select_common_sample(
        df_p, factor_set, target_col=TARGET_COL
    )
    print(f"Common sample size: {df_common.shape[0]:,} observations")
    if dropped_factors:
        dropped_sorted = sorted(dropped_factors, key=lambda f: factor_counts.get(f, 0))
        dropped_msg = ", ".join(f"{f} ({factor_counts.get(f, 0)} valid)" for f in dropped_sorted)
        print(f"[WARN] Dropping sparse factors: {dropped_msg}")
    if df_common.empty or not selected_factors:
        print("[ERROR] No data after forming common sample.")
        return None, pd.DataFrame(), df_common, []

    optimized_factors, improved, _ = optimize_factor_coverage(
        df_p, selected_factors, target_col=TARGET_COL
    )
    if improved:
        factor_set = optimized_factors
        df_common = df_p.dropna(subset=[TARGET_COL] + factor_set).copy()
        print(f"Rebuilt common sample after optimization: {len(df_common):,} observations")
    else:
        factor_set = selected_factors

    if save_common_path is not None:
        df_common.reset_index().to_csv(save_common_path, index=False)
        print(f"[OK] Common sample saved to {save_common_path}")

    regressors, dropped_const = drop_constant_columns(df_common, factor_set)
    if dropped_const:
        print(f"\n[WARN] Dropped near-constant columns: {dropped_const}")

    if not regressors:
        print("\n[ERROR] No valid regressors available after dropping constant columns.")
        return None, pd.DataFrame(), df_common, []

    payload = run_fe_model(
        df_common,
        regressors,
        model_key=key,
        model_name=f"{name} ({len(regressors)} vars, FE+TE)",
        plot_paths=plot_paths,
    )
    if payload is None:
        return None, pd.DataFrame(), df_common, regressors
    if "SectorGroup" in df_common.columns:
        try:
            payload["df_results"]["SectorGroup"] = df_common["SectorGroup"]
        except Exception:
            payload["df_results"]["SectorGroup"] = df_common["SectorGroup"].reindex(payload["df_results"].index)

    metrics_df = pd.DataFrame([payload["metrics"]])
    return payload, metrics_df, df_common, regressors


def export_fe_outputs(payloads, metrics_df=None, output_xlsx=OUTPUT_XLSX):
    with pd.ExcelWriter(output_xlsx, engine="openpyxl") as writer:
        if metrics_df is not None and not metrics_df.empty:
            metrics_df.to_excel(writer, sheet_name="Model_Comparison", index=False)

        for payload in payloads:
            df_to_export = payload["df_results"].reset_index()
            bubble_q = payload["bubble_by_quarter"]
            bubble_t = payload["bubble_by_ticker"]
            top_over = payload["top_overvalued"]
            top_under = payload["top_undervalued"]

            key = payload["key"]
            export_cols = ["Ticker", "QuarterDate", "Actual_MarketCap", "Predicted_MarketCap",
                          "Residual_LogMCap", "Overvaluation_pct"]
            if "SectorGroup" in df_to_export.columns:
                export_cols.append("SectorGroup")

            df_to_export[export_cols].to_excel(
                writer, sheet_name=f"{key}_All"[:31], index=False
            )
            bubble_q.to_excel(writer, sheet_name=f"{key}_ByQuarter"[:31], index=False)
            bubble_t.to_excel(writer, sheet_name=f"{key}_ByCompany"[:31], index=False)
            top_over.to_excel(writer, sheet_name=f"{key}_MostOver"[:31], index=False)
            top_under.to_excel(writer, sheet_name=f"{key}_MostUnder"[:31], index=False)

    print(f"\n[OK] FE+TE results exported to: {output_xlsx}")


def main():
    print("\n" + "="*70)
    print(" BIST MARKET VALUATION - Fixed + Time Effects Model")
    print("="*70)

    df_p, factor_set, factor_component_map = prepare_panel_data()
    full_lasso_pool = [c for c in df_p.columns if c.endswith("_z") and c != TARGET_COL]
    print(f"\nLasso candidate pool size: {len(full_lasso_pool)} z-scored fundamentals (full set)")

    payloads = []
    metrics_frames = []

    # Overall model (lasso-selected)
    overall_features = run_lasso_feature_selection(
        df_p,
        full_lasso_pool,
        sector_label="ALL",
        min_features=5,
    )
    if not overall_features:
        print("[WARN] Lasso returned no features for ALL; falling back to pooled factor set")
        overall_features = factor_set

    overall_payload, overall_metrics, overall_df_common, overall_regressors = run_fe_pipeline(
        df_p,
        overall_features,
        key="Pooled_FE",
        name="Pooled Fixed+Time",
        save_common_path=OUTPUT_DIR / "bist_df_common_fixed_time.csv",
        plot_paths=(
            OUTPUT_DIR / "bist_bubbleness_timeseries_pooled_fe.png",
            OUTPUT_DIR / "bist_bubbleness_distribution_pooled_fe.png",
        ),
    )
    if overall_payload is not None:
        payloads.append(overall_payload)
        metrics_frames.append(overall_metrics)
        export_vif_and_corr(
            overall_df_common,
            overall_regressors,
            label="Overall FE",
            vif_path=OUTPUT_DIR / "bist_vif_table_pooled_fe.csv",
            corr_path=OUTPUT_DIR / "bist_correlation_vs_log_mcap_pooled_fe.csv",
        )

    # Sector-specific FE models
    sector_codes = ["CAP", "COM", "CON", "FIN", "IND", "INT", "RE"]
    sector_series = df_p["SectorGroup"] if "SectorGroup" in df_p.columns else pd.Series("", index=df_p.index)
    for code in sector_codes:
        df_sector = df_p[sector_series.fillna("") == code]
        if df_sector.empty:
            print(f"[WARN] Skipping sector {code}: no data after merge.")
            continue
        sector_label = SECTOR_DISPLAY_NAMES.get(code, f"{code} Sector")
        sector_features = run_lasso_feature_selection(
            df_sector,
            full_lasso_pool,
            sector_label=sector_label,
            min_features=3,
        )
        if not sector_features:
            print(f"[WARN] No Lasso-selected features for {sector_label}; falling back to pooled factor set")
            sector_features = factor_set

        ts_path = OUTPUT_DIR / f"bist_bubbleness_timeseries_{code.lower()}_fe.png"
        dist_path = OUTPUT_DIR / f"bist_bubbleness_distribution_{code.lower()}_fe.png"
        payload, metrics_df, df_common, regressors = run_fe_pipeline(
            df_sector,
            sector_features,
            key=f"{code}_FE",
            name=f"{sector_label} FE+Time",
            save_common_path=None,
            plot_paths=(ts_path, dist_path),
        )
        if payload is not None:
            payloads.append(payload)
            metrics_frames.append(metrics_df)
            export_vif_and_corr(
                df_common,
                regressors,
                label=f"{sector_label} FE",
                vif_path=OUTPUT_DIR / f"bist_vif_table_{code.lower()}_fe.csv",
                corr_path=OUTPUT_DIR / f"bist_correlation_vs_log_mcap_{code.lower()}_fe.csv",
            )

    if not payloads:
        print("[ERROR] No FE models ran successfully. Check data coverage.")
        return

    metrics_df = pd.concat(metrics_frames, ignore_index=True) if metrics_frames else None
    export_fe_outputs(payloads, metrics_df=metrics_df)

    print("\n" + "="*70)
    print(" FE+TIME ANALYSIS COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()
