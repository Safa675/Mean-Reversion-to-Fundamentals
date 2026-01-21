import pandas as pd

from pooled_ols_residuals_bist import (
    TARGET_COL,
    prepare_panel_data,
    build_lasso_feature_pool,
    add_forward_returns,
    select_features_by_ic,
    run_pooled_ols,
)


def main():
    df_p, _, _ = prepare_panel_data()
    df_reset = df_p.reset_index()
    df_reset = add_forward_returns(df_reset, horizons=(1, 2, 4))

    candidate_pool = build_lasso_feature_pool(df_reset)
    selected = select_features_by_ic(
        df_reset,
        candidate_pool,
        model_type="elasticnet",
        l1_grid=[0.1, 0.3, 0.5, 0.7, 0.9],
        sector_label="Pooled IC-ElasticNet",
        min_features=5,
    )
    print(f"[RESULT] IC-ElasticNet selected features: {selected}")

    if not selected:
        print("[WARN] No features selected; aborting OLS run.")
        return

    df_common = df_p.dropna(subset=[TARGET_COL] + selected).copy()
    payload = run_pooled_ols(
        df_common,
        selected,
        model_key="Pooled_OLS_IC_ElasticNet",
        model_name=f"Pooled OLS (IC-ElasticNet, {len(selected)} vars)",
    )
    if payload is None:
        print("[ERROR] OLS run failed.")
    else:
        print("[OK] IC-ElasticNet OLS run complete.")


if __name__ == "__main__":
    main()
