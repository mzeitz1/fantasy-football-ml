"""
Feature importance deep-dive on the tuned XGBoost model: compare the three
native importance types (weight/gain/cover), which can rank features
differently and disagree for informative reasons, then use SHAP for a more
rigorous global ranking AND a concrete per-prediction explanation.
"""

import polars as pl
import numpy as np
import xgboost as xgb
import shap

TEST_SEASONS = [2023, 2024]
NUMERIC_FEATURES = [
    "last_game_fantasy_points",
    "fantasy_points_ppr_rolling3g",
    "carries_rolling3g",
    "targets_rolling3g",
    "snap_pct_rolling3g",
    "season_avg_fantasy_points_to_date",
    "target_share_rolling3g",
    "air_yards_share_rolling3g",
    "rz_share_rolling3g",
    "opp_def_rolling3g_points_allowed",
    "opp_def_season_avg_points_allowed_to_date",
    "is_home",
    "team_spread",
    "total_line",
    "fantasy_points_trend",
    "target_share_gap_to_team_leader",
    "report_status_ordinal",
    "practice_status_ordinal",
]
TARGET = "fantasy_points_ppr"


def load_data():
    df = pl.read_parquet("data/processed/features_leakage_safe.parquet")
    df = df.to_dummies(columns=["position"])
    position_cols = [c for c in df.columns if c.startswith("position_")]
    feature_cols = NUMERIC_FEATURES + position_cols
    test = df.filter(pl.col("season").is_in(TEST_SEASONS))
    return test, feature_cols


def main():
    test, feature_cols = load_data()
    X_test = test.select(feature_cols).to_pandas()
    y_test = test[TARGET].to_pandas()

    model = xgb.XGBRegressor()
    model.load_model("models/xgboost_tuned.json")
    booster = model.get_booster()

    print("=" * 70)
    print("NATIVE IMPORTANCE TYPES -- note where these disagree")
    print("=" * 70)
    for imp_type in ["weight", "gain", "cover"]:
        scores = booster.get_score(importance_type=imp_type)
        ranked = sorted(scores.items(), key=lambda x: -x[1])
        print(f"\n--- {imp_type} ---")
        for name, val in ranked[:8]:
            print(f"  {name:45s} {val:10.2f}")

    print("\n" + "=" * 70)
    print("SHAP VALUES -- global importance (mean |SHAP| per feature)")
    print("=" * 70)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_test)

    mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
    ranked_shap = sorted(zip(feature_cols, mean_abs_shap), key=lambda x: -x[1])
    for name, val in ranked_shap:
        print(f"  {name:45s} {val:8.3f}")

    print("\n" + "=" * 70)
    print("ONE PREDICTION, EXPLAINED -- why did the model predict THIS number, for THIS player-week")
    print("=" * 70)
    # pick a high-confidence, high-scoring prediction to make the explanation concrete
    preds = model.predict(X_test)
    idx = int(np.argmax(preds))
    row_info = test.select(["player_display_name", "season", "week", "fantasy_points_ppr"]).to_pandas().iloc[idx]
    print(f"Player: {row_info['player_display_name']}, {row_info['season']} week {row_info['week']}")
    print(f"Model predicted: {preds[idx]:.2f}   Actual: {row_info['fantasy_points_ppr']:.2f}   Base rate (avg prediction): {explainer.expected_value:.2f}")
    print("\nTop contributors to THIS prediction (feature: SHAP contribution, feature's actual value):")
    row_shap = shap_values.values[idx]
    row_features = X_test.iloc[idx]
    contribs = sorted(zip(feature_cols, row_shap, row_features), key=lambda x: -abs(x[1]))
    for name, contrib, val in contribs[:8]:
        sign = "+" if contrib >= 0 else ""
        print(f"  {name:45s} {sign}{contrib:6.2f}   (value: {val:.2f})")


if __name__ == "__main__":
    main()
