"""
Walk-forward multi-season backtest: for each test season, train only on
seasons strictly before it, then predict that season. This is the only
methodologically honest way to test multiple historical seasons against
consensus projections -- our model was previously trained through 2024 (to
predict 2025), so directly "testing" on 2021-2024 with that same model would
be circular (it already saw that data as training).

  train 2016-2020 -> predict 2021
  train 2016-2021 -> predict 2022
  train 2016-2022 -> predict 2023
  train 2016-2023 -> predict 2024
  train 2016-2024 -> predict 2025  (same as the earlier single-season exercise)

Hyperparameters are reused from the earlier chronological-CV search rather
than re-tuned per fold -- re-running a full search 5 times is expensive, and
the earlier tuning already showed results are fairly insensitive to exact
hyperparameters once naive-baseline territory is cleared.
"""

import polars as pl
import xgboost as xgb

TEST_SEASONS = [2021, 2022, 2023, 2024, 2025]

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

BEST_PARAMS = {
    "subsample": 0.8,
    "reg_lambda": 20.0,
    "n_estimators": 800,
    "min_child_weight": 3,
    "max_depth": 6,
    "learning_rate": 0.01,
    "colsample_bytree": 0.6,
    "random_state": 42,
}


def load_all():
    df = pl.read_parquet("data/processed/features_leakage_safe.parquet")
    df = df.to_dummies(columns=["position"])
    position_cols = [c for c in df.columns if c.startswith("position_")]
    feature_cols = NUMERIC_FEATURES + position_cols
    df = df.sort(["season", "week"])
    return df, feature_cols, position_cols


def main():
    df, feature_cols, position_cols = load_all()
    all_predictions = []

    for test_season in TEST_SEASONS:
        train = df.filter(pl.col("season") < test_season)
        test = df.filter(pl.col("season") == test_season)

        X_train = train.select(feature_cols).to_pandas()
        y_train = train[TARGET].to_pandas()
        X_test = test.select(feature_cols).to_pandas()

        model = xgb.XGBRegressor(**BEST_PARAMS)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        position_expr = pl.coalesce([
            pl.when(pl.col(c) == 1).then(pl.lit(c.removeprefix("position_"))) for c in position_cols
        ]).alias("position")
        out = test.select(
            ["player_id", "player_display_name", "team", "opponent_team",
             "season", "week", "fantasy_points_ppr"] + position_cols
        ).with_columns(pl.Series("our_prediction", preds)).with_columns(position_expr).drop(position_cols)

        all_predictions.append(out)
        print(f"test_season={test_season}: train={len(X_train)} rows, test={len(X_test)} rows")

    combined = pl.concat(all_predictions)
    combined.write_parquet("data/processed/walk_forward_predictions.parquet")
    print(f"\ntotal predictions: {combined.shape[0]}")
    print("Saved to data/processed/walk_forward_predictions.parquet")


if __name__ == "__main__":
    main()
