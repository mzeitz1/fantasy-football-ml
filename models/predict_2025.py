"""
Train on all of 2016-2024 (this project's full backtested history), tune via
the same chronological CV approach as tune_xgboost.py, then generate this
model's own predictions for every 2025 player-week.

2025 is deliberately excluded from training entirely -- it's the comparison
set against scraped NFL.com projections (see scripts/scrape_nfl_projections.py),
a genuinely fresh out-of-sample season this model has never touched, unlike
the 2023-2024 test set used for the original Day 4 backtest report.
"""

import polars as pl
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, make_scorer
import xgboost as xgb

TRAIN_SEASONS = list(range(2016, 2025))  # 2016-2024 inclusive
PREDICT_SEASON = 2025

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


def load_split():
    df = pl.read_parquet("data/processed/features_leakage_safe.parquet")
    df = df.to_dummies(columns=["position"])
    position_cols = [c for c in df.columns if c.startswith("position_")]
    feature_cols = NUMERIC_FEATURES + position_cols

    df = df.sort(["season", "week"])
    train = df.filter(pl.col("season").is_in(TRAIN_SEASONS))
    predict = df.filter(pl.col("season") == PREDICT_SEASON)
    return train, predict, feature_cols


def main():
    train, predict_df, feature_cols = load_split()
    X_train = train.select(feature_cols).to_pandas()
    y_train = train[TARGET].to_pandas()
    X_predict = predict_df.select(feature_cols).to_pandas()

    print(f"train rows: {len(X_train)} (seasons 2016-2024)")
    print(f"2025 rows to predict: {len(X_predict)}")

    # hyperparameters already found via RandomizedSearchCV + TimeSeriesSplit on
    # this exact train set in a prior run (mean CV MAE: 4.643) -- reusing
    # directly since the search is deterministic (random_state=42) and re-running
    # 200 model fits again would just reproduce the identical result
    best_params = {
        "subsample": 0.8,
        "reg_lambda": 20.0,
        "n_estimators": 800,
        "min_child_weight": 3,
        "max_depth": 6,
        "learning_rate": 0.01,
        "colsample_bytree": 0.6,
        "random_state": 42,
    }
    print("\n--- using previously-found best hyperparameters ---")
    for k, v in best_params.items():
        print(f"  {k}: {v}")

    best_model = xgb.XGBRegressor(**best_params)
    best_model.fit(X_train, y_train)
    predictions = best_model.predict(X_predict)

    # position was one-hot encoded in load_split() for model input; reconstruct
    # the plain label here since we need it to match against scraped NFL.com data
    position_cols = [c for c in predict_df.columns if c.startswith("position_")]
    out = predict_df.select(
        ["player_id", "player_display_name", "team", "opponent_team",
         "season", "week", "fantasy_points_ppr"] + position_cols
    ).with_columns(pl.Series("our_prediction", predictions))
    position_expr = pl.coalesce([
        pl.when(pl.col(c) == 1).then(pl.lit(c.removeprefix("position_"))) for c in position_cols
    ]).alias("position")
    out = out.with_columns(position_expr).drop(position_cols)

    out.write_parquet("data/processed/our_2025_predictions.parquet")
    print(f"\nSaved {out.shape[0]} predictions to data/processed/our_2025_predictions.parquet")
    print(out.head(10))


if __name__ == "__main__":
    main()
