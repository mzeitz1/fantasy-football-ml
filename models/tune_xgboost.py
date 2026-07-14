"""
Hyperparameter tuning for XGBoost using chronological (time-series) cross-
validation, entirely within the training period (2016-2022). The 2023-2024
test set is never touched during tuning -- it gets exactly one evaluation,
at the very end, with the already-chosen hyperparameters.

Why TimeSeriesSplit and not random k-fold: k-fold shuffles rows into random
folds, so a model could "validate" on an early season using training data
that includes a later one -- the same lookahead problem as Session 1's
leakage-safety work, just relocated into the tuning loop. TimeSeriesSplit
only ever validates on data chronologically after what it trained on.
"""

import polars as pl
import numpy as np
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, make_scorer
import xgboost as xgb

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


def load_split():
    df = pl.read_parquet("data/processed/features_leakage_safe.parquet")
    df = df.to_dummies(columns=["position"])
    position_cols = [c for c in df.columns if c.startswith("position_")]
    feature_cols = NUMERIC_FEATURES + position_cols

    # sort chronologically -- TimeSeriesSplit assumes row order IS time order
    df = df.sort(["season", "week"])

    train = df.filter(~pl.col("season").is_in(TEST_SEASONS))
    test = df.filter(pl.col("season").is_in(TEST_SEASONS))
    return train, test, feature_cols


def main():
    train, test, feature_cols = load_split()
    X_train = train.select(feature_cols).to_pandas()
    y_train = train[TARGET].to_pandas()
    X_test = test.select(feature_cols).to_pandas()
    y_test = test[TARGET].to_pandas()

    print(f"train rows: {len(X_train)} (seasons 2016-2022)")
    print(f"test rows (held out, untouched until the end): {len(X_test)} (seasons 2023-2024)")

    # 5 chronological splits WITHIN the training period. Each split trains on
    # an expanding early window and validates on the chunk immediately after
    # it -- never on anything earlier.
    tscv = TimeSeriesSplit(n_splits=5)
    print(f"\nTimeSeriesSplit produces {tscv.n_splits} splits. Sizes (rows):")
    for i, (tr_idx, val_idx) in enumerate(tscv.split(X_train)):
        print(f"  split {i}: train={len(tr_idx):6d}  validate={len(val_idx):6d}")

    # hyperparameter search space -- see accompanying explanation of what each
    # one actually controls
    param_dist = {
        "max_depth": [2, 3, 4, 5, 6],
        "learning_rate": [0.01, 0.03, 0.05, 0.1, 0.2],
        "n_estimators": [100, 200, 300, 500, 800],
        "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
        "min_child_weight": [1, 3, 5, 10, 20],
        "reg_lambda": [0.1, 1.0, 5.0, 10.0, 20.0],
    }

    mae_scorer = make_scorer(mean_absolute_error, greater_is_better=False)

    search = RandomizedSearchCV(
        estimator=xgb.XGBRegressor(random_state=42),
        param_distributions=param_dist,
        n_iter=40,
        scoring=mae_scorer,
        cv=tscv,
        n_jobs=-1,
        random_state=42,
        verbose=1,
    )
    search.fit(X_train, y_train)

    print("\n--- best hyperparameters found (by mean CV MAE across the 5 chronological splits) ---")
    for k, v in search.best_params_.items():
        print(f"  {k}: {v}")
    print(f"  mean CV MAE: {-search.best_score_:.3f}")

    # final, ONE-TIME evaluation on the untouched test set
    best_model = search.best_estimator_
    test_preds = best_model.predict(X_test)
    test_mae = mean_absolute_error(y_test, test_preds)
    print(f"\n--- final test-set MAE (2023-2024, evaluated once) ---")
    print(f"  Tuned XGBoost: {test_mae:.3f}")
    print(f"  (yesterday's untuned XGBoost: 4.480, naive baseline: 4.574)")

    best_model.save_model("models/xgboost_tuned.json")
    print("\nSaved tuned model to models/xgboost_tuned.json")


if __name__ == "__main__":
    main()
