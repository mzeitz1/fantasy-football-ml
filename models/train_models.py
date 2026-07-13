"""
First model pass: Ridge regression, then XGBoost, both on the leakage-safe
features from Session 1, evaluated on the same chronological 2023-2024 test
split as the naive baselines. The number to beat: 4.57 MAE (season-to-date
average baseline).

Missing-value handling differs by model on purpose:
- Ridge can't handle NaN, so missing values (mostly week-1-of-season rows with
  no season-to-date average yet) are median-imputed, fit on TRAIN ONLY so no
  test-set information leaks into the imputation values.
- XGBoost handles NaN natively (it learns a default split direction for missing
  values), so it gets the raw features with nulls intact -- no imputation
  needed or wanted.
"""

import polars as pl
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
import xgboost as xgb

TEST_SEASONS = [2023, 2024]

NUMERIC_FEATURES = [
    "last_game_fantasy_points",
    "fantasy_points_ppr_rolling3g",
    "carries_rolling3g",
    "targets_rolling3g",
    "snap_pct_rolling3g",
    "season_avg_fantasy_points_to_date",
    "opp_def_rolling3g_points_allowed",
    "opp_def_season_avg_points_allowed_to_date",
    "is_home",
    "report_status_ordinal",
    "practice_status_ordinal",
]
TARGET = "fantasy_points_ppr"


def load_split():
    df = pl.read_parquet("data/processed/features_leakage_safe.parquet")
    df = df.to_dummies(columns=["position"])
    position_cols = [c for c in df.columns if c.startswith("position_")]

    train = df.filter(~pl.col("season").is_in(TEST_SEASONS))
    test = df.filter(pl.col("season").is_in(TEST_SEASONS))
    feature_cols = NUMERIC_FEATURES + position_cols
    return train, test, feature_cols


def run_ridge(train, test, feature_cols):
    X_train = train.select(feature_cols).to_pandas()
    X_test = test.select(feature_cols).to_pandas()
    y_train = train[TARGET].to_pandas()
    y_test = test[TARGET].to_pandas()

    # median-impute using TRAIN statistics only
    medians = X_train.median()
    X_train = X_train.fillna(medians)
    X_test = X_test.fillna(medians)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = Ridge(alpha=1.0)
    model.fit(X_train_scaled, y_train)
    preds = model.predict(X_test_scaled)

    mae = mean_absolute_error(y_test, preds)
    rmse = root_mean_squared_error(y_test, preds)
    print("\n--- Ridge regression ---")
    print(f"MAE:  {mae:.3f}")
    print(f"RMSE: {rmse:.3f}")

    coefs = sorted(zip(feature_cols, model.coef_), key=lambda x: -abs(x[1]))
    print("\ncoefficients (standardized, so magnitude = relative importance):")
    for name, coef in coefs:
        print(f"  {name:45s} {coef:+.3f}")

    return mae, rmse


def run_xgboost(train, test, feature_cols):
    X_train = train.select(feature_cols).to_pandas()
    X_test = test.select(feature_cols).to_pandas()
    y_train = train[TARGET].to_pandas()
    y_test = test[TARGET].to_pandas()

    model = xgb.XGBRegressor(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
    )
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    rmse = root_mean_squared_error(y_test, preds)
    print("\n--- XGBoost ---")
    print(f"MAE:  {mae:.3f}")
    print(f"RMSE: {rmse:.3f}")

    importances = sorted(zip(feature_cols, model.feature_importances_), key=lambda x: -x[1])
    print("\nfeature importances:")
    for name, imp in importances:
        print(f"  {name:45s} {imp:.3f}")

    return mae, rmse


def main():
    train, test, feature_cols = load_split()
    print(f"train rows: {train.shape[0]}, test rows: {test.shape[0]}")
    print(f"features: {feature_cols}")

    ridge_mae, ridge_rmse = run_ridge(train, test, feature_cols)
    xgb_mae, xgb_rmse = run_xgboost(train, test, feature_cols)

    print("\n" + "=" * 50)
    print("SUMMARY vs. naive baseline (season-avg: MAE 4.574, RMSE 6.448)")
    print("=" * 50)
    print(f"Ridge:   MAE {ridge_mae:.3f}  RMSE {ridge_rmse:.3f}")
    print(f"XGBoost: MAE {xgb_mae:.3f}  RMSE {xgb_rmse:.3f}")


if __name__ == "__main__":
    main()
