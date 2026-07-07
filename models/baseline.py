"""
Naive baselines, evaluated on a chronological train/test split.

Split: train on 2016-2022, test on 2023-2024. This is not a random shuffle --
the model (and these baselines) never see 2023-2024 data during "training"
(baselines don't train, but the split boundary is what any real model will
also respect), mimicking how the system will actually be used: predicting
future weeks from past ones.

Any real model built next has to beat these numbers -- reporting error with no
baseline is meaningless.
"""

import polars as pl

TEST_SEASONS = [2023, 2024]


def mae(pred_col: str, actual_col: str, df: pl.DataFrame) -> float:
    valid = df.filter(pl.col(pred_col).is_not_null())
    return valid.select((pl.col(pred_col) - pl.col(actual_col)).abs().mean()).item()


def rmse(pred_col: str, actual_col: str, df: pl.DataFrame) -> float:
    valid = df.filter(pl.col(pred_col).is_not_null())
    return valid.select(((pl.col(pred_col) - pl.col(actual_col)) ** 2).mean().sqrt()).item()


def main():
    df = pl.read_parquet("data/processed/features_leakage_safe.parquet")

    test = df.filter(pl.col("season").is_in(TEST_SEASONS))
    train = df.filter(~pl.col("season").is_in(TEST_SEASONS))
    print(f"train rows: {train.shape[0]} (seasons {sorted(train['season'].unique().to_list())})")
    print(f"test rows: {test.shape[0]} (seasons {sorted(test['season'].unique().to_list())})")

    coverage_last_game = test["last_game_fantasy_points"].is_not_null().mean()
    coverage_season_avg = test["season_avg_fantasy_points_to_date"].is_not_null().mean()
    print(f"\ntest rows with a last-game value: {coverage_last_game:.1%}")
    print(f"test rows with a season-avg value: {coverage_season_avg:.1%}")

    print("\n--- Baseline 1: predict last week's actual score ---")
    print(f"MAE:  {mae('last_game_fantasy_points', 'fantasy_points_ppr', test):.3f}")
    print(f"RMSE: {rmse('last_game_fantasy_points', 'fantasy_points_ppr', test):.3f}")

    print("\n--- Baseline 2: predict season-to-date average ---")
    print(f"MAE:  {mae('season_avg_fantasy_points_to_date', 'fantasy_points_ppr', test):.3f}")
    print(f"RMSE: {rmse('season_avg_fantasy_points_to_date', 'fantasy_points_ppr', test):.3f}")

    print("\n--- Baseline 3: predict rolling 3-game average ---")
    print(f"MAE:  {mae('fantasy_points_ppr_rolling3g', 'fantasy_points_ppr', test):.3f}")
    print(f"RMSE: {rmse('fantasy_points_ppr_rolling3g', 'fantasy_points_ppr', test):.3f}")

    print(f"\ntest-set actual fantasy_points_ppr mean: {test['fantasy_points_ppr'].mean():.3f}")
    print(f"test-set actual fantasy_points_ppr std:  {test['fantasy_points_ppr'].std():.3f}")


if __name__ == "__main__":
    main()
