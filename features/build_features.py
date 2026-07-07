"""
Build leakage-safe features for predicting a player's week-N fantasy performance.

Two categories of feature, handled differently -- this distinction is the whole
point of the leakage-safety requirement:

1. Outcome-derived features (rolling/season-to-date stats, opponent defense
   strength): these are built FROM game results, so they must only ever use
   weeks 1..N-1 relative to the week being predicted. Using week N's own
   carries/targets/snap% to predict week N's score would be circular -- the
   model would effectively already know the answer.

2. Pre-game context for week N itself (home/away, opponent identity, injury
   report status): these are legitimately known before week N's game is played
   (the injury report is published Wed-Fri before Sunday's game), so they are
   NOT lagged -- using them for week N is not leakage, it's just using
   information that genuinely existed before kickoff.
"""

import polars as pl

pl.Config.set_tbl_rows(20)


def add_player_rolling_features(df: pl.DataFrame) -> pl.DataFrame:
    df = df.sort(["player_id", "season", "week"])

    # raw last-game actual (not averaged) -- used for the "predict last week's
    # score" naive baseline, distinct from the rolling3g average
    df = df.with_columns(
        pl.col("fantasy_points_ppr").shift(1).over("player_id").alias("last_game_fantasy_points")
    )

    # rolling 3-game form: allowed to carry across a season boundary (a player's
    # most recent 3 games are informative even in week 1 of a new season)
    rolling_cols = ["fantasy_points_ppr", "carries", "targets", "snap_pct"]
    for col in rolling_cols:
        df = df.with_columns(
            pl.col(col)
            .shift(1)
            .rolling_mean(window_size=3, min_samples=1)
            .over("player_id")
            .alias(f"{col}_rolling3g")
        )

    # season-to-date average: resets each season by definition, null in week 1
    # of a season (no prior data that season) -- left null intentionally, to be
    # handled explicitly at model-training time rather than silently imputed here
    df = df.with_columns(
        (pl.col("fantasy_points_ppr").cum_sum().shift(1).over(["player_id", "season"]))
        .alias("_cum_sum_fp"),
        (pl.col("fantasy_points_ppr").cum_count().shift(1).over(["player_id", "season"]))
        .alias("_cum_count_fp"),
    )
    df = df.with_columns(
        (pl.col("_cum_sum_fp") / pl.col("_cum_count_fp")).alias("season_avg_fantasy_points_to_date")
    ).drop(["_cum_sum_fp", "_cum_count_fp"])

    return df


def add_lagged_opponent_defense(df: pl.DataFrame) -> pl.DataFrame:
    # rebuild the raw same-week defense-allowed series (from the CURRENT df,
    # which already has the leaky same-week column from Session 3 -- recompute
    # cleanly and then lag it; the same-week column itself gets dropped later)
    def_history = (
        df.group_by(["season", "week", "opponent_team", "position"])
        .agg(pl.col("fantasy_points_ppr").sum().alias("points_allowed"))
        .rename({"opponent_team": "defense_team"})
        .sort(["defense_team", "position", "season", "week"])
    )
    def_history = def_history.with_columns(
        pl.col("points_allowed")
        .shift(1)
        .rolling_mean(window_size=3, min_samples=1)
        .over(["defense_team", "position"])
        .alias("opp_def_rolling3g_points_allowed"),
        (pl.col("points_allowed").cum_sum().shift(1).over(["defense_team", "position", "season"]))
        .alias("_cum_sum_pa"),
        (pl.col("points_allowed").cum_count().shift(1).over(["defense_team", "position", "season"]))
        .alias("_cum_count_pa"),
    )
    def_history = def_history.with_columns(
        (pl.col("_cum_sum_pa") / pl.col("_cum_count_pa")).alias("opp_def_season_avg_points_allowed_to_date")
    ).drop(["_cum_sum_pa", "_cum_count_pa", "points_allowed"])

    df = df.join(
        def_history,
        left_on=["season", "week", "opponent_team", "position"],
        right_on=["season", "week", "defense_team", "position"],
        how="left",
    )
    return df


def add_home_away(df: pl.DataFrame) -> pl.DataFrame:
    # schedules uses period-accurate team codes for relocated franchises (SD,
    # OAK) while the player-stats table retroactively uses current codes
    # (LAC, LV) across all seasons -- remap before joining or those team-weeks
    # silently fail to match
    relocation_map = {"SD": "LAC", "OAK": "LV"}
    sched = pl.read_parquet("data/raw/schedules_2016_2024.parquet").select(
        ["season", "week", "home_team", "away_team"]
    ).with_columns(
        pl.col("home_team").replace(relocation_map),
        pl.col("away_team").replace(relocation_map),
    )
    home = sched.select(["season", "week", pl.col("home_team").alias("team")]).with_columns(
        pl.lit(1).alias("is_home")
    )
    away = sched.select(["season", "week", pl.col("away_team").alias("team")]).with_columns(
        pl.lit(0).alias("is_home")
    )
    home_away = pl.concat([home, away]).unique()
    df = df.join(home_away, on=["season", "week", "team"], how="left")
    return df


def main():
    df = pl.read_parquet("data/processed/weekly_player_table.parquet")
    df = df.drop("opp_def_points_allowed_to_position")  # same-week, leaky, recomputed lagged below

    df = add_player_rolling_features(df)
    df = add_lagged_opponent_defense(df)
    df = add_home_away(df)

    feature_cols = [
        "player_id", "player_display_name", "position", "team", "opponent_team",
        "season", "week",
        "last_game_fantasy_points",
        "fantasy_points_ppr_rolling3g", "carries_rolling3g", "targets_rolling3g",
        "snap_pct_rolling3g", "season_avg_fantasy_points_to_date",
        "opp_def_rolling3g_points_allowed", "opp_def_season_avg_points_allowed_to_date",
        "is_home", "report_status_ordinal", "practice_status_ordinal",
        "fantasy_points_ppr",  # target
    ]
    out = df.select(feature_cols)

    print("shape:", out.shape)
    print("\nnull rates:")
    print(out.select([pl.col(c).is_null().mean().alias(c) for c in feature_cols]))

    out.write_parquet("data/processed/features_leakage_safe.parquet")
    print("\nSaved to data/processed/features_leakage_safe.parquet")


if __name__ == "__main__":
    main()
