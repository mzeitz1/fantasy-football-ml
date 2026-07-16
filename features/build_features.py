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


def add_redzone(df: pl.DataFrame) -> pl.DataFrame:
    rz = pl.read_parquet("data/processed/redzone_features.parquet").select(
        ["player_id", "season", "week", "rz_share"]
    )
    df = df.join(rz, on=["player_id", "season", "week"], how="left")
    # a player-week absent from the red zone table genuinely had zero red zone
    # touches that week -- not missing data, a real 0
    df = df.with_columns(pl.col("rz_share").fill_null(0.0))
    return df


def add_player_rolling_features(df: pl.DataFrame) -> pl.DataFrame:
    df = df.sort(["player_id", "season", "week"])

    # raw last-game actual (not averaged) -- used for the "predict last week's
    # score" naive baseline, distinct from the rolling3g average
    df = df.with_columns(
        pl.col("fantasy_points_ppr").shift(1).over("player_id").alias("last_game_fantasy_points")
    )

    # rolling 3-game form: allowed to carry across a season boundary (a player's
    # most recent 3 games are informative even in week 1 of a new season)
    # target_share/air_yards_share were already sitting in weekly_player_table
    # (pulled by nflreadpy) but never actually used as model features until now
    rolling_cols = [
        "fantasy_points_ppr", "carries", "targets", "snap_pct",
        "target_share", "air_yards_share", "rz_share",
    ]
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
    sched = pl.read_parquet("data/raw/schedules_2016_2025.parquet").select(
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


def add_vegas_features(df: pl.DataFrame) -> pl.DataFrame:
    # spread_line/total_line are set by sportsbooks days before kickoff, so
    # -- like home/away and injury status -- these are pre-game context, NOT
    # lagged. Sign convention verified empirically against actual results
    # (not assumed from the column name): higher spread_line correlates with
    # the HOME team winning by more, so team_spread = spread_line for the home
    # team and -spread_line for the away team, with higher = more favored.
    relocation_map = {"SD": "LAC", "OAK": "LV"}
    sched = pl.read_parquet("data/raw/schedules_2016_2025.parquet").select(
        ["season", "week", "home_team", "away_team", "spread_line", "total_line"]
    ).with_columns(
        pl.col("home_team").replace(relocation_map),
        pl.col("away_team").replace(relocation_map),
    )
    home = sched.select(
        ["season", "week", pl.col("home_team").alias("team"),
         pl.col("spread_line").alias("team_spread"), "total_line"]
    )
    away = sched.select(
        ["season", "week", pl.col("away_team").alias("team"),
         (-pl.col("spread_line")).alias("team_spread"), "total_line"]
    )
    vegas = pl.concat([home, away]).unique()
    df = df.join(vegas, on=["season", "week", "team"], how="left")
    return df


def add_trend(df: pl.DataFrame) -> pl.DataFrame:
    # momentum: recent form relative to the season baseline, not just the
    # level of either alone -- catches a role changing (positive = trending
    # up vs. their own season average, negative = cooling off/losing role)
    df = df.with_columns(
        (pl.col("fantasy_points_ppr_rolling3g") - pl.col("season_avg_fantasy_points_to_date"))
        .alias("fantasy_points_trend")
    )
    return df


def add_target_competition(df: pl.DataFrame) -> pl.DataFrame:
    # opportunity security: is this player the team's clear top target-getter,
    # or in a timeshare? Built entirely from the already-lagged
    # target_share_rolling3g, so no additional lagging needed here -- 0 for
    # the team's current top target-getter, negative for everyone else
    team_max = (
        df.group_by(["team", "season", "week"])
        .agg(pl.col("target_share_rolling3g").max().alias("team_max_target_share_rolling3g"))
    )
    df = df.join(team_max, on=["team", "season", "week"], how="left")
    df = df.with_columns(
        (pl.col("target_share_rolling3g") - pl.col("team_max_target_share_rolling3g"))
        .alias("target_share_gap_to_team_leader")
    ).drop("team_max_target_share_rolling3g")
    return df


def main():
    df = pl.read_parquet("data/processed/weekly_player_table.parquet")
    df = df.drop("opp_def_points_allowed_to_position")  # same-week, leaky, recomputed lagged below

    df = add_redzone(df)
    df = add_player_rolling_features(df)
    df = add_lagged_opponent_defense(df)
    df = add_home_away(df)
    df = add_vegas_features(df)
    df = add_trend(df)
    df = add_target_competition(df)

    feature_cols = [
        "player_id", "player_display_name", "position", "team", "opponent_team",
        "season", "week",
        "last_game_fantasy_points",
        "fantasy_points_ppr_rolling3g", "carries_rolling3g", "targets_rolling3g",
        "snap_pct_rolling3g", "season_avg_fantasy_points_to_date",
        "target_share_rolling3g", "air_yards_share_rolling3g", "rz_share_rolling3g",
        "opp_def_rolling3g_points_allowed", "opp_def_season_avg_points_allowed_to_date",
        "is_home", "team_spread", "total_line",
        "fantasy_points_trend", "target_share_gap_to_team_leader",
        "report_status_ordinal", "practice_status_ordinal",
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
