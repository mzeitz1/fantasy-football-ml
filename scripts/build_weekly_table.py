"""
Build the unified weekly player table for RB/WR/QB/TE: raw stats + snap % + injury
ordinals (two separate features, see design decision below) + opponent defense
strength allowed to each position.

This is the RAW aggregated material, not yet leakage-safe. It intentionally includes
same-week opponent-defense numbers (e.g. week 5's opponent-defense-vs-RB is computed
from week 5's games). Turning this into point-in-time-safe features (only using weeks
1..N-1 for anything used to predict week N) is Day 2's feature engineering step, not
this one. Do not use this table directly as model input.

Injury encoding decision: report_status (game-day designation) and practice_status
(that day's practice participation) are two different signals that don't always
agree, so we keep them as two separate ordinal features rather than hand-blending
them into one scale. Lets the model learn how much each matters.
"""

import polars as pl

SKILL_POSITIONS = ["QB", "RB", "WR", "TE"]

REPORT_STATUS_ORDINAL = {None: 0, "Questionable": 1, "Doubtful": 2, "Out": 3}
PRACTICE_STATUS_ORDINAL = {
    "Full Participation in Practice": 0,
    "Limited Participation in Practice": 1,
    "Did Not Participate In Practice": 2,
}


def normalize_name(col: pl.Expr) -> pl.Expr:
    return (
        col.str.replace_all(r"\s+(Jr\.?|Sr\.?|III|II|IV)$", "")
        .str.replace_all(r"[.']", "")
        .str.to_lowercase()
        .str.strip_chars()
    )


def main():
    weekly = pl.read_parquet("data/raw/player_stats_weekly_2016_2025.parquet")
    snaps = pl.read_parquet("data/raw/snap_counts_2016_2025.parquet")
    injuries = pl.read_parquet("data/raw/injuries_2016_2025.parquet")

    weekly = weekly.filter(pl.col("position").is_in(SKILL_POSITIONS))
    print("weekly skill-position rows:", weekly.shape[0])

    # --- snap % join ---
    weekly = weekly.with_columns(normalize_name(pl.col("player_display_name")).alias("name_norm"))
    snaps = snaps.with_columns(normalize_name(pl.col("player")).alias("name_norm"))
    weekly = weekly.join(
        snaps.select(["name_norm", "team", "season", "week", "position", "offense_pct"]).rename(
            {"offense_pct": "snap_pct"}
        ),
        on=["name_norm", "team", "season", "week", "position"],
        how="left",
    )
    print("snap_pct null rate after join:", weekly["snap_pct"].is_null().mean())

    # --- injury ordinals join ---
    # a player can have multiple report snapshots within a week (e.g. Thu -> Fri
    # update); keep only the most recent one, which is the final pre-game call
    injuries_clean = (
        injuries.with_columns(pl.col("season").cast(pl.Int32), pl.col("week").cast(pl.Int32))
        .sort("date_modified")
        .group_by(["season", "week", "gsis_id"], maintain_order=True)
        .last()
        .select(["season", "week", "gsis_id", "report_status", "practice_status"])
    )
    weekly = weekly.join(
        injuries_clean,
        left_on=["season", "week", "player_id"],
        right_on=["season", "week", "gsis_id"],
        how="left",
    )
    weekly = weekly.with_columns(
        pl.col("report_status")
        .replace_strict(REPORT_STATUS_ORDINAL, default=0)
        .alias("report_status_ordinal"),
        pl.col("practice_status")
        .replace_strict(PRACTICE_STATUS_ORDINAL, default=0)
        .alias("practice_status_ordinal"),
    )
    print(
        "report_status_ordinal distribution:\n",
        weekly["report_status_ordinal"].value_counts().sort("report_status_ordinal"),
    )
    print(
        "practice_status_ordinal distribution:\n",
        weekly["practice_status_ordinal"].value_counts().sort("practice_status_ordinal"),
    )

    # --- opponent defense strength: fantasy points allowed by each team to each
    # position, per season/week. Raw same-week number -- NOT lagged yet. ---
    # for each (team, position, week), how many fantasy points did that team's
    # defense allow to that position -- keyed by the DEFENSE's own team code
    def_strength = (
        weekly.group_by(["season", "week", "opponent_team", "position"])
        .agg(pl.col("fantasy_points_ppr").sum().alias("opp_def_points_allowed_to_position"))
        .rename({"opponent_team": "defense_team"})
    )
    # attach it to each player-week using THAT WEEK'S opponent as the lookup key
    weekly = weekly.join(
        def_strength,
        left_on=["season", "week", "opponent_team", "position"],
        right_on=["season", "week", "defense_team", "position"],
        how="left",
    )

    out_cols = [
        "player_id", "player_display_name", "position", "team", "opponent_team",
        "season", "week", "carries", "rushing_yards", "targets", "receptions",
        "receiving_yards", "fantasy_points_ppr", "target_share", "air_yards_share",
        "snap_pct", "report_status_ordinal", "practice_status_ordinal",
        "opp_def_points_allowed_to_position",
    ]
    out = weekly.select(out_cols)
    print("\nfinal table shape:", out.shape)
    print(out.head(5))

    out.write_parquet("data/processed/weekly_player_table.parquet")
    print("\nSaved to data/processed/weekly_player_table.parquet")


if __name__ == "__main__":
    main()
