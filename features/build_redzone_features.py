"""
Derive red zone opportunity share from play-by-play: a player's red zone
touches (rush attempts + targets, inside the opponent's 20-yard line) as a
share of their team's total red zone touches that game.

This is outcome-derived (like carries/targets/snap_pct), so it gets lagged
the same way in build_features.py -- this script just produces the raw
per-player-week counts.
"""

import polars as pl

RED_ZONE_YARDLINE = 20


def main():
    pbp = pl.read_parquet("data/raw/pbp_2016_2025.parquet")
    rz = pbp.filter(pl.col("yardline_100") <= RED_ZONE_YARDLINE)

    rz_rushes = (
        rz.filter((pl.col("play_type") == "run") & pl.col("rusher_player_id").is_not_null())
        .group_by(["season", "week", "posteam", "rusher_player_id"])
        .agg(pl.len().alias("rz_touches"))
        .rename({"rusher_player_id": "player_id"})
    )
    rz_targets = (
        rz.filter((pl.col("play_type") == "pass") & pl.col("receiver_player_id").is_not_null())
        .group_by(["season", "week", "posteam", "receiver_player_id"])
        .agg(pl.len().alias("rz_touches"))
        .rename({"receiver_player_id": "player_id"})
    )

    rz_player = (
        pl.concat([rz_rushes, rz_targets])
        .group_by(["season", "week", "posteam", "player_id"])
        .agg(pl.col("rz_touches").sum())
    )

    rz_team_totals = (
        rz_player.group_by(["season", "week", "posteam"])
        .agg(pl.col("rz_touches").sum().alias("team_rz_touches"))
    )

    rz_final = rz_player.join(rz_team_totals, on=["season", "week", "posteam"], how="left")
    rz_final = rz_final.with_columns(
        (pl.col("rz_touches") / pl.col("team_rz_touches")).alias("rz_share")
    ).select(["player_id", "season", "week", "rz_touches", "team_rz_touches", "rz_share"])

    print("red zone player-weeks:", rz_final.shape[0])
    print(rz_final.sort("rz_touches", descending=True).head(5))

    rz_final.write_parquet("data/processed/redzone_features.parquet")
    print("\nSaved to data/processed/redzone_features.parquet")


if __name__ == "__main__":
    main()
