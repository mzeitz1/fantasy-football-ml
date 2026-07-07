"""Explore schema, missingness, and position coverage across the pulled datasets."""

import polars as pl

pl.Config.set_tbl_rows(30)

weekly = pl.read_parquet("data/raw/player_stats_weekly_2016_2024.parquet")
snaps = pl.read_parquet("data/raw/snap_counts_2016_2024.parquet")
injuries = pl.read_parquet("data/raw/injuries_2016_2024.parquet")
rosters = pl.read_parquet("data/raw/rosters_2016_2024.parquet")

print("=" * 60)
print("WEEKLY PLAYER STATS")
print("=" * 60)
print("shape:", weekly.shape)
print("\nposition value counts:")
print(weekly["position"].value_counts().sort("count", descending=True))

fantasy_positions = ["QB", "RB", "WR", "TE"]
skill = weekly.filter(pl.col("position").is_in(fantasy_positions))
print(f"\nrows for {fantasy_positions}:", skill.shape)
print("\nplayer-weeks per season (skill positions only):")
print(skill.group_by("season").agg(pl.len().alias("player_weeks")).sort("season"))

print("\nkey column null rates (skill positions):")
key_cols = [
    "carries", "rushing_yards", "targets", "receptions", "receiving_yards",
    "fantasy_points_ppr", "target_share", "air_yards_share",
]
present = [c for c in key_cols if c in skill.columns]
missing_cols = [c for c in key_cols if c not in skill.columns]
if missing_cols:
    print("not found in schema:", missing_cols)
null_rates = skill.select([pl.col(c).is_null().mean().alias(c) for c in present])
print(null_rates)

print("\n" + "=" * 60)
print("SNAP COUNTS")
print("=" * 60)
print("shape:", snaps.shape)
print("columns:", snaps.columns)
print("seasons covered:", sorted(snaps["season"].unique().to_list()))

print("\n" + "=" * 60)
print("INJURY REPORTS")
print("=" * 60)
print("shape:", injuries.shape)
print("columns:", injuries.columns)
if "report_status" in injuries.columns:
    print("\nreport_status value counts:")
    print(injuries["report_status"].value_counts().sort("count", descending=True))
print("seasons covered:", sorted(injuries["season"].unique().to_list()))

print("\n" + "=" * 60)
print("ROSTERS")
print("=" * 60)
print("shape:", rosters.shape)
print("columns sample:", rosters.columns[:20])
