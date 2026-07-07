"""Investigate injury report_status null meaning and the id fields available for joining snap_counts to weekly stats."""

import polars as pl

pl.Config.set_tbl_rows(30)
pl.Config.set_tbl_cols(20)

injuries = pl.read_parquet("data/raw/injuries_2016_2024.parquet")
weekly = pl.read_parquet("data/raw/player_stats_weekly_2016_2024.parquet")
snaps = pl.read_parquet("data/raw/snap_counts_2016_2024.parquet")

print("=" * 60)
print("INJURY report_status vs practice_status crosstab")
print("=" * 60)
print(
    injuries.group_by(["report_status", "practice_status"])
    .agg(pl.len().alias("count"))
    .sort("count", descending=True)
)

print("\nsample rows where report_status is null:")
print(
    injuries.filter(pl.col("report_status").is_null())
    .select(["season", "week", "full_name", "report_primary_injury", "practice_status", "practice_primary_injury"])
    .head(10)
)

print("\n" + "=" * 60)
print("ID FIELDS AVAILABLE")
print("=" * 60)
print("weekly stats id-like columns:", [c for c in weekly.columns if "id" in c.lower() or "player" in c.lower()][:15])
print("snap_counts id-like columns:", [c for c in snaps.columns if "id" in c.lower() or "player" in c.lower()])
print("injuries id-like columns:", [c for c in injuries.columns if "id" in c.lower() or "name" in c.lower()])
