"""Pull full 2016-2024 play-by-play and weekly player stats from nflverse, save as parquet to data/raw/."""

import nflreadpy as nfl

SEASONS = list(range(2016, 2025))

print(f"Pulling play-by-play for seasons {SEASONS[0]}-{SEASONS[-1]}...")
pbp = nfl.load_pbp(seasons=SEASONS)
print("play-by-play shape:", pbp.shape)
pbp.write_parquet("data/raw/pbp_2016_2024.parquet")

print("Pulling weekly player stats...")
weekly = nfl.load_player_stats(seasons=SEASONS)
print("weekly stats shape:", weekly.shape)
weekly.write_parquet("data/raw/player_stats_weekly_2016_2024.parquet")

print("Pulling snap counts...")
snaps = nfl.load_snap_counts(seasons=SEASONS)
print("snap counts shape:", snaps.shape)
snaps.write_parquet("data/raw/snap_counts_2016_2024.parquet")

print("Pulling injury reports...")
injuries = nfl.load_injuries(seasons=SEASONS)
print("injuries shape:", injuries.shape)
injuries.write_parquet("data/raw/injuries_2016_2024.parquet")

print("Pulling rosters (for position/team metadata)...")
rosters = nfl.load_rosters(seasons=SEASONS)
print("rosters shape:", rosters.shape)
rosters.write_parquet("data/raw/rosters_2016_2024.parquet")

print("Done.")
