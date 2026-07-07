"""Smoke test: confirm nflreadpy can pull play-by-play and weekly data before committing to a full historical pull."""

import nflreadpy as nfl

pbp = nfl.load_pbp(seasons=[2024])
print("play-by-play shape:", pbp.shape)
print("columns sample:", pbp.columns[:15])

weekly = nfl.load_player_stats(seasons=[2024])
print("player weekly stats shape:", weekly.shape)
print(weekly.select(["player_display_name", "position", "week", "fantasy_points_ppr"]).head(5))
