"""
Full-dataset parity check: q-computed point-in-time features vs. the polars
pipeline from features/build_features.py. This is the actual point of building
the KDB layer at all -- two independent implementations of the same
leakage-safe logic agreeing is real evidence of correctness, not just a demo.
"""

import pykx as kx
import polars as pl
import numpy as np

from build_features_q import load_into_q, add_player_features, add_opponent_defense_asof


def main():
    load_into_q()
    add_player_features()
    add_opponent_defense_asof()

    q_weekly = kx.q("weekly").pd()
    q_defense_asof = kx.q("defense_asof").pd()

    polars_features = pl.read_parquet("data/processed/features_leakage_safe.parquet").to_pandas()

    # --- player-level features: join on player_id, season, week ---
    merged = q_weekly.merge(
        polars_features,
        on=["player_id", "season", "week"],
        suffixes=("_q", "_polars"),
    )
    print(f"merged rows for comparison: {len(merged)}")

    checks = [
        ("last_game_q", "last_game_fantasy_points"),
        ("fantasy_points_ppr_rolling3g_q", "fantasy_points_ppr_rolling3g"),
        ("season_avg_q", "season_avg_fantasy_points_to_date"),
    ]
    for q_col, polars_col in checks:
        both_present = merged[[q_col, polars_col]].dropna()
        diff = (both_present[q_col] - both_present[polars_col]).abs()
        max_diff = diff.max()
        mean_diff = diff.mean()
        pct_matching = (diff < 1e-6).mean()
        print(f"\n{q_col} vs {polars_col}:")
        print(f"  rows compared (both non-null): {len(both_present)}")
        print(f"  exact match rate: {pct_matching:.4%}")
        print(f"  max abs diff: {max_diff:.6f}, mean abs diff: {mean_diff:.6f}")

    # --- opponent defense: join q's as-of result to the player table on
    # (player_id, season, actual week = lookup week + 1) ---
    q_def = q_defense_asof.rename(columns={"week": "lookup_week"})
    q_def["week"] = q_def["lookup_week"] + 1
    def_merged = q_def.merge(
        polars_features,
        on=["player_id", "season", "week"],
        suffixes=("_q", "_polars"),
    )
    both_present = def_merged[["def_season_avg_q", "opp_def_season_avg_points_allowed_to_date"]].dropna()
    diff = (both_present["def_season_avg_q"] - both_present["opp_def_season_avg_points_allowed_to_date"]).abs()
    print(f"\nopp_def_season_avg_q vs opp_def_season_avg_points_allowed_to_date:")
    print(f"  rows compared: {len(both_present)}")
    print(f"  exact match rate: {(diff < 1e-6).mean():.4%}")
    print(f"  rows differing (expected: q correctly handles the DEFENSE's own bye weeks, polars does not): {(diff >= 1e-6).sum()}")
    if (diff >= 1e-6).sum() > 0:
        mismatches = def_merged.loc[diff[diff >= 1e-6].index, ["player_id", "season", "week", "def_season_avg_q", "opp_def_season_avg_points_allowed_to_date"]]
        print(mismatches.head(5))


if __name__ == "__main__":
    main()
