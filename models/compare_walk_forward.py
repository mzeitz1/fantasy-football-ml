"""
Full multi-season comparison: our walk-forward model predictions vs. the
hvpkod-derived consensus projections, both against actual results.

IMPORTANT: 2021-2023 are excluded. Investigation found their "_projected.csv"
files in the source repo are heavily contaminated with actual post-game
results mislabeled as pre-game projections -- exact-match rate between
"projection" and actual result is 78-90% for those seasons (a real forecast
cannot exactly equal a continuous multi-component final score by chance).
2024 (5.0% exact-match) and 2025 (6.8%) show a normal background rate
consistent with genuine forecasts, and 2025 independently cross-validates
against our own separately-scraped live NFL.com data from the earlier
single-season comparison (4.084 MAE there vs. consensus MAE computed here).
"""

import polars as pl
from sklearn.metrics import mean_absolute_error

pl.Config.set_tbl_rows(30)

CLEAN_SEASONS = [2024, 2025]
CONTAMINATION_EXACT_MATCH_THRESHOLD = 0.20  # seasons above this are excluded


def check_contamination(pdf) -> dict:
    """Exact-match rate between projection and actual, per season -- the
    signature of a real forecast (should be low, a few percent from players
    projected/scoring exactly 0) vs. leaked actuals (70-90%+)."""
    rates = {}
    for season in sorted(pdf["season"].unique()):
        s = pdf[pdf["season"] == season]
        rates[season] = (s["consensus_projection"] == s["fantasy_points_ppr"]).mean()
    return rates


def normalize_name(col: pl.Expr) -> pl.Expr:
    return (
        col.str.replace_all(r"\s+(Jr\.?|Sr\.?|III|II|IV)$", "")
        .str.replace_all(r"[.']", "")
        .str.to_lowercase()
        .str.strip_chars()
    )


def main():
    ours = pl.read_parquet("data/processed/walk_forward_predictions.parquet")
    consensus = pl.read_parquet("data/processed/consensus_projections_2021_2025.parquet")

    ours = ours.with_columns(normalize_name(pl.col("player_display_name")).alias("name_norm"))

    merged = ours.join(
        consensus.select(["name_norm", "season", "week", "position", "consensus_projection"]),
        on=["name_norm", "season", "week", "position"],
        how="inner",
    )
    print(f"our predictions: {ours.shape[0]}")
    print(f"matched player-weeks: {merged.shape[0]}")

    pdf = merged.to_pandas()

    print("\n" + "=" * 55)
    print("CONTAMINATION CHECK (exact-match rate: projection == actual)")
    print("=" * 55)
    rates = check_contamination(pdf)
    excluded = []
    for season, rate in rates.items():
        flag = "EXCLUDED -- contaminated" if rate > CONTAMINATION_EXACT_MATCH_THRESHOLD else "clean"
        print(f"  {season}: {rate:.1%}  ({flag})")
        if rate > CONTAMINATION_EXACT_MATCH_THRESHOLD:
            excluded.append(season)

    pdf = pdf[~pdf["season"].isin(excluded)]
    print(f"\nProceeding with seasons {sorted(pdf['season'].unique().tolist())} only "
          f"(excluded {excluded} as contaminated)")

    our_mae = mean_absolute_error(pdf["fantasy_points_ppr"], pdf["our_prediction"])
    consensus_mae = mean_absolute_error(pdf["fantasy_points_ppr"], pdf["consensus_projection"])

    print("\n" + "=" * 55)
    print("OVERALL RESULT (clean seasons only, all matched player-weeks)")
    print("=" * 55)
    print(f"Our model MAE:        {our_mae:.3f}")
    print(f"Consensus MAE:        {consensus_mae:.3f}")
    print(f"Difference:           {consensus_mae - our_mae:+.3f}  ({'we beat consensus' if our_mae < consensus_mae else 'consensus beat us'})")

    print("\n" + "=" * 55)
    print("BY SEASON")
    print("=" * 55)
    season_results = []
    for season in sorted(pdf["season"].unique()):
        s = pdf[pdf["season"] == season]
        s_our = mean_absolute_error(s["fantasy_points_ppr"], s["our_prediction"])
        s_con = mean_absolute_error(s["fantasy_points_ppr"], s["consensus_projection"])
        season_results.append((season, len(s), s_our, s_con))
        note = " (2021 RB rushing-TD gap in source)" if season == 2021 else ""
        print(f"  {season} (n={len(s):5d}): our MAE {s_our:.3f}  |  consensus MAE {s_con:.3f}  |  {'US' if s_our < s_con else 'CONSENSUS'}{note}")

    print("\n" + "=" * 55)
    print("BY POSITION (all seasons combined)")
    print("=" * 55)
    for pos in ["QB", "RB", "WR", "TE"]:
        p = pdf[pdf["position"] == pos]
        p_our = mean_absolute_error(p["fantasy_points_ppr"], p["our_prediction"])
        p_con = mean_absolute_error(p["fantasy_points_ppr"], p["consensus_projection"])
        print(f"  {pos}: our MAE {p_our:.3f}  |  consensus MAE {p_con:.3f}  |  {'US' if p_our < p_con else 'CONSENSUS'}  (n={len(p)})")

    print("\n" + "=" * 55)
    print("WEEKS/SEASON-POSITION CELLS WE WON")
    print("=" * 55)
    wins = 0
    total = 0
    for season in sorted(pdf["season"].unique()):
        for week in sorted(pdf[pdf["season"] == season]["week"].unique()):
            wk = pdf[(pdf["season"] == season) & (pdf["week"] == week)]
            if len(wk) < 5:
                continue
            w_our = mean_absolute_error(wk["fantasy_points_ppr"], wk["our_prediction"])
            w_con = mean_absolute_error(wk["fantasy_points_ppr"], wk["consensus_projection"])
            total += 1
            if w_our < w_con:
                wins += 1
    print(f"Weeks we beat consensus: {wins} / {total}")

    merged.write_parquet("data/processed/walk_forward_comparison.parquet")
    print("\nSaved to data/processed/walk_forward_comparison.parquet")


if __name__ == "__main__":
    main()
