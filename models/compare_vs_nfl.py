"""
Compare our model's 2025 predictions against NFL.com's scraped weekly
projections, both against actual results. This is the real test the whole
project has been building toward: does our model have any marginal advantage
over public consensus, week over week, on a season it's never seen before.
"""

import polars as pl
from sklearn.metrics import mean_absolute_error

pl.Config.set_tbl_rows(25)


def normalize_name(col: pl.Expr) -> pl.Expr:
    return (
        col.str.replace_all(r"\s+(Jr\.?|Sr\.?|III|II|IV)$", "")
        .str.replace_all(r"[.']", "")
        .str.to_lowercase()
        .str.strip_chars()
    )


def main():
    ours = pl.read_parquet("data/processed/our_2025_predictions.parquet")
    nfl = pl.read_parquet("data/raw/nfl_projections_2025.parquet")

    ours = ours.with_columns(normalize_name(pl.col("player_display_name")).alias("name_norm"))
    nfl = nfl.with_columns(normalize_name(pl.col("player_name")).alias("name_norm"))

    merged = ours.join(
        nfl.select(["name_norm", "week", "position", "projected_points"]),
        on=["name_norm", "week", "position"],
        how="inner",
    )
    print(f"our predictions: {ours.shape[0]}")
    print(f"nfl.com projections: {nfl.shape[0]}")
    print(f"matched player-weeks (both have data): {merged.shape[0]}")

    # drop rows where NFL.com had no real projection (bye weeks, inactive
    # players show up as null from the scraper)
    merged = merged.filter(pl.col("projected_points").is_not_null())
    print(f"after dropping NFL.com nulls (byes/inactive): {merged.shape[0]}")

    pdf = merged.to_pandas()

    our_mae = mean_absolute_error(pdf["fantasy_points_ppr"], pdf["our_prediction"])
    nfl_mae = mean_absolute_error(pdf["fantasy_points_ppr"], pdf["projected_points"])

    print("\n" + "=" * 50)
    print("SEASON-WIDE RESULT (2025, all matched player-weeks)")
    print("=" * 50)
    print(f"Our model MAE:     {our_mae:.3f}")
    print(f"NFL.com MAE:       {nfl_mae:.3f}")
    print(f"Difference:        {nfl_mae - our_mae:+.3f}  ({'we beat consensus' if our_mae < nfl_mae else 'consensus beat us'})")

    print("\n" + "=" * 50)
    print("WEEK-BY-WEEK BREAKDOWN")
    print("=" * 50)
    weekly_results = []
    for week in sorted(pdf["week"].unique()):
        wk = pdf[pdf["week"] == week]
        if len(wk) == 0:
            continue
        w_our = mean_absolute_error(wk["fantasy_points_ppr"], wk["our_prediction"])
        w_nfl = mean_absolute_error(wk["fantasy_points_ppr"], wk["projected_points"])
        weekly_results.append((week, len(wk), w_our, w_nfl, w_our < w_nfl))
        print(f"  week {week:2d} (n={len(wk):4d}): our MAE {w_our:.3f}  |  NFL.com MAE {w_nfl:.3f}  |  {'US' if w_our < w_nfl else 'NFL'}")

    weeks_we_won = sum(1 for r in weekly_results if r[4])
    print(f"\nWeeks we beat NFL.com: {weeks_we_won} / {len(weekly_results)}")

    print("\n" + "=" * 50)
    print("BY POSITION")
    print("=" * 50)
    for pos in ["QB", "RB", "WR", "TE"]:
        p = pdf[pdf["position"] == pos]
        if len(p) == 0:
            continue
        p_our = mean_absolute_error(p["fantasy_points_ppr"], p["our_prediction"])
        p_nfl = mean_absolute_error(p["fantasy_points_ppr"], p["projected_points"])
        print(f"  {pos}: our MAE {p_our:.3f}  |  NFL.com MAE {p_nfl:.3f}  |  {'US' if p_our < p_nfl else 'NFL'}  (n={len(p)})")

    merged.write_parquet("data/processed/comparison_2025.parquet")
    print("\nSaved merged comparison to data/processed/comparison_2025.parquet")


if __name__ == "__main__":
    main()
