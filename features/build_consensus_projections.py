"""
Convert hvpkod/NFL-Data's raw projected stat lines into a standard full-PPR
projection we can fairly compare against our own model.

Why not use their own PlayerWeekProjectedPts/TotalPoints directly: verified
against our nflverse actuals for matched player-weeks, their own point totals
run ~2.2 points below ours on average (correlated with receptions) -- almost
certainly a different scoring system (their README notes points are "specific
for the MML league point system"). Recomputing from the raw stat columns
(which the README says are scoring-system-agnostic) using a standard full-PPR
formula and validating against our nflverse actuals for the SAME games
confirms the fix: mean abs diff drops to 0.115 (rounding noise), from ~2.2.
"""

import polars as pl

STAT_COLS = [
    "PassingYDS", "PassingTD", "PassingInt", "RushingYDS", "RushingTD",
    "ReceivingRec", "ReceivingYDS", "ReceivingTD", "RetTD", "FumTD", "2PT", "Fum",
]


def normalize_name(col: pl.Expr) -> pl.Expr:
    return (
        col.str.replace_all(r"\s+(Jr\.?|Sr\.?|III|II|IV)$", "")
        .str.replace_all(r"[.']", "")
        .str.to_lowercase()
        .str.strip_chars()
    )


def main():
    hv = pl.read_parquet("data/raw/historical_projections_2021_2025.parquet")
    hv = hv.with_columns([pl.col(c).cast(pl.Float64, strict=False).fill_null(0.0) for c in STAT_COLS])
    hv = hv.with_columns(
        (
            0.04 * pl.col("PassingYDS") + 4 * pl.col("PassingTD") - 2 * pl.col("PassingInt")
            + 0.1 * pl.col("RushingYDS") + 6 * pl.col("RushingTD")
            + 1.0 * pl.col("ReceivingRec") + 0.1 * pl.col("ReceivingYDS") + 6 * pl.col("ReceivingTD")
            + 6 * pl.col("RetTD") + 6 * pl.col("FumTD") + 2 * pl.col("2PT") - 2 * pl.col("Fum")
        ).alias("consensus_projection")
    )
    hv = hv.with_columns(normalize_name(pl.col("player_name")).alias("name_norm"))

    out = hv.select([
        "name_norm", "player_name", "position", "team", "opponent",
        "season", "week", "consensus_projection",
    ])
    print("shape:", out.shape)
    print(out.head(5))

    out.write_parquet("data/processed/consensus_projections_2021_2025.parquet")
    print("\nSaved to data/processed/consensus_projections_2021_2025.parquet")

    print("\nNote: 2021 RB rushing TDs are missing from the source entirely "
          "(that season's RB export lacks the column) -- 2021 RB consensus "
          "projections are therefore understated and should be flagged as such.")


if __name__ == "__main__":
    main()
