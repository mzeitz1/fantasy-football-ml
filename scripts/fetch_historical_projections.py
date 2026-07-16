"""
Pull historical weekly projections + actuals from hvpkod/NFL-Data (github.com/
hvpkod/NFL-Data), extracted from Fantasy.NFL.com by that repo's author. This
gives us multiple real historical seasons (2021-2025) of consensus projections,
unlike our own scrape which only had live 2025 data -- letting us build a
proper multi-season walk-forward backtest instead of a single-season snapshot.

2020 is excluded: that year's folder in the source repo only has season-total
files, no per-week/projected breakdown (checked directly before writing this).

Each position's per-week "_projected.csv" conveniently includes BOTH the
projection (PlayerWeekProjectedPts) and the actual result (TotalPoints) in the
same file, from the same source -- no separate join needed for that part.
"""

import time
import requests
import polars as pl
from io import StringIO

POSITIONS = ["QB", "RB", "WR", "TE"]
YEARS_WEEKS = {
    2021: range(1, 18),
    2022: range(1, 18),
    2023: range(1, 18),
    2024: range(1, 18),
    2025: range(1, 19),
}
BASE = "https://raw.githubusercontent.com/hvpkod/NFL-Data/main/NFL-data-Players"
DELAY_SECONDS = 0.2

KEEP_COLS = [
    "PlayerName", "PlayerId", "Pos", "Team", "PlayerOpponent",
    "PassingYDS", "PassingTD", "PassingInt",
    "RushingYDS", "RushingTD",
    "ReceivingRec", "ReceivingYDS", "ReceivingTD",
    "RetTD", "FumTD", "2PT", "Fum",
    "PlayerWeekProjectedPts", "TotalPoints",
]


def fetch_one(season: int, week: int, position: str) -> pl.DataFrame | None:
    url = f"{BASE}/{season}/{week}/projected/{position}_projected.csv"
    resp = requests.get(url, timeout=20)
    if resp.status_code != 200 or not resp.text.strip():
        return None
    df = pl.read_csv(StringIO(resp.text), infer_schema_length=10000, ignore_errors=True)
    # 2021 RB files only: rushing yards column is misnamed/combined, and
    # rushing TD is entirely absent from that season's RB export -- a real,
    # documented gap in the source data, not something we can recover
    if "RushingPassingYDS" in df.columns and "RushingYDS" not in df.columns:
        df = df.rename({"RushingPassingYDS": "RushingYDS"})
    present_cols = [c for c in KEEP_COLS if c in df.columns]
    df = df.select(present_cols).with_columns(
        pl.lit(season).alias("season"),
        pl.lit(week).alias("week"),
    )
    return df


def main():
    all_frames = []
    for season, weeks in YEARS_WEEKS.items():
        for week in weeks:
            for position in POSITIONS:
                df = fetch_one(season, week, position)
                time.sleep(DELAY_SECONDS)
                if df is not None:
                    all_frames.append(df)
            print(f"{season} week {week}: done")

    combined = pl.concat(all_frames, how="diagonal_relaxed")
    combined = combined.rename({
        "PlayerName": "player_name",
        "PlayerId": "player_id_src",
        "Pos": "position",
        "Team": "team",
        "PlayerOpponent": "opponent",
        "PlayerWeekProjectedPts": "projected_points",
        "TotalPoints": "actual_points_src",
    })
    print(f"\ntotal rows: {combined.shape[0]}")
    print(combined.head(10))

    combined.write_parquet("data/raw/historical_projections_2021_2025.parquet")
    print("\nSaved to data/raw/historical_projections_2021_2025.parquet")


if __name__ == "__main__":
    main()
