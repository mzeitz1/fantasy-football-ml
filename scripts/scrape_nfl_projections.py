"""
Scrape fantasy.nfl.com's week-by-week 2025 projections for QB/RB/WR/TE.

This is the only season NFL.com retains real (non-placeholder) weekly
projection data for -- confirmed by testing 2023/2024 (all-zero placeholders,
current-roster metadata bleeding through) vs 2025 (real varying numbers,
correct period matchups). 2025 also happens to be a season our model has
never touched in training (2016-2022) or testing (2023-2024), making it a
genuinely fresh out-of-sample comparison.

Rate-limited to be a polite scraper of public data: one request per page,
small delay between requests, no parallelism.
"""

import time
import requests
from bs4 import BeautifulSoup
import polars as pl

POSITIONS = {1: "QB", 2: "RB", 3: "WR", 4: "TE"}
SEASON = 2025
WEEKS = range(1, 19)
BASE_URL = "https://fantasy.nfl.com/research/projections"
HEADERS = {"User-Agent": "Mozilla/5.0 (research script, personal project)"}
DELAY_SECONDS = 0.4


def fetch_page(position: int, week: int, offset: int) -> str:
    params = {
        "position": position,
        "statCategory": "projectedStats",
        "statSeason": SEASON,
        "statType": "weekProjectedStats",
        "statWeek": week,
    }
    if offset > 1:
        params["offset"] = offset
    resp = requests.get(BASE_URL, params=params, headers=HEADERS, timeout=20)
    resp.raise_for_status()
    return resp.text


def parse_total_count(html: str) -> int:
    soup = BeautifulSoup(html, "html.parser")
    title = soup.find("span", class_="paginationTitle")
    if title is None:
        return 0
    # format: " 1 - 25 of 242 "
    text = title.get_text(strip=True)
    return int(text.split("of")[-1].strip())


def parse_players(html: str, season: int, week: int, position_label: str) -> list[dict]:
    soup = BeautifulSoup(html, "html.parser")
    rows = soup.find_all("tr", class_=lambda c: c and "player-" in c)
    records = []
    for row in rows:
        name_tag = row.find("a", class_="playerNameFull")
        if name_tag is None:
            continue
        name = name_tag.get_text(strip=True)

        team_tag = row.find("em")
        team = team_tag.get_text(strip=True).split(" - ")[-1] if team_tag else None

        opp_tag = row.find("td", class_="playerOpponent")
        opponent = opp_tag.get_text(strip=True) if opp_tag else None

        pts_tag = row.find("span", class_=lambda c: c and "playerWeekProjectedPts" in c)
        pts_text = pts_tag.get_text(strip=True) if pts_tag else None
        projected_points = None
        if pts_text and pts_text != "-":
            try:
                projected_points = float(pts_text)
            except ValueError:
                projected_points = None

        records.append({
            "season": season,
            "week": week,
            "position": position_label,
            "player_name": name,
            "team": team,
            "opponent": opponent,
            "projected_points": projected_points,
        })
    return records


def scrape_all() -> pl.DataFrame:
    all_records = []
    for position_code, position_label in POSITIONS.items():
        for week in WEEKS:
            first_page_html = fetch_page(position_code, week, offset=1)
            time.sleep(DELAY_SECONDS)
            total = parse_total_count(first_page_html)
            all_records.extend(parse_players(first_page_html, SEASON, week, position_label))

            offset = 26
            while offset <= total:
                html = fetch_page(position_code, week, offset)
                time.sleep(DELAY_SECONDS)
                all_records.extend(parse_players(html, SEASON, week, position_label))
                offset += 25

            print(f"{position_label} week {week}: {total} players scraped")

    return pl.DataFrame(all_records)


def main():
    df = scrape_all()
    print(f"\ntotal rows scraped: {df.shape[0]}")
    print(df.head(10))
    df.write_parquet("data/raw/nfl_projections_2025.parquet")
    print("Saved to data/raw/nfl_projections_2025.parquet")


if __name__ == "__main__":
    main()
