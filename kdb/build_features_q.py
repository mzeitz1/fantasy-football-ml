"""
Point-in-time feature pipeline, built in q via PyKX -- a parallel implementation
of features/build_features.py, used to (a) genuinely exercise KDB+/q on this
project's real event-time data, per the plan's rationale, and (b) cross-validate
the pandas/polars pipeline: two independent implementations agreeing is a much
stronger correctness signal than trusting one.

Key q language elements used here, each doing in one primitive what took several
composed steps in polars:
  xasc          -- sort ascending (q needs explicit ordering; row order isn't
                   assumed to mean anything)
  prev / next   -- shift a list by 1 (like .shift(1)/.shift(-1))
  mavg          -- trailing moving average as a single operator (e.g. `3 mavg x`)
  avgs          -- running/cumulative average of a list at every point
  by ... from   -- group-by, applied per-group to the whole expression
  aj            -- as-of join: for each row in the left table, find the LAST
                   matching row in the right table with time <= the left row's
                   time. This is the one operation that's genuinely awkward to
                   hand-roll correctly, and the textbook reason KDB+/q exists.

IMPORTANT q semantic: `update ... from t` evaluates to a NEW table -- it does
NOT mutate `t` in place unless reassigned (`t: update ... from t`) or the table
is referenced by name (`` update ... from `t ``). Every step below reassigns
explicitly for clarity.
"""

import pykx as kx
import polars as pl


def load_into_q():
    df = pl.read_parquet("data/processed/weekly_player_table.parquet")
    kx.q["weekly"] = kx.toq(df.to_pandas())
    kx.q("weekly: `player_id`season`week xasc weekly")


def add_player_features():
    # last game actual (raw, not averaged) -- for the "predict last week" baseline
    kx.q("weekly: update last_game_q: prev fantasy_points_ppr by player_id from weekly")

    # rolling 3-game form: allowed to cross season boundaries (deliberate design
    # choice, matches build_features.py) -- grouped only by player_id
    for col in ["fantasy_points_ppr", "carries", "targets", "snap_pct"]:
        kx.q(f"weekly: update {col}_rolling3g_q: 3 mavg prev {col} by player_id from weekly")

    # season-to-date average: resets each season -- grouped by player_id AND season
    kx.q("weekly: update season_avg_q: prev avgs fantasy_points_ppr by player_id, season from weekly")


def add_opponent_defense_asof():
    # ground truth: what a defense actually allowed to a position, per week
    # (the "quotes" table -- raw facts as they happened)
    kx.q("""
    defense_history: select points_allowed: sum fantasy_points_ppr
        by defense_team:opponent_team, position, season, week from weekly
    """)
    kx.q("defense_history: `defense_team`position`season`week xasc defense_history")

    # running rolling3g/season-avg of points allowed, INCLUDING each row's own
    # week (deliberately NOT using prev here) -- the lag happens exactly ONCE,
    # via the week-1 lookup key below. Lagging here too would double-shift:
    # a real bug caught by the full-dataset parity check against polars, where
    # 99.9% of rows initially differed instead of the handful expected from
    # bye-week handling alone.
    kx.q("""
    defense_history: update
        def_rolling3g_q: 3 mavg points_allowed,
        def_season_avg_q: avgs points_allowed
        by defense_team, position, season from defense_history
    """)

    # lookup key = week-1: this is what makes the join leakage-safe. aj matches
    # "<=", so joining on the CURRENT week would match that week's own outcome.
    kx.q("""
    lookup: select player_id, position, defense_team:opponent_team, season, week:week-1 from weekly
    """)
    joined = kx.q(
        "aj[`defense_team`position`season`week; lookup; defense_history]"
    )
    # aj also correctly finds the defense's most recent PRIOR game even across
    # a bye week -- an exact equi-join on week-1 would silently return null there
    kx.q["defense_asof"] = joined


def main():
    load_into_q()
    add_player_features()
    add_opponent_defense_asof()

    # pull the q-computed defense lookup back to validate against a known case
    sample = kx.q(
        'select from defense_asof where player_id=`$"00-0019596"'
    )
    print("sample opponent-defense as-of lookup (Andy Dalton / CIN QB, 2016):")
    print(sample)

    print("\nfinal weekly table shape (q):", kx.q("(count weekly; count cols weekly)"))


if __name__ == "__main__":
    main()
