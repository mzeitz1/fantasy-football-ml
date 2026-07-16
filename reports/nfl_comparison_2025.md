# Comparison vs. NFL.com Consensus Projections (2025 Season)

## Why this exists

The original project framing (see README) is a live scoreboard against public
consensus, evaluated as the season unfolds -- not a one-time backtest. No free
source of *historical* weekly projections exists for 2023-2024 (the season
used in the Day 4 backtest report) -- projection sites don't archive past
weeks, only the current one. 2025 turned out to be the one season NFL.com's
projections tool retains real (non-placeholder) data for, and it happens to
be a season our model has never touched in training (2016-2024) or the
original backtest (2023-2024) -- a genuinely fresh out-of-sample comparison.

## Method

- Our model: XGBoost, trained on 2016-2024 (all of it), hyperparameters tuned
  via `TimeSeriesSplit` chronological CV (see `models/predict_2025.py`)
- Consensus: scraped from `fantasy.nfl.com/research/projections`, all 18 weeks
  of 2025, QB/RB/WR/TE (`scripts/scrape_nfl_projections.py`)
- Matched by player name (normalized) + week + position; both compared against
  actual PPR fantasy points for that game
- Verified this is an apples-to-apples PPR comparison: NFL.com's mean
  projected value (6.75) closely matches the actual PPR mean (6.78) for WRs --
  a non-PPR scoring mismatch would show their projections running ~1
  pt/reception lower than that

## Result

| | MAE |
|---|---|
| Our model | 4.394 |
| NFL.com consensus | 4.084 |

**NFL.com's projections beat our model in all 18 weeks and at every position
(QB/RB/WR/TE).** This is a clean, honest negative result, not a mixed or
ambiguous one -- worth stating plainly rather than downplaying.

## Why, honestly

NFL.com's projection system almost certainly incorporates information our
model doesn't have access to: beat-reporter news, depth-chart nuance beyond
official injury designations, and likely a more mature modeling pipeline built
by a team with far more resources than an 18-feature XGBoost model assembled
over a few sessions. This isn't a surprising outcome -- it's a legitimate,
useful data point about where this project currently stands relative to
professional consensus, and a concrete target to close the gap against going
forward (e.g. the planned Stage B sentiment layer, which this comparison
predates).

## Known limitations of this comparison

- Single season (2025) -- more seasons of data would strengthen the finding,
  but no historical projection archives exist to extend it backward
- NFL.com's exact projection methodology is not public, so "why" is informed
  reasoning, not a verified mechanism
- Matched player-weeks (5,952) exclude byes/inactive players and any name-
  matching misses; not literally every player-week in 2025
