# Multi-Season Walk-Forward Backtest vs. Consensus Projections (2021-2025)

## Why this exists

A community-maintained GitHub repo, [hvpkod/NFL-Data](https://github.com/hvpkod/NFL-Data),
extracts weekly player data and projections from Fantasy.NFL.com back to 2015,
with per-week `projected/` subfolders from 2021 onward. This looked like the
historical projection archive the project's earlier research concluded didn't
exist anywhere for free -- a chance to do a real multi-season backtest instead
of the single-season (2025) comparison done previously.

## Two problems found and fixed during verification

**1. Scoring format mismatch.** The source's own point totals
(`PlayerWeekProjectedPts`, `TotalPoints`) are computed under a custom league
scoring system (the repo's README: "specific for the MML league point
system"), not standard full PPR. Verified: matched player-weeks showed the
source's totals running ~2.2 points below our nflverse-derived actual PPR
totals, correlated with reception volume (classic signature of a reduced
reception-scoring rule). **Fix:** recomputed a standard full-PPR projection
from the source's raw stat-line columns (passing/rushing/receiving yards,
TDs, receptions -- confirmed scoring-system-agnostic) instead of trusting
their point totals directly. Validated the fix by applying the same formula
to the source's *actual* (non-projected) weekly files and comparing to our
own nflverse actuals for the same games: mean absolute difference dropped
from ~2.2 to 0.115 (rounding noise) -- confirms the recompute approach is sound.

**2. Historical projection contamination (2021-2023).** After fixing #1, the
2021-2023 seasons showed implausibly good consensus accuracy (season MAE as
low as 0.123 -- a real pre-game forecast cannot get that close to a
continuous, multi-component final score). Investigation: many rows had
`consensus_projection` **exactly equal** to the actual result. Exact-match
rate by season:

| Season | Exact-match rate | Verdict |
|---|---|---|
| 2021 | 78.4% | Contaminated -- excluded |
| 2022 | 79.4% | Contaminated -- excluded |
| 2023 | 89.5% | Contaminated -- excluded |
| 2024 | 5.0% | Clean |
| 2025 | 6.8% | Clean |

The 5-7% background rate in 2024/2025 is expected (some players are
legitimately projected zero and score zero). 78-90% is not a coincidence --
the source's "projected" files for older seasons appear to have been
backfilled with actual post-game results at some point, not genuine
point-in-time forecasts. **2021-2023 are excluded from this backtest as a
result** -- reporting them would have shown a wildly exaggerated, false gap
between our model and consensus.

A smaller, known limitation independent of the above: 2021's RB export is
missing a rushing-TD column entirely (a naming/schema quirk in that year's
file), so any 2021 RB figures would additionally understate the true
consensus projection. Moot here since 2021 is already excluded.

## Method (clean seasons: 2024-2025 only)

Walk-forward: a fresh model trained only on seasons strictly before the test
season, avoiding the leakage of testing on data the model already saw as
training (our earlier 2016-2024-trained model, from the single-season 2025
comparison, cannot be reused here since it saw 2024 in training).

- train 2016-2023 &rarr; predict 2024
- train 2016-2024 &rarr; predict 2025 (consistent with the earlier single-season exercise)

## Result

| | Our model MAE | Consensus MAE |
|---|---|---|
| 2024 | 4.473 | 4.158 |
| 2025 | 4.376 | 4.123 |
| **Combined** | **4.422** | **4.139** |

**Consensus beat our model in both seasons, at every position, and in all 35
season-position-week cells checked.** This closely matches the earlier
single-season NFL.com comparison (4.394 vs. 4.084), which used an
independently-scraped source -- two different data sources agreeing is a
real cross-validation of the finding, not a fluke of one dataset.

## Honest takeaway

The gap is consistent (~0.28-0.31 MAE) and consistent across two independent
consensus sources. Combined with the earlier finding that hyperparameter
tuning and 5 new engineered features didn't meaningfully close this gap, the
most likely explanation remains what was concluded before: consensus
projection systems draw on information this model doesn't have access to
(beat-reporter news, depth-chart nuance, more mature modeling pipelines) --
not a fixable bug in this project's approach. This is precisely the gap the
planned Stage B sentiment layer was designed to help close.

## Known limitations of this backtest

- Only 2 seasons (2024-2025) survived the contamination check -- a small
  sample for a "consensus always wins" claim, even though it's consistent
  across positions and an independent data source
- Consensus projection methodology is not public; "why" is informed
  reasoning, not a verified mechanism
- Matched player-weeks only (name-normalized join); excludes any players
  missed by name-matching or absent from one source
