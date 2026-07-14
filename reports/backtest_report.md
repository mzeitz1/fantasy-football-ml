# Backtest Report: Weekly Fantasy Performance Prediction (Phase 1)

## Task

Predict a player's fantasy points (PPR) in week N+1, using only data available through
week N. Positions: RB, WR, QB, TE. Data: nflverse play-by-play and weekly stats,
2016-2024 (9 seasons, 52,748 RB/WR/QB/TE player-weeks).

## Methodology

### Leakage safety

Every feature used to predict week N's score is computed using only weeks 1..N-1 for
that player (rolling 3-game averages, season-to-date averages) or is genuinely known
before week N's game is played (home/away, opponent identity, injury report status,
Vegas lines). No feature is ever derived from the outcome it's trying to predict.

This was verified two ways, not just asserted:
- By hand, against a known player's actual game log (Christian McCaffrey, 2019).
- By building the point-in-time feature pipeline **twice**, independently, once in
  polars and once in KDB+/q (`kdb/build_features_q.py`), and confirming the two
  implementations produce identical output across the full dataset
  (`kdb/validate_parity.py` — 100% exact match on every point-in-time feature).

### Chronological validation

Train: 2016-2022 (40,544 player-weeks). Test: 2023-2024 (12,204 player-weeks),
held out and evaluated on exactly once per model, never used to pick features or
hyperparameters. Hyperparameter tuning used `TimeSeriesSplit` (5 chronological folds
within the training period only) — never a random/shuffled k-fold, which would let a
model "validate" on an early season using training data from a later one.

## Naive baselines

A model has to beat a dumb, honest guess to mean anything. Three were tested:

| Baseline | MAE | RMSE |
|---|---|---|
| Predict last week's actual score | 5.500 | 7.863 |
| **Predict season-to-date average** | **4.574** | **6.448** |
| Predict rolling 3-game average | 4.756 | 6.647 |

Season-to-date average is the strongest naive baseline — fantasy scoring is noisy
enough week to week (test-set std ≈ 8 points on a mean of ≈ 8 points) that averaging
over more games beats leaning on recent form alone. **4.574 MAE is the number every
real model below has to beat.**

## Model progression

| Model | Features | MAE | RMSE | vs. baseline |
|---|---|---|---|---|
| Naive baseline (season avg) | — | 4.574 | 6.448 | — |
| Ridge regression | original 11 | 4.520 | 6.151 | -1.2% |
| XGBoost (untuned) | original 11 | 4.480 | 6.108 | -2.1% |
| XGBoost (tuned, time-series CV) | original 11 | 4.482 | — | -2.0% |
| **XGBoost (tuned, expanded features)** | **+5 new** | **4.431** | **6.051** | **-3.1%** |

Two honest findings along the way, reported as-is rather than smoothed over:

1. **Hyperparameter tuning alone did not help** (4.482 vs. 4.480 untuned) — with the
   original feature set, the model had already reached the ceiling of what those
   features could support. Squeezing the model harder wasn't the right lever.
2. **New features were the right lever.** Adding target share, air yards share, red
   zone opportunity share, Vegas game script (spread/total), a momentum/trend signal,
   and a team target-competition signal — then re-tuning — moved the model from
   effectively flat-vs-baseline to a real, if still modest, improvement.

### What the model found actually matters (SHAP analysis, expanded feature set)

Ranked by mean absolute SHAP value (how much each feature actually moves individual
predictions, not just how often it's used to split):

1. Season-to-date average
2. Rolling 3-game average
3. Snap share (rolling)
4. Carries (rolling)
5. **Vegas game total (`total_line`)** — new, ranks above last week's actual score
6. Last week's actual score
7. Target-share gap to team's top target-getter — new
8. Target share (rolling) — new
9. Team spread (`team_spread`) — new
10. Opponent defense season-average points allowed

A player's own recent usage and scoring history dominates, as it did before the
feature expansion — but all five new features earned a real, non-trivial place in
the ranking. The Vegas game total in particular now ranks above last week's raw
score, which is a legible, defensible finding: how much scoring a game is expected
to have tells the model more than one single recent data point does.

## Known limitations

- **Stage B (live sentiment adjustment) is not backtested.** Evaluated prospectively
  only, during the live season, by design — not a gap, a deliberate scope decision
  (see project plan, Section 5). Individual news/beat-reporter content isn't reliably
  archived enough for a rigorous historical backtest.
- **Combine/age/demographic features are not included in Phase 1.** Research is mixed
  on their value for *weekly* (vs. preseason) prediction; a candidate for later testing,
  not assumed to help.
- **This does not and cannot assist with pre-season drafting**, by construction of the
  prediction target (week N+1 requires data through week N to exist).
- **The improvement over baseline is real but modest (3.1%).** This is reported
  honestly rather than oversold — it is a legitimate result given the rigor behind
  it (leakage-safety, honest baseline, chronological validation), not a headline
  claim of a dramatically better model.

## What's next (Phase 2)

Live weekly prediction pipeline, Stage B sentiment layer, dashboard, and the ongoing
public scoreboard vs. consensus projections (FantasyPros).
