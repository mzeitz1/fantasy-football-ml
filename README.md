# Fantasy Football Performance Prediction

A leakage-safe, time-aware weekly fantasy performance prediction model for RB/WR/QB/TE,
benchmarked continuously against public consensus projections over a live season, with a
point-in-time feature pipeline (KDB+/q) and an optional live sentiment adjustment layer.

This is not a tool to win your fantasy league — the model predicts week N+1 from data
through week N, so it structurally can't help with drafting. It's a benchmarked,
ongoing artifact: the model's weekly predictions are tracked against consensus
projections (e.g. FantasyPros) all season.

## Status

Phase 1 (backtest) in progress.

## Structure

- `data/raw/` — untouched pulls from nflverse (gitignored)
- `data/processed/` — cleaned/aggregated weekly stats (gitignored)
- `features/` — feature engineering code (pandas + KDB/q point-in-time layer)
- `models/` — training and backtest code
- `kdb/` — KDB+/q as-of-join point-in-time pipeline
- `notebooks/` — exploration
- `reports/` — backtest report, write-ups
- `scripts/` — data pull / pipeline entry points

## Setup

macOS only: XGBoost needs the OpenMP runtime, which isn't bundled — `brew install libomp` before installing requirements.

The KDB+/q pipeline (`kdb/`) uses PyKX, which needs a free KX license to execute q code (the pip package alone only gets "partial access"). Get a free KDB-X Community Edition license at the KX Developer Center (`developer.kx.com`) and install it with `pykx.license.install(license_string, format="STRING")`, or set the `QLIC` env var to a license file's directory.

```
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Data source

Play-by-play and weekly stats via [nflverse](https://github.com/nflverse), pulled with
`nflreadpy` (the maintained successor to `nfl_data_py`).

## Known limitations

- Stage B (live sentiment adjustment) is not backtested — evaluated prospectively only, by design.
- Does not and cannot assist with pre-season drafting, by construction of the prediction target.
