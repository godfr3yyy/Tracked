# EPL Match Outcome Predictor

Predicts win / draw / loss probabilities for upcoming Premier League matches (all 20 teams) and tracks its own accuracy through the season.

## How it works
- **Data (auto-refreshing, no manual downloads):** results + bookmaker odds for every season since 2000/01 from football-data.co.uk; upcoming fixtures and injury flags from the official Fantasy Premier League API. The current season is re-pulled on every run.
- **Features** (all computed from matches *before* kickoff): Elo rating (with home advantage, season regression, promoted-team handling), rolling 5-match form, goals for/against, shots on target for/against, home-only / away-only form, rest days.
- **Models:** multinomial logistic regression (baseline) and XGBoost, plus a 50/50 blend.
- **Honest evaluation:** walk-forward backtest (retrain before each season, predict it blind), compared against bookmaker odds (overround removed) and a base-rate baseline, scored by accuracy, log loss and Brier score.
- **Live tracking:** every prediction is logged before kickoff and frozen once the match starts; `evaluate` scores the log against real results.

## Usage
    python3 -m venv .venv && .venv/bin/pip install -r requirements.txt   # macOS: brew install libomp for xgboost
    .venv/bin/python run.py backtest    # historical walk-forward results
    .venv/bin/python run.py predict     # refresh data, retrain, predict next gameweek
    .venv/bin/python run.py dashboard   # everything above + rebuild docs/index.html (the web dashboard)
    .venv/bin/python run.py evaluate    # accuracy of logged predictions vs actual results

## Backtest (2021/22 – 2025/26, 1,900 matches, never seen in training)
| model | accuracy | log loss |
|---|---|---|
| base rates | 44.2% | 1.070 |
| logistic regression | 53.7% | 0.974 |
| XGBoost | 53.6% | 0.984 |
| blend | 53.7% | 0.977 |
| bookmakers | 55.3% | 0.959 |

## Limitations
- Injuries: the FPL API only gives *current* availability, so there's no historical injury data to train on. Daily snapshots are saved to `data/injuries/` and shown as context; they can become a model feature once enough history accumulates.
- Bookmakers still win. That is the realistic ceiling for a public-data model, not a bug.

## Dashboard
`run.py dashboard` writes `docs/index.html`: one self-contained page (data inlined, no server, no external libraries) with next-gameweek predictions, accuracy by point in the season, a live tracker, team Elo ratings and a calibration chart. Open it locally, or host `docs/` free on GitHub Pages (Settings > Pages > Deploy from branch > /docs). Re-run the command (or schedule it) to refresh.

## Automatic refresh
`.github/workflows/refresh.yml` runs daily at 06:00 UTC (and on demand from the Actions tab). It re-pulls data, logs predictions before kickoff, scores finished matches, rebuilds `docs/index.html`, and commits `docs/`, `predictions/log.csv` and the injury snapshots back to the repo. Needs Settings > Actions > General > Workflow permissions set to "Read and write".
