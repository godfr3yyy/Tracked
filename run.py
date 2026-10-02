"""EPL match outcome predictor.

  python run.py backtest   walk-forward test on past seasons vs. bookmaker odds + naive baselines
  python run.py predict    refresh data, retrain, predict next gameweek, log predictions
  python run.py dashboard  refresh + predict + rebuild docs/index.html
  python run.py evaluate   score logged predictions against real results (accuracy over the season)
"""
from __future__ import annotations

import sys
import warnings

import numpy as np
import pandas as pd

from src import data, features, model

warnings.filterwarnings("ignore")
LOG = data.ROOT / "predictions" / "log.csv"
PCOLS = {n: [f"p_{n}_{c}" for c in model.CLASSES] for n in ["logreg", "xgboost", "blend"]}


def cmd_backtest():
    matches = data.load_matches()
    feat, _ = features.build(matches)
    last = int(feat.season.max())
    done = [s for s in sorted(feat.season.unique()) if (feat.season == s).sum() >= 370]   # complete seasons
    tests = done[-5:]
    bt = model.walk_forward(feat, tests)
    bt = bt.dropna(subset=["oddsH", "oddsD", "oddsA"])
    rows = []
    for s in tests + ["ALL"]:
        d = bt if s == "ALL" else bt[bt.season == s]
        base = np.tile(feat[feat.season < tests[0]].FTR.value_counts(normalize=True).reindex(model.CLASSES).to_numpy(), (len(d), 1))
        res = {"always home/base rates": model.score(d.FTR, base), "bookmakers": model.score(d.FTR, model.market_probs(d))}
        for n in PCOLS:
            res[n] = model.score(d.FTR, d[PCOLS[n]].to_numpy())
        for k, v in res.items():
            rows.append({"season": s, "model": k, **v})
    out = pd.DataFrame(rows)
    pd.set_option("display.width", 140)
    print(f"Walk-forward backtest (retrained before each season; current season {last} excluded as incomplete)\n")
    print(out.pivot(index="model", columns="season", values="accuracy").round(3).to_string())
    print("\nLog loss (lower = better) over ALL test matches:")
    print(out[out.season == "ALL"].set_index("model")[["n", "accuracy", "log_loss", "brier"]].round(4).to_string())
    out.to_csv(data.ROOT / "predictions" / "backtest.csv", index=False)


def cmd_predict():
    matches = data.load_matches(refresh_current=True)
    feat, state = features.build(matches)
    models = model.fit_all(feat[feat.season > feat.season.min()])
    fx = data.fetch_upcoming()
    X = features.upcoming_features(state, fx)
    probs = model.predict_proba(models, X)
    out = fx.copy()
    for n in PCOLS:
        out[PCOLS[n]] = probs[n]
    out["made_at"] = pd.Timestamp.now().floor("s")
    try:
        inj = data.fetch_injury_burden().set_index("team")
        out["out_h"] = out.HomeTeam.map(inj.players_out)
        out["out_a"] = out.AwayTeam.map(inj.players_out)
    except Exception as e:                      # injuries are context only; never block predictions
        print("injury fetch skipped:", e)
    unknown = (set(out.HomeTeam) | set(out.AwayTeam)) - set(matches.HomeTeam)
    if unknown:
        sys.exit(f"Team names in fixtures but not in results data - add them to FPL_TO_FD in src/data.py: {unknown}")

    LOG.parent.mkdir(exist_ok=True)
    if LOG.exists():
        old = pd.read_csv(LOG, parse_dates=["Date", "made_at"])
        key = ["Date", "HomeTeam", "AwayTeam"]
        keep_old = old.merge(out[key], on=key, how="left", indicator=True)
        frozen = keep_old[(keep_old._merge == "left_only") | (keep_old.Date < pd.Timestamp.now().normalize())].drop(columns="_merge")
        out = pd.concat([frozen, out], ignore_index=True)       # re-predicting a future fixture overwrites; kicked-off ones stay frozen
    out.to_csv(LOG, index=False)

    show = out[out.made_at == out.made_at.max()]
    print(f"\nGameweek {int(show.gameweek.iloc[0]) if len(show) else '?'} predictions (blend of logreg + xgboost)\n")
    for r in show.itertuples():
        p = [getattr(r, f"p_blend_{c}") for c in model.CLASSES]
        pick = ["home win", "draw", "away win"][int(np.argmax(p))]
        print(f"{r.Date:%a %d %b}  {r.HomeTeam:>14} v {r.AwayTeam:<14} H {p[0]:.0%}  D {p[1]:.0%}  A {p[2]:.0%}   -> {pick}")


def cmd_evaluate():
    if not LOG.exists():
        sys.exit("No predictions logged yet. Run: python run.py predict")
    log = pd.read_csv(LOG, parse_dates=["Date", "made_at"])
    res = data.load_matches(refresh_current=True)[["Date", "HomeTeam", "AwayTeam", "FTR", "oddsH", "oddsD", "oddsA"]]
    j = log.merge(res, on=["Date", "HomeTeam", "AwayTeam"], how="inner").sort_values("Date")
    print(f"{len(log)} predictions logged, {len(j)} already played.")
    if j.empty:
        return
    for n in PCOLS:
        print(f"{n:>8}:", {k: round(v, 3) for k, v in model.score(j.FTR, j[PCOLS[n]].to_numpy()).items()})
    jj = j.dropna(subset=["oddsH"])
    print(f"{'market':>8}:", {k: round(v, 3) for k, v in model.score(jj.FTR, model.market_probs(jj)).items()})
    j["correct"] = (j[PCOLS["blend"]].to_numpy().argmax(1) == j.FTR.map(model.LABEL).to_numpy())
    j["running_acc"] = j.correct.expanding().mean()
    j[["Date", "HomeTeam", "AwayTeam", "FTR", "correct", "running_acc"]].to_csv(data.ROOT / "predictions" / "running_accuracy.csv", index=False)
    print("\nRunning accuracy (blend):", f"{j.running_acc.iloc[-1]:.1%} over {len(j)} matches")


def cmd_dashboard():
    from src import dashboard
    cmd_predict()
    print("\nWrote", dashboard.write(LOG))


if __name__ == "__main__":
    {"backtest": cmd_backtest, "predict": cmd_predict, "evaluate": cmd_evaluate, "dashboard": cmd_dashboard}.get(
        sys.argv[1] if len(sys.argv) > 1 else "", lambda: print(__doc__))()
