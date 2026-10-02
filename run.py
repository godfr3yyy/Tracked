"""EPL match outcome predictor.

  python run.py backtest   walk-forward test on past seasons vs. bookmaker odds + naive baselines
  python run.py predict    refresh data, retrain, predict next gameweek, log predictions
  python run.py dashboard  refresh + predict + rebuild docs/index.html
  python run.py value      does the model beat bookmaker odds? (value-bet backtest)
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


def cmd_value():
    from src import value
    matches = data.load_matches()
    feat, _ = features.build(matches)
    done = [s for s in sorted(feat.season.unique()) if (feat.season == s).sum() >= 370]
    tests = done[-5:]
    d = value.prep(model.walk_forward(feat, tests))
    pd.set_option("display.width", 160)
    n = len(d["df"])
    print(f"{n} matches with full odds, seasons {tests[0]}-{tests[-1]}. Flat 1-unit stakes. ROI = profit / stakes.\n")
    full = value.grid(d)
    fmt = lambda df: df.assign(roi=(df.roi * 100).round(1), lo=(df.lo * 100).round(1), hi=(df.hi * 100).round(1)).rename(columns={"roi": "ROI%", "lo": "ci_lo%", "hi": "ci_hi%"})
    for price in ["avg", "best"]:
        print(f"=== priced at {'MARKET-AVERAGE' if price == 'avg' else 'BEST-AVAILABLE'} odds (all 5 seasons) ===")
        t = full[full.price == price].pivot(index="w_model", columns="edge>", values="roi") * 100
        print(t.round(1).to_string(), "\n")
    # honest check: pick the rule on the first 3 test seasons, judge it on the last 2
    sea = d["season"]
    sel, hold = np.isin(sea, tests[:3]), np.isin(sea, tests[3:])
    sg = value.grid(d, sel)
    sg = sg[sg.bets >= 100].sort_values("roi", ascending=False)
    print("=== choose-then-test: best rule on first 3 seasons -> result on last 2 (unseen) ===")
    for _, r in sg.head(3).iterrows():
        mask, profit = value.bets(d, r["w_model"], r["edge>"], r["price"], hold)
        res = value.summarize(mask, profit)
        print(f"rule {r['price']} w={r['w_model']} edge>{r['edge>']:.0%}: chosen-set ROI {r.roi:+.1%} ({int(r.bets)} bets)  ->  holdout ROI {res['roi']:+.1%} on {res['bets']} bets, 95% range {res['lo']:+.1%}..{res['hi']:+.1%}")
    full.to_csv(data.ROOT / "predictions" / "value_grid.csv", index=False)


def cmd_experiment():
    """Do extra signals (xG, availability) improve accuracy AND turn the value-bet ROI positive?"""
    from src import extra, value
    matches = data.load_matches()
    feat, _ = features.build(matches)
    f = extra.add_extra(feat, extra.load_xg(), extra.load_availability())
    done = [s for s in sorted(f.season.unique()) if (f.season == s).sum() >= 370]
    tests = done[-5:]
    B, X, A = features.FEATURES, extra.XG_FEATURES, extra.AVAIL_FEATURES
    variants = {"baseline (all history)": (B, 0), "baseline (2016+ only)": (B, 2016),
                "+ xG (2016+)": (B + X, 2016), "+ xG + availability (2016+)": (B + X + A, 2016)}
    rules = [("best", 1.0, 0.04), ("best", 0.5, 0.02), ("best", 1.0, 0.0)]
    rows = []
    for name, (cols, frm) in variants.items():
        bt = model.walk_forward(f, tests, cols, frm)
        d = value.prep(bt)
        y = d["df"].FTR
        sc = model.score(y, d["model"])
        mk = model.score(y, d["market"])
        row = {"variant": name, "n": sc["n"], "acc": sc["accuracy"], "logloss": sc["log_loss"]}
        for k, (price, w, th) in enumerate(rules):
            mask, profit = value.bets(d, w, th, price)
            r = value.summarize(mask, profit)
            if k == 0:
                row.update(roi=r["roi"], lo=r["lo"], hi=r["hi"], bets=r["bets"])
            row[f"ROI best w={w} >{th:.0%}"] = f"{r['roi']:+.1%} ({r['bets']} bets, {r['lo']:+.0%}..{r['hi']:+.0%})"
        rows.append(row)
    pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 60)
    out = pd.DataFrame(rows)
    print(f"Test seasons {tests[0]}-{tests[-1]} (same matches for every variant). Bookmakers: acc {mk['accuracy']:.3f}  logloss {mk['log_loss']:.4f}\n")
    print(out.round(4).to_string(index=False))
    out.to_csv(data.ROOT / "predictions" / "experiment.csv", index=False)


def cmd_dashboard():
    from src import dashboard
    cmd_predict()
    print("\nWrote", dashboard.write(LOG))


if __name__ == "__main__":
    {"backtest": cmd_backtest, "predict": cmd_predict, "evaluate": cmd_evaluate, "dashboard": cmd_dashboard, "value": cmd_value, "experiment": cmd_experiment}.get(
        sys.argv[1] if len(sys.argv) > 1 else "", lambda: print(__doc__))()
