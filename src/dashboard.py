"""Builds docs/index.html: a single self-contained page (data inlined, no server, no external libraries)."""
from __future__ import annotations

import json
import math
from datetime import datetime

import numpy as np
import pandas as pd

from . import data, features, model, value

TEMPLATE = data.ROOT / "src" / "dashboard_template.html"
OUT = data.ROOT / "docs" / "index.html"
CLS = model.CLASSES


def _label(season: int) -> str:
    return f"{season}/{(season + 1) % 100:02d}"


def _clean(o):
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if (math.isnan(o) or math.isinf(o)) else float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (pd.Timestamp, datetime)):
        return o.strftime("%Y-%m-%d")
    return o


def _pcols(name):
    return [f"p_{name}_{c}" for c in CLS]


def build_payload(log_path) -> dict:
    matches = data.load_matches(refresh_current=True)
    feat, state = features.build(matches)
    complete = [s for s in sorted(feat.season.unique()) if (feat.season == s).sum() >= 370]
    tests = complete[-5:]
    bt = model.walk_forward(feat, tests).dropna(subset=["oddsH", "oddsD", "oddsA"]).copy()
    mk = model.market_probs(bt)
    bt[[f"p_market_{c}" for c in CLS]] = mk
    y = bt.FTR.map(model.LABEL).to_numpy()
    base = feat[feat.season < tests[0]].FTR.value_counts(normalize=True).reindex(CLS).to_numpy()
    base_acc = float((np.full(len(bt), base.argmax()) == y).mean())

    def acc(d, name):
        return float((d[_pcols(name)].to_numpy().argmax(1) == d.FTR.map(model.LABEL).to_numpy()).mean())

    seasons = [{"season": _label(s), **{n: acc(bt[bt.season == s], n) for n in ["logreg", "xgboost", "blend", "market"]}}
               for s in tests]
    bt["stage"] = np.minimum(bt.matchno // 95, 3)
    stage_names = ["Matches 1–95", "96–190", "191–285", "286–380"]
    stages = [{"label": stage_names[k], "n": int(len(d)), "blend": acc(d, "blend"), "market": acc(d, "market")}
              for k, d in bt.groupby("stage")]

    pooled_p = bt[_pcols("blend")].to_numpy().ravel()
    pooled_y = np.eye(3)[y].ravel()
    edges = np.linspace(0, 0.8, 9)
    calib = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (pooled_p >= lo) & (pooled_p < hi)
        if m.sum() >= 30:
            calib.append({"p": float(pooled_p[m].mean()), "freq": float(pooled_y[m].mean()), "n": int(m.sum())})

    # ---- betting-edge test (value-bet backtest vs bookmaker odds)
    d = value.prep(bt)
    edge_rows = []
    for label, price, w, th in [("Model · market-average odds · any value", "avg", 1.0, 0.0),
                                ("Model · best odds · edge > 4%", "best", 1.0, 0.04),
                                ("50/50 blend with market · best odds · edge > 2%", "best", 0.5, 0.02)]:
        mask, profit = value.bets(d, w, th, price)
        edge_rows.append({"label": label, **value.summarize(mask, profit)})
    exp_path = data.ROOT / "predictions" / "experiment.csv"
    if exp_path.exists():
        ex = pd.read_csv(exp_path).set_index("variant")
        for v, label in [("+ xG (2016+)", "Model + xG · best odds · edge > 4%"),
                         ("+ xG + availability (2016+)", "Model + xG + player availability · best odds · edge > 4%")]:
            if v in ex.index:
                r = ex.loc[v]
                edge_rows.append({"label": label, "bets": int(r.bets), "roi": float(r.roi), "lo": float(r.lo), "hi": float(r.hi)})
    margin = float((1 / d["avg"]).sum(1).mean() - 1)
    market_ll = model.score(d["df"].FTR, d["market"])["log_loss"]
    model_ll = model.score(d["df"].FTR, d["model"])["log_loss"]
    edge = {"rows": edge_rows, "margin": margin, "market_ll": market_ll, "model_ll": model_ll}

    # ---- upcoming + live tracker from the prediction log
    upcoming, live = [], {"n": 0, "blend_acc": None, "market_acc": None, "rows": [], "recent": []}
    if log_path.exists():
        log = pd.read_csv(log_path, parse_dates=["Date", "made_at"])
        latest = log[log.made_at == log.made_at.max()]
        for r in latest.itertuples():
            p = [getattr(r, f"p_blend_{c}") for c in CLS]
            upcoming.append({"date": r.Date, "home": r.HomeTeam, "away": r.AwayTeam, "gw": r.gameweek,
                             "ph": p[0], "pd": p[1], "pa": p[2], "pick": CLS[int(np.argmax(p))],
                             "out_h": getattr(r, "out_h", None), "out_a": getattr(r, "out_a", None)})
        res = matches[["Date", "HomeTeam", "AwayTeam", "FTR", "oddsH", "oddsD", "oddsA"]]
        j = log.merge(res, on=["Date", "HomeTeam", "AwayTeam"], how="inner").sort_values("Date")
        if len(j):
            yi = j.FTR.map(model.LABEL).to_numpy()
            pred = j[_pcols("blend")].to_numpy().argmax(1)
            ok = pred == yi
            live["n"] = int(len(j))
            live["blend_acc"] = float(ok.mean())
            jm = j.dropna(subset=["oddsH"])
            if len(jm):
                live["market_acc"] = float((model.market_probs(jm).argmax(1) == jm.FTR.map(model.LABEL).to_numpy()).mean())
            live["rows"] = [{"i": i + 1, "acc": float(v)} for i, v in enumerate(pd.Series(ok).expanding().mean())]
            live["recent"] = [{"date": r.Date, "home": r.HomeTeam, "away": r.AwayTeam, "pred": CLS[p], "actual": r.FTR, "ok": bool(o)}
                              for r, p, o in zip(j.tail(10).itertuples(), pred[-10:], ok[-10:])][::-1]

    # ---- team strength (current-season clubs)
    cur = matches[matches.season == matches.season.max()]
    teams = sorted(set(cur.HomeTeam) | set(cur.AwayTeam))
    elo = []
    for t in teams:
        form = "".join({3: "W", 1: "D", 0: "L"}[x[0]] for x in state.hist[t])
        elo.append({"team": t, "elo": float(state.elo[t]), "form": form})
    elo.sort(key=lambda d: -d["elo"])

    return _clean({
        "generated": datetime.now().astimezone().strftime("%d %b %Y, %H:%M %Z"),
        "season": _label(int(matches.season.max())),
        "played": int(len(cur)),
        "headline": {"blend": acc(bt, "blend"), "logreg": acc(bt, "logreg"), "xgboost": acc(bt, "xgboost"),
                     "market": acc(bt, "market"), "base": base_acc, "n": int(len(bt)),
                     "range": f"{_label(tests[0])} – {_label(tests[-1])}"},
        "seasons": seasons, "stages": stages, "calib": calib,
        "upcoming": upcoming, "live": live, "elo": elo, "edge": edge,
    })


def write(log_path) -> str:
    payload = build_payload(log_path)
    html = TEMPLATE.read_text().replace("/*__DATA__*/null", json.dumps(payload, allow_nan=False))
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(html)
    return str(OUT)
