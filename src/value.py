"""Does the model have a betting edge? Tests value-betting rules against bookmaker odds.

value = model_prob * odds - 1. We test (a) the raw model, (b) the model shrunk toward the bookmaker
consensus, priced at the market-average odds and at the best available odds. Rules are chosen on early
test seasons and judged on later ones, so we don't just report the best-looking cell of a big grid."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import model

CLS = model.CLASSES
WEIGHTS = [0.0, 0.25, 0.5, 0.75, 1.0]           # weight on the model (0 = pure market, 1 = pure model)
THRESHOLDS = [0.0, 0.02, 0.04, 0.06, 0.10]


def prep(bt: pd.DataFrame) -> dict:
    bt = bt.dropna(subset=["oddsH", "oddsD", "oddsA", "bestH", "bestD", "bestA"]).reset_index(drop=True)
    return {
        "df": bt,
        "model": bt[[f"p_blend_{c}" for c in CLS]].to_numpy(),
        "market": model.market_probs(bt),
        "avg": bt[["oddsH", "oddsD", "oddsA"]].to_numpy(float),
        "best": bt[["bestH", "bestD", "bestA"]].to_numpy(float),
        "y": np.eye(3)[bt.FTR.map(model.LABEL).to_numpy()],
        "season": bt.season.to_numpy(),
    }


def bets(d: dict, w: float, th: float, price: str, rows=None):
    """Return (stake mask, profit per unit staked matrix) for one rule, flat 1-unit stakes."""
    p = w * d["model"] + (1 - w) * d["market"]
    odds = d[price]
    mask = (p * odds - 1) > th
    profit = (d["y"] * odds - 1) * mask
    if rows is not None:
        mask, profit = mask[rows], profit[rows]
    return mask, profit


def summarize(mask, profit, rng=np.random.default_rng(0), boots=1000):
    n = int(mask.sum())
    if n == 0:
        return {"bets": 0, "roi": np.nan, "lo": np.nan, "hi": np.nan}
    per_match = profit.sum(1)                    # profit per match (a match can carry >1 bet)
    stakes = mask.sum(1)
    idx = np.arange(len(per_match))
    rois = []
    for _ in range(boots):                       # resample matches to get an honest uncertainty range
        b = rng.choice(idx, len(idx))
        s = stakes[b].sum()
        rois.append(per_match[b].sum() / s if s else np.nan)
    return {"bets": n, "roi": float(profit.sum() / n), "lo": float(np.nanpercentile(rois, 2.5)), "hi": float(np.nanpercentile(rois, 97.5))}


def grid(d: dict, rows=None) -> pd.DataFrame:
    out = []
    for price in ["avg", "best"]:
        for w in WEIGHTS:
            for th in THRESHOLDS:
                mask, profit = bets(d, w, th, price, rows)
                out.append({"price": price, "w_model": w, "edge>": th, **summarize(mask, profit)})
    return pd.DataFrame(out)
