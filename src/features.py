"""Leak-free feature engineering. One stateful pass over matches in date order: features for a match are
built from state *before* it, then the state is updated. The same code builds features for upcoming
fixtures, so training and prediction can't drift apart."""
from __future__ import annotations

from collections import defaultdict, deque

import numpy as np
import pandas as pd

K = 20.0            # Elo k-factor
HOME_ADV = 65.0     # Elo points of home advantage
START_ELO = 1500.0
NEW_TEAM_ELO = 1440.0
SEASON_REGRESS = 0.25
FORM_N = 5

FEATURES = [
    "elo_h", "elo_a", "elo_diff",
    "form_h", "form_a", "gf_h", "gf_a", "ga_h", "ga_a",
    "sot_h", "sot_a", "sota_h", "sota_a",
    "homeform_h", "awayform_a", "rest_h", "rest_a",
]


def _mean(d, default=np.nan):
    return float(np.mean(d)) if len(d) else default


class State:
    def __init__(self):
        self.elo = defaultdict(lambda: START_ELO)
        self.seen = set()
        self.hist = defaultdict(lambda: deque(maxlen=FORM_N))        # (pts, gf, ga, sot, sot_against)
        self.home_pts = defaultdict(lambda: deque(maxlen=FORM_N))
        self.away_pts = defaultdict(lambda: deque(maxlen=FORM_N))
        self.last_date = {}
        self.season = None

    def new_season(self, season, teams_in_season):
        if self.season is not None:
            for t in list(self.elo):
                self.elo[t] = START_ELO + (self.elo[t] - START_ELO) * (1 - SEASON_REGRESS)
        for t in teams_in_season:
            if t not in self.seen and self.season is not None:
                self.elo[t] = NEW_TEAM_ELO          # promoted side with no top-flight history
            self.seen.add(t)
        self.season = season

    def features(self, home, away, date):
        h, a = self.hist[home], self.hist[away]
        rest = lambda t: min((date - self.last_date[t]).days, 14) if t in self.last_date else np.nan
        eh, ea = self.elo[home], self.elo[away]
        return {
            "elo_h": eh, "elo_a": ea, "elo_diff": eh + HOME_ADV - ea,
            "form_h": _mean([x[0] for x in h]), "form_a": _mean([x[0] for x in a]),
            "gf_h": _mean([x[1] for x in h]), "gf_a": _mean([x[1] for x in a]),
            "ga_h": _mean([x[2] for x in h]), "ga_a": _mean([x[2] for x in a]),
            "sot_h": _mean([x[3] for x in h]), "sot_a": _mean([x[3] for x in a]),
            "sota_h": _mean([x[4] for x in h]), "sota_a": _mean([x[4] for x in a]),
            "homeform_h": _mean(self.home_pts[home]), "awayform_a": _mean(self.away_pts[away]),
            "rest_h": rest(home), "rest_a": rest(away),
        }

    def update(self, r):
        hg, ag = r.FTHG, r.FTAG
        res = r.FTR
        ph, pa = {"H": (3, 0), "D": (1, 1), "A": (0, 3)}[res]
        eh, ea = self.elo[r.HomeTeam], self.elo[r.AwayTeam]
        exp_h = 1 / (1 + 10 ** (-(eh + HOME_ADV - ea) / 400))
        s_h = {"H": 1.0, "D": 0.5, "A": 0.0}[res]
        mult = np.log(abs(hg - ag) + 1) + 1          # bigger wins move ratings more
        delta = K * mult * (s_h - exp_h)
        self.elo[r.HomeTeam], self.elo[r.AwayTeam] = eh + delta, ea - delta
        hst = r.HST if pd.notna(r.HST) else np.nan
        ast = r.AST if pd.notna(r.AST) else np.nan
        self.hist[r.HomeTeam].append((ph, hg, ag, hst, ast))
        self.hist[r.AwayTeam].append((pa, ag, hg, ast, hst))
        self.home_pts[r.HomeTeam].append(ph)
        self.away_pts[r.AwayTeam].append(pa)
        self.last_date[r.HomeTeam] = self.last_date[r.AwayTeam] = r.Date


def build(matches: pd.DataFrame) -> tuple[pd.DataFrame, State]:
    """Feature table for every completed match + the final state (for predicting upcoming fixtures)."""
    st, rows = State(), []
    for season, g in matches.groupby("season", sort=True):
        st.new_season(season, set(g.HomeTeam) | set(g.AwayTeam))
        for i, r in enumerate(g.itertuples(index=False)):
            f = st.features(r.HomeTeam, r.AwayTeam, r.Date)
            f.update(season=season, Date=r.Date, HomeTeam=r.HomeTeam, AwayTeam=r.AwayTeam, FTR=r.FTR,
                     oddsH=r.oddsH, oddsD=r.oddsD, oddsA=r.oddsA, matchno=i)
            rows.append(f)
            st.update(r)
    return pd.DataFrame(rows), st


def upcoming_features(st: State, fixtures: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for r in fixtures.itertuples(index=False):
        f = st.features(r.HomeTeam, r.AwayTeam, r.Date)
        f.update(Date=r.Date, HomeTeam=r.HomeTeam, AwayTeam=r.AwayTeam, gameweek=r.gameweek)
        rows.append(f)
    return pd.DataFrame(rows)
