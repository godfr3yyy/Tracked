"""Extra signals for the 'can we find an edge' experiment.

  xG          match expected goals from Understat (2014/15+)  -> rolling attack/defence quality
  availability proxy from the FPL archive (2016/17+): share of a team's regular-starter value that missed
              its PREVIOUS match. Only uses information available before kickoff (no lineup leakage).
"""
from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from .data import RAW, ROOT, current_season_start

UNDERSTAT = "https://understat.com/getLeagueData/EPL/{y}"
VAASTAV = "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data/{s}/{f}"
U_TO_FD = {"Manchester City": "Man City", "Manchester United": "Man United", "Newcastle United": "Newcastle",
           "Nottingham Forest": "Nott'm Forest", "Queens Park Rangers": "QPR", "West Bromwich Albion": "West Brom",
           "Wolverhampton Wanderers": "Wolves", "Sheffield United": "Sheffield United", "Leeds": "Leeds"}
V_TO_FD = {"Man Utd": "Man United", "Spurs": "Tottenham", "Sheffield Utd": "Sheffield United", "Leeds": "Leeds",
           "Coventry City": "Coventry", "Hull City": "Hull", "Ipswich Town": "Ipswich"}
FIRST_XG, FIRST_FPL = 2014, 2016
ROLL = 5


def _get(url, **kw):
    r = requests.get(url, timeout=60, **kw)
    r.raise_for_status()
    return r


def load_xg(refresh_current=True) -> pd.DataFrame:
    """One row per played match: season, HomeTeam, AwayTeam, xg_h, xg_a."""
    RAW.mkdir(parents=True, exist_ok=True)
    cur, rows = current_season_start(), []
    for y in range(FIRST_XG, cur + 1):
        path = RAW / f"understat_{y}.json"
        if not path.exists() or (refresh_current and y == cur):
            path.write_text(_get(UNDERSTAT.format(y=y), headers={"X-Requested-With": "XMLHttpRequest", "User-Agent": "Mozilla/5.0"}).text)
        for m in json.loads(path.read_text())["dates"]:
            if m["isResult"]:
                rows.append({"season": y, "HomeTeam": U_TO_FD.get(m["h"]["title"], m["h"]["title"]),
                             "AwayTeam": U_TO_FD.get(m["a"]["title"], m["a"]["title"]),
                             "xg_h": float(m["xG"]["h"]), "xg_a": float(m["xG"]["a"])})
    return pd.DataFrame(rows)


def _season_str(y):
    return f"{y}-{(y + 1) % 100:02d}"


def _fpl_season(y, refresh) -> pd.DataFrame | None:
    path = RAW / f"fpl_{y}.csv"
    if not path.exists() or refresh:
        s = _season_str(y)
        try:
            gw = _get(VAASTAV.format(s=s, f="gws/merged_gw.csv")).content
        except requests.HTTPError:
            return None
        df = pd.read_csv(io.BytesIO(gw), encoding="latin-1", on_bad_lines="skip")
        if "team" not in df.columns:                    # older seasons: attach team via players_raw + master team list
            pr = pd.read_csv(io.BytesIO(_get(VAASTAV.format(s=s, f="players_raw.csv")).content), encoding="latin-1")
            tl = pd.read_csv(io.BytesIO(_get("https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data/master_team_list.csv").content))
            names = tl[tl.season == s].set_index("team")["team_name"]
            df["team"] = df["element"].map(pr.set_index("id")["team"].map(names))
        df[["element", "team", "kickoff_time", "minutes", "value"]].to_csv(path, index=False)
    return pd.read_csv(path)


def load_availability(refresh_current=True) -> pd.DataFrame:
    """Per team, per match date: share of regular-starter value that missed THAT match (post-match fact).
    The caller attaches it to the team's NEXT fixture, so features only use the past."""
    cur, out = current_season_start(), []
    for y in range(FIRST_FPL, cur + 1):
        df = _fpl_season(y, refresh=refresh_current and y == cur)
        if df is None or df.empty:
            continue
        df = df.dropna(subset=["team", "kickoff_time"]).copy()
        df["team"] = df["team"].replace(V_TO_FD)
        df["date"] = pd.to_datetime(df["kickoff_time"]).dt.tz_localize(None).dt.normalize()
        df = df.groupby(["team", "date", "element"], as_index=False).agg(minutes=("minutes", "sum"), value=("value", "max"))
        for team, g in df.groupby("team"):
            piv = g.pivot(index="date", columns="element", values="minutes").sort_index().fillna(0)
            val = g.groupby("element")["value"].max()
            prior_avg = piv.shift(1).rolling(ROLL, min_periods=2).mean()
            for date, row in piv.iterrows():
                reg = prior_avg.loc[date]
                regulars = reg.index[reg >= 55]
                if len(regulars) < 6:
                    continue
                v = val.reindex(regulars)
                missed = v[row.reindex(regulars) == 0]
                out.append({"season": y, "team": team, "date": date, "abs_val": float(missed.sum() / v.sum()), "abs_n": int(len(missed))})
    return pd.DataFrame(out)


def add_extra(feat: pd.DataFrame, xg: pd.DataFrame, avail: pd.DataFrame) -> pd.DataFrame:
    """Attach xG-form and availability features to the baseline feature table (same leak-free principle)."""
    f = feat.merge(xg, on=["season", "HomeTeam", "AwayTeam"], how="left")
    long = pd.concat([
        f[["season", "Date", "HomeTeam", "xg_h", "xg_a"]].rename(columns={"HomeTeam": "team", "xg_h": "xf", "xg_a": "xa"}),
        f[["season", "Date", "AwayTeam", "xg_a", "xg_h"]].rename(columns={"AwayTeam": "team", "xg_a": "xf", "xg_h": "xa"}),
    ]).dropna(subset=["xf"]).sort_values("Date")
    long["xf_roll"] = long.groupby("team")["xf"].transform(lambda s: s.shift(1).rolling(ROLL, min_periods=3).mean())
    long["xa_roll"] = long.groupby("team")["xa"].transform(lambda s: s.shift(1).rolling(ROLL, min_periods=3).mean())
    key = long.set_index(["team", "Date"])[["xf_roll", "xa_roll"]]
    for side, col in [("h", "HomeTeam"), ("a", "AwayTeam")]:
        j = f[[col, "Date"]].merge(key.reset_index(), left_on=[col, "Date"], right_on=["team", "Date"], how="left")
        f[f"xgf_{side}"], f[f"xga_{side}"] = j["xf_roll"].to_numpy(), j["xa_roll"].to_numpy()
    f["xg_diff_h"], f["xg_diff_a"] = f.xgf_h - f.xga_h, f.xgf_a - f.xga_a
    # availability: latest record strictly before this match's date
    f = f.sort_values("Date")
    av = avail.sort_values("date")[["team", "date", "abs_val", "abs_n"]]
    for side, col in [("h", "HomeTeam"), ("a", "AwayTeam")]:
        m = pd.merge_asof(f[["Date", col]].reset_index().sort_values("Date"), av.rename(columns={"abs_val": f"abs_val_{side}", "abs_n": f"abs_n_{side}"}),
                          left_on="Date", right_on="date", left_by=col, right_by="team", allow_exact_matches=False, tolerance=pd.Timedelta(days=14)).set_index("index")
        f[f"abs_val_{side}"], f[f"abs_n_{side}"] = m[f"abs_val_{side}"], m[f"abs_n_{side}"]
    return f.sort_index()


XG_FEATURES = ["xgf_h", "xga_h", "xgf_a", "xga_a", "xg_diff_h", "xg_diff_a"]
AVAIL_FEATURES = ["abs_val_h", "abs_n_h", "abs_val_a", "abs_n_a"]
