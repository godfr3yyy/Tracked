"""Data layer: historical results (football-data.co.uk) + upcoming fixtures & injuries (official FPL API)."""
from __future__ import annotations

import io
from datetime import date
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
INJ = ROOT / "data" / "injuries"
FD_URL = "https://www.football-data.co.uk/mmz4281/{code}/E0.csv"
FPL = "https://fantasy.premierleague.com/api/"
FIRST_SEASON = 2000

# FPL team names -> football-data.co.uk names
FPL_TO_FD = {"Man Utd": "Man United", "Spurs": "Tottenham", "Coventry City": "Coventry",
             "Hull City": "Hull", "Ipswich Town": "Ipswich"}

KEEP = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR", "HS", "AS", "HST", "AST",
        "B365H", "B365D", "B365A", "AvgH", "AvgD", "AvgA"]


def current_season_start(today: date | None = None) -> int:
    today = today or date.today()
    return today.year if today.month >= 7 else today.year - 1


def season_code(start: int) -> str:
    return f"{start % 100:02d}{(start + 1) % 100:02d}"


def _fetch_season(start: int, refresh: bool) -> pd.DataFrame | None:
    path = RAW / f"E0_{season_code(start)}.csv"
    if refresh or not path.exists():
        r = requests.get(FD_URL.format(code=season_code(start)), timeout=30)
        if r.status_code != 200 or len(r.content) < 200:
            return None
        path.write_bytes(r.content)
    try:
        df = pd.read_csv(path, encoding="utf-8-sig", on_bad_lines="skip")
    except UnicodeDecodeError:                  # a few older season files are latin-1
        df = pd.read_csv(path, encoding="latin-1", on_bad_lines="skip")
    df = df.dropna(subset=["HomeTeam", "AwayTeam", "FTR"])
    for c in KEEP:
        if c not in df.columns:
            df[c] = pd.NA
    df = df[KEEP].copy()
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, format="mixed")
    df["season"] = start
    return df


def load_matches(refresh_current: bool = True) -> pd.DataFrame:
    """All completed Premier League matches since 2000/01, oldest first. Past seasons are cached;
    the current season is re-downloaded each call so results stay fresh."""
    RAW.mkdir(parents=True, exist_ok=True)
    cur = current_season_start()
    frames = []
    for s in range(FIRST_SEASON, cur + 1):
        f = _fetch_season(s, refresh=refresh_current and s == cur)
        if f is not None:
            frames.append(f)
    df = pd.concat(frames, ignore_index=True)
    # one set of odds columns: prefer market average, fall back to Bet365
    for o in "HDA":
        df[f"odds{o}"] = pd.to_numeric(df[f"Avg{o}"], errors="coerce").fillna(pd.to_numeric(df[f"B365{o}"], errors="coerce"))
    return df.sort_values("Date", kind="stable").reset_index(drop=True)


def _fpl_get(endpoint: str):
    r = requests.get(FPL + endpoint, timeout=30)
    r.raise_for_status()
    return r.json()


def fetch_upcoming(max_gameweeks: int = 1) -> pd.DataFrame:
    """Next unplayed Premier League fixtures from the FPL API (all teams, next `max_gameweeks` gameweeks)."""
    boot = _fpl_get("bootstrap-static/")
    names = {t["id"]: FPL_TO_FD.get(t["name"], t["name"]) for t in boot["teams"]}
    fx = pd.DataFrame(_fpl_get("fixtures/?future=1"))
    fx = fx[fx["kickoff_time"].notna()].copy()
    fx["Date"] = pd.to_datetime(fx["kickoff_time"]).dt.tz_localize(None).dt.normalize()
    fx["HomeTeam"] = fx["team_h"].map(names)
    fx["AwayTeam"] = fx["team_a"].map(names)
    fx = fx.sort_values("kickoff_time")
    gws = sorted(fx["event"].dropna().unique())[:max_gameweeks]
    return fx[fx["event"].isin(gws)][["event", "Date", "HomeTeam", "AwayTeam"]].rename(columns={"event": "gameweek"}).reset_index(drop=True)


def fetch_injury_burden() -> pd.DataFrame:
    """Per-team share of squad 'value' (FPL total points) that is currently unavailable. Snapshotted daily
    so a real historical injury dataset builds up over time. Used as context, not (yet) as a model input."""
    boot = _fpl_get("bootstrap-static/")
    names = {t["id"]: FPL_TO_FD.get(t["name"], t["name"]) for t in boot["teams"]}
    p = pd.DataFrame(boot["elements"])
    p["team_name"] = p["team"].map(names)
    chance = pd.to_numeric(p["chance_of_playing_next_round"], errors="coerce")
    p["out"] = p["status"].isin(["i", "s", "u"]) | ((p["status"] == "d") & (chance.fillna(100) <= 50))
    p["pts"] = p["total_points"].astype(float)
    g = p.groupby("team_name").apply(
        lambda d: pd.Series({"players_out": int(d["out"].sum()),
                             "burden": d.loc[d["out"], "pts"].sum() / max(d["pts"].sum(), 1.0)}))
    INJ.mkdir(parents=True, exist_ok=True)
    g.assign(snapshot=str(date.today())).to_csv(INJ / f"snapshot_{date.today()}.csv")
    return g.reset_index().rename(columns={"team_name": "team"})
