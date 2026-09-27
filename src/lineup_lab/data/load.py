"""Load and clean the raw 2025-26 data into tidy player-game and team-game tables."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .. import config as C

STAT_COLS = ["MIN", "FGM", "FGA", "FG3M", "FG3A", "FTM", "FTA", "OREB", "DREB",
             "REB", "AST", "TOV", "STL", "BLK", "PF", "PTS"]


def _game_type(game_id: pd.Series) -> pd.Series:
    prefix = game_id.str[:5]
    out = pd.Series("OTHER", index=game_id.index)
    out[prefix == C.REGULAR_SEASON_PREFIX] = "RS"
    out[prefix.isin(C.PLAYOFF_PREFIXES)] = "PO"
    return out


def load_player_games(include_preseason: bool = False) -> pd.DataFrame:
    """One row per player per game (regular season + play-in/playoffs)."""
    g = pd.read_csv(C.GAMELOGS_FILE, dtype={"GAME_ID": str})
    g["GAME_ID"] = g["GAME_ID"].str.zfill(10)
    g["GAME_DATE"] = pd.to_datetime(g["GAME_DATE"].str[:10])
    g = g.rename(columns={"TEAM_ABBREVIATION": "TEAM"})
    g = g[g["TEAM"].isin(C.TEAMS)].copy()          # drop All-Star exhibition teams
    g["GAME_TYPE"] = _game_type(g["GAME_ID"])
    g.loc[g["GAME_ID"].str[:5] == "00125", "GAME_TYPE"] = "PRE"
    keep = ["RS", "PO", "PRE"] if include_preseason else ["RS", "PO"]
    g = g[g["GAME_TYPE"].isin(keep)].copy()

    # "OKC vs. DEN" = home game, "OKC @ DEN" = road game
    g["HOME"] = g["MATCHUP"].str.contains(" vs. ").astype(int)
    g["OPP"] = g["MATCHUP"].str.split(r" vs\. | @ ", regex=True).str[-1].str.strip()
    g[STAT_COLS] = g[STAT_COLS].fillna(0.0).astype(float)
    g = g.sort_values(["GAME_DATE", "GAME_ID", "TEAM", "MIN"],
                      ascending=[True, True, True, False]).reset_index(drop=True)
    return g


def attach_starters(pg: pd.DataFrame) -> pd.DataFrame:
    """Flag each team's starting five.

    If `data/raw/starters_2025-26.csv.gz` exists (written by the nba_api fetcher
    from BoxScoreTraditionalV3) the true starters are used. Otherwise we fall
    back to a minutes proxy: the five players who logged the most minutes.
    Across a season this matches the listed starters for the large majority of
    team-games; the misses are mostly foul-trouble or blowout nights.
    """
    pg = pg.copy()
    if C.STARTERS_FILE.exists():
        st = pd.read_csv(C.STARTERS_FILE, dtype={"GAME_ID": str})
        st["GAME_ID"] = st["GAME_ID"].str.zfill(10)
        key = set(zip(st["GAME_ID"], st["PLAYER_ID"]))
        pg["STARTER"] = [int((a, b) in key) for a, b in zip(pg["GAME_ID"], pg["PLAYER_ID"])]
        pg.attrs["starter_source"] = "boxscore"
    else:
        rank = pg.groupby(["GAME_ID", "TEAM"])["MIN"].rank(method="first", ascending=False)
        pg["STARTER"] = (rank <= 5).astype(int)
        pg.attrs["starter_source"] = "minutes_proxy"
    return pg


def team_games(pg: pd.DataFrame) -> pd.DataFrame:
    """One row per team per game with box totals, possessions and result."""
    agg = pg.groupby(["GAME_ID", "GAME_DATE", "GAME_TYPE", "TEAM", "OPP", "HOME"],
                     as_index=False)[STAT_COLS].sum()
    wl = pg.groupby(["GAME_ID", "TEAM"])["WL"].first().rename("WL").reset_index()
    agg = agg.merge(wl, on=["GAME_ID", "TEAM"], how="left")
    # Data-quality repair: some team-games in the bundled snapshot are missing
    # player rows (team minutes < 240). Scale that team's box totals up to 240
    # minutes so ratings aren't dragged down by absent rows. Ratios (pace,
    # AST/FG, usage denominators) are unchanged by the scaling.
    agg["RAW_MIN"] = agg["MIN"]
    scale = np.where(agg["MIN"] < 239.5, 240.0 / agg["MIN"].clip(lower=1), 1.0)
    for c in STAT_COLS:
        agg[c] = agg[c] * scale
    agg["POSS"] = agg["FGA"] + 0.44 * agg["FTA"] - agg["OREB"] + agg["TOV"]
    opp = agg[["GAME_ID", "TEAM", "PTS", "POSS", "OREB", "DREB", "REB"]].rename(
        columns={"TEAM": "OPP", "PTS": "OPP_PTS", "POSS": "OPP_POSS",
                 "OREB": "OPP_OREB", "DREB": "OPP_DREB", "REB": "OPP_REB"})
    tg = agg.merge(opp, on=["GAME_ID", "OPP"], how="left")
    # The official W/L flag is authoritative. A few dozen games in the source
    # snapshot are missing one or two player rows, so summed points can
    # disagree with the result; those box scores are flagged incomplete.
    tg["WIN"] = (tg["WL"] == "W").astype(int)
    tg["BOX_COMPLETE"] = (tg["RAW_MIN"] >= 239.5).astype(int)
    tg = tg.drop(columns="WL")
    tg["GAME_POSS"] = (tg["POSS"] + tg["OPP_POSS"]) / 2
    tg["ORTG"] = 100 * tg["PTS"] / tg["GAME_POSS"]
    tg["DRTG"] = 100 * tg["OPP_PTS"] / tg["GAME_POSS"]
    tg["PACE"] = 48 * tg["GAME_POSS"] / (tg["MIN"] / 5)
    tg = tg.sort_values(["TEAM", "GAME_DATE"]).reset_index(drop=True)
    tg["TEAM_GAME_NO"] = tg.groupby("TEAM").cumcount()   # 0-based, RS then playoffs
    return tg


def load_results_history() -> pd.DataFrame:
    """Game results 2021-22 onward (for multi-season head-to-head records).

    Preseason and All-Star games are excluded; playoff games are kept - a
    playoff meeting is the most informative head-to-head game there is.
    """
    r = pd.read_csv(C.RESULTS_FILE)
    r = r[(r["round"].fillna("") != "Pre-season") & (~r["is_allstar"].astype(bool))].copy()
    r["home"] = r["home_team"].map(C.NAME_TO_ABBR)
    r["away"] = r["away_team"].map(C.NAME_TO_ABBR)
    r = r.dropna(subset=["home", "away"])
    r["date"] = pd.to_datetime(r["date"])
    r["home_win"] = (r["home_score"] > r["away_score"]).astype(int)
    return r[["season", "date", "home", "away", "home_win"]].sort_values("date").reset_index(drop=True)


def league_constants(pg: pd.DataFrame) -> dict:
    """League-wide constants used by Hollinger's PER (regular season).

    These are league-level normalizers (value of a possession, DRB%, pace), not
    team or player outcomes, and they are extremely stable after the first
    weeks of a season, so using full-season values introduces no meaningful
    look-ahead into player features.
    """
    rs = pg[pg["GAME_TYPE"] == "RS"]
    lg = rs[STAT_COLS].sum()
    tg = team_games(rs)
    factor = (2 / 3) - (0.5 * (lg.AST / lg.FGM)) / (2 * (lg.FGM / lg.FTM))
    vop = lg.PTS / (lg.FGA - lg.OREB + lg.TOV + 0.44 * lg.FTA)
    drbp = (lg.REB - lg.OREB) / lg.REB
    return {
        "factor": float(factor), "VOP": float(vop), "DRBP": float(drbp),
        "lgFT_PF": float(lg.FTM / lg.PF), "lgFTA_PF": float(lg.FTA / lg.PF),
        "lgPace": float(tg["PACE"].mean()),
        "lgORTG": float((100 * tg["PTS"].sum() / tg["GAME_POSS"].sum())),
    }


def assign_roles(pg: pd.DataFrame) -> pd.DataFrame:
    """Infer a positional role (G / W / B) from each player's statistical profile.

    Box-score logs carry no position field, so we build a 'position score' from
    per-36 rates: rebounding and rim protection push toward big, playmaking and
    three-point volume push toward guard. Rotation players are split into
    terciles. The same score orders players into 1-5 slots when we pair each
    player with his most likely primary defender.
    """
    rs = pg[pg["GAME_TYPE"] == "RS"]
    tot = rs.groupby(["PLAYER_ID", "PLAYER_NAME"])[STAT_COLS].sum()
    per36 = tot[["REB", "BLK", "AST", "FG3A"]].div(tot["MIN"].clip(lower=1), axis=0) * 36
    rot = tot["MIN"] >= 300
    z = (per36 - per36[rot].mean()) / per36[rot].std()
    score = 0.9 * z["REB"] + 0.6 * z["BLK"] - 0.8 * z["AST"] - 0.5 * z["FG3A"]
    lo, hi = score[rot].quantile([1 / 3, 2 / 3])
    role = np.where(score <= lo, "G", np.where(score >= hi, "B", "W"))
    out = pd.DataFrame({"POS_SCORE": score.values, "ROLE": role}, index=tot.index).reset_index()
    return out[["PLAYER_ID", "POS_SCORE", "ROLE"]]
