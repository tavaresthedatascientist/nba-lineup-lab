"""Lineup optimizer: pick the starting five that maximizes win probability
against a specific opponent's starting five, built around the core players.

    python -m lineup_lab optimize --team OKC --opp DEN --home
    python -m lineup_lab optimize --team OKC --opp HOU --away --out "Jalen Williams"

The league state (rosters, profiles, matchup history, team context) is the
same structure exported to site/data/, so the website and this module score
lineups identically.
"""
from __future__ import annotations

from collections import Counter
from itertools import combinations

import numpy as np
import pandas as pd

from . import config as C
from .features import FEATURES, FeatureBundle, game_feature_vector, lineup_side_features


# ---------------------------------------------------------------------------
# League state as of the end of the data
# ---------------------------------------------------------------------------
def league_state(b: FeatureBundle) -> dict:
    tl = b.timeline.sort_values("GAME_DATE")
    last = tl.groupby("PLAYER_ID").tail(1).set_index("PLAYER_ID")
    roles = b.roles.set_index("PLAYER_ID")
    pg = b.pg
    games_for_team = pg.groupby(["PLAYER_ID", "TEAM"]).size()

    tg = b.tg.sort_values("GAME_DATE")
    teams = {}
    for team in C.TEAMS:
        t = tg[tg["TEAM"] == team]
        rs = t[t["GAME_TYPE"] == "RS"]
        net_all = (t["PTS"] - t["OPP_PTS"]).sum() / t["GAME_POSS"].sum() * 100
        pdiff, poss = (t["PTS"] - t["OPP_PTS"]).sum(), t["GAME_POSS"].sum()
        last10 = t.tail(10)
        roster_ids = [pid for pid in last.index[last["TEAM"] == team]
                      if games_for_team.get((pid, team), 0) >= C.ROSTER_MIN_GAMES]
        players = []
        for pid in roster_ids:
            r = last.loc[pid]
            players.append({
                "id": int(pid), "name": r["PLAYER_NAME"],
                "role": roles.loc[pid, "ROLE"] if pid in roles.index else "W",
                "pos": round(float(roles.loc[pid, "POS_SCORE"]) if pid in roles.index else 0.0, 4),
                "per": round(float(r["PER"]), 3), "ts": round(float(r["TS"]), 4),
                "usg": round(float(r["USG"]), 3), "mpg": round(float(r["MPG"]), 1),
                "gp": int(r["GP"]),
                "per_raw": round(float(r["PER_RAW"]), 1), "ts_raw": round(float(r["TS_RAW"]), 3),
                "usg_raw": round(float(r["USG_RAW"]), 1),
            })
        players.sort(key=lambda p: -p["mpg"])
        roster_set = {p["id"] for p in players}

        # core: overrides, then PER x MPG
        names = {p["name"]: p["id"] for p in players}
        core = [names[n] for n in C.CORE_OVERRIDES.get(team, []) if n in names]
        for p in sorted(players, key=lambda p: -p["per"] * p["mpg"]):
            if len(core) >= C.CORE_SIZE:
                break
            if p["id"] not in core and p["gp"] >= 10:
                core.append(p["id"])

        # most common starting five over the last 15 games, restricted to current roster
        recent_ids = t.tail(15)["GAME_ID"]
        st = pg[(pg["TEAM"] == team) & (pg["STARTER"] == 1) & pg["GAME_ID"].isin(recent_ids)]
        counts = Counter(st["PLAYER_ID"])
        default5 = [pid for pid, _ in counts.most_common() if pid in roster_set][:5]
        for p in players:
            if len(default5) >= 5:
                break
            if p["id"] not in default5:
                default5.append(p["id"])

        teams[team] = {
            "abbr": team, "name": C.TEAM_NAMES[team], "conf": C.conference(team),
            "w": int(rs["WIN"].sum()), "l": int(len(rs) - rs["WIN"].sum()),
            "winpct": round(float((rs["WIN"].sum() + 2) / (len(rs) + 4)), 4),
            "net_szn": round(float(100 * pdiff / (poss + 500)), 3),
            "net_l10": round(float(((last10["ORTG"] - last10["DRTG"])).mean()), 3),
            "net_rating": round(float(net_all), 1),
            "core": [int(x) for x in core], "default5": [int(x) for x in default5],
            "players": players,
        }

    # head-to-head as of the end of the data, team perspective
    from .data.load import load_results_history
    hist = load_results_history()
    past = hist[hist["season"] != "2025-2026"][["date", "home", "away", "home_win"]]
    ref = b.tg.sort_values(["GAME_ID", "HOME", "TEAM"], ascending=[True, False, True]).drop_duplicates("GAME_ID")
    cur = ref.rename(columns={"GAME_DATE": "date", "TEAM": "home", "OPP": "away", "WIN": "home_win"})[
        ["date", "home", "away", "home_win"]]
    allg = pd.concat([past, cur]).sort_values("date")
    allg["winner"] = np.where(allg["home_win"] == 1, allg["home"], allg["away"])
    h2h = {}
    for a in C.TEAMS:
        h2h[a] = {}
        for bb in C.TEAMS:
            if a == bb:
                continue
            n = C.H2H_SAME_CONF if C.conference(a) == C.conference(bb) else C.H2H_CROSS_CONF
            m = allg[((allg["home"] == a) & (allg["away"] == bb)) | ((allg["home"] == bb) & (allg["away"] == a))].tail(n)
            w = int((m["winner"] == a).sum())
            h2h[a][bb] = {"w": w, "n": int(len(m)), "pct": round(w / len(m), 4) if len(m) else 0.5}

    # matchup history totals (only pairs of current-roster players)
    on_roster = {p["id"] for t in teams.values() for p in t["players"]}
    mh = b.mhist.sort_values("GAME_DATE").groupby(["OFF_PLAYER_ID", "DEF_PLAYER_ID"]).tail(1)
    mh = mh[mh["OFF_PLAYER_ID"].isin(on_roster) & mh["DEF_PLAYER_ID"].isin(on_roster)]
    matchups = {f"{int(o)}-{int(d)}": [int(n), round(float(s), 4)]
                for o, d, n, s in zip(mh["OFF_PLAYER_ID"], mh["DEF_PLAYER_ID"], mh["N"], mh["S"])}
    return {"teams": teams, "h2h": h2h, "matchups": matchups}


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------
def _players(state, team, ids):
    by_id = {p["id"]: p for p in state["teams"][team]["players"]}
    return [by_id[i] for i in ids]


def _mlookup(state):
    m = state["matchups"]
    return lambda o, d: tuple(m.get(f"{o}-{d}", (0, 0.0)))


def team_vector(state, team, opp, lineup, opp_lineup, is_home, rest=1, opp_rest=1):
    T, O = state["teams"][team], state["teams"][opp]
    look = _mlookup(state)
    side = lineup_side_features(lineup, opp_lineup, T["core"], look)
    oside = lineup_side_features(opp_lineup, lineup, O["core"], look)
    ctx = {"h2h": state["h2h"][team][opp]["pct"], "winpct": T["winpct"], "opp_winpct": O["winpct"],
           "netszn": T["net_szn"], "opp_netszn": O["net_szn"],
           "net10": T["net_l10"], "opp_net10": O["net_l10"],
           "rest": rest, "opp_rest": opp_rest, "is_home": int(is_home)}
    return game_feature_vector(side, oside, ctx), side


def win_probability(forest, state, team, opp, lineup_ids, opp_ids, home: bool | None = True) -> float:
    """Symmetric P(team wins): average of the team's view and 1 - opponent's view.
    home=None means a neutral site."""
    me, them = _players(state, team, lineup_ids), _players(state, opp, opp_ids)
    h_me = 0 if home is None else int(home)
    h_them = 0 if home is None else int(not home)
    v1, _ = team_vector(state, team, opp, me, them, h_me)
    v2, _ = team_vector(state, opp, team, them, me, h_them)
    X = pd.DataFrame([v1, v2], columns=FEATURES)
    p = forest.predict_proba(X)[:, 1]
    return float((p[0] + 1 - p[1]) / 2)


def valid_lineup(players: list[dict]) -> bool:
    roles = Counter(p["role"] for p in players)
    return (roles["G"] >= C.MIN_BALLHANDLERS and roles["B"] >= C.MIN_BIGS
            and sum(p["usg"] for p in players) <= C.MAX_LINEUP_USG)


def optimize(forest, state, team, opp, home: bool | None = True, opp_ids=None,
             unavailable=(), pool_size: int = 11, top_k: int = 5) -> dict:
    T = state["teams"][team]
    opp_ids = opp_ids or state["teams"][opp]["default5"]
    out = set(unavailable)
    pool = [p for p in T["players"] if p["id"] not in out][:pool_size]
    core = [c for c in T["core"] if c in {p["id"] for p in pool}]
    rest = [p["id"] for p in pool if p["id"] not in core]
    results = []
    for combo in combinations(rest, 5 - len(core)):
        ids = core + list(combo)
        if not valid_lineup(_players(state, team, ids)):
            continue
        results.append((win_probability(forest, state, team, opp, ids, opp_ids, home), ids))
    results.sort(key=lambda x: -x[0])
    default = [i for i in T["default5"] if i not in out]
    base = win_probability(forest, state, team, opp, default, opp_ids, home) if len(default) == 5 else None
    names = {p["id"]: p["name"] for p in T["players"]}
    return {
        "team": team, "opp": opp, "home": home,
        "core": [names[c] for c in core],
        "opp_lineup": [p["name"] for p in _players(state, opp, opp_ids)],
        "default_lineup": [names[i] for i in default], "default_win_prob": base,
        "evaluated": len(results),
        "best": [{"win_prob": round(p, 4), "lineup": [names[i] for i in ids]} for p, ids in results[:top_k]],
    }
