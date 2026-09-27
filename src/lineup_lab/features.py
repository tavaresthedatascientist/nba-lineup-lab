"""Leakage-free feature engineering.

Every feature attached to a game is computed only from games played *before*
that game's date. The guiding structure:

    player profile  ->  lineup profile  ->  lineup-vs-lineup differences
    (PER, TS%, USG%)    (5 starters,        + team context (form, rest,
                         core players,        head-to-head record)
                         matchup edges)

`lineup_side_features` and `game_feature_vector` are deliberately written in
plain Python (no pandas) because the website re-implements them in JavaScript;
`tests/test_parity.py` checks both implementations agree.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import config as C
from .data.load import (STAT_COLS, assign_roles, attach_starters, league_constants,
                        load_player_games, load_results_history, team_games)

FEATURES = [
    "LU_PER_DIFF",      # mean PER of the starting five
    "LU_TS_DIFF",       # usage-weighted TS% of the five (efficiency on the shots they take)
    "LU_USG_DIFF",      # total usage (ball-dominance overlap)
    "CORE_N_DIFF",      # how many of the team's 3 core players start
    "CORE_PER_DIFF",    # core players' PER, averaged over the 3 core slots
    "STAR_PER_DIFF",    # best player on the floor
    "MATCHUP_DIFF",     # TS% edge of each player vs. his likely primary defender
    "H2H_WINPCT",       # record over the last 12 (same conf) / 6 (cross conf) meetings
    "WINPCT_DIFF",      # season-to-date win %, lightly shrunk
    "NET_SZN_DIFF",     # season-to-date net rating (points per 100 possessions, shrunk)
    "NET_L10_DIFF",     # net rating over the last 10 games (current form)
    "REST_DIFF",        # days of rest
    "IS_HOME",
]
SIDE_KEYS = ["LU_PER", "LU_TS", "LU_USG", "CORE_N", "CORE_PER", "STAR_PER", "MATCHUP"]


# ---------------------------------------------------------------------------
# 1. Player advanced metrics (season-to-date, shrunk)
# ---------------------------------------------------------------------------
def per_components(pg: pd.DataFrame, tg: pd.DataFrame, lc: dict) -> pd.Series:
    """Hollinger PER numerator for each player-game, pace-adjusted.

    uPER = (1/MP) * [3P + 2/3*AST + (2 - factor*tmAST/tmFG)*FG + ... - PF*(...)]
    aPER = uPER * lgPace / tmPace;  PER = aPER * 15 / lg_aPER
    We keep the bracketed sum per game so it can be accumulated over any window.
    """
    t = tg.set_index(["GAME_ID", "TEAM"])
    key = pd.MultiIndex.from_arrays([pg["GAME_ID"], pg["TEAM"]])
    tm_ast = t["AST"].reindex(key).to_numpy()
    tm_fg = t["FGM"].reindex(key).to_numpy().clip(min=1)
    tm_pace = t["PACE"].reindex(key).to_numpy()
    r = tm_ast / tm_fg
    f, vop, drbp = lc["factor"], lc["VOP"], lc["DRBP"]
    comp = (pg.FG3M + (2 / 3) * pg.AST + (2 - f * r) * pg.FGM
            + pg.FTM * 0.5 * (1 + (1 - r) + (2 / 3) * r)
            - vop * pg.TOV - vop * drbp * (pg.FGA - pg.FGM)
            - vop * 0.44 * (0.44 + 0.56 * drbp) * (pg.FTA - pg.FTM)
            + vop * (1 - drbp) * pg.DREB + vop * drbp * pg.OREB
            + vop * pg.STL + vop * drbp * pg.BLK
            - pg.PF * (lc["lgFT_PF"] - 0.44 * lc["lgFTA_PF"] * vop))
    return comp * (lc["lgPace"] / tm_pace)


def player_timeline(pg: pd.DataFrame, tg: pd.DataFrame, lc: dict) -> pd.DataFrame:
    """Cumulative player profile *after* each game (look up with an as-of join
    that excludes the current date to get the pre-game profile)."""
    t = tg.set_index(["GAME_ID", "TEAM"])
    key = pd.MultiIndex.from_arrays([pg["GAME_ID"], pg["TEAM"]])
    x = pg[["PLAYER_ID", "PLAYER_NAME", "TEAM", "GAME_ID", "GAME_DATE", "GAME_TYPE"] + STAT_COLS].copy()
    for c in ["MIN", "FGA", "FTA", "TOV"]:
        x["TM_" + c] = t[c].reindex(key).to_numpy()
    x["PER_COMP"] = per_components(pg, tg, lc).to_numpy()
    x = x.sort_values(["PLAYER_ID", "GAME_DATE"]).reset_index(drop=True)

    cum_cols = STAT_COLS + ["TM_MIN", "TM_FGA", "TM_FTA", "TM_TOV", "PER_COMP"]
    cum = x.groupby("PLAYER_ID")[cum_cols].cumsum()
    gp = x.groupby("PLAYER_ID").cumcount() + 1

    rs = x["GAME_TYPE"] == "RS"
    lg_aper = x.loc[rs, "PER_COMP"].sum() / x.loc[rs, "MIN"].sum()

    mins = cum["MIN"].clip(lower=1e-6)
    tsa = 2 * (cum["FGA"] + 0.44 * cum["FTA"])
    ts = np.where(tsa > 0, cum["PTS"] / tsa.replace(0, np.nan), C.LEAGUE_PRIOR["TS"])
    usg = 100 * ((cum["FGA"] + 0.44 * cum["FTA"] + cum["TOV"]) * (cum["TM_MIN"] / 5)
                 / (mins * (cum["TM_FGA"] + 0.44 * cum["TM_FTA"] + cum["TM_TOV"])))
    per = (cum["PER_COMP"] / mins) * 15 / lg_aper

    w = cum["MIN"] / (cum["MIN"] + C.PRIOR_MINUTES)          # shrink toward league avg
    out = x[["PLAYER_ID", "PLAYER_NAME", "TEAM", "GAME_ID", "GAME_DATE"]].copy()
    out["GP"] = gp
    out["MIN_TOT"] = cum["MIN"]
    out["MPG"] = cum["MIN"] / gp
    out["PER_RAW"] = per
    out["TS_RAW"] = ts
    out["USG_RAW"] = usg
    out["PER"] = w * per + (1 - w) * C.LEAGUE_PRIOR["PER"]
    out["TS"] = w * ts + (1 - w) * C.LEAGUE_PRIOR["TS"]
    out["USG"] = w * usg + (1 - w) * C.LEAGUE_PRIOR["USG"]
    out.attrs["lg_aper"] = lg_aper
    return out


PRIOR_PROFILE = {"PER": C.LEAGUE_PRIOR["PER"], "TS": C.LEAGUE_PRIOR["TS"],
                 "USG": C.LEAGUE_PRIOR["USG"], "MPG": 0.0, "GP": 0}


def profiles_asof(timeline: pd.DataFrame, queries: pd.DataFrame) -> pd.DataFrame:
    """For each (PLAYER_ID, GAME_DATE) query return the profile from games
    strictly before that date. Players with no prior games get the league prior."""
    q = queries[["PLAYER_ID", "GAME_DATE"]].reset_index().sort_values("GAME_DATE")
    tl = timeline.sort_values("GAME_DATE")
    m = pd.merge_asof(q, tl[["PLAYER_ID", "GAME_DATE", "PER", "TS", "USG", "MPG", "GP"]],
                      on="GAME_DATE", by="PLAYER_ID", allow_exact_matches=False)
    m = m.set_index("index").sort_index()
    for k, v in PRIOR_PROFILE.items():
        m[k] = m[k].fillna(v)
    return m[["PER", "TS", "USG", "MPG", "GP"]]


# ---------------------------------------------------------------------------
# 2. Player-vs-defender matchup history
# ---------------------------------------------------------------------------
def matchup_events(pg: pd.DataFrame, timeline: pd.DataFrame) -> pd.DataFrame:
    """One row per (game, offensive player, defender) with the offensive
    player's TS% that night minus his pre-game TS% baseline.

    * With tracking data (BoxScoreMatchupsV3): the defender is whoever guarded
      him for the most partial possessions.
    * Without it: every opposing rotation player who shared the floor with him
      for 15+ minutes. At lookup time the likely primary defender is chosen by
      positional slot (see `assign_defenders`), so the history used is
      "how did X shoot in games Y played heavy minutes against him".
    """
    base = pg[["GAME_ID", "GAME_DATE", "PLAYER_ID", "TEAM", "OPP", "MIN", "PTS", "FGA", "FTA"]].copy()
    base["TSA"] = 2 * (base["FGA"] + 0.44 * base["FTA"])
    base = base[base["TSA"] >= 4]                           # need a real shooting sample
    base["TS_GAME"] = base["PTS"] / base["TSA"]
    pre = profiles_asof(timeline, base)
    base["D_TS"] = base["TS_GAME"] - pre["TS"].to_numpy()

    if C.MATCHUPS_FILE.exists():
        mu = pd.read_csv(C.MATCHUPS_FILE, dtype={"GAME_ID": str})
        mu["GAME_ID"] = mu["GAME_ID"].str.zfill(10)
        idx = mu.groupby(["GAME_ID", "OFF_PLAYER_ID"])["PARTIAL_POSS"].idxmax()
        prim = mu.loc[idx, ["GAME_ID", "OFF_PLAYER_ID", "DEF_PLAYER_ID"]]
        ev = base.merge(prim, left_on=["GAME_ID", "PLAYER_ID"],
                        right_on=["GAME_ID", "OFF_PLAYER_ID"])
    else:
        defs = pg.loc[pg["MIN"] >= 15, ["GAME_ID", "TEAM", "PLAYER_ID"]].rename(
            columns={"TEAM": "OPP", "PLAYER_ID": "DEF_PLAYER_ID"})
        ev = base[base["MIN"] >= 15].merge(defs, on=["GAME_ID", "OPP"])
    return ev[["GAME_DATE", "PLAYER_ID", "DEF_PLAYER_ID", "D_TS"]].rename(
        columns={"PLAYER_ID": "OFF_PLAYER_ID"})


def matchup_history(events: pd.DataFrame) -> pd.DataFrame:
    """Cumulative (n, sum of TS deltas) per offensive/defender pair after each date."""
    e = events.groupby(["OFF_PLAYER_ID", "DEF_PLAYER_ID", "GAME_DATE"], as_index=False).agg(
        n=("D_TS", "size"), s=("D_TS", "sum"))
    e = e.sort_values(["OFF_PLAYER_ID", "DEF_PLAYER_ID", "GAME_DATE"])
    g = e.groupby(["OFF_PLAYER_ID", "DEF_PLAYER_ID"])
    e["N"] = g["n"].cumsum()
    e["S"] = g["s"].cumsum()
    return e[["OFF_PLAYER_ID", "DEF_PLAYER_ID", "GAME_DATE", "N", "S"]]


def matchup_edge(n: float, s: float) -> float:
    """Shrunk TS% edge in percentage points: s / (n + k) * 100."""
    return 100.0 * s / (n + C.MATCHUP_PRIOR_GAMES)


# ---------------------------------------------------------------------------
# 3. Lineup features (plain Python: mirrored in site/js/model.js)
# ---------------------------------------------------------------------------
def assign_defenders(lineup: list[dict], opp_lineup: list[dict]) -> list[dict]:
    """Most-likely primary defender for each player: sort both fives from
    guard-most to big-most (position score) and match slot to slot."""
    a = sorted(lineup, key=lambda p: p["pos"])
    b = sorted(opp_lineup, key=lambda p: p["pos"])
    return [{"off": x["id"], "def": y["id"]} for x, y in zip(a, b)]


def lineup_side_features(lineup: list[dict], opp_lineup: list[dict], core_ids: list[int],
                         mhist) -> dict:
    """Aggregate five player profiles into lineup features.

    lineup / opp_lineup: dicts with id, per, ts, usg, pos
    core_ids: the team's core (backbone) player ids
    mhist(off_id, def_id) -> (n, sum_d_ts) prior matchup history
    """
    usg_sum = sum(p["usg"] for p in lineup)
    ts_w = sum(p["usg"] * p["ts"] for p in lineup) / usg_sum
    core_in = [p for p in lineup if p["id"] in core_ids]
    edges = {}
    for pair in assign_defenders(lineup, opp_lineup):
        n, s = mhist(pair["off"], pair["def"])
        edges[pair["off"]] = matchup_edge(n, s)
    matchup = sum(p["usg"] * edges[p["id"]] for p in lineup) / usg_sum
    return {
        "LU_PER": sum(p["per"] for p in lineup) / 5.0,
        "LU_TS": 100.0 * ts_w,
        "LU_USG": usg_sum,
        "CORE_N": float(len(core_in)),
        "CORE_PER": sum(p["per"] for p in core_in) / C.CORE_SIZE,
        "STAR_PER": max(p["per"] for p in lineup),
        "MATCHUP": matchup,
    }


def game_feature_vector(side: dict, opp_side: dict, ctx: dict) -> list[float]:
    """Team-perspective feature vector in FEATURES order."""
    diffs = [side[k] - opp_side[k] for k in SIDE_KEYS]
    return diffs + [ctx["h2h"], ctx["winpct"] - ctx["opp_winpct"],
                    ctx["netszn"] - ctx["opp_netszn"],
                    ctx["net10"] - ctx["opp_net10"], ctx["rest"] - ctx["opp_rest"],
                    float(ctx["is_home"])]


# ---------------------------------------------------------------------------
# 4. Team context: form, rest, head-to-head, core players
# ---------------------------------------------------------------------------
def team_context(tg: pd.DataFrame) -> pd.DataFrame:
    t = tg.sort_values(["TEAM", "GAME_DATE"]).copy()
    g = t.groupby("TEAM")
    w_prior = g["WIN"].cumsum() - t["WIN"]
    n_prior = g.cumcount()
    t["WINPCT"] = (w_prior + 2) / (n_prior + 4)              # 2-2 prior: early games -> ~.500
    # season net rating from totals, shrunk as if 5 extra games at net 0
    pdiff, poss = t["PTS"] - t["OPP_PTS"], t["GAME_POSS"]
    cum_pd = pdiff.groupby(t["TEAM"]).cumsum() - pdiff
    cum_poss = poss.groupby(t["TEAM"]).cumsum() - poss
    t["NET_SZN"] = 100 * cum_pd / (cum_poss + 5 * 100)
    net = t["ORTG"] - t["DRTG"]
    t["NET_L10"] = net.groupby(t["TEAM"]).transform(
        lambda s: s.shift(1).rolling(10, min_periods=1).mean()).fillna(0.0)
    rest = g["GAME_DATE"].diff().dt.days.sub(1)
    t["REST"] = rest.fillna(3).clip(upper=4)
    return t


def head_to_head(tg: pd.DataFrame, history: pd.DataFrame) -> pd.DataFrame:
    """Home team's win % over the last N meetings before each game date.

    N = 12 for same-conference opponents, 6 for cross-conference opponents.
    Prior seasons come from the multi-season results file; 2025-26 meetings
    come from the game logs themselves. No meetings -> 0.5.
    """
    # reference team per game = home team (alphabetical first at neutral sites)
    ref = tg.sort_values(["GAME_ID", "HOME", "TEAM"], ascending=[True, False, True]).drop_duplicates("GAME_ID")
    cur = ref[["GAME_ID", "GAME_DATE", "TEAM", "OPP", "WIN"]].rename(
        columns={"GAME_DATE": "date", "TEAM": "home", "OPP": "away", "WIN": "home_win"})
    past = history[history["season"] != "2025-2026"][["date", "home", "away", "home_win"]]
    allg = pd.concat([past.assign(GAME_ID=None), cur], ignore_index=True).sort_values("date")
    allg["pair"] = [tuple(sorted(p)) for p in zip(allg["home"], allg["away"])]

    rows = []
    for pair, grp in allg.groupby("pair", sort=False):
        n_lookback = C.H2H_SAME_CONF if C.conference(pair[0]) == C.conference(pair[1]) else C.H2H_CROSS_CONF
        winners = []   # chronological list of winning team abbreviations
        dates = []
        for r in grp.itertuples(index=False):
            if isinstance(r.GAME_ID, str):
                prior = [w for w, d in zip(winners, dates) if d < r.date][-n_lookback:]
                pct = (sum(w == r.home for w in prior) / len(prior)) if prior else 0.5
                rows.append({"GAME_ID": r.GAME_ID, "H2H_REF": r.home, "H2H_HOME": pct, "H2H_N": len(prior)})
            winners.append(r.home if r.home_win == 1 else r.away)
            dates.append(r.date)
    return pd.DataFrame(rows)


def core_players(pg: pd.DataFrame, timeline: pd.DataFrame, tg: pd.DataFrame) -> dict:
    """Core (backbone) players for every team-game: config overrides first,
    then the top players by pre-game PER x MPG among recent contributors."""
    name_to_id = pg.drop_duplicates("PLAYER_ID").set_index("PLAYER_NAME")["PLAYER_ID"].to_dict()
    # Cold start (a team's first 3 games): rank by preseason points+rebounds+
    # assists per game - stars play limited preseason minutes, but still produce
    # the most per appearance, so per-game production identifies them well.
    pre = load_player_games(include_preseason=True)
    pre = pre[pre["GAME_TYPE"] == "PRE"]
    pre = (pre["PTS"] + pre["REB"] + pre["AST"]).groupby([pre["TEAM"], pre["PLAYER_ID"]]).mean()
    recent = pg[["TEAM", "GAME_DATE", "PLAYER_ID"]].drop_duplicates()
    out = {}
    for team, tgs in tg.groupby("TEAM"):
        dates = sorted(tgs["GAME_DATE"].unique())
        roster_hist = recent[recent["TEAM"] == team]
        forced = [name_to_id[n] for n in C.CORE_OVERRIDES.get(team, []) if n in name_to_id]
        for i, d in enumerate(dates):
            window_start = dates[max(0, i - 20)]
            cands = roster_hist[(roster_hist["GAME_DATE"] < d) &
                                (roster_hist["GAME_DATE"] >= window_start)]["PLAYER_ID"].unique()
            if i < 3:
                pre_rank = [p for p in pre.get(team, pd.Series(dtype=float)).sort_values(ascending=False).index
                            if p not in forced]
                out[(team, d)] = (forced + pre_rank)[: C.CORE_SIZE]
                continue
            prof = profiles_asof(timeline, pd.DataFrame({"PLAYER_ID": cands, "GAME_DATE": d}))
            prof["PLAYER_ID"] = cands
            prof = prof[prof["GP"] >= 1]
            prof["VALUE"] = prof["PER"] * prof["MPG"]
            ranked = [p for p in prof.sort_values("VALUE", ascending=False)["PLAYER_ID"] if p not in forced]
            out[(team, d)] = (forced + ranked)[: C.CORE_SIZE]
    return out


# ---------------------------------------------------------------------------
# 5. Assemble the modelling table
# ---------------------------------------------------------------------------
@dataclass
class FeatureBundle:
    pg: pd.DataFrame
    tg: pd.DataFrame
    timeline: pd.DataFrame
    roles: pd.DataFrame
    mhist: pd.DataFrame
    cores: dict
    lc: dict
    table: pd.DataFrame          # one row per team-game, FEATURES + WIN
    starter_source: str


def build() -> FeatureBundle:
    pg = attach_starters(load_player_games())
    starter_source = pg.attrs.get("starter_source", "minutes_proxy")
    tg = team_games(pg)
    lc = league_constants(pg)
    timeline = player_timeline(pg, tg, lc)
    roles = assign_roles(pg)
    pg = pg.merge(roles, on="PLAYER_ID", how="left")
    pg["POS_SCORE"] = pg["POS_SCORE"].fillna(0.0)
    mh = matchup_history(matchup_events(pg, timeline))
    cores = core_players(pg, timeline, tg)
    ctx = team_context(tg)
    h2h = head_to_head(tg, load_results_history())

    # pre-game profiles for every starter
    st = pg[pg["STARTER"] == 1].copy()
    prof = profiles_asof(timeline, st)
    st[["PER", "TS", "USG"]] = prof[["PER", "TS", "USG"]].to_numpy()

    # fast pre-game matchup lookup: (off, def) -> list of (date, N, S)
    mh_idx: dict = {}
    for r in mh.itertuples(index=False):
        mh_idx.setdefault((r.OFF_PLAYER_ID, r.DEF_PLAYER_ID), []).append((r.GAME_DATE, r.N, r.S))

    def lookup_factory(date):
        def f(o, d):
            best = (0, 0.0)
            for dt, n, s in mh_idx.get((o, d), ()):
                if dt < date:
                    best = (n, s)
                else:
                    break
            return best
        return f

    lineups = {k: [{"id": int(r.PLAYER_ID), "per": r.PER, "ts": r.TS, "usg": r.USG, "pos": r.POS_SCORE}
                   for r in grp.itertuples(index=False)]
               for k, grp in st.groupby(["GAME_ID", "TEAM"])}

    ctx_idx = ctx.set_index(["GAME_ID", "TEAM"])
    h2h_idx = h2h.set_index("GAME_ID")
    rows = []
    for r in ctx.itertuples(index=False):
        mine, theirs = lineups.get((r.GAME_ID, r.TEAM)), lineups.get((r.GAME_ID, r.OPP))
        if not mine or not theirs or len(mine) != 5 or len(theirs) != 5:
            continue
        look = lookup_factory(r.GAME_DATE)
        side = lineup_side_features(mine, theirs, cores[(r.TEAM, r.GAME_DATE)], look)
        opp_side = lineup_side_features(theirs, mine, cores[(r.OPP, r.GAME_DATE)], look)
        o = ctx_idx.loc[(r.GAME_ID, r.OPP)]
        hh = h2h_idx.loc[r.GAME_ID, "H2H_HOME"]
        c = {"h2h": hh if h2h_idx.loc[r.GAME_ID, "H2H_REF"] == r.TEAM else 1 - hh,
             "winpct": r.WINPCT, "opp_winpct": o.WINPCT,
             "netszn": r.NET_SZN, "opp_netszn": o.NET_SZN,
             "net10": r.NET_L10, "opp_net10": o.NET_L10,
             "rest": r.REST, "opp_rest": o.REST, "is_home": r.HOME}
        vec = game_feature_vector(side, opp_side, c)
        rows.append({"GAME_ID": r.GAME_ID, "GAME_DATE": r.GAME_DATE, "GAME_TYPE": r.GAME_TYPE,
                     "TEAM": r.TEAM, "OPP": r.OPP, "TEAM_GAME_NO": r.TEAM_GAME_NO,
                     "PTS": r.PTS, "OPP_PTS": r.OPP_PTS, "WIN": r.WIN,
                     "H2H_N": int(h2h_idx.loc[r.GAME_ID, "H2H_N"]),
                     **dict(zip(FEATURES, vec))})
    table = pd.DataFrame(rows).sort_values(["GAME_DATE", "GAME_ID", "IS_HOME"]).reset_index(drop=True)
    return FeatureBundle(pg, tg, timeline, roles, mh, cores, lc, table, starter_source)
