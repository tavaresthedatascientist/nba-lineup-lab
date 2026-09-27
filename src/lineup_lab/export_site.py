"""Write the JSON files the static website reads (site/data/*.json)."""
from __future__ import annotations

import datetime as dt

import pandas as pd

from . import config as C
from .features import FEATURES, FeatureBundle, profiles_asof
from .model import forest_to_json, importances, save_json, tree_to_json

FEATURE_HELP = {
    "LU_PER_DIFF": "Starting-five average PER (Hollinger), vs. opponent",
    "LU_TS_DIFF": "Starting-five usage-weighted true shooting %, vs. opponent",
    "LU_USG_DIFF": "Starting-five total usage rate, vs. opponent",
    "CORE_N_DIFF": "Core players in the starting five, vs. opponent",
    "CORE_PER_DIFF": "Core players' PER per core slot, vs. opponent",
    "STAR_PER_DIFF": "Best player's PER, vs. opponent's best",
    "MATCHUP_DIFF": "TS% edge vs. likely primary defender (usage-weighted), vs. opponent",
    "H2H_WINPCT": "Win % in last 12 meetings (6 if cross-conference)",
    "WINPCT_DIFF": "Season win %, vs. opponent",
    "NET_SZN_DIFF": "Season net rating, vs. opponent",
    "NET_L10_DIFF": "Net rating over last 10 games, vs. opponent",
    "REST_DIFF": "Days of rest, vs. opponent",
    "IS_HOME": "Home court",
}


def _strip(bt: dict) -> dict:
    return {k: v for k, v in bt.items() if k != "predictions"}


def games_payload(b: FeatureBundle, preds: pd.DataFrame) -> list[dict]:
    """Every game that has an out-of-sample model prediction, for the
    'Beat the Model' game mode."""
    pg, tg, table = b.pg, b.tg.sort_values(["TEAM", "GAME_DATE"]).copy(), b.table
    g = tg.groupby("TEAM")
    tg["W_PRE"] = g["WIN"].cumsum() - tg["WIN"]
    tg["L_PRE"] = g.cumcount() - tg["W_PRE"]
    tgi = tg.set_index(["GAME_ID", "TEAM"])

    st = pg[pg["STARTER"] == 1].copy()
    prof = profiles_asof(b.timeline, st)
    st[["PER", "TS", "USG"]] = prof[["PER", "TS", "USG"]].to_numpy()
    lineups = {k: grp for k, grp in st.groupby(["GAME_ID", "TEAM"])}
    names = pg.drop_duplicates("PLAYER_ID").set_index("PLAYER_ID")["PLAYER_NAME"].to_dict()
    tab = table.set_index(["GAME_ID", "TEAM"])

    out = []
    for r in preds.itertuples(index=False):
        a, h = r.TEAM, r.OPP                      # a = reference (home) team
        ra, rh = tgi.loc[(r.GAME_ID, a)], tgi.loc[(r.GAME_ID, h)]
        feat = tab.loc[(r.GAME_ID, a)]

        def five(team):
            lu = lineups[(r.GAME_ID, team)].sort_values("POS_SCORE")
            return [{"id": int(x.PLAYER_ID), "name": x.PLAYER_NAME, "role": x.ROLE,
                     "per": round(float(x.PER), 1), "ts": round(float(x.TS), 3),
                     "usg": round(float(x.USG), 1)} for x in lu.itertuples(index=False)]

        def core_out(team):
            starters = set(lineups[(r.GAME_ID, team)]["PLAYER_ID"])
            return [names[c] for c in b.cores[(team, ra.GAME_DATE)] if c not in starters]

        n = int(feat["H2H_N"])
        out.append({
            "id": r.GAME_ID, "date": ra.GAME_DATE.strftime("%Y-%m-%d"),
            "type": ra.GAME_TYPE, "neutral": int(ra.HOME == 0),
            "home": a, "away": h,
            "home_rec": [int(ra.W_PRE), int(ra.L_PRE)], "away_rec": [int(rh.W_PRE), int(rh.L_PRE)],
            "home_win": int(ra.WIN),
            "home_pts": int(round(ra.PTS)) if ra.BOX_COMPLETE and rh.BOX_COMPLETE else None,
            "away_pts": int(round(rh.PTS)) if ra.BOX_COMPLETE and rh.BOX_COMPLETE else None,
            "home5": five(a), "away5": five(h),
            "home_core_out": core_out(a), "away_core_out": core_out(h),
            "h2h": {"w": int(round(float(feat["H2H_WINPCT"]) * n)), "n": n},
            "p_home": round(float(r.P_REF), 4),
        })
    return out


def export(b: FeatureBundle, tree, forest, state: dict, rolling: dict, expanding: dict,
           playoffs: dict) -> None:
    C.SITE_DATA.mkdir(parents=True, exist_ok=True)
    preds = pd.concat([expanding["predictions"], playoffs["predictions"]], ignore_index=True)
    meta = {
        "season": C.SEASON,
        "generated": dt.date.today().isoformat(),
        "data_through": b.tg["GAME_DATE"].max().strftime("%Y-%m-%d"),
        "starter_source": b.starter_source,
        "matchup_source": "tracking" if C.MATCHUPS_FILE.exists() else "positional_proxy",
        "features": FEATURES, "feature_help": FEATURE_HELP,
        "backtest": {"rolling_5_10": _strip(rolling), "expanding_10": _strip(expanding),
                     "playoffs": _strip(playoffs)},
        "importance_forest": importances(forest), "importance_tree": importances(tree),
        "tree": tree_to_json(tree),
        "config": {"h2h_same": C.H2H_SAME_CONF, "h2h_cross": C.H2H_CROSS_CONF,
                   "core_size": C.CORE_SIZE, "max_usg": C.MAX_LINEUP_USG,
                   "matchup_prior": C.MATCHUP_PRIOR_GAMES,
                   "train_n": C.TRAIN_GAMES_PER_TEAM, "test_n": C.TEST_GAMES_PER_TEAM},
    }
    save_json(meta, C.SITE_DATA / "meta.json")
    save_json(forest_to_json(forest), C.SITE_DATA / "model.json", compact=True)
    # matchup histories are split per team (offensive player's team) so the
    # site only downloads the two teams it is comparing
    by_team = {t: {} for t in C.TEAMS}
    team_of = {p["id"]: t for t, T in state["teams"].items() for p in T["players"]}
    for key, val in state["matchups"].items():
        off = int(key.split("-")[0])
        by_team[team_of[off]][key] = val
    (C.SITE_DATA / "matchups").mkdir(parents=True, exist_ok=True)
    for t, m in by_team.items():
        save_json(m, C.SITE_DATA / "matchups" / f"{t}.json", compact=True)
    save_json({k: v for k, v in state.items() if k != "matchups"}, C.SITE_DATA / "league.json", compact=True)
    save_json(games_payload(b, preds), C.SITE_DATA / "games.json", compact=True)
