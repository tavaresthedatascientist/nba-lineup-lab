"""Features must only use information available before tip-off."""
import pandas as pd

from lineup_lab import config as C
from lineup_lab.features import profiles_asof


def test_profile_uses_only_prior_games(bundle):
    pg = bundle.pg
    sga = pg[pg["PLAYER_NAME"] == "Shai Gilgeous-Alexander"].sort_values("GAME_DATE")
    d = sga["GAME_DATE"].iloc[20]
    prior = sga[sga["GAME_DATE"] < d]
    prof = profiles_asof(bundle.timeline, pd.DataFrame({"PLAYER_ID": [sga["PLAYER_ID"].iloc[0]], "GAME_DATE": [d]}))
    ts_raw = prior["PTS"].sum() / (2 * (prior["FGA"].sum() + 0.44 * prior["FTA"].sum()))
    w = prior["MIN"].sum() / (prior["MIN"].sum() + C.PRIOR_MINUTES)
    assert abs(prof["TS"].iloc[0] - (w * ts_raw + (1 - w) * C.LEAGUE_PRIOR["TS"])) < 1e-9
    assert prof["GP"].iloc[0] == len(prior)


def test_first_game_profiles_are_priors(bundle):
    t = bundle.table
    first = t[t["GAME_DATE"] == t["GAME_DATE"].min()]      # opening night: nobody has played
    # season-to-date stats don't exist yet, so both lineups look league-average
    assert first["LU_PER_DIFF"].abs().max() < 1e-9
    assert first["MATCHUP_DIFF"].abs().max() < 1e-9


def test_h2h_window_sizes(bundle):
    t = bundle.table
    same = t.apply(lambda r: C.conference(r.TEAM) == C.conference(r.OPP), axis=1)
    assert t.loc[same, "H2H_N"].max() <= C.H2H_SAME_CONF
    assert t.loc[~same, "H2H_N"].max() <= C.H2H_CROSS_CONF
    assert t.loc[same, "H2H_N"].max() == C.H2H_SAME_CONF
    assert t.loc[~same, "H2H_N"].max() == C.H2H_CROSS_CONF


def test_matchup_history_is_prior_only(bundle):
    mh = bundle.mhist
    # cumulative counts must be strictly increasing within a pair over time
    g = mh.sort_values("GAME_DATE").groupby(["OFF_PLAYER_ID", "DEF_PLAYER_ID"])["N"]
    assert (g.diff().dropna() > 0).all()
