"""Our advanced metrics, computed from raw box scores, should match the
published Basketball-Reference 2025-26 regular-season values."""
import pytest

# (PER, TS%, USG%) from basketball-reference.com/leagues/NBA_2026_advanced.html
BBREF = {
    "Shai Gilgeous-Alexander": (30.8, 0.665, 33.4),
    "Chet Holmgren": (21.6, 0.653, 21.9),
    "Luguentz Dort": (8.3, 0.524, 14.1),
    "Cason Wallace": (12.1, 0.538, 14.5),
    "Jaylin Williams": (13.8, 0.597, 15.6),
}


@pytest.mark.parametrize("name", list(BBREF))
def test_matches_basketball_reference(bundle, name):
    tl = bundle.timeline
    rs = tl[tl["GAME_ID"].str.startswith("00225")].sort_values("GAME_DATE")
    row = rs[rs["PLAYER_NAME"] == name].iloc[-1]
    per, ts, usg = BBREF[name]
    assert row["PER_RAW"] == pytest.approx(per, abs=0.5)
    assert row["TS_RAW"] == pytest.approx(ts, abs=0.003)
    assert row["USG_RAW"] == pytest.approx(usg, abs=0.6)


def test_regular_season_complete(bundle):
    rs = bundle.tg[bundle.tg["GAME_TYPE"] == "RS"]
    assert rs["GAME_ID"].nunique() == 1230
    assert (rs.groupby("TEAM").size() == 82).all()
    okc = rs[rs["TEAM"] == "OKC"]
    assert okc["WIN"].sum() == 64   # 64-18
