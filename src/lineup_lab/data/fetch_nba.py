"""Refresh the raw data straight from stats.nba.com with `nba_api`.

Run this on your own machine (stats.nba.com blocks most cloud/CI IP ranges):

    pip install nba_api
    python -m lineup_lab fetch                 # game logs + starters + matchups
    python -m lineup_lab fetch --skip-matchups # faster: game logs + starters only

It writes three files into data/raw/ that the rest of the pipeline picks up
automatically:

* player_gamelogs_2025-26.csv.gz   - same schema as the bundled snapshot
* starters_2025-26.csv.gz          - true starting fives (BoxScoreTraditionalV3)
* boxscore_matchups_2025-26.csv.gz - who guarded whom, in partial possessions
                                     (BoxScoreMatchupsV3). When present, the
                                     "primary defender" feature uses real
                                     tracking data instead of the positional proxy.

The box-score endpoints are called once per game (~1,300 calls), so the script
sleeps between requests and caches finished games; it is safe to stop and rerun.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd

from .. import config as C

CACHE = C.DATA_RAW / "_cache"
SLEEP = 0.7  # seconds between calls - be polite to stats.nba.com


def _require_nba_api():
    try:
        import nba_api  # noqa: F401
    except ImportError as e:  # pragma: no cover
        raise SystemExit("nba_api is not installed. Run: pip install nba_api") from e


def fetch_gamelogs(season: str = C.SEASON) -> pd.DataFrame:
    from nba_api.stats.endpoints import playergamelogs

    frames = []
    for season_type in ("Regular Season", "PlayIn", "Playoffs"):
        df = playergamelogs.PlayerGameLogs(
            season_nullable=season, season_type_nullable=season_type, timeout=60
        ).get_data_frames()[0]
        frames.append(df)
        time.sleep(SLEEP)
    g = pd.concat(frames, ignore_index=True)
    keep = ["SEASON_YEAR", "PLAYER_ID", "PLAYER_NAME", "TEAM_ID", "TEAM_ABBREVIATION",
            "GAME_ID", "GAME_DATE", "MATCHUP", "WL", "MIN", "FGM", "FGA", "FG3M", "FG3A",
            "FTM", "FTA", "OREB", "DREB", "REB", "AST", "TOV", "STL", "BLK", "PF", "PTS",
            "PLUS_MINUS"]
    return g[keep]


def _cached(kind: str, game_id: str, fn):
    path = CACHE / kind / f"{game_id}.json"
    if path.exists():
        return pd.DataFrame(json.loads(path.read_text()))
    df = fn()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(df.to_json(orient="records"))
    time.sleep(SLEEP)
    return df


def fetch_starters(game_ids: list[str]) -> pd.DataFrame:
    """A player started if BoxScoreTraditionalV3 lists a position for him."""
    from nba_api.stats.endpoints import boxscoretraditionalv3

    rows = []
    for i, gid in enumerate(game_ids, 1):
        df = _cached("traditional", gid, lambda: boxscoretraditionalv3.BoxScoreTraditionalV3(
            game_id=gid, timeout=60).get_data_frames()[0])
        pos_col = "position" if "position" in df.columns else "START_POSITION"
        pid_col = "personId" if "personId" in df.columns else "PLAYER_ID"
        started = df[df[pos_col].fillna("").astype(str).str.len() > 0]
        rows += [{"GAME_ID": gid, "PLAYER_ID": int(p)} for p in started[pid_col]]
        if i % 100 == 0:
            print(f"  starters: {i}/{len(game_ids)} games")
    return pd.DataFrame(rows)


def fetch_matchups(game_ids: list[str]) -> pd.DataFrame:
    """Offensive player x defensive player partial possessions for every game."""
    from nba_api.stats.endpoints import boxscorematchupsv3

    frames = []
    for i, gid in enumerate(game_ids, 1):
        df = _cached("matchups", gid, lambda: boxscorematchupsv3.BoxScoreMatchupsV3(
            game_id=gid, timeout=60).get_data_frames()[0])
        if df.empty:
            continue
        cols = {c.lower(): c for c in df.columns}
        off = cols.get("personid") or cols.get("person_id_off") or "personId"
        dfn = cols.get("personiddef") or cols.get("matchup_person_id") or "personIdDef"
        poss = cols.get("partialpossessions") or cols.get("matchupminutes") or "partialPossessions"
        frames.append(pd.DataFrame({
            "GAME_ID": gid,
            "OFF_PLAYER_ID": df[off].astype(int),
            "DEF_PLAYER_ID": df[dfn].astype(int),
            "PARTIAL_POSS": pd.to_numeric(df[poss], errors="coerce").fillna(0.0),
        }))
        if i % 100 == 0:
            print(f"  matchups: {i}/{len(game_ids)} games")
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def run(season: str = C.SEASON, skip_matchups: bool = False) -> None:
    _require_nba_api()
    C.DATA_RAW.mkdir(parents=True, exist_ok=True)
    print(f"Fetching {season} player game logs ...")
    g = fetch_gamelogs(season)
    g.to_csv(C.GAMELOGS_FILE, index=False)
    game_ids = sorted(g["GAME_ID"].astype(str).str.zfill(10).unique())
    print(f"  {len(g):,} player-games across {len(game_ids):,} games")

    print("Fetching true starting lineups ...")
    fetch_starters(game_ids).to_csv(C.STARTERS_FILE, index=False)

    if not skip_matchups:
        print("Fetching defensive matchup (who-guarded-whom) data ...")
        fetch_matchups(game_ids).to_csv(C.MATCHUPS_FILE, index=False)
    print("Done. Re-run `python -m lineup_lab all` to rebuild features, models and site data.")


if __name__ == "__main__":  # pragma: no cover
    run()
