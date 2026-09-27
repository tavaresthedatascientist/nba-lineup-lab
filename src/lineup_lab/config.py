"""Project-wide configuration: paths, league structure, and basketball assumptions.

Every constant that encodes a basketball judgment call lives here so a front
office can see (and change) the assumptions in one place.
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
SITE_DATA = ROOT / "site" / "data"

SEASON = "2025-26"
GAMELOGS_FILE = DATA_RAW / "player_gamelogs_2025-26.csv.gz"
RESULTS_FILE = DATA_RAW / "game_results_2021-22_to_2025-26.csv.gz"
# Optional: produced by `python -m lineup_lab fetch` (nba_api, run locally).
MATCHUPS_FILE = DATA_RAW / "boxscore_matchups_2025-26.csv.gz"
STARTERS_FILE = DATA_RAW / "starters_2025-26.csv.gz"

# NBA game-id prefixes (the 3rd-5th chars): 002 = regular season, 004 = playoffs,
# 005 = play-in. The NBA Cup final (006) does not count toward standings.
REGULAR_SEASON_PREFIX = "00225"
PLAYOFF_PREFIXES = ("00425", "00525")

# ---------------------------------------------------------------------------
# League structure
# ---------------------------------------------------------------------------
EAST = {"ATL", "BOS", "BKN", "CHA", "CHI", "CLE", "DET", "IND",
        "MIA", "MIL", "NYK", "ORL", "PHI", "TOR", "WAS"}
WEST = {"DAL", "DEN", "GSW", "HOU", "LAC", "LAL", "MEM", "MIN",
        "NOP", "OKC", "PHX", "POR", "SAC", "SAS", "UTA"}
TEAMS = sorted(EAST | WEST)

TEAM_NAMES = {
    "ATL": "Atlanta Hawks", "BOS": "Boston Celtics", "BKN": "Brooklyn Nets",
    "CHA": "Charlotte Hornets", "CHI": "Chicago Bulls", "CLE": "Cleveland Cavaliers",
    "DAL": "Dallas Mavericks", "DEN": "Denver Nuggets", "DET": "Detroit Pistons",
    "GSW": "Golden State Warriors", "HOU": "Houston Rockets", "IND": "Indiana Pacers",
    "LAC": "Los Angeles Clippers", "LAL": "Los Angeles Lakers", "MEM": "Memphis Grizzlies",
    "MIA": "Miami Heat", "MIL": "Milwaukee Bucks", "MIN": "Minnesota Timberwolves",
    "NOP": "New Orleans Pelicans", "NYK": "New York Knicks", "OKC": "Oklahoma City Thunder",
    "ORL": "Orlando Magic", "PHI": "Philadelphia 76ers", "PHX": "Phoenix Suns",
    "POR": "Portland Trail Blazers", "SAC": "Sacramento Kings", "SAS": "San Antonio Spurs",
    "TOR": "Toronto Raptors", "UTA": "Utah Jazz", "WAS": "Washington Wizards",
}
NAME_TO_ABBR = {v: k for k, v in TEAM_NAMES.items()}


def conference(team: str) -> str:
    return "East" if team in EAST else "West"


# ---------------------------------------------------------------------------
# Head-to-head window
# ---------------------------------------------------------------------------
# Same-conference teams meet 3-4x per season, cross-conference teams only 2x.
# Looking back 12 same-conference / 6 cross-conference meetings covers roughly
# the same 3-season span for both, so the H2H signal is equally "fresh".
H2H_SAME_CONF = 12
H2H_CROSS_CONF = 6

# ---------------------------------------------------------------------------
# Core ("backbone") players
# ---------------------------------------------------------------------------
# By default a team's core is its top-3 players by season-to-date
# PER x minutes-per-game (impact x role). Overrides encode front-office intent
# that pure box-score ranking can miss - e.g. Jalen Williams missed ~half the
# 2025-26 season, but he is unambiguously part of OKC's championship core.
CORE_SIZE = 3
CORE_OVERRIDES: dict[str, list[str]] = {
    "OKC": ["Shai Gilgeous-Alexander", "Jalen Williams", "Chet Holmgren"],
}

# ---------------------------------------------------------------------------
# Walk-forward evaluation
# ---------------------------------------------------------------------------
TRAIN_GAMES_PER_TEAM = 5    # train on a block of 5 games per team (~75 league games)
TEST_GAMES_PER_TEAM = 10    # ...then predict the next 10 games per team (~150 games)
STEP_GAMES_PER_TEAM = 10    # slide forward so every test block is new games

# ---------------------------------------------------------------------------
# Small-sample shrinkage (empirical-Bayes style priors)
# ---------------------------------------------------------------------------
# Early in the season a player's numbers are noisy. Each rate stat is blended
# with the league average as if the player had PRIOR_MINUTES of average play.
PRIOR_MINUTES = 150.0
LEAGUE_PRIOR = {"PER": 15.0, "TS": 0.575, "USG": 20.0}
# Matchup splits (player vs. a specific defender) are shrunk toward the
# player's own baseline as if he had MATCHUP_PRIOR_GAMES "average" games.
MATCHUP_PRIOR_GAMES = 3.0

# Minimum minutes for a player-game to count as a real appearance.
MIN_MINUTES = 5.0

# Lineup optimizer constraints (domain guardrails, not learned).
MAX_LINEUP_USG = 125.0      # five high-usage players don't share one ball
MIN_BALLHANDLERS = 1        # at least one primary creator (role = G)
MIN_BIGS = 1                # at least one big (role = B)
ROSTER_MIN_GAMES = 5        # candidate must have played >= 5 games for the team
