# Lineup Lab

**Decision-tree lineup optimization and game prediction for the 2025-26 NBA season.**

[Play the prediction game →](https://tavaresthedatascientist.github.io/nba-lineup-lab/) · [Backtest report](reports/backtest.md) · [OKC lineup examples](reports/okc_lineup_examples.md)

Lineup Lab answers one question a coaching staff asks before every game: *given who the other team is starting tonight, which five should we start?* It learns what wins from the 2025-26 season, builds each lineup around the team's core players (for OKC: Shai Gilgeous-Alexander, Jalen Williams and Chet Holmgren), and scores every valid starting five against the opponent's.

It works for any of the 30 teams, and the website turns it into a game: pick winners of real 2025-26 games and see whether you can beat the model.

---

## What the model looks at

Each game becomes one row per team. Every feature is the **difference** between that team's starting five and the opponent's, computed only from games played *before* tip-off.

| Feature | What it captures |
|---|---|
| Lineup PER | Average Player Efficiency Rating (Hollinger, pace-adjusted) of the starting five |
| Lineup TS% | True shooting %, weighted by usage, so the efficiency of the players taking the shots counts most |
| Lineup usage | Total usage rate: five ball-dominant players don't share one ball |
| Core starting / core PER | How many of the team's 3 core players start, and how well they are playing |
| Best player PER | The best player on the floor |
| Matchup edge | Each player's true shooting against his likely primary defender in past meetings, relative to his own average |
| Head-to-head | Win % over the last **12** meetings (**6** if the opponent is in the other conference) |
| Win %, net rating (season and last 10) | Team quality and current form |
| Rest, home court | Schedule context |

Why 12 and 6 meetings: same-conference teams play 3-4 times a season and cross-conference teams twice, so both windows cover roughly the last three seasons.

### Player metrics are computed, not downloaded

PER, TS% and USG% are rebuilt from raw box scores so they can be measured *as of any date*. They match Basketball-Reference's published 2025-26 values (checked in `tests/test_metrics.py`):

| Player | PER (ours / BBRef) | TS% (ours / BBRef) | USG% (ours / BBRef) |
|---|---|---|---|
| Shai Gilgeous-Alexander | 31.0 / 30.8 | .665 / .665 | 33.7 / 33.4 |
| Chet Holmgren | 21.6 / 21.6 | .653 / .653 | 22.1 / 21.9 |
| Luguentz Dort | 8.3 / 8.3 | .524 / .524 | 14.2 / 14.1 |
| Cason Wallace | 12.0 / 12.1 | .538 / .538 | 14.6 / 14.5 |

Early in the season each player is blended with league average as if he had 150 minutes of average play, so a hot first week doesn't make a bench player look like an All-Star.

## How it's evaluated

The brief was to **train on 5 games and test on the next 10**. The backtest does exactly that, for every team at once: train on a block of 5 games per team (~75 league games), predict the next 10 per team (~150 games), slide forward 10, repeat through game 82. Every prediction is out-of-sample.

| Protocol (regular season, 1,150 games) | Decision tree | Random forest | Better record wins |
|---|---|---|---|
| Rolling: train 5 / test next 10 | 64.7% | 66.6% | 67.7% |
| Expanding: train all prior / test next 10 | 67.7% | **68.1%** | 67.7% |
| Postseason holdout (91 games) | 59.3% | 61.5% | 61.5% |

What this shows:

- **A 5-game training window is too small for a tree.** 150 rows is thin, so the rolling model sits just under the simple "better record wins" rule. Same test blocks, expanding training window: the forest edges past it and has clearly better calibrated probabilities (log loss 0.604 vs. 0.614). The site uses the expanding version for its picks.
- **Beating "better record wins" is hard in the NBA.** It is right about two-thirds of the time, and published game-level models typically land in the high 60s. The value of this model is less in raw accuracy than in being **lineup-aware**: it can price a specific five against a specific five, which a team record can't.
- **Lineup PER and season net rating do most of the work** (feature importance in [the report](reports/backtest.md)). Matchup edges and head-to-head add a little; they are noisy at one season of data.

The single depth-3 decision tree is kept alongside the forest because a coach can read it: [decision_tree_rules.txt](reports/decision_tree_rules.txt).

## The optimizer

```bash
python -m lineup_lab optimize --team OKC --opp DEN --away
python -m lineup_lab optimize --team OKC --opp SAS --home --out "Jalen Williams"
```

1. Start every available core player.
2. Try every way to fill the remaining spots from the top 11 players by minutes.
3. Drop lineups that break basketball rules of thumb: at least one guard, at least one big, combined usage ≤ 125%.
4. Score each against the opponent's starting five and rank by win probability.

Example (OKC at Denver, full roster): the model rates SGA / Jalen Williams / Chet Holmgren / Ajay Mitchell / Isaiah Hartenstein at **78.4%**, against 67.5% for OKC's most-used five from late in the season, the group without Jalen Williams. [More examples](reports/okc_lineup_examples.md).

## The website

`site/` is a static site (no server, no build step) deployed to GitHub Pages:

- **Beat the Model** · 10 real 2025-26 games with both starting fives, records and head-to-head. Pick the winner, then see the model's pre-game probability and the result.
- **Lineup Builder** · Pick your team and an opponent, mark players out, build a five and watch the win probability move, or let the optimizer find the best five.
- **How it works** · Backtest results, feature importance and the decision tree.

The browser runs the **same** random forest as Python (exported tree by tree to `site/data/model.json`); `tests/test_parity.py` runs the JavaScript under Node and checks it matches scikit-learn.

## Run it

```bash
git clone https://github.com/tavaresthedatascientist/nba-lineup-lab.git
cd nba-lineup-lab
pip install -r requirements.txt
pip install -e .

python -m lineup_lab all        # features → backtest → models → reports → site data (~1 min)
pytest                          # metric accuracy, leakage, walk-forward, JS/Python parity
python -m http.server -d site   # open http://localhost:8000
```

### Refresh the data from stats.nba.com

The repo ships with a 2025-26 snapshot so everything runs offline. To pull fresh data, including **official starting lineups** and **NBA tracking data on who guarded whom**, run this on your own machine (stats.nba.com blocks most cloud servers):

```bash
pip install nba_api
python -m lineup_lab fetch      # ~1,300 box-score calls, cached, safe to re-run
python -m lineup_lab all
```

With the tracking data present, the "primary defender" feature switches from the positional-slot approximation to the defender with the most partial possessions against each player.

## Project layout

```
src/lineup_lab/
  config.py          every basketball assumption in one place (core overrides, H2H windows, priors, constraints)
  data/load.py       cleaning, team box scores, possessions, roles (G/W/B)
  data/fetch_nba.py  nba_api refresh: game logs, starters, defensive matchups
  features.py        PER / TS% / USG% timelines, matchup history, H2H, core players, lineup features
  model.py           decision tree + random forest, walk-forward backtest, export
  optimizer.py       lineup search around the core
  export_site.py     JSON for the website
site/                static website (GitHub Pages)
reports/             backtest, tree rules, OKC examples
tests/               pytest suite
```

## Data and known limits

- **Source:** 2025-26 player game logs (stats.nba.com via `nba_api`) and 2021-22 to 2025-26 game results, from the MIT-licensed [wippa-nba-data](https://github.com/wippa-studios/wippa-nba-data) snapshot. All 1,230 regular-season games are present; OKC's 64-18 record reproduces exactly.
- **Missing rows in the snapshot:** about a fifth of team-games are missing one or more player rows. Results come from the official W/L flag, and team box totals for those games are scaled to 240 minutes so ratings aren't dragged down. `python -m lineup_lab fetch` replaces the snapshot with complete data.
- **Starters** are approximated as the five players with the most minutes unless the official-starters file has been fetched.
- **Defense is under-weighted.** PER, TS% and USG% are offensive-leaning, so the optimizer tends to prefer shooters to stoppers like Lu Dort. The next step is adding defensive inputs (matchup FG% allowed from the tracking data, defensive rebounding, rim deterrence).
- **Opponent starters must be known.** Starting lineups are announced about 30 minutes before tip-off, which is when this tool would be used.

## How this was built

Built with an AI coding assistant (Claude) working from my specification: decision trees, the 2025-26 season, train on 5 games and test on the next 10, the opponent's starting five, head-to-head over the last 12 meetings (6 cross-conference), performance against the primary defender, and lineups built around each team's best players. The judgment calls made along the way (shrinkage priors, the usage cap, how core players are chosen, why the expanding window is also reported) are written down in `config.py` and in this README so they can be challenged.

## License

MIT for the code. Bundled data keeps its source license (`data/raw/SOURCE_LICENSE_wippa-nba-data.txt`). Not affiliated with the NBA or any team.
