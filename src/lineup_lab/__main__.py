"""Command-line entry point.

    python -m lineup_lab all                         # features -> backtest -> models -> reports -> site data
    python -m lineup_lab optimize --team OKC --opp DEN --home
    python -m lineup_lab fetch                       # refresh raw data from stats.nba.com (run locally)
"""
from __future__ import annotations

import argparse
import json
import pickle
import time

from . import config as C

BUNDLE = C.DATA_PROCESSED / "bundle.pkl"
MODELS = C.DATA_PROCESSED / "models.pkl"


def _fmt_pct(x):
    return f"{100 * x:.1f}%"


def write_reports(rolling, expanding, playoffs, tree, forest) -> None:
    from .model import importances, save_json, tree_rules

    C.REPORTS.mkdir(parents=True, exist_ok=True)
    lines = ["# Walk-forward backtest - 2025-26 NBA season", "",
             "All numbers are **out-of-sample**: every prediction is made by a model that never saw",
             "that game, using features built only from games played before it.", "",
             "## Summary (regular season, games 6-82 for each team)", "",
             "| Protocol | Model | Games | Accuracy | Log loss | Brier |",
             "|---|---|---|---|---|---|"]
    for label, bt in [("Rolling: train 5 games / test next 10", rolling),
                      ("Expanding: train all prior / test next 10", expanding)]:
        for m in ("decision_tree", "random_forest"):
            p = bt["pooled"][m]
            lines.append(f"| {label} | {m.replace('_', ' ')} | {p['games']} | {_fmt_pct(p['accuracy'])} "
                         f"| {p['log_loss']:.3f} | {p['brier']:.3f} |")
    b = rolling["pooled"]["baselines"]
    lines += ["", "Baselines on the same games: "
              f"home team wins {_fmt_pct(b['home_team'])}, better record wins {_fmt_pct(b['better_record'])}, "
              f"better last-10 net rating wins {_fmt_pct(b['better_net_rating_l10'])}.", "",
              "## Playoff holdout (trained on the full regular season)", "",
              "| Model | Games | Accuracy | Log loss |", "|---|---|---|---|"]
    for m in ("decision_tree", "random_forest"):
        p = playoffs[m]
        lines.append(f"| {m.replace('_', ' ')} | {p['games']} | {_fmt_pct(p['accuracy'])} | {p['log_loss']:.3f} |")
    pb = playoffs["baselines"]
    lines += ["", f"Playoff baselines: home team {_fmt_pct(pb['home_team'])}, better record {_fmt_pct(pb['better_record'])}.",
              "", "## Fold detail (rolling 5 / 10)", "",
              "| Train games | Test games | Test n | Tree acc. | Forest acc. | Better-record acc. |",
              "|---|---|---|---|---|---|"]
    for f in rolling["folds"]:
        lines.append(f"| {f['train_games'][0] + 1}-{f['train_games'][1]} | {f['test_games'][0] + 1}-{f['test_games'][1]} "
                     f"| {f['test_games_n']} | {_fmt_pct(f['decision_tree']['accuracy'])} "
                     f"| {_fmt_pct(f['random_forest']['accuracy'])} | {_fmt_pct(f['baselines']['better_record'])} |")
    lines += ["", "## Feature importance (random forest, full regular season)", ""]
    lines += [f"- `{k}`: {v:.3f}" for k, v in importances(forest).items()]
    (C.REPORTS / "backtest.md").write_text("\n".join(lines) + "\n")
    (C.REPORTS / "decision_tree_rules.txt").write_text(tree_rules(tree))
    save_json({k: {kk: vv for kk, vv in v.items() if kk != "predictions"}
               for k, v in {"rolling": rolling, "expanding": expanding, "playoffs": playoffs}.items()},
              C.REPORTS / "backtest.json")


def cmd_all(args) -> None:
    from .export_site import export
    from .features import build
    from .model import fit_final, playoff_holdout, walk_forward
    from .optimizer import league_state

    t0 = time.time()
    print("1/5 building leakage-free features ...")
    b = build()
    print(f"    {len(b.table):,} team-game rows | starters: {b.starter_source}")
    print("2/5 walk-forward backtests (train 5 / test 10, and expanding) ...")
    rolling = walk_forward(b.table, expanding=False)
    expanding = walk_forward(b.table, expanding=True)
    playoffs = playoff_holdout(b.table)
    for name, bt in [("rolling 5/10", rolling), ("expanding", expanding)]:
        p = bt["pooled"]
        print(f"    {name:13s} tree {_fmt_pct(p['decision_tree']['accuracy'])} | forest "
              f"{_fmt_pct(p['random_forest']['accuracy'])} | better-record baseline "
              f"{_fmt_pct(p['baselines']['better_record'])}")
    print("3/5 fitting final models on the full regular season ...")
    tree, forest = fit_final(b.table)
    print("4/5 writing reports/ ...")
    write_reports(rolling, expanding, playoffs, tree, forest)
    print("5/5 exporting site data ...")
    state = league_state(b)
    export(b, tree, forest, state, rolling, expanding, playoffs)
    C.DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    with open(MODELS, "wb") as fh:
        pickle.dump({"tree": tree, "forest": forest, "state": state}, fh)
    print(f"done in {time.time() - t0:.0f}s")


def cmd_optimize(args) -> None:
    from .optimizer import optimize

    if not MODELS.exists():
        raise SystemExit("Run `python -m lineup_lab all` first.")
    m = pickle.loads(MODELS.read_bytes())
    state = m["state"]
    team, opp = args.team.upper(), args.opp.upper()
    name_to_id = {p["name"]: p["id"] for t in state["teams"].values() for p in t["players"]}
    out_ids = [name_to_id[n] for n in args.out if n in name_to_id]
    home = None if args.neutral else args.home
    res = optimize(m["forest"], state, team, opp, home=home, unavailable=out_ids, top_k=args.top)
    if args.json:
        print(json.dumps(res, indent=2))
        return
    where = "neutral site" if home is None else ("home" if home else "away")
    print(f"\n{C.TEAM_NAMES[team]} vs {C.TEAM_NAMES[opp]} ({where})")
    print(f"Core (always starts when available): {', '.join(res['core'])}")
    print(f"Opponent starting five: {', '.join(res['opp_lineup'])}")
    if res["default_win_prob"] is not None:
        print(f"Most-used recent five: {', '.join(res['default_lineup'])} -> {_fmt_pct(res['default_win_prob'])}")
    print(f"\nTop lineups ({res['evaluated']} valid lineups evaluated):")
    for i, r in enumerate(res["best"], 1):
        print(f"  {i}. {_fmt_pct(r['win_prob'])}  {', '.join(r['lineup'])}")


def cmd_fetch(args) -> None:
    from .data.fetch_nba import run
    run(skip_matchups=args.skip_matchups)


def main() -> None:
    ap = argparse.ArgumentParser(prog="lineup_lab")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("all", help="build features, backtest, fit, report, export site data").set_defaults(fn=cmd_all)
    o = sub.add_parser("optimize", help="best starting five vs. an opponent")
    o.add_argument("--team", default="OKC")
    o.add_argument("--opp", required=True)
    g = o.add_mutually_exclusive_group()
    g.add_argument("--home", dest="home", action="store_true", default=True)
    g.add_argument("--away", dest="home", action="store_false")
    g.add_argument("--neutral", action="store_true")
    o.add_argument("--out", nargs="*", default=[], help="players unavailable tonight")
    o.add_argument("--top", type=int, default=5)
    o.add_argument("--json", action="store_true")
    o.set_defaults(fn=cmd_optimize)
    f = sub.add_parser("fetch", help="refresh raw data from stats.nba.com (needs nba_api, run locally)")
    f.add_argument("--skip-matchups", action="store_true")
    f.set_defaults(fn=cmd_fetch)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
