"""Decision-tree game models and the 5-train / 10-test walk-forward backtest."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.tree import DecisionTreeClassifier, export_text

from . import config as C
from .features import FEATURES


def make_tree() -> DecisionTreeClassifier:
    """Single interpretable tree. Shallow + min leaf size: a 5-game training
    block is only ~150 team-game rows, so anything deeper memorizes noise."""
    return DecisionTreeClassifier(max_depth=3, min_samples_leaf=10, random_state=7)


def make_forest() -> RandomForestClassifier:
    """Bagged decision trees: same building block, smoother probabilities
    (a single tree's probability is a step function, which is useless for
    ranking near-identical lineups)."""
    return RandomForestClassifier(n_estimators=300, max_depth=4, min_samples_leaf=5,
                                  max_features=0.5, random_state=7, n_jobs=-1)


MODELS = {"decision_tree": make_tree, "random_forest": make_forest}


def game_level(table: pd.DataFrame, p: np.ndarray) -> pd.DataFrame:
    """Collapse the two team-perspective rows of each game into one
    symmetric probability for the reference (home) team:
        p_ref = (p(ref row) + 1 - p(other row)) / 2
    """
    t = table[["GAME_ID", "GAME_DATE", "TEAM", "OPP", "IS_HOME", "WIN",
               "WINPCT_DIFF", "NET_L10_DIFF"]].copy()
    t["P"] = p
    t = t.sort_values(["GAME_ID", "IS_HOME", "TEAM"], ascending=[True, False, True])
    ref = t.groupby("GAME_ID").nth(0).set_index("GAME_ID")
    oth = t.groupby("GAME_ID").nth(1).set_index("GAME_ID")
    ref = ref.loc[ref.index.intersection(oth.index)]
    ref["P_REF"] = (ref["P"] + 1 - oth.loc[ref.index, "P"]) / 2
    return ref.reset_index()


def _metrics(g: pd.DataFrame) -> dict:
    y = g["WIN"].to_numpy()
    p = g["P_REF"].clip(1e-4, 1 - 1e-4).to_numpy()
    return {
        "games": int(len(g)),
        "accuracy": float(((p >= 0.5) == y).mean()),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "brier": float(brier_score_loss(y, p)),
    }


def _baselines(g: pd.DataFrame) -> dict:
    y = g["WIN"].to_numpy()
    home = np.ones(len(g), dtype=int)                  # ref team = home team (neutral sites: alphabetical)
    better_rec = (g["WINPCT_DIFF"] >= 0).astype(int).to_numpy()
    better_net = (g["NET_L10_DIFF"] >= 0).astype(int).to_numpy()
    return {"home_team": float((home == y).mean()),
            "better_record": float((better_rec == y).mean()),
            "better_net_rating_l10": float((better_net == y).mean())}


def walk_forward(table: pd.DataFrame, expanding: bool = False,
                 train_n: int = C.TRAIN_GAMES_PER_TEAM, test_n: int = C.TEST_GAMES_PER_TEAM,
                 step: int = C.STEP_GAMES_PER_TEAM) -> dict:
    """Train on 5 games (per team), test on the next 10, slide forward 10.

    A game's position in the season is the average of the two teams' game
    numbers, so a 'block of 5 games per team' is ~75 league games (~150
    team-perspective rows) and a test block of 10 is ~150 games.
    With expanding=True the training set is every earlier game instead.
    """
    rs = table[table["GAME_TYPE"] == "RS"].copy()
    gno = rs.groupby("GAME_ID")["TEAM_GAME_NO"].transform("mean")
    rs["BLOCK_POS"] = gno
    out = {"folds": [], "pooled": {}}
    pooled = {name: [] for name in MODELS}
    k = 0
    while k + train_n < 82:
        tr_lo = 0 if expanding else k
        tr = rs[(rs["BLOCK_POS"] >= tr_lo) & (rs["BLOCK_POS"] < k + train_n)]
        te = rs[(rs["BLOCK_POS"] >= k + train_n) & (rs["BLOCK_POS"] < k + train_n + test_n)]
        if te.empty:
            break
        fold = {"train_games": [tr_lo, k + train_n], "test_games": [k + train_n, min(82, k + train_n + test_n)],
                "train_rows": int(len(tr)), "test_games_n": int(te["GAME_ID"].nunique())}
        for name, make in MODELS.items():
            m = make().fit(tr[FEATURES], tr["WIN"])
            g = game_level(te, m.predict_proba(te[FEATURES])[:, 1])
            fold[name] = _metrics(g)
            pooled[name].append(g)
        fold["baselines"] = _baselines(game_level(te, np.full(len(te), 0.5)))
        out["folds"].append(fold)
        k += step
    for name, gs in pooled.items():
        out["pooled"][name] = _metrics(pd.concat(gs))
    allg = pd.concat(pooled["random_forest"])
    out["pooled"]["baselines"] = _baselines(allg)
    out["predictions"] = allg[["GAME_ID", "TEAM", "OPP", "P_REF", "WIN"]].reset_index(drop=True)
    return out


def playoff_holdout(table: pd.DataFrame) -> dict:
    rs = table[table["GAME_TYPE"] == "RS"]
    po = table[table["GAME_TYPE"] == "PO"]
    res = {}
    for name, make in MODELS.items():
        m = make().fit(rs[FEATURES], rs["WIN"])
        g = game_level(po, m.predict_proba(po[FEATURES])[:, 1])
        res[name] = _metrics(g)
    res["baselines"] = _baselines(game_level(po, np.full(len(po), 0.5)))
    res["predictions"] = g[["GAME_ID", "TEAM", "OPP", "P_REF", "WIN"]].reset_index(drop=True)
    return res


def fit_final(table: pd.DataFrame):
    """Production models trained on the full regular season."""
    rs = table[table["GAME_TYPE"] == "RS"]
    tree = make_tree().fit(rs[FEATURES], rs["WIN"])
    forest = make_forest().fit(rs[FEATURES], rs["WIN"])
    return tree, forest


def tree_rules(tree: DecisionTreeClassifier) -> str:
    return export_text(tree, feature_names=FEATURES, decimals=2, show_weights=True)


def importances(model) -> dict:
    return {f: round(float(v), 4) for f, v in
            sorted(zip(FEATURES, model.feature_importances_), key=lambda x: -x[1])}


def forest_to_json(forest: RandomForestClassifier) -> dict:
    """Serialize every tree as flat arrays so the website can evaluate the
    exact same model in the browser (see site/js/model.js)."""
    trees = []
    for est in forest.estimators_:
        t = est.tree_
        val = t.value[:, 0, :]
        p_win = (val[:, 1] / val.sum(axis=1)).round(5)
        trees.append({
            "f": t.feature.tolist(), "t": t.threshold.tolist(),  # float32 split values, kept exact
            "l": t.children_left.tolist(), "r": t.children_right.tolist(),
            "p": p_win.tolist(),
        })
    return {"features": FEATURES, "trees": trees}


def tree_to_json(tree: DecisionTreeClassifier) -> dict:
    """Nested representation of the single tree for the site's explainer."""
    t = tree.tree_

    def node(i):
        val = t.value[i, 0, :]
        d = {"n": int(t.n_node_samples[i]), "p": round(float(val[1] / val.sum()), 3)}
        if t.children_left[i] != -1:
            d.update({"feature": FEATURES[t.feature[i]], "threshold": round(float(t.threshold[i]), 3),
                      "left": node(t.children_left[i]), "right": node(t.children_right[i])})
        return d
    return node(0)


def save_json(obj, path, compact: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if compact:
        path.write_text(json.dumps(obj, separators=(",", ":"), default=str))
    else:
        path.write_text(json.dumps(obj, indent=1, default=str))
