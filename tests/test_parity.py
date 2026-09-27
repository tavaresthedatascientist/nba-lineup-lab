"""The website's JavaScript model must give the same answers as Python."""
import json
import pickle
import shutil
import subprocess

import pytest

from lineup_lab import config as C
from lineup_lab.optimizer import optimize, win_probability

MODELS = C.DATA_PROCESSED / "models.pkl"
CASES = [("OKC", "DEN", True), ("OKC", "SAS", False), ("BOS", "NYK", None), ("LAL", "GSW", True)]

JS = r"""
const fs = require('fs');
const { LineupModel } = require(process.argv[2] + '/js/model.js');
const d = process.argv[2] + '/data/';
const rd = (f) => JSON.parse(fs.readFileSync(d + f));
const m = new LineupModel(rd('model.json'), rd('league.json'), rd('meta.json'));
const cases = JSON.parse(process.argv[3]);
const out = cases.map(([t, o, h]) => {
  m.addMatchups(rd('matchups/' + t + '.json')); m.addMatchups(rd('matchups/' + o + '.json'));
  const T = m.league.teams[t], O = m.league.teams[o];
  const p = m.winProbability(t, o, T.default5, O.default5, h).p;
  const best = m.optimize(t, o, O.default5, h).best.map(b => [b.p, b.ids]);
  return { p, best };
});
console.log(JSON.stringify(out));
"""


@pytest.mark.skipif(shutil.which("node") is None or not MODELS.exists(),
                    reason="needs node and `python -m lineup_lab all`")
def test_js_matches_python(tmp_path):
    m = pickle.loads(MODELS.read_bytes())
    forest, state = m["forest"], m["state"]
    script = tmp_path / "parity.js"
    script.write_text(JS)
    res = subprocess.run(["node", str(script), str(C.ROOT / "site"), json.dumps(CASES)],
                         capture_output=True, text=True, check=True)
    js = json.loads(res.stdout)
    for (t, o, h), j in zip(CASES, js):
        T, O = state["teams"][t], state["teams"][o]
        p = win_probability(forest, state, t, o, T["default5"], O["default5"], h)
        assert p == pytest.approx(j["p"], abs=1e-6)
        py_best = optimize(forest, state, t, o, home=h)["best"]
        assert [round(b["win_prob"], 4) for b in py_best] == [round(b[0], 4) for b in j["best"]]
