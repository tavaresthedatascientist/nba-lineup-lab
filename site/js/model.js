/* Lineup Lab - browser implementation of the Python model.
 *
 * Mirrors src/lineup_lab/features.py (lineup_side_features, game_feature_vector)
 * and src/lineup_lab/optimizer.py (win_probability, optimize). The random
 * forest is the exact scikit-learn model, exported tree-by-tree to model.json.
 * tests/test_parity.py runs this file under Node and checks it matches Python.
 */
(function (root) {
  "use strict";

  function matchupEdge(n, s, prior) {
    return 100.0 * s / (n + prior);
  }

  function assignDefenders(lineup, oppLineup) {
    const a = lineup.slice().sort((x, y) => x.pos - y.pos);
    const b = oppLineup.slice().sort((x, y) => x.pos - y.pos);
    return a.map((x, i) => ({ off: x.id, def: b[i].id }));
  }

  function lineupSideFeatures(lineup, oppLineup, coreIds, mhist, cfg) {
    let usgSum = 0, tsNum = 0, perSum = 0, star = -Infinity, coreN = 0, corePer = 0;
    for (const p of lineup) {
      usgSum += p.usg; tsNum += p.usg * p.ts; perSum += p.per;
      if (p.per > star) star = p.per;
      if (coreIds.includes(p.id)) { coreN += 1; corePer += p.per; }
    }
    const edges = {};
    for (const pair of assignDefenders(lineup, oppLineup)) {
      const [n, s] = mhist(pair.off, pair.def);
      edges[pair.off] = matchupEdge(n, s, cfg.matchup_prior);
    }
    let m = 0;
    for (const p of lineup) m += p.usg * edges[p.id];
    return {
      LU_PER: perSum / 5.0,
      LU_TS: 100.0 * (tsNum / usgSum),
      LU_USG: usgSum,
      CORE_N: coreN,
      CORE_PER: corePer / cfg.core_size,
      STAR_PER: star,
      MATCHUP: m / usgSum,
      _edges: edges,
    };
  }

  const SIDE_KEYS = ["LU_PER", "LU_TS", "LU_USG", "CORE_N", "CORE_PER", "STAR_PER", "MATCHUP"];

  function gameFeatureVector(side, oppSide, ctx) {
    const d = SIDE_KEYS.map((k) => side[k] - oppSide[k]);
    return d.concat([ctx.h2h, ctx.winpct - ctx.opp_winpct, ctx.netszn - ctx.opp_netszn,
      ctx.net10 - ctx.opp_net10, ctx.rest - ctx.opp_rest, ctx.is_home]);
  }

  // scikit-learn trees compare float32(x) <= threshold
  function treeProb(t, x) {
    let i = 0;
    while (t.l[i] !== -1) {
      i = Math.fround(x[t.f[i]]) <= t.t[i] ? t.l[i] : t.r[i];
    }
    return t.p[i];
  }

  function forestProb(forest, x) {
    let s = 0;
    for (const t of forest.trees) s += treeProb(t, x);
    return s / forest.trees.length;
  }

  class LineupModel {
    constructor(forest, league, meta) {
      this.forest = forest;
      this.league = league;
      this.cfg = meta.config;
      this.matchups = {};
      this.byId = {};
      for (const [abbr, T] of Object.entries(league.teams)) {
        for (const p of T.players) this.byId[p.id] = Object.assign({ team: abbr }, p);
      }
    }
    addMatchups(obj) { Object.assign(this.matchups, obj); }
    mhist() {
      const m = this.matchups;
      return (o, d) => m[o + "-" + d] || [0, 0.0];
    }
    players(ids) { return ids.map((i) => this.byId[i]); }

    teamVector(team, opp, lineup, oppLineup, isHome, rest = 1, oppRest = 1) {
      const T = this.league.teams[team], O = this.league.teams[opp];
      const look = this.mhist();
      const side = lineupSideFeatures(lineup, oppLineup, T.core, look, this.cfg);
      const oside = lineupSideFeatures(oppLineup, lineup, O.core, look, this.cfg);
      const ctx = {
        h2h: this.league.h2h[team][opp].pct, winpct: T.winpct, opp_winpct: O.winpct,
        netszn: T.net_szn, opp_netszn: O.net_szn, net10: T.net_l10, opp_net10: O.net_l10,
        rest, opp_rest: oppRest, is_home: isHome ? 1 : 0,
      };
      return { x: gameFeatureVector(side, oside, ctx), side, oside };
    }

    /* home: true | false | null (neutral site) */
    winProbability(team, opp, ids, oppIds, home = true) {
      const me = this.players(ids), them = this.players(oppIds);
      const hMe = home === null ? 0 : (home ? 1 : 0);
      const hThem = home === null ? 0 : (home ? 0 : 1);
      const a = this.teamVector(team, opp, me, them, hMe);
      const b = this.teamVector(opp, team, them, me, hThem);
      const p = (forestProb(this.forest, a.x) + 1 - forestProb(this.forest, b.x)) / 2;
      return { p, x: a.x, side: a.side, oside: a.oside };
    }

    validLineup(ids) {
      const ps = this.players(ids);
      const g = ps.filter((p) => p.role === "G").length;
      const b = ps.filter((p) => p.role === "B").length;
      const usg = ps.reduce((s, p) => s + p.usg, 0);
      const problems = [];
      if (g < 1) problems.push("No primary ball-handler (guard)");
      if (b < 1) problems.push("No big man");
      if (usg > this.cfg.max_usg) problems.push(`Usage adds to ${usg.toFixed(0)}% (limit ${this.cfg.max_usg}%): too many players who need the ball`);
      return { ok: problems.length === 0, problems };
    }

    optimize(team, opp, oppIds, home = true, unavailable = [], poolSize = 11, topK = 5) {
      const T = this.league.teams[team];
      const pool = T.players.filter((p) => !unavailable.includes(p.id)).slice(0, poolSize);
      const poolIds = pool.map((p) => p.id);
      const core = T.core.filter((c) => poolIds.includes(c));
      const rest = poolIds.filter((i) => !core.includes(i));
      const need = 5 - core.length;
      const out = [];
      const combo = [];
      const rec = (start) => {
        if (combo.length === need) {
          const ids = core.concat(combo);
          if (this.validLineup(ids).ok) out.push({ p: this.winProbability(team, opp, ids, oppIds, home).p, ids });
          return;
        }
        for (let i = start; i < rest.length; i++) { combo.push(rest[i]); rec(i + 1); combo.pop(); }
      };
      rec(0);
      out.sort((a, b) => b.p - a.p);
      return { core, evaluated: out.length, best: out.slice(0, topK) };
    }
  }

  const api = { LineupModel, lineupSideFeatures, gameFeatureVector, forestProb, assignDefenders };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.LineupLab = api;
})(typeof window !== "undefined" ? window : globalThis);
