/* Lineup Lab - UI. Model math lives in model.js. */
(function () {
  "use strict";
  const $ = (id) => document.getElementById(id);
  const el = (tag, attrs = {}, ...kids) => {
    const n = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs)) {
      if (k === "class") n.className = v;
      else if (k === "text") n.textContent = v;
      else if (k.startsWith("on")) n.addEventListener(k.slice(2), v);
      else n.setAttribute(k, v);
    }
    for (const k of kids) if (k != null) n.append(k.nodeType ? k : document.createTextNode(k));
    return n;
  };
  const put = (node, ...kids) => node.replaceChildren(...kids.flat().filter((k) => k != null));
  const pct = (x, d = 1) => (100 * x).toFixed(d) + "%";
  const sgn = (x, d = 1) => (x > 0 ? "+" : x < 0 ? "−" : "±") + Math.abs(x).toFixed(d);

  const S = { meta: null, league: null, forest: null, games: null, model: null, mLoaded: new Set() };

  async function getJSON(path) {
    const r = await fetch(path);
    if (!r.ok) throw new Error(`Couldn't load ${path} (${r.status})`);
    return r.json();
  }
  async function ensureMatchups(team) {
    if (S.mLoaded.has(team)) return;
    S.model.addMatchups(await getJSON(`data/matchups/${team}.json`));
    S.mLoaded.add(team);
  }
  const teamName = (a) => S.league.teams[a].name;

  /* ---------------- tabs ---------------- */
  function showTab(name) {
    document.querySelectorAll(".tabs button").forEach((b) => b.setAttribute("aria-selected", String(b.dataset.tab === name)));
    document.querySelectorAll(".panel").forEach((p) => { p.hidden = p.dataset.panel !== name; });
    if (name === "lab") renderLab();
  }

  /* ================= PLAY ================= */
  const P = { set: [], i: 0, you: 0, model: 0, picked: null, filter: "ALL" };

  function newSet() {
    let pool = S.games;
    if (P.filter !== "ALL") pool = pool.filter((g) => g.home === P.filter || g.away === P.filter);
    const idx = pool.map((_, i) => i);
    for (let i = idx.length - 1; i > 0; i--) { const j = Math.floor(Math.random() * (i + 1)); [idx[i], idx[j]] = [idx[j], idx[i]]; }
    P.set = idx.slice(0, Math.min(10, idx.length)).map((i) => pool[i]);
    P.i = 0; P.you = 0; P.model = 0; P.picked = null;
    $("final").hidden = true; $("matchup").hidden = false;
    renderGame();
  }

  function playerList(five) {
    const ul = el("ul", { class: "players" },
      el("li", { class: "head" }, el("span"), el("span", { text: "Starter" }),
        el("span", { class: "n", text: "PER" }), el("span", { class: "n", text: "TS%" }), el("span", { class: "n", text: "USG" })));
    for (const p of five) {
      ul.append(el("li", {},
        el("span", { class: "role", title: roleName(p.role), text: p.role }),
        el("span", { class: "pname", text: p.name }),
        el("span", { class: "n", text: p.per.toFixed(1) }),
        el("span", { class: "n", text: (100 * p.ts).toFixed(1) }),
        el("span", { class: "n", text: p.usg.toFixed(1) })));
    }
    return ul;
  }
  const roleName = (r) => ({ G: "Guard / primary creator", W: "Wing", B: "Big" }[r] || r);

  function teamBlock(abbr, rec, five, coreOut) {
    return [
      el("h2", { text: abbr }),
      el("div", { class: "tname", text: teamName(abbr) }),
      el("div", { class: "rec", text: `${rec[0]}–${rec[1]} before tip-off` }),
      coreOut.length ? el("div", { class: "out", text: `Core not starting: ${coreOut.join(", ")}` }) : null,
      playerList(five),
    ];
  }

  function renderGame() {
    const g = P.set[P.i];
    $("score-you").textContent = P.you; $("score-model").textContent = P.model;
    $("round").replaceChildren(String(P.i + 1), el("small", { text: `/${P.set.length}` }));
    const d = new Date(g.date + "T12:00:00");
    put($("m-meta"),
      el("span", { text: d.toLocaleDateString(undefined, { weekday: "short", month: "short", day: "numeric", year: "numeric" }) }),
      g.type === "PO" ? el("span", { class: "badge po", text: "Postseason" }) : el("span", { class: "badge", text: "Regular season" }),
      g.neutral ? el("span", { class: "badge", text: "Neutral site" }) : null);
    put($("m-away"), teamBlock(g.away, g.away_rec, g.away5, g.away_core_out));
    put($("m-home"), teamBlock(g.home, g.home_rec, g.home5, g.home_core_out));
    document.querySelector(".m-at").textContent = g.neutral ? "vs" : "@";
    const n = S.meta.config;
    const same = S.league.teams[g.home].conf === S.league.teams[g.away].conf;
    $("m-context").replaceChildren(
      el("span", {}, el("b", { text: "Head-to-head: " }),
        g.h2h.n ? `${g.home} ${g.h2h.w}–${g.h2h.n - g.h2h.w} in the last ${g.h2h.n} meetings` : "no recent meetings",
        el("span", { class: "muted", text: ` (window: ${same ? n.h2h_same : n.h2h_cross}, ${same ? "same" : "cross"}-conference)` })));
    $("pick-row").replaceChildren(
      el("button", { id: "pick-away", onclick: () => pick(g.away) }, `${g.away} wins`),
      el("button", { id: "pick-home", onclick: () => pick(g.home) }, `${g.home} wins`));
    $("reveal").hidden = true;
  }

  function pick(team) {
    const g = P.set[P.i];
    document.querySelectorAll("#pick-row button").forEach((b) => { b.disabled = true; });
    $(team === g.home ? "pick-home" : "pick-away").classList.add("chosen");
    const winner = g.home_win ? g.home : g.away;
    const loser = g.home_win ? g.away : g.home;
    const modelPick = g.p_home >= 0.5 ? g.home : g.away;
    const youRight = team === winner, modelRight = modelPick === winner;
    P.you += youRight; P.model += modelRight;
    $("score-you").textContent = P.you; $("score-model").textContent = P.model;
    const score = g.home_pts != null
      ? ` ${Math.max(g.home_pts, g.away_pts)}–${Math.min(g.home_pts, g.away_pts)}` : "";
    const last = P.i === P.set.length - 1;
    const bar = el("div", { class: "pbar", role: "img", "aria-label": `Model: ${g.home} ${pct(g.p_home, 0)}, ${g.away} ${pct(1 - g.p_home, 0)}` },
      el("span", { class: "a", style: `width:${100 * (1 - g.p_home)}%`, text: `${g.away} ${pct(1 - g.p_home, 0)}` }),
      el("span", { class: "h", text: `${g.home} ${pct(g.p_home, 0)}` }));
    $("reveal").replaceChildren(
      el("div", { class: "result", text: `${winner} beat ${loser}${score}` }),
      el("div", {}, el("span", { class: "lbl", text: "Model's pre-game win probability" }), bar),
      el("div", { class: "verdicts" },
        el("span", { class: "v " + (youRight ? "ok" : "no"), text: youRight ? "You: correct" : "You: missed" }),
        el("span", { class: "v " + (modelRight ? "ok" : "no"), text: modelRight ? `Model: correct (${modelPick})` : `Model: missed (${modelPick})` })),
      el("div", {}, el("button", { class: "primary", id: "next-game", onclick: next }, last ? "See your result" : "Next game")));
    $("reveal").hidden = false;
    $("next-game").focus({ preventScroll: true });
  }

  function next() {
    if (P.i < P.set.length - 1) { P.i++; renderGame(); return; }
    $("matchup").hidden = true;
    const diff = P.you - P.model;
    const head = diff > 0 ? "You beat the model" : diff < 0 ? "The model wins this set" : "Dead heat";
    $("final").replaceChildren(
      el("h2", { text: head }),
      el("p", { text: `You went ${P.you} for ${P.set.length}. The model went ${P.model} for ${P.set.length}. Across the whole season it picks about ${pct(S.meta.backtest.expanding_10.pooled.random_forest.accuracy, 0)} of games correctly.` }),
      el("div", {}, el("button", { class: "primary", onclick: newSet }, "Play another set")));
    $("final").hidden = false;
  }

  function initPlay() {
    const sel = $("play-filter");
    sel.append(el("option", { value: "ALL", text: "All teams" }));
    for (const a of Object.keys(S.league.teams).sort((x, y) => teamName(x).localeCompare(teamName(y))))
      sel.append(el("option", { value: a, text: teamName(a) }));
    sel.value = "OKC"; P.filter = "OKC";
    sel.addEventListener("change", () => { P.filter = sel.value; newSet(); });
    $("play-restart").addEventListener("click", newSet);
    newSet();
  }

  /* ================= LAB ================= */
  const L = { team: "OKC", opp: "DEN", venue: "home", my: [], their: [], out: new Set(), best: [] };
  const venueArg = () => (L.venue === "neutral" ? null : L.venue === "home");

  function resetLineups() {
    L.out = new Set(); L.best = [];
    L.my = S.league.teams[L.team].default5.slice();
    L.their = S.league.teams[L.opp].default5.slice();
  }

  async function renderLab() {
    await Promise.all([ensureMatchups(L.team), ensureMatchups(L.opp)]);
    const T = S.league.teams[L.team], O = S.league.teams[L.opp];
    $("lab-you-h").textContent = `${L.team} starting five`;
    $("lab-opp-h").textContent = `${L.opp} starting five`;
    renderFive($("lab-five"), L.my, T, true);
    renderFive($("lab-opp-five"), L.their, O, false);
    renderRoster($("lab-roster"), T, L.my, true);
    renderRoster($("lab-opp-roster"), O, L.their, false);
    renderBest();

    const warn = $("lab-warn"); warn.replaceChildren();
    if (L.my.length !== 5 || L.their.length !== 5) {
      $("lab-prob").textContent = "–"; $("lab-delta").textContent = "Pick five starters for each side";
      $("lab-delta").className = "prob-delta"; $("lab-bar").style.width = "50%";
      $("lab-matchups").replaceChildren(); $("lab-features").replaceChildren();
      return;
    }
    const v = S.model.validLineup(L.my);
    for (const p of v.problems) warn.append(el("li", { text: p }));
    const res = S.model.winProbability(L.team, L.opp, L.my, L.their, venueArg());
    $("lab-prob").textContent = pct(res.p);
    $("lab-bar").style.width = (100 * res.p).toFixed(1) + "%";
    const base = T.default5.filter((i) => !L.out.has(i));
    const d = $("lab-delta");
    if (base.length === 5 && !sameSet(base, L.my)) {
      const bp = S.model.winProbability(L.team, L.opp, base, L.their, venueArg()).p;
      const dd = res.p - bp;
      d.textContent = `${sgn(100 * dd)} pts vs. ${L.team}'s most-used five (${pct(bp)})`;
      d.className = "prob-delta " + (dd > 0.0005 ? "up" : dd < -0.0005 ? "down" : "");
    } else {
      d.textContent = base.length === 5 ? `${L.team}'s most-used five` : "";
      d.className = "prob-delta";
    }
    renderMatchups(res);
    renderFeatures(res);
  }
  const sameSet = (a, b) => a.length === b.length && a.every((x) => b.includes(x));

  function statsText(p) { return `PER ${p.per.toFixed(1)} · TS ${(100 * p.ts).toFixed(1)} · USG ${p.usg.toFixed(1)}`; }

  function renderFive(ol, ids, T, mine) {
    ol.replaceChildren();
    const ps = S.model.players(ids).sort((a, b) => a.pos - b.pos);
    for (const p of ps) {
      const core = T.core.includes(p.id);
      ol.append(el("li", {},
        el("span", { class: "role", title: roleName(p.role), text: p.role }),
        el("span", {}, p.name, core ? el("span", { class: "star", title: "Core player", text: " ★" }) : null),
        el("span", { class: "stats", text: statsText(p) })));
    }
    for (let i = ps.length; i < 5; i++) ol.append(el("li", { class: "empty" }, el("span"), el("span", { text: mine ? "Open slot: pick from the roster" : "Open slot" }), el("span")));
  }

  function renderRoster(ul, T, ids, mine) {
    ul.replaceChildren();
    for (const p of T.players) {
      const on = ids.includes(p.id), out = mine && L.out.has(p.id), core = T.core.includes(p.id);
      const li = el("li", { class: (on ? "on " : "") + (out ? "isout" : "") },
        el("span", { class: "role", title: roleName(p.role), text: p.role }),
        el("button", { class: "pick pname", "aria-pressed": String(on), disabled: out ? "" : null,
          onclick: () => toggleStart(p.id, mine) },
          p.name, core ? el("span", { class: "star", text: " ★" }) : null),
        el("span", { class: "stats", text: `${p.mpg.toFixed(1)} mpg · PER ${p.per.toFixed(1)}` }),
        mine ? el("button", { class: "outbtn", "aria-pressed": String(out), title: "Unavailable tonight", onclick: () => toggleOut(p.id) }, "Out") : el("span"));
      if (!out) li.querySelector(".pick").removeAttribute("disabled");
      ul.append(li);
    }
  }

  function toggleStart(id, mine) {
    const arr = mine ? L.my : L.their;
    const i = arr.indexOf(id);
    if (i >= 0) arr.splice(i, 1);
    else if (arr.length < 5) arr.push(id);
    else { arr.pop(); arr.push(id); }
    L.best = [];
    renderLab();
  }
  function toggleOut(id) {
    if (L.out.has(id)) L.out.delete(id);
    else { L.out.add(id); L.my = L.my.filter((x) => x !== id); }
    L.best = [];
    renderLab();
  }

  function optimize() {
    const r = S.model.optimize(L.team, L.opp, L.their, venueArg(), [...L.out]);
    L.best = r.best;
    if (r.best.length) L.my = r.best[0].ids.slice();
    renderLab();
  }

  function renderBest() {
    const ol = $("lab-best"); ol.replaceChildren();
    if (!L.best.length) { ol.append(el("li", { class: "muted", text: "Press “Find the best five”. Core players (★) always start when available." })); return; }
    for (const b of L.best) {
      const names = S.model.players(b.ids).sort((x, y) => x.pos - y.pos).map((p) => p.name).join(", ");
      ol.append(el("li", {}, el("button", { onclick: () => { L.my = b.ids.slice(); renderLab(); } },
        el("span", { class: "p", text: pct(b.p) }), el("span", { class: "names", text: names }))));
    }
  }

  function renderMatchups(res) {
    const me = S.model.players(L.my), them = S.model.players(L.their);
    const look = S.model.mhist();
    const prior = S.meta.config.matchup_prior;
    const t = $("lab-matchups");
    t.replaceChildren(el("tr", {}, el("th", { text: L.team }), el("th", { text: "Likely defender" }),
      el("th", { class: "n", text: "Games" }), el("th", { class: "n", text: "TS edge" })));
    for (const pr of LineupLab.assignDefenders(me, them)) {
      const [n, s] = look(pr.off, pr.def);
      const e = 100 * s / (n + prior);
      t.append(el("tr", {}, el("td", { text: S.model.byId[pr.off].name }), el("td", { text: S.model.byId[pr.def].name }),
        el("td", { class: "n", text: String(n) }),
        el("td", { class: "n " + (e > 0.05 ? "pos" : e < -0.05 ? "neg" : ""), text: n ? sgn(e) : "–" })));
    }
  }

  function renderFeatures(res) {
    const a = res.side, b = res.oside, T = S.league.teams[L.team], O = S.league.teams[L.opp];
    const h = S.league.h2h[L.team][L.opp];
    const rows = [
      ["Average PER of the five", a.LU_PER, b.LU_PER, 1],
      ["True shooting %, usage-weighted", a.LU_TS, b.LU_TS, 1],
      ["Total usage rate", a.LU_USG, b.LU_USG, 1],
      ["Core players starting", a.CORE_N, b.CORE_N, 0],
      ["Core PER (per core slot)", a.CORE_PER, b.CORE_PER, 1],
      ["Best player's PER", a.STAR_PER, b.STAR_PER, 1],
      ["Matchup TS edge", a.MATCHUP, b.MATCHUP, 2],
      ["Net rating, season", T.net_szn, O.net_szn, 1],
      ["Net rating, last 10", T.net_l10, O.net_l10, 1],
    ];
    const t = $("lab-features");
    t.replaceChildren(el("tr", {}, el("th", { text: "Feature" }), el("th", { class: "n", text: L.team }),
      el("th", { class: "n", text: L.opp }), el("th", { class: "n", text: "Edge" })));
    for (const [label, x, y, d] of rows) {
      const diff = x - y;
      t.append(el("tr", {}, el("td", { text: label }), el("td", { class: "n", text: x.toFixed(d) }),
        el("td", { class: "n", text: y.toFixed(d) }),
        el("td", { class: "n " + (diff > 0 ? "pos" : diff < 0 ? "neg" : ""), text: sgn(diff, d) })));
    }
    t.append(el("tr", {}, el("td", { text: `Head-to-head, last ${h.n} meetings` }),
      el("td", { class: "n", text: `${h.w}–${h.n - h.w}` }), el("td", { class: "n", text: `${h.n - h.w}–${h.w}` }), el("td", { class: "n", text: pct(h.pct, 0) })));
    t.append(el("tr", {}, el("td", { text: "Regular-season record" }),
      el("td", { class: "n", text: `${T.w}–${T.l}` }), el("td", { class: "n", text: `${O.w}–${O.l}` }), el("td")));
  }

  function initLab() {
    const teams = Object.keys(S.league.teams).sort((x, y) => teamName(x).localeCompare(teamName(y)));
    for (const id of ["lab-team", "lab-opp"]) for (const a of teams) $(id).append(el("option", { value: a, text: teamName(a) }));
    $("lab-team").value = L.team; $("lab-opp").value = L.opp;
    $("lab-team").addEventListener("change", (e) => {
      L.team = e.target.value;
      if (L.team === L.opp) { L.opp = teams.find((t) => t !== L.team); $("lab-opp").value = L.opp; }
      resetLineups(); renderLab();
    });
    $("lab-opp").addEventListener("change", (e) => {
      L.opp = e.target.value;
      if (L.team === L.opp) { L.team = teams.find((t) => t !== L.opp); $("lab-team").value = L.team; }
      L.their = S.league.teams[L.opp].default5.slice(); L.best = []; renderLab();
    });
    document.querySelectorAll(".seg button").forEach((b) => b.addEventListener("click", () => {
      L.venue = b.dataset.venue;
      document.querySelectorAll(".seg button").forEach((x) => x.setAttribute("aria-checked", String(x === b)));
      L.best = []; renderLab();
    }));
    $("lab-optimize").addEventListener("click", optimize);
    $("lab-reset").addEventListener("click", () => { L.my = S.league.teams[L.team].default5.filter((i) => !L.out.has(i)); L.best = []; renderLab(); });
    resetLineups();
  }

  /* ================= MODEL ================= */
  const FEAT_SHORT = {
    LU_PER_DIFF: "Lineup PER", LU_TS_DIFF: "Lineup TS%", LU_USG_DIFF: "Lineup usage", CORE_N_DIFF: "Core starting",
    CORE_PER_DIFF: "Core PER", STAR_PER_DIFF: "Best player PER", MATCHUP_DIFF: "Matchup edge", H2H_WINPCT: "Head-to-head",
    WINPCT_DIFF: "Win %", NET_SZN_DIFF: "Net rtg (season)", NET_L10_DIFF: "Net rtg (last 10)", REST_DIFF: "Rest days", IS_HOME: "Home court",
  };

  function initModel() {
    const bt = S.meta.backtest;
    const R = bt.rolling_5_10.pooled, E = bt.expanding_10.pooled, PO = bt.playoffs;
    $("model-summary").replaceChildren(
      stat(pct(R.random_forest.accuracy), "Train 5 / test 10", `${R.random_forest.games} games predicted. Better-record rule: ${pct(R.baselines.better_record)}. Single tree: ${pct(R.decision_tree.accuracy)}.`),
      stat(pct(E.random_forest.accuracy), "Expanding window", `Same test blocks, trained on every earlier game. Log loss ${E.random_forest.log_loss.toFixed(3)} vs. ${R.random_forest.log_loss.toFixed(3)}.`),
      stat(pct(PO.random_forest.accuracy), "Postseason holdout", `${PO.random_forest.games} play-in and playoff games, model trained on the regular season only. Better-record rule: ${pct(PO.baselines.better_record)}.`));
    foldChart(bt.rolling_5_10.folds);
    impChart(S.meta.importance_forest);
    $("tree").replaceChildren(treeNode(S.meta.tree));
    notes();
  }
  function stat(big, label, sub) {
    return el("div", { class: "stat" }, el("span", { class: "lbl", text: label }), el("span", { class: "big", text: big }), el("span", { class: "sub", text: sub }));
  }

  function foldChart(folds) {
    const W = 640, H = 240, m = { l: 40, r: 10, t: 14, b: 34 };
    const ymin = 0.4, ymax = 0.9;
    const x0 = m.l, bw = (W - m.l - m.r) / folds.length;
    const y = (v) => m.t + (H - m.t - m.b) * (1 - (v - ymin) / (ymax - ymin));
    const ns = "http://www.w3.org/2000/svg";
    const svg = document.createElementNS(ns, "svg");
    svg.setAttribute("viewBox", `0 0 ${W} ${H}`); svg.setAttribute("width", "100%");
    svg.setAttribute("role", "img"); svg.setAttribute("aria-label", "Accuracy of the random forest in each 10-game test block");
    svg.style.minWidth = "520px";
    const add = (tag, a, text) => { const n = document.createElementNS(ns, tag); for (const k in a) n.setAttribute(k, a[k]); if (text) n.textContent = text; svg.append(n); return n; };
    for (const v of [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]) {
      add("line", { x1: m.l, x2: W - m.r, y1: y(v), y2: y(v), stroke: "var(--line)", "stroke-width": 1 });
      add("text", { x: m.l - 6, y: y(v) + 4, "text-anchor": "end" }, (100 * v).toFixed(0) + "%");
    }
    const pts = [];
    folds.forEach((f, i) => {
      const cx = x0 + bw * i + bw / 2, a = f.random_forest.accuracy;
      add("rect", { x: cx - bw * 0.3, y: y(a), width: bw * 0.6, height: y(ymin) - y(a), fill: "var(--accent)", rx: 2 });
      add("text", { x: cx, y: y(a) + 16, "text-anchor": "middle", style: "fill: var(--accent-ink); font-weight: 500" }, (100 * a).toFixed(0));
      add("text", { x: cx, y: H - m.b + 16, "text-anchor": "middle" }, `${f.test_games[0] + 1}–${f.test_games[1]}`);
      pts.push([cx, y(f.baselines.better_record)]);
    });
    add("polyline", { points: pts.map((p) => p.join(",")).join(" "), fill: "none", stroke: "var(--ball)", "stroke-width": 2 });
    pts.forEach(([cx, cy]) => add("circle", { cx, cy, r: 3.5, fill: "var(--ball)" }));
    add("text", { x: W / 2, y: H - 2, "text-anchor": "middle" }, "Team games in test block");
    $("fold-chart").replaceChildren(svg);
  }

  function impChart(imp) {
    const max = Math.max(...Object.values(imp));
    const box = el("div", { class: "imp" });
    for (const [k, v] of Object.entries(imp)) {
      box.append(el("div", { class: "imp-row" }, el("span", { text: FEAT_SHORT[k] || k }),
        el("div", {}, el("div", { class: "bar", style: `width:${(100 * v / max).toFixed(1)}%` })),
        el("span", { class: "v", text: v.toFixed(3) })));
    }
    $("imp-chart").replaceChildren(box);
  }

  function treeNode(n) {
    if (!n.feature) {
      const c = n.p >= 0.5 ? "var(--win)" : "var(--loss)";
      return el("div", { class: "tleaf" }, el("span", { class: "dot", style: `background:${c}` }), `wins ${pct(n.p, 0)} · n=${n.n}`);
    }
    const lab = FEAT_SHORT[n.feature] || n.feature;
    return el("div", { class: "tnode" },
      el("div", { class: "tsplit", text: `${lab} ≤ ${n.threshold}` }), treeNode(n.left),
      el("div", { class: "tsplit", text: `${lab} > ${n.threshold}` }), treeNode(n.right));
  }

  function notes() {
    const m = S.meta, c = m.config;
    const box = $("model-notes");
    const add = (h, ...ps) => { box.append(el("h3", { text: h })); for (const p of ps) box.append(el("p", { text: p })); };
    add("Player metrics",
      "PER (Hollinger, pace-adjusted, league average 15), true shooting % and usage rate are computed from raw box scores, season-to-date as of each game. They match Basketball-Reference's published 2025-26 values (Shai Gilgeous-Alexander: PER 31.0 vs. 30.8, TS% .665 vs. .665). Early in the season each player is blended with league average as if he had 150 minutes of average play.");
    add("Core players",
      `Each team's backbone is its top ${c.core_size} players by PER × minutes, with front-office overrides (OKC: Shai Gilgeous-Alexander, Jalen Williams, Chet Holmgren). The lineup optimizer always starts available core players and fills the other spots around them, requiring at least one guard and one big and capping combined usage at ${c.max_usg}%.`);
    add("Matchups and head-to-head",
      `Each player is paired with his likely primary defender by positional slot. His edge is how his true shooting in past meetings with that defender compares to his own average, shrunk toward zero (${c.matchup_prior} phantom average games). ${m.matchup_source === "tracking" ? "Pairings come from NBA tracking data (who guarded whom)." : "With the tracking-data fetch (see README) the pairing uses real partial possessions instead."} Head-to-head uses the last ${c.h2h_same} meetings for same-conference opponents and ${c.h2h_cross} for cross-conference, which covers about three seasons either way.`);
    add("Evaluation",
      `The walk-forward test trains on a block of ${c.train_n} games per team and predicts the next ${c.test_n}, then slides forward. A 5-game block is only about 150 team-game rows, which is thin for a tree; the expanding-window run keeps the same test blocks but trains on everything before them, and it is the version used for the game's picks. Picking the team with the better record is a strong baseline in the NBA, so the comparison is shown next to every number.`);
    add("Limits",
      `Box-score metrics undervalue defensive specialists (the optimizer tends to prefer shooters over stoppers). Starters are ${m.starter_source === "boxscore" ? "taken from official box scores" : "approximated as the five players with the most minutes"}. The data runs through ${m.data_through}.`);
  }

  /* ================= boot ================= */
  async function boot() {
    try {
      const [meta, league, forest, games] = await Promise.all(
        ["meta", "league", "model", "games"].map((f) => getJSON(`data/${f}.json`)));
      Object.assign(S, { meta, league, forest, games });
      S.model = new LineupLab.LineupModel(forest, league, meta);
      document.querySelectorAll(".tabs button").forEach((b) => b.addEventListener("click", () => showTab(b.dataset.tab)));
      initPlay(); initLab(); initModel();
      $("loading").hidden = true;
      const h = (location.hash || "").replace("#", "");
      showTab(["play", "lab", "model"].includes(h) ? h : "play");
    } catch (e) {
      $("loading").textContent = `The season data didn't load: ${e.message}. If you opened index.html from disk, serve the folder instead (python -m http.server).`;
    }
  }
  boot();
})();
