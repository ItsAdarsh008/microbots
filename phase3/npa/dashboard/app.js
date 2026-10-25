/* NPA lab dashboard: experiment tree, repair clips and loss curves. Data comes from data.js (made by lab/dashboard.py). */
(function () {
  "use strict";
  const $ = (id) => document.getElementById(id);
  const out = $("out"), gif = $("gif");
  let D = window.NPA_DASH, layout = "tree", view = "clip", current = null, filter = "", tab = "lizard";
  const TABS = {
    lizard: { note: "the lizard, ideal dots (the paper's setting)", pick: "pa8_scales2_all_damage", empty: "pick a lizard run" },
    shapes: { note: "unseen shapes, no compass, ideal dots", pick: null, empty: "pick a shape run (no clips yet: only plain networks have one)" },
    robot: { note: "bot body: MuJoCo on the CAD, and runs through the body model (assumed)", pick: "roll + quarter-turn gait", empty: "pick the MuJoCo clip, or a body-model run" },
  };
  const inTab = (name) => (D.runs[name].tab || "lizard") === tab;
  const open = new Set();   // collapsed groups are the default; ids here are expanded

  const el = (tag, cls, text) => { const n = document.createElement(tag); if (cls) n.className = cls; if (text !== undefined) n.textContent = text; return n; };
  const f2 = (v) => (v === null || v === undefined ? "–" : v >= 100 ? v.toFixed(0) : v.toFixed(3));

  // dot kind: good / finished / dropped / running / queued
  function kind(r) {
    if (r.state === "demo") return "main";
    if (r.state === "queued") return "q";
    if (r.state === "running") return "run";
    if (r.state === "failed" || r.broken || r.diverged) return "dead";
    const comparable = r.shape === "lizard" && !r.agnostic;
    return comparable && (r.score < D.good || (r.rhard !== null && r.rhard >= D.good_repair)) ? "main" : "fin";
  }

  function runNode(name, kids, depth) {
    const r = D.runs[name], k = kind(r);
    const li = el("li", "node"); li.dataset.kind = k; li.dataset.run = name;
    if (name === current) li.setAttribute("aria-current", "true");
    const b = el("button", "nhead"); b.type = "button";
    b.append(el("span", "name", name));
    const sc = el("span", "sc");
    if (r.gif) sc.append(el("span", "play", "▶")); else sc.append(el("span", "play", ""));
    if (r.state === "done") {
      if (r.rhard !== null) sc.append(el("span", "", "R " + r.rhard.toFixed(2)));
      const bar = el("span", "bar"), i = el("i"); i.style.width = Math.max(2, Math.min(100, (1 - Math.min(r.score, 1)) * 100)) + "%"; bar.append(i); sc.append(bar);
      sc.append(el("b", "", f2(r.score)));
    } else sc.append(el("b", "", r.state));
    b.append(sc);
    b.onclick = () => select(name);
    li.append(b);
    if (name === current) li.append(detail(name));
    if (kids && kids.length) { const ul = el("ul"); kids.forEach((c) => ul.append(runNode(c.run, c.kids, depth + 1))); li.append(ul); }
    return li;
  }

  function detail(name) {
    const r = D.runs[name], d = el("div", "detail");
    if (r.state === "done") {
      const t = el("table");
      [["score", f2(r.score)], ["grown lizard (ctrl)", f2(r.ctrl)], ["drift", r.drift === null ? "–" : (r.drift > 0 ? "+" : "") + r.drift.toFixed(3)],
       ["hard-damage repair", r.rhard === null ? "–" : r.rhard.toFixed(2)], ["iterations", (r.iters || 0) + " / " + (r.want || "?") + (r.broken ? " (stopped early)" : "")],
       ["params · minutes", (r.params || 0).toLocaleString() + " · " + r.minutes]].forEach(([a, c]) => { const tr = el("tr"); tr.append(el("td", "", a), el("td", "", c)); t.append(tr); });
      d.append(t);
    }
    if (r.cfg) d.append(el("p", "cfg", r.cfg));
    return d;
  }

  function matches(name) { return inTab(name) && (!filter || name.toLowerCase().includes(filter) || (D.runs[name].cfg || "").toLowerCase().includes(filter)); }
  function anyMatch(n) { return matches(n.run) || (n.kids || []).some(anyMatch); }
  function prune(n) { return { run: n.run, kids: (n.kids || []).filter(anyMatch).map(prune) }; }

  function renderTree() {
    out.textContent = "";
    D.tree.forEach((g) => {
      const kids = g.kids.filter(anyMatch).map(prune);
      if (!kids.length) return;
      const count = (ns) => ns.reduce((a, n) => a + 1 + count(n.kids || []), 0);
      const sec = el("section", "group");
      const h = el("button", "ghead"); h.type = "button";
      h.append(el("b", "", g.id), el("span", "", g.text), el("i", "", count(kids)));
      const isOpen = filter || tab === "robot" || open.has(g.id) || (current && JSON.stringify(kids).includes('"' + current + '"'));
      h.onclick = () => { open.has(g.id) ? open.delete(g.id) : open.add(g.id); render(); };
      sec.append(h);
      if (isOpen) { const ol = el("ol", "kids"); kids.forEach((k) => ol.append(runNode(k.run, k.kids, 0))); sec.append(ol); }
      out.append(sec);
    });
  }

  function renderList() {
    out.textContent = "";
    const order = { run: 0, main: 1, fin: 2, q: 3, dead: 4 };
    const names = Object.keys(D.runs).filter(matches).sort((a, b) => {
      const ka = kind(D.runs[a]), kb = kind(D.runs[b]);
      return order[ka] - order[kb] || (D.runs[a].score ?? 9e9) - (D.runs[b].score ?? 9e9);
    });
    const ol = el("ol", "kids"); names.forEach((n) => ol.append(runNode(n, [], 0))); out.append(ol);
  }

  function render() { layout === "tree" ? renderTree() : renderList(); header(); }

  function header() {
    const mine = Object.keys(D.runs).filter(inTab).map((n) => D.runs[n].state);
    const n = (s) => mine.filter((x) => x === s).length;
    $("meta").textContent = `${n("running")} running · ${n("queued")} queued · ${n("done")} done`;
    document.querySelectorAll(".tabs button").forEach((b) => {
      const t = b.dataset.tab, cnt = Object.keys(D.runs).filter((k) => (D.runs[k].tab || "lizard") === t).length;
      b.textContent = b.textContent.replace(/ \d+$/, "") + " " + cnt;
      b.setAttribute("aria-pressed", String(t === tab));
    });
    $("tabnote").textContent = TABS[tab].note;
    $("foot").textContent = `updated ${D.updated} · ideal-dot results on a 1,024-particle lizard unless a run says otherwise · base score ${D.base ?? "–"}`;
  }

  function select(name) {
    current = current === name ? null : name;
    render(); stage();
  }

  function stage() {
    const r = current && D.runs[current];
    $("card").hidden = !r;
    if (r) {
      $("cl").textContent = r.state === "done" ? "finished" : r.state;
      $("ct").textContent = current;
      $("co").textContent = r.state === "done" ? `score ${f2(r.score)} · repair ${r.rhard === null ? "–" : r.rhard.toFixed(2)}` : "";
    }
    const showLoss = view === "loss";
    $("lossv").hidden = !showLoss;
    const hasGif = !showLoss && r && r.gif;
    gif.hidden = !hasGif;
    if (hasGif && gif.getAttribute("src") !== r.gif) gif.src = r.gif;
    if (!hasGif) gif.removeAttribute("src");
    $("empty").hidden = showLoss || hasGif;
    if (!showLoss && !hasGif) {
      $("etitle").textContent = r ? current : TABS[tab].empty;
      $("enote").textContent = r ? "no clip for this one (only the plain 66-128-18 network has one)" : (tab === "robot" ? "the body-model runs are numbers only; the MuJoCo clip shows the real CAD" : "runs with ▶ have a repair clip");
    }
    if (showLoss) drawLoss();
  }

  function drawLoss() {
    const cv = $("cv"), box = cv.parentElement, dpr = window.devicePixelRatio || 1;
    cv.width = box.clientWidth * dpr; cv.height = box.clientHeight * dpr;
    const g = cv.getContext("2d"); g.scale(dpr, dpr);
    const W = box.clientWidth, H = box.clientHeight, m = { l: 56, r: 24, t: 70, b: 44 };
    g.clearRect(0, 0, W, H);
    const sel = current && D.runs[current], ref = D.runs["base_10k"];
    const series = [];
    if (ref && ref.curve.length) series.push({ pts: ref.curve, col: "#c4c4bd", w: 1.5, label: "base_10k" });
    if (sel && sel.curve.length) series.push({ pts: sel.curve, col: "#1f3a8a", w: 2, label: current });
    if (!series.length) { g.fillStyle = "#8b8b84"; g.font = "14px Arial"; g.fillText("no loss curve yet", m.l, m.t); return; }
    const xs = series.flatMap((s) => s.pts.map((p) => p[0])), ys = series.flatMap((s) => s.pts.map((p) => Math.max(p[1], 1e-3)));
    const xmax = Math.max(...xs, 1), ymin = Math.min(...ys), ymax = Math.max(...ys);
    const ly = (v) => Math.log10(Math.max(v, 1e-3));
    const X = (v) => m.l + (v / xmax) * (W - m.l - m.r);
    const Y = (v) => m.t + (1 - (ly(v) - ly(ymin)) / Math.max(ly(ymax) - ly(ymin), 0.3)) * (H - m.t - m.b);
    g.strokeStyle = "#e6e5e0"; g.fillStyle = "#8b8b84"; g.font = "11px Arial"; g.lineWidth = 1;
    for (let e = Math.floor(ly(ymin)); e <= Math.ceil(ly(ymax)); e++) {
      const v = Math.pow(10, e), y = Y(v); if (y < m.t - 1 || y > H - m.b + 1) continue;
      g.beginPath(); g.moveTo(m.l, y); g.lineTo(W - m.r, y); g.stroke(); g.fillText(v >= 1 ? v.toFixed(0) : String(v), 12, y + 4);
    }
    g.fillText("iteration", W - m.r - 46, H - 14); g.fillText("loss (log)", 12, m.t - 18);
    series.forEach((s) => {
      g.strokeStyle = s.col; g.lineWidth = s.w; g.beginPath();
      s.pts.forEach((p, i) => { const x = X(p[0]), y = Y(Math.max(p[1], 1e-3)); i ? g.lineTo(x, y) : g.moveTo(x, y); });
      g.stroke();
    });
    g.font = "12px Arial"; let lx = m.l; series.slice().reverse().forEach((s) => { g.fillStyle = s.col; g.fillRect(lx, H - 24, 14, 3); g.fillStyle = "#5a5a5a"; g.fillText(s.label, lx + 20, H - 19); lx += 40 + s.label.length * 7; });
  }

  function setTab(t) {
    tab = t; filter = ""; $("q").value = "";
    const want = TABS[t].pick && D.runs[TABS[t].pick] ? TABS[t].pick : null;
    current = want;
    render(); stage();
  }
  document.querySelectorAll(".tabs button").forEach((b) => { b.onclick = () => setTab(b.dataset.tab); });

  function setPressed(ids, on) { ids.forEach((id) => $(id).setAttribute("aria-pressed", String(id === on))); }
  $("l-tree").onclick = () => { layout = "tree"; setPressed(["l-tree", "l-list"], "l-tree"); render(); };
  $("l-list").onclick = () => { layout = "list"; setPressed(["l-tree", "l-list"], "l-list"); render(); };
  $("v-clip").onclick = () => { view = "clip"; setPressed(["v-clip", "v-loss"], "v-clip"); stage(); };
  $("v-loss").onclick = () => { view = "loss"; setPressed(["v-clip", "v-loss"], "v-loss"); stage(); };
  $("q").oninput = (e) => { filter = e.target.value.trim().toLowerCase(); render(); };
  window.addEventListener("resize", () => view === "loss" && drawLoss());

  // follow training: reload data.js every minute without a page refresh
  function reload() {
    const s = document.createElement("script"); s.src = "data.js?t=" + Date.now();
    s.onload = () => { D = window.NPA_DASH; render(); stage(); s.remove(); }; s.onerror = () => s.remove();
    document.head.append(s);
  }
  setInterval(reload, 60000);
  const hashTab = location.hash.slice(1);
  if (TABS[hashTab]) setTab(hashTab); else setTab("lizard");
})();
