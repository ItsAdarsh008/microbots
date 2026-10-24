"""Build the monitoring dashboard: dashboard/data.js (runs, loss curves, the experiment tree) and repair GIFs.

    python lab/dashboard.py [--gifs 3] [--deploy]

* data.js: every run (finished, running, queued) with its scores, loss curve (parsed from train.log), and a parent, so the
  runs form a spanning tree under the hypotheses of lab/hypotheses.md.
* gifs: for finished runs that use the plain 66-128-18 network (the only one the JavaScript engine can run), a three-panel
  repair clip (wipe the head, cut the tail, rip a foot off). --gifs N makes at most N new ones per call, best score first.
* --deploy copies dashboard/ to the demo server (http://129.80.219.24/lab/).
"""

import argparse
import glob
import json
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
NPA = HERE.parent
sys.path.insert(0, str(NPA))
sys.path.insert(0, str(HERE))
DASH = NPA / "dashboard"
RUNS = NPA / "runs"
GOOD_SCORE = 0.30          # lizard runs below this count as "good"; (a number picked to separate the clear leaders)
GOOD_REPAIR = 0.85
LOSS_RE = re.compile(r"^it\s+(\d+)\s.*?loss\s+([\d.]+)\s+density\s+([\d.]+)")


def read_curve(name, limit=120):
    log = RUNS / name / "train.log"
    if not log.exists():
        return []
    pts = [[int(m.group(1)), float(m.group(2)), float(m.group(3))] for line in log.read_text(errors="ignore").splitlines()
           if (m := LOSS_RE.match(line))]
    if len(pts) > limit:
        pts = pts[:: -(-len(pts) // limit)]
    return pts


def repair_hard(d):
    ev = d.get("eval", {})
    if not (ev.get("damage") and ev.get("growth")):
        return None
    ctrl = np.array(ev["growth"]["loss_control"])
    out = []
    for kind in ("cut", "drag", "clump"):
        kd = ev["damage"].get(kind)
        if not kd:
            continue
        hurt = np.array(kd["loss_after_damage"]) - ctrl[None, :]
        left = np.array(kd["loss_final"]) - ctrl[None, :]
        out.append(float(np.clip(1.0 - left.mean() / max(hurt.mean(), 0.02), -1, 1)))
    return float(np.mean(out)) if out else None


def short_cfg(cfg):
    c, t = cfg.get("cell", {}), cfg.get("train", {})
    bits = [cfg.get("shape", "lizard") if "shapes" not in cfg else "multi-shape"]
    if c:
        bits.append(", ".join(f"{k} {v}" for k, v in c.items() if k != "embody") or "")
        if "embody" in c:
            bits.append("body model (assumed, not simulated)")
    mode = t.get("mode", "bptt")
    if mode != "bptt":
        bits.append(mode)
    dm = t.get("damage")
    if dm:
        bits.append("damage: " + "+".join(sorted({x["kind"] for x in dm})))
    if t.get("pool_max_age"):
        bits.append(f"pool age {t['pool_max_age']}")
    if cfg.get("eps"):
        bits.append(f"eps {cfg['eps']}")
    return " · ".join(b for b in bits if b)


def load_hypotheses():
    rows = []
    for line in (HERE / "hypotheses.md").read_text().splitlines():
        if not line.startswith("| H"):
            continue
        cols = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cols) < 6:
            continue
        rows.append({"id": cols[0], "text": re.sub(r"`", "", cols[2])[:200], "names": re.findall(r"`([^`]+)`", cols[3]), "status": cols[5]})
    return rows


def parent_of(name, known):
    if "__ft_" in name:
        base = name.split("__ft_")[0]
        return base if base in known else None
    for pat in (r"_r\d$", r"_s\d+$"):
        m = re.search(pat, name)
        if m and name[: m.start()] in known:
            return name[: m.start()]
    m = re.match(r"(.+)_on_[a-z]+$", name)
    if m and m.group(1) in known:
        return m.group(1)
    return None


def collect():
    runs = {}
    now = time.time()
    for p in sorted(glob.glob(str(RUNS / "*/result.json"))):
        try:
            d = json.loads(Path(p).read_text())
        except ValueError:
            continue
        if not d.get("summary"):
            continue
        cfg, sm, name = d.get("config", {}), d["summary"], d["name"]
        want, got = cfg.get("train", {}).get("iters"), d.get("iters_done")
        broken = bool(want and got is not None and want > 0 and got < 0.5 * want)
        diverged = (not sm.get("finite")) or sm["score"] > 100
        runs[name] = {"state": "done", "score": round(sm["score"], 4), "ctrl": round(sm["control_loss"], 4), "drift": round(sm["drift"], 4),
                      "rhard": repair_hard(d), "params": d.get("params"), "iters": got, "want": want,
                      "minutes": round(d.get("train_minutes") or 0, 1), "broken": broken, "diverged": diverged,
                      "shape": "multi" if "shapes" in cfg else cfg.get("shape", "lizard"),
                      "agnostic": cfg.get("train", {}).get("mode") == "agnostic", "cfg": short_cfg(cfg)}
    for d in RUNS.glob("*/train.log"):
        name = d.parent.name
        if name in runs:
            continue
        runs[name] = {"state": "running" if now - d.stat().st_mtime < 900 else "failed", "cfg": ""}   # silent for 15 min and no result: dead
        for cand in (NPA / "lab/queue").rglob(f"*_{name}.json"):
            try:
                runs[name]["cfg"] = short_cfg(json.loads(cand.read_text()))
            except ValueError:
                pass
            break
    for sub in ("master", "dispatched", "pending"):
        for cand in (NPA / "lab/queue" / sub).rglob("*.json"):
            m = re.match(r"\d+_(.+)\.json$", cand.name)
            if not m or m.group(1) in runs:
                continue
            try:
                runs[m.group(1)] = {"state": "queued", "cfg": short_cfg(json.loads(cand.read_text()))}
            except ValueError:
                pass
    for name, r in runs.items():
        r["curve"] = read_curve(name)
        r["gif"] = f"gifs/{name}.gif" if (DASH / "gifs" / f"{name}.gif").exists() else None
    return runs


def build_tree(runs, hyps):
    assigned, nodes = set(), []
    def match(h):
        out = []
        for tok in h["names"]:
            if "*" in tok:
                rx = re.compile("^" + re.escape(tok).replace(r"\*", ".*") + "$")
                out += [n for n in runs if rx.match(n)]
            else:
                out += [n for n in runs if n == tok or n.startswith(tok + "_") or n.startswith(tok + "__")]
        return list(dict.fromkeys(out))
    def node(name, group):
        kids = [n for n in group if parent_of(n, set(runs)) == name]
        return {"run": name, "kids": [node(k, group) for k in sorted(kids)]}
    for h in hyps:
        group = [n for n in match(h) if n not in assigned]
        if not group:
            continue
        assigned |= set(group)
        tops = [n for n in group if parent_of(n, set(group)) is None]
        nodes.append({"id": h["id"], "text": h["text"], "status": h["status"], "kids": [node(n, group) for n in sorted(tops)]})
    rest = [n for n in runs if n not in assigned]
    tops = [n for n in rest if parent_of(n, set(rest)) is None]
    if tops:
        nodes.append({"id": "·", "text": "other runs", "status": "", "kids": [node(n, rest) for n in sorted(tops)]})
    return nodes


def make_gifs(runs, limit):
    import weights
    import repair
    todo = [(r["score"], n) for n, r in runs.items() if r["state"] == "done" and not r["broken"] and not r["diverged"]
            and not r["gif"] and not r["agnostic"] and r["shape"] == "lizard"]
    made = 0
    for _, name in sorted(todo):
        cfgp = RUNS / name / "result.json"
        cfg = json.loads(cfgp.read_text())["config"]
        cell = cfg.get("cell") or {}
        if set(cell) - {"scales"} or cfg.get("train", {}).get("mode", "bptt") != "bptt" or not (RUNS / name / "params.npz").exists():
            continue          # the browser engine runs the plain network, optionally sensing at several radii
        params, eps, eps0 = weights.load_npz(RUNS / name / "params.npz")
        scales = cell.get("scales")
        with tempfile.TemporaryDirectory() as tmp:
            js = Path(tmp) / "w.js"
            weights.to_js(js, name, params, eps, eps0, particles=cfg["particles"], scales=scales)
            repair.SNAP_EVERY = 8
            res = repair.run_scenarios(str(js), ["erase_head", "cut_tail", "drag_limb"], cfg["particles"], 0)
            repair.make_gif(DASH / "gifs" / f"{name}.gif", ["erase_head", "cut_tail", "drag_limb"], res, size=200)
        made += 1
        if made >= limit:
            break
    return made


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gifs", type=int, default=0)
    ap.add_argument("--deploy", action="store_true")
    a = ap.parse_args()
    DASH.mkdir(exist_ok=True)
    (DASH / "gifs").mkdir(exist_ok=True)
    runs = collect()
    if a.gifs:
        print("new gifs:", make_gifs(runs, a.gifs))
        for n, r in runs.items():
            r["gif"] = f"gifs/{n}.gif" if (DASH / "gifs" / f"{n}.gif").exists() else None
    base = [r["score"] for n, r in runs.items() if re.fullmatch(r"base(_s\d+)?", n) and r["state"] == "done"]
    data = {"updated": time.strftime("%Y-%m-%d %H:%M"), "base": round(float(np.mean(base)), 3) if base else None,
            "counts": {s: sum(r["state"] == s for r in runs.values()) for s in ("running", "queued", "done")},
            "good": GOOD_SCORE, "good_repair": GOOD_REPAIR, "runs": runs, "tree": build_tree(runs, load_hypotheses())}
    (DASH / "data.js").write_text("window.NPA_DASH = " + json.dumps(data, separators=(",", ":")) + ";\n")
    print("data.js:", len(runs), "runs,", data["counts"])
    if a.deploy:
        r = subprocess.run(["rsync", "-a", "--delete", "-e", "ssh -o BatchMode=yes -o ConnectTimeout=15", f"{DASH}/",
                            "ubuntu@129.80.219.24:~/npa-demo/lab/"], capture_output=True, text=True)
        print("deployed" if r.returncode == 0 else "deploy failed: " + r.stderr[:200])


if __name__ == "__main__":
    main()
