"""Summarise every finished run in runs/*/result.json as a table (and write lab/RESULTS.md).

    python lab/report.py [--shape lizard] [--top 30]

score = mean loss after damage and recovery (lower is better; it includes how well the lizard grew);
ctrl = loss of the undamaged lizard at the same step (growth quality); drift = how the loss changes by step
1000 (negative: still improving, large positive: falling apart). easy / hard = excess loss over the control,
averaged over wipe, shuffle, noise / over cut, drag, clump: THIS is the repair signal (0 = fully repaired).
Rhard = repair fraction on cut/drag/clump: 1 - (loss still missing) / (loss the damage caused at once); 1 = fully
repaired, 0 = nothing repaired. It does not reward a lizard that simply grew badly. The per-kind columns are the excess for each kind. Early runs (3,000 iterations) mostly differ in how fast
they learn to grow, so compare repair through easy / hard, and confirm at 10,000 iterations.
Rows with seeds are averaged when 'base', 'base_s1', 'base_s2' (etc.) exist.
"""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RUNS = HERE.parent / "runs"
KINDS = ("wipe", "cut", "drag", "clump", "shuffle", "noise")


def load():
    rows = []
    for p in sorted(RUNS.glob("*/result.json")):
        try:
            d = json.loads(p.read_text())
        except ValueError:
            continue
        sm = d.get("summary")
        if not sm:
            continue
        cfg = d.get("config", {})
        repair = {}
        ev = d.get("eval", {})
        if ev.get("damage") and ev.get("growth"):
            ctrl = np.array(ev["growth"]["loss_control"])
            for kind, kd in ev["damage"].items():
                hurt = np.array(kd["loss_after_damage"]) - ctrl[None, :]       # what the damage cost at once
                left = np.array(kd["loss_final"]) - ctrl[None, :]               # what is still missing afterwards
                repair[kind] = float(np.clip(1.0 - left.mean() / max(hurt.mean(), 0.02), -1.0, 1.0))
        shape = d.get("shape", cfg.get("shape", "lizard"))
        if cfg.get("train", {}).get("mode") == "agnostic":      # shape-agnostic healing: held-out shapes
            shape = "agnostic"
        if "shapes" in cfg:      # one network, many shapes: its own table
            shape = "multi:" + ",".join(cfg["shapes"])
        rows.append({"name": d["name"], "shape": shape, "params": d.get("params"),
                     "iters": d.get("iters_done"), "minutes": d.get("train_minutes"), "seed": cfg.get("seed", 0),
                     "score": sm["score"], "excess": sm["excess"], "control": sm["control_loss"], "drift": sm["drift"],
                     "kinds": sm["per_kind_score"], "kind_excess": sm["per_kind_excess"], "repair": repair, "cfg": cfg})
    return rows


def fmt(r, base=None):
    ke = r["kind_excess"]
    easy = np.mean([ke[x] for x in ("wipe", "shuffle", "noise")])
    hard = np.mean([ke[x] for x in ("cut", "drag", "clump")])
    rp = r["repair"]
    rh = np.mean([rp[x] for x in ("cut", "drag", "clump")]) if rp else float("nan")
    k = f"{easy:+6.3f} {hard:+6.3f} {rh:5.2f} " + " ".join(f"{ke[x]:+5.2f}" for x in KINDS)
    delta = f"{r['score'] - base:+6.3f}" if base is not None else "      "
    mins = f"{r['minutes']:4.0f}" if r["minutes"] else "   -"
    return (f"{r['name'][:30]:30s} {r['score']:6.3f} {delta} {r['control']:6.3f} {r['drift']:+6.3f}  {k}  "
            f"{(r['params'] or 0):7,d} {mins}m")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shape", default=None)
    ap.add_argument("--top", type=int, default=60)
    args = ap.parse_args()
    rows = load()
    by_shape = defaultdict(list)
    for r in rows:
        by_shape[r["shape"]].append(r)
    lines = []
    header = (f"{'name':30s} {'score':>6s} {'d.base':>6s} {'ctrl':>6s} {'drift':>6s}  {'easy':>6s} {'hard':>6s} "
              + " ".join(f"{x[:5]:>5s}" for x in KINDS) + "   params  train")
    for shape, rs in sorted(by_shape.items()):
        if args.shape and shape != args.shape:
            continue
        base_rows = [r for r in rs if re.fullmatch(r"base(_s\d+)?", r["name"])]
        base = float(np.mean([r["score"] for r in base_rows])) if base_rows else None
        lines += [f"\n## {shape}  ({len(rs)} runs" + (f"; base score {base:.3f} +- {np.std([r['score'] for r in base_rows]):.3f} over {len(base_rows)} seeds" if base is not None else "") + ")", header]
        for r in sorted(rs, key=lambda r: r["score"])[:args.top]:
            lines.append(fmt(r, base))
    text = "\n".join(lines)
    print(text)
    (HERE / "RESULTS.md").write_text("```\n" + text.strip() + "\n```\n")


if __name__ == "__main__":
    main()
