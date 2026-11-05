"""Table of every finished run in results/ (markdown), main numbers with references.

    python summarize.py [prefix]       # prints the table and writes results/TABLE.md
"""
import glob
import json
import os
import sys

pre = sys.argv[1] if len(sys.argv) > 1 else ""
rows = []
for f in sorted(glob.glob("results/*/result.json")):
    r = json.load(open(f))
    n = r["config"]["name"]
    if not n.startswith(pre):
        continue
    v, h = r["metrics"]["val"], r["metrics"]["held"]
    rows.append((n, v, h, r["config"]))


def f2(x):
    return f"{x:.2f}"


hdr = ("| run | val IoU build | val IoU hold [95% CI] | val hold if all grip | val fallen | held IoU build | held IoU hold | held fallen | "
       "tri/bot overhang (learned / even-frozen) | refs val: pile, ceiling, even-frozen hold |")
lines = [hdr, "|" + "---|" * (hdr.count("|") - 1)]
for n, v, h, c in rows:
    ci = v.get("iou_hold_ci", [0, 0])
    lines.append(f"| {n} | {f2(v['iou_build'])} | {f2(v['iou_hold'])} [{f2(ci[0])}, {f2(ci[1])}] | {f2(v.get('iou_hold_gripall', float('nan')))} | {f2(v['fallen'])} | "
                 f"{f2(h['iou_build'])} | {f2(h['iou_hold'])} | {f2(h['fallen'])} | "
                 f"{f2(v['tri_ov'])} / {f2(v['base_ceilfreeze_tri_ov'])} | "
                 f"{f2(v['base_pile_iou'])}, {f2(v['base_ceiling_iou'])}, {f2(v['base_ceilfreeze_iou_hold'])} |")
out = "\n".join(lines)
print(out)
os.makedirs("results", exist_ok=True)
open("results/TABLE.md", "w").write(out + "\n")
