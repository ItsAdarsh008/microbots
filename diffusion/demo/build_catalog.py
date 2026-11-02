"""Merge the Hub manifest (config + K=64 footprint IoU per run) with results/*/result.json into demo/catalog.json,
the table behind the dashboard.   python demo/build_catalog.py"""
import json, os
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
SPACINGS_PER_RADIUS = 16.4        # r=0.25 -> 4.1 spacings (journal tick 44)
rows = []
for r in json.load(open(os.path.join(HERE, "manifest.json"))):
    name, c = r["run"], r["config"]
    m = r.get("metrics_K64_val_ood_emoji") or {}
    d = os.path.join(ROOT, "results", name)
    full = json.load(open(os.path.join(d, "result.json"))) if os.path.exists(os.path.join(d, "result.json")) else {}
    start = c.get("start") or "gauss"
    rows.append(dict(
        run=name, group=name.split("_")[0], data=c.get("data", "shapes"), seed=c.get("seed", 0),
        radius=c.get("radius"), spacings=round(c["radius"] * SPACINGS_PER_RADIUS, 1) if c.get("radius") else None,
        layers=c.get("layers"), cond=c.get("cond"), coarse=c.get("coarse") or 0, time=c.get("time", True), tnoise=c.get("tnoise") or 0,
        start=start, repair=bool(c.get("damage_p")), damage_p=c.get("damage_p") or 0, aux_in=c.get("aux_in") or 0,
        gstate=c.get("gstate") or "none", iters=c.get("iters"),
        iou={k: m.get(k, {}).get("iou") for k in ("val", "ood_family", "emoji")},
        cover={k: m.get(k, {}).get("cover") for k in ("val", "ood_family", "emoji")},
        coll={k: m.get(k, {}).get("coll") for k in ("val", "ood_family", "emoji")},
        w2={k: m.get(k, {}).get("w2") for k in ("val", "ood_family", "emoji")},
        train_min=round(full.get("train_seconds", 0) / 60, 1) or None,
        has_pics=os.path.exists(os.path.join(d, "samples.png")), has_gif=os.path.exists(os.path.join(d, "traj.gif")),
        config=c))
json.dump(rows, open(os.path.join(HERE, "catalog.json"), "w"), separators=(",", ":"))
print(len(rows), "runs;", sum(1 for x in rows if x["iou"]["ood_family"] is not None), "with IoU")
