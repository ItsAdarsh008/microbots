"""Score every map-conditioned run on ONE fixed test set (never-trained shapes), same protocol for all:
256 bots, 64 steps, seed 0, each model from its own default start.  Reference rows: the no-learning rule and the ceiling.
    python demo/eval_all.py SHARD NSHARDS [--keep]        writes demo/lb/shard_<i>.json (resumable)
Weights are downloaded one at a time and deleted after scoring unless --keep."""
import json, os, sys, time
import numpy as np
os.environ.setdefault("OMP_NUM_THREADS", "2")
import torch
import engine as E
shard, nshard = int(sys.argv[1]), int(sys.argv[2]); keep = "--keep" in sys.argv
torch.set_num_threads(2)
lib = E.shape_library()
LET = [f"letter {c} ({t})" for c in "AEHKRSTY" for t in ("thick", "thin")]
EMO = sorted(n for n in lib if n.startswith("emoji"))
HELD = [f"held-out #{i}" for i in range(0, 80, 5)]
SUITE = {"emoji": EMO, "letters": LET, "held-out": HELD}
out = os.path.join(E.HERE, "lb"); os.makedirs(out, exist_ok=True)
path = os.path.join(out, f"shard_{shard}.json")
done = json.load(open(path)) if os.path.exists(path) else {}
def agg(rows):
    return {g: float(np.mean([rows[s]["iou"] for s in names])) for g, names in SUITE.items()} | \
           {"all": float(np.mean([r["iou"] for r in rows.values()])), "cover": float(np.mean([r["cover"] for r in rows.values()])),
            "coll": float(np.mean([r["coll"] for r in rows.values()])), "prec": float(np.mean([r["prec"] for r in rows.values()])),
            "rec": float(np.mean([r["rec"] for r in rows.values()]))}
def keep_keys(r): return {k: round(float(r[k]), 4) for k in ("iou", "prec", "rec", "cover", "coll")}
if shard == 0 and "_rule" not in done:
    for nm, fn in (("_rule", E.run_rule), ("_ceiling", E.ceiling)):
        rows = {s: keep_keys(fn(*lib[s])) for g in SUITE.values() for s in g}
        done[nm] = dict(agg=agg(rows), per_shape={s: r["iou"] for s, r in rows.items()})
    json.dump(done, open(path, "w"))
runs = sorted(n for n, r in E.CATALOG.items() if r["cond"] == "map")
for name in runs[shard::nshard]:
    if name in done: continue
    t = time.time()
    try:
        rows = {s: keep_keys(E.run_model(name, *lib[s], want_frames=False)) for g in SUITE.values() for s in g}
        done[name] = dict(agg=agg(rows), per_shape={s: r["iou"] for s, r in rows.items()}, secs=round(time.time() - t, 1))
    except Exception as e:
        done[name] = dict(error=repr(e)[:300])
    json.dump(done, open(path, "w"))
    if not keep:
        p = os.path.join(E.CACHE, "weights", name + ".pt")
        if os.path.exists(p): os.remove(p)
    print(shard, name, done[name].get("agg", {}).get("all"), flush=True)
print("shard done", shard)
