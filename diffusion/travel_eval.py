"""Two-phase swarm, no retraining: (1) the pile travels as a rigid body until its centre sits on the shape's
centre (read from the map), (2) the model trained on a CENTRED pile forms the shape.
    python travel_eval.py runs/<pfc run>
Compared: offset pile with no travel phase, offset pile after travel, and the centred pile the model was trained on."""
import json, sys
import numpy as np, torch
from model import Denoiser, build
import evalm
from evalm import metrics, agg, spacing
run = sys.argv[1]
cfg = json.load(open(f"{run}/result.json"))["config"]
evalm.COARSE = cfg.get("coarse", 0)
D = np.load(f"data/{cfg.get('data','shapes')}.npz")
m = build(cfg).cuda().eval()
m.load_state_dict(torch.load(f"{run}/ckpt.pt")["ema"])
AX = (np.arange(64) + 0.5) * evalm.PX - 1
for name, p in (("val", "va"), ("held-out family", "ood")):
    n = 96
    occ, sdf, ref = D[p + "_occ"][:n], D[p + "_sdf"][:n], D[p + "_pts"][:n]
    maps = evalm.make_maps(torch.tensor(occ).cuda(), torch.tensor(sdf).cuda())
    cen = np.stack([np.array([(AX[None, :] * (o > .5)).sum() / (o > .5).sum(), (AX[:, None] * (o > .5)).sum() / (o > .5).sum()]) for o in occ])
    rng = np.random.default_rng(11)
    off = evalm.pile_fps(rng, n, 256, R0=0.7, off=0.5)
    ctr = evalm.pile_fps(np.random.default_rng(12), n, 256, R0=0.7, off=0.0)
    moved = (off - off.mean(1, keepdims=True) + cen[:, None]).astype(np.float32)
    sp = np.array([spacing(o) for o in occ])
    travel = np.linalg.norm(cen - off.mean(1), axis=1) / sp
    for tag, x0 in (("offset pile, no travel", off), ("offset pile, after travel", moved), ("centred pile (as trained)", ctr)):
        X = torch.cat([evalm.sample(m, torch.tensor(x0[i:i + 64]).cuda(), maps[i:i + 64], 32) for i in range(0, n, 64)]).cpu().numpy()
        r = agg([metrics(X[i], ref[i], occ[i], sdf[i]) for i in range(n)])
        print(f"{name:16s} {tag:28s} w2 {r['w2']:.2f} inside {r['inside']:.2f} cover {r['cover']:.2f} coll {r['coll']:.3f}")
    print(f"{name:16s} travel distance to the shape centre: mean {travel.mean():.1f} spacings (max {travel.max():.1f})")
