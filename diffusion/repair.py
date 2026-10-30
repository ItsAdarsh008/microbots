"""Self-repair test, no retraining: form the shape, damage it, rewind every particle's clock to t0, re-run.
    python repair.py runs/<name>
Damage: 'chunk' removes all particles inside a disc of radius 0.3 around a random point on the shape;
'random' removes a random 20% of particles. Time-free models just keep stepping (no clock to rewind).
Metrics use the surviving particle count for spacing (cover = pixels with a particle within 1 spacing).
before / damaged / repaired (rewound to t0 = 0.5 and 0.8)."""
import json, sys
import numpy as np, torch
from model import Denoiser, build
import evalm
from evalm import metrics, agg
run = sys.argv[1]
cfg = json.load(open(f"{run}/result.json"))["config"]
evalm.COARSE = cfg.get("coarse", 0)
D = np.load(f"data/{cfg.get('data','shapes')}.npz")
m = build(cfg).cuda().eval()
m.load_state_dict(torch.load(f"{run}/ckpt.pt")["ema"])
n = 64
occ, sdf, ref = D["va_occ"][:n], D["va_sdf"][:n], D["va_pts"][:n]
maps = evalm.make_maps(torch.tensor(occ).cuda(), torch.tensor(sdf).cuda())
K = 16


@torch.no_grad()
def run_flow(x, t0, steps):
    for i in range(steps):
        t = torch.full((x.shape[0],), t0 + (1 - t0) * i / steps, device="cuda")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            v = m(x, t, maps).float()
        x = x + v * (1 - t0) / steps
    return x


g = torch.Generator().manual_seed(7)
x = evalm.init_noise((n, 256, 2), g, cfg)
x = run_flow(x, 0.0, K)
rng = np.random.default_rng(3)
Xf = x.cpu().numpy()


def score(Xs, tag):
    # per-set arrays with different counts: evaluate set by set
    r = agg([metrics(Xs[i], ref[i], occ[i], sdf[i]) | {} for i in range(n)]) if False else None
    rows = []
    for i in range(n):
        X = Xs[i]
        sp_ref = ref[i][: len(X)]   # reference of matching size (subset; w2 not reported)
        mm = metrics(X, sp_ref, occ[i], sdf[i]); mm.pop("w2"); rows.append(mm)
    r = agg(rows)
    print(f"  {tag:34s} N~{np.mean([len(X) for X in Xs]):.0f} inside {r['inside']:.2f} cover {r['cover']:.2f} coll {r['coll']:.3f} nn {r['nn']:.2f}")


def damage(kind):
    out = []
    for i in range(n):
        X = Xf[i]
        if kind == "chunk":
            c = ref[i][rng.integers(256)]
            keep = np.linalg.norm(X - c, axis=1) > 0.3
        else:
            keep = rng.random(256) > 0.2
        out.append(X[keep])
    return out


print(run, "(conditional, K=%d)" % K)
score(list(Xf), "formed (no damage)")
for kind in ("chunk", "random"):
    dm = damage(kind)
    print(f" damage = {kind}")
    score(dm, "damaged, no repair")
    for t0 in (((0.0,) if cfg.get("damage_p") else ()) + ((0.5, 0.8) if cfg["time"] else (0.5,))):
        reps = []
        for i in range(n):
            xi = torch.tensor(dm[i], device="cuda")[None]
            mp = maps[i:i + 1]
            xs = xi
            for s in range(K):
                t = torch.full((1,), t0 + (1 - t0) * s / K, device="cuda") if cfg["time"] else torch.zeros(1, device="cuda")
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    v = m(xs, t, mp).float()
                xs = xs + v * (1 - t0) / K
            reps.append(xs[0].cpu().numpy())
        score(reps, f"repaired, rewound to t0={t0}" if cfg["time"] else "repaired, kept stepping")
