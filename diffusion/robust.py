"""S1d-lite: how fragile is a trained model to things a real swarm has and the training did not?
No retraining. Conditional runs only.   python robust.py runs/<name> [K]
Variants: clock jitter (each particle has its own wrong clock), asynchronous updates (only half of the
particles move at each step), motion noise, and a particle count the model never saw.
Reported per variant: inside / cover / coll / nn (spacing units; sp uses the actual particle count) and
w2 where the count is 256 (the reference has 256 points)."""
import json, sys
import numpy as np, torch
from model import Denoiser
import evalm
from evalm import metrics, spacing, agg
run, K = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 16
cfg = json.load(open(f"{run}/result.json"))["config"]
assert cfg["cond"] == "map"
evalm.COARSE = cfg.get("coarse", 0)
D = np.load("data/shapes.npz")
m = Denoiser(cfg["d"], cfg["heads"], cfg["layers"], cfg["radius"], cfg["cond"], cfg["time"]).cuda().eval()
m.load_state_dict(torch.load(f"{run}/ckpt.pt")["ema"])
n_sets = 96
occ, sdf, ref = D["va_occ"][:n_sets], D["va_sdf"][:n_sets], D["va_pts"][:n_sets]
maps = evalm.make_maps(torch.tensor(occ).cuda(), torch.tensor(sdf).cuda())
R_ax = (np.arange(64) + 0.5) * evalm.PX - 1


def fps(o, n, rng):
    ys, xs = np.nonzero(o > 0.5); c = np.stack([R_ax[xs], R_ax[ys]], 1)
    s = [int(rng.integers(len(c)))]; d = ((c - c[s[0]]) ** 2).sum(1)
    for _ in range(n - 1):
        j = int(np.argmax(d)); s.append(j); d = np.minimum(d, ((c - c[j]) ** 2).sum(1))
    return c[s]


@torch.no_grad()
def run_variant(N=256, jitter=0.0, async_p=1.0, motion=0.0, seed=7):
    g = torch.Generator().manual_seed(seed)
    x = (torch.randn((n_sets, N, 2), generator=g) * cfg["sigma"]).cuda()
    steps = int(round(K / async_p))
    for i in range(steps):
        t = torch.full((n_sets, N), i / steps, device="cuda")
        if jitter:
            t = (t + torch.randn_like(t) * jitter).clamp(0, 1)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            v = m(x, t, maps).float()
        upd = (torch.rand(n_sets, N, 1, device="cuda") < async_p).float()
        x = x + upd * v / K   # an update moves a particle by one nominal step; async just needs K/p rounds to get K updates on average
        if motion:
            sp = torch.tensor([spacing(o, N) for o in occ], device="cuda")[:, None, None]
            x = x + torch.randn_like(x) * motion * sp / np.sqrt(steps)
    return x.cpu().numpy()


rng = np.random.default_rng(0)
rows = {}
variants = {"base": {}, "jitter0.1": dict(jitter=0.1), "jitter0.25": dict(jitter=0.25), "async0.5": dict(async_p=0.5),
            "motion0.3sp": dict(motion=0.3), "N128": dict(N=128), "N512": dict(N=512)}
for name, kw in variants.items():
    X = run_variant(**kw)
    N = kw.get("N", 256)
    refs = ref if N == 256 else np.stack([fps(o, N, rng) for o in occ])
    res = agg([metrics(X[i], refs[i], occ[i], sdf[i]) for i in range(n_sets)])
    rows[name] = res
    print(f"{name:12s} " + " ".join(f"{k} {v:.2f}" for k, v in res.items() if N == 256 or k != "w2"))
json.dump(rows, open(f"{run}/robust.json", "w"), indent=1)
