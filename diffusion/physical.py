"""What happens when the sampler is made physical? No retraining; conditional runs.
    python physical.py runs/<name>
Variants: speed cap (a particle moves at most vmax spacings per step), hard-core spacing (after every step push
apart any pair closer than 0.7 spacings), and the number of steps K. Starts as the run was trained (gauss or pile).
Metrics per variant (spacing units): w2 to the reference, inside, cover, coll (<0.35 sp), min-gap (5th percentile of
nearest-neighbour distance), and the total path length per particle (spacings) = how far a bot must travel."""
import json, sys
import numpy as np, torch
from model import Denoiser
import evalm
from evalm import metrics, spacing, agg
run = sys.argv[1]
cfg = json.load(open(f"{run}/result.json"))["config"]
evalm.COARSE = cfg.get("coarse", 0)
D = np.load(f"data/{cfg.get('data','shapes')}.npz")
m = Denoiser(cfg["d"], cfg["heads"], cfg["layers"], cfg["radius"], cfg["cond"], cfg["time"]).cuda().eval()
m.load_state_dict(torch.load(f"{run}/ckpt.pt")["ema"])
n = 96
occ, sdf, ref = D["va_occ"][:n], D["va_sdf"][:n], D["va_pts"][:n]
maps = evalm.make_maps(torch.tensor(occ).cuda(), torch.tensor(sdf).cuda())
sp = torch.tensor([spacing(o) for o in occ], device="cuda")[:, None, None]


def separate(x, dmin, iters=3):
    for _ in range(iters):
        d = x[:, :, None] - x[:, None]
        dist = d.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        push = torch.relu(dmin[..., None] - dist) / 2 * d / dist
        push = push * (dist > 1e-5)
        x = x + push.sum(2)
    return x


@torch.no_grad()
def go(K, vmax=None, hard=False):
    g = torch.Generator().manual_seed(7)
    x = evalm.init_noise((n, 256, 2), g, cfg)
    path = torch.zeros(n, 256, device="cuda")
    for i in range(K):
        t = torch.full((n,), i / K, device="cuda")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            v = m(x, t, maps).float()
        dx = v / K
        if vmax is not None:
            nrm = dx.norm(dim=-1, keepdim=True).clamp_min(1e-9)
            dx = dx * torch.clamp(vmax * sp / nrm, max=1.0)
        x1 = x + dx
        if hard:
            x1 = separate(x1, 0.7 * sp)
        path += (x1 - x).norm(dim=-1)
        x = x1
    return x.cpu().numpy(), (path / sp[:, :, 0]).mean().item()


print(f"{'variant':26s} w2   inside cover coll  gap5%  path(sp)")
for name, kw in [("K=8", dict(K=8)), ("K=32", dict(K=32)), ("K=64", dict(K=64)),
                 ("K=32 + hardcore", dict(K=32, hard=True)),
                 ("K=32 vmax 1sp/step", dict(K=32, vmax=1.0)), ("K=32 vmax 1sp + hardcore", dict(K=32, vmax=1.0, hard=True)),
                 ("K=64 vmax 0.5sp", dict(K=64, vmax=0.5)), ("K=64 vmax 0.5sp + hardcore", dict(K=64, vmax=0.5, hard=True))]:
    X, path = go(**kw)
    r = agg([metrics(X[i], ref[i], occ[i], sdf[i]) for i in range(n)])
    gaps = np.concatenate([np.sort(np.sqrt(((x[:, None] - x[None]) ** 2).sum(-1) + np.eye(256) * 1e9).min(1))[:13] / spacing(occ[i]) for i, x in enumerate(X)])
    print(f"{name:26s} {r['w2']:.2f} {r['inside']:.2f}   {r['cover']:.2f}  {r['coll']:.3f} {np.percentile(gaps, 50):.2f}   {path:.1f}")
