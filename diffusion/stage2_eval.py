"""Stage 2, version 0: the trained swarm with physical bodies. No retraining.
Each ROUND every bot moves by the model's step, but at most vmax spacings; then overlapping bodies are pushed apart so that no
two bodies of diameter b (in spacings) overlap. Sensing is the model's own radius (r=0.14 is about 2.3 spacings = the v1 arm reach).
    python stage2_eval.py runs/<name>
b = body diameter / spacing: the v1 bot is 30 mm and touches at 34.5 mm, so b is about 0.87 to 1.0 when 1 spacing = 1 touching bot."""
import json, sys
import numpy as np, torch
from model import build, make_maps
import evalm
from evalm import metrics, spacing, agg
run = sys.argv[1]
cfg = json.load(open(f"runs/{run}/result.json"))["config"]; evalm.COARSE = cfg.get("coarse", 0)
D = np.load(f"data/{cfg.get('data','shapes')}.npz")
m = build(cfg).cuda().eval(); m.load_state_dict(torch.load(f"runs/{run}/ckpt.pt")["ema"])
n = 96
occ, sdf, ref = D["va_occ"][:n], D["va_sdf"][:n], D["va_pts"][:n]
maps = make_maps(torch.tensor(occ).cuda(), torch.tensor(sdf).cuda(), cfg.get("coarse", 0))
sp = torch.tensor([spacing(o) for o in occ], device="cuda")[:, None, None]


def separate(x, dmin, iters):
    for _ in range(iters):
        d = x[:, :, None] - x[:, None]
        dist = d.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        x = x + (torch.relu(dmin[..., None] - dist) / 2 * d / dist * (dist > 1e-5)).sum(2)
    return x


@torch.no_grad()
def go(K, vmax, b):
    g = torch.Generator().manual_seed(7)
    x = evalm.init_noise((n, 256, 2), g, cfg)
    travelled = torch.zeros(n, 256, device="cuda")
    for i in range(K):
        t = torch.full((n,), i / K, device="cuda")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            v = m(x, t, maps).float()
        dx = v / K
        if vmax:
            dx = dx * torch.clamp(vmax * sp / dx.norm(dim=-1, keepdim=True).clamp_min(1e-9), max=1.0)
        x1 = separate(x + dx, b * sp, 6)
        travelled += (x1 - x).norm(dim=-1)
        x = x1
    x = separate(x, b * sp, 30)
    return x.cpu().numpy(), (travelled / sp[:, :, 0]).mean().item()


print(f"{run}: no retraining; pile start; sensing = the model's radius; 96 val shapes")
print(f"{'rounds':>6s} {'max step (spacings/round)':>26s} {'body (spacings)':>16s}   IoU  cover  overlap-pairs  path(sp)")
for K in (16, 32, 64):
    for vmax in (None, 1.0, 0.5, 0.25):
        for b in (0.0, 0.85, 1.0):
            X, path = go(K, vmax, b)
            r = agg([metrics(X[i], ref[i], occ[i], sdf[i]) for i in range(n)])
            spn = sp[:, 0, 0].cpu().numpy()
            ov = np.mean([((np.sqrt(((X[i][:, None] - X[i][None]) ** 2).sum(-1)) + np.eye(256) * 9) < 0.95 * b * spn[i]).sum() / 2 / 256 for i in range(n)]) if b else 0.0
            print(f"{K:6d} {('none' if vmax is None else vmax):>26} {b:16.2f}   {r['iou']:.2f}  {r['cover']:.2f}  {ov:12.3f}  {path:6.1f}", flush=True)
