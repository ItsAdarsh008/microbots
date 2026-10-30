"""Does it LOOK right?  Mask IoU / precision / recall of the bots' footprint (pixels within 0.7 spacings of a bot) against
the target mask, for the no-learning rule, trained models, and learned-then-polished hybrids.   Run on the demo box."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch
from scipy.spatial import cKDTree
import serve, evalm
from model import make_maps
D = np.load(os.path.join(serve.ROOT, "data", "shapes2.npz"))
AX = (np.arange(64) + 0.5) * 2 / 64 - 1
PIX = np.stack(np.meshgrid(AX, AX), -1).reshape(-1, 2)          # (x, y) per pixel, row-major in y
def foot(X, sp):
    d, _ = cKDTree(X).query(PIX); return (d < 0.7 * sp).reshape(64, 64)
def score(X, occ):
    sp = evalm.spacing(occ, len(X)); f = foot(X, sp); t = occ > 0.5
    inter = (f & t).sum(); return inter / max((f | t).sum(), 1), inter / max(f.sum(), 1), inter / max(t.sum(), 1)
sets = {"emoji (12 x 4 starts)": [(D["em_occ"][i], D["em_sdf"][i]) for i in range(12)] * 4,
        "held-out family (96)": [(D["ood_occ"][i], D["ood_sdf"][i]) for i in range(96)],
        "familiar (96)": [(D["va_occ"][i], D["va_sdf"][i]) for i in range(96)]}
def gen(model, occ, sdf, start, k=16, seed=0):
    m, cfg = serve.MODELS[model]; maps = make_maps(torch.tensor(occ)[None].cuda(), torch.tensor(sdf)[None].cuda(), cfg.get("coarse", 0))
    g = torch.Generator().manual_seed(seed); x = evalm.init_noise((1, 256, 2), g, dict(cfg, start={"noise": "gauss"}.get(start, "pile_fps_c" if start == "pile_c" else start)))
    with torch.no_grad():
        for i in range(k):
            t = torch.full((1,), i / k, device="cuda"); x = x + m(x, t, maps) / k
    return x, maps
methods = [("no-learning rule, 200 steps", None, None, 0), ("global (scattered start), 16 steps", "s2_cond_global", "noise", 0),
           ("global + 60 polish steps of the rule", "s2_cond_global", "noise", 60), ("local r=0.6 (scattered)", "s2_cond_local_r60_L4", "noise", 0),
           ("local r=0.6 + polish", "s2_cond_local_r60_L4", "noise", 60), ("reference swarm (pile, 64-number map)", "s4_ref_local_r60_g8_tn15_dmg_pfc", "pile_c", 0)]
print(f"{'method':42s}" + "".join(f"{n:>30s}" for n in sets))
for name, model, start, polish in methods:
    row = f"{name:42s}"
    for sn, S in sets.items():
        sc = []
        for i, (occ, sdf) in enumerate(S):
            seed = i // 12 if "emoji" in sn else 0
            full_maps = make_maps(torch.tensor(occ)[None].cuda(), torch.tensor(sdf)[None].cuda())
            sp = torch.tensor([evalm.spacing(occ)]).cuda()
            if model is None:
                g = torch.Generator().manual_seed(seed); x0 = (torch.randn((1, 256, 2), generator=g) * 0.7).cuda()
                X = evalm.classical(x0, full_maps, sp, steps=200)
            else:
                X, _ = gen(model, occ, sdf, start, seed=seed)
                if polish:
                    X = evalm.classical(X, full_maps, sp, steps=polish)
            sc.append(score(X[0].cpu().numpy(), occ))
        a = np.mean(sc, 0); row += f"   IoU {a[0]:.2f} prec {a[1]:.2f} rec {a[2]:.2f}  "
    print(row)
