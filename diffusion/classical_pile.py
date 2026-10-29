"""Is the no-learning baseline just under-run from a pile start?  Run it for more steps and a bigger step size.
    python classical_pile.py            (v2 data, well-spaced pile, 96 val + 96 held-out-family shapes)"""
import numpy as np, torch
import evalm
from evalm import metrics, spacing, agg, classical
D = np.load("data/shapes2.npz")
cfg = dict(start="pile_fps", sigma=0.7)
for name, p in (("val", "va"), ("ood_family", "ood")):
    occ, sdf, ref = D[p + "_occ"][:96], D[p + "_sdf"][:96], D[p + "_pts"][:96]
    maps = evalm.make_maps(torch.tensor(occ).cuda(), torch.tensor(sdf).cuda())
    sp = torch.tensor([spacing(o) for o in occ], device="cuda")
    x0 = evalm.init_noise((96, 256, 2), torch.Generator().manual_seed(7), cfg)
    for steps in (200, 600, 1500):
        X = classical(x0, maps, sp, steps=steps).cpu().numpy()
        r = agg([metrics(X[i], ref[i], occ[i], sdf[i]) for i in range(96)])
        print(f"{name:10s} classical {steps:5d} steps: w2 {r['w2']:.2f} inside {r['inside']:.2f} cover {r['cover']:.2f} coll {r['coll']:.3f}")
