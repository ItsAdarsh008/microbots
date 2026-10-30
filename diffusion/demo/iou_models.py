"""IoU (with and without polish) and a picture grid for the models in demo_models/, next to the no-learning rule.
    python demo/iou_models.py out.png name1 name2 ...    (run on the demo box)"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import serve, evalm
import iou_eval_lib as L
from model import make_maps
D = L.D
sets = {"emoji(12x4)": [(D["em_occ"][i], D["em_sdf"][i]) for i in range(12)] * 4, "held-out(96)": [(D["ood_occ"][i], D["ood_sdf"][i]) for i in range(96)],
        "familiar(96)": [(D["va_occ"][i], D["va_sdf"][i]) for i in range(96)]}
def gen(name, occ, sdf, seed, polish=0):
    m, cfg = serve.MODELS[name]; full = make_maps(torch.tensor(occ)[None].cuda(), torch.tensor(sdf)[None].cuda())
    maps = make_maps(torch.tensor(occ)[None].cuda(), torch.tensor(sdf)[None].cuda(), cfg.get("coarse", 0)) if cfg["cond"] == "map" else None
    g = torch.Generator().manual_seed(seed); x = (torch.randn((1, 256, 2), generator=g) * 0.7).cuda()
    with torch.no_grad():
        for i in range(16):
            x = x + m(x, torch.full((1,), i / 16, device="cuda"), maps) / 16
    if polish:
        x = evalm.classical(x, full, torch.tensor([evalm.spacing(occ)]).cuda(), steps=polish)
    return x[0].cpu().numpy()
def classical_x(occ, sdf, seed):
    full = make_maps(torch.tensor(occ)[None].cuda(), torch.tensor(sdf)[None].cuda()); g = torch.Generator().manual_seed(seed)
    x0 = (torch.randn((1, 256, 2), generator=g) * 0.7).cuda()
    return evalm.classical(x0, full, torch.tensor([evalm.spacing(occ)]).cuda(), steps=200)[0].cpu().numpy()
names = sys.argv[2:]
print(f"{'model':36s}" + "".join(f"{n:>26s}" for n in sets))
def row(label, fn):
    r = f"{label:36s}"
    for sn, S in sets.items():
        sc = [L.score(fn(o, s, (i // 12 if 'emoji' in sn else 0)), o) for i, (o, s) in enumerate(S)]
        r += f"     IoU {np.mean([a[0] for a in sc]):.2f} rec {np.mean([a[2] for a in sc]):.2f}   "
    print(r, flush=True)
row("no-learning rule (200 steps)", classical_x)
for n in names:
    row(n, lambda o, s, sd, n=n: gen(n, o, s, sd)); row(n + " +polish60", lambda o, s, sd, n=n: gen(n, o, s, sd, 60))
shp = ["star", "ring", "anchor", "butterfly", "key", "snowflake", "fish", "wrench"]
fig, ax = plt.subplots(len(names) + 1, len(shp), figsize=(2.0 * len(shp), 2.0 * (len(names) + 1)))
for c, s in enumerate(shp):
    occ, sdf = serve.PRE[s]
    for r in range(len(names) + 1):
        X = classical_x(occ, sdf, 0) if r == 0 else gen(names[r - 1], occ, sdf, 0)
        a = ax[r, c]; a.imshow(occ > 0.5, extent=(-1, 1, -1, 1), origin="lower", cmap="Greys", alpha=0.25, vmin=0, vmax=1.4)
        a.scatter(X[:, 0], X[:, 1], s=3.5, c="#d1432b"); a.set_xlim(-1.3, 1.3); a.set_ylim(-1.3, 1.3); a.set_aspect("equal"); a.set_xticks([]); a.set_yticks([])
        if c == 0: a.set_ylabel(("no-learning rule" if r == 0 else names[r - 1].replace("s5a_r25_", "").replace("s6_", "")), fontsize=7)
        if r == 0: a.set_title(s, fontsize=7)
fig.tight_layout(); fig.savefig(sys.argv[1], dpi=72); print("saved")
