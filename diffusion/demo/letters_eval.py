"""IoU on letters and a picture grid.   python demo/letters_eval.py out.png model1 model2 ...  (run in diffusion/ on the demo box)"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import serve, evalm
import iou_eval_lib as L
from model import make_maps
Lt = np.load(os.path.join(serve.ROOT, "data", "letters.npz")); names = [str(n) for n in Lt["names"]]
def maps_of(occ, sdf, coarse=0):
    return make_maps(torch.tensor(occ)[None].cuda(), torch.tensor(sdf)[None].cuda(), coarse)
def rule(occ, sdf):
    g = torch.Generator().manual_seed(0); x0 = (torch.randn((1, 256, 2), generator=g) * 0.7).cuda()
    return evalm.classical(x0, maps_of(occ, sdf), torch.tensor([evalm.spacing(occ)]).cuda(), steps=200)[0].cpu().numpy()
def model_x(name, occ, sdf, polish=0):
    m, cfg = serve.MODELS[name]; maps = maps_of(occ, sdf, cfg.get("coarse", 0))
    g = torch.Generator().manual_seed(0); x = evalm.init_noise((1, 256, 2), g, cfg)      # each model starts the way it was trained
    with torch.no_grad():
        for i in range(16):
            x = x + m(x, torch.full((1,), i / 16, device="cuda"), maps) / 16
    if polish:
        x = evalm.classical(x, maps_of(occ, sdf), torch.tensor([evalm.spacing(occ)]).cuda(), steps=polish)
    return x[0].cpu().numpy()
models = sys.argv[2:]
ceil = np.mean([L.score(Lt["pts"][i], Lt["occ"][i]) for i in range(len(names))], 0)
print(f"ceiling (even sample): IoU {ceil[0]:.2f} rec {ceil[2]:.2f}")
def table(label, fn):
    bold = [i for i, n in enumerate(names) if n.endswith("bold")]; thin = [i for i, n in enumerate(names) if n.endswith("thin")]
    out = []
    for idx in (bold, thin):
        sc = [L.score(fn(Lt["occ"][i], Lt["sdf"][i]), Lt["occ"][i]) for i in idx]; out.append((np.mean([a[0] for a in sc]), np.mean([a[2] for a in sc])))
    print(f"{label:38s} bold letters IoU {out[0][0]:.2f} rec {out[0][1]:.2f}   thin letters IoU {out[1][0]:.2f} rec {out[1][1]:.2f}", flush=True)
table("no-learning rule (200 steps)", rule)
for n in models:
    table(n, lambda o, s, n=n: model_x(n, o, s)); table(n + " +polish60", lambda o, s, n=n: model_x(n, o, s, 60))
show = ["A-bold", "R-bold", "K-thin", "S-thin", "E-thin", "H-bold", "Z-thin", "B-thin"]
fig, ax = plt.subplots(len(models) + 1, len(show), figsize=(2.0 * len(show), 2.0 * (len(models) + 1)))
for c, s in enumerate(show):
    i = names.index(s); occ, sdf = Lt["occ"][i], Lt["sdf"][i]
    for r in range(len(models) + 1):
        X = rule(occ, sdf) if r == 0 else model_x(models[r - 1], occ, sdf)
        a = ax[r, c]; a.imshow(occ > 0.5, extent=(-1, 1, -1, 1), origin="lower", cmap="Greys", alpha=0.25, vmin=0, vmax=1.4)
        a.scatter(X[:, 0], X[:, 1], s=3.5, c="#d1432b"); a.set_xlim(-1.3, 1.3); a.set_ylim(-1.3, 1.3); a.set_aspect("equal"); a.set_xticks([]); a.set_yticks([])
        if c == 0: a.set_ylabel("no-learning rule" if r == 0 else models[r - 1].replace("s6_", "").replace("s2_cond_", ""), fontsize=7)
        if r == 0: a.set_title(s, fontsize=7)
fig.tight_layout(); fig.savefig(sys.argv[1], dpi=72); print("saved")
