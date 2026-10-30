"""Side-by-side pictures: the no-learning rule (project onto the shape, spread by local repulsion) vs trained models."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import serve, evalm
from model import make_maps
shp = ["star", "ring", "anchor", "butterfly", "key", "snowflake", "fish", "wrench"]
fig, ax = plt.subplots(3, len(shp), figsize=(2.2 * len(shp), 2.2 * 3))
def panel(a, X, occ, title):
    a.imshow(occ > 0.5, extent=(-1, 1, -1, 1), origin="lower", cmap="Greys", alpha=0.25, vmin=0, vmax=1.4)
    a.scatter(X[:, 0], X[:, 1], s=4, c="#d1432b"); a.set_xlim(-1.3, 1.3); a.set_ylim(-1.3, 1.3); a.set_aspect("equal"); a.set_xticks([]); a.set_yticks([]); a.set_title(title, fontsize=7)
rows = ["no-learning rule (200 steps)", "global, scattered start (16 steps)", "global + hard-core rule"]
for c, s in enumerate(shp):
    occ, sdf = serve.PRE[s]
    maps = make_maps(torch.tensor(occ)[None].cuda(), torch.tensor(sdf)[None].cuda())
    g = torch.Generator().manual_seed(0)
    x0 = (torch.randn((1, 256, 2), generator=g) * 0.7).cuda()
    Xc = evalm.classical(x0, maps, torch.tensor([evalm.spacing(occ)]).cuda(), steps=200)[0].cpu().numpy()
    mc = evalm.metrics(Xc, Xc, occ, sdf); panel(ax[0, c], Xc, occ, f"in {mc['inside']:.2f} cov {mc['cover']:.2f} coll {mc['coll']:.2f}")
    j = serve.run(dict(model="s2_cond_global", shape=s, start="noise", k=16, seed=0)); panel(ax[1, c], np.array(j["frames"][-1]), occ, f"in {j['inside']:.2f} cov {j['cover']:.2f} coll {j['coll']:.2f}")
    j = serve.run(dict(model="s2_cond_global", shape=s, start="noise", k=16, seed=0, hardcore=True)); panel(ax[2, c], np.array(j["frames"][-1]), occ, f"in {j['inside']:.2f} cov {j['cover']:.2f} coll {j['coll']:.2f}")
    ax[0, c].set_xlabel(s, fontsize=7); ax[0, c].xaxis.set_label_position("top")
for r, l in enumerate(rows):
    ax[r, 0].set_ylabel(l, fontsize=7)
fig.tight_layout(); fig.savefig(sys.argv[1], dpi=75); print("saved")
