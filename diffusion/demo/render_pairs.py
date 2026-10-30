"""Pictures of chosen models, each started the way it was trained, next to the no-learning rule.
   python demo/render_pairs.py out.png model:start[:label] ...   (run in diffusion/ on the demo box)"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import serve, evalm
from model import make_maps
Lt = np.load(os.path.join(serve.ROOT, "data", "letters.npz")); ln = [str(n) for n in Lt["names"]]
items = [("emoji", s, serve.PRE[s]) for s in ["star", "ring", "anchor", "butterfly", "key", "snowflake", "fish", "wrench"]] + \
        [("letter", n, (Lt["occ"][ln.index(n)], Lt["sdf"][ln.index(n)])) for n in ["A-bold", "R-bold", "K-thin", "S-thin", "E-thin", "H-bold", "Z-thin", "B-thin"]]
specs = [a.split(":") for a in sys.argv[2:]]
def maps_of(o, s, c=0): return make_maps(torch.tensor(o)[None].cuda(), torch.tensor(s)[None].cuda(), c)
def rule(o, s):
    g = torch.Generator().manual_seed(0); x0 = (torch.randn((1, 256, 2), generator=g) * 0.7).cuda()
    return evalm.classical(x0, maps_of(o, s), torch.tensor([evalm.spacing(o)]).cuda(), steps=200)[0].cpu().numpy()
def run(name, start, o, s):
    m, cfg = serve.MODELS[name]; maps = maps_of(o, s, cfg.get("coarse", 0)); g = torch.Generator().manual_seed(0)
    x = evalm.init_noise((1, 256, 2), g, dict(cfg, start={"noise": "gauss", "pile_c": "pile_fps_c"}.get(start, start)))
    with torch.no_grad():
        for i in range(16): x = x + m(x, torch.full((1,), i / 16, device="cuda"), maps) / 16
    return x[0].cpu().numpy()
fig, ax = plt.subplots(len(specs) + 1, len(items), figsize=(1.6 * len(items), 1.6 * (len(specs) + 1)))
for c, (kind, nm, (o, s)) in enumerate(items):
    for r in range(len(specs) + 1):
        X = rule(o, s) if r == 0 else run(specs[r - 1][0], specs[r - 1][1], o, s)
        a = ax[r, c]; a.imshow(o > 0.5, extent=(-1, 1, -1, 1), origin="lower", cmap="Greys", alpha=0.25, vmin=0, vmax=1.4)
        a.scatter(X[:, 0], X[:, 1], s=2.5, c="#d1432b"); a.set_xlim(-1.3, 1.3); a.set_ylim(-1.3, 1.3); a.set_aspect("equal"); a.set_xticks([]); a.set_yticks([])
        if c == 0: a.set_ylabel("rule" if r == 0 else (specs[r - 1][2] if len(specs[r - 1]) > 2 else specs[r - 1][0][:18]), fontsize=6)
        if r == 0: a.set_title(nm, fontsize=6)
fig.tight_layout(); fig.savefig(sys.argv[1], dpi=70); print("saved")
