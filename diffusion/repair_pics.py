"""Pictures of repair: form the shape from the pile, cut out a chunk, restart the clock and let the same model refill it.
    python repair_pics.py out.png runA runB ...      (run dirs under runs/)"""
import sys, json
import numpy as np, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from model import build, make_maps
import evalm
D = np.load("data/shapes2.npz"); Lt = np.load("data/letters.npz"); ln = [str(n) for n in Lt["names"]]
targets = [("anchor", D["em_occ"][0][::-1].copy(), D["em_sdf"][0][::-1].copy()), ("snowflake", D["em_occ"][9][::-1].copy(), D["em_sdf"][9][::-1].copy()),
           ("B thick", Lt["occ"][ln.index("B-bold")], Lt["sdf"][ln.index("B-bold")]), ("S thin", Lt["occ"][ln.index("S-thin")], Lt["sdf"][ln.index("S-thin")])]
runs = sys.argv[2:]
fig, ax = plt.subplots(len(runs), 2 * len(targets), figsize=(1.9 * 2 * len(targets), 1.9 * len(runs)))
AX = (np.arange(64) + 0.5) * 2 / 64 - 1
for r, run in enumerate(runs):
    cfg = json.load(open(f"runs/{run}/result.json"))["config"]; m = build(cfg).cuda().eval(); m.load_state_dict(torch.load(f"runs/{run}/ckpt.pt")["ema"])
    for c, (nm, occ, sdf) in enumerate(targets):
        maps = make_maps(torch.tensor(occ)[None].cuda(), torch.tensor(sdf)[None].cuda(), cfg.get("coarse", 0))
        g = torch.Generator().manual_seed(0); x = evalm.init_noise((1, 256, 2), g, cfg)
        def flow(x, t0=0.0):
            with torch.no_grad():
                for i in range(16): x = x + m(x, torch.full((1,), t0 + (1 - t0) * i / 16, device="cuda"), maps) * (1 - t0) / 16
            return x
        x = flow(x); X = x[0].cpu().numpy()
        ys, xs = np.nonzero(occ > 0.5); k = np.random.default_rng(c).integers(len(xs)); centre = np.array([AX[xs[k]], AX[ys[k]]])
        keep = np.linalg.norm(X - centre, axis=1) > 0.3; cut = torch.tensor(X[keep], dtype=torch.float32, device="cuda")[None]
        rep = flow(cut)[0].cpu().numpy()
        for j, (P, title) in enumerate(((X[keep], f"cut ({(~keep).sum()} bots gone)"), (rep, "repaired"))):
            a = ax[r, 2 * c + j]; a.imshow(occ > 0.5, extent=(-1, 1, -1, 1), origin="lower", cmap="Greys", alpha=0.25, vmin=0, vmax=1.4)
            a.scatter(P[:, 0], P[:, 1], s=2.5, c="#d1432b"); a.set_xlim(-1.3, 1.3); a.set_ylim(-1.3, 1.3); a.set_aspect("equal"); a.set_xticks([]); a.set_yticks([])
            if r == 0: a.set_title(f"{nm}: {title}", fontsize=6)
            if 2 * c + j == 0: a.set_ylabel(run.replace("s17_refv3_r14_", "").replace("s16_refv3_r14_g16", "base (no shared state)"), fontsize=6)
fig.tight_layout(); fig.savefig(sys.argv[1], dpi=75); print("saved")
