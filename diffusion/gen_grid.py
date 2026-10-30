"""Sample many DISTINCT sets from an unconditional run and plot them next to real validation shapes.
    python gen_grid.py runs/<name> [K]
"""
import json, os, sys
import numpy as np, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from model import Denoiser, build
import evalm
run, K = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 32
cfg = json.load(open(f"{run}/result.json"))["config"]
D = np.load("data/shapes.npz")
m = build(cfg).cuda().eval()
m.load_state_dict(torch.load(f"{run}/ckpt.pt")["ema"])
g = torch.Generator().manual_seed(99)
x0 = (torch.randn((48, 256, 2), generator=g) * cfg["sigma"]).cuda()
X = evalm.sample(m, x0, None, K).cpu().numpy()
fig, ax = plt.subplots(6, 16, figsize=(24, 9))
for i in range(48):
    a = ax[i // 16 * 2, i % 16]; a.scatter(*X[i].T, s=2, c="C3")
    b = ax[i // 16 * 2 + 1, i % 16]; b.scatter(*D["va_pts"][i].T, s=2, c="gray")
for a in ax.flat:
    a.set_xlim(-1.3, 1.3); a.set_ylim(-1.3, 1.3); a.set_aspect("equal"); a.axis("off")
ax[0, 0].set_title("generated (rows 1,3,5) vs real val (rows 2,4,6)", loc="left", fontsize=9)
fig.savefig(f"{run}/grid.png", dpi=55, bbox_inches="tight")
