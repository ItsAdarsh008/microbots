"""Pictures of one swarm model at 256 / 1024 / 2048 bots (sensing range kept in spacings). Run on the demo box in diffusion/."""
import sys, os, json, urllib.request
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
def post(d):
    r = urllib.request.Request("http://localhost:8000/api/run", data=json.dumps(d).encode(), method="POST"); return json.load(urllib.request.urlopen(r, timeout=300))
shapes = ["star", "anchor", "snowflake", "letter K (thin)", "letter B (thick)", "letter S (thin)"]
Ns = (256, 1024, 2048)
fig, ax = plt.subplots(len(Ns), len(shapes), figsize=(2.1 * len(shapes), 2.1 * len(Ns)))
for c, s in enumerate(shapes):
    for r, n in enumerate(Ns):
        j = post(dict(model=sys.argv[2], shape=s, start="pile_c", k=16, n=n, keep_spacings=True)); X = np.array(j["frames"][-1]); occ = np.array(j["occ"])
        a = ax[r, c]; a.imshow(occ > 0.5, extent=(-1, 1, -1, 1), origin="lower", cmap="Greys", alpha=0.25, vmin=0, vmax=1.4)
        a.scatter(X[:, 0], X[:, 1], s=max(0.5, 3.2 * (256 / n) ** 0.5), c="#d1432b"); a.set_xlim(-1.3, 1.3); a.set_ylim(-1.3, 1.3); a.set_aspect("equal"); a.set_xticks([]); a.set_yticks([])
        a.set_title(f"{s}  N={n}  cov {j['cover']:.2f}", fontsize=6)
fig.tight_layout(); fig.savefig(sys.argv[1], dpi=75); print("saved")
