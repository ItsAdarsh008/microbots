"""Render what the models actually produce, so it can be looked at (not only scored).
    python demo/render_grid.py out.png   (run on the demo box; imports the server's loader)"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import serve
rows = [("s2_cond_global", "noise", "global, scattered start"),
        ("s2_cond_local_r60_L4", "noise", "local r=0.6, scattered"),
        ("s2_cond_local_r25_L4", "noise", "local r=0.25, scattered"),
        ("s4_ref_local_r60_g8_tn15_dmg_pfc", "pile_c", "REFERENCE swarm (pile)"),
        ("s4_ref_global_g8_tn15_dmg_pfc", "pile_c", "reference, global"),
        ("s2pf_cond_global_pf", "pile_off", "global, offset pile")]
shapes = ["star", "ring", "anchor", "butterfly", "key", "snowflake", "fish", "wrench"]
K = int(sys.argv[2]) if len(sys.argv) > 2 else 16
fig, ax = plt.subplots(len(rows), len(shapes), figsize=(2.2 * len(shapes), 2.2 * len(rows)))
for r, (m, st, lab) in enumerate(rows):
    for c, sh in enumerate(shapes):
        j = serve.run(dict(model=m, shape=sh, start=st, k=K, seed=0))
        a = ax[r, c]; occ = np.array(j["occ"]); X = np.array(j["frames"][-1])
        a.imshow(occ > 0.5, extent=(-1, 1, -1, 1), origin="lower", cmap="Greys", alpha=0.25, vmin=0, vmax=1.4)
        a.scatter(X[:, 0], X[:, 1], s=4, c="#d1432b")
        a.set_xlim(-1.3, 1.3); a.set_ylim(-1.3, 1.3); a.set_aspect("equal"); a.set_xticks([]); a.set_yticks([])
        a.set_title(f"in {j['inside']:.2f} cov {j['cover']:.2f}", fontsize=7)
        if c == 0: a.set_ylabel(lab, fontsize=7)
        if r == 0: a.set_xlabel(sh, fontsize=7); a.xaxis.set_label_position("top")
fig.tight_layout(); fig.savefig(sys.argv[1], dpi=75)
print("saved")
