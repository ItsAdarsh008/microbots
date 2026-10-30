"""Do the bots hold together? Contact graph of the finished swarm: bots whose bodies touch (centre distance below c * body diameter).
Reports the largest connected piece and the mean number of contacts per bot (a rigid 2D frame needs about 4).
    python stage2_contacts.py runs/<name>"""
import json, sys
import numpy as np, torch
from scipy.sparse.csgraph import connected_components
from scipy.sparse import csr_matrix
sys.argv = [sys.argv[0], sys.argv[1]]
exec(open("stage2_eval.py").read().split("print(f\"{run}: no retraining")[0])
for K, vmax in ((32, 0.5), (64, 0.25)):
    X, path = go(K, vmax, 1.0)
    for c in (1.15, 1.5, 2.0, 2.35):
        frac, deg = [], []
        for i in range(n):
            d = np.sqrt(((X[i][:, None] - X[i][None]) ** 2).sum(-1)); s = sp[i, 0, 0].item()
            A = (d < c * 1.0 * s) & (d > 0); k, lab = connected_components(csr_matrix(A)); frac.append(np.bincount(lab).max() / 256); deg.append(A.sum(1).mean())
        print(f"{run}  rounds {K}, max step {vmax}, body 1.0 spacing, contact if centres within {c:.2f} bodies: largest connected piece {np.mean(frac):.2f}, mean contacts per bot {np.mean(deg):.1f}")
