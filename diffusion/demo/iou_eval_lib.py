"""Does it LOOK right?  Mask IoU / precision / recall of the bots' footprint (pixels within 0.7 spacings of a bot) against
the target mask, for the no-learning rule, trained models, and learned-then-polished hybrids.   Run on the demo box."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch
from scipy.spatial import cKDTree
import evalm
D = np.load(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "shapes2.npz"))
AX = (np.arange(64) + 0.5) * 2 / 64 - 1
PIX = np.stack(np.meshgrid(AX, AX), -1).reshape(-1, 2)          # (x, y) per pixel, row-major in y
def foot(X, sp):
    d, _ = cKDTree(X).query(PIX); return (d < 0.7 * sp).reshape(64, 64)
def score(X, occ):
    sp = evalm.spacing(occ, len(X)); f = foot(X, sp); t = occ > 0.5
    inter = (f & t).sum(); return inter / max((f | t).sum(), 1), inter / max(f.sum(), 1), inter / max(t.sum(), 1)
