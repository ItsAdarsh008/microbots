"""Ceiling for the footprint IoU: an independent, perfectly even sample of the same shape."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import iou_eval_lib as L
D = L.D
sets = {"emoji": [(D["em_pts"][i], D["em_occ"][i]) for i in range(12)], "held-out family": [(D["ood_pts"][i], D["ood_occ"][i]) for i in range(96)],
        "familiar": [(D["va_pts"][i], D["va_occ"][i]) for i in range(96)]}
for n, S in sets.items():
    a = np.mean([L.score(p, o) for p, o in S], 0); print(f"{n:16s} ceiling (the data's own evenly spaced sample): IoU {a[0]:.2f} prec {a[1]:.2f} rec {a[2]:.2f}")
