"""Do generated shapes have the real set's spread of 'branchiness'?  solidity = area / convex-hull area of the
dilated particle raster (1 = convex blob, small = arms, C-shapes, holes). Compares generated vs real val shapes.
    python solidity.py runs/<name> [runs/<name2> ...]   (unconditional runs)"""
import json, sys
import numpy as np, torch
from scipy.spatial import ConvexHull
from scipy.stats import wasserstein_distance
from model import Denoiser
import evalm
D = np.load("data/shapes.npz")


def solidity(X):
    m = evalm.raster(X)
    ys, xs = np.nonzero(m)
    if len(xs) < 5:
        return 0.0
    try:
        h = ConvexHull(np.stack([xs, ys], 1))
    except Exception:
        return 1.0
    return float(m.sum() / h.volume)


real = np.array([solidity(x) for x in D["va_pts"]])
print(f"real val: solidity mean {real.mean():.3f}  10th pct {np.percentile(real,10):.3f}  frac<0.8 {np.mean(real<0.8):.2f}")
for run in sys.argv[1:]:
    cfg = json.load(open(f"{run}/result.json"))["config"]
    m = Denoiser(cfg["d"], cfg["heads"], cfg["layers"], cfg["radius"], cfg["cond"], cfg["time"]).cuda().eval()
    m.load_state_dict(torch.load(f"{run}/ckpt.pt")["ema"])
    g = torch.Generator().manual_seed(21)
    x0 = (torch.randn((256, 256, 2), generator=g) * cfg["sigma"]).cuda()
    X = torch.cat([evalm.sample(m, x0[i:i + 64], None, 32) for i in range(0, 256, 64)]).cpu().numpy()
    s = np.array([solidity(x) for x in X])
    print(f"{run:40s} mean {s.mean():.3f}  10th pct {np.percentile(s,10):.3f}  frac<0.8 {np.mean(s<0.8):.2f}  W1 vs real {wasserstein_distance(s, real):.3f}")
