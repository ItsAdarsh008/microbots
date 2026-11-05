"""How much of each target can stand at all: the even sample placed with every grip on, held passively, vs bond strength.
    python standability.py            # prints IoU after the hold per bond strength (val and held), per family
"""
import os, torch, train
from sim import Physics
KS = float(os.environ.get("KSTIFF", 200)); DT = float(os.environ.get("DT", 0.02))
dev = "cuda"
data = train.Data("data/world128.npz", dev)
for split in ("val", "held"):
    S = data.sets[split]
    Q, occ, fam = S["pts"], S["occ"], S["fam"]
    for fb in (8.0, 16.0, 32.0, 64.0, 1e4):
        for mu in (0.5,):
            ph = Physics(f_bond=fb, mu=mu, W=data.W, k_c=KS, k_b=KS, dt=DT)
            x, v = Q.clone(), torch.zeros_like(Q)
            one = torch.ones(Q.shape[:2], device=dev)
            for _ in range(int(10 / DT)):
                x, v = ph.step(x, v, None, one, one)
            i = train.iou(train.footprint(x, one, data.R, data.W), occ)
            pf = {data.families[f]: round(i[fam == f].mean().item(), 2) for f in fam.unique().tolist()}
            print(f"{split} f_bond {fb:7.0f}: IoU after hold {i.mean().item():.3f} (placed {train.iou(train.footprint(Q, one, data.R, data.W), occ).mean().item():.3f}) {pf}", flush=True)


def hex_fill(sdf_map, n, W, R, dev):
    """Reference arrangement: the n hexagonal-lattice points (spacing 1 = touching) deepest inside the shape,
    best of a few lattice offsets. sdf_map: (R, R) world units."""
    best = None
    for ox in (0.0, 0.25, 0.5, 0.75):
        for oy in (0.0, 0.29, 0.58):
            r = torch.arange(int(W / 0.866) + 2, device=dev).float()
            c = torch.arange(int(W) + 2, device=dev).float()
            rr, cc = torch.meshgrid(r, c, indexing="ij")
            px = cc + 0.5 * (rr % 2) + ox
            py = 0.5 + rr * 0.866 + oy
            P = torch.stack([px.reshape(-1), py.reshape(-1)], -1)
            P = P[(P[:, 0] > 0.5) & (P[:, 0] < W - 0.5) & (P[:, 1] < W - 0.5)]
            s = train.map_features(sdf_map[None, None], P[None], W)[0, :, 0]
            idx = torch.argsort(s)[:n]
            score = s[idx].mean().item()
            if best is None or score < best[0]:
                best = (score, P[idx])
    return best[1]


if __name__ == "__main__" and True:
    import numpy as np
    z = np.load("data/world128.npz")
    for split in ("val", "held"):
        S = data.sets[split]
        sdf = torch.tensor(z[f"{split}_sdf"].astype("float32"), device=dev)
        Hx = torch.stack([hex_fill(sdf[i], data.N, data.W, data.R, dev) for i in range(len(sdf))])
        occ, fam = S["occ"], S["fam"]
        one = torch.ones(Hx.shape[:2], device=dev)
        placed = train.iou(train.footprint(Hx, one, data.R, data.W), occ).mean().item()
        for fb in (16.0, 64.0, 1e4):
            ph = Physics(f_bond=fb, W=data.W)
            x, v = Hx.clone(), torch.zeros_like(Hx)
            for _ in range(int(10 / DT)):
                x, v = ph.step(x, v, None, one, one)
            i = train.iou(train.footprint(x, one, data.R, data.W), occ)
            pf = {data.families[f]: round(i[fam == f].mean().item(), 2) for f in fam.unique().tolist()}
            print(f"HEX {split} f_bond {fb:7.0f}: IoU after hold {i.mean().item():.3f} (placed {placed:.3f}) {pf}", flush=True)
