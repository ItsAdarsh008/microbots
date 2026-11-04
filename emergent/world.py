"""Floor-standing 2D target shapes for the emergent-physics swarm.

World: a square arena [0, W] x [0, W], floor at y = 0, gravity along -y. One bot = a disc of diameter 1.
Each shape is scaled so its area is `n_bots * area_per_bot`, stands on the floor, and is centred in x.
Stored per shape: occ (R, R) bool and sdf (R, R) in world units (negative inside); row index = y (row 0 at the floor);
pts (n_bots, 2): an even sample of the shape (farthest-point sampling + Lloyd), the "ceiling" arrangement.

    python world.py --out data/world128.npz --n-bots 128 --n-train 4000
"""
import argparse
import os
import numpy as np
from scipy import ndimage

FAMILIES = ["tower", "tee", "arch", "stairs", "blocks", "tree", "mushroom", "yshape"]
HELD_OUT = {"mushroom", "yshape"}
G = 128          # abstract raster for generation, unit square, row 0 = floor
W = 24.0         # arena size in bot diameters
R = 96           # world raster (cell = W / R = 0.25)

_a = (np.arange(G) + 0.5) / G
UX, UY = np.meshgrid(_a, _a)   # UX[i, j] = x of column j, UY[i, j] = y of row i (row 0 = bottom)


def rect(x0, x1, y0, y1):
    return (UX >= x0) & (UX <= x1) & (UY >= y0) & (UY <= y1)


def capsule(p, q, w):
    p, q = np.asarray(p, float), np.asarray(q, float)
    d = q - p
    L2 = max(float(d @ d), 1e-9)
    t = np.clip(((UX - p[0]) * d[0] + (UY - p[1]) * d[1]) / L2, 0, 1)
    return (UX - (p[0] + t * d[0])) ** 2 + (UY - (p[1] + t * d[1])) ** 2 <= w * w


def ellipse(cx, cy, rx, ry):
    return ((UX - cx) / rx) ** 2 + ((UY - cy) / ry) ** 2 <= 1


def f_tower(rng):
    w = rng.uniform(0.15, 0.6)
    h = rng.uniform(0.3, 0.95)
    return rect(0.5 - w / 2, 0.5 + w / 2, 0, h)


def f_tee(rng):
    """Column with a beam on top: T, L, Gamma (one-sided cantilever) shapes."""
    wc = rng.uniform(0.08, 0.22)
    hc = rng.uniform(0.3, 0.8)
    wb = rng.uniform(0.35, 0.9)
    t = rng.uniform(0.07, 0.18)
    xc = 0.5
    # beam start: anywhere such that the beam overlaps the column
    xb0 = rng.uniform(xc + wc / 2 - wb, xc - wc / 2)
    m = rect(xc - wc / 2, xc + wc / 2, 0, hc) | rect(xb0, xb0 + wb, hc, hc + t)
    return m


def f_arch(rng):
    wo = rng.uniform(0.45, 0.95)
    ho = rng.uniform(0.3, 0.8)
    wi = rng.uniform(0.3, 0.8) * wo
    hi = rng.uniform(0.4, 0.85) * ho
    m = rect(0.5 - wo / 2, 0.5 + wo / 2, 0, ho) & ~rect(0.5 - wi / 2, 0.5 + wi / 2, -1, hi)
    if rng.random() < 0.5:   # rounded opening
        m &= ~ellipse(0.5, hi * 0.6, wi / 2, hi * 0.5 + 0.02)
    return m


def f_stairs(rng):
    k = rng.integers(2, 6)
    w = rng.uniform(0.08, 0.2)
    h0 = rng.uniform(0.1, 0.25)
    dh = rng.uniform(0.08, 0.2)
    m = np.zeros((G, G), bool)
    x0 = 0.5 - k * w / 2
    for i in range(k):
        m |= rect(x0 + i * w, x0 + (i + 1) * w, 0, h0 + i * dh)
    return m[:, ::-1] if rng.random() < 0.5 else m


def f_blocks(rng):
    """Random stack of rectangles; later blocks sit on earlier ones and may overhang."""
    blocks = []
    w = rng.uniform(0.15, 0.5)
    h = rng.uniform(0.1, 0.3)
    blocks.append((0.5 - w / 2, 0.5 + w / 2, 0.0, h))
    for _ in range(rng.integers(2, 5)):
        b = blocks[rng.integers(len(blocks))]
        w = rng.uniform(0.1, 0.5)
        h = rng.uniform(0.06, 0.25)
        x0 = rng.uniform(b[0] - w * 0.8, b[1] - w * 0.2)
        blocks.append((x0, x0 + w, b[3], b[3] + h))
    m = np.zeros((G, G), bool)
    for b in blocks:
        m |= rect(*b)
    return m


def f_tree(rng):
    wt = rng.uniform(0.03, 0.07)
    top = rng.uniform(0.5, 0.9)
    m = capsule((0.5, 0.0), (0.5, top), wt) | rect(0.5 - wt, 0.5 + wt, 0, 0.05)
    for _ in range(rng.integers(2, 5)):
        y = rng.uniform(0.25, top)
        ang = rng.uniform(0.15, 0.85) * np.pi * (1 if rng.random() < 0.5 else -1) / 2
        L = rng.uniform(0.15, 0.4)
        q = (0.5 + L * np.sin(ang), y + L * np.cos(ang) * rng.uniform(0.2, 1.0))
        m |= capsule((0.5, y), q, wt * rng.uniform(0.7, 1.0))
    return m


def f_mushroom(rng):
    ws = rng.uniform(0.06, 0.18)
    hs = rng.uniform(0.25, 0.6)
    rx = rng.uniform(0.2, 0.45)
    ry = rng.uniform(0.08, 0.25)
    m = rect(0.5 - ws / 2, 0.5 + ws / 2, 0, hs) | (ellipse(0.5, hs, rx, ry) & (UY >= hs - 0.02))
    return m


def f_yshape(rng):
    w = rng.uniform(0.03, 0.08)
    hs = rng.uniform(0.2, 0.5)
    L = rng.uniform(0.2, 0.45)
    a = rng.uniform(0.3, 0.9)
    m = rect(0.5 - w, 0.5 + w, 0, hs) | capsule((0.5, hs), (0.5 - L * np.sin(a), hs + L * np.cos(a)), w) \
        | capsule((0.5, hs), (0.5 + L * np.sin(a), hs + L * np.cos(a)), w)
    return m


GEN = {f: globals()[f"f_{f}"] for f in FAMILIES}


def place(m, n_bots, area_per_bot=1.0):
    """Scale a unit-square mask to the world so its area is n_bots*area_per_bot, standing on the floor, centred.
    Returns the (R, R) world occupancy or None if it does not fit / is not one connected piece on the floor."""
    ys, xs = np.nonzero(m)
    if len(xs) < 20 or ys.min() > 1:
        return None
    m = m[: ys.max() + 1, xs.min(): xs.max() + 1]
    s = np.sqrt(n_bots * area_per_bot / m.sum())   # world units per abstract pixel
    bw, bh = m.shape[1] * s, m.shape[0] * s
    if bw > W - 2 or bh > W - 2 or min(bw, bh) < 1.5:
        return None
    c = (np.arange(R) + 0.5) * (W / R)
    cx, cy = np.meshgrid(c, c)
    x0 = W / 2 - bw / 2
    px = np.floor((cx - x0) / s).astype(int)
    py = np.floor(cy / s).astype(int)
    ok = (px >= 0) & (px < m.shape[1]) & (py >= 0) & (py < m.shape[0])
    occ = np.zeros((R, R), bool)
    occ[ok] = m[py[ok], px[ok]]
    lab, k = ndimage.label(occ)
    if k != 1 or not occ[0].any():
        return None
    area = occ.sum() * (W / R) ** 2
    if abs(area / (n_bots * area_per_bot) - 1) > 0.15:
        return None
    return occ


def sdf_of(occ):
    cell = W / R
    return ((ndimage.distance_transform_edt(~occ) - ndimage.distance_transform_edt(occ)) * cell).astype(np.float32)


def even_sample(occ, n, rng, lloyd=8):
    cell = W / R
    iy, ix = np.nonzero(occ)
    sub = 3
    off = (np.arange(sub) + 0.5) / sub
    oy, ox = np.meshgrid(off, off)
    cand = np.stack([(ix[:, None] + ox.ravel()[None]) * cell, (iy[:, None] + oy.ravel()[None]) * cell], -1).reshape(-1, 2)
    idx = [rng.integers(len(cand))]
    d = np.linalg.norm(cand - cand[idx[0]], axis=1)
    for _ in range(n - 1):
        j = int(d.argmax())
        idx.append(j)
        d = np.minimum(d, np.linalg.norm(cand - cand[j], axis=1))
    p = cand[idx].copy()
    for _ in range(lloyd):
        a = np.argmin(((cand[:, None] - p[None]) ** 2).sum(-1), 1)
        for k in range(n):
            sel = cand[a == k]
            if len(sel):
                p[k] = sel.mean(0)
    return p.astype(np.float32)


def make(fam, n, n_bots, seed):
    rng = np.random.default_rng(seed)
    occs, sdfs, pts = [], [], []
    tries = 0
    while len(occs) < n:
        tries += 1
        m = GEN[fam](rng)
        if rng.random() < 0.5:
            m = m[:, ::-1]
        occ = place(m, n_bots)
        if occ is None:
            continue
        occs.append(occ)
        sdfs.append(sdf_of(occ))
        pts.append(even_sample(occ, n_bots, rng))
    return np.array(occs), np.array(sdfs), np.array(pts), tries


def build(out, n_bots, n_train, n_val, seed=0):
    train_f = [f for f in FAMILIES if f not in HELD_OUT]
    parts = {}
    for si, (split, fams, per) in enumerate([("train", train_f, n_train // len(train_f)),
                                             ("val", train_f, n_val // len(train_f)),
                                             ("held", sorted(HELD_OUT), n_val // len(HELD_OUT))]):
        O, S, P, F = [], [], [], []
        for k, f in enumerate(fams):
            o, s, p, tries = make(f, per, n_bots, seed=seed * 10000 + si * 100 + FAMILIES.index(f))
            print(f"{split:5s} {f:9s} {per} shapes ({tries} tries)", flush=True)
            O.append(o); S.append(s); P.append(p); F += [FAMILIES.index(f)] * per
        parts[f"{split}_occ"] = np.concatenate(O)
        parts[f"{split}_sdf"] = np.concatenate(S).astype(np.float16)
        parts[f"{split}_pts"] = np.concatenate(P)
        parts[f"{split}_fam"] = np.array(F, np.int16)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    tmp = out + f".tmp{os.getpid()}.npz"
    np.savez_compressed(tmp, families=np.array(FAMILIES), W=W, R=R, n_bots=n_bots, **parts)
    os.replace(tmp, out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/world128.npz")
    ap.add_argument("--n-bots", type=int, default=128)
    ap.add_argument("--n-train", type=int, default=3000)
    ap.add_argument("--n-val", type=int, default=192)
    a = ap.parse_args()
    build(a.out, a.n_bots, a.n_train, a.n_val)
