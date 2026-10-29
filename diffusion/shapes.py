"""Procedural 2D shape generator and dataset builder for the diffusion-swarm study.

Every shape lives in [-1, 1]^2, is one connected blob (possibly with holes), and is stored as
  pts   (N, 2)   N well-spaced "particles" inside the shape (farthest-point sampling)
  occ   (R, R)   occupancy mask, R = 64
  sdf   (R, R)   signed distance in world units (negative inside)
  fam   int      family id (see FAMILIES)
Two families (star, annulus) are held out of training so we can test unseen kinds of shape.

    python shapes.py --out data/shapes.npz --n-train 10000
"""
import argparse
import os
import numpy as np
from scipy import ndimage

FAMILIES = ["blob", "discs", "strokes", "poly", "holey", "csg", "star", "annulus"]
HELD_OUT = {"star", "annulus"}
G = 128   # raster used for generation
R = 64    # raster stored / used as the map
N_PTS = 256

_ax = (np.arange(G) + 0.5) / G * 2 - 1
X, Y = np.meshgrid(_ax, _ax)  # X[i, j] = x of column j, Y[i, j] = y of row i


def disc(cx, cy, r):
    return (X - cx) ** 2 + (Y - cy) ** 2 <= r * r


def capsule(p, q, w):
    p, q = np.asarray(p), np.asarray(q)
    d = q - p
    L2 = max(float(d @ d), 1e-9)
    t = np.clip(((X - p[0]) * d[0] + (Y - p[1]) * d[1]) / L2, 0, 1)
    return (X - (p[0] + t * d[0])) ** 2 + (Y - (p[1] + t * d[1])) ** 2 <= w * w


def polar(rfun):
    th = np.arctan2(Y, X)
    return np.hypot(X, Y) <= rfun(th)


def f_blob(rng):
    ks = np.arange(2, 7)
    a = rng.uniform(0, 0.5, ks.size) / ks ** rng.uniform(0.5, 1.2)
    ph = rng.uniform(0, 2 * np.pi, ks.size)
    r0 = rng.uniform(0.45, 0.8)
    return polar(lambda th: r0 * (1 + sum(a[i] * np.cos(ks[i] * th + ph[i]) for i in range(ks.size))))


def f_discs(rng):
    m = np.zeros((G, G), bool)
    c = rng.uniform(-0.3, 0.3, 2)
    for _ in range(rng.integers(2, 6)):
        c = np.clip(c + rng.uniform(-0.4, 0.4, 2), -0.6, 0.6)
        m |= disc(c[0], c[1], rng.uniform(0.15, 0.4))
    return m


def f_strokes(rng):
    m = np.zeros((G, G), bool)
    p = rng.uniform(-0.6, 0.6, 2)
    for _ in range(rng.integers(2, 6)):
        q = np.clip(p + rng.uniform(-0.8, 0.8, 2), -0.8, 0.8)
        m |= capsule(p, q, rng.uniform(0.07, 0.18))
        p = q if rng.random() < 0.6 else rng.uniform(-0.6, 0.6, 2)
    return m


def f_poly(rng):
    k = rng.integers(3, 9)
    ang = np.sort(rng.uniform(0, 2 * np.pi, k))
    rad = rng.uniform(0.35, 0.85, k)
    px, py = rad * np.cos(ang), rad * np.sin(ang)
    m = np.ones((G, G), bool)
    # intersection of half-planes of the convex hull edges
    from scipy.spatial import ConvexHull
    try:
        h = ConvexHull(np.stack([px, py], 1))
    except Exception:
        return f_blob(rng)
    for eq in h.equations:
        m &= (eq[0] * X + eq[1] * Y + eq[2]) <= 0
    return m


def f_holey(rng):
    m = f_blob(rng) if rng.random() < 0.5 else f_discs(rng)
    for _ in range(rng.integers(1, 4)):
        ys, xs = np.nonzero(m)
        if len(ys) == 0:
            break
        i = rng.integers(len(ys))
        m = m & ~disc(_ax[xs[i]], _ax[ys[i]], rng.uniform(0.08, 0.2))
    return m


def f_star(rng):
    k = rng.integers(4, 9)
    ratio = rng.uniform(0.3, 0.6)
    R0 = rng.uniform(0.6, 0.9)

    def rf(th):
        s = (th * k / (2 * np.pi)) % 1.0
        tri = np.abs(2 * s - 1)           # 1 at valleys, 0 at tips
        return R0 * (ratio + (1 - ratio) * (1 - tri))
    return polar(rf)


def f_annulus(rng):
    ro = rng.uniform(0.55, 0.9)
    ri = ro * rng.uniform(0.35, 0.75)
    m = disc(0, 0, ro) & ~disc(0, 0, ri)
    if rng.random() < 0.5:   # a gap makes it a "C"
        a0, w = rng.uniform(0, 2 * np.pi), rng.uniform(0.4, 1.0)
        d = (np.arctan2(Y, X) - a0 + np.pi) % (2 * np.pi) - np.pi
        m &= ~(np.abs(d) < w / 2)
    return m


def f_csg(rng):
    gens = [f_blob, f_discs, f_strokes, f_poly]
    a, b = gens[rng.integers(4)](rng), gens[rng.integers(4)](rng)
    sh = rng.integers(-25, 25, 2)
    b = np.roll(b, tuple(sh), (0, 1))
    return (a | b) if rng.random() < 0.65 else (a & ~b)


GEN = {"blob": f_blob, "discs": f_discs, "strokes": f_strokes, "poly": f_poly,
       "holey": f_holey, "csg": f_csg, "star": f_star, "annulus": f_annulus}


def normalise(m, rng=None, rad=None):
    """Keep the biggest piece, centre on the centroid, scale so the farthest pixel sits at `rad`."""
    lab, n = ndimage.label(m)
    if n == 0:
        return None
    sizes = ndimage.sum(m, lab, range(1, n + 1))
    m = lab == (1 + int(np.argmax(sizes)))
    frac = m.mean()
    if frac < 0.02 or frac > 0.7:
        return None
    ys, xs = np.nonzero(m)
    px, py = _ax[xs], _ax[ys]
    cx, cy = px.mean(), py.mean()
    r = np.hypot(px - cx, py - cy).max()
    if rad is None:
        rad = rng.uniform(0.7, 0.95)
    s = rad / r
    # resample the mask onto the new frame (nearest)
    u = (X / s + cx + 1) / 2 * G - 0.5
    v = (Y / s + cy + 1) / 2 * G - 0.5
    out = ndimage.map_coordinates(m.astype(np.float32), [v, u], order=1, mode="constant") > 0.5
    lab, n = ndimage.label(out)
    if n == 0:
        return None
    sizes = ndimage.sum(out, lab, range(1, n + 1))
    out = lab == (1 + int(np.argmax(sizes)))
    if out.mean() < 0.02:
        return None
    return out


def signed_distance(m):
    """World-unit signed distance on the G grid (negative inside)."""
    px = 2.0 / G
    return (ndimage.distance_transform_edt(~m) - ndimage.distance_transform_edt(m)) * px


def fps_points(m, n, rng):
    ys, xs = np.nonzero(m)
    cand = np.stack([_ax[xs], _ax[ys]], 1) + rng.uniform(-1, 1, (len(xs), 2)) / G
    if len(cand) > 12 * n:
        cand = cand[rng.choice(len(cand), 12 * n, replace=False)]
    sel = [int(rng.integers(len(cand)))]
    d = ((cand - cand[sel[0]]) ** 2).sum(1)
    for _ in range(n - 1):
        j = int(np.argmax(d))
        sel.append(j)
        d = np.minimum(d, ((cand - cand[j]) ** 2).sum(1))
    return cand[sel].astype(np.float32)


def pack(m, rng):
    sdf = signed_distance(m)
    pts = fps_points(m, N_PTS, rng)
    k = G // R
    occ = m.reshape(R, k, R, k).mean((1, 3)).astype(np.float32)
    sdf_r = sdf[k // 2::k, k // 2::k][:R, :R].astype(np.float32)
    return pts, occ, sdf_r


def make_one(seed, families):
    rng = np.random.default_rng(seed)
    for _ in range(50):
        fam = families[rng.integers(len(families))]
        m = normalise(GEN[fam](rng), rng)
        if m is not None:
            pts, occ, sdf = pack(m, rng)
            return pts, occ, sdf, FAMILIES.index(fam)
    raise RuntimeError("generator failed")


def _job(args):
    return make_one(*args)


def build(n, families, seed0, pool):
    out = pool.map(_job, [(seed0 + i, families) for i in range(n)], chunksize=64)
    return (np.stack([o[0] for o in out]), np.stack([o[1] for o in out]),
            np.stack([o[2] for o in out]), np.array([o[3] for o in out], np.int8))


def load_emoji(folder):
    from PIL import Image
    names, P, O, S = [], [], [], []
    rng = np.random.default_rng(0)
    for f in sorted(os.listdir(folder)):
        if not f.endswith(".png"):
            continue
        im = Image.open(os.path.join(folder, f)).convert("RGBA").resize((G, G), Image.LANCZOS)
        m = np.asarray(im)[..., 3] > 127
        m = normalise(m, rad=0.9)
        if m is None:
            continue
        pts, occ, sdf = pack(m, rng)
        names.append(f[:-4]); P.append(pts); O.append(occ); S.append(sdf)
    return names, np.stack(P), np.stack(O), np.stack(S)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/shapes.npz")
    ap.add_argument("--n-train", type=int, default=10000)
    ap.add_argument("--n-val", type=int, default=512)
    ap.add_argument("--emoji", default="data/emoji_shapes")
    a = ap.parse_args()
    from multiprocessing import Pool
    train_fams = [f for f in FAMILIES if f not in HELD_OUT]
    with Pool(min(24, os.cpu_count())) as pool:
        tr = build(a.n_train, train_fams, 1_000_000, pool)
        va = build(a.n_val, train_fams, 2_000_000, pool)
        oo = build(a.n_val, sorted(HELD_OUT), 3_000_000, pool)
    en, ep, eo, es = load_emoji(a.emoji)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    np.savez_compressed(a.out + ".tmp.npz",
                        tr_pts=tr[0], tr_occ=tr[1], tr_sdf=tr[2], tr_fam=tr[3],
                        va_pts=va[0], va_occ=va[1], va_sdf=va[2], va_fam=va[3],
                        ood_pts=oo[0], ood_occ=oo[1], ood_sdf=oo[2], ood_fam=oo[3],
                        em_pts=ep, em_occ=eo, em_sdf=es, em_names=np.array(en),
                        families=np.array(FAMILIES))
    os.replace(a.out + ".tmp.npz", a.out)
    print("train", tr[0].shape, "val", va[0].shape, "ood", oo[0].shape, "emoji", en)
    # quick montage for eyeballing
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(4, 12, figsize=(24, 8))
    sets = [(tr, "train"), (tr, "train"), (oo, "OOD family"), ((ep, eo, es, None), "emoji")]
    for r, (S, nm) in enumerate(sets):
        for c in range(12):
            i = c + (12 if r == 1 else 0)
            ax[r, c].scatter(*S[0][i].T, s=2); ax[r, c].set_xlim(-1.05, 1.05); ax[r, c].set_ylim(-1.05, 1.05)
            ax[r, c].set_aspect("equal"); ax[r, c].axis("off")
        ax[r, 0].set_title(nm, loc="left", fontsize=9)
    fig.savefig(os.path.splitext(a.out)[0] + "_montage.png", dpi=70, bbox_inches="tight")
