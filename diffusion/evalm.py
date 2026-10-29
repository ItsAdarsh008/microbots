"""Evaluation for particles-forming-shapes. Everything is in units of the shape's own particle
spacing sp = sqrt(area / N), so a number means the same for a fat blob and a thin stroke.

Conditional metrics (target known, compared with the reference point set):
  w2      optimal-matching RMS distance to the reference points, in sp (0 = perfect)
  inside  fraction of particles within 0.5 sp of the shape
  cover   fraction of shape pixels with a particle within 1.0 sp   (the reference scores ~1 by construction)
  coll    fraction of particles whose nearest neighbour is closer than 0.35 sp
  nn      mean nearest-neighbour distance in sp (reference ~ 0.9)
Every table has: noise (K=0, chance), 'project' (one-shot nearest-point projection),
'classical' (projection + local repulsion, 200 steps; no learning), 'ref' (the data itself) and the model at each K.
skill = 1 - w2_model / w2_noise.
Unconditional metrics compare descriptor distributions of generated vs real sets (see descr()).
"""
import json, os
import numpy as np
import torch
from scipy import ndimage
from scipy.optimize import linear_sum_assignment
from scipy.stats import wasserstein_distance
from model import make_maps as _make_maps, read_map

COARSE = 0


def make_maps(occ, sdf):
    return _make_maps(occ, sdf, COARSE)

R = 64
PX = 2.0 / R


@torch.no_grad()
def sample(model, x0, maps, K, traj=False):
    x, t = x0.clone(), torch.zeros(x0.shape[0], device=x0.device)
    tr = [x.cpu()] if traj else None
    for i in range(K):
        t = torch.full((x.shape[0],), i / K, device=x.device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            v = model(x, t, maps).float()
        x = x + v / K
        if traj:
            tr.append(x.cpu())
    return (x, tr) if traj else x


def spacing(occ, n=256):
    return float(np.sqrt(occ.sum() * PX ** 2 / n))


def sdf_at(x, sdf):
    """sdf (R,R) numpy -> value at x (N,2) numpy, with distance-to-box added outside."""
    m = torch.tensor(sdf, dtype=torch.float32)[None, None]
    c = np.clip(x, -1, 1)
    g = torch.tensor(c, dtype=torch.float32)[None, None]
    v = torch.nn.functional.grid_sample(m, g, align_corners=False, padding_mode="border")[0, 0, 0].numpy()
    return v + np.linalg.norm(x - c, axis=1)


def metrics(X, ref, occ, sdf):
    sp = spacing(occ, len(X))
    C = ((X[:, None] - ref[None]) ** 2).sum(-1)
    r, c = linear_sum_assignment(C)
    w2 = float(np.sqrt(C[r, c].mean()) / sp)
    inside = float((sdf_at(X, sdf) < 0.5 * sp).mean())
    gx = (np.arange(R) + 0.5) * PX - 1
    ys, xs = np.nonzero(occ > 0.5)
    pix = np.stack([gx[xs], gx[ys]], 1)
    dmin = np.sqrt(((pix[:, None] - X[None]) ** 2).sum(-1).min(1))
    cover = float((dmin < 1.0 * sp).mean())
    D = np.sqrt(((X[:, None] - X[None]) ** 2).sum(-1)); np.fill_diagonal(D, 1e9)
    nn = D.min(1)
    return dict(w2=w2, inside=inside, cover=cover, coll=float((nn < 0.35 * sp).mean()), nn=float(nn.mean() / sp))


def project(x, maps, iters=1):
    for _ in range(iters):
        f = read_map(x, maps)
        sd, g = f[..., :1], f[..., 1:3]
        g = g / g.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        x = x - torch.relu(sd) * g
    return x


def classical(x, maps, sp, steps=200):
    """No learning: slide to the shape, then spread out by local repulsion (radius 2 sp)."""
    sp = sp[:, None, None]
    x = project(x, maps, 3)
    for _ in range(steps):
        d = x[:, None] - x[:, :, None]                      # (B,N,N,2) x_j - x_i
        dist = d.norm(dim=-1, keepdim=True).clamp_min(1e-4)
        w = torch.relu(1 - dist / (2 * sp[..., None]))
        w = w * (dist > 1e-3)
        f = -(w * d / dist).sum(2)
        x = x + 0.12 * sp * f / (w.sum(2) + 1.0)
        x = project(x, maps, 2)
    return x


def descr(X):
    """Descriptors of an unordered particle set, computed identically for real and generated sets."""
    nnd = np.sqrt(((X[:, None] - X[None]) ** 2).sum(-1) + np.eye(len(X)) * 1e9).min(1)
    r = 2.0 * np.median(nnd)
    g = 128
    ax = (np.arange(g) + 0.5) / g * 2 - 1
    H, _, _ = np.histogram2d(X[:, 1], X[:, 0], bins=g, range=[[-1.5, 1.5], [-1.5, 1.5]])
    ax = (np.arange(g) + 0.5) / g * 3 - 1.5
    dist = ndimage.distance_transform_edt(H == 0) * (3.0 / g)
    m = dist <= r
    _, ncomp = ndimage.label(m)
    lab, nbg = ndimage.label(~m)
    border = set(lab[0].tolist()) | set(lab[-1].tolist()) | set(lab[:, 0].tolist()) | set(lab[:, -1].tolist())
    holes = len([i for i in range(1, nbg + 1) if i not in border])
    c = np.cov((X - X.mean(0)).T)
    ev = np.sort(np.linalg.eigvalsh(c))
    return dict(ncomp=ncomp, holes=holes, rg=float(np.sqrt(ev.sum())), aniso=float(np.sqrt(ev[0] / ev[1])),
                nn=float(np.median(nnd)), cx=float(np.abs(X.mean(0)).max()), extent=float(np.abs(X).max()))


def raster(X, nnd_scale=2.0):
    """Occupancy raster (R*R bool) of a set on the standard [-1,1]^2 grid, dilated by 2x median NN distance."""
    nnd = np.sqrt(((X[:, None] - X[None]) ** 2).sum(-1) + np.eye(len(X)) * 1e9).min(1)
    r = nnd_scale * np.median(nnd)
    H, _, _ = np.histogram2d(X[:, 1], X[:, 0], bins=R, range=[[-1, 1], [-1, 1]])
    return ndimage.distance_transform_edt(H == 0) * PX <= r


def agg(rows):
    return {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}


def evaluate(model, cfg, D, out, log):
    global COARSE
    COARSE = cfg.get("coarse", 0)
    dev = "cuda"
    cond = cfg["cond"] == "map"
    rng = np.random.default_rng(1234)
    res = {"sets": {}}
    sets = {}
    nev = cfg["n_eval"]
    sets["val"] = [(D["va_pts"][i], D["va_occ"][i], D["va_sdf"][i]) for i in range(nev)]
    sets["ood_family"] = [(D["ood_pts"][i], D["ood_occ"][i], D["ood_sdf"][i]) for i in range(nev)]
    em = [(D["em_pts"][i], D["em_occ"][i], D["em_sdf"][i]) for i in range(len(D["em_pts"]))]
    sets["emoji"] = em * 8
    figdata = {}
    for sname, S in sets.items():
        ref = np.stack([s[0] for s in S]); occ = np.stack([s[1] for s in S]); sdf = np.stack([s[2] for s in S])
        g = torch.Generator().manual_seed(7)
        x0 = (torch.randn(ref.shape, generator=g) * cfg["sigma"]).to(dev)
        maps = make_maps(torch.tensor(occ, device=dev), torch.tensor(sdf, device=dev)) if cond else None
        sp = torch.tensor([spacing(o) for o in occ], device=dev)
        table = {}

        def score(X, refs=ref, occs=occ, sdfs=sdf):
            X = X.cpu().numpy() if torch.is_tensor(X) else X
            return agg([metrics(X[i], refs[i], occs[i], sdfs[i]) for i in range(len(X))])
        table["ref"] = score(torch.tensor(ref))   # the data itself: w2 is 0 by construction, the rest are the reference values
        table["noise"] = score(x0)
        if cond:
            table["project"] = score(project(x0, maps, 1))
            table["classical"] = score(classical(x0, maps, sp))
        for K in cfg["eval_steps"]:
            parts = [sample(model, x0[i:i + 64], None if maps is None else maps[i:i + 64], K) for i in range(0, len(x0), 64)]
            X = torch.cat(parts)
            table[f"K{K}"] = score(X)
            if K == max(cfg["eval_steps"]):
                figdata[sname] = X.cpu().numpy()
        wn = table["noise"]["w2"]
        for k, v in table.items():
            v["skill"] = 1 - v["w2"] / wn
        res["sets"][sname] = table
        log(sname, json.dumps({k: {m: round(x, 3) for m, x in v.items()} for k, v in table.items() if k in ("noise", "classical", f"K{cfg['eval_steps'][-1]}")}))

    # control: does the model follow the map it is given?  feed the map of the NEXT shape.
    if cond:
        ref = np.stack([s[0] for s in sets["val"]]); occ = np.stack([s[1] for s in sets["val"]]); sdf = np.stack([s[2] for s in sets["val"]])
        sh = np.roll(np.arange(len(ref)), 1)
        g = torch.Generator().manual_seed(7)
        x0 = (torch.randn(ref.shape, generator=g) * cfg["sigma"]).to(dev)
        maps = make_maps(torch.tensor(occ[sh], device=dev), torch.tensor(sdf[sh], device=dev))
        K = max(cfg["eval_steps"])
        X = torch.cat([sample(model, x0[i:i + 64], maps[i:i + 64], K) for i in range(0, len(x0), 64)]).cpu().numpy()
        fed = agg([metrics(X[i], ref[sh][i], occ[sh][i], sdf[sh][i]) for i in range(len(X))])
        asked = agg([metrics(X[i], ref[i], occ[i], sdf[i]) for i in range(len(X))])
        res["control_wrong_map"] = dict(w2_vs_fed_map=fed["w2"], w2_vs_other_shape=asked["w2"])
        log("control wrong-map: w2 vs map fed", round(fed["w2"], 3), "vs other shape", round(asked["w2"], 3))

    # shape-level statistics: generated vs real descriptor distributions (works for every model)
    g = torch.Generator().manual_seed(11)
    n = 256
    x0 = (torch.randn((n, 256, 2), generator=g) * cfg["sigma"]).to(dev)
    if cond:
        i0 = np.arange(n) % len(D["va_pts"])
        maps = make_maps(torch.tensor(D["va_occ"][i0], device=dev), torch.tensor(D["va_sdf"][i0], device=dev))
        real = D["va_pts"][i0]
    else:
        maps, real = None, D["va_pts"]
    K = max(cfg["eval_steps"])
    Xg = torch.cat([sample(model, x0[i:i + 64], None if maps is None else maps[i:i + 64], K) for i in range(0, n, 64)]).cpu().numpy()
    dg = [descr(x) for x in Xg]; dr = [descr(x) for x in real]
    stat = {}
    for k in dg[0]:
        stat[k] = dict(gen=float(np.mean([d[k] for d in dg])), real=float(np.mean([d[k] for d in dr])),
                       w1=float(wasserstein_distance([d[k] for d in dg], [d[k] for d in dr])))
    stat["connected_frac"] = dict(gen=float(np.mean([d["ncomp"] == 1 for d in dg])), real=float(np.mean([d["ncomp"] == 1 for d in dr])))
    # memorisation: best IoU with a training shape (10k) vs the same for real val shapes
    mt = torch.tensor(D["tr_occ"].reshape(len(D["tr_occ"]), -1) > 0.5, device=dev).float()

    def best_iou(Xs):
        M = torch.tensor(np.stack([raster(x) for x in Xs]).reshape(len(Xs), -1), device=dev).float()
        inter = M @ mt.T
        iou = inter / (M.sum(1, keepdim=True) + mt.sum(1)[None] - inter)
        return float(iou.max(1).values.mean())
    stat["best_iou_train"] = dict(gen=best_iou(Xg), real=best_iou(real))
    res["stats"] = stat
    log("stats connected", stat["connected_frac"], "best_iou_train", stat["best_iou_train"])

    # pictures
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cols = [("val", i) for i in range(6)] + [("ood_family", i) for i in (0, 1, 2, 3, 4, 5)] + [("emoji", i) for i in range(12)]
    cols = cols[:18]
    fig, ax = plt.subplots(2, len(cols), figsize=(2 * len(cols), 4.2))
    for c, (sn, i) in enumerate(cols):
        ref = sets[sn][i][0]
        ax[0, c].scatter(*ref.T, s=3, c="gray"); ax[1, c].scatter(*figdata[sn][i].T, s=3, c="C3")
        ax[0, c].set_title(sn, fontsize=7)
        for r in range(2):
            ax[r, c].set_xlim(-1.3, 1.3); ax[r, c].set_ylim(-1.3, 1.3); ax[r, c].set_aspect("equal"); ax[r, c].axis("off")
    fig.savefig(os.path.join(out, "samples.png"), dpi=60, bbox_inches="tight"); plt.close(fig)

    # trajectory gif for four emoji
    from PIL import Image
    ids = [i for i in range(len(D["em_pts"])) if D["em_names"][i] in ("star", "butterfly", "key", "snowflake")][:4]
    g = torch.Generator().manual_seed(3)
    x0 = (torch.randn((len(ids), 256, 2), generator=g) * cfg["sigma"]).to(dev)
    mp = make_maps(torch.tensor(D["em_occ"][ids], device=dev), torch.tensor(D["em_sdf"][ids], device=dev)) if cond else None
    _, tr = sample(model, x0, mp, 64, traj=True)
    frames = []
    for f in range(0, len(tr), 2):
        fig, ax = plt.subplots(1, len(ids), figsize=(2.4 * len(ids), 2.4))
        for c in range(len(ids)):
            ax[c].scatter(*tr[f][c].numpy().T, s=2, c="C3"); ax[c].set_xlim(-1.6, 1.6); ax[c].set_ylim(-1.6, 1.6)
            ax[c].set_aspect("equal"); ax[c].axis("off")
        fig.canvas.draw(); frames.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[..., :3])); plt.close(fig)
    frames[0].save(os.path.join(out, "traj.gif"), save_all=True, append_images=frames[1:] + [frames[-1]] * 6, duration=80, loop=0)
    return res
