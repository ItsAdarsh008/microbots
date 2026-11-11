"""Inference engine for the dashboard: lazy weights from the Hub, one sampler for every model, the no-learning rule,
the even-sample ceiling, and footprint IoU / precision / recall.  CPU only (models are 2.5-3.6M parameters)."""
import json, os, sys, threading
import numpy as np
import torch
from scipy.spatial import cKDTree
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
import evalm, shapes
from model import build, make_maps

REPO = "KuanKuanKuan/microbots-flow-swarm"
CACHE = os.path.join(ROOT, "weights_cache")
R = 64
AX = (np.arange(R) + 0.5) * 2 / R - 1
PIX = np.stack(np.meshgrid(AX, AX), -1).reshape(-1, 2)           # (x, y) per pixel, row-major in y
torch.set_num_threads(max(1, (os.cpu_count() or 4) - 2))
CATALOG = {r["run"]: r for r in json.load(open(os.path.join(HERE, "catalog.json")))}
_models, _lock = {}, threading.Lock()


def get_model(name):
    with _lock:
        if name not in _models:
            p = os.path.join(CACHE, "weights", name + ".pt")
            if not os.path.exists(p):
                from huggingface_hub import hf_hub_download
                hf_hub_download(REPO, f"weights/{name}.pt", local_dir=CACHE)
            blob = torch.load(p, map_location="cpu")
            cfg = blob["config"]; m = build(cfg); m.load_state_dict(blob["ema"]); m.eval()
            _models[name] = (m, cfg)
        return _models[name]


def shape_library():
    """name -> (occ, sdf), all (64,64) float32, row 0 = bottom."""
    lib = {}
    for fn in ("shapes2", "shapes"):
        p = os.path.join(ROOT, "data", fn + ".npz")
        if os.path.exists(p):
            D = np.load(p)
            for i, nm in enumerate(D["em_names"]):
                lib.setdefault(f"emoji {nm}", (D["em_occ"][i][::-1].copy(), D["em_sdf"][i][::-1].copy()))   # stored upside-down
            for i in range(len(D["ood_occ"])):
                lib.setdefault(f"held-out #{i}", (D["ood_occ"][i], D["ood_sdf"][i]))
            break
    p = os.path.join(ROOT, "data", "letters.npz")
    if os.path.exists(p):
        L = np.load(p)
        for i, nm in enumerate(L["names"]):
            ch, tag = str(nm).split("-")
            lib[f"letter {ch} ({'thick' if tag == 'bold' else 'thin'})"] = (L["occ"][i], L["sdf"][i])
    return lib


def maps_from_mask(mask64):
    m = np.kron(np.asarray(mask64, bool).reshape(R, R), np.ones((2, 2), bool))
    m = shapes.normalise(m, rad=0.85)
    if m is None:
        raise ValueError("shape too small, too big or empty")
    sdf = shapes.signed_distance(m)
    return m.reshape(R, 2, R, 2).mean((1, 3)).astype(np.float32), sdf[1::2, 1::2][:R, :R].astype(np.float32)


def footprint(X, sp):
    d, _ = cKDTree(X).query(PIX)
    return (d < 0.7 * sp).reshape(R, R)


def score(X, occ):
    """Footprint IoU / precision / recall of the bots against the target mask (pixels within 0.7 spacings of a bot)."""
    sp = evalm.spacing(occ, len(X)); f = footprint(X, sp); t = occ > 0.5
    inter = (f & t).sum()
    return dict(iou=float(inter / max((f | t).sum(), 1)), prec=float(inter / max(f.sum(), 1)), rec=float(inter / max(t.sum(), 1)))


def extra(X, occ, sdf):
    mt = evalm.metrics(X, X, occ, sdf)
    return dict(inside=float(mt["inside"]), cover=float(mt["cover"]), coll=float(mt["coll"]))


def centroid(occ):
    ys, xs = np.nonzero(occ > 0.5)
    return np.array([AX[xs].mean(), AX[ys].mean()])


def separate(x, dmin, iters=3):
    for _ in range(iters):
        d = x[:, :, None] - x[:, None]
        dist = d.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        x = x + (torch.relu(dmin - dist) / 2 * d / dist * (dist > 1e-5)).sum(2)
    return x


def default_start(cfg):
    s = cfg.get("start") or "gauss"
    return "noise" if s == "gauss" else ("pile_c" if s.endswith("_c") else "pile_off")


def start_positions(kind, cfg, N, rng, occ):
    """-> (x0 (N,2) tensor, rigid-travel shift or None)."""
    if kind == "noise":
        return torch.tensor(rng.normal(size=(1, N, 2)) * cfg.get("sigma", 0.7), dtype=torch.float32), None
    x = torch.tensor(evalm.pile_fps(rng, 1, N, R0=0.7, off=0.5 if kind == "pile_off" else 0.0))
    if kind == "pile_off":
        return x, torch.tensor(centroid(occ), dtype=torch.float32) - x[0].mean(0)
    return x, None


@torch.no_grad()
def run_model(name, occ, sdf, N=256, K=64, seed=0, start="auto", jitter=0.0, motion=0.0, hardcore=False, vmax=None,
              x0=None, t0=0.0, keep_spacings=False, want_frames=True, max_frames=65, n_ref=None):
    m, cfg = get_model(name)
    if cfg.get("radius"):     # keep_spacings: hold the sensing range fixed in spacings of a swarm of n_ref (default N) bots
        m.radius = cfg["radius"] * (float(np.sqrt(256.0 / (n_ref or N))) if keep_spacings else 1.0)
    rng = np.random.default_rng(seed); torch.manual_seed(seed)
    maps = make_maps(torch.tensor(occ)[None], torch.tensor(sdf)[None], cfg.get("coarse", 0)) if cfg["cond"] == "map" else None
    sp = evalm.spacing(occ, N)
    kind = default_start(cfg) if start == "auto" else start
    frames, travel = [], 0
    if x0 is not None:
        x = torch.tensor(x0, dtype=torch.float32)[None]
    else:
        x, shift = start_positions(kind, cfg, N, rng, occ); t0 = 0.0
        if shift is not None:                  # phase 1: the pile travels as a rigid body to the shape's centre
            travel = 10
            frames += [(x[0] + shift * j / travel).numpy() for j in range(travel)]
            x = x + shift
    frames.append(x[0].numpy())
    for i in range(K):
        t = torch.full((1, N if x0 is None else x.shape[1]), t0 + (1 - t0) * i / K)
        if jitter:
            t = (t + torch.randn_like(t) * jitter).clamp(0, 1)
        v = m(x, t if jitter else t[:, 0], maps)
        dx = v * (1 - t0) / K
        if vmax:
            dx = dx * torch.clamp(float(vmax) * sp / dx.norm(dim=-1, keepdim=True).clamp_min(1e-9), max=1.0)
        x = x + dx
        if motion:
            x = x + torch.randn_like(x) * motion * sp / np.sqrt(K)
        if hardcore:
            x = separate(x, 0.7 * sp)
        frames.append(x[0].numpy())
    X = x[0].numpy()
    out = dict(final=X, sp=sp, travel=travel, start=kind, **score(X, occ), **extra(X, occ, sdf))
    if want_frames:
        idx = np.unique(np.linspace(0, len(frames) - 1, min(max_frames, len(frames))).round().astype(int))
        out["frames"] = [frames[i] for i in idx]
    return out


def run_rule(occ, sdf, N=256, seed=0, steps=200):
    """No learning: slide onto the shape, then spread out by local repulsion (evalm.classical)."""
    rng = np.random.default_rng(seed)
    maps = make_maps(torch.tensor(occ)[None], torch.tensor(sdf)[None], 0)
    x0 = torch.tensor(rng.normal(size=(1, N, 2)) * 0.7, dtype=torch.float32)
    with torch.no_grad():
        X = evalm.classical(x0, maps, torch.tensor([evalm.spacing(occ, N)]), steps=steps)[0].numpy()
    return dict(final=X, sp=evalm.spacing(occ, N), **score(X, occ), **extra(X, occ, sdf))


def ceiling(occ, sdf, N=256, seed=0):
    """The data's own best case: N evenly spaced (farthest-point) points inside the shape."""
    rng = np.random.default_rng(seed)
    P = PIX[(occ > 0.5).reshape(-1)]
    idx = [int(rng.integers(len(P)))]; d = ((P - P[idx[0]]) ** 2).sum(1)
    for _ in range(N - 1):
        j = int(d.argmax()); idx.append(j); d = np.minimum(d, ((P - P[j]) ** 2).sum(1))
    X = P[idx].astype(np.float32)
    return dict(final=X, sp=evalm.spacing(occ, N), **score(X, occ), **extra(X, occ, sdf))
