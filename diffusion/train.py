"""Flow-matching trainer + evaluator for particles forming shapes.

    python train.py configs/s1_cond_global.json

One config = one job. Writes runs/<name>/{ckpt.pt, log.txt, result.json, samples.png, traj.gif}.
Resumes from ckpt.pt. Training pairs noise with target points by an exact optimal-transport
matching *inside each set* (256 noise points <-> 256 shape points), so paths are straight and
do not cross.
"""
import json, os, sys, time, math, subprocess, copy
import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from model import Denoiser, build, make_maps, read_map
import evalm

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT = dict(name="x", cond="map", radius=None, layers=6, d=192, heads=6, time=True, n_train=10000,
               iters=30000, bs=32, lr=3e-4, seed=0, sigma=0.7, shuffle_labels=False, ot=True,
               eval_steps=[4, 8, 16, 32, 64], n_eval=128)


_AX = (np.arange(64) + 0.5) * (2 / 64) - 1


def fps_occ(occ, n, rng):
    """n well-spaced points on the shape given by a 64x64 occupancy map (rows = y, cols = x)."""
    ys, xs = np.nonzero(occ > 0.5)
    c = np.stack([_AX[xs], _AX[ys]], 1) + rng.uniform(-0.5, 0.5, (len(xs), 2)) * (2 / 64)
    s = [int(rng.integers(len(c)))]; d = ((c - c[s[0]]) ** 2).sum(1)
    for _ in range(n - 1):
        j = int(np.argmax(d)); s.append(j); d = np.minimum(d, ((c - c[j]) ** 2).sum(1))
    return c[s].astype(np.float32)


class Pairs(torch.utils.data.IterableDataset):
    """Yields (shape index, aug code, noise, matched target) batches; matching done on the CPU."""
    def __init__(s, pts, bs, sigma, ot, seed, start="gauss", occ=None, damage_p=0.0):
        s.pts, s.bs, s.sigma, s.ot, s.seed, s.start, s.occ, s.damage_p = pts, bs, sigma, ot, seed, start, occ, damage_p

    def __iter__(s):
        wi = torch.utils.data.get_worker_info()
        rng = np.random.default_rng(s.seed * 1000 + (wi.id if wi else 0) + 1)
        n, N = len(s.pts), s.pts.shape[1]
        while True:
            idx = rng.integers(n, size=s.bs)
            k, flip = int(rng.integers(4)), int(rng.integers(2))
            x1 = s.pts[idx].copy()
            if flip:
                x1[..., 0] *= -1
            for _ in range(k):
                x1 = np.stack([-x1[..., 1], x1[..., 0]], -1)
            if s.damage_p and rng.random() < s.damage_p:
                # repair batch: start from the shape with a contiguous chunk of bots removed (N' < 256 left);
                # the target is the same shape at N' bots, evenly spread
                Np = int(rng.integers(192, 241)); Dn = x1.shape[1] - Np
                ob = s.occ[idx]
                if flip:
                    ob = ob[..., ::-1]
                ob = np.rot90(ob, k, (1, 2))
                x0 = np.empty((s.bs, Np, 2), np.float32); x1n = np.empty((s.bs, Np, 2), np.float32)
                for b in range(s.bs):
                    c = x1[b][rng.integers(x1.shape[1])]
                    x0[b] = x1[b][np.argsort(((x1[b] - c) ** 2).sum(1))[Dn:]]
                    x1n[b] = fps_occ(ob[b], Np, rng)
                x1 = x1n
            elif s.start.startswith("pile_fps"):
                x0 = evalm.pile_fps(rng, s.bs, x1.shape[1], **evalm.pile_opts(s.start))
            elif s.start == "pile":   # a compact pile of radius 0.7 somewhere in the arena, like bots dropped together
                c = rng.uniform(-0.5, 0.5, (s.bs, 1, 2)); r = 0.7 * np.sqrt(rng.uniform(size=x1.shape[:2] + (1,)))
                th = rng.uniform(0, 2 * np.pi, x1.shape[:2] + (1,))
                x0 = (c + r * np.concatenate([np.cos(th), np.sin(th)], -1)).astype(np.float32)
            else:
                x0 = rng.normal(size=x1.shape).astype(np.float32) * s.sigma
            if s.ot:
                for b in range(s.bs):
                    C = ((x0[b][:, None] - x1[b][None]) ** 2).sum(-1)
                    _, col = linear_sum_assignment(C)
                    x1[b] = x1[b][col]
            yield idx, k, flip, x0, x1


def aug_maps(occ, sdf, k, flip):
    """Apply the same flip + 90-degree rotation as the points got (maps are indexed [row=y, col=x])."""
    if flip:
        occ, sdf = occ.flip(-1), sdf.flip(-1)
    if k:
        occ, sdf = torch.rot90(occ, k, (1, 2)), torch.rot90(sdf, k, (1, 2))
    return occ, sdf


def git_info():
    try:
        h = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=HERE, stderr=subprocess.DEVNULL).decode().strip()
        return h
    except Exception:
        p = os.path.join(HERE, "GIT_COMMIT")
        return open(p).read().strip() if os.path.exists(p) else "unknown"


def main(cfg_path):
    cfg = {**DEFAULT, **json.load(open(cfg_path))}
    name = cfg["name"]
    out = os.path.join(HERE, "runs", name); os.makedirs(out, exist_ok=True)
    logf = open(os.path.join(out, "log.txt"), "a")

    def log(*a):
        s = " ".join(str(x) for x in a); print(s, flush=True); logf.write(s + "\n"); logf.flush()

    torch.manual_seed(cfg["seed"]); np.random.seed(cfg["seed"])
    dev = "cuda"
    D = np.load(os.path.join(HERE, f"data/{cfg.get('data', 'shapes')}.npz"))
    ntr = cfg["n_train"]
    tr_pts = D["tr_pts"][:ntr]
    tr_occ = torch.tensor(D["tr_occ"][:ntr], device=dev); tr_sdf = torch.tensor(D["tr_sdf"][:ntr], device=dev)
    model = build(cfg).to(dev)
    ema = copy.deepcopy(model).eval()
    for p in ema.parameters():
        p.requires_grad_(False)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=0.01)
    log("params", sum(p.numel() for p in model.parameters()), "cfg", json.dumps(cfg))
    it0, ck = 0, os.path.join(out, "ckpt.pt")
    if os.path.exists(ck):
        s = torch.load(ck, map_location=dev)
        model.load_state_dict(s["model"]); ema.load_state_dict(s["ema"]); opt.load_state_dict(s["opt"]); it0 = s["it"]
        log("resumed at", it0)
    dl = torch.utils.data.DataLoader(Pairs(tr_pts, cfg["bs"], cfg["sigma"], cfg["ot"], cfg["seed"] + it0, cfg.get("start", "gauss"), D["tr_occ"][:ntr], cfg.get("damage_p", 0.0)),
                                     batch_size=None, num_workers=10, prefetch_factor=4, persistent_workers=True)
    t_start, losses = time.time(), []
    total = cfg["iters"]
    if it0 < total:
        for it, (idx, k, flip, x0, x1) in zip(range(it0, total), dl):
            lr = cfg["lr"] * min(1, (it + 1) / 500) * (0.5 * (1 + math.cos(math.pi * it / total)) * 0.97 + 0.03)
            for g in opt.param_groups:
                g["lr"] = lr
            x0, x1 = x0.to(dev), x1.to(dev)
            maps = None
            if cfg["cond"] == "map":
                mi = idx if not cfg["shuffle_labels"] else np.random.permutation(idx)
                mi = torch.as_tensor(mi, device=dev)
                o, sd = aug_maps(tr_occ[mi], tr_sdf[mi], k, flip)
                maps = make_maps(o, sd, cfg.get("coarse", 0))
            t = torch.rand(x0.shape[0], device=dev)
            if cfg.get("tjit", 0):
                # every particle gets its own slightly wrong clock; the straight-line target is unchanged
                t = (t[:, None] + torch.randn(x0.shape[:2], device=dev) * cfg["tjit"]).clamp(0, 1)
                xt = (1 - t[..., None]) * x0 + t[..., None] * x1
            else:
                xt = (1 - t[:, None, None]) * x0 + t[:, None, None] * x1
            t_in = t
            if cfg.get("tnoise", 0):
                # the particle's clock reading is wrong; where it actually is (xt) is unchanged
                t_in = (t[:, None] if t.dim() == 1 else t) + torch.randn(x0.shape[:2], device=dev) * cfg["tnoise"]
                t_in = t_in.clamp(0, 1)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                v = model(xt, t_in, maps)
            loss = F.mse_loss(v.float(), x1 - x0)
            opt.zero_grad(set_to_none=True); loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            with torch.no_grad():
                dec = min(0.999, (1 + it) / (10 + it))
                for pe, pm in zip(ema.parameters(), model.parameters()):
                    pe.lerp_(pm, 1 - dec)
            losses.append(loss.item())
            if not math.isfinite(losses[-1]):
                log("NaN loss, abort"); sys.exit(2)
            if (it + 1) % 500 == 0:
                log(f"it {it+1} loss {np.mean(losses[-500:]):.4f} gn {gn:.2f} lr {lr:.1e} {time.time()-t_start:.0f}s")
            if (it + 1) % 2000 == 0 or it + 1 == total:
                torch.save(dict(model=model.state_dict(), ema=ema.state_dict(), opt=opt.state_dict(), it=it + 1), ck + ".tmp")
                os.replace(ck + ".tmp", ck)
    train_time = time.time() - t_start
    res = evalm.evaluate(ema, cfg, D, out, log)
    res.update(config=cfg, git=git_info(), train_seconds=train_time, final_loss=float(np.mean(losses[-500:])) if losses else None)
    json.dump(res, open(os.path.join(out, "result.json.tmp"), "w"), indent=1)
    os.replace(os.path.join(out, "result.json.tmp"), os.path.join(out, "result.json"))
    log("done")


if __name__ == "__main__":
    main(sys.argv[1])
