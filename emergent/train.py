"""Train one controller end to end through the physics (backprop through time), evaluate, draw.

    python train.py configs/<name>.json            # writes runs/<name>/{result.json, log.txt, samples.png, traj.gif, ema.pt}
Resumes from runs/<name>/ckpt.pt if present.
"""
import json
import math
import os
import sys
import time
import copy
import numpy as np
import torch
from torch.utils.checkpoint import checkpoint

from sim import Physics, pile
from policy import Policy, map_features, build_maps

DEF = dict(data="data/world128.npz", iters=3000, B=16, lr=1e-3, T=64, K=4, hold=100, hold_eval=500, hold_w=1.0,
           g=1.0, g_warm=0, f_bond=8.0, f_act=3.0, mu=0.5, reaction="internal", r_sense=3.0, H=32, M=96, rounds=2,
           use_h=True, see_v=True, out_w=1.0, act_w=0.0, seed=0, offset=0.0, shuffle_targets=False, eval_every=500,
           ema=0.995, clip=1.0, n_eval=192, k_c=200.0, r_bond=1.25, r_hold=1.15, damage_p=0.0, damage_r=2.5,
           grip_fixed=None, families=None, g_eval=None, compile=True)


def load_cfg(path):
    cfg = dict(DEF)
    cfg.update(json.load(open(path)))
    cfg.setdefault("name", os.path.splitext(os.path.basename(path))[0])
    return cfg


class Data:
    def __init__(self, path, dev, families=None):
        z = np.load(path)
        self.W = float(z["W"])
        self.R = int(z["R"])
        self.families = [str(f) for f in z["families"]]
        self.sets = {}
        for s in ("train", "val", "held"):
            occ = torch.tensor(z[f"{s}_occ"])
            sdf = torch.tensor(z[f"{s}_sdf"].astype(np.float32))
            fam = torch.tensor(z[f"{s}_fam"].astype(np.int64))
            keep = torch.ones(len(fam), dtype=torch.bool)
            if families is not None and s == "train":
                keep = torch.tensor([self.families[int(f)] in families for f in fam])
            maps = torch.cat([build_maps(sdf[i:i + 256].to(dev), occ[i:i + 256].to(dev), self.W) for i in range(0, len(occ), 256)])
            self.sets[s] = dict(maps=maps[keep].half(), pts=torch.tensor(z[f"{s}_pts"])[keep].to(dev),
                                occ=occ[keep].to(dev), fam=fam[keep].to(dev))
        self.N = self.sets["train"]["pts"].shape[1]


def rollout(pol, ph, maps, x0, cfg, g_scale=1.0, grad=True, alive=None, record=False, damage=None, hold=None):
    """Build for T control steps (K physics steps each), then hold passively (motors off, grips kept).
    Returns dict with x at end of build, x at end of hold, grip, mean squared action, and frames if record."""
    B, N, _ = x0.shape
    dev = x0.device
    x, v = x0, torch.zeros_like(x0)
    h = torch.zeros(B, N, pol.H, device=dev)
    alive = torch.ones(B, N, device=dev) if alive is None else alive
    hold = cfg["hold"] if hold is None else hold
    W = ph.W
    frames = []
    asq = 0.0
    g_scale = torch.tensor(float(g_scale), device=dev)

    def ctrl(x, v, h, alive):
        mf = map_features(maps, x, W)
        a, grip, h = pol(x, v, h, mf, alive)
        if cfg["grip_fixed"] is not None:
            grip = torch.full_like(grip, float(cfg["grip_fixed"]))
        for _ in range(cfg["K"]):
            x, v = ph.step(x, v, a, grip, alive, g_scale)
        return x, v, h, grip, (a * a).mean()

    def passive(x, v, grip, alive):
        for _ in range(cfg["K"]):
            x, v = ph.step(x, v, None, grip, alive, g_scale)
        return x, v

    grip = torch.ones(B, N, device=dev)
    for t in range(cfg["T"]):
        if damage is not None and t == cfg["T"] // 2:
            centres, rad = damage           # cut a disc of radius rad around each centre
            dm = ((x - centres[:, None, :]).norm(dim=-1) > rad).float()
            alive = alive * dm
            x = torch.where(alive[..., None] > 0, x, torch.full_like(x, -50.0))
            v = v * alive[..., None]
        if grad:
            x, v, h, grip, aq = checkpoint(ctrl, x, v, h, alive, use_reentrant=False)
        else:
            x, v, h, grip, aq = ctrl(x, v, h, alive)
        asq = asq + aq / cfg["T"]
        if record:
            frames.append((x.detach().cpu(), grip.detach().cpu()))
    xb = x
    for t in range(hold // cfg["K"]):
        if grad:
            x, v = checkpoint(passive, x, v, grip, alive, use_reentrant=False)
        else:
            x, v = passive(x, v, grip, alive)
        if record and t % 2 == 0:
            frames.append((x.detach().cpu(), grip.detach().cpu()))
    return dict(xb=xb, xh=x, grip=grip, asq=asq, frames=frames, alive=alive)


def shape_loss(x, alive, Q, maps, W, out_w):
    D = torch.cdist(x, Q) ** 2
    big = 1e4 * (1 - alive)[:, :, None]
    t1 = ((D.min(2).values) * alive).sum(1) / alive.sum(1).clamp(min=1)
    t2 = (D + big).min(1).values.mean(1)
    sd = map_features(maps[:, :1], x, W)[..., 0] * 4.0
    out = ((torch.relu(sd) ** 2) * alive).sum(1) / alive.sum(1).clamp(min=1)
    return (t1 + t2 + out_w * out).mean(), dict(t1=t1.mean().item(), t2=t2.mean().item(), out=out.mean().item())


# ---------------------------------------------------------------- metrics
def footprint(x, alive, R, W, r=0.6):
    c = (torch.arange(R, device=x.device, dtype=torch.float32) + 0.5) * (W / R)
    cy, cx = torch.meshgrid(c, c, indexing="ij")
    cells = torch.stack([cx.reshape(-1), cy.reshape(-1)], -1)
    out = []
    for b in range(x.shape[0]):
        xa = x[b][alive[b] > 0]
        if len(xa) == 0:
            out.append(torch.zeros(R, R, dtype=torch.bool, device=x.device))
            continue
        out.append((torch.cdist(cells, xa).min(1).values < r).reshape(R, R))
    return torch.stack(out)


def iou(fp, occ):
    inter = (fp & occ).flatten(1).sum(1).float()
    uni = (fp | occ).flatten(1).sum(1).float()
    return inter / uni.clamp(min=1)


def overhang_map(occ):
    """A shape cell is an overhang if some cell below it in the same column is empty."""
    below_empty = torch.cumsum((~occ).int(), dim=1) > 0          # row index = y; any empty at or below
    return occ & below_empty


def struct_stats(ph, x, grip, alive, occ, W):
    """Bonds and bonded triangles per bot, split into bots inside overhang cells and inside supported cells."""
    A = ph.bonds(x, grip, alive).float()
    deg = A.sum(2)
    tri = torch.einsum("bij,bjk,bki->bi", A, A, A) / 2
    R = occ.shape[-1]
    ix = (x[..., 0] / W * R).long().clamp(0, R - 1)
    iy = (x[..., 1] / W * R).long().clamp(0, R - 1)
    ov = overhang_map(occ)
    b = torch.arange(x.shape[0], device=x.device)[:, None]
    in_shape = occ[b, iy, ix] & (alive > 0)
    in_ov = ov[b, iy, ix] & in_shape
    in_sup = in_shape & ~in_ov
    res = {}
    for k, m in [("all", alive > 0), ("ov", in_ov), ("sup", in_sup)]:
        n = m.float().sum().clamp(min=1)
        res[f"deg_{k}"] = (deg * m).sum().item() / n.item()
        res[f"tri_{k}"] = (tri * m).sum().item() / n.item()
    return res


def evaluate(pol, ph, data, cfg, split, n=None, dev="cuda", bs=32, seed=123, with_baselines=False):
    S = data.sets[split]
    n = min(n or cfg["n_eval"], len(S["fam"]))
    gen = torch.Generator(device="cpu").manual_seed(seed)
    acc = {}
    per_fam = {}

    def add(d, k, val, fam):
        d.setdefault(k, []).append(val)
        for f, vv in zip(fam.tolist(), val.tolist()):
            per_fam.setdefault(data.families[f], {}).setdefault(k, []).append(vv)

    for i in range(0, n, bs):
        sl = slice(i, min(n, i + bs))
        maps, Q, occ, fam = S["maps"][sl].float(), S["pts"][sl], S["occ"][sl], S["fam"][sl]
        B = Q.shape[0]
        x0 = pile(B, data.N, W=data.W, jitter=0.05, offset=0.0, gen=gen).to(dev)
        with torch.no_grad():
            r = rollout(pol, ph, maps, x0, cfg, grad=False, hold=cfg["hold_eval"])
            al = r["alive"]
            fb, fh = footprint(r["xb"], al, data.R, data.W), footprint(r["xh"], al, data.R, data.W)
            add(acc, "iou_build", iou(fb, occ), fam)
            add(acc, "iou_hold", iou(fh, occ), fam)
            dQ = torch.cdist(Q, r["xh"]).min(2).values
            add(acc, "cover_hold", (dQ < 0.75).float().mean(1), fam)
            drop = (r["xb"][..., 1] - r["xh"][..., 1])
            add(acc, "sag", drop.mean(1), fam)
            add(acc, "fallen", (drop > 1.5).float().mean(1), fam)
            add(acc, "grip", r["grip"].mean(1), fam)
            st = struct_stats(ph, r["xh"], r["grip"], al, occ, data.W)
            for k, vv in st.items():
                acc.setdefault(k, []).append(torch.tensor([vv]))
            if with_baselines:
                ones = torch.ones(B, data.N, device=dev)
                fp0 = footprint(x0, ones, data.R, data.W)
                add(acc, "base_pile_iou", iou(fp0, occ), fam)
                add(acc, "base_ceiling_iou", iou(footprint(Q, ones, data.R, data.W), occ), fam)
                xq, vq = Q.clone(), torch.zeros_like(Q)
                for _ in range(cfg["hold_eval"]):
                    xq, vq = ph.step(xq, vq, None, ones, ones)
                add(acc, "base_ceilfreeze_iou_hold", iou(footprint(xq, ones, data.R, data.W), occ), fam)
                dropq = Q[..., 1] - xq[..., 1]
                add(acc, "base_ceilfreeze_sag", dropq.mean(1), fam)
                add(acc, "base_ceilfreeze_fallen", (dropq > 1.5).float().mean(1), fam)
                stq = struct_stats(ph, xq, ones, ones, occ, data.W)
                for k, vv in stq.items():
                    acc.setdefault("base_ceilfreeze_" + k, []).append(torch.tensor([vv]))
    out = {k: float(torch.cat([t.float().reshape(-1) for t in v]).mean()) for k, v in acc.items()}
    # bootstrap 95% interval over shapes for the main numbers
    for k in ("iou_build", "iou_hold", "base_ceilfreeze_iou_hold"):
        if k in acc:
            vals = torch.cat([t.float().reshape(-1) for t in acc[k]]).cpu().numpy()
            rng = np.random.default_rng(0)
            bs_ = [vals[rng.integers(0, len(vals), len(vals))].mean() for _ in range(1000)]
            out[k + "_ci"] = [float(np.percentile(bs_, 2.5)), float(np.percentile(bs_, 97.5))]
    pf = {f: {k: float(np.mean(v)) for k, v in d.items()} for f, d in per_fam.items()}
    return out, pf


# ---------------------------------------------------------------- pictures
def draw(pol, ph, data, cfg, path_png, path_gif, dev):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image
    rows = []
    for split in ("val", "held"):
        S = data.sets[split]
        idx = torch.linspace(0, len(S["fam"]) - 1, 8).long()
        rows.append((split, idx))
    fig, axes = plt.subplots(4, 8, figsize=(20, 10.5))
    gif_src = None
    for ri, (split, idx) in enumerate(rows):
        S = data.sets[split]
        maps, Q, occ = S["maps"][idx].float(), S["pts"][idx], S["occ"][idx]
        x0 = pile(len(idx), data.N, W=data.W, gen=torch.Generator().manual_seed(7)).to(dev)
        with torch.no_grad():
            r = rollout(pol, ph, maps, x0, cfg, grad=False, record=(ri == 0), hold=cfg["hold_eval"])
            ones = torch.ones_like(Q[..., 0])
            xq, vq = Q.clone(), torch.zeros_like(Q)
            for _ in range(cfg["hold_eval"]):
                xq, vq = ph.step(xq, vq, None, ones, ones)
        if ri == 0:
            gif_src = (r["frames"], occ.cpu())
        A = ph.bonds(r["xh"], r["grip"], r["alive"]).cpu()
        Aq = ph.bonds(xq, ones, ones).cpu()
        for k in range(8):
            for rr, (xs, AA, title) in enumerate([(r["xh"][k].cpu(), A[k], "learned"), (xq[k].cpu(), Aq[k], "even sample, frozen")]):
                ax = axes[ri * 2 + rr, k]
                ax.imshow(occ[k].cpu(), origin="lower", extent=[0, data.W, 0, data.W], cmap="Greys", alpha=0.35, vmin=0, vmax=1)
                if rr == 0:
                    xb = r["xb"][k].cpu()
                    ax.scatter(xb[:, 0], xb[:, 1], s=6, c="#9ab", alpha=0.6)
                ii, jj = torch.nonzero(torch.triu(AA), as_tuple=True)
                for a_, b_ in zip(ii.tolist(), jj.tolist()):
                    ax.plot([xs[a_, 0], xs[b_, 0]], [xs[a_, 1], xs[b_, 1]], c="#c63", lw=0.5)
                ax.scatter(xs[:, 0], xs[:, 1], s=9, c="#236")
                ax.set_xlim(0, data.W); ax.set_ylim(0, data.W); ax.set_xticks([]); ax.set_yticks([])
                fp = footprint(xs[None].to(dev), torch.ones(1, xs.shape[0], device=dev), data.R, data.W)
                ax.set_title(f"{split} {title} IoU {iou(fp, occ[k:k+1]).item():.2f}", fontsize=7)
    plt.suptitle(f"{cfg['name']}: rows 1,3 learned (grey = end of build, dark = after passive hold, lines = bonds); "
                 f"rows 2,4 the even sample placed and frozen in the same physics", fontsize=9)
    plt.tight_layout()
    plt.savefig(path_png, dpi=70)
    plt.close(fig)
    frames, occ = gif_src
    imgs = []
    for t, (xs, gr) in enumerate(frames):
        fig, axs = plt.subplots(1, 4, figsize=(10, 2.8))
        for k in range(4):
            ax = axs[k]
            ax.imshow(occ[k], origin="lower", extent=[0, data.W, 0, data.W], cmap="Greys", alpha=0.35, vmin=0, vmax=1)
            ax.scatter(xs[k, :, 0], xs[k, :, 1], s=8, c=gr[k], cmap="viridis", vmin=0, vmax=1)
            ax.set_xlim(0, data.W); ax.set_ylim(0, data.W); ax.set_xticks([]); ax.set_yticks([])
        axs[0].set_title("build" if t < cfg["T"] else "passive hold", fontsize=8)
        fig.canvas.draw()
        imgs.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[..., :3]))
        plt.close(fig)
    imgs[0].save(path_gif, save_all=True, append_images=imgs[1:], duration=60, loop=0)


# ---------------------------------------------------------------- main
def main(path):
    cfg = load_cfg(path)
    dev = "cuda"
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    out = os.path.join("runs", cfg["name"])
    os.makedirs(out, exist_ok=True)
    log = open(os.path.join(out, "log.txt"), "a")

    def P(*a):
        s = " ".join(str(x) for x in a)
        print(s, flush=True)
        log.write(s + "\n"); log.flush()

    data = Data(cfg["data"], dev, cfg["families"])
    ph = Physics(g=cfg["g"], f_bond=cfg["f_bond"], f_act=cfg["f_act"], mu=cfg["mu"], reaction=cfg["reaction"],
                 W=data.W, k_c=cfg["k_c"], r_bond=cfg["r_bond"], r_hold=cfg["r_hold"])
    ph_eval = Physics(g=cfg["g"] if cfg["g_eval"] is None else cfg["g_eval"], f_bond=cfg["f_bond"], f_act=cfg["f_act"],
                      mu=cfg["mu"], reaction=cfg["reaction"], W=data.W, k_c=cfg["k_c"], r_bond=cfg["r_bond"], r_hold=cfg["r_hold"])
    pol = Policy(H=cfg["H"], M=cfg["M"], rounds=cfg["rounds"], r_sense=cfg["r_sense"], f_act=cfg["f_act"],
                 use_h=cfg["use_h"], see_v=cfg["see_v"]).to(dev)
    ema = copy.deepcopy(pol)
    run_pol, run_ema = pol, ema
    if cfg["compile"]:
        ph.step = torch.compile(ph.step, dynamic=False)
        ph_eval.step = torch.compile(ph_eval.step, dynamic=False)
        run_pol, run_ema = torch.compile(pol, dynamic=False), torch.compile(ema, dynamic=False)
    opt = torch.optim.Adam(pol.parameters(), lr=cfg["lr"])
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda i: min(1, (i + 1) / 100) * 0.5 * (1 + math.cos(math.pi * min(i, cfg["iters"]) / cfg["iters"])))
    start = 0
    ck = os.path.join(out, "ckpt.pt")
    if os.path.exists(ck):
        s = torch.load(ck, map_location=dev)
        pol.load_state_dict(s["pol"]); ema.load_state_dict(s["ema"]); opt.load_state_dict(s["opt"])
        sched.load_state_dict(s["sched"]); start = s["it"]
        P(f"resumed at {start}")
    P(json.dumps(cfg))
    P(f"params {sum(p.numel() for p in pol.parameters())}")
    T = data.sets["train"]
    gen = torch.Generator().manual_seed(cfg["seed"] + 1000 * start)
    t0 = time.time()
    bad = 0
    for it in range(start, cfg["iters"]):
        idx = torch.randint(0, len(T["fam"]), (cfg["B"],), generator=gen).to(dev)
        maps, Q = T["maps"][idx].float(), T["pts"][idx]
        if cfg["shuffle_targets"]:
            Q = T["pts"][torch.randint(0, len(T["fam"]), (cfg["B"],), generator=gen).to(dev)]
        off = (torch.rand(1, generator=gen).item() * 2 - 1) * cfg["offset"]
        x0 = pile(cfg["B"], data.N, W=data.W, offset=off, gen=gen).to(dev)
        gs = min(1.0, (it + 1) / cfg["g_warm"]) if cfg["g_warm"] else 1.0
        damage = None
        if cfg["damage_p"] > 0 and torch.rand(1, generator=gen).item() < cfg["damage_p"]:
            c = Q[torch.arange(cfg["B"], device=dev), torch.randint(0, data.N, (cfg["B"],), generator=gen).to(dev)]
            damage = (c, cfg["damage_r"])   # at T/2, remove the bots within damage_r of a random target point
        r = rollout(run_pol, ph, maps, x0, cfg, g_scale=gs, grad=True, damage=damage)
        lb, pb = shape_loss(r["xb"], r["alive"], Q, maps, data.W, cfg["out_w"])
        lh, ph_ = shape_loss(r["xh"], r["alive"], Q, maps, data.W, cfg["out_w"])
        loss = lb + cfg["hold_w"] * lh + cfg["act_w"] * r["asq"]
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(pol.parameters(), cfg["clip"])
        if not torch.isfinite(gn):
            bad += 1
            P(f"it {it} non-finite grad, skipped ({bad})")
            if bad > 50:
                raise RuntimeError("too many non-finite gradients")
            continue
        opt.step(); sched.step()
        with torch.no_grad():
            for pe, pp in zip(ema.parameters(), pol.parameters()):
                pe.mul_(cfg["ema"]).add_(pp, alpha=1 - cfg["ema"])
        if it % 25 == 0:
            el = time.time() - t0
            P(f"it {it} loss {loss.item():.3f} build {lb.item():.3f} (t1 {pb['t1']:.2f} t2 {pb['t2']:.2f} out {pb['out']:.2f}) "
              f"hold {lh.item():.3f} asq {float(r['asq']):.2f} grip {r['grip'].mean().item():.2f} gn {gn.item():.2f} g {gs:.2f} "
              f"{el / (it - start + 1):.2f}s/it")
        if (it + 1) % 250 == 0 or it + 1 == cfg["iters"]:
            tmp = ck + f".tmp{os.getpid()}"
            torch.save(dict(pol=pol.state_dict(), ema=ema.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(), it=it + 1), tmp)
            os.replace(tmp, ck)
        if (it + 1) % cfg["eval_every"] == 0 and it + 1 < cfg["iters"]:
            m, _ = evaluate(run_ema, ph_eval, data, cfg, "val", n=64, dev=dev)
            P(f"eval it {it + 1}: " + " ".join(f"{k} {v:.3f}" for k, v in m.items() if not k.endswith("_ci")))
    torch.save(dict(ema=ema.state_dict(), cfg=cfg), os.path.join(out, "ema.pt"))
    res = dict(config=cfg, commit=open("GIT_COMMIT").read().strip() if os.path.exists("GIT_COMMIT") else None,
               train_seconds=time.time() - t0, metrics={}, per_family={})
    for split in ("val", "held"):
        m, pf = evaluate(run_ema, ph_eval, data, cfg, split, dev=dev, with_baselines=True)
        res["metrics"][split] = m
        res["per_family"][split] = pf
        P(f"final {split}: " + " ".join(f"{k} {v:.3f}" for k, v in m.items() if not k.endswith("_ci")))
    try:
        draw(run_ema, ph_eval, data, cfg, os.path.join(out, "samples.png"), os.path.join(out, "traj.gif"), dev)
    except Exception as e:  # pictures must not lose the result
        P(f"draw failed: {e!r}")
    tmp = os.path.join(out, f"result.json.tmp{os.getpid()}")
    json.dump(res, open(tmp, "w"), indent=1)
    os.replace(tmp, os.path.join(out, "result.json"))
    P("done")


if __name__ == "__main__":
    main(sys.argv[1])
