"""Evolution strategies for a local lattice swarm controller under gravity and a cantilever limit. No teacher.

The controller is a small neural cellular automaton run at every bot: three 3x3 convolutions (each bot sees a 7x7
window: occupancy, the target blueprint, which neighbours are held, and their hidden state), then per bot 9 move
logits (stay, 4 slides, 4 pivots; invalid moves masked) and a new hidden state that travels with the bot.
Fitness = fraction of target cells filled after the build and a settling phase (nothing is said about supports,
build order or not dropping bots: unheld bots simply fall).

    python lat/es.py lat/configs/<name>.json        # from emergent/; writes runs/<name>/
"""
import json
import math
import os
import sys
import time
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from latworld import MOVES, INF, support_span, valid_moves, apply_moves, gravity, look

DEF = dict(H=20, W=40, L=1, T=160, settle=60, spares=0.3, C=8, M=32, pop=128, shapes=8, sigma=0.04, lr=0.03,
           gens=1500, eval_every=50, n_eval=64, seed=0, temp=1.0, families=None, held_out=["mushroom", "tree"],
           support_iters=8, speed_w=0.2, n_bank=4000, damage=0.0)

FAMS = ["tower", "gamma", "tee", "bridge", "stairs", "blocks", "mushroom", "tree"]


# ---------------------------------------------------------------- targets
def gen_target(fam, rng, H, W):
    t = np.zeros((H, W), bool)
    if fam == "tower":
        w, h = rng.integers(2, 6), rng.integers(6, 15)
        t[:h, :w] = True
    elif fam == "gamma":                      # one-sided cantilever
        w, h, b, th = rng.integers(1, 4), rng.integers(5, 13), rng.integers(3, 10), rng.integers(1, 3)
        t[:h, :w] = True
        t[h:h + th, :w + b] = True
        if rng.random() < 0.5:
            t = t[:, ::-1]
    elif fam == "tee":
        w, h, b, th = rng.integers(1, 4), rng.integers(5, 13), rng.integers(2, 7), rng.integers(1, 3)
        t[:h, b:b + w] = True
        t[h:h + th, :2 * b + w] = True
    elif fam == "bridge":
        lw, h, gap, th = rng.integers(1, 3), rng.integers(3, 9), rng.integers(2, 12), rng.integers(1, 3)
        t[:h, :lw] = True
        t[:h, lw + gap:2 * lw + gap] = True
        t[h:h + th, :2 * lw + gap] = True
    elif fam == "stairs":
        k, w, dh = rng.integers(2, 6), rng.integers(1, 4), rng.integers(1, 4)
        for q in range(k):
            t[:dh * (q + 1), q * w:(q + 1) * w] = True
        if rng.random() < 0.5:
            t = t[:, ::-1]
    elif fam == "blocks":
        x0, w, h = 4, rng.integers(2, 6), rng.integers(2, 6)
        t[:h, x0:x0 + w] = True
        top = [(x0, x0 + w, h)]
        for _ in range(rng.integers(1, 4)):
            a, b, z = top[rng.integers(len(top))]
            w2, h2 = rng.integers(2, 7), rng.integers(1, 4)
            s = rng.integers(a - w2 + 1, b)
            s = max(0, s)
            t[z:z + h2, s:s + w2] = True
            top.append((s, s + w2, z + h2))
    elif fam == "mushroom":
        w, h, cap, th = rng.integers(1, 3), rng.integers(4, 11), rng.integers(2, 6), rng.integers(1, 3)
        t[:h, cap:cap + w] = True
        t[h:h + th, :2 * cap + w] = True
        t[h + th:h + th + 1, 1:2 * cap + w - 1] = True
    elif fam == "tree":
        h = rng.integers(7, 15)
        t[:h, 6] = True
        for _ in range(rng.integers(2, 4)):
            z = rng.integers(3, h)
            ln = rng.integers(2, 6)
            if rng.random() < 0.5:
                t[z, 7:7 + ln] = True
            else:
                t[z, max(0, 6 - ln):6] = True
    ys, xs = np.nonzero(t)
    if len(xs) == 0:
        return None
    t = t[:, xs.min():xs.max() + 1]
    if t.shape[1] > W - 20 or t[:H].sum() < 12 or t.shape[0] > H or not t[0].any():
        return None
    out = np.zeros((H, W), bool)
    off = rng.integers(18, W - t.shape[1])
    out[:t.shape[0], off:off + t.shape[1]] = t[:H]
    return out


def bank(fams, n, H, W, seed):
    rng = np.random.default_rng(seed)
    out, fid = [], []
    while len(out) < n:
        f = fams[len(out) % len(fams)]
        t = gen_target(f, rng, H, W)
        if t is not None and 12 <= t.sum() <= 110:
            out.append(t)
            fid.append(FAMS.index(f))
    return torch.tensor(np.array(out), dtype=torch.float32), torch.tensor(fid)


def pile(tgt, spares, H):
    """Bots stacked bottom-up in columns 1..12 on the floor, |target| * (1 + spares) of them."""
    E, Hh, W = tgt.shape
    n = (tgt.flatten(1).sum(1) * (1 + spares)).round().long()
    occ = torch.zeros_like(tgt)
    width = 12
    idx = torch.arange(Hh * width, device=tgt.device)
    rows, cols = idx // width, idx % width + 1
    occ[:, rows, cols] = (idx[None] < n[:, None]).float()
    return occ


# ---------------------------------------------------------------- controller (population in one grouped conv)
class NCA:
    def __init__(self, C, M, cin=4):
        self.C, self.M, self.cin = C, M, cin + C
        self.shapes = [(M, self.cin, 3, 3), (M,), (M, M, 3, 3), (M,), (M, M, 3, 3), (M,), (9 + C, M, 1, 1), (9 + C,)]
        self.n = sum(int(np.prod(s)) for s in self.shapes)

    def init(self, gen):
        parts = []
        for k, s in enumerate(self.shapes):
            if len(s) == 1:
                parts.append(torch.zeros(s))
            elif k == 6:
                parts.append(torch.zeros(s))          # last layer zero: start uniform-random
            else:
                fan = s[1] * s[2] * s[3]
                parts.append(torch.randn(s, generator=gen) * math.sqrt(2.0 / fan))
        return torch.cat([p.reshape(-1) for p in parts])

    def unpack(self, theta):          # theta (P, n) -> list of (P, *shape)
        out, o = [], 0
        for s in self.shapes:
            k = int(np.prod(s))
            out.append(theta[:, o:o + k].reshape(theta.shape[0], *s))
            o += k
        return out

    def forward(self, theta, x):
        """theta (P, n); x (P*S, cin, H, W) member-major -> (P*S, 9 + C, H, W)."""
        P = theta.shape[0]
        E, cin, H, W = x.shape
        S = E // P
        w = self.unpack(theta)
        h = x.reshape(P, S, cin, H, W).transpose(0, 1).reshape(S, P * cin, H, W)
        for li in range(3):
            Wt, b = w[2 * li], w[2 * li + 1]
            h = F.conv2d(h, Wt.reshape(-1, *Wt.shape[2:]), b.reshape(-1), padding=1, groups=P)
            h = F.relu(h)
        Wt, b = w[6], w[7]
        h = F.conv2d(h, Wt.reshape(-1, *Wt.shape[2:]), b.reshape(-1), groups=P)
        return h.reshape(S, P, -1, H, W).transpose(0, 1).reshape(E, -1, H, W)


def rollout(nca, theta, tgt, cfg, gen, record=False, damage_at=None):
    """theta (P, n); tgt (P*S, H, W). Returns fitness (P*S,) and stats."""
    E, H, W = tgt.shape
    dev = tgt.device
    occ = pile(tgt, cfg["spares"], H)
    hid = torch.zeros(E, cfg["C"], H, W, device=dev)
    floor = torch.zeros(E, 1, H, W, device=dev)
    floor[:, :, 0, :] = 1
    s = support_span(occ, cfg["L"], None, H + W)
    fill_sum = torch.zeros(E, device=dev)
    falls = torch.zeros(E, device=dev)
    frames = []
    nt = tgt.flatten(1).sum(1)
    for t in range(cfg["T"]):
        held = occ * (s <= cfg["L"]).float()
        inp = torch.cat([occ[:, None], tgt[:, None], held[:, None], floor, hid], 1)
        out = nca.forward(theta, inp)
        logits = out[:, :9] / cfg["temp"]
        hid_new = torch.tanh(out[:, 9:]) * occ[:, None]
        mask = valid_moves(occ, held)
        g = -torch.log(-torch.log(torch.rand(logits.shape, device=dev, generator=gen).clamp(1e-9, 1 - 1e-9)))
        act = torch.where(mask > 0, logits + g, torch.full_like(logits, -1e9)).argmax(1)
        prio = torch.rand(occ.shape, device=dev, generator=gen)
        occ, hid, _ = apply_moves(occ, hid_new, act, prio)
        s = support_span(occ, cfg["L"], s, cfg["support_iters"])
        occ, hid, unheld = gravity(occ, hid, cfg["L"], s)
        falls += unheld
        if damage_at is not None and t == damage_at:
            occ = occ * (1 - cut_mask(tgt, gen))
            hid = hid * occ[:, None]
        fill_sum += (occ * tgt).flatten(1).sum(1) / nt
        if record:
            frames.append(occ.cpu())
    for _ in range(cfg["settle"]):
        s = support_span(occ, cfg["L"], s, cfg["support_iters"])
        occ, hid, unheld = gravity(occ, hid, cfg["L"], s)
        falls += unheld
        if record and _ % 3 == 0:
            frames.append(occ.cpu())
    fill = (occ * tgt).flatten(1).sum(1) / nt
    fit = fill + cfg["speed_w"] * fill_sum / cfg["T"]
    outside = (occ * (1 - tgt)).flatten(1).sum(1)
    return fit, dict(fill=fill, falls=falls, outside=outside, occ=occ, frames=frames)


def cut_mask(tgt, gen):
    """Remove the upper half of the target region (damage), as a mask over cells."""
    E, H, W = tgt.shape
    rows = torch.arange(H, device=tgt.device)[None, :, None].float()
    top = (tgt * rows).flatten(1).amax(1)
    return ((rows > (top / 2)[:, None, None]).float() * (tgt > 0).float())


def struct_stats(occ, tgt):
    """Scaffolding: bots outside the target that sit under a target overhang (a target cell somewhere above them)."""
    above_tgt = torch.flip(torch.cumsum(torch.flip(tgt, [1]), 1), [1]) - tgt > 0
    under = (occ * (1 - tgt) * above_tgt.float()).flatten(1).sum(1)
    return under


# ---------------------------------------------------------------- main
def main(path):
    cfg = dict(DEF)
    cfg.update(json.load(open(path)))
    cfg.setdefault("name", os.path.splitext(os.path.basename(path))[0])
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    out = os.path.join("runs", cfg["name"])
    os.makedirs(out, exist_ok=True)
    log = open(os.path.join(out, "log.txt"), "a")

    def P_(*a):
        s = " ".join(str(x) for x in a)
        print(s, flush=True)
        log.write(s + "\n"); log.flush()

    H, W = cfg["H"], cfg["W"]
    train_f = cfg["families"] or [f for f in FAMS if f not in cfg["held_out"]]
    TR, TRf = bank(train_f, cfg["n_bank"], H, W, 1)
    VA, VAf = bank(train_f, cfg["n_eval"], H, W, 2)
    HO, HOf = bank(cfg["held_out"], cfg["n_eval"], H, W, 3) if cfg["held_out"] else (None, None)
    TR, VA = TR.to(dev), VA.to(dev)
    HO = HO.to(dev) if HO is not None else None
    nca = NCA(cfg["C"], cfg["M"])
    gen = torch.Generator().manual_seed(cfg["seed"])
    ggen = torch.Generator(device=dev).manual_seed(cfg["seed"])
    theta = nca.init(gen).to(dev)
    m = torch.zeros_like(theta); v = torch.zeros_like(theta)
    start = 0
    ck = os.path.join(out, "ckpt.pt")
    if os.path.exists(ck):
        s = torch.load(ck, map_location=dev)
        theta, m, v, start = s["theta"], s["m"], s["v"], s["gen"]
        P_(f"resumed at {start}")
    P_(json.dumps(cfg))
    P_(f"params {nca.n}")
    Pp, S = cfg["pop"], cfg["shapes"]
    t0 = time.time()
    for g in range(start, cfg["gens"]):
        eps = torch.randn(Pp // 2, nca.n, device=dev, generator=ggen)
        eps = torch.cat([eps, -eps])
        thetas = theta[None] + cfg["sigma"] * eps
        idx = torch.randint(0, len(TR), (S,), device=dev, generator=ggen)
        tgt = TR[idx].repeat(Pp, 1, 1)                      # member-major: same shapes for every member
        dmg = cfg["T"] // 2 if (cfg["damage"] > 0 and torch.rand(1, generator=gen).item() < cfg["damage"]) else None
        with torch.no_grad():
            fit, st = rollout(nca, thetas, tgt, cfg, ggen, damage_at=dmg)
        f = fit.reshape(Pp, S).mean(1)
        r = torch.argsort(torch.argsort(f)).float()
        r = r / (Pp - 1) - 0.5                                # centred ranks
        grad = (r[:, None] * eps).sum(0) / (Pp * cfg["sigma"])
        # Adam ascent
        m = 0.9 * m + 0.1 * grad
        v = 0.999 * v + 0.001 * grad * grad
        theta = theta + cfg["lr"] * (m / (1 - 0.9 ** (g + 1))) / (torch.sqrt(v / (1 - 0.999 ** (g + 1))) + 1e-8) - cfg["lr"] * 0.005 * theta
        if g % 10 == 0:
            P_(f"gen {g} fit mean {f.mean().item():.3f} max {f.max().item():.3f} fill {st['fill'].mean().item():.3f} "
               f"falls/ep {st['falls'].mean().item():.1f} |theta| {theta.norm().item():.1f} {(time.time() - t0) / (g - start + 1):.2f}s/gen")
        if (g + 1) % 25 == 0 or g + 1 == cfg["gens"]:
            tmp = ck + f".tmp{os.getpid()}"
            torch.save(dict(theta=theta, m=m, v=v, gen=g + 1), tmp)
            os.replace(tmp, ck)
        if (g + 1) % cfg["eval_every"] == 0 and g + 1 < cfg["gens"]:
            ev = evaluate(nca, theta, VA, VAf, cfg, dev)
            P_(f"eval gen {g + 1}: " + " ".join(f"{k} {vv:.3f}" for k, vv in ev.items() if not isinstance(vv, dict)))
    res = dict(config=cfg, commit=open("GIT_COMMIT").read().strip() if os.path.exists("GIT_COMMIT") else None,
               seconds=time.time() - t0, metrics={})
    res["metrics"]["val"] = evaluate(nca, theta, VA, VAf, cfg, dev)
    if HO is not None:
        res["metrics"]["held"] = evaluate(nca, theta, HO, HOf, cfg, dev)
    if cfg["damage"] > 0:
        res["metrics"]["val_damage"] = evaluate(nca, theta, VA, VAf, cfg, dev, damage=True)
    for k, vv in res["metrics"].items():
        P_(f"final {k}: " + " ".join(f"{a} {b:.3f}" for a, b in vv.items() if not isinstance(b, dict)))
    torch.save(dict(theta=theta, cfg=cfg), os.path.join(out, "theta.pt"))
    try:
        draw(nca, theta, VA, HO, cfg, dev, out)
    except Exception as e:
        P_(f"draw failed: {e!r}")
    tmp = os.path.join(out, f"result.json.tmp{os.getpid()}")
    json.dump(res, open(tmp, "w"), indent=1)
    os.replace(tmp, os.path.join(out, "result.json"))
    P_("done")


def evaluate(nca, theta, T_, Tf, cfg, dev, reps=2, damage=False):
    accs = []
    for rep in range(reps):
        gen = torch.Generator(device=dev).manual_seed(1000 + rep)
        with torch.no_grad():
            fit, st = rollout(nca, theta[None], T_, cfg, gen, damage_at=cfg["T"] // 2 if damage else None)
        accs.append(dict(fill=st["fill"], falls=st["falls"], outside=st["outside"],
                         scaffold=struct_stats(st["occ"], T_), fit=fit))
    res = {k: float(torch.stack([a[k] for a in accs]).mean()) for k in accs[0]}
    fill = torch.stack([a["fill"] for a in accs]).mean(0)
    res["per_family"] = {FAMS[f]: float(fill[Tf == f].mean()) for f in Tf.unique().tolist()}
    # reference: a perfect placement of the target (every target cell filled), then the same settling physics
    occ = T_.clone()
    s = support_span(occ, cfg["L"], None, cfg["H"] + cfg["W"])
    hid = torch.zeros(occ.shape[0], cfg["C"], *occ.shape[1:], device=dev)
    for _ in range(cfg["settle"]):
        s = support_span(occ, cfg["L"], s, cfg["support_iters"])
        occ, hid, _ = gravity(occ, hid, cfg["L"], s)
    res["ref_target_stands"] = float(((occ * T_).flatten(1).sum(1) / T_.flatten(1).sum(1)).mean())
    res["ref_pile_fill"] = float(((pile(T_, cfg["spares"], cfg["H"]) * T_).flatten(1).sum(1) / T_.flatten(1).sum(1)).mean())
    return res


def draw(nca, theta, VA, HO, cfg, dev, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image
    sets = [("val", VA)] + ([("held", HO)] if HO is not None else [])
    fig, axes = plt.subplots(len(sets) * 2, 6, figsize=(18, 3.2 * len(sets) * 2))
    gif = None
    for si, (name, T_) in enumerate(sets):
        idx = torch.linspace(0, len(T_) - 1, 6).long()
        tg = T_[idx]
        gen = torch.Generator(device=dev).manual_seed(7)
        with torch.no_grad():
            _, st = rollout(nca, theta[None], tg, cfg, gen, record=(si == 0))
        if si == 0:
            gif = (st["frames"], tg.cpu())
        for k in range(6):
            for r, (img, title) in enumerate([(tg[k].cpu(), "target"), (st["occ"][k].cpu(), "after build + settle")]):
                ax = axes[si * 2 + r, k]
                rgb = np.ones((cfg["H"], cfg["W"], 3))
                t, o = tg[k].cpu().numpy(), st["occ"][k].cpu().numpy() if r else None
                if r == 0:
                    rgb[t > 0] = [0.7, 0.7, 0.7]
                else:
                    rgb[(t > 0) & (o == 0)] = [0.88, 0.88, 0.88]
                    rgb[(t > 0) & (o > 0)] = [0.25, 0.35, 0.6]
                    rgb[(t == 0) & (o > 0)] = [0.85, 0.45, 0.2]
                ax.imshow(rgb, origin="lower", interpolation="nearest")
                ax.set_xticks([]); ax.set_yticks([])
                if r:
                    ax.set_title(f"{name}: fill {float(st['fill'][k]):.2f}", fontsize=8)
    plt.suptitle(f"{cfg['name']}: blue = target cell filled, orange = bot outside the target (spare or support), grey = missing", fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(out, "samples.png"), dpi=70)
    plt.close(fig)
    frames, tg = gif
    imgs = []
    for fr in frames:
        fig, axs = plt.subplots(1, 3, figsize=(12, 2.6))
        for k in range(3):
            rgb = np.ones((cfg["H"], cfg["W"], 3))
            t, o = tg[k].numpy(), fr[k].numpy()
            rgb[(t > 0) & (o == 0)] = [0.88, 0.88, 0.88]
            rgb[(t > 0) & (o > 0)] = [0.25, 0.35, 0.6]
            rgb[(t == 0) & (o > 0)] = [0.85, 0.45, 0.2]
            axs[k].imshow(rgb, origin="lower", interpolation="nearest"); axs[k].set_xticks([]); axs[k].set_yticks([])
        fig.canvas.draw()
        imgs.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[..., :3]))
        plt.close(fig)
    imgs[0].save(os.path.join(out, "traj.gif"), save_all=True, append_images=imgs[1:], duration=70, loop=0)


if __name__ == "__main__":
    main(sys.argv[1])
