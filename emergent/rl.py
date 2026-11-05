"""PPO for the same swarm (no gradients through the physics): exploration instead of backprop.

The actor is the same local controller, now stochastic (Gaussian on the raw velocity/force output and the grip
logit); every bot samples its own action. The critic sees the whole swarm (mean of bot embeddings), used only in
training. Reward per control step = decrease of the shape loss (shared by all bots of an episode), plus at the end
-hold_w * shape loss after the passive hold (motors off, grips kept).

    python rl.py configs/<name>.json     # same outputs as train.py (result.json, samples.png, traj.gif, ema.pt)
"""
import copy
import json
import math
import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn

import train
from train import Data, shape_loss, evaluate, draw, map_features
from sim import Physics, pile
from policy import Policy

RL_DEF = dict(envs=32, epochs=4, mb=16, clip_eps=0.2, gamma=0.99, lam=0.95, ent_w=0.0, vf_w=0.5, log_std0=-0.5,
              updates=600, rl_lr=3e-4, rew_scale=1.0, per_bot=False)


class RLPolicy(Policy):
    def __init__(self, log_std0=-0.5, **kw):
        super().__init__(**kw)
        self.log_std = nn.Parameter(torch.full((3,), float(log_std0)))
        self.vhead = nn.Sequential(nn.Linear(self.M, self.M), nn.SiLU(), nn.Linear(self.M, 1))
        self.vhead_bot = nn.Sequential(nn.Linear(2 * self.M, self.M), nn.SiLU(), nn.Linear(self.M, 1))

    def raw(self, x, v, h, mapf, alive):
        """Returns the raw mean output o (B, N, 3), new memory, embedding e."""
        B, N, _ = x.shape
        vin = v if self.see_v else torch.zeros_like(v)
        y = x[..., 1:2]
        own = torch.cat([mapf, vin, torch.clamp(y, max=3.0) / 3.0, (y < 0.65).float(), h if self.use_h else torch.zeros_like(h)], -1)
        e = self.enc(own)
        d = x[:, None, :, :] - x[:, :, None, :]
        dist = torch.sqrt((d * d).sum(-1) + 1e-8)
        w = torch.clamp((self.r_sense - dist) / 0.5, 0, 1) * alive[:, :, None] * alive[:, None, :]
        w = w * (1 - torch.eye(N, device=x.device))
        dv = vin[:, None, :, :] - vin[:, :, None, :]
        geo = torch.cat([d / self.r_sense, dist[..., None] / self.r_sense, dv, (dist[..., None] < 1.15).float()], -1)
        cnt = w.sum(2, keepdim=True)
        for r in range(self.rounds):
            z = torch.nn.functional.silu(self.ea[r](e)[:, :, None, :] + self.eb[r](e)[:, None, :, :] + self.eg[r](geo))
            m = self.e2[r]((w[..., None] * z).sum(2) / (1 + cnt))
            e = self.norm[r](e + self.upd[r](torch.cat([e, m], -1)))
        hn = self.gru(e.reshape(B * N, -1), h.reshape(B * N, -1)).reshape(B, N, -1) if self.use_h else h
        o = self.head(torch.cat([e, hn], -1))
        return o, hn, e

    def act_from(self, o):
        a = self.out_scale * torch.tanh(o[..., :2])
        grip = torch.sigmoid(o[..., 2] + self.grip_bias)
        return a, grip

    def value(self, e, alive, per_bot=False):
        g = (e * alive[..., None]).sum(1) / alive.sum(1, keepdim=True).clamp(min=1)
        if per_bot:
            return self.vhead_bot(torch.cat([e, g[:, None].expand_as(e)], -1))[..., 0]   # (B, N)
        return self.vhead(g)[..., 0]


def main(path):
    cfg = dict(train.DEF)
    cfg.update(RL_DEF)
    cfg.update(json.load(open(path)))
    cfg.setdefault("name", os.path.splitext(os.path.basename(path))[0])
    dev = "cuda"
    torch.manual_seed(cfg["seed"])
    out = os.path.join("runs", cfg["name"])
    os.makedirs(out, exist_ok=True)
    log = open(os.path.join(out, "log.txt"), "a")

    def P(*a):
        s = " ".join(str(x) for x in a)
        print(s, flush=True)
        log.write(s + "\n"); log.flush()

    data = Data(cfg["data"], dev, cfg["families"])
    mk = lambda: Physics(g=1.0, f_bond=cfg["f_bond"], f_act=cfg["f_act"], mu=cfg["mu"], reaction=cfg["reaction"], W=data.W,
                         k_c=cfg["k_c"], r_bond=cfg["r_bond"], r_hold=cfg["r_hold"], r_touch=cfg["r_touch"],
                         ctrl=cfg["ctrl"], kp=cfg["kp"], air=cfg["air"], k_b=cfg["k_b"], dt=cfg["dt"],
                         c_c=cfg["c_c"], c_b=cfg["c_b"])
    ph, ph_eval = mk(), mk()
    pol = RLPolicy(log_std0=cfg["log_std0"], H=cfg["H"], M=cfg["M"], rounds=cfg["rounds"], r_sense=cfg["r_sense"],
                   f_act=cfg["f_act"], use_h=cfg["use_h"], see_v=cfg["see_v"], grip_bias=cfg["grip_bias"],
                   out_scale=cfg["v_max"] if cfg["ctrl"] == "velocity" else None).to(dev)
    if cfg["compile"]:
        ph.step = torch.compile(ph.step, dynamic=False)
        ph_eval.step = torch.compile(ph_eval.step, dynamic=False)
    opt = torch.optim.Adam(pol.parameters(), lr=cfg["rl_lr"])
    start = 0
    ck = os.path.join(out, "ckpt.pt")
    if os.path.exists(ck):
        s = torch.load(ck, map_location=dev)
        pol.load_state_dict(s["pol"]); opt.load_state_dict(s["opt"]); start = s["it"]
        P(f"resumed at {start}")
    P(json.dumps(cfg))
    TR = data.sets["train"]
    gen = torch.Generator().manual_seed(cfg["seed"] + 7919 * start)
    E, T, K, N = cfg["envs"], cfg["T"], cfg["K"], data.N
    ret_rms = [1.0]
    t0 = time.time()
    for upd in range(start, cfg["updates"]):
        # ---------------- rollout
        idx = torch.randint(0, len(TR["fam"]), (E,), generator=gen).to(dev)
        maps, Q = TR["maps"][idx].float(), TR["pts"][idx]
        x = pile(E, N, W=data.W, gen=gen).to(dev)
        v = torch.zeros_like(x)
        h = torch.zeros(E, N, cfg["H"], device=dev)
        alive = torch.ones(E, N, device=dev)
        gs = torch.tensor(float(cfg["g"]), device=dev)
        buf = dict(x=[], v=[], h=[], u=[], logp=[], val=[], rew=[])
        PB = cfg["per_bot"]
        lossf = (lambda x_: shape_loss_per_bot(x_, alive, Q, maps, data.W, cfg["out_w"])) if PB else \
                (lambda x_: shape_loss_per_env(x_, alive, Q, maps, data.W, cfg["out_w"]))
        with torch.no_grad():
            L_prev = lossf(x)
            for t in range(T):
                mf = map_features(maps, x, data.W)
                o, hn, e = pol.raw(x, v, h, mf, alive)
                std = pol.log_std.exp()
                u = o + std * torch.randn_like(o)
                logp = (-0.5 * ((u - o) / std) ** 2 - pol.log_std - 0.5 * math.log(2 * math.pi)).sum(-1)  # per bot (E, N)
                val = pol.value(e, alive, PB)
                buf["x"].append(x); buf["v"].append(v); buf["h"].append(h); buf["u"].append(u)
                buf["logp"].append(logp); buf["val"].append(val)
                a, grip = pol.act_from(u)
                for _ in range(K):
                    x, v = ph.step(x, v, a, grip, alive, gs, 1.0)
                h = hn
                L = lossf(x)
                buf["rew"].append((L_prev - L) * cfg["rew_scale"])
                L_prev = L
            xb = x
            for _ in range(cfg["hold"] // K):
                for _ in range(K):
                    x, v = ph.step(x, v, None, grip, alive, gs, 1.0)
            Lh_ = lossf(x)
            buf["rew"][-1] = buf["rew"][-1] - cfg["hold_w"] * (Lh_ - lossf(xb)) * cfg["rew_scale"]
            Lh = shape_loss_per_env(x, alive, Q, maps, data.W, cfg["out_w"])
            rew = torch.stack(buf["rew"])          # (T, E) or (T, E, N)
            val = torch.stack(buf["val"])
            # reward normalisation by a running scale of returns
            sd_now = float(rew.sum(0).std().clamp(min=1e-3))
            ret_rms[0] = sd_now if upd == start else 0.99 * ret_rms[0] + 0.01 * sd_now
            rn = rew / ret_rms[0]
            adv = torch.zeros_like(rn)
            last = torch.zeros_like(rn[0])
            for t in reversed(range(T)):
                nv = val[t + 1] if t + 1 < T else torch.zeros_like(rn[0])
                delta = rn[t] + cfg["gamma"] * nv - val[t]
                last = delta + cfg["gamma"] * cfg["lam"] * last
                adv[t] = last
            ret = adv + val
        # ---------------- PPO update over (t, env) frames
        X, V, Hh, U = (torch.stack(buf[k]) for k in ("x", "v", "h", "u"))      # (T, E, ...)
        LP = torch.stack(buf["logp"])
        A = (adv - adv.mean()) / (adv.std() + 1e-8)
        frames = [(t, e) for t in range(T) for e in range(E)]
        stats = []
        for ep in range(cfg["epochs"]):
            perm = torch.randperm(len(frames), generator=gen)
            for i in range(0, len(frames), cfg["mb"]):
                sel = perm[i:i + cfg["mb"]]
                ts = torch.tensor([frames[j][0] for j in sel], device=dev)
                es = torch.tensor([frames[j][1] for j in sel], device=dev)
                xs, vs, hs, us = X[ts, es], V[ts, es], Hh[ts, es], U[ts, es]
                ms = maps[es]
                al = torch.ones(len(sel), N, device=dev)
                o, _, e = pol.raw(xs, vs, hs, map_features(ms, xs, data.W), al)
                lp = (-0.5 * ((us - o) / pol.log_std.exp()) ** 2 - pol.log_std - 0.5 * math.log(2 * math.pi)).sum(-1)
                ratio = torch.exp(torch.clamp(lp - LP[ts, es], -20, 20))      # per bot (mb, N)
                a_ = A[ts, es] if PB else A[ts, es][:, None]                     # per-bot or shared team advantage
                pl = -torch.min(ratio * a_, torch.clamp(ratio, 1 - cfg["clip_eps"], 1 + cfg["clip_eps"]) * a_).mean()
                vl = ((pol.value(e, al, PB) - ret[ts, es]) ** 2).mean()
                ent = pol.log_std.sum()
                loss = pl + cfg["vf_w"] * vl - cfg["ent_w"] * ent + cfg["pre_w"] * (o[..., :2] ** 2).mean()
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(pol.parameters(), 1.0)
                opt.step()
                stats.append((pl.item(), vl.item(), (ratio - 1).abs().mean().item()))
        if upd % 10 == 0:
            s = np.mean(stats, 0)
            P(f"upd {upd} build_loss {shape_loss_per_env(xb, alive, Q, maps, data.W, cfg['out_w']).mean().item():.2f} "
              f"hold_loss {Lh.mean().item():.2f} ret {rew.sum(0).mean().item():.3f} pl {s[0]:.3f} vl {s[1]:.3f} "
              f"|r-1| {s[2]:.3f} std {pol.log_std.exp().tolist()} grip {grip.mean().item():.2f} {(time.time() - t0) / (upd - start + 1):.1f}s/upd")
        if (upd + 1) % 25 == 0 or upd + 1 == cfg["updates"]:
            tmp = ck + f".tmp{os.getpid()}"
            torch.save(dict(pol=pol.state_dict(), opt=opt.state_dict(), it=upd + 1), tmp)
            os.replace(tmp, ck)
        if (upd + 1) % cfg["eval_every"] == 0 and upd + 1 < cfg["updates"]:
            m, _ = evaluate(DetWrap(pol), ph_eval, data, cfg, "val", n=64, dev=dev)
            P(f"eval upd {upd + 1}: " + " ".join(f"{k} {v:.3f}" for k, v in m.items() if not k.endswith("_ci")))
    torch.save(dict(ema=pol.state_dict(), cfg=cfg), os.path.join(out, "ema.pt"))
    res = dict(config=cfg, commit=open("GIT_COMMIT").read().strip() if os.path.exists("GIT_COMMIT") else None,
               train_seconds=time.time() - t0, metrics={}, per_family={})
    det = DetWrap(pol)
    for split in ("val", "held"):
        m, pf = evaluate(det, ph_eval, data, cfg, split, dev=dev, with_baselines=True)
        res["metrics"][split] = m
        res["per_family"][split] = pf
        P(f"final {split}: " + " ".join(f"{k} {v:.3f}" for k, v in m.items() if not k.endswith("_ci")))
    try:
        draw(det, ph_eval, data, cfg, os.path.join(out, "samples.png"), os.path.join(out, "traj.gif"), dev)
    except Exception as e:
        P(f"draw failed: {e!r}")
    tmp = os.path.join(out, f"result.json.tmp{os.getpid()}")
    json.dump(res, open(tmp, "w"), indent=1)
    os.replace(tmp, os.path.join(out, "result.json"))
    P("done")


class DetWrap(nn.Module):
    """Deterministic (mean-action) view of the RL policy with the train.py controller interface."""
    def __init__(self, pol):
        super().__init__()
        self.pol = pol
        self.H = pol.H

    def forward(self, x, v, h, mf, alive):
        o, hn, _ = self.pol.raw(x, v, h, mf, alive)
        a, grip = self.pol.act_from(o)
        return a, grip, hn, (o[..., :2] ** 2).mean()


def shape_loss_per_bot(x, alive, Q, maps, W, out_w):
    """The Chamfer shape loss split exactly into per-bot parts: a bot's own squared distance to the shape, plus the
    squared distances of the target points it is the nearest bot to, plus its outside penalty. (B, N)."""
    D = torch.cdist(x, Q) ** 2                          # (B, N, M)
    big = 1e4 * (1 - alive)[:, :, None]
    own = D.min(2).values * alive
    dq, iq = (D + big).min(1)                           # nearest bot per target point
    cov = torch.zeros_like(own).scatter_add_(1, iq, dq)
    sd = map_features(maps[:, :1], x, W)[..., 0] * 4.0
    out = (torch.relu(sd) ** 2) * alive
    return own + cov + out_w * out


def shape_loss_per_env(x, alive, Q, maps, W, out_w):
    D = torch.cdist(x, Q) ** 2
    big = 1e4 * (1 - alive)[:, :, None]
    t1 = ((D.min(2).values) * alive).sum(1) / alive.sum(1).clamp(min=1)
    t2 = (D + big).min(1).values.mean(1)
    sd = map_features(maps[:, :1], x, W)[..., 0] * 4.0
    out = ((torch.relu(sd) ** 2) * alive).sum(1) / alive.sum(1).clamp(min=1)
    return t1 + t2 + out_w * out


if __name__ == "__main__":
    main(sys.argv[1])
