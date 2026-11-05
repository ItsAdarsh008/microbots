"""Time series of a trained lattice controller on the validation set: does it build falsework and then remove it?

Per step (mean over validation shapes): fill, scaffold (bots outside the target under a target overhang),
site_out (bots outside the target in the build site), cured fraction of filled target cells.
usage: python lat/traj_stats.py runs/<name> [every]
"""
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import es  # noqa: E402
from latworld import support_span, valid_moves, apply_moves, gravity, solid_at  # noqa: E402


def main(d, every=20):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ck = os.path.join(d, "theta.pt") if os.path.exists(os.path.join(d, "theta.pt")) else os.path.join(d, "ckpt.pt")
    s_ = torch.load(ck, map_location=dev)
    cfg = dict(es.DEF)
    cfg.update(json.loads([l for l in open(os.path.join(d, "log.txt")) if l.startswith("{")][-1]))
    H, W = cfg["H"], cfg["W"]
    train_f = cfg["families"] or [f for f in es.FAMS if f not in cfg["held_out"] and f != "chasm"]
    tgt, _ = es.bank(train_f, cfg["n_eval"], H, W, 2)
    tgt = tgt.to(dev)
    nca = es.make_nca(cfg)
    theta = s_["theta"][None]
    gen = torch.Generator(device=dev).manual_seed(1000)
    E = tgt.shape[0]
    import latworld
    latworld.set_floor(es.floor_mask(tgt) if cfg.get("chasm") else None)
    occ = es.pile(tgt, cfg["spares"], H)
    hid = torch.zeros(E, cfg["C"], H, W, device=dev)
    floor = torch.zeros(E, 1, H, W, device=dev); floor[:, :, 0, :] = 1
    K = cfg.get("cure", 0)
    age = torch.zeros_like(occ); inc = None
    s = support_span(occ, cfg["L"], None, H + W)
    nt = tgt.flatten(1).sum(1)
    cols = (torch.arange(W, device=dev) >= 14).float()[None, None, :]
    print(f"{d}: step fill scaffold site_out cured_frac", flush=True)
    with torch.no_grad():
        for t in range(cfg["T"]):
            held = occ * (s <= cfg["L"]).float()
            extra = [(age >= K).float()[:, None] * occ[:, None]] if K > 0 else []
            inp = torch.cat([occ[:, None], tgt[:, None], held[:, None], floor] + extra + [hid], 1)
            out = nca.forward(theta, inp)
            logits = out[:, :9] / cfg["temp"]
            hid_new = torch.tanh(out[:, 9:]) * occ[:, None]
            mask = valid_moves(occ, held)
            g = -torch.log(-torch.log(torch.rand(logits.shape, device=dev, generator=gen).clamp(1e-9, 1 - 1e-9)))
            act = torch.where(mask > 0, logits + g, torch.full_like(logits, -1e9)).argmax(1)
            prio = torch.rand(occ.shape, device=dev, generator=gen)
            occ_b = occ
            occ, hid, mv = apply_moves(occ, hid_new, act, prio)
            if K > 0:
                age = (age + 1) * occ_b * (1 - mv)
                if cfg.get("cure_below"):
                    age = torch.where((solid_at(occ, -1, 0) > 0) | (age > K), age, torch.zeros_like(age))
                inc = 1.0 - (age >= K).float()
            s = support_span(occ, cfg["L"], s, cfg["support_iters"], inc)
            if K > 0:
                age = age * (s <= cfg["L"]).float()
            occ, hid, _ = gravity(occ, hid, cfg["L"], s)
            age = age * occ
            if t % every == 0 or t == cfg["T"] - 1:
                fill = ((occ * tgt).flatten(1).sum(1) / nt).mean()
                scaf = es.struct_stats(occ, tgt).mean()
                site = (occ * (1 - tgt) * cols).flatten(1).sum(1).mean()
                cured = (((age >= K).float() * occ * tgt).flatten(1).sum(1) / nt).mean() if K > 0 else 0.0
                print(f"  {t:4d} {fill:.3f} {scaf:6.1f} {site:6.1f} {float(cured):.3f}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 20)
