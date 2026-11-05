"""Gap-width sweep for chasm runs: is crossing a general local procedure or fitted to the trained widths?

    python lat/chasm_gen.py runs/lat6_chasm_cure_warm [more runs]      # gaps 3..20 (training used 3..12 by default)
Prints, per gap width: deck fill, far-tower fill, and whole-target fill (mean over 32 targets x 2 seeds).
Writes runs/<name>/chasm_gen_T<T>.json. CG_T=600 overrides the episode length.
"""
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import es  # noqa: E402
from gallery import run_cfg  # noqa: E402


def targets(gap, n, H, W, seed):
    rng = np.random.default_rng(seed)
    out = []
    while len(out) < n:
        t = es.chasm_target(rng, H, W, gap=gap)
        if t is not None:
            out.append(t)
    return torch.tensor(np.array(out), dtype=torch.float32)


def main(args):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    for d in args:
        cfg = run_cfg(d)
        if os.environ.get("CG_T"):                     # override the episode length (is a failure a time limit?)
            cfg["T"] = int(os.environ["CG_T"])
        ck = os.path.join(d, "theta.pt") if os.path.exists(os.path.join(d, "theta.pt")) else os.path.join(d, "ckpt.pt")
        theta = torch.load(ck, map_location=dev)["theta"]
        nca = es.make_nca(cfg)
        H, W = cfg["H"], cfg["W"]
        res = {}
        print(os.path.basename(d.rstrip("/")), "trained gaps", cfg.get("chasm_gap", [3, 13]), flush=True)
        for gap in range(3, 21):
            tg = targets(gap, 32, H, W, 100 + gap).to(dev)
            deck = tg.clone(); deck[:, 1:] = 0
            tower = tg - deck
            r = []
            for rep in range(2):
                g = torch.Generator(device=dev).manual_seed(500 + rep)
                with torch.no_grad():
                    _, st = es.rollout(nca, theta[None], tg, cfg, g)
                occ = st["occ"]
                r.append([float(((occ * deck).flatten(1).sum(1) / deck.flatten(1).sum(1)).mean()),
                          float(((occ * tower).flatten(1).sum(1) / tower.flatten(1).sum(1)).mean()),
                          float(st["fill"].mean())])
            dk, tw, fl = np.mean(r, 0)
            res[gap] = dict(deck=dk, tower=tw, fill=fl)
            print(f"  gap {gap:2d}  deck {dk:.2f}  tower {tw:.2f}  fill {fl:.2f}", flush=True)
        json.dump(res, open(os.path.join(d, f"chasm_gen_T{cfg['T']}.json"), "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1:])
