"""Look at a running lattice job from its checkpoint: metrics (with the 'propped' scaffolding test) and a picture.

propped = target cells that are filled at the end but would NOT be held if every bot outside the target were removed.
Nonzero propped means the swarm is holding part of the target up with bots outside it (scaffolding, or a column of
spares under an overhang). It is measured per shape, so it cannot be faked by averaging across shapes.

usage: python lat/peek.py runs/<name> [runs/<name> ...]   (writes runs/<name>/peek.png and prints one line each)
"""
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import es  # noqa: E402
from latworld import support_span  # noqa: E402


def propped(occ, tgt, L, H, W):
    inside = occ * tgt
    s = support_span(inside, L, None, H + W)
    return (inside * (s > L).float()).flatten(1).sum(1)


def main(dirs):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    for d in dirs:
        ck = os.path.join(d, "theta.pt") if os.path.exists(os.path.join(d, "theta.pt")) else os.path.join(d, "ckpt.pt")
        if not os.path.exists(ck):
            print(d, "no checkpoint"); continue
        s = torch.load(ck, map_location=dev)
        cfg = dict(es.DEF)
        logp = os.path.join(d, "log.txt")
        cfg.update(json.loads([l for l in open(logp) if l.startswith("{")][-1]))
        H, W = cfg["H"], cfg["W"]
        train_f = cfg["families"] or [f for f in es.FAMS if f not in cfg["held_out"]]
        VA, VAf = es.bank(train_f, cfg["n_eval"], H, W, 2)
        HO, HOf = es.bank(cfg["held_out"], cfg["n_eval"], H, W, 3)
        VA, HO = VA.to(dev), HO.to(dev)
        nca = es.NCA(cfg["C"], cfg["M"])
        theta = s["theta"]
        line = [f"{d} gen {s.get('gen', 'final')}"]
        if os.environ.get("PEEK_INIT"):     # control: the untrained controller (uniform over the allowed moves)
            theta = nca.init(torch.Generator().manual_seed(0)).to(dev)
            line = [f"{d} UNTRAINED (uniform random moves)"]
        for name, T_ in [("val", VA), ("held", HO)]:
            g = torch.Generator(device=dev).manual_seed(1000)
            with torch.no_grad():
                _, st = es.rollout(nca, theta[None], T_, cfg, g)
            pr = propped(st["occ"], T_, cfg["L"], H, W)
            nt = T_.flatten(1).sum(1)
            line.append(f"{name}: fill {st['fill'].mean():.3f} propped/target {(pr / nt).mean():.3f} "
                        f"shapes_with_propped {(pr > 0).float().mean():.2f} outside {st['outside'].mean():.1f} "
                        f"falls {st['falls'].mean():.0f}")
        print(" | ".join(line), flush=True)
        if os.environ.get("PEEK_INIT"):
            continue
        es.draw(nca, theta, VA, HO, cfg, dev, d)
        os.replace(os.path.join(d, "samples.png"), os.path.join(d, "peek.png")) if not os.path.exists(
            os.path.join(d, "theta.pt")) else None
        if not os.path.exists(os.path.join(d, "theta.pt")):
            os.replace(os.path.join(d, "traj.gif"), os.path.join(d, "peek.gif"))


if __name__ == "__main__":
    main(sys.argv[1:])
