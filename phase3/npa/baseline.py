"""Grow the lizard from a seed blob, wipe its head, and watch it regrow.

This is the baseline run: the paper's published setup (4,096 particles, EPS = 0.1) with the
authors' pretrained lizard weights, run through this repo's own NPA code. If the lizard
appears and heals, the reimplementation matches the paper.

    python phase3/npa/baseline.py path/to/lizard.pth            # 4,096 particles
    python phase3/npa/baseline.py ours.npz --particles 1024     # one of our own models

Writes results/<tag>.png (a strip of snapshots) and results/<tag>.json (the numbers).
It is slow on a CPU (about 3 s per step at 4,096 particles).
"""

import argparse
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

import npa
import target as tgt
import viz
import weights

HERE = Path(__file__).resolve().parent
GROW_STEPS = 160            # a trained rollout is 32 to 96 steps; this runs well past that
HEAL_STEPS = 160
SNAPSHOTS = (0, 16, 32, 64, 96, 160)
HEAD_CUT_RADIUS = 0.30      # in world units; training damage only uses EPS (0.1)


def advance(params, key, x, s, steps, eps, eps0, block):
    run = jax.jit(lambda p, k, x, s: npa.rollout(p, k, x, s, steps, eps, eps0, block), static_argnums=())
    return run(params, key, x, s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint", help="the authors' .pth, or one of our .npz files")
    ap.add_argument("--particles", type=int, default=4096)
    ap.add_argument("--eps", type=float, default=None, help="neighbour radius (default: 0.2*sqrt(1024/N))")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.checkpoint.endswith(".pth"):
        params, eps0, _ = weights.load_pth(args.checkpoint)
        eps_default = eps0
    else:
        params, eps_default, eps0 = weights.load_npz(args.checkpoint)
    params = {k: jnp.asarray(v) for k, v in params.items()}
    n = args.particles
    eps = args.eps if args.eps is not None else eps_default
    block = 512 if n > 2048 else None
    tag = args.tag or f"baseline_N{n}"
    print(f"{tag}: {n} particles, eps={eps}, eps0={eps0}")

    target = tgt.load_target()
    evaluate = jax.jit(tgt.make_eval(target))
    x, s = npa.seed(jax.random.PRNGKey(args.seed), n)
    key = jax.random.PRNGKey(args.seed + 1)

    log, frames, t_step = [], [], []

    def record(phase, step, x, s, mark=None):
        m = {k: float(v) for k, v in evaluate(x, s).items()}
        log.append({"phase": phase, "step": step, **m})
        frames.append((phase, step, viz.draw(x, s, mark=mark)))
        print(f"  {phase:5s} step {step:3d}  density {m['density']:.3f}  colour {m['colour']:.3f}  iou {m['iou']:.3f}", flush=True)

    # --- grow from the seed ---
    record("grow", 0, x, s)
    done = 0
    for target_step in SNAPSHOTS[1:]:
        key, k = jax.random.split(key)
        t0 = time.time()
        x, s = advance(params, k, x, s, target_step - done, eps, eps0, block)
        jax.block_until_ready(x)
        t_step.append((time.time() - t0) / (target_step - done))
        done = target_step
        record("grow", done, x, s)
    grown = (x, s)

    # --- wipe the head, then heal ---
    head = int(jnp.argmax(-(x[:, 0]) + x[:, 1]))     # the top-left-most particle: the lizard's snout
    centre = x[head]
    s = npa.damage(x, s, centre, HEAD_CUT_RADIUS)
    mark = (float(centre[0]), float(centre[1]), HEAD_CUT_RADIUS)
    record("heal", 0, x, s, mark=mark)
    done = 0
    for target_step in SNAPSHOTS[1:]:
        key, k = jax.random.split(key)
        x, s = advance(params, k, x, s, target_step - done, eps, eps0, block)
        jax.block_until_ready(x)
        done = target_step
        record("heal", done, x, s)

    out = HERE / "results"
    out.mkdir(exist_ok=True)
    grow = [f for p, st, f in frames if p == "grow"]
    heal = [f for p, st, f in frames if p == "heal"]
    viz.strip(grow).save(out / f"{tag}_grow.png")
    viz.strip(heal).save(out / f"{tag}_heal.png")
    meta = {"particles": n, "eps": eps, "eps0": eps0, "seconds_per_step": float(np.mean(t_step)),
            "head_cut_radius": HEAD_CUT_RADIUS, "log": log}
    (out / f"{tag}.json").write_text(json.dumps(meta, indent=2))
    print("wrote", out / f"{tag}_grow.png", out / f"{tag}_heal.png")


if __name__ == "__main__":
    main()
