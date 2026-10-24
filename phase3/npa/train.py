"""Train an NPA so that a seed blob grows into the lizard and heals itself.

The recipe is the paper's "growing" setup (configs/growing.yaml in the authors' release):
  * a pool of 512 swarms, each a snapshot of an earlier rollout; every iteration pulls a batch
    from it, runs 32-96 steps, scores the result against the target, and puts it back
  * every 16th iteration the first swarm in the batch is replaced with a fresh seed
  * damage: each swarm gets a random particle's neighbourhood wiped before it runs, so
    the rule learns to regrow what is missing
  * loss: how far the splatted density and colour are from the lizard's (target.py), plus
    small penalties for large moves and for states or positions leaving [-1, 1]
  * each parameter tensor's gradient is normalised to unit length, then Adam, with the
    learning rate cut by 0.3 at 20/40/60/80% of the run

    python phase3/npa/train.py --particles 1024 --iters 3000 --out models/lizard_1024.npz

The full recipe is 30,000 iterations at batch 8 and 4,096 particles, which needs a GPU. On a CPU,
lower --batch and --iters, or stop early with --minutes. Checkpoints are written as it goes.
"""

import argparse
import json
import math
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

import npa
import target as tgt
import weights

HERE = Path(__file__).resolve().parent

POOL_SIZE = 512
SEED_EVERY = 16
STEP_RANGE = (32, 96)      # the number of steps per iteration is drawn from this range
LR = 5e-4
LR_DROP = 0.3              # multiply the learning rate by this at each milestone
LR_MILESTONES = (0.2, 0.4, 0.6, 0.8)   # fractions of the run
ADAM = (0.9, 0.999, 1e-8)


def scaled_eps(n):
    """The paper's rule (appendix C.2): keep the neighbour count fixed as the particle count changes."""
    return 0.2 * math.sqrt(1024 / n)


def numpy_seed(rng, count, n, radius):
    """`count` fresh seeds: uniform discs of n particles with zero state."""
    r = np.sqrt(rng.random((count, n))) * radius
    t = rng.random((count, n)) * 2 * np.pi
    x = np.stack([r * np.cos(t), r * np.sin(t)], axis=-1).astype(np.float32)
    return x, np.zeros((count, n, npa.STATE_DIMS), np.float32)


def numpy_damage(rng, x, s, radius):
    """Wipe the state around one random particle in every swarm of the batch."""
    for b in range(x.shape[0]):
        centre = x[b, rng.integers(x.shape[1])]
        s[b][np.sum((x[b] - centre) ** 2, axis=-1) < radius * radius] = 0.0


# =============================================================================
# The training step
# =============================================================================
def make_train_step(loss_fn, batch, n, eps, eps0, block):
    """Build train_step(params, opt, key, x, s, steps, lr) -> new params, new opt state, new x, s, loss, parts."""
    t_max = STEP_RANGE[1]

    def rollout_loss(params, key, x, s, steps):
        def run_one(params, k, carry):
            x, s, moves = carry
            update = jax.random.bernoulli(k, npa.UPDATE_PROB, (batch, n, 1)).astype(x.dtype)
            x, s, move = jax.vmap(npa.step, in_axes=(None, 0, 0, 0, None, None, None))(
                params, x, s, update, eps, eps0, block)
            return x, s, moves + jnp.mean(move)

        def body(carry, inp):
            t, k = inp
            # steps past `steps` are skipped, so one compiled program serves every rollout length.
            # checkpoint: recompute each step's internals on the backward pass instead of storing
            # them, which is what keeps memory small enough for 1,000+ particles on a CPU.
            carry = lax.cond(t < steps, jax.checkpoint(run_one), lambda p, k, c: c, params, k, carry)
            return carry, None

        keys = jax.random.split(key, t_max)
        (x, s, moves), _ = lax.scan(body, (x, s, jnp.zeros(())), (jnp.arange(t_max), keys))
        loss, parts = loss_fn(x, s, moves)
        return loss, (x, s, parts)

    def train_step(params, opt, key, x, s, steps, lr):
        (loss, (x, s, parts)), grads = jax.value_and_grad(rollout_loss, has_aux=True)(params, key, x, s, steps)
        # normalise each tensor's gradient to unit length (standard for NCAs; makes lr scale-free)
        grads = jax.tree_util.tree_map(
            lambda g: (lambda g: g / (jnp.linalg.norm(g) + 1e-8))(jnp.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0)),
            grads)
        b1, b2, e = ADAM
        t = opt["t"] + 1
        m = jax.tree_util.tree_map(lambda m, g: b1 * m + (1 - b1) * g, opt["m"], grads)
        v = jax.tree_util.tree_map(lambda v, g: b2 * v + (1 - b2) * g * g, opt["v"], grads)
        params = jax.tree_util.tree_map(
            lambda p, m, v: p - lr * (m / (1 - b1**t)) / (jnp.sqrt(v / (1 - b2**t)) + e), params, m, v)
        return params, {"m": m, "v": v, "t": t}, x, s, loss, parts

    return jax.jit(train_step)


def init_opt(params):
    zeros = jax.tree_util.tree_map(jnp.zeros_like, params)
    return {"m": zeros, "v": zeros, "t": jnp.zeros((), jnp.int32)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--particles", type=int, default=1024)
    ap.add_argument("--eps", type=float, default=None, help="neighbour radius (default: 0.2*sqrt(1024/N))")
    ap.add_argument("--iters", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--grid", type=int, default=128, help="side of the picture the loss compares (paper: 256)")
    ap.add_argument("--seed-radius", type=float, default=None, help="radius of the seed disc (default: 2*eps)")
    ap.add_argument("--minutes", type=float, default=None, help="stop after this long")
    ap.add_argument("--init", default=None, help="start from an .npz of weights instead of a random init")
    ap.add_argument("--out", default=str(HERE / "models" / "lizard.npz"))
    ap.add_argument("--rng", type=int, default=0)
    args = ap.parse_args()

    n, batch = args.particles, args.batch
    eps = args.eps if args.eps is not None else scaled_eps(n)
    eps0 = eps                                   # the radius this model is trained at
    seed_radius = args.seed_radius if args.seed_radius is not None else 2 * eps
    block = 512 if n > 2048 else None
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    print(f"{n} particles, eps={eps:.4f}, batch {batch}, {args.iters} iterations, loss grid {args.grid}", flush=True)

    target = tgt.load_target()
    loss_fn = tgt.make_loss(target, grid=args.grid)
    train_step = make_train_step(loss_fn, batch, n, eps, eps0, block)

    if args.init:
        params = {k: jnp.asarray(v) for k, v in weights.load_npz(args.init)[0].items()}
    else:
        params = npa.init_params(jax.random.PRNGKey(args.rng))
    opt = init_opt(params)

    rng = np.random.default_rng(args.rng)
    pool_x, pool_s = numpy_seed(rng, POOL_SIZE, n, seed_radius)
    milestones = {int(f * args.iters) for f in LR_MILESTONES}
    key = jax.random.PRNGKey(args.rng + 1)
    lr, log, t_start = LR, [], time.time()
    history = out.with_suffix(".log.json")

    def save():
        weights.save_npz(out, params, eps, eps0)
        history.write_text(json.dumps({"particles": n, "eps": eps, "batch": batch, "grid": args.grid,
                                       "iterations_done": len(log), "log": log}, indent=1))

    for it in range(args.iters):
        if it in milestones:
            lr *= LR_DROP
        idx = rng.permutation(POOL_SIZE)[:batch]
        x, s = pool_x[idx].copy(), pool_s[idx].copy()
        if it % SEED_EVERY == 0:
            x[:1], s[:1] = numpy_seed(rng, 1, n, seed_radius)
        numpy_damage(rng, x, s, eps)

        steps = int(rng.integers(*STEP_RANGE))
        key, k = jax.random.split(key)
        t0 = time.time()
        params, opt, x, s, loss, parts = train_step(params, opt, k, jnp.asarray(x), jnp.asarray(s), steps, lr)
        loss = float(loss)
        if not math.isfinite(loss):
            raise SystemExit(f"loss became {loss} at iteration {it}; stopping (the last checkpoint is intact)")
        pool_x[idx], pool_s[idx] = np.asarray(x), np.asarray(s)
        log.append({"iter": it, "steps": steps, "loss": loss, "lr": lr, "seconds": time.time() - t0,
                    **{k: float(v) for k, v in parts.items()}})
        if it % 10 == 0 or it == args.iters - 1:
            print(f"it {it:5d}  steps {steps:2d}  loss {loss:8.4f}  density {log[-1]['density']:.4f}  "
                  f"colour {log[-1]['colour']:.4f}  {log[-1]['seconds']:.1f}s/it", flush=True)
        if it % 25 == 0:
            save()
        if args.minutes is not None and time.time() - t_start > args.minutes * 60:
            print(f"stopping after {args.minutes} minutes ({it + 1} iterations)")
            break
    save()
    print("saved", out)


if __name__ == "__main__":
    main()
