"""Shape-agnostic training (selected by train.mode == "agnostic" in lab/run.py; scored by lab/evaluate_agnostic.py).

Each iteration draws fresh swarms: a mix of procedural shapes (seeds >= 1000) and the training emoji, sampled into n
particles with the colour pattern in the last three state channels. The reference picture is rendered from the START
swarm. The cell then runs WARM undamaged steps, a damage is applied, it runs RECOVER more steps (backprop through all
of it, one checkpoint per step, variable lengths through lax.cond), and the loss compares the final swarm with the
reference: density + 5 colour of target.make_loss, plus its regularisers with the same weights. Nothing about a shape
reaches the weights except through the swarm itself.

Extra train options (all optional; the rest are run.py's: iters, batch, lr, grid, grad, noise_pos, noise_state,
max_minutes, damage):
    proc_fraction   share of procedural shapes in a batch (default 0.7); the rest are train emoji
    warm            [lo, hi] undamaged steps before the damage (default [16, 48])
    recover         [lo, hi] steps after the damage (default [32, 96])
    warm_loss       weight of an extra loss at the end of the warm-up, so settling without damage is trained too (default 0.0)
    damage          default: all six kinds, each with probability 0.2, at the evaluator's level ranges
    pool            not implemented (the reference must stay paired with its swarm); must be false

Same optimiser, per-tensor gradient normalisation, learning-rate schedule and NaN recovery as run.py (shared constants).
"""

import math
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
import agnostic  # noqa: E402
import cells  # noqa: E402
import npa  # noqa: E402
import run  # noqa: E402
import target as tgt  # noqa: E402

AGNOSTIC_DEFAULTS = {
    "proc_fraction": 0.7, "warm": [16, 48], "recover": [32, 96], "warm_loss": 0.0, "pool": False,
    "damage": [{"kind": "wipe", "level": [0.15, 0.65], "p": 0.2},
               {"kind": "cut", "level": [0.1, 0.3], "p": 0.2},
               {"kind": "drag", "level": [0.15, 0.8], "p": 0.2},
               {"kind": "clump", "level": [0.2, 0.7], "p": 0.2},
               {"kind": "shuffle", "level": [0.15, 0.65], "p": 0.2},
               {"kind": "noise", "level": [0.1, 0.5], "p": 0.2}],
}


def make_loss(grid):
    """loss(x, s, sum_move, alive, ref, total) for a batch: like target.make_loss, but each sample has its own reference."""
    def loss(x, s, sum_move, alive, ref, total):
        img = jax.vmap(lambda x, s, a, t: agnostic.render(x, s, a, t, grid))(x, s, alive, total)
        den_diff = tgt.l1l2(img[..., 3:] - ref[..., 3:])
        w = jnp.exp(-jax.lax.stop_gradient(den_diff))       # colour only has to match where the density does
        col_loss = jnp.mean(tgt.l1l2(img[..., :3] - ref[..., :3]) * w)
        den_loss = jnp.mean(den_diff)
        splat_loss = tgt.DENSITY_WEIGHT * den_loss + tgt.COLOUR_WEIGHT * col_loss
        overflow = jnp.mean(jnp.abs(s - jnp.clip(s, -1.0, 1.0)))
        bound = jnp.mean(jnp.abs(x - jnp.clip(x, -1.0, 1.0)))
        total_loss = (tgt.SPLAT_LOSS_WEIGHT * splat_loss + tgt.DISPLACEMENT_WEIGHT * sum_move
                      + tgt.OVERFLOW_WEIGHT * overflow + tgt.BOUND_WEIGHT * bound)
        return total_loss, {"density": den_loss, "colour": col_loss, "splat": splat_loss,
                            "move": sum_move, "overflow": overflow, "bound": bound}
    return loss


def make_train_step(cell, loss_fn, damager, batch, n, eps, eps0, block, warm_max, rec_max, grad_mode,
                    noise_pos, noise_state, warm_loss):
    def rollout_loss(params, key, x, s, mem, ref, total, n_warm, n_rec):
        alive = jnp.ones((batch, n))

        def phase(carry, alive, steps, t_max, key):
            def run_one(params, alive, k, carry):
                x, s, mem, moves = carry
                ku, kx, ks = jax.random.split(k, 3)
                update = jax.random.bernoulli(ku, npa.UPDATE_PROB, (batch, n, 1)).astype(x.dtype) * alive[..., None]
                if noise_pos:
                    x = x + alive[..., None] * noise_pos * eps * jax.random.normal(kx, x.shape)
                if noise_state:
                    s = s + alive[..., None] * noise_state * jax.random.normal(ks, s.shape)
                x, s, mem, move = jax.vmap(cell.step, in_axes=(None, 0, 0, 0, 0, None, None, None, None, 0))(
                    params, x, s, mem, update, eps, eps0, block, n, alive)
                return x, s, mem, moves + jnp.mean(move)

            def body(carry, inp):
                t, k = inp
                return lax.cond(t < steps, jax.checkpoint(run_one), lambda p, a, k, c: c, params, alive, k, carry), None

            carry, _ = lax.scan(body, carry, (jnp.arange(t_max), jax.random.split(key, t_max)))
            return carry

        k_warm, k_dmg, k_rec = jax.random.split(key, 3)
        carry = phase((x, s, mem, jnp.zeros(())), alive, n_warm, warm_max, k_warm)
        x, s, mem, moves = carry
        early = 0.0
        if warm_loss:
            early = warm_loss * loss_fn(x, s, moves, alive, ref, total)[0]
        x, s, mem, alive = damager(k_dmg, x, s, mem, alive)
        x, s, mem, moves = phase((x, s, mem, moves), alive, n_rec, rec_max, k_rec)
        loss, parts = loss_fn(x, s, moves, alive, ref, total)
        return loss + early, (parts, loss)

    def train_step(params, opt, key, x, s, mem, ref, total, n_warm, n_rec, lr):
        (_, (parts, loss)), grads = jax.value_and_grad(rollout_loss, has_aux=True)(
            params, key, x, s, mem, ref, total, n_warm, n_rec)
        grads = jax.tree_util.tree_map(lambda g: jnp.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0), grads)
        if grad_mode == "per_tensor":       # each tensor's gradient to unit length, as in the paper's code
            grads = jax.tree_util.tree_map(lambda g: g / (jnp.linalg.norm(g) + 1e-8), grads)
        else:                               # a global clip at the given norm
            gnorm = jnp.sqrt(sum(jnp.sum(g * g) for g in jax.tree_util.tree_leaves(grads)))
            grads = jax.tree_util.tree_map(lambda g: g * jnp.minimum(1.0, grad_mode / (gnorm + 1e-8)), grads)
        b1, b2, e = run.ADAM
        t = opt["t"] + 1
        m = jax.tree_util.tree_map(lambda m, g: b1 * m + (1 - b1) * g, opt["m"], grads)
        v = jax.tree_util.tree_map(lambda v, g: b2 * v + (1 - b2) * g * g, opt["v"], grads)
        params = jax.tree_util.tree_map(
            lambda p, m, v: p - lr * (m / (1 - b1**t)) / (jnp.sqrt(v / (1 - b2**t)) + e), params, m, v)
        return params, {"m": m, "v": v, "t": t}, loss, parts

    return jax.jit(train_step)


def train(cfg, log=print):
    n = cfg["particles"]
    tc = {**run.TRAIN_DEFAULTS, **AGNOSTIC_DEFAULTS, **cfg["train"]}
    if tc["pool"]:
        raise ValueError("train.pool is not implemented for the agnostic trainer (the reference must stay paired with its swarm)")
    eps = cfg.get("eps") or 0.2 * math.sqrt(1024 / n)
    cell = cells.build(**dict(cfg["cell"]))
    batch, grid = tc["batch"], tc["grid"]
    block = 512 if n > 2048 else None
    grad_mode = "per_tensor" if tc["grad"] == "per_tensor" else float(tc["grad"])
    train_step = make_train_step(cell, make_loss(grid), run.make_damager(tc["damage"], batch, eps), batch, n, eps, eps, block,
                                 tc["warm"][1], tc["recover"][1], grad_mode, tc["noise_pos"], tc["noise_state"], tc["warm_loss"])
    render = jax.jit(jax.vmap(lambda x, s, t: agnostic.render(x, s, None, t, grid)))

    params = cell.init(jax.random.PRNGKey(cfg.get("seed", 0)))
    opt = {"m": jax.tree_util.tree_map(jnp.zeros_like, params), "v": jax.tree_util.tree_map(jnp.zeros_like, params),
           "t": jnp.zeros((), jnp.int32)}
    rng = np.random.default_rng(cfg.get("seed", 0))
    key = jax.random.PRNGKey(cfg.get("seed", 0) + 1)
    milestones = {int(f * tc["iters"]) for f in run.LR_MILESTONES}
    lr, curve, t0 = tc["lr"], [], time.time()
    recoveries, last_blowup, lr_cap = 0, -10_000, lr
    mem0 = jnp.zeros((batch, n, cell.mem_dims))

    for it in range(tc["iters"]):
        if it in milestones:
            lr *= run.LR_DROP
            lr_cap *= run.LR_DROP
        if recoveries and it - last_blowup > 300:
            recoveries = 0
        x, s, total = map(jnp.asarray, agnostic.train_batch(rng, batch, n, tc["proc_fraction"]))
        ref = render(x, s, total)                 # the reference is the START swarm
        n_warm, n_rec = int(rng.integers(tc["warm"][0], tc["warm"][1] + 1)), int(rng.integers(tc["recover"][0], tc["recover"][1] + 1))
        key, kt = jax.random.split(key)
        ts = time.time()
        good = (params, opt)
        params, opt, loss, parts = train_step(params, opt, kt, x, s, mem0, ref, total, n_warm, n_rec, lr)
        loss = float(loss)
        if not math.isfinite(loss) or not all(bool(jnp.isfinite(v).all()) for v in params.values()):
            recoveries += 1
            if recoveries > run.MAX_RECOVERIES:
                log(f"loss became {loss} at iteration {it}; giving up after {run.MAX_RECOVERIES} recoveries")
                break
            params, opt = good
            last_blowup = it
            lr = max(lr * 0.5, lr_cap * 0.05)
            log(f"loss became {loss} at iteration {it}; back to the last good weights, lr now {lr:.2e} (recovery {recoveries})")
            continue
        lr = min(lr * 1.02, lr_cap)
        curve.append({"iter": it, "loss": loss, "density": float(parts["density"]), "colour": float(parts["colour"])})
        if it % 100 == 0 or it == tc["iters"] - 1:
            log(f"it {it:5d} steps {n_warm}+{n_rec} loss {loss:7.4f} density {curve[-1]['density']:.4f} "
                f"colour {curve[-1]['colour']:.4f} {time.time() - ts:.2f}s/it")
        if tc["max_minutes"] and time.time() - t0 > tc["max_minutes"] * 60:
            log(f"time limit reached after {it + 1} iterations")
            break
    return cell, params, eps, curve, time.time() - t0
