"""Multi-shape training (selected by train.mode == "multishape" in lab/run.py): one network, several shapes.

Config: {"name", "particles", "shapes": ["lizard", "star", ...], "cell": {...}, "train": {"mode": "multishape", ...}}

The shape is selected by a code in the seed: s[:, :K] = one-hot(shape), the rest 0 (evaluate_multi.seed_swarm). The cell sees
the code only through its state. Everything else is run.train: sample pool, seed injection every SEED_EVERY iterations,
damage curriculum (run.make_damager), Adam with per-tensor gradient normalisation, the LR schedule, noise options and the
NaN-recovery rule.

What changes:
  * every pool entry stores its shape index (pool_k); the initial pool has the shapes in equal numbers and every
    injected seed draws its shape uniformly, so each batch element has its shape's code in its state from the start;
  * the loss of batch element b compares its swarm with the splatted image of ITS shape's target, selected with
    pool_k[b] from the stacked targets (multi_loss below). It is target.make_loss line by line: density l1l2, colour l1l2
    weighted by exp(-stop_gradient(density diff)), the same weights, regularisers, alive/mass support, and a centring
    on that target's centre. With one shape it is target.make_loss.
  * wipe damage zeroes the state including the code in the wiped particles. That is intended: the neighbours that still
    carry the code have to restore it.

Extra train options (all optional):
    code_scale   scale of the one-hot code in the seed (default 1.0). Only for tests: 0 turns the code off, which with one
                 shape reproduces run.train exactly. Evaluation always uses 1.0.

The random stream of run.train is untouched: the shape draws use a separate generator, so with one shape (and code_scale 0)
iteration 0 sees the same seeds, damage and update masks as run.train for the same config.
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
import cells  # noqa: E402
import evaluate_multi as em  # noqa: E402
import npa  # noqa: E402
import run  # noqa: E402
import target as tgt  # noqa: E402

MULTI_DEFAULTS = {"code_scale": 1.0}


def make_multi_loss(stack, grid):
    """loss(x, s, sum_move, alive, idx) -> (total, parts), like target.make_loss but with a shape index per batch element.

    x (B, N, 2), s (B, N, 16), alive (B, N), idx (B,) ints into the stacked targets of evaluate_multi.stack_targets.
    parts has the keys of make_loss's plus "elem" (B,), each element's splat loss, for per-shape logging.
    """
    l1l2 = tgt.l1l2

    def render(x, s, w, k):
        x_c = x - jnp.sum(x * w[:, None], axis=0) / jnp.maximum(jnp.sum(w), 1.0) + stack["centre"][k]
        return tgt.splat(x_c, npa.colour(s), stack["pixel"][k], grid, total=stack["count"][k], weight=w)

    def loss(x, s, sum_move, alive, idx):
        img = jax.vmap(render)(x, s, alive, idx)                          # (B, grid, grid, 4)
        den_diff = l1l2(img[..., 3:] - stack["den"][idx])
        w = jnp.exp(-jax.lax.stop_gradient(den_diff))                     # colour only has to match where density does
        col_map = l1l2(img[..., :3] - stack["rgb"][idx]) * w
        col_loss = jnp.mean(col_map)
        den_loss = jnp.mean(den_diff)
        splat_loss = tgt.DENSITY_WEIGHT * den_loss + tgt.COLOUR_WEIGHT * col_loss
        overflow = jnp.mean(jnp.abs(s - jnp.clip(s, -1.0, 1.0)))
        bound = jnp.mean(jnp.abs(x - jnp.clip(x, -1.0, 1.0)))
        total = (tgt.SPLAT_LOSS_WEIGHT * splat_loss + tgt.DISPLACEMENT_WEIGHT * sum_move
                 + tgt.OVERFLOW_WEIGHT * overflow + tgt.BOUND_WEIGHT * bound)
        elem = tgt.DENSITY_WEIGHT * jnp.mean(den_diff, axis=(1, 2, 3)) + tgt.COLOUR_WEIGHT * jnp.mean(col_map, axis=(1, 2, 3))
        return total, {"density": den_loss, "colour": col_loss, "splat": splat_loss, "move": sum_move,
                       "overflow": overflow, "bound": bound, "elem": elem}

    return loss


def make_train_step(cell, loss_fn, batch, n, eps, eps0, block, t_max, grad_mode, noise_pos, noise_state):
    """run.make_train_step with one more argument, idx (B,) shape indices, passed to the loss. Same update rule."""
    def rollout_loss(params, key, x, s, mem, alive, idx, steps):
        def run_one(params, k, carry):
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
            carry = lax.cond(t < steps, jax.checkpoint(run_one), lambda p, k, c: c, params, k, carry)
            return carry, None

        keys = jax.random.split(key, t_max)
        (x, s, mem, moves), _ = lax.scan(body, (x, s, mem, jnp.zeros(())), (jnp.arange(t_max), keys))
        loss, parts = loss_fn(x, s, moves, alive, idx)
        return loss, (x, s, mem, parts)

    def train_step(params, opt, key, x, s, mem, alive, idx, steps, lr):
        (loss, (x, s, mem, parts)), grads = jax.value_and_grad(rollout_loss, has_aux=True)(
            params, key, x, s, mem, alive, idx, steps)
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
        return params, {"m": m, "v": v, "t": t}, x, s, mem, loss, parts

    return jax.jit(train_step)


def train(cfg, log=print):
    n, tc = cfg["particles"], {**run.TRAIN_DEFAULTS, **MULTI_DEFAULTS, **cfg["train"]}
    names = list(cfg["shapes"])
    num_shapes = len(names)
    eps = cfg.get("eps") or 0.2 * math.sqrt(1024 / n)
    cell = cells.build(**cfg["cell"])
    batch, t_max = tc["batch"], tc["steps"][1]
    block = 512 if n > 2048 else None
    stack = em.stack_targets(names, tc["grid"])
    loss_fn = make_multi_loss(stack, tc["grid"])
    grad_mode = "per_tensor" if tc["grad"] == "per_tensor" else float(tc["grad"])
    train_step = make_train_step(cell, loss_fn, batch, n, eps, eps, block, t_max, grad_mode,
                                 tc["noise_pos"], tc["noise_state"])
    damager = run.make_damager(tc["damage"], batch, eps)
    code_scale = float(tc["code_scale"])
    seed_batch = jax.jit(jax.vmap(lambda k, c: em.seed_swarm(k, n, 2 * eps, c, num_shapes, code_scale)))

    params = cell.init(jax.random.PRNGKey(cfg.get("seed", 0)))
    opt = {"m": jax.tree_util.tree_map(jnp.zeros_like, params), "v": jax.tree_util.tree_map(jnp.zeros_like, params),
           "t": jnp.zeros((), jnp.int32)}
    rng = np.random.default_rng(cfg.get("seed", 0))
    shape_rng = np.random.default_rng([cfg.get("seed", 0), 7])      # separate stream: run.train's draws stay unchanged
    key = jax.random.PRNGKey(cfg.get("seed", 0) + 1)

    def fresh(ks):
        """Seeds for shape indices ks (count,): positions as in run.train, state = the shape's code."""
        keys = jax.random.split(jax.random.fold_in(key, rng.integers(1 << 30)), len(ks))
        x, s = seed_batch(keys, jnp.asarray(ks, jnp.int32))
        return np.array(x), np.array(s)      # writable copies: the pool is updated in place

    pool_k = shape_rng.permutation(np.arange(run.POOL_SIZE) % num_shapes).astype(np.int32)     # shapes in equal numbers
    pool_x, pool_s = fresh(pool_k)
    pool_m = np.zeros((run.POOL_SIZE, n, cell.mem_dims), np.float32)
    pool_a = np.ones((run.POOL_SIZE, n), np.float32)
    pool_age = np.zeros(run.POOL_SIZE, np.int32)
    max_age = tc.get("pool_max_age")      # an entry rolled out this many times goes back to a fresh seed (None: never)
    milestones = {int(f * tc["iters"]) for f in run.LR_MILESTONES}
    lr, curve, t0 = tc["lr"], [], time.time()
    recoveries, last_blowup, lr_cap = 0, -10_000, lr

    for it in range(tc["iters"]):
        if it in milestones:
            lr *= run.LR_DROP
            lr_cap *= run.LR_DROP
        if recoveries and it - last_blowup > 300:
            recoveries = 0
        idx = rng.permutation(run.POOL_SIZE)[:batch]
        if max_age:
            old = idx[pool_age[idx] >= max_age]
            if len(old):
                pool_x[old], pool_s[old] = fresh(pool_k[old])
                pool_m[old], pool_a[old], pool_age[old] = 0.0, 1.0, 0
        x, s, m, a, k = pool_x[idx].copy(), pool_s[idx].copy(), pool_m[idx].copy(), pool_a[idx].copy(), pool_k[idx].copy()
        if it % run.SEED_EVERY == 0:
            k[:1] = shape_rng.integers(num_shapes, size=1)
            x[:1], s[:1] = fresh(k[:1])
            m[:1], a[:1] = 0.0, 1.0
        key, kd, kt = jax.random.split(key, 3)
        x, s, m, a = damager(kd, *map(jnp.asarray, (x, s, m, a)))
        steps = int(rng.integers(*tc["steps"]))
        ts = time.time()
        good = (params, opt)
        params, opt, x, s, m, loss, parts = train_step(params, opt, kt, x, s, m, a, jnp.asarray(k), steps, lr)
        loss = float(loss)
        if not math.isfinite(loss) or not all(bool(jnp.isfinite(v).all()) for v in params.values()):
            # a blow-up: go back to the last good weights, halve the learning rate and carry on
            recoveries += 1
            if recoveries > run.MAX_RECOVERIES:
                log(f"loss became {loss} at iteration {it}; giving up after {run.MAX_RECOVERIES} recoveries")
                break
            params, opt = good
            last_blowup = it
            lr = max(lr * 0.5, lr_cap * 0.05)
            pool_x[idx], pool_s[idx] = fresh(pool_k[idx])
            pool_m[idx], pool_a[idx] = 0.0, 1.0
            log(f"loss became {loss} at iteration {it}; back to the last good weights, lr now {lr:.2e} (recovery {recoveries})")
            continue
        pool_x[idx], pool_s[idx], pool_m[idx], pool_a[idx] = map(np.asarray, (x, s, m, a))
        pool_k[idx] = k       # unchanged for pool samples; the injected seed has its new shape
        lr = min(lr * 1.02, lr_cap)
        pool_age[idx] += 1
        pool_age[idx[:1]] = np.where(it % run.SEED_EVERY == 0, 0, pool_age[idx[:1]])   # the injected seed starts young
        elem = np.asarray(parts["elem"])
        curve.append({"iter": it, "loss": loss, "density": float(parts["density"]), "colour": float(parts["colour"])})
        if it % 100 == 0 or it == tc["iters"] - 1:
            per = "  ".join(f"{names[j]} {elem[k == j].mean():.3f}" for j in range(num_shapes) if (k == j).any())
            log(f"it {it:5d} steps {steps:2d} loss {loss:7.4f} density {curve[-1]['density']:.4f} "
                f"colour {curve[-1]['colour']:.4f} [{per}] {time.time() - ts:.2f}s/it")
        if tc["max_minutes"] and time.time() - t0 > tc["max_minutes"] * 60:
            log(f"time limit reached after {it + 1} iterations")
            break
    return cell, params, eps, curve, time.time() - t0
