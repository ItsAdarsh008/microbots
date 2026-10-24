"""Truncated-window training with a persistent state (selected by train.mode == "trunc" in lab/run.py).

Same sample pool, seed injection, Adam, per-tensor gradient normalisation, LR schedule and noise options as
run.train, but the rollout of T steps is cut into windows of `window` steps. Gradients flow only inside a
window (the carried x, s, mem are constants at window boundaries, i.e. a stop-gradient), and a loss is taken at
the end of every window. Because the windows are independent for the gradient, each window's gradient is computed
and accumulated on its own inside a lax.scan, so the autodiff residuals span `window` steps, not T.

Extra train options (all optional):
    window        steps per window (default 8; 0 or >= steps[1] means one window = plain BPTT)
    window_loss   "all" (default): mean of the losses at the end of every window; "last": final window only
    anneal        true (default): damage levels are scaled by a factor going from anneal_start to 1.0 over
                  the first half of training, then held at 1.0 (levels are capped at 1)
    anneal_start  initial damage scale (default 1.5)
    reset_p       default 0.0. At each seed-injection iteration, with this probability the fresh seed is replaced by
                  a pool sample, i.e. the final state of an earlier rollout, which is then damaged as usual.

Notes on the loss: the displacement penalty of a window is the movement summed over that window times the number
of active windows, so that the mean over windows has the same size as the penalty of a full rollout.
The curve reports loss/density/colour of the FINAL window (comparable with run.train's end-of-rollout loss);
the log line also shows the training objective (mean over windows).
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
import damage  # noqa: E402
import npa  # noqa: E402
import run  # noqa: E402
import target as tgt  # noqa: E402

TRUNC_DEFAULTS = {"window": 8, "window_loss": "all", "anneal": True, "anneal_start": 1.5, "reset_p": 0.0}


def make_train_step(cell, loss_fn, batch, n, eps, eps0, block, t_max, grad_mode, noise_pos, noise_state,
                    window=8, window_loss="all"):
    """Returns jitted train_step(params, opt, key, x, s, mem, alive, steps, lr) like run.make_train_step."""
    assert window_loss in ("all", "last"), window_loss
    window = t_max if not window or window >= t_max else int(window)
    n_win = -(-t_max // window)
    last_only = window_loss == "last"

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

    step_fn = jax.checkpoint(run_one)

    def run_window(params, alive, x, s, mem, wkeys, t0, steps):
        """`window` steps starting at global step t0 (steps past `steps` are skipped). Returns x, s, mem, moves."""
        def body(carry, inp):
            t, k = inp
            carry = lax.cond(t < steps, step_fn, lambda p, a, k, c: c, params, alive, k, carry)
            return carry, None

        ts = t0 + jnp.arange(window)
        (x, s, mem, moves), _ = lax.scan(body, (x, s, mem, jnp.zeros(())), (ts, wkeys))
        return x, s, mem, moves

    def rollout_grads(params, key, x, s, mem, alive, steps):
        keys = jax.random.split(key, n_win * window).reshape((n_win, window) + key.shape)
        n_active = (steps + window - 1) // window                       # windows that have at least one step
        zeros_like = lambda t: jax.tree_util.tree_map(jnp.zeros_like, t)
        _, shapes = jax.eval_shape(lambda: loss_fn(x, s, jnp.zeros(()), alive))
        zparts = jax.tree_util.tree_map(lambda sd: jnp.zeros(sd.shape, sd.dtype), shapes)

        def win_loss(params, x, s, mem, wkeys, t0):
            x, s, mem, moves = run_window(params, alive, x, s, mem, wkeys, t0, steps)
            loss, parts = loss_fn(x, s, moves * n_active, alive)
            return loss, (x, s, mem, parts)

        def body(carry, inp):
            j, wkeys = inp
            x, s, mem, gacc, lacc, pacc, llast, plast = carry
            t0 = j * window
            is_last = j == n_active - 1
            weight = 1.0 if last_only else 1.0 / n_active
            case = jnp.where(j >= n_active, 0, jnp.where(last_only & ~is_last, 1, 2))

            def idle(c):
                return c

            def advance(c):                                             # forward only: no residuals, no gradient
                x, s, mem, gacc, lacc, pacc, llast, plast = c
                x, s, mem, _ = run_window(params, alive, x, s, mem, wkeys, t0, steps)
                return x, s, mem, gacc, lacc, pacc, llast, plast

            def learn(c):
                x, s, mem, gacc, lacc, pacc, llast, plast = c
                (loss, (x, s, mem, parts)), g = jax.value_and_grad(win_loss, has_aux=True)(
                    params, x, s, mem, wkeys, t0)
                gacc = jax.tree_util.tree_map(lambda a, b: a + weight * b, gacc, g)
                pacc = jax.tree_util.tree_map(lambda a, b: a + weight * b, pacc, parts)
                lacc = lacc + weight * loss
                llast = jnp.where(is_last, loss, llast)
                plast = jax.tree_util.tree_map(lambda o, nw: jnp.where(is_last, nw, o), plast, parts)
                return x, s, mem, gacc, lacc, pacc, llast, plast

            return lax.switch(case, (idle, advance, learn), carry), None

        init = (x, s, mem, zeros_like(params), jnp.zeros(()), zparts, jnp.zeros(()), zparts)
        (x, s, mem, grads, loss_mean, parts_mean, loss_last, parts_last), _ = lax.scan(
            body, init, (jnp.arange(n_win), keys))
        return grads, x, s, mem, loss_mean, loss_last, parts_last

    def train_step(params, opt, key, x, s, mem, alive, steps, lr):
        grads, x, s, mem, loss_mean, loss, parts = rollout_grads(params, key, x, s, mem, alive, steps)
        grads = jax.tree_util.tree_map(lambda g: jnp.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0), grads)
        if grad_mode == "per_tensor":
            grads = jax.tree_util.tree_map(lambda g: g / (jnp.linalg.norm(g) + 1e-8), grads)
        else:
            gnorm = jnp.sqrt(sum(jnp.sum(g * g) for g in jax.tree_util.tree_leaves(grads)))
            grads = jax.tree_util.tree_map(lambda g: g * jnp.minimum(1.0, grad_mode / (gnorm + 1e-8)), grads)
        b1, b2, e = run.ADAM
        t = opt["t"] + 1
        m = jax.tree_util.tree_map(lambda m, g: b1 * m + (1 - b1) * g, opt["m"], grads)
        v = jax.tree_util.tree_map(lambda v, g: b2 * v + (1 - b2) * g * g, opt["v"], grads)
        params = jax.tree_util.tree_map(
            lambda p, m, v: p - lr * (m / (1 - b1**t)) / (jnp.sqrt(v / (1 - b2**t)) + e), params, m, v)
        parts = {**parts, "loss_mean": loss_mean}
        return params, {"m": m, "v": v, "t": t}, x, s, mem, loss, parts

    jitted = jax.jit(train_step)
    jitted.rollout_grads = jax.jit(rollout_grads)  # exposed for tests
    return jitted


def make_damager(specs, batch, eps):
    """run.make_damager with one extra traced argument `scale` multiplying every level range (capped at 1)."""
    @jax.jit
    def damager(key, x, s, mem, alive, scale):
        for i, spec in enumerate(specs):
            kl, kp, kd = jax.random.split(jax.random.fold_in(key, i), 3)
            lo, hi = spec["level"]
            level = jax.random.uniform(kl, (batch,), minval=lo * scale, maxval=jnp.minimum(hi * scale, 1.0))
            hurt = jax.random.bernoulli(kp, spec.get("p", 1.0), (batch,))
            nx, ns, nm, na = jax.vmap(lambda lv, k, x, s, m, a: damage.apply(spec["kind"], lv, k, x, s, m, a, eps))(
                level, jax.random.split(kd, batch), x, s, mem, alive)
            pick = lambda new, old: jnp.where(hurt.reshape((batch,) + (1,) * (old.ndim - 1)), new, old)
            x, s, mem, alive = pick(nx, x), pick(ns, s), pick(nm, mem), pick(na, alive)
        return x, s, mem, alive
    return damager


def anneal_scale(it, iters, tc):
    if not tc["anneal"]:
        return 1.0
    frac = min(1.0, it / max(1.0, 0.5 * iters))
    return tc["anneal_start"] + (1.0 - tc["anneal_start"]) * frac


def train(cfg, log=print):
    n, tc = cfg["particles"], {**run.TRAIN_DEFAULTS, **TRUNC_DEFAULTS, **cfg["train"]}
    eps = cfg.get("eps") or 0.2 * math.sqrt(1024 / n)
    cell = cells.build(**cfg["cell"])
    batch, t_max = tc["batch"], tc["steps"][1]
    block = 512 if n > 2048 else None
    target = tgt.load_target()
    loss_fn = tgt.make_loss(target, grid=tc["grid"])
    grad_mode = "per_tensor" if tc["grad"] == "per_tensor" else float(tc["grad"])
    train_step = make_train_step(cell, loss_fn, batch, n, eps, eps, block, t_max, grad_mode,
                                 tc["noise_pos"], tc["noise_state"], tc["window"], tc["window_loss"])
    damager = make_damager(tc["damage"], batch, eps)

    params = cell.init(jax.random.PRNGKey(cfg.get("seed", 0)))
    opt = {"m": jax.tree_util.tree_map(jnp.zeros_like, params), "v": jax.tree_util.tree_map(jnp.zeros_like, params),
           "t": jnp.zeros((), jnp.int32)}
    rng = np.random.default_rng(cfg.get("seed", 0))
    key = jax.random.PRNGKey(cfg.get("seed", 0) + 1)
    pool_size = run.POOL_SIZE

    def fresh(count):
        x, s = jax.vmap(lambda k: npa.seed(k, n, radius=2 * eps))(jax.random.split(jax.random.fold_in(key, rng.integers(1 << 30)), count))
        return np.array(x), np.array(s)

    pool_x, pool_s = fresh(pool_size)
    pool_m = np.zeros((pool_size, n, cell.mem_dims), np.float32)
    pool_a = np.ones((pool_size, n), np.float32)
    milestones = {int(f * tc["iters"]) for f in run.LR_MILESTONES}
    lr, curve, t0 = tc["lr"], [], time.time()

    for it in range(tc["iters"]):
        if it in milestones:
            lr *= run.LR_DROP
        idx = rng.permutation(pool_size)[:batch]
        x, s, m, a = pool_x[idx].copy(), pool_s[idx].copy(), pool_m[idx].copy(), pool_a[idx].copy()
        if it % run.SEED_EVERY == 0 and not (tc["reset_p"] and rng.random() < tc["reset_p"]):
            x[:1], s[:1] = fresh(1)
            m[:1], a[:1] = 0.0, 1.0
        key, kd, kt = jax.random.split(key, 3)
        x, s, m, a = damager(kd, *map(jnp.asarray, (x, s, m, a)), jnp.float32(anneal_scale(it, tc["iters"], tc)))
        steps = int(rng.integers(*tc["steps"]))
        ts = time.time()
        params, opt, x, s, m, loss, parts = train_step(params, opt, kt, x, s, m, a, steps, lr)
        loss = float(loss)
        if not math.isfinite(loss):
            log(f"loss became {loss} at iteration {it}; stopping")
            break
        pool_x[idx], pool_s[idx], pool_m[idx], pool_a[idx] = map(np.asarray, (x, s, m, a))
        curve.append({"iter": it, "loss": loss, "density": float(parts["density"]), "colour": float(parts["colour"])})
        if it % 100 == 0 or it == tc["iters"] - 1:
            log(f"it {it:5d} steps {steps:2d} loss {loss:7.4f} (window mean {float(parts['loss_mean']):7.4f}) "
                f"density {curve[-1]['density']:.4f} colour {curve[-1]['colour']:.4f} "
                f"damage x{anneal_scale(it, tc['iters'], tc):.2f} {time.time() - ts:.2f}s/it")
        if tc["max_minutes"] and time.time() - t0 > tc["max_minutes"] * 60:
            log(f"time limit reached after {it + 1} iterations")
            break
    return cell, params, eps, curve, time.time() - t0
