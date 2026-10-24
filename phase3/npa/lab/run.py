"""Train one configuration, score it with the immutable evaluator, and log the result.

    python lab/run.py configs/baseline.json
    python lab/run.py '{"name": "tiny", "particles": 128, "cell": {}, "train": {"iters": 5}}'

A config is JSON:
    name        run name; outputs go to runs/<name>/
    seed        training seed (the evaluator uses its own held-out seeds)
    particles   swarm size N; eps follows the paper's rule 0.2 * sqrt(1024 / N) unless "eps" is given
    cell        options for cells.Cell (block, width, depth, norm, mem, ...)
    train       iters, batch, steps [lo, hi], lr, grid, grad ("per_tensor" or a global clip value),
                noise_pos, noise_state (per-step noise injected during training, in units of eps / state),
                damage: [{"kind": "wipe", "level": [lo, hi], "p": 1.0}, ...] applied to each pool sample,
                max_minutes

Training follows train.py (sample pool, seed injection, backprop through 32-96 steps) but works with any
cell, deleted particles and private memory. Appends one row to results.jsonl.
"""

import json
import math
import subprocess
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
import evaluate  # noqa: E402
import npa  # noqa: E402
import target as tgt  # noqa: E402

POOL_SIZE = 512
SEED_EVERY = 16
LR_DROP, LR_MILESTONES = 0.3, (0.2, 0.4, 0.6, 0.8)
ADAM = (0.9, 0.999, 1e-8)
TRAIN_DEFAULTS = {"mode": "bptt", "iters": 3000, "batch": 8, "steps": [32, 96], "lr": 5e-4, "grid": 128, "grad": "per_tensor",
                  "noise_pos": 0.0, "noise_state": 0.0, "max_minutes": None,
                  "damage": [{"kind": "wipe", "level": [0.2, 0.2], "p": 1.0}]}


def make_train_step(cell, loss_fn, batch, n, eps, eps0, block, t_max, grad_mode, noise_pos, noise_state):
    def rollout_loss(params, key, x, s, mem, alive, steps):
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
        loss, parts = loss_fn(x, s, moves, alive)
        return loss, (x, s, mem, parts)

    def train_step(params, opt, key, x, s, mem, alive, steps, lr):
        (loss, (x, s, mem, parts)), grads = jax.value_and_grad(rollout_loss, has_aux=True)(
            params, key, x, s, mem, alive, steps)
        grads = jax.tree_util.tree_map(lambda g: jnp.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0), grads)
        if grad_mode == "per_tensor":       # each tensor's gradient to unit length, as in the paper's code
            grads = jax.tree_util.tree_map(lambda g: g / (jnp.linalg.norm(g) + 1e-8), grads)
        else:                               # a global clip at the given norm
            gnorm = jnp.sqrt(sum(jnp.sum(g * g) for g in jax.tree_util.tree_leaves(grads)))
            grads = jax.tree_util.tree_map(lambda g: g * jnp.minimum(1.0, grad_mode / (gnorm + 1e-8)), grads)
        b1, b2, e = ADAM
        t = opt["t"] + 1
        m = jax.tree_util.tree_map(lambda m, g: b1 * m + (1 - b1) * g, opt["m"], grads)
        v = jax.tree_util.tree_map(lambda v, g: b2 * v + (1 - b2) * g * g, opt["v"], grads)
        params = jax.tree_util.tree_map(
            lambda p, m, v: p - lr * (m / (1 - b1**t)) / (jnp.sqrt(v / (1 - b2**t)) + e), params, m, v)
        return params, {"m": m, "v": v, "t": t}, x, s, mem, loss, parts

    return jax.jit(train_step)


def make_damager(specs, batch, eps):
    """Damage each pool sample with each spec's probability and a level drawn from its range."""
    @jax.jit
    def damager(key, x, s, mem, alive):
        for i, spec in enumerate(specs):
            kl, kp, kd = jax.random.split(jax.random.fold_in(key, i), 3)
            lo, hi = spec["level"]
            level = jax.random.uniform(kl, (batch,), minval=lo, maxval=hi)
            hurt = jax.random.bernoulli(kp, spec.get("p", 1.0), (batch,))
            nx, ns, nm, na = jax.vmap(lambda lv, k, x, s, m, a: damage.apply(spec["kind"], lv, k, x, s, m, a, eps))(
                level, jax.random.split(kd, batch), x, s, mem, alive)
            pick = lambda new, old: jnp.where(hurt.reshape((batch,) + (1,) * (old.ndim - 1)), new, old)
            x, s, mem, alive = pick(nx, x), pick(ns, s), pick(nm, mem), pick(na, alive)
        return x, s, mem, alive
    return damager


def train(cfg, log=print):
    n, tc = cfg["particles"], {**TRAIN_DEFAULTS, **cfg["train"]}
    eps = cfg.get("eps") or 0.2 * math.sqrt(1024 / n)
    if tc.get("mode", "bptt") != "bptt":      # other training schemes live in lab/trainer_<mode>.py
        import importlib
        return importlib.import_module(f"trainer_{tc['mode']}").train(cfg, log)
    cell = cells.build(**cfg["cell"])
    batch, t_max = tc["batch"], tc["steps"][1]
    block = 512 if n > 2048 else None
    target = tgt.load_target()
    loss_fn = tgt.make_loss(target, grid=tc["grid"])
    grad_mode = "per_tensor" if tc["grad"] == "per_tensor" else float(tc["grad"])
    train_step = make_train_step(cell, loss_fn, batch, n, eps, eps, block, t_max, grad_mode,
                                 tc["noise_pos"], tc["noise_state"])
    damager = make_damager(tc["damage"], batch, eps)

    params = cell.init(jax.random.PRNGKey(cfg.get("seed", 0)))
    opt = {"m": jax.tree_util.tree_map(jnp.zeros_like, params), "v": jax.tree_util.tree_map(jnp.zeros_like, params),
           "t": jnp.zeros((), jnp.int32)}
    rng = np.random.default_rng(cfg.get("seed", 0))
    key = jax.random.PRNGKey(cfg.get("seed", 0) + 1)

    def fresh(count):
        x, s = jax.vmap(lambda k: npa.seed(k, n, radius=2 * eps))(jax.random.split(jax.random.fold_in(key, rng.integers(1 << 30)), count))
        return np.array(x), np.array(s)      # writable copies: the pool is updated in place

    pool_x, pool_s = fresh(POOL_SIZE)
    pool_m = np.zeros((POOL_SIZE, n, cell.mem_dims), np.float32)
    pool_a = np.ones((POOL_SIZE, n), np.float32)
    milestones = {int(f * tc["iters"]) for f in LR_MILESTONES}
    lr, curve, t0 = tc["lr"], [], time.time()

    for it in range(tc["iters"]):
        if it in milestones:
            lr *= LR_DROP
        idx = rng.permutation(POOL_SIZE)[:batch]
        x, s, m, a = pool_x[idx].copy(), pool_s[idx].copy(), pool_m[idx].copy(), pool_a[idx].copy()
        if it % SEED_EVERY == 0:
            x[:1], s[:1] = fresh(1)
            m[:1], a[:1] = 0.0, 1.0
        key, kd, kt = jax.random.split(key, 3)
        x, s, m, a = damager(kd, *map(jnp.asarray, (x, s, m, a)))
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
            log(f"it {it:5d} steps {steps:2d} loss {loss:7.4f} density {curve[-1]['density']:.4f} "
                f"colour {curve[-1]['colour']:.4f} {time.time() - ts:.2f}s/it")
        if tc["max_minutes"] and time.time() - t0 > tc["max_minutes"] * 60:
            log(f"time limit reached after {it + 1} iterations")
            break
    return cell, params, eps, curve, time.time() - t0


def main():
    arg = sys.argv[1]
    cfg = json.loads(arg) if arg.lstrip().startswith("{") else json.loads(Path(arg).read_text())
    name = cfg["name"]
    out_dir = HERE.parent / "runs" / name
    out_dir.mkdir(parents=True, exist_ok=True)
    logf = open(out_dir / "train.log", "a")

    def log(msg):
        print(msg, flush=True)
        logf.write(msg + "\n")
        logf.flush()

    log(f"== {name}: {json.dumps(cfg)}")
    cell, params, eps, curve, seconds = train(cfg, log)
    np.savez(out_dir / "params.npz", eps=eps, **{k: np.asarray(v) for k, v in params.items()})
    log(f"trained in {seconds / 60:.1f} min; scoring")
    t0 = time.time()
    result = evaluate.run(cell, params, eps, eps, cfg["particles"])
    sm = result["summary"]
    log(f"scored in {time.time() - t0:.0f}s: score {sm['score']:.4f}  excess {sm['excess']:.4f}  "
        f"control {sm['control_loss']:.4f}  drift {sm['drift']:.4f}")
    log("per-kind score: " + "  ".join(f"{k} {v:.3f}" for k, v in sm["per_kind_score"].items()))
    row = {"name": name, "time": time.strftime("%Y-%m-%d %H:%M:%S"), "config": cfg,
           "params": cell.param_count(params), "iters_done": len(curve), "train_minutes": seconds / 60,
           "final_train_loss": float(np.mean([c["loss"] for c in curve[-50:]])) if curve else None,
           "summary": sm, "evaluator": evaluate.fingerprint(),
           "params_sha": __import__("hashlib").sha256(b"".join(np.asarray(params[k]).tobytes() for k in sorted(params))).hexdigest()[:16]}
    (out_dir / "result.json").write_text(json.dumps({**row, "eval": result, "curve": curve[::10]}))
    with open(HERE / "results.jsonl", "a") as f:
        f.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
