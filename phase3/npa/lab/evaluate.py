"""The repair evaluator. Experiments may not edit this file; run.py records its hash with every score.

Given a trained cell, it grows the lizard from fixed held-out seeds, damages it in six ways at three
severities each, lets it recover, and scores what is left. Every damaged swarm is paired with an
undamaged control from the same seed, run for the same number of steps.

    growth       the control after GROW and after GROW + RECOVER steps
    stability    the control after LONG steps: does the finished lizard hold still, or drift and blow up?
    damage[k]    for each damage kind k and severity: the loss after RECOVER steps, the excess over
                 the control (0 = fully repaired), and IoU
    summary      score = mean loss over all damaged runs (lower is better). It folds in growth quality, so a model
                 that cannot grow a lizard cannot win by "repairing" nothing. excess says how much
                 of the score is damage rather than a bad lizard.

Loss is target.make_eval's loss: density + 5 * colour, as in training but without the regularisers.
"""

import hashlib
import sys
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import damage  # noqa: E402
import npa  # noqa: E402
import target as tgt  # noqa: E402

# NOTE: the seed disc has radius max(2 * eps, 0.4). For eps >= 0.2 (every run before the physical-regime
# experiments) this is exactly the old 2 * eps; for small eps a 2 * eps disc packs 1,024 particles ~20x too densely
# (initial density loss 44 at eps 0.066) and training diverges (H18).
PROTOCOL = {
    "version": 1,
    "grow": 200,          # steps to grow from the seed before any damage
    "recover": 200,       # steps the damaged swarm gets to repair itself
    "long": 1000,         # steps for the stability check (counted from the seed)
    "seeds": (10_000, 10_001, 10_002, 10_003, 10_004),   # never used for training
    "levels": {
        "wipe": (0.2, 0.4, 0.6),
        "cut": (0.12, 0.2, 0.3),
        "drag": (0.2, 0.4, 0.8),
        "clump": (0.3, 0.5, 0.7),
        "shuffle": (0.2, 0.4, 0.6),
        "noise": (0.1, 0.25, 0.5),
    },
    "grid": 256,
}

IMMUTABLE = ("lab/evaluate.py", "lab/damage.py", "target.py", "npa.py", "data/lizard_u1f98e.png")


def fingerprint():
    """Hashes of everything an experiment must not change. run.py stores this with every result."""
    root = Path(__file__).resolve().parent.parent
    return {f: hashlib.sha256((root / f).read_bytes()).hexdigest()[:16] for f in IMMUTABLE}


def _roll(cell, params, eps, eps0, n0, block, steps, x, s, mem, alive, key):
    """Run one swarm for `steps` steps; deleted particles neither move nor are seen."""
    def body(carry, k):
        x, s, mem = carry
        update = jax.random.bernoulli(k, npa.UPDATE_PROB, (x.shape[0], 1)).astype(x.dtype) * alive[:, None]
        x, s, mem, _ = cell.step(params, x, s, mem, update, eps, eps0, block, n0, alive)
        return (x, s, mem), None

    (x, s, mem), _ = lax.scan(body, (x, s, mem), jax.random.split(key, steps))
    return x, s, mem


def run(cell, params, eps, eps0, n, protocol=PROTOCOL, block=None, target=None):
    """Score a cell. Returns a JSON-friendly dict."""
    target = target or tgt.load_target()
    score = jax.vmap(tgt.make_eval(target, protocol["grid"]))
    seeds, levels = protocol["seeds"], protocol["levels"]
    n_seeds, n_levels = len(seeds), len(next(iter(levels.values())))
    batch = n_seeds * n_levels          # every rollout is batch-sized, so each step count compiles once

    # layout: index = level * n_seeds + seed, so reshape(n_levels, n_seeds) recovers the grid
    seed_keys = jnp.tile(jnp.stack([jax.random.PRNGKey(s) for s in seeds]), (n_levels, 1))
    x0, s0 = jax.vmap(lambda k: npa.seed(k, n, radius=max(2 * eps, 0.4)))(seed_keys)
    mem0 = jnp.zeros((batch, n, cell.mem_dims))
    alive0 = jnp.ones((batch, n))

    @partial(jax.jit, static_argnames=("steps",))
    def roll(x, s, mem, alive, key, steps):
        keys = jax.random.split(key, batch)
        return jax.vmap(lambda *a: _roll(cell, params, eps, eps0, n, block, steps, *a))(x, s, mem, alive, keys)

    def metrics(x, s, alive):
        m = score(x, s, alive)
        return {k: np.asarray(v).reshape(n_levels, n_seeds)[0] for k, v in m.items()}   # first row: the seed grid

    rollout_key = jax.random.PRNGKey(7_000_000 + protocol["version"])
    g, r, long_ = protocol["grow"], protocol["recover"], protocol["long"]

    # grow, then run the undamaged control on to the recovery horizon and to the long horizon
    x, s, mem = roll(x0, s0, mem0, alive0, jax.random.fold_in(rollout_key, 0), g)
    pre = (x, s, mem)
    m_grown = metrics(x, s, alive0)
    xc, sc, memc = roll(*pre, alive0, jax.random.fold_in(rollout_key, 1), r)
    m_ctrl = metrics(xc, sc, alive0)
    xl, sl, _ = roll(xc, sc, memc, alive0, jax.random.fold_in(rollout_key, 2), long_ - g - r)
    m_long = metrics(xl, sl, alive0)

    out = {"protocol": {k: protocol[k] for k in ("version", "grow", "recover", "long", "grid")},
           "growth": {"loss_grown": m_grown["loss"].tolist(), "loss_control": m_ctrl["loss"].tolist(),
                      "iou_control": m_ctrl["iou"].tolist()},
           "stability": {"loss_long": m_long["loss"].tolist()}, "damage": {}}

    for ki, kind in enumerate(damage.KINDS):
        lv = jnp.repeat(jnp.asarray(levels[kind], dtype=jnp.float32), n_seeds)
        dkeys = jax.random.split(jax.random.fold_in(jax.random.PRNGKey(8_000_000), ki), batch)
        dkeys = jnp.tile(dkeys[:n_seeds], (n_levels, 1))        # same centre for every severity of a seed
        xd, sd, memd, alived = jax.vmap(partial(damage.apply, kind, eps=eps))(lv, dkeys, *pre, alive0)
        m_hurt = score(xd, sd, alived)
        xr, sr, _ = roll(xd, sd, memd, alived, jax.random.fold_in(rollout_key, 1), r)   # same update masks as the control: a paired comparison
        m_end = score(xr, sr, alived)
        loss_end = np.asarray(m_end["loss"]).reshape(n_levels, n_seeds)
        out["damage"][kind] = {
            "levels": list(levels[kind]),
            "loss_after_damage": np.asarray(m_hurt["loss"]).reshape(n_levels, n_seeds).tolist(),
            "loss_final": loss_end.tolist(),
            "excess": (loss_end - m_ctrl["loss"][None, :]).tolist(),
            "iou_final": np.asarray(m_end["iou"]).reshape(n_levels, n_seeds).tolist(),
            "alive_final": np.asarray(alived).reshape(n_levels, n_seeds, n).sum(-1).mean(-1).tolist(),
        }

    finals = np.array([out["damage"][k]["loss_final"] for k in damage.KINDS])
    excess = np.array([out["damage"][k]["excess"] for k in damage.KINDS])
    finite = bool(np.isfinite(finals).all() and np.isfinite(m_long["loss"]).all())
    big = 1e3
    out["summary"] = {
        "score": float(np.nan_to_num(finals, nan=big, posinf=big).mean()) if finite else big,
        "excess": float(np.nan_to_num(excess, nan=big, posinf=big).mean()) if finite else big,
        "per_kind_score": {k: float(np.nan_to_num(finals[i], nan=big, posinf=big).mean()) for i, k in enumerate(damage.KINDS)},
        "per_kind_excess": {k: float(np.nan_to_num(excess[i], nan=big, posinf=big).mean()) for i, k in enumerate(damage.KINDS)},
        "control_loss": float(np.nan_to_num(m_ctrl["loss"], nan=big).mean()),
        "control_iou": float(np.nan_to_num(m_ctrl["iou"], nan=0.0).mean()),
        "drift": float(np.nan_to_num(m_long["loss"], nan=big).mean() - np.nan_to_num(m_ctrl["loss"], nan=big).mean()),
        "finite": finite,
    }
    return out
