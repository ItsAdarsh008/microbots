"""The multi-shape repair evaluator (task 2a). Experiments may not edit this file; run.py records its hash.

One network has to grow and heal every shape in cfg["shapes"]. The shape is chosen by a CODE in the seed: with K shapes the
seed swarm has s[:, :K] = one-hot(shape) and everything else 0 (see seed_swarm); the cell sees the code only through its state.

The protocol is evaluate.PROTOCOL's (same step counts, seeds, damage kinds, levels and grid; paired controls with the
same update masks), run once per shape with that shape's code and target. Per shape the result has the layout of
evaluate.run's; the top-level dict has the same layout with the numbers averaged over shapes, plus

    shapes        the shape names, in cfg["shapes"] order
    per_shape     {name: the full evaluate.run-style result for that shape}
    summary       exactly evaluate.run's summary keys averaged over shapes (score, excess, per_kind_score, per_kind_excess,
                  control_loss, control_iou, drift, finite), plus per_shape_score and per_shape_excess {name: value}

Loss is target.make_eval's (density + 5 * colour), computed against the shape's own target with a stacked-target version
that takes the shape index as a traced value, so every function here compiles once for all shapes.
"""

import hashlib
import sys
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
import damage  # noqa: E402
import npa  # noqa: E402
import shapes as shape_lib  # noqa: E402
import target as tgt  # noqa: E402

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

IMMUTABLE = ("lab/evaluate_multi.py", "lab/damage.py", "lab/shapes.py", "target.py", "npa.py", "data/lizard_u1f98e.png")
MAX_SHAPES = npa.STATE_DIMS - 3      # the last three state channels are the colour, so the code must fit before them


def fingerprint():
    """Hashes of everything an experiment must not change. run.py stores this with every result."""
    root = Path(__file__).resolve().parent.parent
    return {f: hashlib.sha256((root / f).read_bytes()).hexdigest()[:16] for f in IMMUTABLE}


# =============================================================================
# Seeds and stacked targets (the trainer imports these, so training and scoring agree)
# =============================================================================
def seed_swarm(key, n, radius, k, num_shapes, scale=1.0):
    """The seed blob with the shape code: s[:, :num_shapes] = scale * one_hot(k), everything else 0. k may be traced."""
    assert 1 <= num_shapes <= MAX_SHAPES, f"{num_shapes} shapes do not fit before the colour channels"
    x, s = npa.seed(key, n, radius=radius)
    return x, s.at[:, :num_shapes].set(scale * jax.nn.one_hot(k, num_shapes))


def stack_targets(names, grid):
    """Splat every shape's target once and stack them: arrays indexed by shape number (so a traced index selects one).

    Targets differ in point count and pixel size, so those are stacked too. Returns a dict of jnp arrays:
    rgb (K, g, g, 3), den (K, g, g, 1), fill (K, g, g), centre (K, 2), count (K,), pixel (K,).
    """
    imgs, centres, counts, pixels = [], [], [], []
    for name in names:
        t = tgt.load_target(shape_lib.path(name))
        pos = jnp.asarray(t["pos"])
        imgs.append(tgt.splat(pos, jnp.asarray(t["rgb"]), t["pixel"], grid))
        centres.append(pos.mean(axis=0))
        counts.append(float(pos.shape[0]))
        pixels.append(float(t["pixel"]))
    img = jnp.stack(imgs)
    return {"rgb": img[..., :3], "den": img[..., 3:], "fill": img[..., 3] > 0.5, "centre": jnp.stack(centres),
            "count": jnp.asarray(counts, jnp.float32), "pixel": jnp.asarray(pixels, jnp.float32)}


def make_stacked_eval(stack, grid):
    """evaluate(x, s, alive, k) for one swarm and shape index k (may be traced): target.make_eval against shape k's target."""
    l1l2 = tgt.l1l2

    def evaluate(x, s, alive, k):
        mean = jnp.sum(x * alive[:, None], axis=0) / jnp.maximum(jnp.sum(alive), 1.0)
        img = tgt.splat(x - mean + stack["centre"][k], npa.colour(s), stack["pixel"][k], grid,
                        total=stack["count"][k], weight=alive)
        den_diff = l1l2(img[..., 3:] - stack["den"][k])
        colour = jnp.mean(l1l2(img[..., :3] - stack["rgb"][k]) * jnp.exp(-den_diff))
        density = jnp.mean(den_diff)
        fill = img[..., 3] > 0.5
        iou = jnp.sum(fill & stack["fill"][k]) / jnp.maximum(jnp.sum(fill | stack["fill"][k]), 1)
        return {"density": density, "colour": colour, "loss": tgt.DENSITY_WEIGHT * density + tgt.COLOUR_WEIGHT * colour,
                "iou": iou}

    return evaluate


def _roll(cell, params, eps, eps0, n0, block, steps, x, s, mem, alive, key):
    """Run one swarm for `steps` steps; deleted particles neither move nor are seen."""
    def body(carry, k):
        x, s, mem = carry
        update = jax.random.bernoulli(k, npa.UPDATE_PROB, (x.shape[0], 1)).astype(x.dtype) * alive[:, None]
        x, s, mem, _ = cell.step(params, x, s, mem, update, eps, eps0, block, n0, alive)
        return (x, s, mem), None

    (x, s, mem), _ = lax.scan(body, (x, s, mem), jax.random.split(key, steps))
    return x, s, mem


def _mean_tree(trees):
    """Elementwise mean of nested dicts/lists of numbers, leaving non-numeric leaves (level lists) from the first."""
    first = trees[0]
    if isinstance(first, dict):
        return {k: _mean_tree([t[k] for t in trees]) for k in first}
    return np.mean(np.array(trees, dtype=np.float64), axis=0).tolist()


def run(cell, params, eps, eps0, n, cfg, protocol=PROTOCOL, block=None):
    """Score a cell on every shape in cfg["shapes"]. Returns a JSON-friendly dict (see the module docstring)."""
    names = list(cfg["shapes"])
    num_shapes = len(names)
    stack = stack_targets(names, protocol["grid"])
    score = jax.jit(jax.vmap(make_stacked_eval(stack, protocol["grid"]), in_axes=(0, 0, 0, None)))
    seeds, levels = protocol["seeds"], protocol["levels"]
    n_seeds, n_levels = len(seeds), len(next(iter(levels.values())))
    batch = n_seeds * n_levels          # every rollout is batch-sized, so each step count compiles once (for all shapes)

    # layout: index = level * n_seeds + seed, so reshape(n_levels, n_seeds) recovers the grid
    seed_keys = jnp.tile(jnp.stack([jax.random.PRNGKey(s) for s in seeds]), (n_levels, 1))
    make_seeds = jax.jit(lambda k: jax.vmap(lambda key: seed_swarm(key, n, 2 * eps, k, num_shapes))(seed_keys))
    mem0 = jnp.zeros((batch, n, cell.mem_dims))
    alive0 = jnp.ones((batch, n))

    @partial(jax.jit, static_argnames=("steps",))
    def roll(x, s, mem, alive, key, steps):
        keys = jax.random.split(key, batch)
        return jax.vmap(lambda *a: _roll(cell, params, eps, eps0, n, block, steps, *a))(x, s, mem, alive, keys)

    hurt = {kind: jax.jit(jax.vmap(partial(damage.apply, kind, eps=eps))) for kind in damage.KINDS}
    rollout_key = jax.random.PRNGKey(7_000_000 + protocol["version"])
    g, r, long_ = protocol["grow"], protocol["recover"], protocol["long"]

    def run_shape(k):
        def metrics(x, s, alive):
            m = score(x, s, alive, k)
            return {key: np.asarray(v).reshape(n_levels, n_seeds)[0] for key, v in m.items()}   # first row: the seed grid

        x0, s0 = make_seeds(k)
        # grow, then run the undamaged control on to the recovery horizon and to the long horizon
        x, s, mem = roll(x0, s0, mem0, alive0, jax.random.fold_in(rollout_key, 0), g)
        pre = (x, s, mem)
        m_grown = metrics(x, s, alive0)
        xc, sc, memc = roll(*pre, alive0, jax.random.fold_in(rollout_key, 1), r)
        m_ctrl = metrics(xc, sc, alive0)
        xl, sl, _ = roll(xc, sc, memc, alive0, jax.random.fold_in(rollout_key, 2), long_ - g - r)
        m_long = metrics(xl, sl, alive0)

        out = {"growth": {"loss_grown": m_grown["loss"].tolist(), "loss_control": m_ctrl["loss"].tolist(),
                          "iou_control": m_ctrl["iou"].tolist()},
               "stability": {"loss_long": m_long["loss"].tolist()}, "damage": {}}

        for ki, kind in enumerate(damage.KINDS):
            lv = jnp.repeat(jnp.asarray(levels[kind], dtype=jnp.float32), n_seeds)
            dkeys = jax.random.split(jax.random.fold_in(jax.random.PRNGKey(8_000_000), ki), batch)
            dkeys = jnp.tile(dkeys[:n_seeds], (n_levels, 1))        # same centre for every severity of a seed
            xd, sd, memd, alived = hurt[kind](lv, dkeys, *pre, alive0)
            m_hurt = score(xd, sd, alived, k)
            xr, sr, _ = roll(xd, sd, memd, alived, jax.random.fold_in(rollout_key, 1), r)   # same update masks as the control: paired
            m_end = score(xr, sr, alived, k)
            loss_end = np.asarray(m_end["loss"]).reshape(n_levels, n_seeds)
            out["damage"][kind] = {
                "levels": list(levels[kind]),
                "loss_after_damage": np.asarray(m_hurt["loss"]).reshape(n_levels, n_seeds).tolist(),
                "loss_final": loss_end.tolist(),
                "excess": (loss_end - m_ctrl["loss"][None, :]).tolist(),
                "iou_final": np.asarray(m_end["iou"]).reshape(n_levels, n_seeds).tolist(),
                "alive_final": np.asarray(alived).reshape(n_levels, n_seeds, n).sum(-1).mean(-1).tolist(),
            }

        finals = np.array([out["damage"][kd]["loss_final"] for kd in damage.KINDS])
        excess = np.array([out["damage"][kd]["excess"] for kd in damage.KINDS])
        finite = bool(np.isfinite(finals).all() and np.isfinite(m_long["loss"]).all())
        big = 1e3
        out["summary"] = {
            "score": float(np.nan_to_num(finals, nan=big, posinf=big).mean()) if finite else big,
            "excess": float(np.nan_to_num(excess, nan=big, posinf=big).mean()) if finite else big,
            "per_kind_score": {kd: float(np.nan_to_num(finals[i], nan=big, posinf=big).mean()) for i, kd in enumerate(damage.KINDS)},
            "per_kind_excess": {kd: float(np.nan_to_num(excess[i], nan=big, posinf=big).mean()) for i, kd in enumerate(damage.KINDS)},
            "control_loss": float(np.nan_to_num(m_ctrl["loss"], nan=big).mean()),
            "control_iou": float(np.nan_to_num(m_ctrl["iou"], nan=0.0).mean()),
            "drift": float(np.nan_to_num(m_long["loss"], nan=big).mean() - np.nan_to_num(m_ctrl["loss"], nan=big).mean()),
            "finite": finite,
        }
        return out

    per_shape = {name: run_shape(jnp.int32(k)) for k, name in enumerate(names)}
    results = list(per_shape.values())
    summaries = [res["summary"] for res in results]
    mean = lambda key: float(np.mean([sm[key] for sm in summaries]))
    out = {
        "protocol": {key: protocol[key] for key in ("version", "grow", "recover", "long", "grid")},
        "shapes": names,
        "growth": _mean_tree([res["growth"] for res in results]),
        "stability": _mean_tree([res["stability"] for res in results]),
        "damage": {kd: {**_mean_tree([{f: v for f, v in res["damage"][kd].items() if f != "levels"} for res in results]),
                        "levels": list(levels[kd])} for kd in damage.KINDS},
        "per_shape": per_shape,
        "summary": {
            "score": mean("score"),
            "excess": mean("excess"),
            "per_kind_score": {kd: float(np.mean([sm["per_kind_score"][kd] for sm in summaries])) for kd in damage.KINDS},
            "per_kind_excess": {kd: float(np.mean([sm["per_kind_excess"][kd] for sm in summaries])) for kd in damage.KINDS},
            "control_loss": mean("control_loss"),
            "control_iou": mean("control_iou"),
            "drift": mean("drift"),
            "finite": all(sm["finite"] for sm in summaries),
            "per_shape_score": {name: sm["score"] for name, sm in zip(names, summaries)},
            "per_shape_excess": {name: sm["excess"] for name, sm in zip(names, summaries)},
        },
    }
    return out
