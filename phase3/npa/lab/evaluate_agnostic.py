"""The shape-agnostic repair evaluator (task 2b). Experiments may not edit this file; run.py records its hash.

Nothing about a shape is in the weights, so the evaluator uses shapes the model never saw: 5 held-out emoji and 20
procedural shapes (seeds 0..19, branches, holes of genus 0 to 3, thin parts), each sampled with 2 seeds into n particles.

For every swarm:
    start      x0, s0 (zero memory): positions are the shape, colour channels the pattern. The REFERENCE is this swarm.
    warm       WARM undamaged steps: the model may imprint memory and settle. Drifting away from the start costs.
    damage     each of six kinds (lab/damage.py) at three severities, centres shared across severities
    recover    RECOVER more steps; the loss against the start swarm is the score
    control    the same swarm, same update masks, no damage: the loss the model gets for doing nothing wrong

A DO-NOTHING model (a NullCell: x, s, mem unchanged) goes through the identical protocol. It is the floor: the damage
costs that much with no repair at all. repair_fraction = 1 - (score - control) / (do_nothing_score - do_nothing_control),
clipped to [-1, 1], per kind and overall: 1 is a full repair, 0 is as bad as doing nothing, negative is worse.

summary has the keys of evaluate.py's summary (score, excess, per_kind_score, per_kind_excess, control_loss,
control_iou, drift, finite) so lab/report.py works, plus do_nothing_score, do_nothing_control, repair_fraction ...
Here drift = control loss after WARM + RECOVER minus the loss after WARM (positive: the undamaged swarm keeps degrading).

Loss is agnostic.score: density + 5 * colour against the start swarm, as target.make_eval does for the lizard.
Cost: (1 control + 18 damaged) rollouts of RECOVER steps for 50 swarms, one compile per rollout length.
"""

import hashlib
import sys
from functools import lru_cache, partial
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
import agnostic  # noqa: E402
import damage  # noqa: E402
import npa  # noqa: E402

PROTOCOL = {
    "version": 1,
    "warm": 32,           # undamaged steps before the damage
    "recover": 120,       # steps the damaged swarm gets
    "emoji": agnostic.HELDOUT_EMOJI,
    "proc_seeds": agnostic.HELDOUT_PROC_SEEDS,
    "sample_seeds": (910_000, 910_001),     # how each shape is sampled into particles; training never uses these
    "levels": {           # the same lists as evaluate.PROTOCOL["levels"]
        "wipe": (0.2, 0.4, 0.6),
        "cut": (0.12, 0.2, 0.3),
        "drag": (0.2, 0.4, 0.8),
        "clump": (0.3, 0.5, 0.7),
        "shuffle": (0.2, 0.4, 0.6),
        "noise": (0.1, 0.25, 0.5),
    },
    "grid": 128,
    "chunk": 25,          # swarms per rollout batch (memory only; does not change the numbers)
}

IMMUTABLE = ("lab/evaluate_agnostic.py", "lab/agnostic.py", "lab/damage.py", "lab/shapes.py", "target.py", "npa.py",
             "data/lizard_u1f98e.png", *(f"data/shapes/{n}.png" for n in agnostic.HELDOUT_EMOJI if n != "lizard"))
DEFAULT_REPORT_KEYS = ("score", "excess", "per_kind_score", "per_kind_excess", "control_loss", "control_iou", "drift", "finite")


def fingerprint():
    """Hashes of everything an experiment must not change."""
    root = Path(__file__).resolve().parent.parent
    return {f: hashlib.sha256((root / f).read_bytes()).hexdigest()[:16] for f in IMMUTABLE}


class NullCell:
    """The floor: does nothing."""

    def __init__(self, mem_dims=0):
        self.mem_dims = mem_dims

    def step(self, p, x, s, mem, update, eps, eps0, block=None, total=None, mass=None):
        return x, s, mem, jnp.zeros(())


def _roll(cell, params, eps, eps0, n0, block, steps, x, s, mem, alive, key):
    """Run one swarm for `steps` steps; deleted particles neither move nor are seen."""
    def body(carry, k):
        x, s, mem = carry
        update = jax.random.bernoulli(k, npa.UPDATE_PROB, (x.shape[0], 1)).astype(x.dtype) * alive[:, None]
        x, s, mem, _ = cell.step(params, x, s, mem, update, eps, eps0, block, n0, alive)
        return (x, s, mem), None

    (x, s, mem), _ = lax.scan(body, (x, s, mem), jax.random.split(key, steps))
    return x, s, mem


def _chunked(fn, chunk, *arrays):
    """Apply fn to slices of the leading axis (the last slice padded to full size, so fn compiles once); concatenate."""
    total = arrays[0].shape[0]
    outs = []
    for i in range(0, total, chunk):
        part = [a[i:i + chunk] for a in arrays]
        pad = chunk - part[0].shape[0]
        if pad:
            part = [jnp.concatenate([a, jnp.repeat(a[-1:], pad, axis=0)]) for a in part]
        out = fn(*part)
        outs.append(jax.tree_util.tree_map(lambda o: o[:chunk - pad], out))
    return jax.tree_util.tree_map(lambda *o: jnp.concatenate(o), *outs)


def _build_swarms(protocol, n):
    """All held-out swarms, shape-major: index = shape * n_seeds + seed. Returns numpy arrays and labels."""
    suite = agnostic.heldout_suite(protocol["emoji"], protocol["proc_seeds"])
    xs, ss, tot, labels = [], [], [], []
    for sh in suite:
        for sd in protocol["sample_seeds"]:
            x, s = agnostic.sample_swarm(sh["mask"], sh["rgb"], n, int(sd))
            xs.append(x), ss.append(s), tot.append(agnostic.shape_total(sh["mask"]))
            labels.append({"shape": sh["name"], "source": sh["source"], "genus": int(sh["genus"]), "sample_seed": int(sd)})
    return np.stack(xs), np.stack(ss), np.asarray(tot, np.float32), labels


@lru_cache(maxsize=None)
def _scorer():
    return jax.jit(jax.vmap(agnostic.score))


@lru_cache(maxsize=None)
def _damager(kind, eps):
    return jax.jit(jax.vmap(partial(damage.apply, kind, eps=eps), in_axes=(None, 0, 0, 0, 0, 0)))


def _model_pass(cell, params, eps, eps0, n, protocol, x0, s0, total, ref, rollout_key):
    """Warm, control and the six damage kinds for one model. Returns numpy arrays (S = swarms, L = levels)."""
    S, chunk, grid = x0.shape[0], protocol["chunk"], protocol["grid"]
    levels = protocol["levels"]
    mem0 = jnp.zeros((S, n, cell.mem_dims))
    alive0 = jnp.ones((S, n))
    keys_warm = jax.random.split(jax.random.fold_in(rollout_key, 0), S)
    keys_rec = jax.random.split(jax.random.fold_in(rollout_key, 1), S)

    @partial(jax.jit, static_argnames=("steps",))
    def roll(params, x, s, mem, alive, keys, steps):
        return jax.vmap(lambda *a: _roll(cell, params, eps, eps0, n, None, steps, *a))(x, s, mem, alive, keys)

    score = _scorer()

    def roll_c(x, s, mem, alive, keys, steps):
        return _chunked(lambda *a: roll(params, *a, steps=steps), chunk, x, s, mem, alive, keys)

    def metrics(x, s, alive):
        m = _chunked(score, chunk, x, s, alive, ref, jnp.asarray(total))
        return {k: np.asarray(v) for k, v in m.items()}

    xw, sw, memw = roll_c(x0, s0, mem0, alive0, keys_warm, protocol["warm"])
    m_warm = metrics(xw, sw, alive0)
    xc, sc, _ = roll_c(xw, sw, memw, alive0, keys_rec, protocol["recover"])
    m_ctrl = metrics(xc, sc, alive0)
    out = {"warm": m_warm, "control": m_ctrl, "damage": {}, "finite_state": bool(np.isfinite(np.asarray(xc)).all() and np.isfinite(np.asarray(sc)).all())}

    for ki, kind in enumerate(damage.KINDS):
        dkeys = jax.random.split(jax.random.fold_in(jax.random.PRNGKey(8_100_000), ki), S)    # same centre for every severity
        dmg = _damager(kind, float(eps))
        rows = {"after": [], "final": [], "iou": [], "alive": []}
        for lv in levels[kind]:
            xd, sd, memd, alived = dmg(jnp.float32(lv), dkeys, xw, sw, memw, alive0)
            rows["after"].append(metrics(xd, sd, alived)["loss"])
            xr, sr, _ = roll_c(xd, sd, memd, alived, keys_rec, protocol["recover"])     # same update masks as the control
            m_end = metrics(xr, sr, alived)
            rows["final"].append(m_end["loss"])
            rows["iou"].append(m_end["iou"])
            rows["alive"].append(np.asarray(alived).sum(-1))
        out["damage"][kind] = {k: np.stack(v) for k, v in rows.items()}      # each (L, S)
    return out


def _fraction(score, control, dn_score, dn_control):
    """1 - (score - control) / (do_nothing - do_nothing_control), clipped to [-1, 1]; 0 when the floor itself has no damage."""
    denom = dn_score - dn_control
    if not np.isfinite(denom) or denom < 1e-6:
        return 0.0
    return float(np.clip(1.0 - (score - control) / denom, -1.0, 1.0))


def run(cell, params, eps, eps0, n, cfg=None, protocol=PROTOCOL):
    """Score a cell on held-out shapes. Returns a JSON-friendly dict (see the module docstring)."""
    protocol = {**PROTOCOL, **protocol}
    kinds, levels = damage.KINDS, protocol["levels"]
    x0, s0, total, labels = _build_swarms(protocol, n)
    S = len(labels)
    ref = jax.jit(jax.vmap(lambda x, s, t: agnostic.render(x, s, None, t, protocol["grid"])))(jnp.asarray(x0), jnp.asarray(s0), jnp.asarray(total))
    rollout_key = jax.random.PRNGKey(7_100_000 + protocol["version"])
    x0, s0 = jnp.asarray(x0), jnp.asarray(s0)

    mine = _model_pass(cell, params, eps, eps0, n, protocol, x0, s0, total, ref, rollout_key)
    null = _model_pass(NullCell(cell.mem_dims), {}, eps, eps0, n, protocol, x0, s0, total, ref, rollout_key)

    big = 1e3
    clean = lambda a: np.nan_to_num(np.asarray(a, np.float64), nan=big, posinf=big, neginf=big)
    ctrl, dn_ctrl = clean(mine["control"]["loss"]), clean(null["control"]["loss"])      # (S,)
    final = {k: clean(mine["damage"][k]["final"]) for k in kinds}                        # (L, S)
    dn_final = {k: clean(null["damage"][k]["final"]) for k in kinds}
    excess = {k: final[k] - ctrl[None] for k in kinds}
    dn_excess = {k: dn_final[k] - dn_ctrl[None] for k in kinds}
    finite = bool(mine["finite_state"] and all(np.isfinite(mine["damage"][k]["final"]).all() for k in kinds)
                  and np.isfinite(mine["control"]["loss"]).all())

    out = {"protocol": {k: protocol[k] for k in ("version", "warm", "recover", "grid", "emoji", "proc_seeds", "sample_seeds")},
           "swarms": labels, "damage": {}, "per_shape": {}}
    for k in kinds:
        out["damage"][k] = {
            "levels": list(levels[k]),
            "loss_after_damage": clean(mine["damage"][k]["after"]).tolist(),
            "loss_final": final[k].tolist(), "excess": excess[k].tolist(),
            "iou_final": np.asarray(mine["damage"][k]["iou"]).tolist(),
            "alive_final": np.asarray(mine["damage"][k]["alive"]).mean(-1).tolist(),
            "do_nothing_loss_final": dn_final[k].tolist(),
        }

    names = list(dict.fromkeys(l["shape"] for l in labels))
    for nm in names:
        idx = [i for i, l in enumerate(labels) if l["shape"] == nm]
        pk_score = {k: float(final[k][:, idx].mean()) for k in kinds}
        pk_dn = {k: float(dn_final[k][:, idx].mean()) for k in kinds}
        c, dc = float(ctrl[idx].mean()), float(dn_ctrl[idx].mean())
        out["per_shape"][nm] = {
            "source": labels[idx[0]]["source"], "genus": labels[idx[0]]["genus"], "control_loss": c,
            "control_iou": float(np.nan_to_num(mine["control"]["iou"][idx]).mean()),
            "score": float(np.mean(list(pk_score.values()))), "excess": float(np.mean([pk_score[k] - c for k in kinds])),
            "per_kind_excess": {k: pk_score[k] - c for k in kinds},
            "do_nothing_score": float(np.mean(list(pk_dn.values()))), "do_nothing_control": dc,
            "repair_fraction": {k: _fraction(pk_score[k], c, pk_dn[k], dc) for k in kinds},
        }

    def pooled(idx):
        """score / excess / repair fraction over a subset of swarms."""
        c, dc = float(ctrl[idx].mean()), float(dn_ctrl[idx].mean())
        sc = {k: float(final[k][:, idx].mean()) for k in kinds}
        dn = {k: float(dn_final[k][:, idx].mean()) for k in kinds}
        rf = {k: _fraction(sc[k], c, dn[k], dc) for k in kinds}
        return {"score": float(np.mean(list(sc.values()))), "excess": float(np.mean([sc[k] - c for k in kinds])), "control_loss": c,
                "repair_fraction": {**rf, "overall": float(np.mean(list(rf.values())))}}

    per_kind_score = {k: float(final[k].mean()) for k in kinds}
    per_kind_excess = {k: float(excess[k].mean()) for k in kinds}
    rf_kind = {k: _fraction(per_kind_score[k], float(ctrl.mean()), float(dn_final[k].mean()), float(dn_ctrl.mean())) for k in kinds}
    easy = ("wipe", "shuffle", "noise")
    out["summary"] = {
        "score": float(np.mean([final[k].mean() for k in kinds])) if finite else big,
        "excess": float(np.mean([excess[k].mean() for k in kinds])) if finite else big,
        "per_kind_score": per_kind_score, "per_kind_excess": per_kind_excess,
        "control_loss": float(ctrl.mean()),
        "control_iou": float(np.nan_to_num(mine["control"]["iou"], nan=0.0).mean()),
        "drift": float(ctrl.mean() - clean(mine["warm"]["loss"]).mean()),
        "finite": finite,
        "do_nothing_score": float(np.mean([dn_final[k].mean() for k in kinds])),
        "do_nothing_control": float(dn_ctrl.mean()),
        "do_nothing_per_kind_score": {k: float(dn_final[k].mean()) for k in kinds},
        "repair_fraction": {**rf_kind, "overall": float(np.mean(list(rf_kind.values())))},
        # stricter: judged against the START swarm (the do-nothing control, loss 0), so a model that wrecks the shape even
        # without damage (control_loss high) cannot score well; repair_fraction above only credits the damage-specific part
        "restoration": {**{k: _fraction(per_kind_score[k], float(dn_ctrl.mean()), float(dn_final[k].mean()), float(dn_ctrl.mean())) for k in kinds},
                        "overall": float(np.mean([_fraction(per_kind_score[k], float(dn_ctrl.mean()), float(dn_final[k].mean()), float(dn_ctrl.mean())) for k in kinds]))},
        "repair_fraction_easy": float(np.mean([rf_kind[k] for k in easy])),       # wipe, shuffle, noise: the 2b pass criterion is >= 0.5
        "by_source": {src: pooled([i for i, l in enumerate(labels) if l["source"] == src])
                      for src in dict.fromkeys(l["source"] for l in labels)},
        "by_genus": {str(g): pooled([i for i, l in enumerate(labels) if l["genus"] == g])
                     for g in sorted({l["genus"] for l in labels})},
    }
    return out
