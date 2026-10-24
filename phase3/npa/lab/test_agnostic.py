"""Checks for the shape-agnostic task (agnostic.py, evaluate_agnostic.py, trainer_agnostic.py). CPU only, a few minutes.

    python lab/test_agnostic.py
"""

import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
import agnostic as ag  # noqa: E402
import cells  # noqa: E402
import damage  # noqa: E402
import evaluate  # noqa: E402
import evaluate_agnostic as ea  # noqa: E402
import run  # noqa: E402

results = []


def check(name, ok, detail=""):
    results.append(bool(ok))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))


# 1. shape generators: deterministic, in frame, connected, coloured
seeds = list(ag.HELDOUT_PROC_SEEDS) + [1000, 1001, 1002, 54321]
same = all(np.array_equal(ag.proc_shape(sd)[0], ag.proc_shape.__wrapped__(sd)[0])
           and np.array_equal(ag.proc_shape(sd)[1], ag.proc_shape.__wrapped__(sd)[1]) for sd in seeds)
check("procedural shapes are deterministic in the seed", same)
check("different seeds give different shapes", not np.array_equal(ag.proc_shape(0)[0], ag.proc_shape(1)[0]))
frame_ok = True
for name in list(ag.HELDOUT_EMOJI + ag.TRAIN_EMOJI) + [f"proc:{sd}" for sd in seeds]:
    m, r = ag.get_shape(name)
    border = np.zeros_like(m)
    border[:2], border[-2:], border[:, :2], border[:, -2:] = True, True, True, True
    frame_ok &= bool(m.shape == (128, 128) and r.shape == (128, 128, 3) and 1500 < m.sum() < 12000 and not (m & border).any()
                     and np.isfinite(r).all() and r.min() >= 0 and r.max() <= 1)
check("every shape is a non-empty 128x128 mask well inside the frame with rgb in [0, 1]", frame_ok)
check("procedural shapes are one connected piece", all(ag.n_pieces(ag.proc_shape(sd)[0]) == 1 for sd in seeds))
thin = [ag.proc_shape(sd)[0] for sd in range(1000, 1060)]
check("the genus of training shapes covers 0..3", {ag.genus(m) for m in thin} >= {0, 1, 2, 3})

# 2. genus labels of the held-out set
genus_ok = all(ag.genus(ag.proc_shape(sd)[0]) == g for sd, g in ag.PROC_GENUS.items())
check("documented genus of held-out seeds 0..19 matches the flood fill", genus_ok and len(ag.PROC_GENUS) == 20)
check("held-out procedural suite has genus 0, 1, 2 and 3", set(ag.PROC_GENUS.values()) == {0, 1, 2, 3},
      str({g: [sd for sd, v in ag.PROC_GENUS.items() if v == g] for g in range(4)}))
ring = np.zeros((128, 128), bool)
ring[20:100, 20:100] = True
ring[40:60, 40:60] = False
ring[70:80, 70:80] = False
check("genus counts a synthetic mask with two holes as 2", ag.genus(ring) == 2)

# 3. sampling
for n in (128, 1024):
    ok, near = True, True
    for name in ("lizard", "donut", "proc:1", "proc:14", "proc:1003"):
        m, r = ag.get_shape(name)
        x, s = ag.sample_swarm(m, r, n, 5)
        x2, s2 = ag.sample_swarm(m, r, n, 5)
        inside = ag._lookup(m, x).all()
        want = ag._bilinear(r, x) - 0.5
        ok &= bool(x.shape == (n, 2) and s.shape == (n, 16) and inside and np.array_equal(x, x2) and np.array_equal(s, s2)
                   and np.abs(s[:, -3:] - want).max() < 1e-5 and np.abs(s[:, :-3]).max() == 0)
        d = np.sqrt(((x[:, None] - x[None]) ** 2).sum(-1) + np.eye(n) * 9).min(1)
        spacing = np.sqrt(m.sum() * (2 / 128) ** 2 / n * 2 / np.sqrt(3))
        near &= bool(d.min() > 0.25 * spacing and np.median(d) > 0.7 * spacing)
    check(f"sample_swarm: exactly {n} in-mask particles, deterministic, colours from the picture", ok)
    check(f"sample_swarm: near-uniform packing at n={n}", near)
x_a, _ = ag.sample_swarm(*ag.get_shape("star"), 128, 1)
x_b, _ = ag.sample_swarm(*ag.get_shape("star"), 128, 2)
check("different sampling seeds give different swarms", not np.array_equal(x_a, x_b))

# 4. score
m, r = ag.get_shape("proc:3")
x, s = ag.sample_swarm(m, r, 256, 3)
total = ag.shape_total(m)
ref = ag.render(jnp.asarray(x), jnp.asarray(s), None, total, 64)
sc = ag.score(jnp.asarray(x), jnp.asarray(s), jnp.ones(256), ref, total)
check("a swarm scored against its own reference has loss < 1e-3 and iou 1", float(sc["loss"]) < 1e-3 and float(sc["iou"]) == 1.0,
      f"loss {float(sc['loss']):.1e} iou {float(sc['iou']):.2f}")
sc2 = ag.score(jnp.asarray(x) + 3.0, jnp.asarray(s), jnp.ones(256), ref, total)
check("the score ignores where the swarm sits (centred on its own mean)", float(sc2["loss"]) < 1e-3)
sc3 = ag.score(jnp.asarray(x) * 0.7, jnp.asarray(s), jnp.ones(256), ref, total)
check("a shrunken swarm scores clearly worse", float(sc3["loss"]) > 0.05 and float(sc3["iou"]) < 0.9, f"loss {float(sc3['loss']):.3f}")
x1k, s1k = ag.sample_swarm(m, r, 1024, 3)
ref1k = ag.render(jnp.asarray(x1k), jnp.asarray(s1k), None, total, 64)
med = float(jnp.median(ref1k[..., 3][ref1k[..., 3] > 0.5]))
check("density is about 1 inside the shape at n=1024 (so the iou threshold 0.5 means something)", 0.8 < med < 1.25, f"{med:.2f}")
batch = jax.vmap(lambda a, b: ag.score(a, b, jnp.ones(256), ref, total))(jnp.stack([jnp.asarray(x)] * 2), jnp.stack([jnp.asarray(s)] * 2))
check("score vmaps", batch["loss"].shape == (2,))
alive = jnp.ones(256).at[:60].set(0.0)
check("score with deleted particles is finite", bool(jnp.isfinite(ag.score(jnp.asarray(x), jnp.asarray(s), alive, ref, total)["loss"])))

# 5. evaluator, tiny protocol
N, EPS = 128, 0.2 * (1024 / 128) ** 0.5
cell = cells.build()
params = cell.init(jax.random.PRNGKey(1))
tiny = {"warm": 3, "recover": 3, "emoji": ("lizard",), "proc_seeds": (0, 1), "sample_seeds": (1,), "grid": 64, "chunk": 2}
check("the evaluator's level lists equal evaluate.PROTOCOL's", dict(ea.PROTOCOL["levels"]) == dict(evaluate.PROTOCOL["levels"]))
t0 = time.time()
res = ea.run(cell, params, EPS, EPS, N, protocol=tiny)
sm = res["summary"]
keys = {"score", "excess", "per_kind_score", "per_kind_excess", "control_loss", "control_iou", "drift", "finite",
        "do_nothing_score", "do_nothing_control", "repair_fraction"}
check("evaluator summary has evaluate.py's keys and the new ones", keys <= set(sm) and set(sm["per_kind_score"]) == set(damage.KINDS))
flat = [sm[k] for k in ("score", "excess", "control_loss", "control_iou", "drift", "do_nothing_score")] + list(sm["repair_fraction"].values())
check("evaluator values are finite and repair fractions lie in [-1, 1]", sm["finite"] and np.isfinite(flat).all()
      and all(-1 <= v <= 1 for v in sm["repair_fraction"].values()), f"{time.time() - t0:.0f}s")
check("the do-nothing control is exactly the reference (loss 0)", sm["do_nothing_control"] < 1e-6, f"{sm['do_nothing_control']:.1e}")
check("per_shape has every shape; cutting really removes particles", set(res["per_shape"]) == {"lizard", "proc:0", "proc:1"}
      and res["damage"]["cut"]["alive_final"][-1] < N)

zero = {**tiny, "levels": {k: (0.0, 0.0, 0.0) for k in damage.KINDS}}
nullres = ea.run(ea.NullCell(), {}, EPS, EPS, N, protocol=zero)
ex = max(abs(v) for k in damage.KINDS for row in nullres["damage"][k]["excess"] for v in row)
check("do-nothing under zero-strength damage has zero excess", ex < 1e-5, f"max {ex:.1e}")
modres = ea.run(cell, params, EPS, EPS, N, protocol=zero)
ex = max(abs(v) for k in damage.KINDS for row in modres["damage"][k]["excess"] for v in row)
check("the trained-or-not cell under zero-strength damage has zero excess (paired update masks)", ex < 1e-4, f"max {ex:.1e}")
fp = ea.fingerprint()
check("the evaluator fingerprint hashes its files", len(fp) >= 8 and all(len(v) == 16 for v in fp.values()))

# 6. tiny training runs: default cell and a gated cell with memory
for label, ccfg in (("default cell", {}), ("gated cell with memory", {"block": "gated", "depth": 3, "norm": "rms", "mem": 4})):
    cfg = {"name": "t", "particles": 128, "seed": 2, "cell": ccfg,
           "train": {"mode": "agnostic", "iters": 3, "batch": 2, "grid": 64, "warm": [2, 4], "recover": [3, 5], "warm_loss": 0.5}}
    t0 = time.time()
    cell2, p2, eps, curve, secs = run.train(cfg, log=lambda m: None)       # also checks the run.py dispatch
    moved = any(float(jnp.abs(v - cell2.init(jax.random.PRNGKey(2))[k]).max()) > 0 for k, v in p2.items())
    check(f"tiny agnostic training, {label}: finite loss, finite parameters, parameters moved",
          len(curve) == 3 and all(np.isfinite(c["loss"]) for c in curve) and all(bool(jnp.isfinite(v).all()) for v in p2.values()) and moved,
          f"loss {curve[0]['loss']:.2f} -> {curve[-1]['loss']:.2f}, {time.time() - t0:.0f}s")
r2 = ea.run(cell2, p2, EPS, EPS, N, protocol=tiny)
check("a trained gated cell with memory goes through the evaluator", r2["summary"]["finite"])

print(f"\n{sum(results)}/{len(results)} agnostic checks passed")
raise SystemExit(0 if all(results) else 1)
