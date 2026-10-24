"""Checks for the multi-shape trainer and evaluator (lab/trainer_multishape.py, lab/evaluate_multi.py). CPU only.

    python lab/test_multishape.py
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
import cells  # noqa: E402
import damage  # noqa: E402
import evaluate  # noqa: E402
import evaluate_multi as em  # noqa: E402
import npa  # noqa: E402
import run  # noqa: E402
import shapes  # noqa: E402
import target as tgt  # noqa: E402
import trainer_multishape as tm  # noqa: E402

results = []


def check(name, ok, detail=""):
    results.append(bool(ok))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))


N, GRID = 128, 64
EPS = 0.2 * (1024 / N) ** 0.5
SHAPES = ["lizard", "star", "bone", "butterfly"]
GATED = {"block": "gated", "depth": 3, "norm": "rms", "mem": 4}
DAMAGE = [{"kind": "wipe", "level": [0.1, 0.3]}, {"kind": "cut", "level": [0.1, 0.3], "p": 0.5}]

# 1. seed codes: one-hot in channels 0..K-1 for the right shape, zero elsewhere, positions as npa.seed
K = len(SHAPES)
key = jax.random.PRNGKey(0)
ok, same_x = True, True
for k in range(K):
    x, s = em.seed_swarm(key, N, 2 * EPS, k, K)
    expect = np.zeros((N, npa.STATE_DIMS), np.float32)
    expect[:, k] = 1.0
    ok &= bool(np.array_equal(np.asarray(s), expect))
    same_x &= bool(np.array_equal(np.asarray(x), np.asarray(npa.seed(key, N, 2 * EPS)[0])))
check("seed codes are one-hot in channel `shape index`, zero in every other channel", ok)
check("the seed positions are npa.seed's", same_x)
_, s_scaled = em.seed_swarm(key, N, 2 * EPS, jnp.int32(2), K, 0.5)
check("a traced shape index and a scale work", float(s_scaled[0, 2]) == 0.5 and float(jnp.abs(s_scaled).sum()) == 0.5 * N)
try:
    em.seed_swarm(key, N, 2 * EPS, 0, 14)
    check("more than 13 shapes are refused (the code must not reach the colour channels)", False)
except AssertionError:
    check("more than 13 shapes are refused (the code must not reach the colour channels)", True)
x, s = em.seed_swarm(key, N, 2 * EPS, 1, K)
_, sw, _, _ = damage.apply("wipe", 0.5, jax.random.PRNGKey(1), x, s, jnp.zeros((N, 0)), jnp.ones(N), EPS)
wiped = np.asarray(jnp.all(sw == 0, axis=-1))
check("wipe damage zeroes the code in the wiped particles and leaves it elsewhere", 0 < wiped.sum() < N
      and bool(np.all(np.asarray(sw)[~wiped, 1] == 1.0)))

# 2. the stacked loss is target.make_loss against the right target, per batch element
stack = em.stack_targets(SHAPES, GRID)
loss_fn = tm.make_multi_loss(stack, GRID)
kx = jax.random.split(jax.random.PRNGKey(5), 3)
bx = 0.4 * jax.random.normal(kx[0], (4, N, 2))
bs = 0.3 * jax.random.normal(kx[1], (4, N, npa.STATE_DIMS))
balive = jnp.ones((4, N)).at[1, :20].set(0.0).at[3, 50:70].set(0.0)
idx = jnp.array([0, 1, 2, 3])
total, parts = loss_fn(bx, bs, jnp.float32(0.7), balive, idx)
gap = 0.0
for b in range(4):
    single = tgt.make_loss(tgt.load_target(shapes.path(SHAPES[b])), GRID)
    _, p = single(bx[b:b + 1], bs[b:b + 1], jnp.float32(0.7), balive[b:b + 1])
    gap = max(gap, abs(float(parts["elem"][b]) - float(p["splat"])))
check("per-element splat loss equals target.make_loss for that element's own shape (4 shapes, alive mask)", gap < 1e-4,
      f"max gap {gap:.1e}")
single = tgt.make_loss(tgt.load_target(shapes.path("star")), GRID)
t_ref, _ = single(bx, bs, jnp.float32(0.7), balive)
t_got, _ = loss_fn(bx, bs, jnp.float32(0.7), balive, jnp.array([1, 1, 1, 1]))
check("a batch of one shape gives make_loss's total (regularisers and move penalty included)", abs(float(t_ref - t_got)) < 1e-4,
      f"{float(t_ref):.5f} vs {float(t_got):.5f}")
check("different shapes give different losses for the same swarm", abs(float(parts["elem"][0] - loss_fn(bx, bs, jnp.float32(0.), balive, jnp.array([1, 1, 1, 1]))[1]["elem"][0])) > 1e-3)

# 3. K=1 'lizard' reproduces run.train's first-iteration loss (exactly, with the code switched off; with it on the code only
#    changes what the untrained cell sees)
base = {"name": "t", "particles": N, "seed": 3, "cell": {"block": "residual", "depth": 3, "norm": "rms", "zero_head": False},
        "train": {"iters": 1, "batch": 2, "steps": [4, 8], "grid": GRID, "damage": DAMAGE}}
_, _, _, ref_curve, _ = run.train(base, log=lambda m: None)
single_cfg = {**base, "shapes": ["lizard"], "train": {**base["train"], "mode": "multishape", "code_scale": 0.0}}
_, _, _, c0, _ = run.train(single_cfg, log=lambda m: None)       # also tests the run.train dispatch
check("K=1 lizard, code off: first-iteration loss, density and colour equal run.train's",
      all(abs(c0[0][f] - ref_curve[0][f]) < 1e-4 for f in ("loss", "density", "colour")),
      f"{c0[0]['loss']:.6f} vs {ref_curve[0]['loss']:.6f}")
code_cfg = {**single_cfg, "train": {**single_cfg["train"], "code_scale": 1.0}}
_, _, _, c1, _ = tm.train(code_cfg, log=lambda m: None)
check("K=1 lizard, code on: first-iteration loss is finite and within 5% of run.train's (only the code channel differs)",
      np.isfinite(c1[0]["loss"]) and abs(c1[0]["loss"] - ref_curve[0]["loss"]) < 0.05 * abs(ref_curve[0]["loss"]),
      f"{c1[0]['loss']:.6f} vs {ref_curve[0]['loss']:.6f}")

# 4. the evaluator, with a tiny protocol
tiny = {**em.PROTOCOL, "grow": 6, "recover": 6, "long": 20, "seeds": (1, 2), "grid": GRID,
        "levels": {k: (0.0, 0.3) for k in damage.KINDS}}
cell = cells.Cell()
p = cell.init(jax.random.PRNGKey(1))
t0 = time.time()
res = em.run(cell, p, EPS, EPS, N, {"shapes": SHAPES}, protocol=tiny)
wall = time.time() - t0
sm = res["summary"]
ref_keys = {"score", "excess", "per_kind_score", "per_kind_excess", "control_loss", "control_iou", "drift", "finite"}
check("evaluator summary has evaluate.py's keys plus per_shape_score / per_shape_excess",
      set(sm) == ref_keys | {"per_shape_score", "per_shape_excess"} and set(sm["per_shape_score"]) == set(SHAPES)
      and set(sm["per_shape_excess"]) == set(SHAPES) and set(res["per_shape"]) == set(SHAPES))
vals = [sm["score"], sm["excess"], sm["control_loss"], sm["control_iou"], sm["drift"], *sm["per_kind_score"].values(),
        *sm["per_kind_excess"].values(), *sm["per_shape_score"].values(), *sm["per_shape_excess"].values()]
check("evaluator returns finite values", sm["finite"] is True and all(np.isfinite(v) for v in vals),
      f"score {sm['score']:.3f}, {wall:.0f}s for K={K}, N={N}, tiny protocol (includes compilation)")
check("per-kind keys are the damage kinds", list(sm["per_kind_score"]) == list(damage.KINDS))
check("score is the mean of the per-shape scores", abs(sm["score"] - np.mean(list(sm["per_shape_score"].values()))) < 1e-9)
zero = max(abs(v) for sh in res["per_shape"].values() for kd in damage.KINDS for v in sh["damage"][kd]["excess"][0])
check("zero-strength damage has exactly zero excess over its control, for every shape", zero < 1e-5, f"max {zero:.1e}")
check("top-level growth / stability / damage follow evaluate.run's layout",
      set(res["damage"]) == set(damage.KINDS) and len(res["damage"]["wipe"]["loss_final"]) == 2
      and len(res["damage"]["wipe"]["loss_final"][0]) == 2 and len(res["growth"]["loss_control"]) == 2)
check("cutting really removes particles", res["per_shape"]["star"]["damage"]["cut"]["alive_final"][1] < N)
check("different shapes get different scores (their own targets)", len({round(v, 5) for v in sm["per_shape_score"].values()}) == K)
fp = em.fingerprint()
check("fingerprint covers the evaluator, damage, shapes, target and npa", len(fp) == len(em.IMMUTABLE)
      and all(len(v) == 16 for v in fp.values()))

# 4b. K=1 against the original evaluator: a cell that does nothing (zero head) cannot see the code, so the numbers must match
null = cells.Cell(block="residual", depth=3, zero_head=True)
pn = null.init(jax.random.PRNGKey(2))
ref = evaluate.run(null, pn, EPS, EPS, N, protocol={**evaluate.PROTOCOL, **{k: tiny[k] for k in ("grow", "recover", "long", "seeds", "levels", "grid")}})
got = em.run(null, pn, EPS, EPS, N, {"shapes": ["lizard"]}, protocol=tiny)
d = max(abs(ref["summary"]["score"] - got["summary"]["score"]), abs(ref["summary"]["excess"] - got["summary"]["excess"]),
        abs(ref["summary"]["control_loss"] - got["summary"]["control_loss"]), abs(ref["summary"]["drift"] - got["summary"]["drift"]))
check("K=1 lizard with a do-nothing cell: summary equals evaluate.run's", d < 1e-4, f"max gap {d:.1e}")

# 4c. the code is really in the state when the cell runs: a cell that only reads it can tell the shapes apart
hx, hs = em.seed_swarm(key, N, 2 * EPS, 3, K)
check("the cell input contains the code (state channel 3 is 1 for shape 3)", bool(np.all(np.asarray(npa.perceive(hx, hs, EPS, EPS)[:, 3]) == 1.0)))

# 5. tiny training runs: default cell and a gated cell with memory, K=2, finite, params move
for label, cell_cfg in (("default cell", {}), ("gated cell with memory", GATED)):
    cfg = {"name": "t", "particles": N, "seed": 1, "shapes": ["lizard", "star"], "cell": cell_cfg,
           "train": {"mode": "multishape", "iters": 3, "batch": 2, "steps": [3, 6], "grid": GRID, "damage": DAMAGE}}
    t0 = time.time()
    c, params, eps, curve, secs = run.train(cfg, log=lambda m: None)
    init = c.init(jax.random.PRNGKey(1))
    check(f"training, {label}, K=2: 3 iterations, finite loss and parameters, weights moved",
          len(curve) == 3 and all(np.isfinite(r["loss"]) and np.isfinite(r["density"]) and np.isfinite(r["colour"]) for r in curve)
          and all(bool(jnp.isfinite(v).all()) for v in params.values())
          and any(float(jnp.abs(v - init[k]).max()) > 0 for k, v in params.items())
          and set(curve[0]) == {"iter", "loss", "density", "colour"},
          f"loss {curve[0]['loss']:.3f} -> {curve[-1]['loss']:.3f}, {time.time() - t0:.0f}s")

# 6. a longer pass over the pool logic: seeds are injected every SEED_EVERY iterations, so run past one injection (K=4)
msgs = []
cfg = {"name": "t", "particles": N, "seed": 2, "shapes": SHAPES, "cell": {},
       "train": {"mode": "multishape", "iters": run.SEED_EVERY + 2, "batch": 4, "steps": [2, 3], "grid": GRID, "damage": DAMAGE}}
_, _, _, curve, _ = run.train(cfg, log=msgs.append)
check("K=4 run past a seed injection stays finite and logs per-shape losses", len(curve) == run.SEED_EVERY + 2
      and all(np.isfinite(r["loss"]) for r in curve) and any("lizard" in m or "star" in m or "bone" in m for m in msgs))

print(f"\n{sum(results)}/{len(results)} multishape checks passed")
raise SystemExit(0 if all(results) else 1)
