"""Checks for the engineered rest-memory cell (cells_rest.py). CPU only.

    python lab/test_rest.py [--eval]      (--eval also runs the tiny end-to-end evaluator pass through run.train)
"""

import sys
import time
from functools import partial
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
import evaluate_agnostic as ea  # noqa: E402
import run  # noqa: E402

results = []


def check(name, ok, detail=""):
    results.append(bool(ok))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""), flush=True)


N = 128
EPS = 0.2 * (1024 / N) ** 0.5
MASK, RGB = ag.get_shape("proc:3")
X0, S0 = (jnp.asarray(a) for a in ag.sample_swarm(MASK, RGB, N, 3))
TOTAL = ag.shape_total(MASK)
REF = ag.render(X0, S0, None, TOTAL, 64)
ONES = jnp.ones(N)


@partial(jax.jit, static_argnums=(0, 5))
def roll(cell, x, s, mem, alive, steps, key, p_update=0.5):
    def body(c, k):
        x, s, mem = c
        upd = jax.random.bernoulli(k, p_update, (N, 1)).astype(x.dtype) * alive[:, None]
        x, s, mem, mv = cell.step({}, x, s, mem, upd, EPS, EPS, None, N, alive)
        return (x, s, mem), mv
    (x, s, mem), mv = jax.lax.scan(body, (x, s, mem), jax.random.split(key, steps))
    return x, s, mem, mv


def make(**cfg):
    return cells.build(kind="rest", **cfg)


def warmed(cell, steps=70, seed=0):
    return roll(cell, X0, S0, jnp.zeros((N, cell.mem_dims)), ONES, steps, jax.random.PRNGKey(seed))[:3]


def mean_disp(x):
    return float(jnp.mean(jnp.sqrt(jnp.sum((x - X0) ** 2, -1))))


def col_err(s, idx=None):
    e = jnp.abs(s[:, -3:] - S0[:, -3:]).mean(-1)
    return float(e.mean() if idx is None else e[idx].mean())


def loss(x, s, alive=ONES):
    return float(ag.score(x, s, alive, REF, TOTAL)["loss"])


# 1. interface, runs, finite
for K in (6, 3):
    for frame in ("world", "distance"):
        cell = make(K=K, frame=frame)
        x, s, mem, mv = roll(cell, X0, S0, jnp.zeros((N, cell.mem_dims)), ONES, 20, jax.random.PRNGKey(1))
        ok = all(bool(jnp.isfinite(v).all()) for v in (x, s, mem, mv))
        check(f"K={K} frame={frame}: runs and is finite", ok and mem.shape == (N, 4 + 3 * K))
        check(f"K={K} frame={frame}: mem_dims = 4 + 3K, init = {{}}, param_count 0",
              cell.mem_dims == 4 + 3 * K and cell.init(jax.random.PRNGKey(0)) == {} and cell.param_count({}) == 0)
check("cells.build routes {'kind': 'rest'} to cells_rest", type(make()).__module__ == "cells_rest")
try:
    make(bogus=1)
    bad = False
except ValueError:
    bad = True
check("unknown options raise", bad)

# 2. imprint happens exactly once (every particle updates every tick, so all imprint together)
cell = make()
mem = jnp.zeros((N, cell.mem_dims))
x, s = X0, S0
snap, ages, changed_after, changed_before = None, [], 0, 0
prev = mem
for t in range(1, 31):
    x, s, mem, _ = cell.step({}, x, s, mem, jnp.ones((N, 1)), EPS, EPS, None, N, ONES)
    ages.append(float(mem[0, 0]))
    if t < 16:
        changed_before += int(bool(jnp.any(mem[:, 1:] != 0)))
    if t == 16:
        snap = mem[:, 1:]
    elif t > 16:
        changed_after += int(bool(jnp.any(jnp.abs(mem[:, 1:] - snap) > 0)))
nb = float((mem[:, 4:10] > -0.5).sum(-1).mean())
check("memory (colour, ids, offsets) is all zero before the imprint update", changed_before == 0)
check("imprint at update 16 stores neighbours/offsets/colour; never rewritten afterwards", float(jnp.abs(snap).sum()) > 0 and changed_after == 0)
check("age counts own updates and is capped at imprint_at + 1", ages[14] == 15 and ages[15] == 16 and ages[-1] == 17, f"{ages[14]}, {ages[15]}, {ages[-1]}")
check("imprinted K=6 neighbour slots are filled (alive, within eps)", nb > 5.5, f"mean filled slots {nb:.2f}")
ids = np.asarray(mem[:, 4:10]).astype(int)
dd = np.linalg.norm(np.asarray(X0)[np.clip(ids, 0, N - 1)] - np.asarray(X0)[:, None], axis=-1)
check("remembered neighbours are within eps and never the particle itself",
      bool(((dd < EPS) | (ids < 0)).all()) and not (ids == np.arange(N)[:, None]).any())
check("the stored own colour equals the particle's colour at imprint", float(jnp.abs(mem[:, 1:4] - S0[:, -3:]).max()) < 1e-6)
# a particle that updates less often imprints later: with update probability 0.5, age lags
_, _, mem_h, _ = roll(cell, X0, S0, jnp.zeros((N, cell.mem_dims)), ONES, 32, jax.random.PRNGKey(3))
check("with 50% updates after 32 ticks only some have imprinted (age counts the particle's own updates)",
      0.1 < float((mem_h[:, 0] >= 16).mean()) < 0.9 and float(mem_h[:, 0].max()) <= 17, f"{float((mem_h[:, 0] >= 16).mean()):.2f} imprinted")

# 3-9. behaviour, world and distance frames
KEY = jax.random.PRNGKey(11)
for frame in ("world", "distance"):
    cell = make(frame=frame)
    x1, s1, m1 = warmed(cell)
    imprinted = float((m1[:, 0] >= 16).mean())
    # no damage: stays put
    _, _, _, mv = roll(cell, x1, s1, m1, ONES, 40, jax.random.PRNGKey(4))
    per_update = float(mv.mean()) / EPS
    xs, ss, _, _ = roll(cell, x1, s1, m1, ONES, 40, jax.random.PRNGKey(4))
    check(f"[{frame}] after warm-up all imprinted ({imprinted:.2f}); undamaged swarm stays put",
          imprinted == 1.0 and per_update < 0.02 and float(jnp.linalg.norm(xs - x1, axis=-1).mean()) < 0.05 * EPS,
          f"mean move per tick {per_update:.4f} eps, displacement over 40 ticks {float(jnp.linalg.norm(xs - x1, axis=-1).mean()) / EPS:.3f} eps")
    check(f"[{frame}] undamaged colours unchanged", float(jnp.abs(ss - s1).max()) < 1e-5)

    # noise
    xd, sd, md, ad = damage.apply("noise", 0.5, KEY, x1, s1, m1, ONES, EPS)
    xr, sr, _, _ = roll(cell, xd, sd, md, ONES, 60, jax.random.PRNGKey(5))
    d0, d1 = mean_disp(xd), mean_disp(xr)
    check(f"[{frame}] noise: positions return toward the original configuration", d1 < 0.6 * d0, f"distance {d0 / EPS:.3f} -> {d1 / EPS:.3f} eps")
    check(f"[{frame}] noise: colour noise is removed", col_err(sr) < 0.3 * col_err(sd), f"colour error {col_err(sd):.3f} -> {col_err(sr):.3f}")

    # shuffle (states AND memory permuted within a patch). EXPECTED FAILURE MODE of this rule: memory is bound to a slot, so a
    # shuffled particle goes to where its memory's OWNER stood while the unshuffled neighbours (who still remember this slot at
    # its old place) pull it back; the colours follow the memory, so they do not return either. The swarm collapses locally.
    xd, sd, md, ad = damage.apply("shuffle", 0.4, KEY, x1, s1, m1, ONES, EPS)
    hit = np.asarray(jnp.any(sd != s1, axis=-1))
    xr, sr, _, _ = roll(cell, xd, sd, md, ONES, 120, jax.random.PRNGKey(6))
    err_after = float(jnp.abs(sd[:, -3:] - S0[:, -3:]).mean(-1)[hit].mean())
    err_end = float(jnp.abs(sr[:, -3:] - S0[:, -3:]).mean(-1)[hit].mean())
    l0, l1 = loss(xd, sd), loss(xr, sr)
    check(f"[{frame}] shuffle: stays finite; documented limitation, NOT repaired and made worse than doing nothing ({hit.sum()} shuffled)",
          bool(jnp.isfinite(xr).all()) and l1 > l0 and err_end > 0.5 * err_after,
          f"loss {l0:.3f} -> {l1:.3f}; per-particle colour error {err_after:.3f} -> {err_end:.3f}")

    # wipe
    xd, sd, md, ad = damage.apply("wipe", 0.4, KEY, x1, s1, m1, ONES, EPS)
    hit = np.asarray(jnp.all(sd == 0, axis=-1))
    xr, sr, mr, _ = roll(cell, xd, sd, md, ONES, 120, jax.random.PRNGKey(7))
    e0, e1 = col_err(sd, hit), col_err(sr, hit)
    check(f"[{frame}] wipe: wiped colours are refilled from the surroundings ({hit.sum()} wiped)", e1 < 0.4 * e0,
          f"colour error of wiped particles {e0:.3f} -> {e1:.3f}; imprinted again {float((mr[hit, 0] >= 16).mean()):.2f}")

    # drag below eps returns; beyond sense_r stays stranded
    d_small = 0.5 * EPS
    xd, sd, md, ad = damage.apply("drag", d_small, KEY, x1, s1, m1, ONES, EPS)
    hit = np.asarray(jnp.any(xd != x1, axis=-1))
    xr, _, _, _ = roll(cell, xd, sd, md, ONES, 120, jax.random.PRNGKey(8))
    dd0 = float(jnp.linalg.norm(xd - x1, axis=-1)[hit].mean())
    dd1 = float(jnp.linalg.norm(xr - x1, axis=-1)[hit].mean())
    check(f"[{frame}] drag by {d_small / EPS:.1f} eps: the patch ({hit.sum()} particles) returns", dd1 < 0.5 * dd0, f"{dd0 / EPS:.2f} -> {dd1 / EPS:.2f} eps")
    d_big = 2.0 * EPS
    xd, sd, md, ad = damage.apply("drag", d_big, KEY, x1, s1, m1, ONES, EPS)
    xr, _, _, _ = roll(cell, xd, sd, md, ONES, 120, jax.random.PRNGKey(8))
    dd0 = float(jnp.linalg.norm(xd - x1, axis=-1)[hit].mean())
    dd1 = float(jnp.linalg.norm(xr - x1, axis=-1)[hit].mean())
    check(f"[{frame}] drag by {d_big / EPS:.1f} eps (beyond sense_r): the patch stays stranded (EXPECTED: no memory link is sensed)",
          dd1 > 0.6 * dd0, f"{dd0 / EPS:.2f} -> {dd1 / EPS:.2f} eps")

    # cut: the hole stays
    xd, sd, md, ad = damage.apply("cut", 0.3, KEY, x1, s1, m1, ONES, EPS)
    xr, _, _, _ = roll(cell, xd, sd, md, ad, 120, jax.random.PRNGKey(9))
    gone = np.asarray(ad == 0)
    c0 = np.asarray(x1)[gone].mean(0)
    inside = float((jnp.linalg.norm(xr - c0, axis=-1) * (ad > 0) + 9 * (ad == 0) < 0.15).sum())
    check(f"[{frame}] cut leaves the hole (EXPECTED: nothing regrows; {int(gone.sum())} deleted, {int(inside)} survivors moved into it)",
          gone.sum() > 0 and inside == 0 and bool(jnp.isfinite(xr).all()))
    check(f"[{frame}] deleted particles neither move nor change", bool(jnp.all(xr[gone] == xd[gone])))

# 10. optional: the tiny end-to-end evaluator pass
if "--eval" in sys.argv:
    cfg = {"name": "rest_eval", "particles": 128, "cell": {"kind": "rest"},
           "train": {"mode": "agnostic", "iters": 0, "batch": 2, "grid": 64}}
    tiny = {"warm": 8, "recover": 8, "emoji": ("lizard",), "proc_seeds": (0, 1), "sample_seeds": (1,), "grid": 64, "chunk": 2}
    t0 = time.time()
    cell, params, eps, curve, secs = run.train(cfg, log=lambda m: None)
    res = ea.run(cell, params, eps, eps, 128, protocol=tiny)
    sm = res["summary"]
    check("run.train with iters=0 and an empty parameter dict works, and the evaluator accepts the rest cell",
          params == {} and curve == [] and sm["finite"] and cell.param_count(params) == 0, f"{time.time() - t0:.0f}s")

print(f"\n{sum(results)}/{len(results)} rest checks passed")
raise SystemExit(0 if all(results) else 1)
