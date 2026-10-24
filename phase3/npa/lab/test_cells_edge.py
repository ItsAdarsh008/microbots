"""Checks for the edge-message cell (lab/cells_edge.py).

    python lab/test_cells_edge.py
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
import cells_edge  # noqa: E402
import npa  # noqa: E402
import run  # noqa: E402

results = []


def check(name, ok, detail=""):
    results.append(bool(ok))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))


N, EPS, K = 128, 0.2 * (1024 / 128) ** 0.5, 64
key = jax.random.PRNGKey(0)
x, s = npa.seed(key, N, radius=2 * EPS)
s = s + 0.3 * jax.random.normal(jax.random.PRNGKey(5), s.shape)
on = jnp.ones((N, 1))
half = (jax.random.uniform(jax.random.PRNGKey(6), (N, 1)) < 0.5).astype(jnp.float32)


def make(**cfg):
    c = cells_edge.Cell(K=K, **cfg)
    return c, c.init(jax.random.PRNGKey(1))


def rand_mem(c, n=N):
    return 0.3 * jax.random.normal(jax.random.PRNGKey(7), (n, c.mem_dims))


# 0. build() dispatches to this module, and bad options are rejected
check("cells.build(kind='edge') gives an edge cell", type(cells.build(kind="edge")).__module__ == "cells_edge")
try:
    cells_edge.Cell(bogus=1)
    check("unknown option is rejected", False)
except ValueError:
    check("unknown option is rejected", True)

# 1. every variant runs and is finite
variants = [{"move": mv, "sph": sph, "mem": mem, "norm": nm}
            for mv in ("direct", "radial") for sph in (False, True) for mem in (0, 4) for nm in ("none", "rms")]
for cfg in variants:
    c, p = make(**cfg)
    mem = rand_mem(c)
    out = c.step(p, x, s, mem, half, EPS, EPS, total=N)
    ok = all(bool(jnp.isfinite(o).all()) for o in out) and out[2].shape == (N, c.mem_dims) and out[0].shape == x.shape
    check(f"runs and is finite {cfg}", ok, f"{c.param_count(p):,} parameters")

# the first steps barely move at init (small output layer)
c, p = make()
x1, s1, _, mv = c.step(p, x, s, jnp.zeros((N, 0)), on, EPS, EPS)
check("small output layer: first step moves little", float(mv) < 0.05 * EPS and float(jnp.abs(s1 - s).max()) < 1.0,
      f"mean move {float(mv) / EPS:.4f} eps")
# update mask: particles with update 0 do not change at all
c, p = make(mem=4)
mem0 = rand_mem(c)
x2, s2, m2, _ = c.step(p, x, s, mem0, half, EPS, EPS)
frozen = np.asarray(half[:, 0]) == 0
check("update mask 0 leaves x, s and mem untouched",
      float(jnp.abs(x2 - x)[frozen].max()) == 0 and float(jnp.abs(s2 - s)[frozen].max()) == 0
      and float(jnp.abs(m2 - mem0)[frozen].max()) == 0)

# make the cell non-trivial for the invariance tests: a large head so outputs are not ~0
def loud(cfg):
    c, p = make(**cfg)
    p = dict(p)
    p["w2"] = p["w2"] * 10.0
    if "phi" in p:
        p["phi"] = p["phi"] * 10.0
    return c, p


# 2. translation: ds and mem unchanged, x shifts by exactly the translation, move unchanged
shift = jnp.array([3.7, -2.1])
for cfg in ({"move": "direct", "mem": 4}, {"move": "radial", "mem": 4}, {"sph": True, "mem": 4}):
    c, p = loud(cfg)
    mem = rand_mem(c)
    a = c.step(p, x, s, mem, half, EPS, EPS, total=N)
    b = c.step(p, x + shift, s, mem, half, EPS, EPS, total=N)
    tol = 2e-4 if cfg.get("sph") else 1e-5   # SPH path has float32 round-off in x - x_j
    d_s = float(jnp.abs(a[1] - b[1]).max())
    d_x = float(jnp.abs((b[0] - shift) - a[0]).max())
    d_m = float(jnp.abs(a[2] - b[2]).max())
    check(f"translation leaves ds, mem, move unchanged and shifts x {cfg}",
          d_s < tol and d_m < tol and d_x < 1e-4 and abs(float(a[3] - b[3])) < 1e-5,
          f"ds {d_s:.1e} mem {d_m:.1e} x {d_x:.1e}; ds size {float(jnp.abs(a[1] - s).max()):.2f}")

# 3. permutation equivariance
perm = np.random.default_rng(0).permutation(N)
for cfg in ({"move": "direct", "mem": 4}, {"move": "radial", "mem": 4}, {"sph": True, "mem": 4}):
    c, p = loud(cfg)
    mem = rand_mem(c)
    a = c.step(p, x, s, mem, half, EPS, EPS, total=N)
    b = c.step(p, x[perm], s[perm], mem[perm], half[perm], EPS, EPS, total=N)
    gap = max(float(jnp.abs(a[i][perm] - b[i]).max()) for i in range(3))
    check(f"permuting particles permutes outputs {cfg}", gap < 1e-4 and abs(float(a[3] - b[3])) < 1e-5, f"gap {gap:.1e}")

# 4. a mass-0 particle is invisible
gone = np.arange(20)
alive = jnp.ones(N).at[gone].set(0.0)
keep = np.arange(20, N)
from neighbors import knn  # noqa: E402
_, v_all, _ = knn(x, jnp.ones(N), EPS, K)
check(f"K={K} is large enough to include every neighbour", int(v_all.sum(1).max()) < K, f"max {int(v_all.sum(1).max())}")
for cfg in ({"move": "direct", "mem": 4}, {"move": "radial", "mem": 4}, {"sph": True, "mem": 4}):
    c, p = loud(cfg)
    mem = rand_mem(c)
    a = c.step(p, x, s, mem, half, EPS, EPS, total=N, mass=alive)
    b = c.step(p, x[keep], s[keep], mem[keep], half[keep], EPS, EPS, total=N)
    gap = max(float(jnp.abs(a[i][keep] - b[i]).max()) for i in range(3))
    finite = all(bool(jnp.isfinite(o).all()) for o in a)
    check(f"mass-0 particle is invisible {cfg}", gap < 1e-4 and finite, f"gap {gap:.1e}")

# 5. vmap and jit (the way run.py uses it: vmap over a batch with in_axes for mass, then jit)
c, p = make(mem=4, sph=True)
B = 3
xb = jnp.stack([x + 0.01 * i for i in range(B)])
sb, mb = jnp.stack([s] * B), jnp.stack([rand_mem(c)] * B)
ub, ab = jnp.stack([half] * B), jnp.stack([alive] * B)
f = jax.jit(jax.vmap(c.step, in_axes=(None, 0, 0, 0, 0, None, None, None, None, 0)), static_argnums=(5, 6, 7, 8))
vb = f(p, xb, sb, mb, ub, EPS, EPS, None, N, ab)
single = c.step(p, xb[1], sb[1], mb[1], ub[1], EPS, EPS, total=N, mass=ab[1])
gap = max(float(jnp.abs(vb[i][1] - single[i]).max()) for i in range(3))
check("works under jit(vmap(...))", all(bool(jnp.isfinite(o).all()) for o in vb) and gap < 1e-4, f"gap {gap:.1e}")
g = jax.grad(lambda pp: jnp.sum(c.step(pp, x, s, rand_mem(c), half, EPS, EPS, total=N)[1] ** 2))(p)
check("gradients w.r.t. parameters are finite and non-zero",
      all(bool(jnp.isfinite(v).all()) for v in g.values()) and sum(float(jnp.abs(v).sum()) for v in g.values()) > 0)

# 6. tiny end-to-end training through run.train
for cell_cfg in ({"kind": "edge"}, {"kind": "edge", "move": "radial", "mem": 4, "K": 16}):
    cfg = {"name": "t", "particles": N, "cell": dict(cell_cfg),
           "train": {"iters": 3, "batch": 2, "steps": [3, 6], "grid": 64}}
    cell, params, eps, curve, secs = run.train(cfg, log=lambda m: None)
    check(f"run.train works with {cell_cfg}", len(curve) == 3 and all(np.isfinite(r["loss"]) for r in curve),
          f"loss {curve[0]['loss']:.3f} -> {curve[-1]['loss']:.3f}, {secs:.0f}s")

# 7. report: parameter count and CPU time of one forward step at N = 1024 and 4096 (default config)
c = cells_edge.Cell()
p = c.init(jax.random.PRNGKey(1))
print(f"\ndefault config: {c.param_count(p):,} parameters "
      f"(baseline MLP cell: {cells.Cell().param_count(cells.Cell().init(jax.random.PRNGKey(1))):,})")
for n in (1024, 4096):
    eps = 0.2 * (1024 / n) ** 0.5
    xx, ss = npa.seed(jax.random.PRNGKey(2), n, radius=2 * eps)
    mm, uu = jnp.zeros((n, 0)), jnp.ones((n, 1))
    fn = jax.jit(lambda pp, xx, ss: c.step(pp, xx, ss, mm, uu, eps, eps, total=n))
    jax.block_until_ready(fn(p, xx, ss))
    t0 = time.time()
    for _ in range(3):
        jax.block_until_ready(fn(p, xx, ss))
    print(f"N={n}: {(time.time() - t0) / 3 * 1000:.0f} ms per forward step (jitted, CPU, after compile)")

print(f"\n{sum(results)}/{len(results)} edge-cell checks passed")
raise SystemExit(0 if all(results) else 1)
