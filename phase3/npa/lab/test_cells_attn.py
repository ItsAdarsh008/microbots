"""Checks for the attention cell (lab/cells_attn.py). A minute or two on a CPU.

    python lab/test_cells_attn.py            # checks
    python lab/test_cells_attn.py --bench    # also time one forward step at N=1024 and N=4096
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
import cells_attn  # noqa: E402
import npa  # noqa: E402
import run  # noqa: E402

results = []


def check(name, ok, detail=""):
    results.append(bool(ok))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))


N, EPS = 128, 0.2 * (1024 / 128) ** 0.5
key = jax.random.PRNGKey(0)
x, s = npa.seed(key, N, radius=2 * EPS)
s = s + 0.3 * jax.random.normal(key, s.shape)
on = jnp.ones((N, 1))
# strong init so that outputs and attention weights are far from trivial in the invariance checks
STRONG = {"gain": 1.0, "attn_gain": 1.0}


def make(**cfg):
    c = cells_attn.Cell(**cfg)
    return c, c.init(jax.random.PRNGKey(1))


def gap(a, b):
    return float(jnp.abs(a - b).max())


# 1. runs and is finite
for cfg in ({"sph": True, "heads": 4}, {"sph": True, "heads": 1}, {"sph": False, "heads": 4},
            {"sph": False, "heads": 1}, {"sph": True, "heads": 4, "mem": 4, "norm": "rms"},
            {"sph": False, "heads": 4, "mem": 4, "zero_head": True}):
    c, p = make(**cfg)
    mem = 0.1 * jnp.ones((N, c.mem_dims))
    out = c.step(p, x, s, mem, on, EPS, EPS)
    shapes = out[0].shape == x.shape and out[1].shape == s.shape and out[2].shape == mem.shape and out[3].shape == ()
    check(f"{cfg} runs, right shapes, finite", shapes and all(bool(jnp.isfinite(o).all()) for o in out),
          f"{c.param_count(p):,} parameters")

# 2. initial behaviour: logits near uniform, first steps barely move
c, p = make()
m, w, valid, idx = c.attention(p, x, s, jnp.zeros((N, 0)), EPS)
uniform = valid[..., None] / valid.sum(1)[:, None, None]
check("default init: attention weights start near uniform", gap(w, uniform) < 0.1, f"max gap {gap(w, uniform):.3f}")
out = c.step(p, x, s, jnp.zeros((N, 0)), on, EPS, EPS)
check("default init: first step barely moves", float(out[3]) < 0.1 * EPS, f"mean move {float(out[3]) / EPS:.4f} eps")

# 3. attention weights sum to 1 over the valid slots; invalid slots get exactly 0; self slot is valid
for cfg in ({"heads": 4}, {"heads": 1, "K": 8}):
    c, p = make(**cfg, **STRONG)
    _, w, valid, idx = c.attention(p, x, s, jnp.zeros((N, 0)), EPS)
    sums = w.sum(axis=1)
    check(f"attention weights sum to 1 over valid slots {cfg}", gap(sums, jnp.ones_like(sums)) < 1e-5,
          f"max gap {gap(sums, jnp.ones_like(sums)):.1e}")
    check(f"invalid slots have zero weight, self slot valid {cfg}",
          float(jnp.abs(w * (1 - valid)[..., None]).max()) == 0 and bool((valid[:, 0] == 1).all())
          and bool((idx[:, 0] == jnp.arange(N)).all()))
    check(f"attention is not trivially uniform with strong init {cfg}", gap(w, valid[..., None] / valid.sum(1)[:, None, None]) > 0.02)

# 4. translating the swarm leaves ds (and the moves) unchanged
mem0 = 0.2 * jax.random.normal(jax.random.PRNGKey(5), (N, 4))
for sph in (True, False):
    c, p = make(sph=sph, mem=4, **STRONG)
    a = c.step(p, x, s, mem0, on, EPS, EPS)
    b = c.step(p, x + jnp.array([3.7, -2.1]), s, mem0, on, EPS, EPS)
    check(f"translation leaves ds, memory and moves unchanged (sph={sph})",
          gap(a[1], b[1]) < 1e-4 and gap(a[2], b[2]) < 1e-4 and gap(a[0] - x, b[0] - (x + jnp.array([3.7, -2.1]))) < 1e-4,
          f"ds gap {gap(a[1], b[1]):.1e}")

# 5. permuting the particles permutes the outputs
perm = jax.random.permutation(jax.random.PRNGKey(7), N)
for sph in (True, False):
    c, p = make(sph=sph, mem=4, **STRONG)
    a = c.step(p, x, s, mem0, on, EPS, EPS)
    b = c.step(p, x[perm], s[perm], mem0[perm], on, EPS, EPS)
    check(f"permuting particles permutes outputs (sph={sph})",
          gap(a[0][perm], b[0]) < 1e-4 and gap(a[1][perm], b[1]) < 1e-4 and gap(a[2][perm], b[2]) < 1e-4
          and abs(float(a[3]) - float(b[3])) < 1e-5, f"ds gap {gap(a[1][perm], b[1]):.1e}")

# 6. a mass-0 particle is invisible: same as removing it (total = the original count, K covers everyone)
alive = jnp.ones(N).at[:20].set(0.0)
keep = np.arange(20, N)
for sph in (True, False):
    c, p = make(sph=sph, mem=4, K=N - 1, **STRONG)
    a = c.step(p, x, s, mem0, on, EPS, EPS, None, N, alive)
    b = c.step(p, x[keep], s[keep], mem0[keep], on[keep], EPS, EPS, None, N)
    g = max(gap(a[0][keep], b[0]), gap(a[1][keep], b[1]), gap(a[2][keep], b[2]))
    check(f"a mass-0 particle is as good as removing it (sph={sph})", g < 1e-4, f"gap {g:.1e}")
_, w, valid, idx = c.attention(p, x, s, mem0, EPS, alive)
seen_dead = float((w.sum(-1) * (alive[idx] == 0) * (jnp.arange(N)[:, None] != idx)).max())
check("no live particle puts weight on a deleted one", seen_dead == 0.0)

# 7. jit and vmap (the trainer vmaps step over a batch, with eps and total static-ish)
c, p = make(mem=4, **STRONG)
eager = c.step(p, x, s, mem0, on, EPS, EPS)
jitted = jax.jit(c.step, static_argnums=(5, 6))(p, x, s, mem0, on, EPS, EPS)
check("jax.jit gives the same numbers", max(gap(e, j) for e, j in zip(eager[:3], jitted[:3])) < 1e-4)
B = 3
xb, sb = jnp.stack([x] * B), jnp.stack([s + 0.1 * i for i in range(B)])
mb, ub, ab = jnp.stack([mem0] * B), jnp.stack([on] * B), jnp.stack([alive] * B)
vm = jax.jit(jax.vmap(c.step, in_axes=(None, 0, 0, 0, 0, None, None, None, None, 0)), static_argnums=(5, 6, 7, 8))
vout = vm(p, xb, sb, mb, ub, EPS, EPS, None, N, ab)
one = c.step(p, xb[1], sb[1], mb[1], ub[1], EPS, EPS, None, N, ab[1])
check("jax.vmap over a batch matches single calls and is finite",
      all(bool(jnp.isfinite(o).all()) for o in vout) and gap(vout[1][1], one[1]) < 1e-4)

# gradients flow to every parameter and are finite
def loss(pp):
    o = c.step(pp, x, s, mem0, on, EPS, EPS)
    return jnp.sum(o[1] ** 2) + jnp.sum(o[0] ** 2) + jnp.sum(o[2] ** 2)

g = jax.grad(loss)(p)
check("gradients are finite and reach every parameter", all(bool(jnp.isfinite(v).all()) for v in g.values())
      and all(float(jnp.abs(v).max()) > 0 for v in g.values()), str([k for k, v in g.items() if float(jnp.abs(v).max()) == 0]))

# 8. cells.build picks it up by kind, and a tiny end-to-end training run gives a finite loss
check("cells.build(kind='attn') returns this cell", type(cells.build(kind="attn", heads=2)).__module__ == "cells_attn")
cfg = {"name": "t", "particles": 128, "cell": {"kind": "attn"},
       "train": {"iters": 3, "batch": 2, "steps": [3, 6], "grid": 64}}
cell, params, eps, curve, secs = run.train(cfg, log=lambda m: None)
check("run.train with the attn cell: finite loss", len(curve) == 3 and all(np.isfinite(r["loss"]) for r in curve),
      f"loss {curve[0]['loss']:.3f} -> {curve[-1]['loss']:.3f}, {secs:.0f}s")

print(f"\n{sum(results)}/{len(results)} attn checks passed")

if "--bench" in sys.argv:
    for n in (1024, 4096):
        eps = 0.2 * (1024 / n) ** 0.5   # same neighbour count per particle at every N
        xn, sn = npa.seed(jax.random.PRNGKey(0), n, radius=2 * eps)
        sn = sn + 0.3 * jax.random.normal(key, sn.shape)
        c, p = make()
        f = jax.jit(lambda pp, a, b, o: c.step(pp, a, b, jnp.zeros((n, 0)), o, eps, eps, None, n)[0])
        o = jnp.ones((n, 1))
        f(p, xn, sn, o).block_until_ready()
        t = time.time()
        for _ in range(3):
            f(p, xn, sn, o).block_until_ready()
        print(f"N={n}: {1000 * (time.time() - t) / 3:.0f} ms per forward step, {c.param_count(p):,} parameters")

raise SystemExit(0 if all(results) else 1)
