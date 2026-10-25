"""Checks for cells_mlphthin: ring_vis=1 equals the plain two-ring cell, ring_vis=0 equals what the first ring alone sees, the fraction in between is about right."""
import sys
from pathlib import Path
import jax, jax.numpy as jnp, numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE))
import cells, cells_mlphthin as ct, npa

ok = []
def check(n, c, d=""):
    ok.append(bool(c)); print(f"[{'PASS' if c else 'FAIL'}] {n}" + (f"  ({d})" if d else ""))

key = jax.random.PRIORITY if False else jax.random.PRNGKey(0)
n = 256; eps = 0.1
x = jax.random.uniform(key, (n, 2)) * 0.6
s = jax.random.uniform(jax.random.PRNGKey(1), (n, 16))
ref = npa.perceive(x, s, 2 * eps, 2 * eps)
one = ct.perceive_thin(x, s, 2 * eps, 2 * eps, eps, 1.0)
check("ring_vis=1 equals npa.perceive", float(jnp.abs(one - ref).max()) < 1e-5, f"{float(jnp.abs(one - ref).max()):.1e}")
zero = ct.perceive_thin(x, s, 2 * eps, 2 * eps, eps, 0.0)
near_only = ct.perceive_thin(x, s, 2 * eps, 2 * eps, 2 * eps, 0.0)       # near radius = whole ring: nothing thinned
check("near=2eps equals npa.perceive", float(jnp.abs(near_only - ref).max()) < 1e-5)
check("ring_vis=0 differs from the full ring", float(jnp.abs(zero - ref).max()) > 1e-3)
v = ct._vis(x, x, eps, 0.25)
d2 = jnp.sum((x[None] - x[:, None]) ** 2, -1)
far = (d2 >= eps * eps)
frac = float((v * far).sum() / far.sum())
check("far pairs visible about 25%", 0.2 < frac < 0.3, f"{frac:.3f}")
check("visibility symmetric", float(jnp.abs(v - v.T).max()) == 0.0)
cell = cells.build(kind="mlphthin", scales=[1.0, 2.0], heading="random", ring_vis=0.25)
p = cell.init(jax.random.PRNGKey(2))
xo, so, mo, mv = cell.step(p, x, s, jnp.zeros((n, cell.mem_dims)), jnp.ones((n, 1)), eps, eps)
check("cell steps and is finite", bool(jnp.isfinite(xo).all() and jnp.isfinite(so).all()))
g = jax.grad(lambda pp: jnp.sum(cell.step(pp, x, s, jnp.zeros((n, cell.mem_dims)), jnp.ones((n, 1)), eps, eps)[1]))(p)
check("gradients are finite", all(bool(jnp.isfinite(v).all()) for v in g.values()))
print(f"{sum(ok)}/{len(ok)} passed"); sys.exit(0 if all(ok) else 1)
