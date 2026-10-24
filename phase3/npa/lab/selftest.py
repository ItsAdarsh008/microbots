"""Fast end-to-end check of the lab (a few minutes on a CPU): cells, damage, evaluator, trainer.

    python lab/selftest.py
"""

import sys
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

# 1. the plain cell is the paper's network: same parameters and same numbers as npa.step
plain = cells.Cell()
p = plain.init(jax.random.PRNGKey(1))
a = plain.step(p, x, s, jnp.zeros((N, 0)), on, EPS, EPS)
b = npa.step({k: p[k] for k in ("w1", "b1", "w2")}, x, s, on, EPS, EPS)
check("the plain cell reproduces npa.step exactly", float(jnp.abs(a[0] - b[0]).max()) == 0 and float(jnp.abs(a[1] - b[1]).max()) == 0)

# 2. every kind of cell runs, stays finite, and has the parameter count it should
for cfg in ({"block": "plain", "depth": 3}, {"block": "residual", "depth": 3, "norm": "rms", "zero_head": True},
            {"block": "gated", "depth": 3, "norm": "rms", "mem": 4}, {"width": 256}):
    c = cells.Cell(**cfg)
    pp = c.init(jax.random.PRNGKey(2))
    out = c.step(pp, x, s, jnp.zeros((N, c.mem_dims)), on, EPS, EPS)
    check(f"cell {cfg} runs and is finite", all(bool(jnp.isfinite(o).all()) for o in out[:3]),
          f"{c.param_count(pp):,} parameters")

# 2b. parameter shapes are what they should be, and nothing depends on N happening to equal the width
for cfg in ({"block": "residual", "depth": 4, "width": 96, "zero_head": True}, {"block": "gated", "depth": 3, "width": 96, "mem": 3},
            {"block": "plain", "depth": 4, "width": 96}):
    c = cells.Cell(**cfg)
    pp = c.init(jax.random.PRNGKey(4))
    ok = pp["b1"].shape == (96,) and pp["w1"].shape == (c.in_dims, 96) and pp["w2"].shape == (96, c.out_dims)
    out = c.step(pp, x, s, jnp.zeros((N, c.mem_dims)), on, EPS, EPS)
    check(f"cell {cfg} has the right parameter shapes", ok and all(bool(jnp.isfinite(o).all()) for o in out[:3]))

# 3. deleting particles: a deleted particle is invisible, and deleting nothing changes nothing
alive = jnp.ones(N).at[:20].set(0.0)
z_full = npa.perceive(x, s, EPS, EPS)
z_ones = npa.perceive(x, s, EPS, EPS, mass=jnp.ones(N))
z_cut = npa.perceive(x, s, EPS, EPS, total=N, mass=alive)
check("mass of 1 everywhere changes nothing", float(jnp.abs(z_full - z_ones).max()) == 0)
keep = np.arange(20, N)
z_small = npa.perceive(x[keep], s[keep], EPS, EPS, total=N)
check("a mass-0 particle is as good as removing it", float(jnp.abs(z_cut[keep] - z_small).max()) < 1e-4,
      f"gap {float(jnp.abs(z_cut[keep] - z_small).max()):.1e}")

# 4. each damage kind does what it says
mem = jnp.zeros((N, 0))
al = jnp.ones(N)
for kind, level in (("wipe", 0.5), ("cut", 0.5), ("drag", 0.4), ("clump", 0.6), ("shuffle", 0.5), ("noise", 0.3)):
    xd, sd, md, ad = damage.apply(kind, level, jax.random.PRNGKey(3), x, s, mem, al, EPS)
    changed = {"wipe": float(jnp.abs(sd - s).max()) > 0 and float(jnp.abs(xd - x).max()) == 0,
               "cut": float(ad.sum()) < N and float(jnp.abs(sd - s).max()) == 0,
               "drag": float(jnp.abs(xd - x).max()) > 0 and float(jnp.abs(sd - s).max()) == 0,
               "clump": float(jnp.abs(xd - x).max()) > 0,
               "shuffle": float(jnp.abs(sd - s).max()) > 0 and abs(float(sd.sum() - s.sum())) < 1e-3,
               "noise": float(jnp.abs(xd - x).max()) > 0 and float(jnp.abs(sd - s).max()) > 0}[kind]
    check(f"damage '{kind}' changes what it should", changed)

# 5. the evaluator: a tiny protocol. Zero-strength damage must give exactly zero excess (paired controls)
tiny = {**evaluate.PROTOCOL, "grow": 6, "recover": 6, "long": 20, "seeds": (1, 2), "grid": 64,
        "levels": {k: (0.0, 0.3) if k not in ("clump",) else (0.0, 0.3) for k in damage.KINDS}}
tiny["levels"]["cut"] = (0.0, 0.3)
res = evaluate.run(cells.Cell(), p, EPS, EPS, N, protocol=tiny)
zero = max(abs(v) for k in damage.KINDS for v in res["damage"][k]["excess"][0])
check("zero-strength damage has exactly zero excess over its control", zero < 1e-5, f"max {zero:.1e}")
check("the evaluator returns a finite score", res["summary"]["finite"], f"score {res['summary']['score']:.3f}")
check("cutting really removes particles in the evaluator", res["damage"]["cut"]["alive_final"][1] < N, str(res["damage"]["cut"]["alive_final"]))

# 6. the trainer runs a few iterations with deletion damage and a memory cell, and the loss is finite
cfg = {"name": "selftest", "particles": N, "cell": {"block": "gated", "depth": 3, "norm": "rms", "mem": 4},
       "train": {"iters": 3, "batch": 2, "steps": [3, 6], "grid": 64,
                 "damage": [{"kind": "wipe", "level": [0.1, 0.3]}, {"kind": "cut", "level": [0.1, 0.3], "p": 0.5}]}}
cell, params, eps, curve, secs = run.train(cfg, log=lambda m: None)
check("training runs with cut damage and memory", len(curve) == 3 and all(np.isfinite(c["loss"]) for c in curve),
      f"loss {curve[0]['loss']:.3f} -> {curve[-1]['loss']:.3f}")

print(f"\n{sum(results)}/{len(results)} lab checks passed")
raise SystemExit(0 if all(results) else 1)
