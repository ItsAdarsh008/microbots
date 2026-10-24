"""Checks for per-particle headings and the rotation-equivariant cell (lab/heading.py, cells_mlph.py, cells_equiv.py).

    python lab/test_equiv.py
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
import cells_equiv  # noqa: E402
import cells_mlph  # noqa: E402
import heading as hd  # noqa: E402
import npa  # noqa: E402
import run  # noqa: E402

results = []


def check(name, ok, detail=""):
    results.append(bool(ok))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))


def maxabs(a, b):
    return float(jnp.abs(a - b).max())


N, EPS = 128, 0.2 * (1024 / 128) ** 0.5
key = jax.random.PRNGKey(0)
x, s = npa.seed(key, N, radius=2 * EPS)
s = s + 0.3 * jax.random.normal(jax.random.PRNGKey(5), s.shape)
on = jnp.ones((N, 1))
half = (jax.random.uniform(jax.random.PRNGKey(6), (N, 1)) < 0.5).astype(jnp.float32)
PHI = 1.234                                                  # the swarm rotation angle used below


def rotate_all(v, phi):
    return hd.rotate(v, jnp.full(v.shape[:-1], phi))


# ---- 1. heading utilities -------------------------------------------------------
th = hd.headings(N, 3)
check("headings are deterministic, in [0, 2 pi), and depend on the seed",
      bool(jnp.all(hd.headings(N, 3) == th)) and float(th.min()) >= 0 and float(th.max()) < 2 * np.pi
      and maxabs(th, hd.headings(N, 4)) > 0.1 and float(th.std()) > 1.5)
v = jax.random.normal(jax.random.PRNGKey(8), (N, 2))
check("rotate then inverse rotate is the identity", maxabs(hd.rotate(hd.rotate(v, th), -th), v) < 1e-5)
check("rotate keeps lengths and rotates by the right angle (90 degrees: (1,0) -> (0,1))",
      maxabs(jnp.linalg.norm(hd.rotate(v, th), axis=-1), jnp.linalg.norm(v, axis=-1)) < 1e-5
      and maxabs(hd.rotate(jnp.array([1.0, 0.0]), jnp.array(np.pi / 2)), jnp.array([0.0, 1.0])) < 1e-6)
z = npa.perceive(x, s, EPS, EPS)
zr = hd.rotate_perception(z, th)
check("rotate_perception leaves the 32 scalars untouched", maxabs(zr[:, :32], z[:, :32]) == 0)
check("rotate_perception changes the vector blocks", maxabs(zr[:, 32:], z[:, 32:]) > 1e-3)
check("rotate_perception then inverse is the identity", maxabs(hd.rotate_perception(zr, -th), z) < 1e-5)
gl = lambda a: a[:, 32:64].reshape(N, 16, 2)
check("each gradient pair and the density gradient are rotated by that particle's angle",
      maxabs(gl(zr), hd.rotate(gl(z), th[:, None])) < 1e-6 and maxabs(zr[:, 64:], hd.rotate(z[:, 64:], th)) < 1e-6)
z2 = jnp.concatenate([z, z], axis=-1)
check("rotate_perception handles two sensing scales",
      maxabs(hd.rotate_perception(z2, th), jnp.concatenate([zr, zr], axis=-1)) < 1e-6)
# the perception itself is equivariant: rotating the swarm rotates the vector blocks (basis of everything below)
zp = npa.perceive(rotate_all(x, PHI), s, EPS, EPS)
check("npa.perceive is rotation-equivariant (scalars invariant, vectors rotate)",
      maxabs(zp[:, :32], z[:, :32]) < 1e-4 and maxabs(zp, hd.rotate_perception(z, jnp.full(N, PHI))) < 1e-4,
      f"gap {maxabs(zp, hd.rotate_perception(z, jnp.full(N, PHI))):.1e}")

# ---- 2. cells_mlph --------------------------------------------------------------
base = cells.Cell(mem=4)
p = base.init(jax.random.PRNGKey(1))
mem = 0.3 * jax.random.normal(jax.random.PRNGKey(7), (N, 4))
ref = base.step(p, x, s, mem, half, EPS, EPS)
mn = cells_mlph.Cell(mem=4)
out = mn.step(p, x, s, mem, half, EPS, EPS)
check("mlph with heading none reproduces cells.Cell exactly (bit for bit)",
      all(bool(jnp.all(a == b)) for a, b in zip(ref, out)))
mr = cells_mlph.Cell(mem=4, heading="random", heading_seed=0)
out_r = mr.step(p, x, s, mem, half, EPS, EPS)
gap = maxabs(out_r[0], out[0])
move_scale = float(jnp.abs(out[0] - x).max())
check("mlph with random headings differs from heading none (a plain MLP is not invariant)",
      gap > 0.2 * move_scale and maxabs(out_r[1], out[1]) > 1e-6,
      f"x gap {gap:.2e} vs max move {move_scale:.2e}; s gap {maxabs(out_r[1], out[1]):.1e}")
for cfg in ({"block": "gated", "depth": 3, "norm": "rms", "mem": 2}, {"scales": (1.0, 2.0)}):
    c = cells_mlph.Cell(heading="random", **cfg)
    pp = c.init(jax.random.PRNGKey(2))
    o = c.step(pp, x, s, jnp.zeros((N, c.mem_dims)), on, EPS, EPS)
    check(f"mlph accepts baseline options {cfg} and stays finite", all(bool(jnp.isfinite(a).all()) for a in o[:3]))
check("cells.build(kind=mlph / equiv) dispatches",
      type(cells.build(kind="mlph", heading="random")).__module__ == "cells_mlph"
      and type(cells.build(kind="equiv")).__module__ == "cells_equiv")
for mod, bad in ((cells_mlph, {"bogus": 1}), (cells_equiv, {"bogus": 1}), (cells_mlph, {"heading": "x"})):
    try:
        mod.Cell(**bad)
        check(f"{mod.__name__} rejects {bad}", False)
    except ValueError:
        check(f"{mod.__name__} rejects {bad}", True)

# ---- 3. cells_equiv -------------------------------------------------------------
eq = cells_equiv.Cell(mem=4, gain=1.0)          # gain 1 so that the outputs are not tiny and the checks mean something
pe = eq.init(jax.random.PRNGKey(1))
base0 = cells.Cell()
n_eq0 = cells_equiv.Cell().param_count(cells_equiv.Cell().init(jax.random.PRNGKey(0)))
n_base = base0.param_count(base0.init(jax.random.PRNGKey(0)))
print(f"      parameter counts: equiv (K=6, mem 0) {n_eq0:,}, equiv (K=6, mem 4) {eq.param_count(pe):,}, baseline {n_base:,}")
check("equiv parameter count is in the baseline's ballpark", 0.7 * n_base < n_eq0 < 1.5 * n_base, f"{n_eq0:,} vs {n_base:,}")

xe, se, me, mv = eq.step(pe, x, s, mem, half, EPS, EPS)
check("equiv runs, is finite, and moves things", all(bool(jnp.isfinite(a).all()) for a in (xe, se, me)) and float(mv) > 1e-4
      and maxabs(se, s) > 1e-4, f"mean move {float(mv):.2e}")

# (a) rotate the whole swarm
xr = rotate_all(x, PHI)
xe2, se2, me2, mv2 = eq.step(pe, xr, s, mem, half, EPS, EPS)
dx_a, dx_b = xe - x, xe2 - xr
check("(a) swarm rotation: ds unchanged", maxabs(se2 - s, se - s) < 1e-4, f"gap {maxabs(se2 - s, se - s):.1e}")
check("(a) swarm rotation: mem unchanged", maxabs(me2, me) < 1e-4, f"gap {maxabs(me2, me):.1e}")
check("(a) swarm rotation: dx rotates by the same angle", maxabs(dx_b, rotate_all(dx_a, PHI)) < 1e-4,
      f"gap {maxabs(dx_b, rotate_all(dx_a, PHI)):.1e} (max |dx| {float(jnp.abs(dx_a).max()):.1e})")
check("(a) swarm rotation: move length unchanged", abs(float(mv2 - mv)) < 1e-5)
# a swarm that is not centred on the origin, rotated about the origin, translated too
xo = x + jnp.array([0.7, -0.3])
xe3 = eq.step(pe, rotate_all(xo, PHI), s, mem, half, EPS, EPS)[0]
check("(a) off-centre swarm rotated about the origin: same result",
      maxabs(xe3 - rotate_all(xo, PHI), rotate_all(eq.step(pe, xo, s, mem, half, EPS, EPS)[0] - xo, PHI)) < 1e-4)

# (b) random headings change nothing
eqh = cells_equiv.Cell(mem=4, gain=1.0, heading="random", heading_seed=11)
oh = eqh.step(pe, x, s, mem, half, EPS, EPS)
gaps = [maxabs(a, b) for a, b in zip(oh[:3], (xe, se, me))]
check("(b) random headings change nothing (x, s, mem)", max(gaps) < 1e-4, "gaps " + " ".join(f"{g:.1e}" for g in gaps))
# ... and the plain MLP with the same headings and the same gain really does change
mh1 = cells_mlph.Cell(mem=4, gain=1.0)
ph = mh1.init(jax.random.PRNGKey(1))
a0 = mh1.step(ph, x, s, mem, half, EPS, EPS)
a1 = cells_mlph.Cell(mem=4, gain=1.0, heading="random").step(ph, x, s, mem, half, EPS, EPS)
print(f"      contrast: plain MLP, same headings: x gap {maxabs(a0[0], a1[0]):.2e}, s gap {maxabs(a0[1], a1[1]):.2e}; "
      f"equiv: x gap {gaps[0]:.1e}, s gap {gaps[1]:.1e}")

# (c) mass-0 particles and total
mass = jnp.ones(N).at[:20].set(0.0)
keep = np.arange(20, N)
full = eq.step(pe, x, s, mem, half, EPS, EPS, total=N, mass=mass)
small = eq.step(pe, x[keep], s[keep], mem[keep], half[keep], EPS, EPS, total=N)
check("(c) mass-0 particles are invisible to the others (same as removing them)",
      max(maxabs(full[0][keep], small[0]), maxabs(full[1][keep], small[1]), maxabs(full[2][keep], small[2])) < 1e-4,
      f"gap {maxabs(full[0][keep], small[0]):.1e}")
fr = eq.step(pe, rotate_all(x, PHI), s, mem, half, EPS, EPS, total=N, mass=mass)
check("(c) equivariance holds with mass and total set",
      maxabs(fr[0] - rotate_all(x, PHI), rotate_all(full[0] - x, PHI)) < 1e-4 and maxabs(fr[1], full[1]) < 1e-4)
fh = eqh.step(pe, x, s, mem, half, EPS, EPS, total=N, mass=mass)
check("(c) random headings change nothing with mass and total set", maxabs(fh[0], full[0]) < 1e-4 and maxabs(fh[1], full[1]) < 1e-4)
check("(c) update mask respected: non-updating particles do not move or change",
      maxabs((xe - x) * (1 - half), 0 * x) == 0 and maxabs((se - s) * (1 - half), 0 * s) == 0)
check("(c) total changes what is sensed (the density gradient is divided by it)",
      maxabs(eq.step(pe, x, s, mem, half, EPS, EPS, total=4 * N)[0], xe) > 1e-6)

# options
for cfg in ({"K": 3}, {"K": 8, "depth": 3}, {"mem": 0}, {"width": 64}):
    c = cells_equiv.Cell(**cfg)
    pp = c.init(jax.random.PRNGKey(2))
    o = c.step(pp, x, s, jnp.zeros((N, c.mem_dims)), on, EPS, EPS)
    check(f"equiv option {cfg} runs and is finite", all(bool(jnp.isfinite(a).all()) for a in o[:3]), f"{c.param_count(pp):,} params")

# (d) gradients
def loss(pp, cell, mem):
    o = cell.step(pp, x, s, mem, on, EPS, EPS)
    return jnp.sum(o[0] ** 2) + jnp.sum(o[1] ** 2) + jnp.sum(o[2] ** 2)

for cfg in ({"mem": 4, "gain": 1.0}, {"K": 3}):
    c = cells_equiv.Cell(**cfg)
    pp = c.init(jax.random.PRNGKey(1))
    g = jax.grad(loss)(pp, c, jnp.zeros((N, c.mem_dims)) + 0.1)
    finite = all(bool(jnp.isfinite(v).all()) for v in g.values())
    nz = all(float(jnp.abs(v).max()) > 0 for v in g.values())
    check(f"(d) gradients are finite and nonzero for every tensor, {cfg}", finite and nz,
          " ".join(f"{k}:{float(jnp.abs(v).max()):.1e}" for k, v in g.items()))
# zero vectors (an isolated swarm: no neighbours) must not give NaN gradients
xs = jnp.stack([jnp.arange(N) * 10.0 * EPS, jnp.zeros(N)], axis=-1)
c = cells_equiv.Cell()
pp = c.init(jax.random.PRNGKey(1))
m0 = jnp.zeros((N, 0))
g0 = jax.grad(lambda q: jnp.sum(c.step(q, xs, s, m0, on, EPS, EPS)[0] ** 2) + jnp.sum(c.step(q, xs, s, m0, on, EPS, EPS)[1] ** 2))(pp)
check("(d) gradients stay finite when no particle has a neighbour (zero vectors)", all(bool(jnp.isfinite(v).all()) for v in g0.values()))

# (e) cost at N = 1024
N2, EPS2 = 1024, 0.2
x2, s2 = npa.seed(jax.random.PRNGKey(3), N2, radius=2 * EPS2)
s2 = s2 + 0.3 * jax.random.normal(jax.random.PRNGKey(4), s2.shape)
on2 = jnp.ones((N2, 1))
times = {}
for name, c in (("baseline", cells.Cell()), ("mlph/random", cells_mlph.Cell(heading="random")), ("equiv", cells_equiv.Cell()),
                ("equiv/random", cells_equiv.Cell(heading="random"))):
    pp = c.init(jax.random.PRNGKey(1))
    fwd = jax.jit(lambda q, c=c: c.step(q, x2, s2, jnp.zeros((N2, 0)), on2, EPS2, EPS2)[0])
    bwd = jax.jit(jax.grad(lambda q, c=c: jnp.sum(c.step(q, x2, s2, jnp.zeros((N2, 0)), on2, EPS2, EPS2)[0] ** 2)))
    fwd(pp).block_until_ready(); jax.block_until_ready(bwd(pp))
    t0 = time.time()
    for _ in range(5):
        fwd(pp).block_until_ready()
    tf = (time.time() - t0) / 5
    t0 = time.time()
    for _ in range(3):
        jax.block_until_ready(bwd(pp))
    times[name] = (tf, (time.time() - t0) / 3)
    print(f"      N=1024 CPU {name:13s} step {1000 * times[name][0]:7.1f} ms   step+grad {1000 * times[name][1]:7.1f} ms")
check("(e) equiv costs at most 1.5x the baseline at N=1024 (perceive dominates)", times["equiv"][1] < 1.5 * times["baseline"][1])

# ---- 4. cells_edge radial / sph=False: equivariant? (reported, file not edited) -----
ce = cells_edge.Cell(K=64, move="radial", sph=False, mem=4, gain=1.0)
pc = ce.init(jax.random.PRNGKey(1))
e0 = ce.step(pc, x, s, mem, half, EPS, EPS)
e1 = ce.step(pc, xr, s, mem, half, EPS, EPS)
g_dx = maxabs(e1[0] - xr, rotate_all(e0[0] - x, PHI))
g_s, g_m = maxabs(e1[1], e0[1]), maxabs(e1[2], e0[2])
equiv_as_written = g_dx < 1e-4 and g_s < 1e-4 and g_m < 1e-4
# Reported, not asserted either way (the file is not edited). Expected finding: NOT equivariant as written, because the
# message MLP reads the raw offset components (not only |offset|), so m_ij (and with it ds, mem and the radial weights)
# is not rotation invariant; only the final sum over offset_ij is equivariant in form.
print(f"[INFO] cells_edge move='radial', sph=False is {'' if equiv_as_written else 'NOT '}rotation-equivariant as written  "
      f"(dx gap {g_dx:.1e}, ds gap {g_s:.1e}, mem gap {g_m:.1e}; max |dx| {float(jnp.abs(e0[0] - x).max()):.1e})")
check("cells_edge radial test is not vacuous (it moves particles)", float(jnp.abs(e0[0] - x).max()) > 1e-4)
# diagnosis: zero the message MLP's weights on the two raw offset components -> only |offset| is left -> equivariant
pz = {**pc, "m1": pc["m1"].at[:2].set(0.0)}
z0, z1 = ce.step(pz, x, s, mem, half, EPS, EPS), ce.step(pz, xr, s, mem, half, EPS, EPS)
gz = max(maxabs(z1[0] - xr, rotate_all(z0[0] - x, PHI)), maxabs(z1[1], z0[1]), maxabs(z1[2], z0[2]))
check("cells_edge radial: with the raw-offset input weights zeroed it IS equivariant (diagnosis of the gap above)",
      gz < 1e-4 and (equiv_as_written or g_dx > 1e-3), f"gap {gz:.1e}")

# ---- 5. training through run.train ------------------------------------------------
for name, cell_cfg in (("equiv", {"kind": "equiv"}), ("mlph", {"kind": "mlph", "heading": "random"}),
                       ("equiv+heading+mem", {"kind": "equiv", "heading": "random", "mem": 4})):
    cfg = {"name": "t", "particles": 128, "cell": dict(cell_cfg),
           "train": {"iters": 3, "batch": 2, "steps": [3, 6], "grid": 64}}
    cell, params, eps, curve, secs = run.train(cfg, log=lambda m: None)
    check(f"run.train with {name} gives finite losses and parameters",
          len(curve) == 3 and all(np.isfinite(c["loss"]) for c in curve) and all(bool(jnp.isfinite(v).all()) for v in params.values()),
          f"loss {curve[0]['loss']:.3f} -> {curve[-1]['loss']:.3f}, {secs:.0f}s")

print(f"\n{sum(results)}/{len(results)} equivariance checks passed")
raise SystemExit(0 if all(results) else 1)
