"""Checks for lab/embody.py (the physical-robot wrapper).

    python lab/test_embody.py
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
import embody  # noqa: E402
import npa  # noqa: E402
import run  # noqa: E402

results = []


def check(name, ok, detail=""):
    results.append(bool(ok))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))


N, EPS = 128, 0.066
R = embody.Embodied(cells.Cell(), mode="phys").spec["body_r"]
base = cells.Cell()
P = base.init(jax.random.PRNGKey(1))
MEM = jnp.zeros((N, 0))
ON = jnp.ones((N, 1))
M1 = jnp.ones(N)
UPD = (jax.random.uniform(jax.random.PRNGKey(5), (N, 1)) < 0.5).astype(jnp.float32)


def hexgrid(n, spacing):
    k = int(np.ceil(np.sqrt(n)))
    pts = [[(j + 0.5 * (i % 2)) * spacing, i * spacing * 0.866] for i in range(k) for j in range(k)]
    return jnp.array(pts[:n], jnp.float32)


def mind(x, live=None):
    d = jnp.sqrt(jnp.sum((x[:, None] - x[None]) ** 2, -1)) + 9.0 * jnp.eye(x.shape[0])
    if live is not None:
        d = d + 9.0 * (1 - live[:, None] * live[None, :])
    return float(d.min())


# a swarm with a spread of states and spacing of about 3 body radii (so sensing is meaningful and nothing overlaps)
X0 = hexgrid(N, 3.0 * R)
S0 = 0.5 * jax.random.normal(jax.random.PRNGKey(2), (N, 16))

# ---------------------------------------------------------------------------------------------------------------
# ideal
# ---------------------------------------------------------------------------------------------------------------
e_ideal = embody.Embodied(base, mode="ideal")
a = base.step(P, X0, S0, MEM, UPD, EPS, EPS, None, N, M1)
b = e_ideal.step(P, X0, S0, MEM, UPD, EPS, EPS, None, N, M1)
check("ideal mode reproduces base.step exactly", all(u.size == 0 or float(jnp.abs(u - v).max()) == 0 for u, v in zip(a[:3], b[:3])) and float(a[3]) == float(b[3]))
check("wrapper has the cell interface", e_ideal.mem_dims == 0 and e_ideal.in_dims == base.in_dims and e_ideal.out_dims == base.out_dims
      and e_ideal.param_count(P) == base.param_count(P) and set(e_ideal.init(jax.random.PRNGKey(1))) == set(P))
check("body_r and port_d come from report.json", abs(R - 15 / 900) < 1e-9 and abs(embody.Embodied(base).spec["port_d"] - 14 / 900) < 1e-9)

# ---------------------------------------------------------------------------------------------------------------
# sensing model
# ---------------------------------------------------------------------------------------------------------------
class Probe:
    """A cell that records what it perceives and proposes a fixed move and state change."""
    mem_dims = 0

    def __init__(self, move=(0.0, 0.0), ds=0.0):
        self.log, self.move, self.ds = [], jnp.array(move), ds

    def step(self, p, x, s, mem, update, eps, eps0, block=None, total=None, mass=None):
        self.log.append({"x": x, "s": s, "mass": mass, "z": npa.perceive(x, s, eps, eps0, block, total, mass)})
        return x + update * self.move, s + update * self.ds, mem, jnp.zeros(())


pr = Probe()
e = embody.Embodied(pr, mode="phys", dropout=0.0, quant_bits=0)
x_out = e.step(None, X0, S0, MEM, ON, EPS, EPS, None, N, M1)[0]
seen = pr.log[0]
dn = seen["x"] - X0
check("sensing noise changes the perceived positions", float(jnp.abs(dn).max()) > 1e-3 and abs(float(dn.std()) - 3 / 900) < 0.0006,
      f"noise std {float(dn.std()) * 900:.2f} mm")
check("true positions stay noise-free (a cell that does not move leaves them put)", float(jnp.abs(x_out - X0).max()) < 1e-6,
      f"max change {float(jnp.abs(x_out - X0).max()):.1e}")
pr2 = Probe(move=(0.004, 0.0))
x2 = embody.Embodied(pr2, mode="phys", dropout=0.0, quant_bits=0).step(None, X0, S0, MEM, ON, EPS, EPS, None, N, M1)[0]
check("a proposed move is applied to the true position, not the noisy copy",
      float(jnp.abs((x2 - X0)[:, 0] - 0.004).max()) < 5e-4 and float(jnp.abs((x2 - X0)[:, 1]).max()) < 5e-4)
e.step(None, X0, S0, MEM, ON, EPS, EPS, None, N, M1, key=jax.random.PRNGKey(7))
e.step(None, X0, S0, MEM, ON, EPS, EPS, None, N, M1, key=jax.random.PRNGKey(8))
check("explicit key= changes the noise, same key repeats it", float(jnp.abs(pr.log[-1]["x"] - pr.log[-2]["x"]).max()) > 1e-4)
e.step(None, X0, S0, MEM, ON, EPS, EPS, None, N, M1, key=jax.random.PRNGKey(8))
check("same key reproduces the noise exactly", float(jnp.abs(pr.log[-1]["x"] - pr.log[-2]["x"]).max()) == 0)
e.step(None, X0, S0, MEM, ON, EPS, EPS, None, N, M1)
e.step(None, X0, S0, MEM, UPD, EPS, EPS, None, N, M1)
check("derived key changes when the update mask changes", float(jnp.abs(pr.log[-1]["x"] - pr.log[-2]["x"]).max()) > 1e-4)
e.step(None, X0, S0, MEM, UPD, EPS, EPS, None, N, M1)
check("derived key is deterministic given identical inputs", float(jnp.abs(pr.log[-1]["x"] - pr.log[-2]["x"]).max()) == 0)

# perceived neighbours differ from the true ones
z_true = npa.perceive(X0, S0, EPS, EPS, None, N, M1)
check("noisy perception differs from noise-free perception", float(jnp.abs(seen["z"] - z_true).max()) > 1e-3)

# quantisation
pq = Probe()
embody.Embodied(pq, mode="phys", dropout=0.0, sense_noise=0.0).step(None, X0, S0, MEM, ON, EPS, EPS, None, N, M1)
sq = pq.log[0]["s"]
step = 3.0 / 255
inr = jnp.abs(S0) <= 1.5
check("quantisation: |s_q - s| <= half a step in range", float(jnp.where(inr, jnp.abs(sq - S0), 0).max()) <= step / 2 + 1e-6,
      f"max {float(jnp.where(inr, jnp.abs(sq - S0), 0).max()):.4f}, half step {step / 2:.4f}")
check("quantisation: values are on the 8-bit grid and clipped", float(jnp.abs(((sq + 1.5) / step) - jnp.round((sq + 1.5) / step)).max()) < 1e-3
      and float(sq.max()) <= 1.5 + 1e-6 and float(sq.min()) >= -1.5 - 1e-6)
g = jax.grad(lambda s_: jnp.sum(embody.quantise(s_, 8) * 2.0))(S0)
check("quantisation: straight-through gradient is the identity", float(jnp.abs(g - 2.0).max()) == 0)
# true state keeps full precision: s_new = s + ds
pd = Probe(ds=0.01)
s_new = embody.Embodied(pd, mode="phys", dropout=0.0, sense_noise=0.0).step(None, X0, S0, MEM, ON, EPS, EPS, None, N, M1)[1]
check("true state keeps full precision (s + ds)", float(jnp.abs(s_new - S0 - 0.01).max()) < 1e-6)

# dropout
pall = Probe()
embody.Embodied(pall, mode="phys", dropout=1.0, sense_noise=0.0, quant_bits=0).step(None, X0, S0, MEM, ON, EPS, EPS, None, N, M1)
z_drop = pall.log[0]["z"]
z_alone = jnp.concatenate([npa.perceive(X0[i:i + 1], S0[i:i + 1], EPS, EPS, None, N, jnp.ones(1)) for i in range(0, N, 9)])
check("dropout 1.0: every particle sees only itself", float(jnp.abs(z_drop[::9] - z_alone).max()) < 1e-4,
      f"max gap to an isolated particle {float(jnp.abs(z_drop[::9] - z_alone).max()):.1e}")
check("dropout: npa.perceive is restored afterwards", npa.perceive.__module__ == "npa" and npa.perceive.__name__ == "perceive"
      and "orig" not in npa.perceive.__code__.co_freevars)
pd5 = Probe()
embody.Embodied(pd5, mode="phys", dropout=0.5, sense_noise=0.0, quant_bits=0).step(None, X0, S0, MEM, ON, EPS, EPS, None, N, M1)
frac = float((pd5.log[0]["mass"] == 0).mean())
check("dropout 0.5 drops about half of the broadcasts", 0.3 < frac < 0.7, f"{frac:.2f}")
# a dropped particle is invisible to others but sees itself: compare with a hand-built mass vector
mm = pd5.log[0]["mass"]
z_ref = npa.perceive(X0, S0, EPS, EPS, None, N, mm)
z_got = pd5.log[0]["z"]
vis = mm > 0
check("kept particles see exactly the dropout-masked world", float(jnp.abs(z_got - z_ref)[vis].max()) < 1e-5)
check("dropped particles differ from it only by their own smoothed-state term",
      float(jnp.abs(z_got - z_ref)[~vis][:, :16].max()) == 0 and float(jnp.abs(z_got - z_ref)[~vis][:, 32:].max()) == 0
      and float(jnp.abs(z_got - z_ref)[~vis][:, 16:32].max()) > 0)

# deleted particles are untouched in every mode
alive = jnp.ones(N).at[:20].set(0.0)
for mode in ("phys", "lattice"):
    ee = embody.Embodied(base, mode=mode)
    xs = ee.snap_to_lattice(X0, alive) if mode == "lattice" else X0
    xo, so, _, _ = ee.step(P, xs, S0, MEM, ON, EPS, EPS, None, N, alive)
    check(f"{mode}: deleted particles are untouched", float(jnp.abs(xo[:20] - xs[:20]).max()) == 0 and float(jnp.abs(so[:20] - S0[:20]).max()) == 0)

# ---------------------------------------------------------------------------------------------------------------
# phys
# ---------------------------------------------------------------------------------------------------------------
ph = embody.Embodied(base, mode="phys")
M = ph.spec["max_step"]
rng = np.random.default_rng(0)
worst, pos_gap = 0.0, 0.0
for t in range(3):
    Pw = base.init(jax.random.PRNGKey(10 + t))
    Pw = {**Pw, "w2": Pw["w2"] * 40.0, "w1": Pw["w1"] * 10.0}                # huge moves: the cap must bind
    x0 = hexgrid(N, 4.0 * R) + 0.001 * jnp.asarray(rng.standard_normal((N, 2)), jnp.float32)
    ideal_move = float(jnp.abs(base.step(Pw, x0, S0 * 3, MEM, ON, EPS, EPS, None, N, M1)[0] - x0).max())
    xo = ph.step(Pw, x0, S0 * 3, MEM, ON, EPS, EPS, None, N, M1)[0]
    # compare against the no-projection displacement through a copy with proj_iters=0
    xo0 = embody.Embodied(base, mode="phys", proj_iters=0).step(Pw, x0, S0 * 3, MEM, ON, EPS, EPS, None, N, M1)[0]
    worst = max(worst, float(jnp.sqrt(jnp.sum((xo0 - x0) ** 2, -1)).max()))
check("phys: per-tick displacement never exceeds max_step", worst <= M + 1e-6, f"max {worst * 900:.3f} mm vs cap {M * 900:.1f} mm, uncapped would be {ideal_move * 900:.1f} mm")

# no overlap: hex lattice at 2.1 body radii is not overlapping; moves + noise create overlaps; projection resolves
Pw = {**P, "w2": P["w2"] * 40.0, "w1": P["w1"] * 10.0}
x0 = hexgrid(N, 3.0 * R)
Pm = Pw
xo = ph.step(Pm, x0, S0 * 3, MEM, UPD, EPS, EPS, None, N, M1)[0]
xn = embody.Embodied(base, mode="phys", proj_iters=0).step(Pm, x0, S0 * 3, MEM, UPD, EPS, EPS, None, N, M1)[0]
check("phys: sparse overlaps from a move are resolved to 2*body_r", mind(xo) >= 2 * R - 1e-4,
      f"min distance before projection {mind(xn) / (2 * R):.3f}, after {mind(xo) / (2 * R):.3f} of 2*body_r")
# a dense start: ring of pairs on top of each other
xp = hexgrid(N, 5.0 * R)
xp = xp.at[1::2].set(xp[0::2] + jnp.array([1.0, 0.3]) * 0.9 * R)           # every second particle 0.94R from its partner (overlap)
Pz = {**P, "w2": P["w2"] * 0.0}
xo = embody.Embodied(base, mode="phys", dropout=0.0, sense_noise=0.0).step(Pz, xp, S0, MEM, ON, EPS, EPS, None, N, M1)[0]
check("phys: overlapping pairs are separated to >= 2*body_r - 1e-4", mind(xo) >= 2 * R - 1e-4, f"start {mind(xp) / (2 * R):.3f}, after {mind(xo) / (2 * R):.4f} of 2*body_r")
# the honest hard case: a hex packing compressed by 10 %
xc = hexgrid(N, 2.0 * R * 0.9)
xo = embody.Embodied(base, mode="phys", dropout=0.0, sense_noise=0.0).step(Pz, xc, S0, MEM, ON, EPS, EPS, None, N, M1)[0]
print(f"[INFO] bulk-compressed packing (spacing 0.9 * 2R): min distance {mind(xc) / (2 * R):.3f} -> {mind(xo) / (2 * R):.3f} of 2R after {ph.spec['proj_iters']} iterations")
# deleted particles neither collide nor are pushed
xd = xp.at[:20].set(xp[20:40])
xo = embody.Embodied(base, mode="phys", dropout=0.0, sense_noise=0.0).step(Pz, xd, S0, MEM, ON, EPS, EPS, None, N, alive)[0]
check("phys: a deleted particle sitting on a live one pushes nothing", float(jnp.abs(xo[:20] - xd[:20]).max()) == 0 and mind(xo[20:]) >= 2 * R - 1e-4)

# latency
el = embody.Embodied(pr2, mode="phys", dropout=0.0, quant_bits=0, sense_noise=0.0, latency=1)
mem1 = jnp.zeros((N, el.mem_dims))
xa, _, mem1, _ = el.step(None, X0, S0, mem1, ON, EPS, EPS, None, N, M1)
xb, _, mem1, _ = el.step(None, xa, S0, mem1, ON * 0, EPS, EPS, None, N, M1)
check("phys latency 1: the move proposed at tick t is executed at t+1", float(jnp.abs(xa - X0).max()) < 1e-6 and float(jnp.abs((xb - xa)[:, 0] - 0.004).max()) < 5e-4
      and el.mem_dims == 2)

# ---------------------------------------------------------------------------------------------------------------
# lattice
# ---------------------------------------------------------------------------------------------------------------
lat = embody.Embodied(base, mode="lattice")
PITCH = lat.pitch
check("lattice pitch is 2 * port_d = 28 mm", abs(PITCH * 900 - 28.0) < 1e-3)
xc0, _ = npa.seed(jax.random.PRNGKey(0), N, radius=2 * EPS)       # the evaluator's continuous disc
xs = lat.snap_to_lattice(xc0, M1)
ij = np.round(np.asarray(xs) / PITCH)
check("snap_to_lattice: on sites, no duplicates", float(jnp.abs(xs / PITCH - jnp.round(xs / PITCH)).max()) < 1e-4 and len({tuple(r) for r in ij}) == N)
check("snap_to_lattice: deterministic and idempotent", float(jnp.abs(lat.snap_to_lattice(xc0, M1) - xs).max()) == 0 and float(jnp.abs(lat.snap_to_lattice(xs, M1) - xs).max()) == 0)
xm = lat.snap_to_lattice(xc0, alive)
check("snap_to_lattice: deleted particles untouched and occupy nothing", float(jnp.abs(xm[:20] - xc0[:20]).max()) == 0
      and len({tuple(r) for r in np.round(np.asarray(xm[20:]) / PITCH)}) == N - 20)

# a docked blob
side = int(np.ceil(np.sqrt(N)))
sites = np.array([[i, j] for i in range(side) for j in range(side)])[:N]
xl = jnp.asarray(sites * PITCH, jnp.float32)
Pl = {**P, "w2": P["w2"] * 30.0, "w1": P["w1"] * 10.0}                      # decisive moves
xcur, scur, mcur = xl, S0, MEM
ok_site = ok_dup = ok_hop = ok_dock = True
hops_total = 0
det_first = None
for t in range(12):
    upd = (jax.random.uniform(jax.random.PRNGKey(100 + t), (N, 1)) < 0.5).astype(jnp.float32)
    a = lat.step(Pl, xcur, scur, mcur, upd, EPS, EPS, None, N, M1)
    xnew = a[0]
    if t == 0:
        b = lat.step(Pl, xcur, scur, mcur, upd, EPS, EPS, None, N, M1)
        det_first = float(jnp.abs(a[0] - b[0]).max()) + float(jnp.abs(a[1] - b[1]).max())
    ijn = np.round(np.asarray(xnew) / PITCH)
    ijo = np.round(np.asarray(xcur) / PITCH)
    ok_site &= float(jnp.abs(xnew / PITCH - jnp.round(xnew / PITCH)).max()) < 1e-4
    ok_dup &= len({tuple(r) for r in ijn}) == N
    hop = np.abs(ijn - ijo).sum(-1)
    ok_hop &= bool((hop <= 1).all())
    hops_total += int((hop == 1).sum())

    def ndock(ijs):
        d = np.abs(ijs[:, None] - ijs[None]).sum(-1)
        return (d == 1).sum(1)

    ok_dock &= bool(((ndock(ijo) == 0) | (ndock(ijn) > 0)).all())
    xcur, scur = xnew, a[1]
check("lattice: positions are on lattice sites", ok_site)
check("lattice: no two live particles share a site", ok_dup)
check("lattice: a hop moves at most one pitch", ok_hop, f"{hops_total} hops in 12 ticks")
check("lattice: hops actually happen", hops_total > 0)
check("lattice: a particle docked at the start is never isolated after a tick", ok_dock)
check("lattice: deterministic given the same inputs", det_first == 0.0)
# claims on one site: two bots both pushing toward the same empty site, exactly one wins
class Push:
    mem_dims = 0

    def __init__(self, v):
        self.v = jnp.asarray(v, jnp.float32)

    def step(self, p, x, s, mem, update, eps, eps0, block=None, total=None, mass=None):
        return x + update * self.v, s, mem, jnp.zeros(())

# sites: A(0,0) B(2,0) anchor C(1,1) above the gap, A pushes +x, B pushes -x -> both want (1,0)
sites3 = np.array([[0, 0], [2, 0], [1, 1]])
x3 = jnp.asarray(sites3 * PITCH, jnp.float32)
vv = jnp.array([[0.03, 0.0], [-0.03, 0.0], [0.0, 0.0]], jnp.float32)
class PushEach(Push):
    def step(self, p, x, s, mem, update, eps, eps0, block=None, total=None, mass=None):
        return x + update * self.v, s, mem, jnp.zeros(())
winners = set()
for sd in range(12):
    ee = embody.Embodied(PushEach(vv), mode="lattice", sense_noise=0.0, dropout=0.0, quant_bits=0, seed=sd)
    out = ee.step(None, x3, jnp.zeros((3, 16)), jnp.zeros((3, 0)), jnp.ones((3, 1)), EPS, EPS, None, 3, jnp.ones(3), key=jax.random.PRNGKey(sd))[0]
    ij3 = np.round(np.asarray(out) / PITCH).astype(int)
    assert len({tuple(r) for r in ij3}) == 3
    winners.add(tuple(ij3[0]) == (1, 0))
check("lattice: two claimants of one site -> exactly one gets it, resolved at random", winners == {True, False}, "both outcomes seen over 12 keys")
# docking: a lone pair, one pushes away -> blocked
sites2 = np.array([[0, 0], [1, 0]])
x2s = jnp.asarray(sites2 * PITCH, jnp.float32)
ee = embody.Embodied(PushEach(jnp.array([[0.0, 0.03], [0.0, 0.0]])), mode="lattice", sense_noise=0.0, dropout=0.0, quant_bits=0)
out = ee.step(None, x2s, jnp.zeros((2, 16)), jnp.zeros((2, 0)), jnp.ones((2, 1)), EPS, EPS, None, 2, jnp.ones(2))[0]
check("lattice: a hop that would undock the only neighbour is refused", float(jnp.abs(out - x2s).max()) == 0)
# an isolated bot may move
x1s = jnp.asarray(np.array([[0, 0], [5, 0]]) * PITCH, jnp.float32)
out = ee.step(None, x1s, jnp.zeros((2, 16)), jnp.zeros((2, 0)), jnp.ones((2, 1)), EPS, EPS, None, 2, jnp.ones(2))[0]
check("lattice: a bot that began isolated is free to hop", abs(float(out[0, 1]) - PITCH) < 1e-6)
# the first step snaps an off-lattice disc by itself
out = lat.step(P, xc0, S0, MEM, UPD, EPS, EPS, None, N, M1)[0]
check("lattice: first step snaps an off-lattice swarm onto distinct sites",
      float(jnp.abs(out / PITCH - jnp.round(out / PITCH)).max()) < 1e-4 and len({tuple(r) for r in np.round(np.asarray(out) / PITCH)}) == N)

# ---------------------------------------------------------------------------------------------------------------
# stride realism (optional fields)
# ---------------------------------------------------------------------------------------------------------------
MS = 18.0 / 900
quiet = dict(dropout=0.0, sense_noise=0.0, quant_bits=0)
pa = Probe(move=(0.01, 0.0))          # 9 mm: below the 18 mm deadzone
oa = embody.Embodied(pa, mode="phys", min_step=MS, **quiet).step(None, X0, S0, MEM, ON, EPS, EPS, None, N, M1)[0]
pb = Probe(move=(0.03, 0.0))          # 27 mm: above it
ob = embody.Embodied(pb, mode="phys", min_step=MS, max_step=50.0 / 900, **quiet).step(None, X0, S0, MEM, ON, EPS, EPS, None, N, M1)[0]
check("phys min_step: a move below the deadzone becomes 0, above it is executed", float(jnp.abs(oa - X0).max()) < 1e-6
      and float(jnp.abs((ob - X0)[:, 0] - 50.0 / 900 * np.tanh(0.03 / (50.0 / 900))).max()) < 1e-5)
gd = jax.grad(lambda v: jnp.sum(embody.Embodied(Probe(move=(0.01, 0.0)), mode="phys", min_step=MS, **quiet).step(
    None, X0 + v, S0, MEM, ON, EPS, EPS, None, N, M1)[0]))(jnp.zeros(2))
check("phys min_step: gradient still flows through the deadzone", bool(jnp.isfinite(gd).all()) and float(jnp.abs(gd).sum()) > 0)
pn = Probe(move=(0.03, 0.0))
en = embody.Embodied(pn, mode="phys", min_step=MS, max_step=50.0 / 900, step_len_noise=0.2, step_dir_noise=0.087, **quiet)
xn_ = en.step(None, X0, S0, MEM, ON, EPS, EPS, None, N, M1)[0] - X0
ln = jnp.sqrt(jnp.sum(xn_**2, -1))
ang = jnp.arctan2(xn_[:, 1], xn_[:, 0])
check("phys stride noise: length spread ~20 %, direction spread ~5 deg, never above max_step",
      0.1 < float(ln.std() / ln.mean()) < 0.3 and 0.05 < float(ang.std()) < 0.13 and float(ln.max()) <= 50.0 / 900 + 1e-6,
      f"len sigma {float(ln.std() / ln.mean()):.2f}, dir sigma {float(ang.std()) * 57.3:.1f} deg")
bn = Probe()
embody.Embodied(bn, mode="phys", bearing_noise=0.087, dropout=0.0, sense_noise=0.0, quant_bits=0).step(None, X0, S0, MEM, ON, EPS, EPS, None, N, M1)
check("bearing_noise (approximated as positional jitter) perturbs the perceived positions",
      0.5e-3 < float((bn.log[0]["x"] - X0).std()) < 6e-3, f"std {float((bn.log[0]['x'] - X0).std()) * 900:.2f} mm")

# hop_success
class PushR(Push):
    pass
xr = lat.snap_to_lattice(hexgrid(N, 3 * R) * 1.0, M1)
xr = jnp.asarray(sites * PITCH, jnp.float32)
ups = jnp.ones((N, 1))
def hop_count(prob):
    ee = embody.Embodied(PushR([0.05, 0.0]), mode="lattice", hop_success=prob, **quiet)
    tot = 0
    for k in range(6):
        o = ee.step(None, xr, S0, MEM, ups, EPS, EPS, None, N, M1, key=jax.random.PRNGKey(k))[0]
        tot += int((jnp.abs(o - xr).sum(-1) > 1e-6).sum())
    return tot
h1, h9, h0 = hop_count(1.0), hop_count(0.5), hop_count(0.0)
check("lattice hop_success: 0 never hops, lower success gives fewer hops", h0 == 0 and 0 < h9 < h1, f"hops over 6 ticks: p=1 {h1}, p=0.5 {h9}, p=0 {h0}")

# ---------------------------------------------------------------------------------------------------------------
# gradients, jit, vmap
# ---------------------------------------------------------------------------------------------------------------
def rollout_loss(params, e, x0, steps=4):
    x, s, mem = x0, S0, jnp.zeros((N, e.mem_dims))
    for t in range(steps):
        upd = (jax.random.uniform(jax.random.PRNGKey(t), (N, 1)) < 0.5).astype(jnp.float32)
        x, s, mem, mv = e.step(params, x, s, mem, upd, EPS, EPS, None, N, M1)
    return jnp.sum(x[:, 0] ** 2) + jnp.sum(s**2) * 1e-2


for mode in ("phys", "lattice"):
    ee = embody.Embodied(base, mode=mode)
    x0 = X0 if mode == "phys" else lat.snap_to_lattice(X0, M1)
    gr = jax.grad(rollout_loss)(P, ee, x0)
    gn = float(jnp.sqrt(sum(jnp.sum(v * v) for v in gr.values())))
    check(f"{mode}: gradients through a 4-step rollout are finite and nonzero", all(bool(jnp.isfinite(v).all()) for v in gr.values()) and gn > 0, f"|grad| {gn:.3e}")

# jit + vmap with positional args like evaluate.py / run.py
for mode in ("phys", "lattice"):
    ee = embody.Embodied(base, mode=mode)
    B = 2
    f = jax.jit(jax.vmap(ee.step, in_axes=(None, 0, 0, 0, 0, None, None, None, None, 0)))
    xb = jnp.stack([lat.snap_to_lattice(X0, M1)] * B) if mode == "lattice" else jnp.stack([X0] * B)
    out = f(Pl if mode == "lattice" else P, xb, jnp.stack([S0] * B), jnp.zeros((B, N, 0)), jnp.stack([UPD, 1 - UPD]), EPS, EPS, None, N, jnp.ones((B, N)))
    check(f"{mode}: jit + vmap with positional arguments", all(bool(jnp.isfinite(o).all()) for o in out)
          and float(jnp.abs(out[0][0] - out[0][1]).max()) > 0)

# a cell with memory (gated + mem) and the latency buffer
cm = cells.Cell(block="gated", depth=3, mem=4)
pm = cm.init(jax.random.PRNGKey(3))
em = embody.Embodied(cm, mode="phys", latency=1)
o = em.step(pm, X0, S0, jnp.zeros((N, em.mem_dims)), UPD, EPS, EPS, None, N, M1)
check("memory cell: mem_dims = base + 2 pending channels, finite", em.mem_dims == 6 and o[2].shape == (N, 6) and bool(jnp.isfinite(o[0]).all()))
for kind in ("edge", "attn"):
    try:
        import importlib
        mod = importlib.import_module(f"cells_{kind}")
        cc = mod.Cell()
        pc = cc.init(jax.random.PRNGKey(4))
        eo = embody.Embodied(cc, mode="phys")
        o = eo.step(pc, X0, S0, jnp.zeros((N, eo.mem_dims)), UPD, EPS, EPS, None, N, M1)
        check(f"wraps cells_{kind} too", bool(jnp.isfinite(o[0]).all() and jnp.isfinite(o[1]).all()))
    except Exception as err:
        check(f"wraps cells_{kind} too", False, repr(err)[:120])

# ---------------------------------------------------------------------------------------------------------------
# the trainer
# ---------------------------------------------------------------------------------------------------------------
for mode in ("phys", "lattice"):
    cfg = {"name": "t", "particles": 128, "cell": {"embody": {"mode": mode}},
           "train": {"iters": 3, "batch": 2, "steps": [3, 6], "grid": 64}}
    cell, params, eps, curve, secs = run.train(cfg, log=lambda m: None)
    check(f"run.train with embody {mode}: finite losses", len(curve) == 3 and all(np.isfinite(c["loss"]) for c in curve) and isinstance(cell, embody.Embodied),
          f"loss {curve[0]['loss']:.3f} -> {curve[-1]['loss']:.3f}, {secs:.1f}s")

print(f"\n{sum(results)}/{len(results)} embody checks passed")
raise SystemExit(0 if all(results) else 1)
